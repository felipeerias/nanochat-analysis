#!/bin/bash
set -euo pipefail

# Generic sweep runner: pre-collection gates, one instrumented training run
# selected from a sweep manifest, artifact verification, then pod SELF-STOP.
#
# Usage (after runs/telemetry_pod_setup.sh):
#   bash operations/telemetry_run.sh operations/manifests/sweep-d12-d16-v1.json d12-s7
#   STOP_POD=1 bash operations/telemetry_run.sh <manifest> <run_id>   # self-stop at the end
# Env (booleans count only as "1"): CHECK_ONLY (portable checks then stop),
#   GATES_ONLY (gates then stop), STOP_POD (stop the pod when done),
#   SKIP_GATES, ALLOW_DIRTY,
#   ALLOW_EPHEMERAL, CONFIRM_VOLUME; EXPECT_BACKEND (default fa3),
#   TELEMETRY_DIR, MIN_FREE_GB (default 60), NANOCHAT_CHECKOUT (default
#   $VOLUME/nanochat) - the training checkout this controller drives.
# NO extra base_train arguments are accepted: an official manifest run is
# exactly the recipe plus the row's telemetry settings - dev experiments
# call scripts.base_train directly instead. The run_id doubles as the
# model tag and segment tag. Long runs: launch under screen.
# Manifests are IMMUTABLE once any run has used them; each segment embeds
# its exact manifest bytes.
# Self-stop uses the pod-scoped RUNPOD_API_KEY Runpod injects (it can stop
# this pod but not read account state); a loud banner prints if it fails.

MANIFEST=${1:?usage: telemetry_run.sh <manifest.json> <run_id>}
RUN_ID=${2:?usage: telemetry_run.sh <manifest.json> <run_id>}
if [ "$#" -gt 2 ]; then
    echo "FATAL: extra arguments are not accepted on official manifest runs"
    echo "(got: ${*:3}). Call scripts.base_train directly for experiments."
    exit 1
fi
VOLUME=${VOLUME:-/workspace}
# The controller no longer lives inside the checkout it drives, so the
# manifest path is resolved before anything changes directory, and the
# checkout is named explicitly.
MANIFEST=$(cd "$(dirname "$MANIFEST")" && pwd)/$(basename "$MANIFEST")
NANOCHAT_CHECKOUT=${NANOCHAT_CHECKOUT:-$VOLUME/nanochat}

# --------------------------------------------------------------------------
# Portable checks FIRST: everything that needs neither a GPU nor a volume,
# so a run configuration can be verified anywhere. CHECK_ONLY=1 stops here.
# These used to sit behind the volume and Runpod guards, which meant a
# malformed manifest could only be discovered on a paid pod.
PY=$(command -v python3 || command -v python) || true
if [ -z "$PY" ]; then echo "FATAL: no python interpreter found."; exit 1; fi
# rev-parse, not a .git directory test: in a worktree .git is a FILE, and
# worktrees are how two experiment branches get driven side by side.
if ! git -C "$NANOCHAT_CHECKOUT" rev-parse --git-dir >/dev/null 2>&1; then
    echo "FATAL: $NANOCHAT_CHECKOUT is not a git checkout."; exit 1
fi

# ----------------------------------------------------------------------------
# Resolve this run's row (defaults overlaid by the row; manifest is the
# single source of truth). The row is VALIDATED in python - types, allowed
# keys, allowed values - and transferred as tab-separated values, never
# through shell interpolation.

DIRTY=0
if [ -n "$(git -C "$NANOCHAT_CHECKOUT" status --porcelain)" ]; then
    if [ "${ALLOW_DIRTY:-0}" != "1" ]; then
        echo "FATAL: dirty checkout (or set ALLOW_DIRTY=1)."; exit 1
    fi
    DIRTY=1
fi

# HEAD is read here because the gate key needs it; comparing it against what
# the manifest pins is run_manifest.py's job, below.
HEAD_SHA=$(git -C "$NANOCHAT_CHECKOUT" rev-parse HEAD)

# Controller identity, taken from wherever this script lives so it keeps
# working once operations move to their own repository. The committed tree oid
# does not change when the working tree does, so cleanliness is checked
# directly rather than inferred from it.
CTRL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CTRL_COMMIT=$(git -C "$CTRL_DIR" rev-parse HEAD 2>/dev/null || echo "unavailable")
CTRL_PREFIX=$(git -C "$CTRL_DIR" rev-parse --show-prefix 2>/dev/null || echo "")
CTRL_TREE=$(git -C "$CTRL_DIR" rev-parse "HEAD:${CTRL_PREFIX%/}" 2>/dev/null || echo "unavailable")
if [ -n "$(git -C "$CTRL_DIR" status --porcelain -- "$CTRL_DIR" 2>/dev/null)" ]; then
    if [ "${ALLOW_DIRTY:-0}" != "1" ]; then
        echo "FATAL: dirty controller tree at $CTRL_DIR (or set ALLOW_DIRTY=1)."
        exit 1
    fi
    DIRTY=1
fi
echo "[controller] $CTRL_COMMIT tree=$CTRL_TREE dir=$CTRL_DIR"

RESOLVE=("$PY" "$CTRL_DIR/run_manifest.py" "$MANIFEST" "$RUN_ID"
         --checkout "$NANOCHAT_CHECKOUT" --head "$HEAD_SHA"
         --expect-backend "${EXPECT_BACKEND:-fa3}"
         --controller-commit "$CTRL_COMMIT" --controller-tree "$CTRL_TREE")
"${RESOLVE[@]}" --check-only

if [ "${CHECK_ONLY:-0}" = "1" ]; then
    echo "[telemetry_run] CHECK_ONLY=1 - nothing was started"
    exit 0
fi

# ----------------------------------------------------------------------------
# Self-stop trap FIRST: any failure below must still stop the pod.
self_stop() {
    status=$?
    echo "[telemetry_run] exiting with status $status"
    if [ -n "${TEE_PID:-}" ]; then
        echo "[telemetry_run] closing log $LOG"
        exec 1>&3 2>&4
        for _ in $(seq 1 15); do
            kill -0 "$TEE_PID" 2>/dev/null || break
            sleep 1
        done
    fi
    timeout 20 sync || true
    # Nothing is printed when the pod is left running: that is the default.
    [ -n "${RUNPOD_POD_ID:-}" ] && [ "${STOP_POD:-0}" = "1" ] || return 0
    if ! command -v runpodctl &> /dev/null || [ -z "${RUNPOD_API_KEY:-}" ] \
            || ! timeout 60 runpodctl stop pod "$RUNPOD_POD_ID"; then
        outcome="[telemetry_run] STOP_POD=1 but stopping $RUNPOD_POD_ID failed; stop it manually"
    else
        outcome="[telemetry_run] pod $RUNPOD_POD_ID stopped"
    fi
    echo "$outcome"
    if [ -n "${LOG:-}" ]; then
        printf '%s\n' "$outcome" | timeout 5 tee -a "$LOG" > /dev/null 2>&1 || true
    fi
}
trap self_stop EXIT

# ----------------------------------------------------------------------------
# Fail-closed pre-flight: network volume attached, enough space, self-stop
# plausible - all BEFORE any expensive work.
if [ "${ALLOW_EPHEMERAL:-0}" != "1" ]; then
    if [ "$(stat -c %d "$VOLUME" 2>/dev/null)" = "$(stat -c %d /)" ]; then
        echo "FATAL: $VOLUME is not a separate mount (set ALLOW_EPHEMERAL=1 to accept data loss)."
        exit 1
    fi
    VOL_FSTYPE=$(findmnt -no FSTYPE --target "$VOLUME" 2>/dev/null \
        || awk -v m="$VOLUME" '$2 == m {print $3}' /proc/mounts | tail -1)
    case "$VOL_FSTYPE" in
        nfs*|cifs|smb*|ceph*|virtiofs|9p|fuse*) ;;
        *)
            if [ "${CONFIRM_VOLUME:-0}" != "1" ]; then
                echo "FATAL: $VOLUME fstype '$VOL_FSTYPE' does not look network-backed;"
                echo "it may die with the pod. If SURE it persists, set CONFIRM_VOLUME=1."
                exit 1
            fi ;;
    esac
fi
MIN_FREE_GB=${MIN_FREE_GB:-60}
AVAIL_GB=$(df -BG --output=avail "$VOLUME" | tail -1 | tr -dc '0-9')
if [ "$AVAIL_GB" -lt "$MIN_FREE_GB" ]; then
    echo "FATAL: only ${AVAIL_GB}G free on $VOLUME, need ${MIN_FREE_GB}G (lineage"
    echo "checkpoints are large). Grow the volume, clean up, or lower MIN_FREE_GB."
    exit 1
fi
if [ -n "${RUNPOD_POD_ID:-}" ] && [ "${STOP_POD:-0}" = "1" ]; then
    if ! command -v runpodctl &> /dev/null; then
        echo "FATAL: STOP_POD=1 but runpodctl is not installed."; exit 1
    fi
    if [ -z "${RUNPOD_API_KEY:-}" ]; then
        echo "FATAL: STOP_POD=1 but RUNPOD_API_KEY is unset."; exit 1
    fi
    # exit 0 = account-scope key; "Unauthorized" = the injected pod-scoped
    # key (stop works, account reads do not) - accepted; anything else
    # (timeout, unreachable) refuses, because the trap's stop would fail too
    if probe_out=$(timeout 30 runpodctl get pod "$RUNPOD_POD_ID" 2>&1); then
        :
    elif ! printf '%s' "$probe_out" | grep -qi "unauthorized"; then
        echo "FATAL: STOP_POD=1 but the runpodctl API probe failed: $probe_out"; exit 1
    fi
fi

if [ ! -f "$VOLUME/nanochat-env.sh" ]; then
    echo "FATAL: $VOLUME/nanochat-env.sh is missing. Run:"
    echo "  bash $CTRL_DIR/telemetry_pod_setup.sh"
    exit 1
fi
source "$VOLUME/nanochat-env.sh"
# The venv lives on the pod's LOCAL disk, not the network volume, so a new
# pod never has it even though the volume looks fully set up.
if [ ! -f "$UV_PROJECT_ENVIRONMENT/bin/activate" ]; then
    echo "FATAL: no environment at $UV_PROJECT_ENVIRONMENT."
    echo "It lives on this pod's local disk, so a new pod needs setup again"
    echo "even though the volume already holds the data. Run:"
    echo "  bash $CTRL_DIR/telemetry_pod_setup.sh"
    exit 1
fi
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
cd "$NANOCHAT_CHECKOUT"
export PYTHONUNBUFFERED=1

TELEMETRY_DIR=${TELEMETRY_DIR:-$VOLUME/telemetry-data}
EXPECT_BACKEND=${EXPECT_BACKEND:-fa3}
LOG="$VOLUME/logs/${RUN_ID}-$(date +%Y%m%d-%H%M%S).log"
mkdir -p "$VOLUME/logs"
touch "$LOG"
exec 3>&1 4>&2
exec > >(tee -a "$LOG") 2>&1
TEE_PID=$!
echo "[telemetry_run] manifest=$MANIFEST run=$RUN_ID"
echo "[telemetry_run] log: $LOG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python - <<'EOF'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA not available - wrong pod/image?")
from nanochat.common import COMPUTE_DTYPE
if COMPUTE_DTYPE != torch.bfloat16:
    raise SystemExit(f"expected bf16 on GPU, got {COMPUTE_DTYPE}")
EOF
ACTUAL_BACKEND=$(python -c "from nanochat.flash_attention import USE_FA3; print('fa3' if USE_FA3 else 'sdpa')")
if [ "$ACTUAL_BACKEND" != "$EXPECT_BACKEND" ]; then
    echo "FATAL: resolved attention backend '$ACTUAL_BACKEND' != expected '$EXPECT_BACKEND'."
    exit 1
fi
echo "[preflight] attention backend: $ACTUAL_BACKEND (as expected)"


# --------------------------------------------------------------------------
# Pre-collection gates: one FIXED cheap config (they test the instrument,
# not the model size), stamped per full software path plus shadow setting.
# A dirty checkout runs gates but never stamps.
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
TORCH_KEY=$(python -c "import torch; print(torch.__version__, torch.version.cuda)")
# The gates test a fixed cheap config, but the acceptance arm is the run's,
# so the gate key and the gate run both need it.
SHADOW=$("${RESOLVE[@]}" --print telemetry_shadow)
GATE_KEY="$HEAD_SHA|ctrl=$CTRL_TREE|gpu=$GPU_NAME|driver=$DRIVER|torch=$TORCH_KEY|expect=$EXPECT_BACKEND|actual=$ACTUAL_BACKEND|shadow=$SHADOW"
GATE_STAMP="$VOLUME/.telemetry-gates-$(printf '%s' "$GATE_KEY" | sha256sum | cut -c1-16)"
echo "[gates] key: $GATE_KEY"
if [ "$DIRTY" != "1" ] && { [ "${SKIP_GATES:-0}" = "1" ] || [ -f "$GATE_STAMP" ]; }; then
    echo "[gates] skipped (stamp: $GATE_STAMP)"
else
    echo "[gates] stage 1/3: CPU-contract suites (CUDA hidden)"
    CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_telemetry.py tests/test_telemetry_integration.py -m "not slow" -q
    echo "[gates] stage 2/3: poly31 CPU/GPU hash conformance"
    python -m pytest "tests/test_telemetry.py::TestSketch::test_poly31_gpu_conformance" -q
    echo "[gates] stage 3/3: compiled GPU controls"
    python -m pytest \
        "tests/test_telemetry.py::TestMuon::test_replay_matches_compiled_production_gpu" \
        "tests/test_telemetry_integration.py::TestBaseTrainIntegration::test_ab_compiled_control_gpu" \
        -q
    echo "[gates] 3-step instrumented gate run (production deep_steps path + lineage)"
    rm -rf "$VOLUME/.gate-run"
    python -m scripts.base_train --depth=12 --seed=7 \
        --num-iterations=3 --eval-every=-1 --core-metric-every=-1 --sample-every=-1 \
        --save-every=-1 --model-tag=gate --run=dummy \
        --telemetry-dir="$VOLUME/.gate-run" --telemetry-periodic-points=3 \
        --telemetry-checkpoints=1 --telemetry-shadow="$SHADOW"
    python runs/verify_telemetry_run.py "$VOLUME/.gate-run" \
        --expect "attention_backend=$EXPECT_BACKEND" \
        --expect "compute_dtype=torch.bfloat16" \
        --expect "telemetry_config.shadow_arm=$SHADOW" \
        --expect "model_config.n_layer=12" \
        --expect "model_config.n_embd=768" \
        --expect "model_config.n_head=6" \
        --expect "seed=7"
    if [ "$DIRTY" = "1" ]; then
        echo "[gates] all passed (dirty checkout: NOT stamped)"
    else
        touch "$GATE_STAMP"
        echo "[gates] all passed; stamped $GATE_STAMP"
    fi
fi

if [ "${GATES_ONLY:-0}" = "1" ]; then
    echo "[telemetry_run] GATES_ONLY=1 - gates passed, manifest run NOT started"
    exit 0
fi

# ----------------------------------------------------------------------------
# The instrumented run: exactly the recipe plus the manifest row.
# Resolving the row, building the argument list and verifying what came out
# all belong to one program, which does them without a text round trip.
echo "[run] starting $RUN_ID"
"${RESOLVE[@]}" --telemetry-dir "$TELEMETRY_DIR"
echo "[telemetry_run] done; data in $TELEMETRY_DIR (network volume - survives pod stop)"
