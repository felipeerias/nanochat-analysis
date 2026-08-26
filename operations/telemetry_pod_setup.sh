#!/bin/bash
set -euo pipefail

# One-time / idempotent setup for a Runpod pod with a NETWORK VOLUME mounted
# at /workspace. Everything that must survive pod stop/terminate lives on the
# volume (repo, data, tokenizer, HF cache, telemetry output, logs); the venv
# is rebuilt on local disk each fresh pod (fast, thanks to the uv cache on
# the volume).
#
# Bootstrap on a brand-new pod (the only manual step; idempotent when the
# volume already holds the clone):
#   cd /workspace \
#     && { [ -d nanochat-analysis ] || git clone https://github.com/felipeerias/nanochat-analysis.git; } \
#     && bash nanochat-analysis/operations/telemetry_pod_setup.sh
#
# Re-running is safe: every step checks for existing state first.
# Requirements on the pod: a CUDA GPU, internet access, /workspace volume.
#
# NETWORK VOLUME SIZING: 250 GB recommended for the d12-d16 sweep (sweep-d12-d16-v1)
# (volumes grow but never shrink). Checkpoints dominate: SIX physical save
# equivalents per run (theta_0 + 3 interiors + the recipe final + its
# independent in-segment copy; ~4 GB each at d12, more at d14/d16), so
# d12x3+d14+d16 is ~156 GB, plus ~8-20 GB dataset shards, ~6 GB caches,
# ~0.1-1 GB telemetry parquet per run, and a 60 GB working reserve.
# telemetry_run.sh refuses to start when free space is below MIN_FREE_GB
# (default 60).

VOLUME=${VOLUME:-/workspace}
REPO_DIR="${NANOCHAT_CHECKOUT:-$VOLUME/nanochat}"
ENV_FILE="$VOLUME/nanochat-env.sh"
N_SHARDS=${N_SHARDS:-80}   # ~250M chars/shard; d12 at 12:1 needs ~66 + val + margin

# Fail closed unless $VOLUME looks like a mounted NETWORK volume. The device
# comparison catches "no volume at all"; the filesystem type distinguishes a
# Runpod network volume (NFS-family) from a pod-LOCAL volume disk that would
# also be a separate mount yet still dies with the pod.
if [ "${ALLOW_EPHEMERAL:-0}" != "1" ]; then
    if [ "$(stat -c %d "$VOLUME" 2>/dev/null)" = "$(stat -c %d /)" ]; then
        echo "FATAL: $VOLUME is not a separate mount - attach the network volume"
        echo "(or set ALLOW_EPHEMERAL=1 to accept data loss on pod stop)."
        exit 1
    fi
    VOL_FSTYPE=$(findmnt -no FSTYPE --target "$VOLUME" 2>/dev/null \
        || awk -v m="$VOLUME" '$2 == m {print $3}' /proc/mounts | tail -1)
    case "$VOL_FSTYPE" in
        nfs*|cifs|smb*|ceph*|virtiofs|9p|fuse*) ;;   # network-backed: ok
        *)
            if [ "${CONFIRM_VOLUME:-0}" != "1" ]; then
                echo "FATAL: $VOLUME is a separate mount but fstype '$VOL_FSTYPE' does"
                echo "not look network-backed; it may be a pod-local volume that dies"
                echo "with the pod. If you are SURE it persists, set CONFIRM_VOLUME=1."
                exit 1
            fi
            echo "WARNING: CONFIRM_VOLUME=1 - trusting $VOLUME (fstype '$VOL_FSTYPE') to persist"
            ;;
    esac
fi

# ----------------------------------------------------------------------------
# Persistent environment (sourced by the runner and by interactive shells)
cat > "$ENV_FILE" <<EOF
export NANOCHAT_BASE_DIR="$VOLUME/nanochat-data"   # data, tokenizer, checkpoints
export HF_HOME="$VOLUME/.hf"                       # FA3 kernel + HF downloads
# Runpod images export HF_HUB_ENABLE_HF_TRANSFER=1, but the locked venv has
# no hf_transfer package; huggingface_hub then refuses ALL downloads and the
# FA3 kernel fetch fails (silently -> SDPA fallback). Force it off.
export HF_HUB_ENABLE_HF_TRANSFER=0
export UV_CACHE_DIR="$VOLUME/.uv-cache"            # wheel cache -> fast venv rebuilds
export UV_PROJECT_ENVIRONMENT="\$HOME/nanochat-venv"  # venv on LOCAL disk, not the volume
export OMP_NUM_THREADS=1
export PATH="\$HOME/.local/bin:\$PATH"
EOF
source "$ENV_FILE"
mkdir -p "$NANOCHAT_BASE_DIR" "$VOLUME/telemetry-data" "$VOLUME/logs"

# ----------------------------------------------------------------------------
# Training checkout: make sure it exists and that its objects are current, but
# NEVER move HEAD. Manifests pin an exact commit and the runner enforces it, so
# a helpful pull here would silently defeat that check moments before it runs.
if [ ! -d "$REPO_DIR/.git" ]; then
    echo "[setup] cloning nanochat into $REPO_DIR"
    git clone -b telemetry https://github.com/felipeerias/nanochat.git "$REPO_DIR"
fi
cd "$REPO_DIR"
if ! git fetch --all --tags --prune; then
    if [ "${ALLOW_STALE:-0}" = "1" ]; then
        echo "WARNING: git fetch failed; ALLOW_STALE=1 set, continuing with what is here"
    else
        echo "FATAL: git fetch failed (offline?) - refusing to run possibly"
        echo "incomplete objects. Fix the network, or set ALLOW_STALE=1."
        exit 1
    fi
fi
echo "[setup] $REPO_DIR is at $(git rev-parse HEAD) on $(git rev-parse --abbrev-ref HEAD)"
echo "[setup] check out the commit your manifest pins before running it"

# ----------------------------------------------------------------------------
# Python environment (mirrors runs/speedrun.sh, plus the dev group for the gates)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv "$UV_PROJECT_ENVIRONMENT" 2>/dev/null || true
uv sync --frozen --extra gpu --group dev   # --frozen: never rewrite uv.lock
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

# ----------------------------------------------------------------------------
# Data + tokenizer (all idempotent; the dataset downloader skips existing shards)

# The downloader tolerates individual shard failures; verify the EXACT
# expected set (train shards 0..N-1 plus the pinned val shard 6542, which the
# loader's split convention needs as the last file) and open every Parquet
# footer so a truncated download cannot pass on filename alone.
check_shards() {
    python - "$1" <<'PYCHECK'
import os
import sys
import pyarrow.parquet as pq
from nanochat.dataset import MAX_SHARD, index_to_filename, list_parquet_files
need = int(sys.argv[1])
files = {os.path.basename(f): f for f in list_parquet_files()}
expected = [index_to_filename(i) for i in range(need)] + [index_to_filename(MAX_SHARD)]
missing = [n for n in expected if n not in files]
assert not missing, f"shards missing (failed downloads?): {missing} - rerun setup"
rows = 0
for name in expected:
    try:
        md = pq.ParquetFile(files[name]).metadata
    except Exception as e:
        raise SystemExit(f"shard {name} has a corrupt/truncated footer: {e} - "
                         f"delete it and rerun setup")
    assert md.num_rows > 0, f"shard {name} is empty"
    rows += md.num_rows
print(f"dataset ok: {len(expected)} shards verified, {rows} rows total")
PYCHECK
}

python -m nanochat.dataset -n 8   # enough characters for tokenizer training
check_shards 8   # BEFORE tok_train: a lost shard must not yield a bad tokenizer
# Skip retraining only if BOTH artifacts actually LOAD and work (existence
# alone would trust a file truncated by an interrupted write). The telemetry
# provenance hashes the tokenizer; all runs must share one copy on the volume.
if python - <<'TOKCHECK'
import sys
try:
    import torch
    from nanochat.common import get_base_dir
    from nanochat.tokenizer import get_tokenizer
    import os
    tok = get_tokenizer()
    ids = tok.encode("hello world")
    assert ids, "encode returned nothing"
    tb = torch.load(os.path.join(get_base_dir(), "tokenizer", "token_bytes.pt"))
    assert tb.numel() > 0, "token_bytes empty"
    print(f"tokenizer ok: vocab={tok.get_vocab_size()}, token_bytes={tuple(tb.shape)}")
except Exception as e:
    print(f"tokenizer artifacts unusable ({e}); retraining")
    sys.exit(1)
TOKCHECK
then
    echo "tokenizer already trained and loadable, skipping"
else
    python -m scripts.tok_train
    python -m scripts.tok_eval
fi
python -m nanochat.dataset -n "$N_SHARDS"
check_shards "$N_SHARDS"

echo
echo "Setup complete."
echo "  env file : source $ENV_FILE"
echo "  data     : $NANOCHAT_BASE_DIR"
echo "  next     : git -C $REPO_DIR checkout <manifest's nanochat_commit>"
echo "             bash operations/telemetry_run.sh operations/manifests/<manifest>.json <run_id>"
