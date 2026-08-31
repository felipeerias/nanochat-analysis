#!/bin/bash
set -euo pipefail

# Run a queue of manifest rows in sequence: several manifests, each pinning
# its own nanochat commit, on one pod.
#
# An exploratory campaign produces a handful of experiment branches with one
# small manifest each. This script runs them back to back. Each distinct pin
# gets a temporary worktree off $NANOCHAT_CHECKOUT, so the shared checkout's
# HEAD never moves; telemetry_run.sh still enforces pin == HEAD inside the
# worktree, exactly as it does for a single run.
#
# Every item is validated with the runner's portable checks BEFORE the first
# run starts: a typo in item four must surface before item one spends hours.
# During execution a failed item does not stop the queue - one broken branch
# must not waste the pod hours paid for the others - and the summary at the
# end names every outcome. The exit code is nonzero if anything failed.
#
# The gates stamp per training HEAD, so the first row of each branch pays
# the gate suite once and later rows of the same branch reuse the stamp.
#
# Usage (after operations/telemetry_pod_setup.sh):
#   bash operations/telemetry_queue.sh MANIFEST[:RUN_ID] [MANIFEST[:RUN_ID] ...]
# A manifest without :RUN_ID runs every row it holds, in file order.
#
# Env, passed through to telemetry_run.sh: CHECK_ONLY=1 validates the whole
# queue anywhere (no GPU, no volume, no pod); STOP_POD=1 stops the pod after
# the WHOLE queue - this script owns the pod, so the runs inside it are
# always started with their own self-stop off.

if [ "$#" -lt 1 ]; then
    echo "usage: telemetry_queue.sh MANIFEST[:RUN_ID] [MANIFEST[:RUN_ID] ...]"
    exit 1
fi
VOLUME=${VOLUME:-/workspace}
NANOCHAT_CHECKOUT=${NANOCHAT_CHECKOUT:-$VOLUME/nanochat}
OPS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PY=$(command -v python3 || command -v python) || true
if [ -z "$PY" ]; then echo "FATAL: no python interpreter found."; exit 1; fi
if ! git -C "$NANOCHAT_CHECKOUT" rev-parse --git-dir >/dev/null 2>&1; then
    echo "FATAL: $NANOCHAT_CHECKOUT is not a git checkout."; exit 1
fi

manifest_field() {  # manifest_field FILE pin|runs
    "$PY" - "$1" "$2" <<'PYEOF'
import json, sys
m = json.load(open(sys.argv[1]))
if sys.argv[2] == "pin":
    print(str(m.get("nanochat_commit", "")))
else:
    for rid in (m.get("runs") or {}):
        print(rid)
PYEOF
}

# ----------------------------------------------------------------------------
# Expand the queue up front, reading every manifest once, so an unreadable
# file fails here rather than mid-campaign.
ITEMS=()   # "manifest<TAB>run_id"
for arg in "$@"; do
    mf=${arg%%:*}
    rid=""
    [ "$arg" != "$mf" ] && rid=${arg#*:}
    if [ ! -f "$mf" ]; then echo "FATAL: no manifest at $mf"; exit 1; fi
    mf=$(cd "$(dirname "$mf")" && pwd)/$(basename "$mf")
    if [ -n "$rid" ]; then
        ITEMS+=("$mf"$'\t'"$rid")
    else
        rows=$(manifest_field "$mf" runs) \
            || { echo "FATAL: cannot read $mf"; exit 1; }
        if [ -z "$rows" ]; then echo "FATAL: $mf holds no runs"; exit 1; fi
        while IFS= read -r rid; do ITEMS+=("$mf"$'\t'"$rid"); done <<< "$rows"
    fi
done

# ----------------------------------------------------------------------------
# Fail-closed self-stop preflight, same rationale as telemetry_run.sh: the
# runs inside the queue have STOP_POD=0, so this script must prove up front
# that its own stop at the end can work.
if [ "${CHECK_ONLY:-0}" != "1" ] && [ "${STOP_POD:-0}" = "1" ] \
        && [ -n "${RUNPOD_POD_ID:-}" ]; then
    if ! command -v runpodctl &> /dev/null; then
        echo "FATAL: STOP_POD=1 but runpodctl is not installed."; exit 1
    fi
    if [ -z "${RUNPOD_API_KEY:-}" ]; then
        echo "FATAL: STOP_POD=1 but RUNPOD_API_KEY is unset."; exit 1
    fi
    if probe_out=$(timeout 30 runpodctl get pod "$RUNPOD_POD_ID" 2>&1); then
        :
    elif ! printf '%s' "$probe_out" | grep -qi "unauthorized"; then
        echo "FATAL: STOP_POD=1 but the runpodctl API probe failed: $probe_out"
        exit 1
    fi
fi

# ----------------------------------------------------------------------------
# One temporary worktree per distinct pin, removed at exit. A missing commit
# gets one fetch before failing: pod setup fetches, but a branch pushed after
# setup would otherwise need a manual step.
WT_ROOT=$(mktemp -d)
declare -A wt_dirs=()
FETCHED=0
worktree_for() {  # sets REPLY to the worktree holding this pin
    local pin="$1" dir
    if [ -n "${wt_dirs[$pin]:-}" ]; then REPLY=${wt_dirs[$pin]}; return 0; fi
    if ! git -C "$NANOCHAT_CHECKOUT" rev-parse --quiet --verify \
            "$pin^{commit}" >/dev/null && [ "$FETCHED" != "1" ]; then
        echo "[queue] $pin is not here; fetching"
        git -C "$NANOCHAT_CHECKOUT" fetch --all --tags --prune || true
        FETCHED=1
    fi
    dir="$WT_ROOT/wt-${pin:0:12}"
    git -C "$NANOCHAT_CHECKOUT" worktree add -q --detach "$dir" "$pin" || return 1
    wt_dirs[$pin]=$dir
    REPLY=$dir
}

finish() {
    local dir
    for dir in ${wt_dirs[@]+"${wt_dirs[@]}"}; do
        git -C "$NANOCHAT_CHECKOUT" worktree remove --force "$dir" 2>/dev/null || true
    done
    rm -rf "$WT_ROOT"
    if [ "${CHECK_ONLY:-0}" = "1" ]; then return 0; fi
    [ -n "${RUNPOD_POD_ID:-}" ] && [ "${STOP_POD:-0}" = "1" ] || return 0
    if ! command -v runpodctl &> /dev/null || [ -z "${RUNPOD_API_KEY:-}" ] \
            || ! timeout 60 runpodctl stop pod "$RUNPOD_POD_ID"; then
        echo "[queue] STOP_POD=1 but stopping $RUNPOD_POD_ID failed; stop it manually"
    else
        echo "[queue] pod $RUNPOD_POD_ID stopped"
    fi
}
trap finish EXIT

# ----------------------------------------------------------------------------
# Validate everything, then stop if that is all that was asked. A queue that
# does not fully validate runs nothing: nothing has started yet, and fixing a
# manifest is cheaper than discovering a silently skipped item tomorrow.
echo "[queue] ${#ITEMS[@]} item(s); validating all of them before anything runs"
for item in "${ITEMS[@]}"; do
    mf=${item%%$'\t'*}; rid=${item#*$'\t'}
    pin=$(manifest_field "$mf" pin)
    if [ -z "$pin" ]; then
        echo "FATAL: $(basename "$mf") does not pin nanochat_commit"; exit 1
    fi
    if ! worktree_for "$pin"; then
        echo "FATAL: cannot check out $pin for $(basename "$mf")"; exit 1
    fi
    if ! CHECK_ONLY=1 STOP_POD=0 NANOCHAT_CHECKOUT="$REPLY" \
            bash "$OPS_DIR/telemetry_run.sh" "$mf" "$rid"; then
        echo "FATAL: $(basename "$mf"):$rid does not validate; nothing was run"
        exit 1
    fi
done
if [ "${CHECK_ONLY:-0}" = "1" ]; then
    echo "[queue] CHECK_ONLY=1 - every item validates, nothing was started"
    exit 0
fi

# ----------------------------------------------------------------------------
# Run the queue. Each run logs itself under $VOLUME/logs as always; the
# summary is also written there, because a screen scrollback does not
# survive the night.
passed=(); failed=()
for item in "${ITEMS[@]}"; do
    mf=${item%%$'\t'*}; rid=${item#*$'\t'}
    name="$(basename "$mf"):$rid"
    pin=$(manifest_field "$mf" pin)
    worktree_for "$pin" || { failed+=("$name"); continue; }
    echo "[queue] === $name @ ${pin:0:7} $(date -u +%FT%TZ) ==="
    if STOP_POD=0 NANOCHAT_CHECKOUT="$REPLY" \
            bash "$OPS_DIR/telemetry_run.sh" "$mf" "$rid"; then
        passed+=("$name")
    else
        echo "[queue] FAILED: $name (continuing with the rest)"
        failed+=("$name")
    fi
done

summary_log=""
if [ -d "$VOLUME/logs" ]; then
    summary_log="$VOLUME/logs/queue-$(date +%Y%m%d-%H%M%S).log"
fi
{
    echo "[queue] done: ${#passed[@]} passed, ${#failed[@]} failed, ${#ITEMS[@]} total"
    for name in ${passed[@]+"${passed[@]}"}; do echo "[queue]   ok    $name"; done
    for name in ${failed[@]+"${failed[@]}"}; do echo "[queue]   FAIL  $name"; done
} | if [ -n "$summary_log" ]; then tee "$summary_log"; else cat; fi
[ "${#failed[@]}" -eq 0 ]
