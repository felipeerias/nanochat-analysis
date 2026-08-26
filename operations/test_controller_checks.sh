#!/bin/bash
# Local tests for the controller's portable checks (CHECK_ONLY=1).
#
# These need no GPU, no volume and no pod. They cover the failures most
# likely to happen in practice: a manifest that does not parse, a row that
# does not exist, a checkout on the wrong commit, a dirty tree. Everything
# that genuinely needs hardware is covered by GATES_ONLY=1 on a pod instead.
#
#   bash operations/test_controller_checks.sh
#
# Temporary worktrees are created at the commits the shipped manifests pin,
# and removed afterwards.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
OPS=$(pwd)
CHECKOUT=${NANOCHAT_CHECKOUT:-$(cd ../../nanochat && pwd)}
TMP=$(mktemp -d)
trap 'git -C "$CHECKOUT" worktree remove --force "$TMP/wt-a" 2>/dev/null;
      git -C "$CHECKOUT" worktree remove --force "$TMP/wt-b" 2>/dev/null;
      rm -rf "$TMP"' EXIT

pin() { "${PY:-python3}" -c "import json,sys; print(json.load(open(sys.argv[1]))['nanochat_commit'])" "$1"; }
A=$(pin manifests/sweep-d12-d16-v1.json)
B=$(pin manifests/sweep-d12-seeds-v1.json)
git -C "$CHECKOUT" worktree add -q --detach "$TMP/wt-a" "$A"
git -C "$CHECKOUT" worktree add -q --detach "$TMP/wt-b" "$B"

# variants of the shipped manifest. A row is flat: any flag either parser
# declares, nothing the runner owns, nothing that does not exist.
"${PY:-python3}" - "$TMP" <<'PYEOF'
import json, sys, os
tmp = sys.argv[1]
base = json.load(open("manifests/sweep-d12-d16-v1.json"))

def variant(name, mutate):
    m = json.loads(json.dumps(base))
    mutate(m)
    json.dump(m, open(os.path.join(tmp, name), "w"))

def row(name, extra):
    variant(name, lambda m: m["runs"]["d14-s7"].update(extra))

variant("unpinned.json", lambda m: m.pop("nanochat_commit"))
row("badkey.json",   {"softcap": 10})
row("r-width.json",  {"aspect_ratio": 48})
row("r-sched.json",  {"warmdown_ratio": 0.35, "final_lr_frac": 1})
row("r-owned.json",  {"model_tag": "mine"})
row("r-mixed.json",  {"eval_every": -1, "telemetry_periodic_points": 4})
row("r-type.json",   {"warmdown_ratio": "0.35"})
row("r-nested.json", {"recipe": {"aspect_ratio": 48}})
PYEOF

# static first: the gate and run sections only execute on a GPU, so nothing
# below reaches them
"${PY:-python3}" "$OPS/check_shell_vars.py" "$OPS/telemetry_run.sh" \
    "$OPS/telemetry_pod_setup.sh" || exit 1

pass=0; fail=0
check() {
    local name="$1" want="$2"; shift 2
    local out; out=$("$@" 2>&1); local got=$?
    if [ "$got" = "$want" ]; then
        pass=$((pass + 1)); printf "  ok    %-40s exit=%s\n" "$name" "$got"
    else
        fail=$((fail + 1))
        printf "  FAIL  %-40s exit=%s want=%s\n%s\n" "$name" "$got" "$want" "$out"
    fi
}
# ALLOW_DIRTY so the suite is runnable while operations/ is being edited;
# one case deliberately drops it to prove a dirty controller is refused.
run()    { CHECK_ONLY=1 ALLOW_DIRTY=1 NANOCHAT_CHECKOUT="$1" bash "$OPS/telemetry_run.sh" "$2" "$3"; }
strict() { CHECK_ONLY=1 NANOCHAT_CHECKOUT="$1" bash "$OPS/telemetry_run.sh" "$2" "$3"; }

check "d12-d16 manifest at its pinned commit" 0 run "$TMP/wt-a" manifests/sweep-d12-d16-v1.json d14-s7
check "seeds manifest at its pinned commit"   0 run "$TMP/wt-b" manifests/sweep-d12-seeds-v1.json d12-s10
check "relative manifest path resolves"       0 run "$TMP/wt-a" ./manifests/sweep-d12-d16-v1.json d14-s7
check "checkout on the wrong commit"          1 run "$CHECKOUT" manifests/sweep-d12-d16-v1.json d14-s7
check "manifest crossed with wrong worktree"  1 run "$TMP/wt-a" manifests/sweep-d12-seeds-v1.json d12-s10
check "unknown run id"                        1 run "$TMP/wt-a" manifests/sweep-d12-d16-v1.json d99-s1
check "checkout is not a git repo"            1 run /tmp        manifests/sweep-d12-d16-v1.json d14-s7
check "manifest without nanochat_commit"      1 run "$TMP/wt-a" "$TMP/unpinned.json" d14-s7

check "row sets width (E03 shape)"            0 run "$TMP/wt-a" "$TMP/r-width.json" d14-s7
check "row sets the schedule (E02 shape)"     0 run "$TMP/wt-a" "$TMP/r-sched.json" d14-s7
check "base_train and telemetry flags mix"    0 run "$TMP/wt-a" "$TMP/r-mixed.json" d14-s7
check "key no parser declares"                1 run "$TMP/wt-a" "$TMP/badkey.json" d14-s7
check "key the runner owns"                   1 run "$TMP/wt-a" "$TMP/r-owned.json" d14-s7
check "value of the wrong type"               1 run "$TMP/wt-a" "$TMP/r-type.json" d14-s7
check "a nested recipe block is refused"      1 run "$TMP/wt-a" "$TMP/r-nested.json" d14-s7

if [ -n "$(git -C "$OPS" status --porcelain -- "$OPS")" ]; then
    check "dirty controller tree refused"     1 strict "$TMP/wt-a" manifests/sweep-d12-d16-v1.json d14-s7
else
    check "clean controller tree accepted"    0 strict "$TMP/wt-a" manifests/sweep-d12-d16-v1.json d14-s7
fi

echo
echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ]
