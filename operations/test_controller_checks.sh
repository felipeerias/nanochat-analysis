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

# variants of the shipped manifest that must be rejected
"${PY:-python3}" - "$TMP" <<'PYEOF'
import json, sys, os
tmp = sys.argv[1]
base = json.load(open("manifests/sweep-d12-d16-v1.json"))
m = json.loads(json.dumps(base)); del m["nanochat_commit"]
json.dump(m, open(os.path.join(tmp, "unpinned.json"), "w"))
m = json.loads(json.dumps(base)); m["runs"]["d14-s7"]["depth"] = 999
json.dump(m, open(os.path.join(tmp, "badrange.json"), "w"))
m = json.loads(json.dumps(base)); m["runs"]["d14-s7"]["aspect_ratio"] = 48
json.dump(m, open(os.path.join(tmp, "badkey.json"), "w"))

# recipe variants: a manifest may set upstream base_train flags, but only
# the ones in the runner's table, only in range, and only with right types
def recipe(name, r):
    m = json.loads(json.dumps(base))
    m["runs"]["d14-s7"]["recipe"] = r
    json.dump(m, open(os.path.join(tmp, name), "w"))

recipe("r-width.json",   {"aspect_ratio": 48})
recipe("r-sched.json",   {"warmdown_ratio": 0.35, "final_lr_frac": 1.0})
recipe("r-range.json",   {"aspect_ratio": 99999})
recipe("r-unknown.json", {"softcap": 10})
recipe("r-type.json",    {"warmdown_ratio": "0.35"})
PYEOF

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
check "depth out of range"                    1 run "$TMP/wt-a" "$TMP/badrange.json" d14-s7
check "unknown row key (aspect_ratio)"        1 run "$TMP/wt-a" "$TMP/badkey.json" d14-s7

check "recipe sets width (E03 shape)"         0 run "$TMP/wt-a" "$TMP/r-width.json" d14-s7
check "recipe sets the schedule (E02 shape)"  0 run "$TMP/wt-a" "$TMP/r-sched.json" d14-s7
check "recipe value out of range"             1 run "$TMP/wt-a" "$TMP/r-range.json" d14-s7
check "recipe key not in the table (softcap)" 1 run "$TMP/wt-a" "$TMP/r-unknown.json" d14-s7
check "recipe value of the wrong type"        1 run "$TMP/wt-a" "$TMP/r-type.json" d14-s7

# a recipe that changes width must change the width the verifier is told to
# expect, or a wrong model would pass verification
w=$(CHECK_ONLY=1 ALLOW_DIRTY=1 NANOCHAT_CHECKOUT="$TMP/wt-a" \
    bash "$OPS/telemetry_run.sh" "$TMP/r-width.json" d14-s7 2>&1 \
    | sed -n 's/.*width=\([0-9]*\) .*/\1/p')
if [ "$w" = "768" ]; then
    pass=$((pass + 1)); printf "  ok    %-40s width=%s\n" "aspect_ratio 48 at depth 14 -> 768" "$w"
else
    fail=$((fail + 1)); printf "  FAIL  %-40s width=%s want=768\n" "aspect_ratio 48 at depth 14" "$w"
fi

if [ -n "$(git -C "$OPS" status --porcelain -- "$OPS")" ]; then
    check "dirty controller tree refused"     1 strict "$TMP/wt-a" manifests/sweep-d12-d16-v1.json d14-s7
else
    check "clean controller tree accepted"    0 strict "$TMP/wt-a" manifests/sweep-d12-d16-v1.json d14-s7
fi

echo
echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ]
