"""Loading layer for nanochat telemetry segments (Claude's implementation).

Uses the instrument's own reader (nanochat.telemetry.read_telemetry) plus
pandas. The filtering discipline lives HERE so analyses cannot silently
mishandle undefined rows or uncertified verdicts:

- defined(): drops is_defined == False rows EXPLICITLY (never implicitly).
- arm(): schema v2+ per-arm selection; on v1 data only "native" is valid.
- certified(): curvature/update rows whose checkpoint's verdict passed,
  per arm - the only rows a headline claim may use.

Join conventions (from the spec): pre_update rows carry step s, post_update
rows carry step s+1; deep post-update checkpoints therefore appear at
s_deep + 1. normalized_progress is the cross-run x-axis.
"""

import json
import os
import sys

import pandas as pd

# layout: ~/Igalia/nanochat/{analysis,nanochat}; REPO is the repo ROOT
# (the dir CONTAINING the nanochat package, not the package itself)
REPO = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "nanochat")
assert os.path.isfile(os.path.join(REPO, "nanochat", "telemetry.py")), REPO
sys.path.insert(0, REPO)
from nanochat.telemetry import read_telemetry  # noqa: E402

TIERS = ("continuous", "periodic", "sparse", "offline")


def load_segment(root, seg):
    prov = json.load(open(os.path.join(root, seg, "provenance.json")))
    tiers = {}
    for tier in TIERS:
        if os.path.isdir(os.path.join(root, seg, tier)):
            tiers[tier] = read_telemetry(root, seg, tier).to_pandas()
    return {"provenance": prov, "tiers": tiers, "segment": seg}


def defined(df):
    """Explicitly keep only defined rows (callers must OPT IN to dropping)."""
    return df[df["is_defined"]]


def metric(df, name, phase=None, step=None):
    out = df[df["metric"] == name]
    if phase is not None:
        out = out[out["phase"] == phase]
    if step is not None:
        out = out[out["step"] == step]
    return out


def arm(df, which):
    """Per-arm selection. v1 data (no acceptance_arm column) is native-only."""
    if "acceptance_arm" not in df.columns:
        if which != "native":
            raise ValueError(f"schema v1 data has no {which!r} arm")
        return df
    return df[df["acceptance_arm"] == which]


def deep_post_steps(prov):
    """Post-update step labels of the deep checkpoints."""
    steps = prov.get("telemetry_deep_steps")
    if steps:
        return sorted(s + 1 for s in steps)
    de = prov.get("telemetry_deep_every", -1)
    n = prov["num_iterations"]
    if de and de > 0:
        return [s + 1 for s in range(0, n, de)]
    return []


def verdict_by_step(sparse, which="native"):
    """step -> verdict string for one arm's checkpoints."""
    name = ("curvature/native_verdict_code" if which == "native"
            else "curvature/shadow_verdict_code")
    rows = defined(metric(arm(sparse, which), name))
    code = {0.0: "passed", 1.0: "inconclusive", 2.0: "failed"}
    return {int(s): code[v] for s, v in
            zip(rows["step"], rows["value_scalar"])}


def certified(sparse, which="native"):
    """Curvature/update rows at checkpoints whose arm verdict PASSED."""
    ok = {s for s, v in verdict_by_step(sparse, which).items() if v == "passed"}
    a = arm(sparse, which)
    picked = a[a["metric"].str.startswith(("curvature/", "update/"))
               & a["step"].isin(sorted(ok))]
    return picked
