"""Loading layer for nanochat telemetry segments (Claude's implementation).

Uses the instrument's own reader (nanochat.telemetry.read_telemetry) plus
pandas. The filtering discipline lives HERE so analyses cannot silently
mishandle undefined rows or uncertified verdicts:

- defined(): drops is_defined == False rows EXPLICITLY (never implicitly).
- arm(): schema v2+ per-arm selection; on v1 data only "native" is valid.
- certified(): curvature/update rows gated by the verdict for the HVP
  direction they actually depend on. Rows that do not depend on an HVP are
  not verdict-gated. Definedness remains an explicit, separate filter.

Join conventions (from the spec): pre_update rows carry step s, post_update
rows carry step s+1; deep post-update checkpoints therefore appear at
s_deep + 1. normalized_progress is the cross-run x-axis.
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd

# By default the two repositories and telemetry-data share one workspace.
# Environment overrides make the same checkout usable in any layout.
ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(os.environ.get(
    "NANOCHAT_WORKSPACE_ROOT", ANALYSIS_ROOT.parent)).expanduser().resolve()
NANOCHAT_REPO = Path(os.environ.get(
    "NANOCHAT_REPO", WORKSPACE_ROOT / "nanochat")).expanduser().resolve()
DEFAULT_DATA_ROOT = Path(os.environ.get(
    "NANOCHAT_TELEMETRY_DATA_ROOT",
    WORKSPACE_ROOT / "telemetry-data" / "sweep" / "telemetry-data",
)).expanduser().resolve()

# NANOCHAT_REPO is the directory containing the nanochat package.
REPO = str(NANOCHAT_REPO)
instrument = NANOCHAT_REPO / "nanochat" / "telemetry.py"
if not instrument.is_file():
    raise FileNotFoundError(
        f"nanochat telemetry instrument not found at {instrument}; "
        "set NANOCHAT_REPO to the nanochat checkout"
    )
sys.path.insert(0, REPO)
from nanochat.telemetry import read_telemetry  # noqa: E402

TIERS = ("continuous", "periodic", "sparse", "offline")

_DIRECTIONS = ("random", "gradient", "update")
_NO_HVP_METRICS = {
    "update/p1",
    "update/actual",
    "update/residual_p1",
    "update/loss_before",
    "update/loss_after",
    "update/direction_norm",
    "update/direction",
    "curvature/arith_eps",
    "curvature/native_verdict_code",
    "curvature/shadow_verdict_code",
    "curvature/fp32_verdict_code",
    *(f"curvature/verdict_code_{direction}" for direction in _DIRECTIONS),
}
_GRADIENT_HVP_METRICS = {
    "curvature/gHg",
    "curvature/gg",
    "curvature/eta_star",
    "curvature/Hg_norm",
    "curvature/eta_star_rho",
    "curvature/eta_star_rho_threshold",
}
_UPDATE_HVP_METRICS = {
    "update/p2",
    "update/residual_p2",
    "update/normalized_residual",
    "curvature/dhd",
}


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
    """Step -> worst checkpoint verdict for one arm (reporting only)."""
    name = ("curvature/native_verdict_code" if which == "native"
            else "curvature/shadow_verdict_code")
    rows = defined(metric(arm(sparse, which), name))
    code = {0.0: "passed", 1.0: "inconclusive", 2.0: "failed"}
    return {int(s): code[v] for s, v in
            zip(rows["step"], rows["value_scalar"])}


def direction_verdict_by_step(sparse, direction, which="native"):
    """Step -> verdict string for one arm and HVP direction."""
    if direction not in _DIRECTIONS:
        raise ValueError(f"unknown HVP direction {direction!r}")
    name = f"curvature/verdict_code_{direction}"
    rows = defined(metric(arm(sparse, which), name))
    code = {0.0: "passed", 1.0: "inconclusive", 2.0: "failed"}
    return {int(s): code[v] for s, v in
            zip(rows["step"], rows["value_scalar"])}


def metric_certifier(name):
    """Return a metric's HVP direction, or None when it needs no HVP.

    Unknown update/curvature metrics fail closed so a future schema addition
    cannot silently inherit the wrong certification rule.
    """
    if name in _NO_HVP_METRICS:
        return None
    if name in _GRADIENT_HVP_METRICS:
        return "gradient"
    if name in _UPDATE_HVP_METRICS:
        return "update"
    for direction in _DIRECTIONS:
        if name.startswith("curvature/") and name.endswith(f"_{direction}"):
            return direction
    raise ValueError(f"no certification rule for metric {name!r}")


def certified(sparse, which="native"):
    """Return per-direction-certified curvature/update rows for one arm.

    This does not drop undefined rows. Call ``defined()`` explicitly after
    this function when an analysis requires defined measurements.
    """
    a = arm(sparse, which)
    picked = a[a["metric"].str.startswith(("curvature/", "update/"))]
    passed = {
        direction: {
            step for step, verdict in
            direction_verdict_by_step(a, direction, which).items()
            if verdict == "passed"
        }
        for direction in _DIRECTIONS
    }
    keep = []
    for name, step in zip(picked["metric"], picked["step"]):
        direction = metric_certifier(name)
        keep.append(direction is None or int(step) in passed[direction])
    return picked.loc[keep]
