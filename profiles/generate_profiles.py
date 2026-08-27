"""Generate one profile per training run, plus one per collection.

Profiles are descriptive: fixed summaries and data-quality notes. No
comparison, no ranking, no explanation - those belong in an investigation.

Usage from the repository root: uv run python profiles/generate_profiles.py
"""

import collections
import datetime
import json
import os
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
from loader.telemetry_load import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    arm,
    defined,
    load_segment,
    metric,
)

ROOT = str(DEFAULT_DATA_ROOT)
VERDICT = {0.0: "passed", 1.0: "inconclusive", 2.0: "failed"}
DIRECTIONS = ("random", "gradient", "update")


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=HERE,
            text=True).strip()
    except Exception:
        return "uncommitted"


def header(source, commit):
    return (f"---\n"
            f"source: {source}\n"
            f"profile_spec: 1\n"
            f"generator: {commit}:profiles/generate_profiles.py\n"
            f"generated: {datetime.date.today().isoformat()}\n"
            f"author: Claude Code (Fable 5)\n"
            f"---\n")


def summarize(root, seg):
    """All fixed summaries for one segment. Pure description."""
    d = load_segment(root, seg)
    prov, tiers = d["provenance"], d["tiers"]
    mc = prov.get("model_config") or {}
    cfg = prov.get("telemetry_config") or {}
    out = {
        "segment": seg,
        "run": prov.get("manifest_run_id") or seg.split("-s0-")[0],
        "schema": str(cfg.get("schema_version", "?")),
        "seed": prov.get("seed"),
        "depth": mc.get("n_layer"), "width": mc.get("n_embd"),
        "heads": mc.get("n_head"),
        "iters": prov.get("num_iterations"),
        "periodic_every": prov.get("telemetry_every"),
        "deep_steps": prov.get("telemetry_deep_steps") or [],
        "backend": prov.get("attention_backend"),
        "dtype": prov.get("compute_dtype"),
        "shadow": cfg.get("shadow_arm", "off"),
        "lineage": prov.get("lineage_checkpoint_labels"),
        "rows": {t: int(len(df)) for t, df in tiers.items()},
        "undefined": {t: int((~df["is_defined"]).sum())
                      for t, df in tiers.items()},
        "families": {t: int(df["metric"].nunique()) for t, df in tiers.items()},
    }
    sparse = tiers["sparse"]
    has_arm = "acceptance_arm" in sparse.columns
    out["checkpoint_verdicts"] = {}
    out["direction_verdicts"] = {}
    arms = (["native", "shadow_fp32"] if has_arm and out["shadow"] != "off"
            else ["native"])
    for a in arms:
        sub = arm(sparse, a) if has_arm else sparse
        vname = ("curvature/native_verdict_code" if a == "native"
                 else "curvature/shadow_verdict_code")
        vals = defined(metric(sub, vname))["value_scalar"]
        out["checkpoint_verdicts"][a] = dict(
            collections.Counter(VERDICT[v] for v in vals))
        per = {}
        for dname in DIRECTIONS:
            vv = defined(metric(sub, f"curvature/verdict_code_{dname}"))["value_scalar"]
            per[dname] = dict(collections.Counter(VERDICT[v] for v in vv))
        out["direction_verdicts"][a] = per

    cont = tiers["continuous"]
    tl = defined(metric(cont, "loss/train_mean")).sort_values("step")
    out["train_loss"] = {"first": float(tl["value_scalar"].iloc[0]),
                         "last": float(tl["value_scalar"].iloc[-1]),
                         "n": int(len(tl))}
    per_t = tiers["periodic"]
    pl = defined(metric(per_t, "probe/loss"))
    probes = {v: k for k, v in (prov.get("probe_ids") or {}).items()}
    out["probe_loss_final"] = {}
    for pid, g in pl.groupby("probe_id"):
        g = g.sort_values("step")
        out["probe_loss_final"][probes.get(pid, pid[:8])] = float(
            g["value_scalar"].iloc[-1])
    off = tiers["offline"]
    ov = off[off["metric"].str.startswith("overhead/total/")]
    out["overhead"] = {m.split("/")[-1]: float(v) for m, v in
                       zip(ov["metric"], ov["value_scalar"])}
    out["overhead_total"] = float(sum(out["overhead"].values()))
    out["_frames"] = {"train_loss": tl, "probe_loss": pl, "probes": probes}
    return out


def write_profile(s, commit):
    seg = s["segment"]
    d = os.path.join(HERE, seg)
    os.makedirs(os.path.join(d, "figures"), exist_ok=True)
    fr = s.pop("_frames")

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].plot(fr["train_loss"]["step"], fr["train_loss"]["value_scalar"], lw=0.6)
    ax[0].set(title="loss/train_mean", xlabel="step", ylabel="loss")
    for pid, g in fr["probe_loss"].groupby("probe_id"):
        g = g.sort_values("step")
        ax[1].plot(g["step"], g["value_scalar"], "o-", ms=3,
                   label=fr["probes"].get(pid, pid[:8]))
    ax[1].set(title="probe/loss", xlabel="step")
    ax[1].legend(fontsize=7)
    fig.suptitle(s["run"], fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(d, "figures", "loss.png"), dpi=110)
    plt.close(fig)

    L = [header(seg, commit), f"\n# Profile — {s['run']}\n"]
    L.append("\n## Inventory\n\n")
    L.append(f"- schema v{s['schema']}, seed {s['seed']}, "
             f"depth {s['depth']}, width {s['width']}, heads {s['heads']}\n")
    L.append(f"- {s['iters']} steps; periodic every {s['periodic_every']}; "
             f"{len(s['deep_steps'])} deep checkpoints\n")
    L.append(f"- backend {s['backend']}, compute dtype {s['dtype']}, "
             f"shadow arm {s['shadow']}\n")
    L.append(f"- lineage checkpoint labels: {s['lineage']}\n\n")
    L.append("| tier | rows | undefined | metric families |\n|---|---|---|---|\n")
    for t in ("continuous", "periodic", "sparse", "offline"):
        L.append(f"| {t} | {s['rows'][t]:,} | {s['undefined'][t]:,} | "
                 f"{s['families'][t]} |\n")
    L.append("\n## Summaries\n\n")
    L.append(f"- train loss: {s['train_loss']['first']:.4f} at the first step, "
             f"{s['train_loss']['last']:.4f} at the last "
             f"({s['train_loss']['n']:,} points)\n")
    for k, v in sorted(s["probe_loss_final"].items()):
        L.append(f"- final probe loss, {k}: {v:.4f}\n")
    L.append(f"- telemetry overhead: {s['overhead_total']:.1f} s total; "
             + ", ".join(f"{k} {v:.1f} s" for k, v in
                         sorted(s["overhead"].items(),
                                key=lambda x: -x[1])[:5]) + "\n")
    L.append("\n### Acceptance verdicts\n\n")
    L.append("Checkpoint level (worst of the three probe directions):\n\n")
    for a, c in s["checkpoint_verdicts"].items():
        L.append(f"- {a}: {c}\n")
    L.append("\nPer direction:\n\n| arm | direction | verdicts |\n|---|---|---|\n")
    for a, per in s["direction_verdicts"].items():
        for dname, c in per.items():
            L.append(f"| {a} | {dname} | {c} |\n")
    L.append("\n![loss](figures/loss.png)\n")
    L.append("\n## Data quality notes\n\n")
    notes = []
    if s["undefined"]["periodic"] or s["undefined"]["sparse"]:
        notes.append(f"Undefined rows are present and carry reasons: "
                     f"{s['undefined']['periodic']:,} periodic, "
                     f"{s['undefined']['sparse']:,} sparse. They are honest "
                     f"'not measurable here' records, not missing data.")
    if s["checkpoint_verdicts"].get("native", {}).get("failed"):
        notes.append("Every native (bf16) checkpoint verdict is `failed`. This "
                     "is a property of bf16 arithmetic against fp32-era "
                     "thresholds, documented in DATASET.md caveat 4. Native "
                     "curvature values are not certified measurements.")
    if s["shadow"] != "off":
        notes.append("Shadow (IEEE fp32) checkpoint verdicts are the worst of "
                     "three directions, so they understate what is usable; see "
                     "the per-direction table above.")
    notes.append("Lineage checkpoint tensors (`checkpoints/*.pt`) are absent "
                 "from this local copy by design; they remain on the GPU "
                 "volume. Verification of their hashes is not possible here.")
    L.extend(f"- {n}\n" for n in notes)
    open(os.path.join(d, "profile.md"), "w").writelines(L)
    return s


def main():
    commit = git_commit()
    segs = sorted(x for x in os.listdir(ROOT)
                  if os.path.isdir(os.path.join(ROOT, x)))
    done = []
    for seg in segs:
        s = summarize(ROOT, seg)
        if s["schema"] != "3":
            print(f"skip {s['run']}: schema v{s['schema']} (legacy)")
            continue
        done.append(write_profile(s, commit))
        print(f"profiled {s['run']}")

    # collection profile: inventory only
    d = os.path.join(HERE, "sweep-d12-d16")
    os.makedirs(d, exist_ok=True)
    L = [header("all schema-v3 segments in telemetry-data/sweep", commit),
         "\n# Profile — the d12-d16 collection\n\n## Inventory\n\n"]
    L.append("| run | seed | depth | width | heads | steps | periodic every | "
             "deep ckpts | train loss last |\n")
    L.append("|---|---|---|---|---|---|---|---|---|\n")
    for s in sorted(done, key=lambda x: (x["depth"], x["seed"])):
        L.append(f"| {s['run']} | {s['seed']} | {s['depth']} | {s['width']} | "
                 f"{s['heads']} | {s['iters']} | {s['periodic_every']} | "
                 f"{len(s['deep_steps'])} | {s['train_loss']['last']:.4f} |\n")
    grids = collections.defaultdict(list)
    for s in done:
        grids[(s["iters"], s["periodic_every"], tuple(s["deep_steps"]))].append(
            s["run"])
    L.append("\n## Step grids\n\n")
    L.append("Runs that share a step grid can be compared point by point with "
             "no interpolation. Runs that do not must be aligned on "
             "`normalized_progress`.\n\n")
    for i, (k, runs) in enumerate(sorted(grids.items()), 1):
        L.append(f"- grid {i}: {s_fmt(k)} — {', '.join(sorted(runs))}\n")
    L.append("\n## Data quality notes\n\n")
    L.append("- The five d12 runs come from two manifests "
             "(`sweep-d12-d16-v1` and `sweep-d12-seeds-v1`) but are "
             "configuration-identical apart from the seed.\n")
    L.append("- One legacy segment (`d12-iter`, schema v1, head_dim 64, no "
             "shadow arm) is present in the data folder and is not profiled "
             "here. Do not pool it with these runs.\n")
    open(os.path.join(d, "profile.md"), "w").writelines(L)
    print("profiled the collection")


def s_fmt(k):
    iters, every, deep = k
    return f"{iters} steps, periodic every {every}, {len(deep)} deep"


if __name__ == "__main__":
    main()
