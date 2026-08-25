"""First end-to-end analysis of the collected d12 run - the FREEZE GATE.

Everything below is computed from parquet + provenance alone. Outputs:
figures/ *.png, results.json (the cross-implementation answer sheet), and
findings.md (numbers + any data-shape frictions discovered).
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from telemetry_load import (load_segment, defined, metric, arm,
                            deep_post_steps, verdict_by_step)

ROOT = os.path.expanduser("~/Igalia/nanochat/telemetry-data/runpod/telemetry-data")
SEG = "d12-iter-s0-0a3f5527067944708caeb7e1ff638b76"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)
frictions = []

data = load_segment(ROOT, SEG)
prov = data["provenance"]
cont, per, sparse, off = (data["tiers"][t] for t in
                          ("continuous", "periodic", "sparse", "offline"))
dsteps = deep_post_steps(prov)
verdicts = verdict_by_step(sparse, "native")

# ---------------------------------------------------------------- answer sheet
res = {"segment": SEG,
       "row_counts": {t: int(len(data["tiers"][t])) for t in data["tiers"]},
       "undefined_counts": {t: int((~data["tiers"][t]["is_defined"]).sum())
                            for t in data["tiers"]}}
vc = defined(metric(sparse, "curvature/native_verdict_code"))["value_scalar"]
code = {0.0: "passed", 1.0: "inconclusive", 2.0: "failed"}
res["native_verdict_counts"] = {name: int((vc == v).sum())
                                for v, name in code.items()}
res["deep"] = {}
for s in dsteps:
    entry = {}
    for key, m in (("gHg", "curvature/gHg"), ("gg", "curvature/gg"),
                   ("eta_star", "curvature/eta_star"),
                   ("dhd", "curvature/dhd"), ("update_p1", "update/p1"),
                   ("update_p2", "update/p2"),
                   ("update_actual", "update/actual")):
        rows = metric(sparse, m, step=s)
        if len(rows) != 1:
            frictions.append(f"{m} at step {s}: multiplicity {len(rows)} != 1")
        row = rows.iloc[0]
        entry[key] = (float(row["value_scalar"]) if row["is_defined"] else None)
    res["deep"][str(s)] = entry
res["relerr"] = {}
for s in dsteps:
    rr = defined(metric(sparse, "muon/replay_update_relerr", step=s))
    v = rr["value_scalar"]
    res["relerr"][str(s)] = {"n": int(len(v)), "median": float(v.median()),
                             "max": float(v.max()),
                             "zeros": int((v < 1e-12).sum())}
res["train_loss_first"] = float(
    metric(cont, "loss/train_mean", step=0)["value_scalar"].iloc[0])
res["train_loss_last"] = float(
    metric(cont, "loss/train_mean", step=2519)["value_scalar"].iloc[0])
pl = defined(metric(per, "probe/loss"))
pl = pl[pl["probe_id"] == prov["probe_ids"]["val"]]
res["probe_val_loss_last"] = float(
    pl.sort_values("step")["value_scalar"].iloc[-1])
ov = off[off["metric"].str.startswith("overhead/total/")]
res["overhead_total_seconds"] = float(ov["value_scalar"].sum())
res["max_grad_norm_at_1000"] = float(
    metric(per, "grad/norm", phase="pre_update", step=1000)["value_scalar"].max())
json.dump(res, open(os.path.join(os.path.dirname(OUT), "results.json"), "w"),
          indent=2)

# ------------------------------------------------------------------- figure 1
# Muon replay decoherence on real gradients (the reproducible result)
rr = defined(metric(sparse, "muon/replay_update_relerr"))
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for s in dsteps:
    v = rr[rr["step"] == s]["value_scalar"]
    axes[0].scatter([s] * len(v), v, s=8, alpha=0.5)
axes[0].set(xlabel="post-update step", ylabel="update-relative L2 error",
            title="Muon reference-vs-actual divergence (per matrix)")
axes[0].set_yscale("symlog", linthresh=1e-4)
by_role = (rr.groupby(["step", "param_role"])["value_scalar"].median()
           .unstack("param_role"))
by_role.plot(ax=axes[1], marker="o")
axes[1].set(xlabel="post-update step", ylabel="median rel. error",
            title="median by parameter role")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_relerr.png"), dpi=120)

# ------------------------------------------------------------------- figure 2
# Curvature trajectory - honestly labeled with its verdicts
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, key, name in ((axes[0], "gHg", "curvature/gHg"),
                      (axes[1], "eta_star", "curvature/eta_star")):
    xs, ys = [], []
    for s in dsteps:
        val = res["deep"][str(s)][key]
        if val is not None:
            xs.append(s), ys.append(val)
    ax.plot(xs, ys, "o-")
    for s in dsteps:
        ax.annotate(verdicts.get(s, "?"), (s, ax.get_ylim()[0]),
                    fontsize=7, color="red")
    ax.set(title=f"{name}  [verdicts marked - ALL {set(verdicts.values())}]",
           xlabel="step")
    ax.set_yscale("log")
p1 = [res["deep"][str(s)]["update_p1"] for s in dsteps]
p2 = [res["deep"][str(s)]["update_p2"] for s in dsteps]
act = [res["deep"][str(s)]["update_actual"] for s in dsteps]
axes[2].plot(dsteps, p1, "o-", label="p1 (linear)")
axes[2].plot(dsteps, p2, "s-", label="p2 (quadratic)")
axes[2].plot(dsteps, act, "^-", label="actual")
axes[2].axhline(0, color="gray", lw=0.5)
axes[2].legend()
axes[2].set(title="local-model prediction vs actual dL", xlabel="step")
fig.suptitle("UNCERTIFIED (native bf16 verdicts: failed) - shadow arm "
             "certification pending", color="red", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_curvature.png"), dpi=120)

# ------------------------------------------------------------------- figure 3
# Breadth pass: one panel per family class
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
tl = defined(metric(cont, "loss/train_mean")).sort_values("step")
axes[0, 0].plot(tl["step"], tl["value_scalar"], lw=0.6)
pv = defined(metric(per, "probe/loss"))
for pid, g in pv.groupby("probe_id"):
    label = [k for k, v in prov["probe_ids"].items() if v == pid][0]
    g = g.sort_values("step")
    axes[0, 0].plot(g["step"], g["value_scalar"], "o-", ms=3, label=label)
axes[0, 0].legend(fontsize=7)
axes[0, 0].set(title="train loss + probe losses", xlabel="step")

for m, ax, logy in (("noise/b_noise", axes[0, 1], True),
                    ("noise/s2", axes[0, 2], True)):
    rows = defined(metric(per, m)).sort_values("step")
    if rows.empty:
        frictions.append(f"{m}: no defined rows")
        continue
    for role, g in rows.groupby(rows["param_role"].fillna("(global)")):
        ax.plot(g["step"], g["value_scalar"], "o-", ms=3, label=str(role)[:16])
    ax.set(title=m, xlabel="step")
    if logy:
        ax.set_yscale("log")
    ax.legend(fontsize=6)

gn = defined(metric(per, "grad/norm", phase="pre_update"))
for role, g in gn.groupby("param_role"):
    gg = g.groupby("step")["value_scalar"].max()
    axes[1, 0].plot(gg.index, gg.values, "o-", ms=3, label=role[:16])
axes[1, 0].set(title="grad/norm (max over layers, by role)", xlabel="step")
axes[1, 0].set_yscale("log")
axes[1, 0].legend(fontsize=6)

mc = defined(metric(per, "muon/cos_raw_final"))
by_layer = mc.groupby(["step", "layer"])["value_scalar"].median().unstack()
by_layer.plot(ax=axes[1, 1], legend=False, alpha=0.7)
axes[1, 1].set(title="muon cos(raw grad, final update) by layer",
               xlabel="step")

ae = defined(metric(per, "attn/per_head_norm_entropy"))
by_layer = ae.groupby(["step", "layer"])["value_scalar"].median().unstack()
by_layer.plot(ax=axes[1, 2], legend=False, alpha=0.7)
axes[1, 2].set(title="attention norm entropy (median/head) by layer",
               xlabel="step")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_breadth.png"), dpi=120)

# ------------------------------------------------------------------ findings
ov_by = ov.set_index("metric")["value_scalar"].sort_values(ascending=False)
with open(os.path.join(os.path.dirname(OUT), "findings.md"), "w") as f:
    f.write("# First end-to-end analysis - d12-iter (freeze gate)\n\n")
    f.write(f"- rows: {res['row_counts']}, undefined: {res['undefined_counts']}\n")
    f.write(f"- native verdicts: {res['native_verdict_counts']} - every deep "
            f"checkpoint is UNCERTIFIED at bf16; curvature figures are "
            f"labeled accordingly. Certified numbers await the shadow arm.\n")
    f.write(f"- train loss {res['train_loss_first']:.4f} -> "
            f"{res['train_loss_last']:.4f}; final val-probe loss "
            f"{res['probe_val_loss_last']:.4f}\n")
    f.write(f"- Muon decoherence (median/max per checkpoint): "
            + ", ".join(f"step {s}: {res['relerr'][str(s)]['median']:.4f}/"
                        f"{res['relerr'][str(s)]['max']:.4f}"
                        for s in dsteps) + "\n")
    f.write(f"- telemetry overhead {res['overhead_total_seconds']:.1f}s; "
            f"top sections: "
            + ", ".join(f"{k.split('/')[-1]}={v:.1f}s"
                        for k, v in ov_by.head(5).items()) + "\n")
    f.write("\n## Data-shape frictions\n\n")
    if frictions:
        f.writelines(f"- {x}\n" for x in frictions)
    else:
        f.write("- none: every quantity above was computable from parquet + "
                "provenance alone with the documented joins.\n")
print("results.json, findings.md, figures/ written")
print("frictions:", frictions or "none")
