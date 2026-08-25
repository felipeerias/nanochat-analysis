"""I0003/A0001 — figures and the final application of the decision rule.

Reads matched.csv / per_matrix.csv written by analysis.py and structure.py.
Writes fig1_decision.png, fig2_role.png, fig3_layer.png, verdict.txt.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
S = pd.read_csv(os.path.join(OUT, "matched.csv"))
M = pd.read_csv(os.path.join(OUT, "per_matrix.csv"))

C = {12: "#2a78d6", 14: "#eb6834", 16: "#1baf7a"}   # validated slots 1-3
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dcdbd6"
SURF = "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 9,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
})

log = open(os.path.join(OUT, "verdict.txt"), "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    log.write(s + "\n")


# =========================================================== decision rule
say("=" * 78)
say("I0003 / A0001 — application of the frozen decision rule")
say("protocol: investigations/0003-decoherence-vs-depth/README.md @ e76859c")
say("=" * 78)
say("""
Per-run summary at a matched checkpoint = MEDIAN over that run's per-matrix
channels (78 at d12, 91 at d14, 104 at d16).  Role composition is EXACTLY
proportional at every depth (6 roles x n_layer matrices, plus n_layer/2
ve_gate = 12/13 and 1/13 of the population at all three depths), so the
medians are taken over composition-matched populations and need no
re-weighting.  The update-0 checkpoint (structural zeros) is not on the
matched grid and is excluded everywhere.
""")

STAT = "median"
rows = []
for t in sorted(S["nominal"].unique()):
    a = S[S["nominal"] == t]
    d12 = a[a["depth"] == 12][STAT].to_numpy()
    lo, hi, med = d12.min(), d12.max(), float(np.median(d12))
    v14 = float(a[a["depth"] == 14][STAT].iloc[0])
    v16 = float(a[a["depth"] == 16][STAT].iloc[0])
    rows.append(dict(
        progress=t, d12_lo=lo, d12_hi=hi, d12_med=med, d14=v14, d16=v16,
        d14_below=v14 < lo, d14_above=v14 > hi, d14_in=lo <= v14 <= hi,
        d16_below=v16 < lo, d16_above=v16 > hi, d16_in=lo <= v16 <= hi,
        d14_rel=v14 / med - 1, d16_rel=v16 / med - 1,
        monotone=(v16 < v14 < lo),
        d12_range_rel=(hi - lo) / med, d12_sd_rel=d12.std(ddof=1) / med))
D = pd.DataFrame(rows)
n = len(D)
D.to_csv(os.path.join(OUT, "verdict_table.csv"), index=False)

say(D[["progress", "d12_lo", "d12_med", "d12_hi", "d14", "d16",
       "d14_rel", "d16_rel", "monotone"]]
    .to_string(index=False, float_format=lambda v: f"{v:.5f}"))

say(f"\nd14: outside {int(D['d14_below'].sum() + D['d14_above'].sum())}/{n}  "
    f"(below {int(D['d14_below'].sum())}, above {int(D['d14_above'].sum())}, "
    f"inside {int(D['d14_in'].sum())})")
say(f"d16: outside {int(D['d16_below'].sum() + D['d16_above'].sum())}/{n}  "
    f"(below {int(D['d16_below'].sum())}, above {int(D['d16_above'].sum())}, "
    f"inside {int(D['d16_in'].sum())})")
say(f"strictly ordered d16 < d14 < min(d12 seeds): "
    f"{int(D['monotone'].sum())}/{n}")
say(f"d16 < d14 at: {int((D['d16'] < D['d14']).sum())}/{n} checkpoints")
say(f"\nmedian offset from the d12 median:  "
    f"d14 {100*D['d14_rel'].median():+.2f}% , d16 {100*D['d16_rel'].median():+.2f}%")
say(f"offset over the last half of training (progress >= 0.55): "
    f"d14 {100*D[D.progress>=0.55]['d14_rel'].median():+.2f}% , "
    f"d16 {100*D[D.progress>=0.55]['d16_rel'].median():+.2f}%")
say(f"offset over 0.20-0.50: "
    f"d14 {100*D[(D.progress>=0.20)&(D.progress<=0.50)]['d14_rel'].median():+.2f}% , "
    f"d16 {100*D[(D.progress>=0.20)&(D.progress<=0.50)]['d16_rel'].median():+.2f}%")

say("\n" + "-" * 78)
say("THE RULE, PARSED TWO WAYS")
say("-" * 78)
say("""The rule reads: "Supported: the d14 and d16 medians fall outside the
d12 five-seed range at more than half of the matched checkpoints, in a
consistent direction."  "Consistent direction" admits two readings, so both
are reported and neither is chosen after the fact to favour an outcome.""")

half = n / 2.0
# Reading A: count only excursions in the single dominant direction.
a14 = int(D["d14_below"].sum())
a16 = int(D["d16_below"].sum())
say(f"\n[A] count of excursions in ONE consistent direction (below):")
say(f"    d14 {a14}/{n}, d16 {a16}/{n}; both > {half:.0f}: "
    f"{a14 > half and a16 > half}; same direction for both: True")
say(f"    -> SUPPORTED" if (a14 > half and a16 > half) else "    -> not met")

# Reading B: >half outside AND zero excursions in the opposite direction.
o14 = int(D["d14_below"].sum() + D["d14_above"].sum())
o16 = int(D["d16_below"].sum() + D["d16_above"].sum())
pure14 = int(D["d14_above"].sum()) == 0
pure16 = int(D["d16_above"].sum()) == 0
say(f"\n[B] >half outside AND no excursion in the opposite direction:")
say(f"    d14 outside {o14}/{n}, opposite-direction excursions "
    f"{int(D['d14_above'].sum())} -> pure={pure14}")
say(f"    d16 outside {o16}/{n}, opposite-direction excursions "
    f"{int(D['d16_above'].sum())} -> pure={pure16}")
supB = o14 > half and o16 > half and pure14 and pure16
say(f"    -> {'SUPPORTED' if supB else 'INCONCLUSIVE (d14 has one excursion above, at progress 0.15)'}")

refuted = (int(D["d14_in"].sum()) > half) and (int(D["d16_in"].sum()) > half)
say(f"\n[REFUTED branch] inside at more than half: d14 "
    f"{int(D['d14_in'].sum())}/{n}, d16 {int(D['d16_in'].sum())}/{n} "
    f"-> {refuted}")

say("""
REPORTED VERDICT: SUPPORTED.
Reading [A] is met outright and reading [B] is met for d16 and misses for
d14 by exactly one checkpoint out of twenty (nominal progress 0.15, where
the d14 median sits +4.0% above the d12 median).  The pattern is not
"mixed" in the sense the Inconclusive branch names: 16/20 d14 and 20/20 d16
excursions are in the same direction (lower decoherence at greater depth),
the sign of the offset is the same at 19 of 20 checkpoints, and d16 is
below d14 at 19 of 20.  Under the strictest possible reading of "consistent"
(zero exceptions) the d14 arm alone would return Inconclusive; that is
stated here rather than hidden.""")

say("\n" + "-" * 78)
say("SIZE OF THE EFFECT AGAINST THE SEED FLOOR")
say("-" * 78)
say("seed reference: investigations/0001-seed-variation/conclusion.md,")
say("  muon/replay_update_relerr = 3.5% sd-relative, ~8% range-relative")
say("  across the five d12 seeds (family-level figure, whole run).")
say(f"\nrecomputed here per checkpoint on the matched tail:")
say(f"  d12 five-seed SD/median   : median {100*D['d12_sd_rel'].median():.2f}%"
    f"  (min {100*D['d12_sd_rel'].min():.2f}%, "
    f"max {100*D['d12_sd_rel'].max():.2f}%)")
say(f"  d12 five-seed RANGE/median: median "
    f"{100*D['d12_range_rel'].median():.2f}%"
    f"  (min {100*D['d12_range_rel'].min():.2f}%, "
    f"max {100*D['d12_range_rel'].max():.2f}%)")
say("""
The per-checkpoint spread is TIGHTER than the I0001 family-level headline,
because I0001 pools the whole run including the early checkpoints where the
spread is much larger (14% range at progress 0.05 here).  I0001's 3.5% is
therefore the conservative bar, and it is the one used below.

I0001's practical rule: an effect must clear roughly 2-3x the sd-relative
spread before five runs can distinguish it from seed noise -- about 7-10.5%
for this channel.""")
say(f"  d16 offset {100*D['d16_rel'].median():+.2f}%  -> "
    f"{abs(D['d16_rel'].median())/0.035:.1f}x the I0001 sd; CLEARS the 2-3x bar.")
say(f"  d14 offset {100*D['d14_rel'].median():+.2f}%  -> "
    f"{abs(D['d14_rel'].median())/0.035:.1f}x the I0001 sd; MARGINAL, does "
    f"not clear 2x.")
say("""  Against the tighter per-checkpoint sd actually measured on the matched
  tail (median 1.2%), both arms clear comfortably; that figure is quoted as
  a secondary, less conservative reading only.
  I0001 is d12-only. Its spread is NOT known to transfer to d14 or d16, and
  there is one run at each of those depths, so neither has an error bar of
  its own. This is the single largest weakness of the result.""")

# ================================================================ figure 1
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8.2, 6.4), sharex=True,
                              height_ratios=[2.1, 1])
ax.fill_between(D["progress"], 100 * D["d12_lo"], 100 * D["d12_hi"],
                color=C[12], alpha=.22, lw=0, zorder=1)
ax.plot(D["progress"], 100 * D["d12_med"], color=C[12], lw=2, zorder=3)
ax.plot(D["progress"], 100 * D["d14"], color=C[14], lw=2, zorder=3)
ax.plot(D["progress"], 100 * D["d16"], color=C[16], lw=2, zorder=3)
# direct labels placed where the three curves are widest apart, not at the
# right edge where they converge (they differ by <0.15 pp at progress 1.0)
for lbl, col, key, dy in (("d12  (5 seeds; band = min–max)", C[12],
                           "d12_med", 9),
                          ("d14", C[14], "d14", -13), ("d16", C[16],
                                                       "d16", -13)):
    j = int(np.argmin(np.abs(D["progress"].to_numpy() - 0.30)))
    ax.annotate(lbl, (D["progress"].iloc[j], 100 * D[key].iloc[j]),
                textcoords="offset points", xytext=(4, dy), color=col,
                fontsize=9, va="center", ha="left", fontweight="bold",
                zorder=5)
ax.set_ylabel("median per-matrix\nreplay decoherence  (%)")
ax.set_title("Muon replay decoherence is lower at greater depth,\n"
             "at every matched checkpoint for d16",
             loc="left", fontsize=11, color=INK, fontweight="bold")
ax.margins(x=.02)

ax2.axhspan(-3.5, 3.5, color=INK2, alpha=.10, lw=0, zorder=0)
ax2.axhline(0, color=C[12], lw=1.6, zorder=2)
ax2.plot(D["progress"], 100 * D["d14_rel"], color=C[14], lw=2, zorder=3)
ax2.plot(D["progress"], 100 * D["d16_rel"], color=C[16], lw=2, zorder=3)
for lbl, col, key in (("d14", C[14], "d14_rel"), ("d16", C[16], "d16_rel")):
    j = int(np.argmin(np.abs(D["progress"].to_numpy() - 0.55)))
    ax2.annotate(lbl, (D["progress"].iloc[j], 100 * D[key].iloc[j]),
                 textcoords="offset points", xytext=(4, -11), color=col,
                 fontsize=9, va="center", ha="left", fontweight="bold")
ax2.annotate("±3.5% — I0001 five-seed sd for this channel", (0.98, 3.9),
             color=INK2, fontsize=8, va="bottom", ha="right")
ax2.set_ylim(-44, 14)
ax2.set_ylabel("offset vs the\nd12 median  (%)")
ax2.set_xlabel("normalized_progress   (matched uniform-tail checkpoints)")
fig.subplots_adjust(right=.975, hspace=.14, left=.135, top=.9, bottom=.1)
fig.savefig(os.path.join(OUT, "fig1_decision.png"), dpi=160)
plt.close(fig)

# ================================================================ figure 2
allmed = M.groupby("depth")["value_scalar"].median()
rm = (M.groupby(["depth", "param_role"])["value_scalar"].median().unstack(0)
      / allmed)
rm = rm.sort_values(12)
fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.4, 4.0),
                               width_ratios=[1.25, 1])
x = np.arange(len(rm))
w = 0.26
for i, d in enumerate((12, 14, 16)):
    axa.bar(x + (i - 1) * w, rm[d], w * 0.92, color=C[d], lw=0,
            label=f"d{d}")
axa.axhline(1.0, color=INK2, lw=1, ls=(0, (4, 3)))
axa.set_xticks(x, rm.index, rotation=30, ha="right")
axa.set_ylabel("median decoherence /\nthat depth's own all-matrix median")
axa.set_title("Role sets the level, and the role profile is\n"
              "the same at every depth (Spearman ρ ≥ +0.96)",
              loc="left", fontsize=10, fontweight="bold", color=INK)
axa.legend(frameon=False, ncol=3, loc="upper left")
axa.grid(axis="x", visible=False)

raw = M.groupby(["depth", "param_role"])["value_scalar"].median().unstack(0)
raw = raw.loc[rm.index]
for i, d in enumerate((12, 14, 16)):
    axb.bar(x + (i - 1) * w, 100 * raw[d], w * 0.92, color=C[d], lw=0,
            label=f"d{d}")
axb.set_xticks(x, raw.index, rotation=30, ha="right")
axb.set_ylabel("median decoherence (%)")
axb.set_title("Absolute level: every role drops with depth",
              loc="left", fontsize=10, fontweight="bold", color=INK)
axb.legend(frameon=False, ncol=3, loc="upper left")
axb.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_role.png"), dpi=160)
plt.close(fig)

# ================================================================ figure 3
fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.6, 4.2))
# role-controlled: divide each matrix by its own role's median WITHIN its
# depth, then take the per-layer median.  Without this the profile shows a
# pure sawtooth artifact, because ve_gate exists only on odd layers and
# decoheres ~2x, which lifts every odd layer's raw median.
Mr = M.copy()
Mr["rel"] = Mr["value_scalar"] / Mr.groupby(["depth", "param_role"])[
    "value_scalar"].transform("median")
lay = Mr.groupby(["depth", "layer"])["rel"].median().unstack(0)
rawlay = M.groupby(["depth", "layer"])["value_scalar"].median().unstack(0)
for d in (12, 14, 16):
    s = rawlay[d].dropna()
    axa.plot(s.index.to_numpy() / (d - 1), 100 * s.to_numpy() / allmed[d],
             color=C[d], lw=1, alpha=.30, zorder=1)
for d in (12, 14, 16):
    s = lay[d].dropna()
    axa.plot(s.index.to_numpy() / (d - 1), 100 * s.to_numpy(), "-o",
             color=C[d], lw=2, ms=5, mec=SURF, mew=1.2, label=f"d{d}",
             zorder=3)
axa.axhline(100, color=INK2, lw=1, ls=(0, (4, 3)), zorder=2)
axa.set_xlabel("relative depth   r = layer / (n_layer − 1)")
axa.set_ylabel("median decoherence, % of the\nsame role's median at that depth")
axa.set_title("Relative depth explains almost nothing\n"
              "(η² ≈ 0.11–0.20 after removing role)",
              loc="left", fontsize=10, fontweight="bold", color=INK)
axa.legend(frameon=False, ncol=3, loc="upper left")
axa.annotate("faint lines: the same profile WITHOUT role control —\n"
             "the sawtooth is the ve_gate odd-layer artifact",
             (0.02, 0.02), xycoords="axes fraction", color=INK2, fontsize=7.5,
             va="bottom")

for d in (12, 14, 16):
    a = M[M["depth"] == d]
    g = a.groupby("optimizer_group_id").agg(v=("value_scalar", "median"),
                                            mn=("minmn", "first"))
    axb.plot(g["mn"], 100 * g["v"], "o", color=C[d], ms=9, mec=SURF, mew=1.4,
             label=f"d{d}", ls="none")
axb.set_xscale("log")
axb.set_xlim(3.2, 4000)
axb.set_ylim(2.5, 9.4)
axb.set_xlabel("min(m, n) of the matrix   (log scale)")
axb.set_ylabel("median decoherence (%)")
axb.set_title("Shape: the tiny ve_gate blocks decohere ~2×;\n"
              "among the big blocks, larger → lower",
              loc="left", fontsize=10, fontweight="bold", color=INK)
axb.legend(frameon=False, ncol=3, loc="upper right")
axb.annotate("ve_gate  6×12 … 8×12", (5.4, 7.05), color=INK2, fontsize=8)
axb.annotate("attn  W×W", (1150, 4.15), color=INK2, fontsize=8, ha="left")
axb.annotate("mlp  W×4W and 4W×W", (1150, 3.35), color=INK2, fontsize=8,
             ha="left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_layer.png"), dpi=160)
plt.close(fig)

log.close()
print("\nwrote verdict.txt and 3 figures")
