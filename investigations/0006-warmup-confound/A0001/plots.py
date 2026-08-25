"""I0006 / A0001 - stage 6: figures."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
S = pd.read_parquet(os.path.join(HERE, "series.parquet"))
P = pd.read_parquet(os.path.join(HERE, "points.parquet"))
F = pd.read_csv(os.path.join(HERE, "families.csv"))
META = json.load(open(os.path.join(HERE, "runs.json")))
D12 = sorted(r for r, m in META.items() if m["depth"] == 12)

# validated categorical palette, fixed slot order (dataviz reference instance)
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"      # d12, d14, d16 / abs, prog
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#b8b7b2"
BAND = "#e9e9e6"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb", "font.size": 8.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 9,
    "axes.titleweight": "bold", "axes.grid": True, "grid.color": "#ececeb",
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6,
})


def ser(run, metric):
    d = S[(S.run == run) & (S.metric == metric)].sort_values("step")
    return d["step"].to_numpy(float), d["progress"].to_numpy(float), d["value"].to_numpy(float)


# ---------------------------------------------------------------- figure 1
fig, ax = plt.subplots(2, 2, figsize=(9.6, 5.4))
runs = [("d12-s7", C1, "d12"), ("d14-s7", C2, "d14"), ("d16-s7", C3, "d16")]
for j, (m, name) in enumerate((("optim/lr", "learning rate"),
                               ("optim/momentum", "Muon momentum"))):
    for k, (xlab, xi) in enumerate((("absolute step", 0), ("normalized progress", 1))):
        a = ax[j, k]
        for run, col, lab in runs:
            s, pr, v = ser(run, m)
            a.plot(s if xi == 0 else pr, v, color=col, label=lab)
        a.set_xlabel(xlab)
        a.set_ylabel(name)
        if xi == 0:
            a.set_xlim(0, 5376)
            a.axvspan(0, 400, color=BAND, zorder=0)
            a.text(430, a.get_ylim()[0], " absolute warmup window (step<=400)",
                   fontsize=7, color=INK2, va="bottom")
        else:
            a.axvspan(400 / 5376, 400 / 2520, color=BAND, zorder=0)
            a.text(400 / 2520 + .01, a.get_ylim()[0],
                   " phase-mismatch zone", fontsize=7, color=INK2, va="bottom")
        a.set_title(("identical for step 0-882, then d12 anneals first"
                     if xi == 0 else
                     "identical for p>=0.159, mismatched below") if j == 0 else "")
ax[0, 0].legend(loc="lower right", ncols=3)
fig.suptitle("The recipe is a hybrid schedule: the warmups are absolute, the "
             "warmdown is proportional", fontsize=10, fontweight="bold", y=0.99)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig1-schedule.png"), dpi=170)
plt.close(fig)

# ---------------------------------------------------------------- figure 2
fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.5))
m = "loss/train_mean"
band = np.vstack([ser(r, m)[2] for r in D12])
s12, p12, _ = ser(D12[0], m)
s16, p16, v16 = ser("d16-s7", m)
for k, (xl, x12, x16) in enumerate((("absolute step (= tokens/524288)", s12, s16),
                                    ("normalized progress", p12, p16))):
    a = ax[k]
    a.fill_between(x12, band.min(0), band.max(0), color=C1, alpha=.35, lw=0,
                   label="d12, 5-seed range")
    a.plot(x12, np.median(band, 0), color=C1, lw=1.2, label="d12 median")
    a.plot(x16, v16, color=C3, label="d16")
    a.set_xlabel(xl)
    a.set_ylabel("train loss")
    a.set_ylim(2.6, 4.2)
    if k == 0:
        a.axvspan(0, 400, color=BAND, zorder=0)
    else:
        a.axvspan(400 / 5376, 400 / 2520, color=BAND, zorder=0)
    a.set_title("aligned on step (= same tokens, same schedule):\ncurves nearly coincide and cross at step 1667"
                if k == 0 else "aligned on progress (= 2.13x more tokens for d16):\nd16 far ahead everywhere")
ax[0].legend(loc="upper right")

d = P[P.metric == m]
a = ax[2]
b = d[d.nok_abs == 5]
a.plot(b.step, 100 * b.rel_abs, color=C1, lw=1.1, label="aligned on absolute step")
b = d[d.nok_prog == 5]
a.plot(b.step, 100 * b.rel_prog, color=C2, lw=1.1, label="aligned on progress")
a.axhline(0, color=INK2, lw=0.8)
a.axvspan(0, 400, color=BAND, zorder=0)
a.set_xlabel("d16 absolute step")
a.set_ylabel("(d16 - d12) / d12   [%]")
a.set_title("the same d16 sample, two answers")
a.legend(loc="lower right")
a.annotate("d16 becomes WORSE than d12\nat step 1667 on this axis",
           xy=(1667, 0), xytext=(2050, -17.5), fontsize=7, color=INK2,
           arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
a.set_ylim(-31, 6)
fig.suptitle("loss/train_mean - the same d16 run, two reference alignments",
             fontsize=10, fontweight="bold", y=1.0)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig2-loss.png"), dpi=170)
plt.close(fig)

# ---------------------------------------------------------------- figure 3
fig, ax = plt.subplots(1, 2, figsize=(9.2, 4.0))
d = F[(F.group == "dynamics") & (F.verdict != "underpowered")]
for k, (xc, yc, lab) in enumerate(((("w_medabsz_abs"), ("w_medabsz_prog"),
                                    "inside the window (step <= 400)"),
                                   (("p_medabsz_abs"), ("p_medabsz_prog"),
                                    "after the window (step > 400)"))):
    a = ax[k]
    x = d[xc].replace(0, np.nan)
    y = d[yc].replace(0, np.nan)
    a.scatter(x, y, s=22, color=C1, alpha=.7, lw=.8, edgecolor="#fcfcfb")
    lim = [1e-2, 3e2]
    a.plot(lim, lim, color=MUTED, lw=1, ls="--")
    a.axhline(3, color=C2, lw=1)
    a.axvline(3, color=C2, lw=1)
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlim(*lim)
    a.set_ylim(*lim)
    a.set_xlabel("|z| aligned on absolute step")
    a.set_ylabel("|z| aligned on normalized progress")
    a.set_title(lab)
    above = int((y > x).sum())
    a.text(.03, .96, f"{above}/{int((x.notna()&y.notna()).sum())} families sit "
           f"above the diagonal\n(progress alignment inflates the effect)\n"
           f"orange lines: the 3-sigma seed band (I0001)",
           transform=a.transAxes, va="top", fontsize=7.5, color=INK2)
fig.suptitle("Effect size in d12 seed sigmas, same families, two alignments",
             fontsize=10, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig3-alignment.png"), dpi=170)
plt.close(fig)

# ---------------------------------------------------------------- figure 4
fig, ax = plt.subplots(2, 1, figsize=(9.2, 3.4), sharex=False)
d12s = np.sort(S[(S.run == "d12-s7") & (S.tier == "sparse")]["step"].unique())
d16s = np.sort(S[(S.run == "d16-s7") & (S.tier == "sparse")]["step"].unique())
d12p = np.sort(S[(S.run == "d12-s7") & (S.tier == "sparse")]["progress"].unique())
d16p = np.sort(S[(S.run == "d16-s7") & (S.tier == "sparse")]["progress"].unique())
ax[0].eventplot([d12s[d12s <= 420], d16s[d16s <= 420]], colors=[C1, C3],
                lineoffsets=[1, 0], linelengths=.7)
ax[0].set_yticks([1, 0], ["d12", "d16"])
ax[0].set_xlabel("absolute step")
ax[0].set_xlim(-5, 420)
ax[0].set_title("deep checkpoints, absolute-step axis: the geometric prefix "
                "0,1,2,4,8,16,32,40,64 lines up exactly")
sel12 = d12p[d12p <= 0.08]
sel16 = d16p[d16p <= 0.08]
ax[1].eventplot([sel12, sel16], colors=[C1, C3], lineoffsets=[1, 0], linelengths=.7)
ax[1].set_yticks([1, 0], ["d12", "d16"])
ax[1].set_xlabel("normalized progress")
ax[1].set_xlim(-0.001, 0.08)
ax[1].set_title("same checkpoints, progress axis: nothing lines up below "
                "p = 0.05")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig4-grid.png"), dpi=170)
plt.close(fig)
print("wrote figures to", FIG)
