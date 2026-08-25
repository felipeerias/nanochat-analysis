"""I0004 / A0001 — figures.

Palette: three ordered depth colors, validated with the dataviz six-checks
(light surface #fcfcfb): all PASS, worst adjacent CVD dE 9.3 (protan).
Threshold uses a reserved critical status color and is a reference line, not
a series.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

THRESH = 1e-4
DEPTH_COLOR = {12: "#3b6bd6", 14: "#c76b1f", 16: "#1f8a70"}
CRIT = "#b3261e"
INK, INK2, GRID = "#1b1b1a", "#5c5c58", "#dcdcd6"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "grid.color": GRID, "grid.linewidth": 0.6,
    "font.size": 9, "axes.titlesize": 10, "legend.frameon": False,
})


def tidy(ax, minor_labels=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="major", axis="y", alpha=0.7)
    ax.set_axisbelow(True)
    if not minor_labels:
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def fig1(df):
    s = df[(df.arm == "shadow_fp32") & (df.family == "e_sym")
           & (df.direction == "gradient")].sort_values("step")
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    for run, g in s.groupby("run"):
        d = int(g.depth.iloc[0])
        lw, a, z = (1.4, 0.55, 2) if d == 12 else (2.0, 1.0, 3)
        ax.plot(g.progress, g.value, color=DEPTH_COLOR[d], lw=lw, alpha=a,
                marker="o", ms=2.6, zorder=z)
    ax.axhline(THRESH, color=CRIT, lw=1.4, ls="--", zorder=1)
    ax.text(0.985, THRESH * 1.25, "acceptance threshold 1e-4", color=CRIT,
            ha="right", va="bottom", fontsize=8.5, transform=ax.get_yaxis_transform())
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("normalized progress (log)")
    ax.set_ylabel("shadow-fp32  e_sym_gradient  (relative)")
    ax.set_title("Self-consistency error along the gradient direction\n"
                 "only d16, and only in its first five updates, reaches 1e-4",
                 loc="left")
    handles = [plt.Line2D([], [], color=DEPTH_COLOR[d], lw=2,
                          label=f"d{d}" + (" (5 seeds)" if d == 12 else " (seed 7)"))
               for d in (12, 14, 16)]
    ax.legend(handles=handles, loc="lower left", ncol=3)
    tidy(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_esym_gradient_trajectories.png"), dpi=170)
    plt.close(fig)


def fig2(df):
    s = df[df.arm == "shadow_fp32"]
    med = s.groupby(["family", "direction", "depth", "run"])["value"].median()
    fams, dirs = ("e_sym", "e_lin"), ("random", "gradient", "update")
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.6), sharey=True, sharex=True)
    for i, fam in enumerate(fams):
        for j, dr in enumerate(dirs):
            ax = axes[i][j]
            m = med.loc[fam, dr]
            d12 = m.loc[12].values
            ax.fill_between([11.4, 16.6], d12.min(), d12.max(),
                            color=DEPTH_COLOR[12], alpha=0.13, lw=0, zorder=1)
            for depth in (12, 14, 16):
                v = m.loc[depth].values
                ax.plot([depth] * len(v), v, "o", ms=6,
                        color=DEPTH_COLOR[depth], zorder=3,
                        markeredgecolor=SURFACE, markeredgewidth=1.2)
            ax.axhline(THRESH, color=CRIT, lw=1.2, ls="--", zorder=2)
            ax.set_yscale("log")
            ax.set_ylim(2e-7, 3e-4)
            ax.set_xticks([12, 14, 16])
            ax.set_xlim(11.2, 16.8)
            ax.set_title(f"{fam}_{dr}", loc="left", pad=4)
            tidy(ax)
            if j == 0:
                ax.set_ylabel("per-run median (relative)")
            if i == 1:
                ax.set_xlabel("depth")
    axes[0][2].text(16.6, THRESH * 1.3, "1e-4", color=CRIT, ha="right",
                    va="bottom", fontsize=8.5)
    fig.suptitle("Per-run median self-consistency error vs depth, shadow-fp32 arm\n"
                 "shaded band = the five d12 seeds; every median sits ~100x "
                 "below the 1e-4 threshold", x=0.012, y=0.985, ha="left",
                 va="top", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.subplots_adjust(top=0.855)
    fig.savefig(os.path.join(FIG, "fig2_medians_vs_depth.png"), dpi=170)
    plt.close(fig)


def fig4(df):
    """Indexed to the d12 five-seed median, so the small trends are visible.

    Same medians as fig2; only the baseline changes. One axis, no dual scale.
    """
    s = df[df.arm == "shadow_fp32"]
    med = s.groupby(["family", "direction", "depth", "run"])["value"].median()
    fams, dirs = ("e_sym", "e_lin"), ("random", "gradient", "update")
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.4), sharey=True, sharex=True)
    for i, fam in enumerate(fams):
        for j, dr in enumerate(dirs):
            ax = axes[i][j]
            m = med.loc[fam, dr]
            d12 = m.loc[12].values
            base = float(pd.Series(d12).median())
            ax.fill_between([11.4, 16.6], d12.min() / base, d12.max() / base,
                            color=DEPTH_COLOR[12], alpha=0.15, lw=0, zorder=1)
            ax.axhline(1.0, color=GRID, lw=1.0, zorder=2)
            for depth in (12, 14, 16):
                v = m.loc[depth].values / base
                ax.plot([depth] * len(v), v, "o", ms=6,
                        color=DEPTH_COLOR[depth], zorder=3,
                        markeredgecolor=SURFACE, markeredgewidth=1.2)
            d16r = float(m.loc[16].iloc[0]) / base
            ax.annotate(f"{d16r:.2f}x", (16, d16r), xytext=(-4, 9),
                        textcoords="offset points", ha="right",
                        color=DEPTH_COLOR[16], fontsize=8.5)
            ax.set_yscale("log")
            ax.set_ylim(0.35, 3.2)
            ax.set_yticks([0.5, 1, 2, 3])
            ax.set_yticklabels(["0.5x", "1x", "2x", "3x"])
            ax.set_xticks([12, 14, 16])
            ax.set_xlim(11.2, 16.8)
            ax.set_title(f"{fam}_{dr}", loc="left", pad=4)
            tidy(ax, minor_labels=False)
            if j == 0:
                ax.set_ylabel("median / d12 seed median")
            if i == 1:
                ax.set_xlabel("depth")
    fig.suptitle("Same medians, indexed to the d12 five-seed median\n"
                 "shaded band = the d12 seed spread; a point outside it is "
                 "larger than seed noise", x=0.012, y=0.985, ha="left",
                 va="top", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.subplots_adjust(top=0.855)
    fig.savefig(os.path.join(FIG, "fig4_medians_indexed.png"), dpi=170)
    plt.close(fig)


def fig3(df):
    s = df[(df.arm == "shadow_fp32") & (df.family == "e_sym")
           & (df.direction == "gradient")].sort_values("step")
    s = s.groupby("run").head(8).copy()
    s["idx"] = s.groupby("run").cumcount()
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for run, g in s.groupby("run"):
        d = int(g.depth.iloc[0])
        lw, a = (1.4, 0.6) if d == 12 else (2.2, 1.0)
        ax.plot(g.idx, g.value, color=DEPTH_COLOR[d], lw=lw, alpha=a,
                marker="o", ms=4)
        if d != 12:
            ax.annotate(f"d{d}", (g.idx.iloc[-1], g.value.iloc[-1]),
                        xytext=(6, 0), textcoords="offset points",
                        color=DEPTH_COLOR[d], va="center", fontsize=9)
    ax.axhline(THRESH, color=CRIT, lw=1.4, ls="--")
    ax.text(0.02, THRESH * 1.25, "1e-4", color=CRIT, va="bottom", fontsize=8.5,
            transform=ax.get_yaxis_transform())
    ax.set_yscale("log")
    ax.set_xlabel("deep-checkpoint index (first eight; update 0,1,2,4,8,16,32,40)")
    ax.set_ylabel("shadow-fp32  e_sym_gradient")
    ax.set_title("The exceedance is a start-of-training transient\n"
                 "d16 is ~8x the d12 seed median there — but d14 sits below d12",
                 loc="left")
    handles = [plt.Line2D([], [], color=DEPTH_COLOR[d], lw=2,
                          label=f"d{d}" + (" (5 seeds)" if d == 12 else ""))
               for d in (12, 14, 16)]
    ax.legend(handles=handles, loc="lower left", ncol=3)
    tidy(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_early_transient.png"), dpi=170)
    plt.close(fig)


def main():
    df = pd.read_csv(os.path.join(HERE, "rows.csv"))
    df = df[df.family != "verdict_code"]
    fig1(df)
    fig2(df)
    fig3(df)
    fig4(df)
    print("wrote", sorted(os.listdir(FIG)))


if __name__ == "__main__":
    main()
