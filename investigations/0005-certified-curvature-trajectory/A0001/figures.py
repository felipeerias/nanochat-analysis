"""I0005 / A0001 - figures.

Static PNGs for a versioned result.md (no interaction layer, light surface
only - a deliberate deviation from the dataviz hover/dark-mode defaults,
because the artefact is a file in a git repo, not a page).

Palette: categorical slots 1 and 2 of the reference palette,
#2a78d6 (certified) and #eb6834 (excluded), validated with
scripts/validate_palette.js --mode light: all six checks PASS
(worst adjacent CVD dE 24.7, normal-vision dE 33.6).

Seed identity carries no meaning here (the five d12 runs differ only in
seed), so the five runs are drawn as one muted family, never as five
categorical hues. Excluded checkpoints are drawn as hollow markers and are
NEVER connected by a line: the protocol forbids interpolating across them.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
SEED = "#9aa3ad"
BAND = "#c9d7ee"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e1"
SURFACE = "#fcfcfb"
WARMDOWN = 0.35
D12 = ["d12-s7", "d12-s8", "d12-s9", "d12-s10", "d12-s11"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9.5,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
})


def load():
    v = pd.read_csv(os.path.join(OUT, "certified_values.csv"))
    v = v[v.is_defined].copy()
    v["p"] = v.normalized_progress.round(6)
    g = pd.read_csv(os.path.join(OUT, "gate_table.csv"))
    g["p"] = g.p.round(6)
    return v[v.run.isin(D12)], g[g.run.isin(D12)]


def panel(ax, cert, excl, metric, title, ylabel, log=True,
          excl_label="uncertified (excluded)"):
    sub = cert[cert.metric == metric]
    ps = sorted(sub.p.unique())
    # across-seed band on checkpoints present in ALL five runs
    keep = [p for p in ps if sub[sub.p == p].run.nunique() == 5]
    med = np.array([sub[sub.p == p].value.median() for p in keep])
    lo = np.array([sub[sub.p == p].value.min() for p in keep])
    hi = np.array([sub[sub.p == p].value.max() for p in keep])
    ax.fill_between(keep, lo, hi, color=BAND, alpha=0.85, lw=0,
                    label="across-seed min-max (n=5)", zorder=1)
    for run in D12:
        s = sub[sub.run == run].sort_values("p")
        ax.plot(s.p, s.value, color=SEED, lw=0.8, alpha=0.85, zorder=2,
                label="individual seeds (5 runs)" if run == D12[0] else None)
    ax.plot(keep, med, color=BLUE, lw=2.0, zorder=4,
            label="across-seed median", solid_capstyle="round")
    if excl is not None and len(excl):
        ax.scatter(excl.p, excl.val, s=26, facecolors="none",
                   edgecolors=ORANGE, linewidths=1.3, zorder=5,
                   label=excl_label)
    ax.axvline(WARMDOWN, color=INK2, lw=0.8, ls=(0, (4, 3)), alpha=0.55,
               zorder=0)
    ax.set_xscale("log")
    if log:
        ax.set_yscale("log")
    ax.grid(True, which="major", axis="y", zorder=0)
    ax.set_title(title, color=INK, loc="left", pad=6)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("normalized progress")
    ax.set_xlim(3e-3, 1.15)


def fig1(cert, gate):
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.2))
    specs = [
        ("curvature/gHg", "gHg — curvature along the raw gradient",
         "$g^{\\mathsf{T}}Hg$"),
        ("curvature/vhv_gradient",
         "vhv_gradient — curvature per unit gradient direction",
         "$\\hat g^{\\mathsf{T}}H\\hat g = gHg/gg$"),
        ("curvature/eta_star", "eta* — quadratic-model step $gg/gHg$",
         "$\\eta^{*}$"),
        ("curvature/dhd", "dhd — curvature along the applied update",
         "$\\Delta^{\\mathsf{T}}H\\Delta$"),
    ]
    for ax, (m, t, yl) in zip(axes.ravel(), specs):
        panel(ax, cert, None, m, t, yl)
    axes[0, 0].legend(loc="upper left", ncols=1)
    axes[0, 0].text(WARMDOWN * 1.08, 0.03, "warmdown starts\n(p = 0.35)",
                    transform=axes[0, 0].get_xaxis_transform(),
                    color=INK2, fontsize=7)
    fig.suptitle("Certified curvature over training, five d12 seeds",
                 x=0.008, y=0.985, ha="left", color=INK, fontsize=12)
    fig.text(0.008, 0.955,
             "shadow_fp32 arm, gradient direction only, checkpoints whose "
             "per-direction verdict passed: 26 of 30 per run (25 in d12-s9), "
             "25 common to all five.\nThe four uncertified checkpoints at the "
             "head of training are not shown and are never interpolated "
             "across — see fig3.",
             ha="left", va="top", color=INK2, fontsize=7.5, linespacing=1.5)
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    fig.savefig(os.path.join(FIG, "fig1-trajectories.png"), dpi=170)
    plt.close(fig)


def fig3(gate):
    """What the two gates remove, and why it matters for eta*."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    for ax, col, title, yl in (
            (axes[0], "curvature/gHg",
             "gHg at the head of training crosses zero", "$g^{\\mathsf{T}}Hg$"),
            (axes[1], "curvature/eta_star",
             "eta* explodes exactly where gHg is near zero", "$\\eta^{*}$")):
        head = gate[gate.p < 0.02].copy()
        c = head[head.certified]
        u = head[~head.certified]
        ax.scatter(c.p, c[col].abs(), s=26, color=BLUE, zorder=4,
                   label="certified (26/30 per run)")
        pos = u[u[col] > 0]
        neg = u[u[col] < 0]
        ax.scatter(pos.p, pos[col], s=30, facecolors="none", edgecolors=ORANGE,
                   linewidths=1.3, zorder=5, label="uncertified, value > 0")
        ax.scatter(neg.p, neg[col].abs(), s=34, marker="x", color=ORANGE,
                   linewidths=1.3, zorder=5, label="uncertified, value < 0")
        miss = u[u[col].isna()]
        if len(miss):
            ax.scatter(miss.p, np.full(len(miss), 1e-5), s=30, marker="_",
                       color=ORANGE, linewidths=1.3, zorder=5,
                       label="undefined by the reliable-sign gate")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.grid(True, axis="y")
        ax.set_title(title, color=INK, loc="left", pad=6)
        ax.set_ylabel(yl + "   (absolute value)")
        ax.set_xlabel("normalized progress")
        ax.legend(loc="lower right" if col.endswith("gHg") else "upper right")
    fig.text(0.008, 0.01,
             "Every d12 shadow deep checkpoint with p < 0.02, all five seeds. "
             "Dashes on the eta* floor mark checkpoints where no value exists.\n"
             "The reliable-sign gate removes 12 of 150 (all of them gHg <= 0). "
             "It does NOT remove the 9 small-positive-gHg points whose eta* "
             "reaches 750-4700 —\nthe per-direction acceptance verdict is what "
             "removes those, and it removes all 21 head checkpoints.",
             ha="left", va="bottom", color=INK2, fontsize=7.5, linespacing=1.6)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(os.path.join(FIG, "fig3-excluded-head.png"), dpi=170)
    plt.close(fig)


def fig2(cert):
    """Where the gHg rise comes from, and whether the instrument drifted.
    Both panels are single-axis; the left one indexes every series to its
    value at the first common checkpoint so three quantities of different
    units share one scale honestly (never a second y-axis)."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    ax = axes[0]
    base_p = 0.006746
    series = [("curvature/gHg", BLUE, 2.0, "gHg"),
              ("curvature/gg", ORANGE, 1.6, "gg  (gradient norm$^2$)"),
              ("curvature/vhv_gradient", AQUA, 1.6,
               "vhv_gradient  (= gHg/gg)")]
    for m, c, lw, lab in series:
        sub = cert[cert.metric == m]
        ps = sorted(p for p in sub.p.unique()
                    if sub[sub.p == p].run.nunique() == 5)
        med = np.array([sub[sub.p == p].value.median() for p in ps])
        b = med[ps.index(base_p)]
        ax.plot(ps, med / b, color=c, lw=lw, label=lab,
                solid_capstyle="round")
    ax.axhline(1.0, color=GRID, lw=1.0, zorder=0)
    ax.axvline(WARMDOWN, color=INK2, lw=0.8, ls=(0, (4, 3)), alpha=0.55)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.grid(True, axis="y")
    ax.set_title("gHg rises faster than the gradient norm can explain",
                 color=INK, loc="left", pad=6)
    ax.set_ylabel("across-seed median, indexed to p = 0.0067")
    ax.set_xlabel("normalized progress")
    ax.set_xlim(5e-3, 1.15)
    ax.legend(loc="upper left")

    ax = axes[1]
    sub = cert[cert.metric == "curvature/e_curv_gradient"]
    ps = sorted(p for p in sub.p.unique() if sub[sub.p == p].run.nunique() == 5)
    med = np.array([sub[sub.p == p].value.median() for p in ps])
    lo = np.array([sub[sub.p == p].value.min() for p in ps])
    hi = np.array([sub[sub.p == p].value.max() for p in ps])
    ax.fill_between(ps, lo, hi, color=BAND, alpha=0.85, lw=0,
                    label="across-seed min-max (n=5)")
    ax.plot(ps, med, color=BLUE, lw=2.0, label="across-seed median",
            solid_capstyle="round")
    ax.axvline(WARMDOWN, color=INK2, lw=0.8, ls=(0, (4, 3)), alpha=0.55)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.grid(True, axis="y")
    ax.set_title("e_curv_gradient — the certification error does not drift",
                 color=INK, loc="left", pad=6)
    ax.set_ylabel("relative curvature error")
    ax.set_xlabel("normalized progress")
    ax.set_xlim(5e-3, 1.15)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2-decomposition.png"), dpi=170)
    plt.close(fig)


def main():
    cert, gate = load()
    fig1(cert, gate)
    fig2(cert)
    fig3(gate)
    print("wrote", sorted(os.listdir(FIG)))


if __name__ == "__main__":
    main()
