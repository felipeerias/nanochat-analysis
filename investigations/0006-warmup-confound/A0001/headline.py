"""I0006 / A0001 - stage 5: the numbers quoted in result.md."""

import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
S = pd.read_parquet(os.path.join(HERE, "series.parquet"))
P = pd.read_parquet(os.path.join(HERE, "points.parquet"))
F = pd.read_csv(os.path.join(HERE, "families.csv"))
META = json.load(open(os.path.join(HERE, "runs.json")))
D12 = sorted(r for r, m in META.items() if m["depth"] == 12)

lines = []


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    lines.append(s)


def ser(run, metric):
    d = S[(S.run == run) & (S.metric == metric)].sort_values("step")
    return d["step"].to_numpy(float), d["progress"].to_numpy(float), d["value"].to_numpy(float)


p("## 1. the recipe is a hybrid schedule: partly absolute, partly proportional")
for m in ("optim/lr", "optim/momentum", "optim/weight_decay"):
    s12, p12, v12 = ser("d12-s7", m)
    s16, p16, v16 = ser("d16-s7", m)
    # agreement on the step axis
    common = np.intersect1d(s12, s16)
    a = np.interp(common, s12, v12)
    b = np.interp(common, s16, v16)
    q = p16[(p16 >= p12.min()) & (p16 <= p12.max())]
    a2 = np.interp(q, p12, v12)
    b2 = np.interp(q, p16, v16)
    p(f"  {m}: peak value {v12.max():g}")
    p("    d16 minus d12, max |difference| in each phase")
    for lo, hi, lab in ((0, 41, "LR warmup      step   0- 40"),
                        (41, 401, "Muon ramp      step  41-400"),
                        (401, 883, "plateau        step 401-882"),
                        (883, 2521, "d12 warmdown   step 883-2520")):
        s = (common >= lo) & (common < hi)
        p(f"      STEP axis  {lab}: {np.abs(a-b)[s].max():.4e}")
    for lo, hi, lab in ((0, 40/2520, "p < 0.0159 (d12 still in LR warmup) "),
                        (40/2520, 400/2520, "0.0159 <= p < 0.1587 (d12 in ramp)"),
                        (400/2520, 0.35, "0.1587 <= p < 0.35   (both plateau)"),
                        (0.35, 1.0, "p >= 0.35            (both warmdown)")):
        s = (q >= lo) & (q < hi)
        p(f"      PROG axis  {lab}: {np.abs(a2-b2)[s].max():.4e}")
p("  -> LR warmup (40) and Muon momentum ramp (400) are ABSOLUTE;")
p("     the warmdown onset is PROPORTIONAL (progress 0.350 at every depth:")
for r in ("d12-s7", "d14-s7", "d16-s7"):
    lm = META[r]["landmarks"]
    p(f"       {r}: landmarks {lm} of N={META[r]['num_iterations']} -> "
      f"warmdown at p={lm[2]/META[r]['num_iterations']:.4f})")

p("")
p("## 2. the token stream is identical across all seven runs")
same = True
for m in ("batch/bos_count", "batch/valid_targets", "batch/mean_segment_length",
          "batch/segments_per_row_mean", "batch/segments_per_row_max"):
    piv = S[S.metric == m].pivot_table(index="step", columns="run", values="value")
    piv = piv.dropna()
    ok = bool((piv.nunique(axis=1) == 1).all())
    same &= ok
    p(f"  {m}: identical at every common step across all 7 runs: {ok} "
      f"({len(piv)} steps compared)")
p(f"  total_batch_size: {sorted({m['total_batch_size'] for m in META.values()})} "
  f"(one value for every depth) -> tokens_seen == step * 524288 in all runs")
p("  CONSEQUENCE: aligning on absolute step also holds the DATA fixed; aligning")
p("  on normalized progress compares d16 after 2.133x as many tokens as d12.")

p("")
p("## 3. matchability of the measurement grids (d12 vs d16)")
for tier in ("continuous", "periodic", "sparse"):
    a = np.sort(S[(S.run == "d12-s7") & (S.tier == tier)]["step"].unique())
    b = np.sort(S[(S.run == "d16-s7") & (S.tier == tier)]["step"].unique())
    com = np.intersect1d(a, b)
    p(f"  {tier:11s}: d16 samples inside the window (step<=400): "
      f"{int((b<=400).sum())}; common absolute steps there: {int((com<=400).sum())}")
p("  sparse deep checkpoints inside the window share the step-defined geometric")
p("  prefix {0,1,2,4,8,16,32,40,64} and the step-400 landmark, but the earliest")
p("  progress value both depths sample is p=0.05 (d12 step 126, d16 step 269):")
p("  BELOW p=0.05 the deep-checkpoint schedule offers NO cross-depth match on")
p("  normalized progress at all, only interpolation across the geometric gaps.")
p("  Under absolute-step alignment the comparison also stops at d12's horizon:")
p("  d16 steps 2521..5375 (53.1% of its training) have no d12 counterpart.")

p("")
p("## 4. headline channels")
hdr = (f"{'family':42s} {'tier':10s} {'n_w':>4s} {'|z|w_abs':>9s} {'|z|w_prog':>10s} "
       f"{'rel_w_abs':>10s} {'rel_w_prog':>11s} {'|z|p_abs':>9s} {'|z|p_prog':>10s} "
       f"{'rel_p_abs':>10s} {'rel_p_prog':>11s} {'seedsd_w':>9s}")
p(hdr)
show = ["loss/train_mean", "update/loss_before", "update/loss_after",
        "muon/replay_update_relerr", "update/direction_norm", "update/p2",
        "curvature/dhd", "curvature/gHg", "curvature/eta_star",
        "probe/loss", "param/norm", "noise/s2", "noise/b_noise",
        "muon/cos_raw_final"]
for m in show:
    for _, r in F[F.metric == m].iterrows():
        p(f"{m+'|'+r.arm:42s} {r.tier:10s} "
          f"{0 if np.isnan(r.w_n_abs) else int(r.w_n_abs):4d} "
          f"{r.w_medabsz_abs:9.2f} {r.w_medabsz_prog:10.2f} "
          f"{100*r.w_medrel_abs:9.2f}% {100*r.w_medrel_prog:10.2f}% "
          f"{r.p_medabsz_abs:9.2f} {r.p_medabsz_prog:10.2f} "
          f"{100*r.p_medrel_abs:9.2f}% {100*r.p_medrel_prog:10.2f}% "
          f"{100*r.w_seedsdrel_abs:8.3f}%")

p("")
p("## 5. loss/train_mean in detail (the best detector in the dataset, I0001)")
d = P[(P.metric == "loss/train_mean")]
for lo, hi, lab in ((0, 40, "LR warmup, step 0-40"),
                    (41, 400, "Muon ramp, step 41-400"),
                    (401, 882, "post-ramp, pre-warmdown"),
                    (883, 2520, "d12 in warmdown, d16 not"),
                    (2521, 5375, "beyond d12's horizon")):
    s = d[(d.step >= lo) & (d.step <= hi)]
    if not len(s):
        continue
    a = s[s.nok_abs == 5]
    b = s[s.nok_prog == 5]
    p(f"  {lab:28s} n_abs={len(a):4d} n_prog={len(b):4d} | "
      f"abs: rel={100*a.rel_abs.median() if len(a) else np.nan:7.2f}% "
      f"|z|={np.median(np.abs(a.z_abs)) if len(a) else np.nan:8.1f} | "
      f"prog: rel={100*b.rel_prog.median():7.2f}% "
      f"|z|={np.median(np.abs(b.z_prog)):8.1f}")
both = d[(d.nok_abs == 5) & (d.nok_prog == 5)]
flip = np.sign(both.rel_abs) != np.sign(both.rel_prog)
p(f"  sign of the d16-d12 loss difference disagrees between the two alignments "
  f"at {100*flip.mean():.1f}% of the {len(both)} shared comparison points "
  f"(all of them at step > {int(both[flip].step.min()) if flip.any() else -1})")
p(f"  d12 seed sd/median on loss/train_mean: window {100*d[d.warmup].sdrel12_abs.median():.3f}%, "
  f"after {100*d[~d.warmup].sdrel12_abs.median():.3f}% "
  f"(I0001 whole-run figure: 0.06% sd-relative)")

p("")
p("## 6. how far apart are the two candidate d12 references?")
p("  median |d12(step s) - d12(progress s/5376)| in pooled d12 seed sigmas")
for g in ("dynamics", "resource", "config"):
    s = F[F.group == g]
    p(f"  {g:9s} n={len(s):3d} window: median={s.w_align_dz.median():6.2f} "
      f"p90={s.w_align_dz.quantile(.9):7.2f} max={s.w_align_dz.max():9.1f} | "
      f"after: median={s.p_align_dz.median():6.2f} "
      f"p90={s.p_align_dz.quantile(.9):7.2f} max={s.p_align_dz.max():9.1f}")

p("")
p("## 7. verdict stability")
t = F[(F.verdict != "underpowered") & (F.verdict_prog != "underpowered")]
p(f"  {len(t)} families are testable under both alignments; "
  f"{int(t.verdict_flip.sum())} ({100*t.verdict_flip.mean():.0f}%) change verdict")
p(f"  warmup-dominated count: {int((F.verdict.str.startswith('warmup')).sum())} "
  f"under absolute-step alignment vs "
  f"{int((F.verdict_prog.str.startswith('warmup')).sum())} under progress alignment")
ct = pd.crosstab(t.verdict, t.verdict_prog)
p(ct.to_string())

open(os.path.join(HERE, "headline.txt"), "w").write("\n".join(lines) + "\n")
