---
investigation: I0002
analyst: A0001
design: confirmatory
outcome: supported
saw: |
  investigations/0002-bf16-vs-fp32-curvature/README.md@e76859c (the protocol);
  analysis/README.md@e76859c; telemetry-data/sweep/DATASET.md;
  investigations/TEMPLATE-result.md; loader/telemetry_load.py@c0419ef (read for
  its filtering semantics; the analysis reads the parquet directly with pyarrow);
  investigations/0001-seed-variation/conclusion.md@4ac11f3;
  provenance.json of the eight sweep segments; the sparse tier of the seven
  schema-v3 segments. Did NOT read: the sibling blind analyst's directory
  (A0002/), any conclusion.md in this investigation, any other investigation's
  results, or profiles/ (permitted, but not needed; skipped deliberately).
data: |
  sweep; the seven schema-v3 segments —
  d12-s7-s0-6c217854f4174ce884dfb2b2dcc13c45,
  d12-s8-s0-2b2e72e4395440029b92226213d137bb,
  d12-s9-s0-b872e698e81b4a8fa0cd08512f43c2c2,
  d12-s10-s0-0ffeb05742bb4154a3f1afc202dc3955,
  d12-s11-s0-5fd1bf807f764954b30658872fc3e2ad,
  d14-s7-s0-b72acf7ac59942dab902e26fa0b2c80d,
  d16-s7-s0-8e0a39fc8f164343bd01ef6b915e849f.
  d12-iter (schema v1, no shadow arm) excluded as the protocol requires.
selection: |
  sparse tier; metric startswith curvature/ or update/; aggregation == "scalar";
  family present in BOTH acceptance arms. A pair is (segment, step, metric) with
  is_defined == True AND finite value_scalar in BOTH arms. No verdict
  conditioning (the protocol forbids it). 13,330 pair slots -> 11,366 pairs with
  both arms defined -> 10,770 with a defined relative difference (shadow != 0).
  215 deep checkpoints (30x5 + 32 + 33).
universe: |
  62 tested, 62 reported. 100 curvature/update families exist in sparse; 36 are
  vector-valued (aggregation "sweep:eps-ascending-v1") and fall outside the
  protocol's "scalar family" universe; 64 are scalar; 2 of those
  (curvature/native_verdict_code, curvature/shadow_verdict_code) exist in one arm
  only by construction, leaving 62 paired families. All 62 are in the table
  below, including the 9 that yield ZERO usable pairs. Nothing was dropped for
  being uninteresting.
code: (this commit):investigations/0002-bf16-vs-fp32-curvature/A0001/analyze.py
seed_reference: |
  investigations/0001-seed-variation/conclusion.md@4ac11f3 — used only for the
  depth statement, which is the one unpaired comparison here. The per-metric
  distortion table is paired within a checkpoint, so seed variation cancels and
  the reference does not apply to it. For the depth statement I also derive a
  direct, in-dataset error bar: the five-seed spread of the per-run median
  distortion at d12 (sd-relative), which is the right floor for this statistic.
supersedes: none
---

## Result

`kind: question`, so `outcome: supported` here means the declared test was
executable on the declared universe and produced determinate numbers — not that
a hypothesis survived. The depth sub-question is **inconclusive**; that is
stated below and not smoothed over.

**The headline.** bf16 arithmetic does not meaningfully distort the curvature
values themselves. Pooled over the nine curvature *quadratic-form* families
(`gHg`, `dhd`, `gg`, `Hg_norm`, `vhv_{gradient,random,update}`, `eta_star`,
`eta_star_rho`; 1,906 pairs), the typical relative distortion is

> **median |(native − shadow)| / |shadow| = 0.34 %**
> (cluster-bootstrap 95 % CI over runs: 0.29 %–0.38 %)

with a median *signed* distortion of **−0.004 %** — i.e. bf16 is essentially
unbiased, not systematically high or low. The IQR of the signed difference is
**−0.29 % … +0.41 %**, the 90th percentile of |rel| is 2.3 %, and the worst
single pair in the whole set is 77 %. **Sign disagreement is exactly 0 of 1,906
pairs**: the two arms never disagree about whether a curvature quadratic form is
positive or negative. Restricted to the five true energies (`gHg`, `dhd`,
`vhv_*`; 1,075 pairs) the number is the same: median |rel| 0.36 %, sign
disagreement 0 %.

**What bf16 does destroy is the acceptance apparatus, not the measurement.**
The same protocol, applied to the *error* and *floor* channels that decide
whether a measurement is trustworthy, gives distortions three to five orders of
magnitude large:

| class | families | pairs | median rel | median abs-rel | sign disagree |
|---|---:|---:|---:|---:|---:|
| quadratic (the curvature values) | 9 | 1,906 | −0.004 % | **0.34 %** | 0 % |
| update (update effectiveness) | 9 | 1,935 | 0.00 % | **0.54 %** | 5.3 % |
| snr (probe signal-to-noise) | 6 | 1,290 | **−99.8 %** | 99.8 % | 0.9 % |
| floor (arithmetic floors / epsilons) | 14 | 3,010 | **+654.7** | 654.7 | 0 % |
| error (symmetry / linearity residuals) | 9 | 1,935 | **+9,281** | 10,013 | 12 % |
| flag (booleans and verdict codes) | 6 | 1,290 | ordinal — see below | | 34 % |

(`median rel` is `(native − shadow)/|shadow|`, so +654.7 means native ≈ 656×
shadow, and +9,281 means native ≈ 9,282× shadow.)

That is the whole story in one line: **the measurement survives bf16 to about
0.3 %; the evidence that the measurement is valid does not survive at all.**
`curvature/arith_eps` and `curvature/eta_star_rho_threshold` differ by exactly
65,535 (a ratio of exactly 65,536 = 2¹⁶ = bf16 eps / fp32 eps) at every one of
the 215 checkpoints — those channels are set *by* the arm's arithmetic and are
definitional, not measurements. The symmetry and linearity residuals
(`e_sym_*`, `e_lin_*`) are roughly 6,800–14,500× larger in bf16, and the probe SNRs
collapse by ~99.8 % (a ~500× loss). This is the mechanism behind DATASET.md's
caveat 4: the native arm fails everywhere because its *error bars* explode, not
because its *values* are wrong.

**Availability loss is a separate, harder failure.** Nine of the 62 paired
families yield **zero** usable pairs, because the native arm is
`is_defined == False` at 215 of 215 checkpoints with reason
`noise_floor_inconclusive`: `curvature/e_curv_{gradient,random,update}`,
`curvature/e_fd_{gradient,random,update}`, `curvature/fd_cos_{gradient,random,update}`.
Two of those (`e_fd_gradient`, `fd_cos_gradient`) are defined in the shadow arm
at **215 of 215**. `curvature/fd_conclusive_gradient` is 0 in bf16 and 1 in fp32
at every checkpoint; `curvature/verdict_code_*` is 2 (failed) in bf16 at
215/215 in all three directions, while the shadow gradient direction passes at
186/215. `curvature/eta_star` is a milder case: fp32 loses 14 checkpoints
(`gHg_not_positive`), bf16 loses those same 14 plus 15 more
(`sign_below_noise`) — 201/215 defined in fp32 against 186/215 in bf16, an
incremental 7 % availability loss on the optimal-step-size channel. So for a third of the
paired families the honest answer to "how much does bf16 distort this?" is not a
percentage; it is **"bf16 cannot produce a value at all."**

**Does the distortion change over training? It splits cleanly by class.**
(Spearman ρ of |rel| against `normalized_progress`, pooled and per run; `*` =
permutation p < 0.01, 10k shuffles.)

- **Curvature quadratic forms: no — if anything it shrinks.** Pooled ρ = −0.074.
  The median family's late-half distortion is **0.83×** its early-half
  distortion. Four of nine families have ρ < 0 in all 7 runs; only `gg` and
  `Hg_norm` grow, and both grow weakly (1.58× and 1.05×). The mechanism is
  visible in the raw scale: for `gHg` the absolute error grows 0.054 → 1.06
  between the early and late halves, but `gHg` itself grows 7.3 → 145 — the
  *relative* precision is constant. There is an early transient: at
  `normalized_progress < 0.01` the median quadratic distortion is 0.58 %, and
  all six worst pairs in the class (32 %–77 %) are at steps 1–33, where the
  curvature is itself near zero.
- **Update effectiveness: yes, strongly and unanimously.** Every one of the
  eight non-degenerate `update/*` families has ρ > 0 in **7 of 7 runs**. Median
  late/early ratio across those families is **5.1×**; `update/actual` grows
  7.5×, `update/residual_p2` 34×, `update/normalized_residual` 32×. Again the
  mechanism is legible: `update/loss_before` and `update/loss_after` are each
  distorted by only ~0.025 %, but `update/actual` is their difference, and that
  difference shrinks 9× over training (median |actual| 0.052 → 0.0057) while its
  absolute error stays flat at ~9e-4. Catastrophic cancellation, not a curvature
  effect. The practical consequence: **at 7 of 215 checkpoints (3.3 %) the two
  arms disagree about the SIGN of the achieved loss decrease**, and six of those
  seven are at `normalized_progress ≥ 0.30`, four at ≥ 0.95. bf16 can report
  that a step made the loss go up when fp32 says it went down.

**Does the distortion change with depth? Inconclusive.** Pooled quadratic
median |rel| is 0.309 % at d12 (five seeds), 0.366 % at d14, 0.414 % at d16 —
monotone, d16/d12 = 1.34×, and 4 of 9 quadratic families are monotone up in
depth with 0 monotone down. But the five d12 seeds alone span 0.274 %–0.402 %
(sd-relative **17.5 %**), so d16 sits **1.84 sd** above the d12 five-seed mean
and d14 sits 0.96 sd above it. I0001@`4ac11f3` says an effect needs roughly
2–3× the sd-relative spread before five d12 runs can distinguish it from seed
noise; **1.84 sd is below that bar.** The `update/*` class is not even monotone
(d14 = 0.85 %, d16 = 0.54 %, d12 = 0.50 %). With n = 3 depths and one seed each
at d14/d16, I decline to claim a depth effect in either direction. The
suggestion of a mild increase is real enough to be worth re-testing on new runs;
it is not a finding.

**Two sanity checks that the pairing is correct.** `update/direction_norm` is
bit-identical between the arms at all 215 checkpoints (median rel exactly 0,
IQR exactly 0) — as it must be, since both arms measure the norm of the same
applied update from upcast endpoints. And `curvature/arith_eps` has ratio
exactly 65,536 at all 215. If the join were wrong, neither would hold.

**Verdict-conditioning sensitivity (secondary, and the protocol forbids using
it for the main test).** Conditioning on the 186 checkpoints where the *shadow*
gradient direction passed changes nothing material: `curvature/vhv_gradient`
goes from median |rel| 0.67 % (215 pairs) to 0.62 % (186 pairs), sign
disagreement 0 % either way. The distortion figures are not an artefact of
including uncertified checkpoints. `curvature/verdict_code_gradient` shows why
the protocol's prohibition was correct: native = failed at 215/215, so any
native-conditioned selection is empty.

### Every paired family

`class` is my own post-hoc grouping (see Limitations). `median rel` and `IQR of
rel` are the signed `(native − shadow)/|shadow|`; `median abs-rel` is the median
of its absolute value; `sign-disagree` is the fraction of pairs where the two
arms disagree on sign; `rho` is the pooled Spearman of |rel| against
`normalized_progress` (`*` = permutation p < 0.01); `runs w/ rho>0` counts the
7 runs whose within-run ρ is positive. Values ≥ 1 are printed as multiples,
values < 1 as percentages.

| family | class | pairs | median rel | IQR of rel | median abs-rel | sign-disagree | rho vs progress | runs w/ rho>0 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `curvature/gg` | quadratic | 215 | -0.00927% | -0.202% … 0.157% | 0.181% | 0.0% | +0.29* | 7/7 |
| `curvature/vhv_update` | quadratic | 215 | -0.0155% | -0.191% … 0.253% | 0.208% | 0.0% | -0.17 | 0/7 |
| `curvature/eta_star_rho` | quadratic | 215 | 0.0334% | -0.122% … 0.309% | 0.211% | 0.0% | -0.26* | 0/7 |
| `curvature/dhd` | quadratic | 215 | -0.00613% | -0.196% … 0.256% | 0.219% | 0.0% | -0.26* | 0/7 |
| `curvature/vhv_random` | quadratic | 215 | -0.104% | -0.498% … 0.329% | 0.435% | 0.0% | -0.16 | 1/7 |
| `curvature/Hg_norm` | quadratic | 215 | -0.0345% | -0.609% … 0.442% | 0.524% | 0.0% | +0.11 | 7/7 |
| `curvature/eta_star` | quadratic | 186 | 0.0188% | -0.66% … 0.587% | 0.605% | 0.0% | -0.03 | 4/7 |
| `curvature/vhv_gradient` | quadratic | 215 | 0.0514% | -0.568% … 1.14% | 0.673% | 0.0% | -0.20* | 0/7 |
| `curvature/gHg` | quadratic | 215 | 0.112% | -0.65% … 1.43% | 0.935% | 0.0% | -0.13 | 2/7 |
| `update/direction_norm` | update | 215 | 0% | 0% … 0% | 0% | 0.0% | n/a | — |
| `update/loss_before` | update | 215 | 0.00118% | -0.0191% … 0.029% | 0.0242% | 0.0% | +0.40* | 7/7 |
| `update/loss_after` | update | 215 | 0.00568% | -0.0203% … 0.0317% | 0.0281% | 0.0% | +0.34* | 7/7 |
| `update/p1` | update | 215 | 0.0033% | -0.347% … 0.352% | 0.351% | 0.5% | +0.46* | 7/7 |
| `update/p2` | update | 215 | -0.0151% | -0.483% … 0.444% | 0.461% | 0.5% | +0.34* | 7/7 |
| `update/residual_p1` | update | 215 | 0.396% | -2.22% … 3.73% | 2.77% | 2.8% | +0.36* | 7/7 |
| `update/actual` | update | 215 | 0.299% | -2.74% … 2.86% | 2.8% | 3.3% | +0.63* | 7/7 |
| `update/normalized_residual` | update | 215 | 2.15% | -52.3% … 1.75 | 77.9% | 20.5% | +0.40* | 7/7 |
| `update/residual_p2` | update | 215 | 2.94% | -52.4% … 1.83 | 83.7% | 20.5% | +0.42* | 7/7 |
| `curvature/c_fd_gradient` | error | 215 | -1.02% | -5.07% … 2.54% | 3.94% | 5.6% | -0.66* | 0/7 |
| `curvature/c_fd_random` | error | 215 | -1.56 | -1,243 … 498 | 688 | 53.0% | +0.37* | 7/7 |
| `curvature/c_fd_update` | error | 215 | -3.46 | -2,195 … 1,397 | 1,854 | 49.8% | +0.42* | 7/7 |
| `curvature/e_sym_gradient` | error | 215 | 6,834 | 2,013 … 23,620 | 6,834 | 0.0% | +0.03 | 4/7 |
| `curvature/e_lin_gradient` | error | 215 | 10,589 | 8,894 … 11,880 | 10,589 | 0.0% | +0.34* | 7/7 |
| `curvature/e_sym_random` | error | 215 | 11,859 | 4,957 … 36,931 | 11,859 | 0.0% | -0.14 | 3/7 |
| `curvature/e_sym_update` | error | 215 | 13,137 | 4,347 … 51,127 | 13,137 | 0.0% | -0.54* | 0/7 |
| `curvature/e_lin_update` | error | 215 | 14,464 | 12,917 … 17,460 | 14,464 | 0.0% | -0.80* | 0/7 |
| `curvature/e_lin_random` | error | 215 | 14,474 | 13,254 … 15,462 | 14,474 | 0.0% | -0.67* | 0/7 |
| `curvature/e_curv_gradient` | error | **0** | — | — | — | — | — | — |
| `curvature/e_curv_random` | error | **0** | — | — | — | — | — | — |
| `curvature/e_curv_update` | error | **0** | — | — | — | — | — | — |
| `curvature/e_fd_gradient` | error | **0** | — | — | — | — | — | — |
| `curvature/e_fd_random` | error | **0** | — | — | — | — | — | — |
| `curvature/e_fd_update` | error | **0** | — | — | — | — | — | — |
| `curvature/fd_cos_gradient` | error | **0** | — | — | — | — | — | — |
| `curvature/fd_cos_random` | error | **0** | — | — | — | — | — | — |
| `curvature/fd_cos_update` | error | **0** | — | — | — | — | — | — |
| `curvature/curv_snr_update` | snr | 215 | -94.5% | -98% … -89.5% | 94.5% | 2.3% | -0.67* | 0/7 |
| `curvature/curv_snr_random` | snr | 215 | -94.5% | -97.8% … -91.1% | 94.5% | 2.8% | -0.67* | 0/7 |
| `curvature/fd_snr_update` | snr | 215 | -99.4% | -99.8% … -99.2% | 99.4% | 0.0% | -0.88* | 0/7 |
| `curvature/curv_snr_gradient` | snr | 215 | -99.8% | -99.9% … -99.3% | 99.8% | 0.0% | -0.91* | 0/7 |
| `curvature/fd_snr_random` | snr | 215 | -99.9% | -99.9% … -99.9% | 99.9% | 0.0% | -0.55* | 0/7 |
| `curvature/fd_snr_gradient` | snr | 215 | -100% | -100% … -100% | 100% | 0.0% | -0.62* | 0/7 |
| `curvature/fd_eps_update` | floor | 215 | 0% | -66.7% … 0% | 0% | 0.0% | +0.39* | 7/7 |
| `curvature/fd_eps_random` | floor | 215 | -66.7% | -90% … 0% | 66.7% | 0.0% | +0.48* | 7/7 |
| `curvature/curv_eps_update` | floor | 215 | 0% | -70% … 2.33 | 90% | 0.0% | -0.15 | 1/7 |
| `curvature/curv_eps_random` | floor | 215 | 0% | -66.7% … 9 | 97% | 0.0% | -0.02 | 4/7 |
| `curvature/curv_eps_gradient` | floor | 215 | 10.6 | 4.77 … 21.9 | 10.6 | 0.0% | +0.91* | 7/7 |
| `curvature/fd_eps_gradient` | floor | 215 | 29 | 9 … 29 | 29 | 0.0% | +0.77* | 7/7 |
| `curvature/curv_floor_gradient` | floor | 215 | 487 | 128 … 1,967 | 487 | 0.0% | -0.90* | 0/7 |
| `curvature/fd_floor_gradient` | floor | 215 | 2,203 | 2,185 … 6,557 | 2,203 | 0.0% | -0.57* | 0/7 |
| `curvature/curv_floor_random` | floor | 215 | 65,520 | 655 … 589,859 | 65,520 | 0.0% | +0.07 | 4/7 |
| `curvature/arith_eps` | floor | 215 | 65,535 | 65,535 … 65,535 | 65,535 | 0.0% | n/a | — |
| `curvature/eta_star_rho_threshold` | floor | 215 | 65,535 | 65,535 … 65,535 | 65,535 | 0.0% | n/a | — |
| `curvature/curv_floor_update` | floor | 215 | 65,535 | 5,893 … 728,189 | 65,535 | 0.0% | +0.22* | 6/7 |
| `curvature/fd_floor_update` | floor | 215 | 66,144 | 65,530 … 196,904 | 66,144 | 0.0% | +0.37* | 7/7 |
| `curvature/fd_floor_random` | floor | 215 | 195,918 | 65,531 … 655,885 | 195,918 | 0.0% | +0.48* | 7/7 |
| `curvature/fd_conclusive_gradient` | flag | 215 | -1 | -1 … -1 | 1 | 100.0% | n/a | — |
| `curvature/fd_conclusive_random` | flag | 215 | -1 | -1 … -1 | 1 | 18.6% | n/a | — |
| `curvature/fd_conclusive_update` | flag | 215 | -1 | -1 … -1 | 1 | 0.9% | n/a | — |
| `curvature/verdict_code_gradient` | flag | 215 | 1 | 1 … 1 | 1 | 86.5% | +0.28 | — |
| `curvature/verdict_code_random` | flag | 215 | 1 | 1 … 1 | 1 | 0.0% | -0.11 | 0/7 |
| `curvature/verdict_code_update` | flag | 215 | 1 | 1 … 1 | 1 | 0.0% | +0.00 | 1/7 |

### Figures

- `figures/rel_distortion_all_families.png` — every paired family, |rel|
  distribution, coloured by class. The two-population structure is the point.
- `figures/distortion_vs_progress.png` — |rel| vs `normalized_progress` for the
  quadratic forms, by depth: flat-to-shrinking, with an early transient.
- `figures/native_vs_shadow.png` — the paired scatter for `gHg`, `dhd`,
  `eta_star`. Points sit on the identity line over four decades.
- `figures/availability.png` — the ten families where bf16 cannot define a value.
- `figures/depth_vs_seed_floor.png` — d14 and d16 against the d12 five-seed band.

## Limitations

1. **The class partition is exploratory.** The protocol declared one universe
   and one test, which I ran on all 62 families without selection — that part is
   confirmatory. But the split into quadratic / update / error / snr / floor /
   flag was made *after* seeing the numbers, as was the catastrophic-cancellation
   explanation. Both are plausible and mechanistically checkable, and I gave the
   supporting numbers, but neither was pre-registered. Treat the class-level
   pooled statistics as exploratory summaries of a pre-registered per-family
   table, not as pre-registered results.
2. **Relative difference is meaningless for three of the six classes.** For
   `flag` families (`fd_conclusive_*`, `verdict_code_*`) the values are booleans
   and ordinal codes; a ratio is nonsense and the "sign disagreement" of 100 %
   for `fd_conclusive_gradient` just encodes 0-vs-1. For `floor` families the
   difference is definitional (set by the arm's machine epsilon), not measured.
   I computed the protocol's statistic for them anyway, because the protocol says
   *every* paired scalar family and suppressing them would be selection — but
   they should be read as the contingency tables in the text, not as percentages.
3. **Vector families are outside the universe.** 36 of the 100 curvature/update
   families are `sweep:eps-ascending-v1` vectors (the ε-sweep diagnostics). The
   protocol's universe says "scalar family", so I excluded them. They would need
   a summary choice (which element? which norm?) that the protocol does not
   authorize. This is a real coverage gap: the ε-sweeps are exactly where the
   finite-difference floor behaviour lives, and a follow-up should cover them.
4. **The 0.34 % figure is a median over a heavy-tailed distribution.** 445 of
   1,906 quadratic pairs exceed 1 %, and 26 exceed 10 %. The tail is concentrated
   at `normalized_progress < 0.01`, where the curvature is near zero and the
   ratio is unstable by construction. "bf16 curvature is good to 0.3 %" is true
   typically and false at initialization.
5. **DATASET.md caveats that bite.** Caveat 4 (native curvature is uncertified
   everywhere) is the subject here rather than a limitation, and the protocol
   explicitly forbade conditioning on it — I complied, and reported the
   sensitivity check separately. Caveat 1 and 2 (size ray, n = 3 depths) are why
   the depth answer is inconclusive; depth co-varies with width, batch size, LR
   and horizon, so even a resolved trend would not be "depth causes X". Caveat 3
   (absolute warmups: 40-step LR warmup is 1.6 % of d12 progress but 0.7 % of
   d16) contaminates precisely the early-transient region where the quadratic
   tail lives. Caveat 9 (multiple comparisons): 62 families were tested; the
   per-family ρ values marked `*` are not corrected for 53 simultaneous tests,
   and I lean on the 7-of-7 within-run sign agreement rather than on p-values for
   the trend claim. Caveat 7 (compiled training is not bit-reproducible) means
   the ~1 ulp optimizer-moment race is inside my "native" arm; it is far below
   the 0.3 % effect.
6. **The shadow arm is not ground truth, it is a better-conditioned estimate.**
   It is an fp32 upcast of a model trained in bf16, measured along the actual
   applied update with TF32 off and rotary rebuilt. It shares the bf16 training
   trajectory and the same probe draws; it is the right *paired* comparator for
   "does the arithmetic distort the measurement", but it cannot answer "would
   fp32 training have had different curvature".
7. **`scipy` is unavailable**, so Spearman ρ, its permutation p-value (10k
   shuffles) and the cluster bootstrap (2,000 resamples over runs, seed
   20260825) are hand-rolled in `analyze.py`. Ranks use pandas average-rank
   ties. The bootstrap resamples *runs*, not rows, so the CI respects the
   within-run nesting; row-level CIs would be several times too narrow.
8. **Deviations from a literal protocol reading.** (a) The protocol's universe
   says "scalar family present in both arms"; `curvature/native_verdict_code`
   and `curvature/shadow_verdict_code` exist in one arm each by construction, so
   62 rather than 64 families are paired — I report both counts. (b) The
   protocol asks for "the fraction of pairs where the two arms differ in SIGN";
   I define that as `sign(native) != sign(shadow)` including exact zeros, and
   also report the fraction restricted to pairs where neither value is zero
   (`sign_disagree_frac_nonzero` in `per_metric.csv`), because zeros make the
   naive definition misleading for the flag families. (c) I did not read
   `profiles/`, which the protocol permits; the pairing is internal to each
   checkpoint and needed nothing from them.

## Files

`analyze.py` (the whole analysis, one file, deterministic given the seed) ·
`pairs.csv.gz` (all 11,366 pairs) · `per_metric.csv` (the full table with
bootstrap CIs) · `availability.csv` and `native_undefined_reasons.csv` ·
`per_metric_depth.csv` / `per_metric_depth_wide.csv` · `seed_spread.csv` ·
`depth_vs_seed_floor.csv` · `trend.csv` · `per_class.csv` ·
`verdict_sensitivity.csv` · `summary.json` · `run.log` (stdout of the run) ·
`table.md` (the table above) · `figures/`.
