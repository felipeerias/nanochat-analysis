# I0008 — conclusion

Status: **closed**. Outcome: **unavailable as posed** — the hypothesis is
untested, not refuted. Evidence level: **reproduced** (both analyses reach the
same negative conclusion by independent routes).

Runs: [A0001](A0001/result.md) (Claude Code, Opus) and
[A0002](A0002/result.md) (Codex), both blind against protocol commit
`e76859c`.

## The proposal cannot be tested here, for two structural reasons

The first was known and stated in the protocol: **there are no data-group
labels**, so `g_k` has no `k`. Both analyses confirm it.

The second was **not** known, and A0001 found it: **this dataset contains
exactly one data ordering.** `nanochat/dataloader.py` contains no RNG at all —
it walks parquet files and row groups in index order into a deterministic
best-fit packer. The `--seed` flag seeds initialization only.

I verified this independently before accepting it. Across the five d12 seeds,
`batch/bos_count`, `batch/valid_targets` and `batch/mean_segment_length` are
**bitwise identical at every one of the 2,520 steps**, while `loss/train_mean`
differs (max 0.079). d12's stream is a prefix of d14's, which is a prefix of
d16's.

The consequence is stronger than a missing measurement: there is **zero
variation in the treatment**. Question 2 — would emphasizing a group at time t
have helped later? — is therefore unanswerable **in principle** on this
dataset, and would remain unanswerable on a future dataset that records far
more telemetry but keeps one fixed data order.

As A0001 puts it: the seven runs look like replications, but they are one data
ordering measured seven times.

## What the proxies did establish

**Data selection is a large effect on this recipe — larger than initialization
by an order of magnitude.** Because the five d12 seeds share the batch sequence
exactly, the component of step-to-step loss variation shared across seeds *is*
the batch effect. A0001 measures it at ICC 0.991, accounting for 99.1% of
step-to-step loss variance, with a standard deviation of 0.0232 nats —
**10.5x the I0001 seed floor**.

And the instrument explains almost none of it: the four recorded batch
descriptors give R² = 0.0037, with no persistence (lag-1 autocorrelation
−0.03). We can see that data selection matters a great deal, and we have
recorded nothing that says why.

**Gradient alignment between disjoint data slices collapses over training.**
A0002 measures the median cosine falling from 0.192 early to 0.021 late,
consistently across all seven runs. A0001 measures the same decay (0.21 to
0.015) *and* pins the measurement floor: the CountSketch cosine floor is
0.0086–0.016, established on disjoint-support role blocks where the true inner
product is exactly zero. So the late value sits **at** the floor — the decay is
real, but its endpoint is not resolvable with the current sketch size. The
sketched inner product itself is unbiased (median error +0.2% against an exact
identity).

**The recorded noise scale is optimistic by at least 8x.** A0001 found that
across-step decorrelation implies at least 1,400 rows where `noise/b_noise`
reports 162, because the sub-batches are a *clustered* sample drawn from a
1,000-document rolling buffer over consecutive row groups rather than an
independent one. This sharpens `DATASET.md` caveat 8 considerably: `b_noise`
must not be used to size a per-group estimator.

**A first-order value model is insufficient.** `s_k = lambda^T g_k` is
first-order; tested against realized loss change, `update/p1` gives R² = −0.57
(97% median magnitude error), while adding the curvature term gives R² = 0.87
(2.5%). The quadratic model that the bracket approximation rests on is, in
contrast, excellent late in training (normalized residual 2e-4).

**One finding was refuted in flight.** A0001 found a striking correlation
(r = −0.66, surviving a circular-shift null, reproducing in all seven runs)
between window batch composition and probe progress — and then showed it
evaporates under log-progress detrending. Reporting the refutation rather than
the correlation is exactly the discipline the logbook is for.

## The five questions

| | |
|---|---|
| Q1 relative usefulness changes over training | unavailable |
| Q2 retrospective estimate of future usefulness | **unavailable in principle** |
| Q3 which local quantities predict usefulness | partially supported — only `g^T Δ` carries signal (r = +0.78); `‖g‖²`, `g^T H g`, `η*`, `Δ^T H Δ` are near-useless alone |
| Q4 evidence for noncommutativity | unavailable — with one loss the bracket is identically zero; A0001 warns explicitly that `curvature/e_sym_*` is an arithmetic diagnostic, not a commutator, and must not be mistaken for one |
| Q5 minimal sufficient telemetry | unavailable as posed; a costed proposal is given instead |

## The instrumentation proposal, with measured costs

A0001 costed this against the recorded `overhead/total/*`, which is the right
way to do it.

- **Tier 0 (free, and unavoidable):** a loader sidecar carrying `group_id`;
  nominal proportions `p_k` in provenance; and **at least two data orderings**.
  The last is not telemetry at all, but without it Q2 stays unanswerable no
  matter what else is recorded.
- **Tier 1 (+3.4% wall clock, under 55 MB per run):** make accumulation
  microbatches group-pure, which turns `g_k` into a by-product at no cost; per
  group sketches and norms; store the G×G Gram matrix rather than raw sketches;
  emit the per-sub-batch quantities the instrument already computes and
  discards; a validation-probe gradient sketch for a myopic estimate of lambda;
  exact `g_k · g_j` calibration at a handful of deep steps; and an
  independent-draw noise estimate to replace the clustered one.
- **Tier 2 (+3%):** group-restricted HVPs in the shadow arm, giving
  `lambda^T (H_j g_i − H_i g_j)` on a designated handful of pairs.

**Sizing rule, derived from the data:** the rows per group needed for a
self-cosine of at least 0.5 rises from 15 early to 265 late — and those are
themselves at least 8x underestimates given the clustering finding. So `g_k`
would have to be accumulated with an EMA across steps late in training, and the
number of groups should be 4–8 coarse ones rather than tens.

## What this is worth

The honest answer to Sol is: not yet, and here is exactly why and what it would
take. The negative result is sharp rather than vague — one missing label, one
missing degree of freedom, both cheap to add — and the proxy work has already
established the measurement floors, the sizing rule, and the fact that data
selection is a 10x-larger effect than initialization on this recipe. That last
point is arguably the most useful thing to come out of the whole exercise for
future work on batch construction.
