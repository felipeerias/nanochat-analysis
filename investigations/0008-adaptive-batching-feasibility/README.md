---
id: I0008
kind: question
status: closed
data: sweep; all seven schema-v3 segments
selection: as required by the assessment; state what you used
universe: the full metric inventory is in scope; the deliverable is a mapping,
  so report every theoretical quantity you assessed, including the ones with
  no available proxy
allowed inputs: DATASET.md; the seven v3 segments; ../../loader/;
  ../../profiles/; ../0001-seed-variation/conclusion.md;
  ../../../nanochat/nanochat/telemetry.py (to see what is measured and how);
  ../../../nanochat/docs/telemetry-spec.md; and the question document in
  ../../experiments/sol-batch-construction-question.txt
---

## Question

The question document proposes an optimal-control model of adaptive data
mixing: a costate lambda(t), a per-group value s_k(t) = lambda^T g_k(t), a
sampling policy q_k proportional to p_k exp(beta s_k), and a prediction that
changes in relative group value are governed by the noncommutativity of group
gradient fields, approximated by H_j g_i - H_i g_j.

Can this dataset support any of it, and if not, what instrumentation would?

## Start from the honest constraint

**This dataset records no data-group labels.** There is no loader sidecar: no
document identity, no domain or source, no crop flags. See caveat 8 in
DATASET.md. Any notion of "group" must therefore be constructed from what
exists, or declared unavailable.

Do not manufacture groups that the data does not contain. If a quantity cannot
be estimated, say so and move on. A clear negative result here is the useful
outcome, because it defines what a future run must record.

## Test

1. Build a table: for each theoretical quantity in the question document
   (g_k, |g_k|, g_i^T g_j, g_k^T g_nominal, H_k, lambda, s_k, the Lie bracket
   proxy), state what the dataset offers, how good a proxy it is, and what it
   cannot do.
2. Where a proxy exists, use it. Consider at least: the K-way sub-batch
   gradient capture used for the noise-scale estimator (disjoint data slices,
   though randomly drawn rather than semantic); count-sketched gradients,
   which support cosines between gradients; per-role and per-layer gradient
   decompositions; the HVP machinery, which can apply H to a chosen direction;
   and the update-effectiveness records, which relate an applied update to the
   loss change that followed.
3. Answer the document's five questions to whatever extent the data allows,
   marking each answer as supported, partially supported, or unavailable.
4. Specify the minimum additional instrumentation that would make the
   hypothesis testable: exactly what to record, at what cadence, and at what
   estimated cost.

## Output

A feasibility mapping, whatever partial empirical results the proxies support,
and a concrete instrumentation proposal. State plainly which of the five
questions this dataset can and cannot answer.
