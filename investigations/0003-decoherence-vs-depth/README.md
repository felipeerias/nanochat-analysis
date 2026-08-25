---
id: I0003
kind: hypothesis
status: closed
data: sweep; all seven schema-v3 segments
selection: muon/replay_update_relerr; defined rows only; per matrix
  (parameter_name); aligned on normalized_progress
universe: the decoherence family and all of its per-matrix channels
allowed inputs: DATASET.md; the seven v3 segments; ../../loader/;
  ../../profiles/; ../0001-seed-variation/conclusion.md
---

## Claim to test

The divergence between the compiled bf16 Muon update and its eager reference
decomposition (`muon/replay_update_relerr`) changes with model depth.

## Decision rule, fixed before looking

d12 has five seeds; d14 and d16 have one run each. At matched
`normalized_progress`, compute the d12 across-seed range for each checkpoint.

- **Supported**: the d14 and d16 medians fall outside the d12 five-seed range
  at more than half of the matched checkpoints, in a consistent direction.
- **Refuted**: they fall inside the d12 range at more than half of checkpoints.
- **Inconclusive**: anything else, including a mixed or non-monotone pattern.

I0001 gives 3.5% as the standard-deviation seed spread of this family, which
makes it one of the few channels with power at one run per depth. Cite it.

## Also report

Per-matrix structure: is decoherence a function of relative depth in the
network (layer index divided by depth), of parameter role, or of matrix shape?
Depth changes the number of layers, so state clearly which normalization you
used before comparing.

Report the init-time structural zeros separately: at step 0 many matrices have
exactly zero decoherence because zero-initialized projections block backprop.
Including them would bias any average.

## Output

The decision above, with the evidence. A per-matrix picture of how
decoherence is distributed within a model, and whether that distribution is
preserved across depths.
