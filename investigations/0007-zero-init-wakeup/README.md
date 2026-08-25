---
id: I0007
kind: question
status: open
data: sweep; all seven schema-v3 segments
selection: muon/replay_update_relerr and the Muon stage families at the early
  deep checkpoints; per matrix (parameter_name)
universe: all per-matrix Muon channels; report every matrix, not a sample
allowed inputs: DATASET.md; the seven v3 segments; ../../loader/;
  ../../profiles/; ../0001-seed-variation/conclusion.md
---

## Question

At initialization, many matrices have exactly zero decoherence because
zero-initialized output projections block backpropagation into them. In which
order do those matrices become active, and does the order depend on depth?

The deep schedule places checkpoints at steps 0, 1, 2, 4, 8, 16, 32, 40, 64
and so on, so the early window has good resolution.

## Test

1. For each matrix, find the first checkpoint at which its decoherence is
   nonzero, and the first at which its gradient norm is nonzero.
2. Order matrices by that wake-up point. Group by parameter role and by layer
   index.
3. Compare the ordering across the five d12 seeds: is it the same order every
   time, or does it vary? Quantify the agreement rather than asserting it.
4. Compare d12, d14 and d16: is wake-up governed by absolute step, by relative
   depth in the network, or by neither?

Exact zero is meaningful here and is not the same as a small value. Treat it
as a distinct state and say how you tested for it.

## Output

The wake-up ordering, its stability across seeds, and its behaviour across
depths.
