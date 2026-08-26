# E01 — packing and composition screen

Status: **draft, not frozen**. Depends on telemetry v4.

## Question

Does how documents are packed into sequences, or how the corpus is composed,
move a held-out training outcome at all — and by how much relative to noise?

This is a screen, not a solution. It exists because every downstream data-side
question assumes the answer is yes, and that assumption is currently untested.

## Why a screen comes first

The three data-side problems are nested: packing determines what a sequence
is, batch selection determines what is in a step, and adaptive mixing
determines how composition changes over time. Each assumes the level below it
matters.

An adaptive-mixing experiment presupposes that group value differs between
groups and varies over time. We have not established that composition matters
at all. Testing an adaptive policy before that would risk a null result that
means "this particular policy did not help" rather than anything about
adaptive mixing — an uninformative use of the budget.

So the second factor here is a **static** composition change, not a controller.
Adaptive mixing becomes the next design, informed by an effect size and by the
interference structure the row-gradient Gram reveals.

## Design

Two factors at d12, architecture and optimizer recipe fixed:

| arm | packing | composition |
|---|---|---|
| A | current mixed best-fit | nominal |
| B | group-pure rows | nominal |
| C | current mixed best-fit | shifted static mixture |
| D | group-pure rows | shifted static mixture |

Blocking: 2 initialization seeds crossed with 3 data-order seeds = 6 blocks,
all four arms within each block. **24 runs, about $55 and 16 GPU-hours.**

Held fixed: architecture, optimizer recipe, token budget, schedule, dataset
snapshot, tokenizer, group taxonomy, probe banks, evaluation batches, and the
candidate-stream seed within each block.

Estimators: packing main effect ½[(B−A)+(D−C)]; composition main effect
½[(C−A)+(D−B)]; interaction D−C−B+A.

## Primary outcome

Precommitted before the first run: final validation bits-per-byte on the
**sealed** probe partition, plus a validation-trajectory summary. The sealed
partition is never read by any training process.

## Power, honestly

Six paired blocks detect roughly 1.14 within-block standard deviations at
conventional levels. Against the loss channel that is a small effect and the
design is adequate. Against curvature (25–29% relative standard deviation for
initialization alone, and the data-order floor is not yet known) it is not — so
curvature outcomes here are descriptive, not decisive.

The data-order noise floor is a **deliverable of this design**, not an input to
it: three order seeds per block give the first measurement of it.

## Confirmatory arms carried for free

- **Data-noise dominance (I0008).** Crossed initialization and order seeds are
  already in the design; compare clustered against independent noise estimates
  at the same θ and test whether order variation still dominates initialization
  variation.
- **The warmdown lock (I0005).** Preregister the change point on the new-order
  baselines. The stronger version adds a paired arm with a shifted warmdown
  onset and asks whether sharpening moves with it — that is a separate small
  design, not part of this factorial.

Not carried: the decoherence-versus-scale trend (I0003). It is confounded with
width and needs a fixed-depth width sweep, not cheap re-observation.

## What this does not answer

- How documents *should* be packed. Two policies is a screen, not a design
  space.
- Which subset of documents belongs in a batch. That requires the interference
  analysis the row-gradient Gram enables, as follow-on work on this data.
- How composition should change over time. No controller is present by design.
- Anything cross-depth or cross-architecture.

## Instrument dependencies

Telemetry v4 items 1, 2, 3, 4, 6, 7 — separated seeds, the lineage sidecar,
`batch_id` joins, sealed probe partitions, the sidecar verifier, and the
row-gradient Gram. Item 8 (independent noise) is required only because the
first confirmatory arm quotes noise scale.

## Open questions before freezing

1. What defines the group taxonomy? The corpus has no labels today; a coarse
   grouping (4–8 groups) must be defined and versioned before any composition
   factor is meaningful.
2. What shift does "shifted static mixture" mean, and is it large enough to
   detect while remaining a plausible training recipe?
3. Does group-pure-row packing materially change token yield? If it crops far
   more, the packing arm confounds composition with token waste, and that must
   be measured in the shakedown rather than assumed.
