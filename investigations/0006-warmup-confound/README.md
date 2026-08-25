---
id: I0006
kind: question
status: open
data: sweep; all seven schema-v3 segments
selection: all tiers; defined rows only; scalar families
universe: every scalar family present in all three depths; report all
allowed inputs: DATASET.md; the seven v3 segments; ../../loader/;
  ../../profiles/; ../0001-seed-variation/conclusion.md
---

## Question

Do the recipe's absolute warmup windows distort comparisons between depths?

The learning-rate warmup ends at step 40 and the Muon momentum ramp at step
400 in every run, regardless of depth. As a fraction of training that is about
16% at d12 and about 7% at d16. Any cross-depth difference concentrated inside
that window is an artifact of the schedule, not of scale.

## Test

For each family, compute the d12-to-d16 difference twice:
1. inside the warmup window (absolute step <= 400);
2. after it (absolute step > 400).

Use the five d12 seeds to form the d12 reference band, and cite I0001 for the
seed spread. Classify each family as warmup-dominated, uniformly different, or
not different.

Report separately what happens when runs are aligned on absolute step versus
on `normalized_progress`. These two alignments disagree precisely because the
warmup is absolute, and the size of that disagreement is the answer.

## Output

A list of families whose cross-depth difference is warmup-dominated. Those
families are unsafe for depth claims and should be flagged for every later
investigation.
