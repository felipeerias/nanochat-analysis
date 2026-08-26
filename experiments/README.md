# Experiments

Designs for **generating** new data. Analysis of data we already have lives in
`../`; what the instrument records lives in `../telemetry-v4-plan.md`.

Keep these separate. The instrument is built once and serves many experiments,
so it is justified by the class of questions it makes answerable. An
experiment is justified by one question and a power calculation. Mixing them
makes the instrument special-purpose and makes experiments hostage to
engineering.

## What an experiment design must state

1. The question, and what result would answer it either way.
2. The factors, their levels, and what is held fixed.
3. The blocking structure and the number of runs.
4. The primary outcome, precommitted, and which probe partition measures it.
5. The power: what effect size this design can detect, against which noise
   floor, citing the reference that establishes that floor.
6. What the design does **not** answer.
7. The instrument capabilities it depends on, by version.

A design is frozen before the first run. Changing it afterwards makes a new
design with a new id, not an edit.

## Status

| id | design | runs | status |
|---|---|---|---|
| E01 | packing and composition screen | 24 | draft |
| E02 | warmdown to sharpening lock | 15 | draft |
| E03 | width versus decoherence (fixed depth) | 18 | draft |
| E04 | shape-preserving architecture geometry | 37 | draft |
| E05 | does Muon decoherence matter? | — | draft |
| E06 | spectral measurement pilot (offline) | 0 GPU training | draft |
| E07 | adaptive mixture | — | proposal, blocked on E01 |
| E08 | depth extension to d18/d20 | staged | draft — recommends NOT extending yet |
| E09 | sequence-length geometry (offline) | 0 GPU training | draft |

None are frozen. Each ends with open questions that must be settled first.
Several impose requirements back on the instrument — see the notes in
`../telemetry-v4-plan.md`.

## Standing constraints on any design here

- Cross-depth comparison is severely limited: 160 of 248 metric families are
  unsafe for depth claims, and the alignment axis (absolute step versus
  normalized progress) changes conclusions and sometimes their sign. Prefer
  fixed-depth designs. See `../investigations/0006-warmup-confound/`.
- The initialization noise floor is known per channel and is narrow: loss
  0.06%, decoherence 3.5%, curvature 25–29% relative standard deviation. It
  covers **initialization only** — a design that varies data order or batching
  must establish its own floor. See
  `../investigations/0001-seed-variation/`.
- Curvature is a weak outcome measure at realistic seed counts. Detecting a
  20% effect needs about 29 seeds per arm; 50% needs about 5.
- Nothing measured so far is confirmed. Every finding is a same-data
  reproduction, so confirmatory arms on new runs are worth building into
  designs that can carry them cheaply.
