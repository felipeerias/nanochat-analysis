---
id: F0000
question: Q0000            # or: none
design: exploratory        # exploratory | confirmatory
outcome: inconclusive      # supported | refuted | inconclusive | superseded
analyst: A0000             # this analysis run
saw: DATASET.md, the parquet, F0001    # what this run was allowed to read
data: sweep-d12-d16; segments d12-s7, d12-s8, d12-s9, d12-s10, d12-s11
selection: defined rows only; arm=shadow_fp32; direction=gradient; verdict=passed
universe: 12 metric families tested; 1 reported
code: <commit>:<path or command>
supersedes: none           # or: F0000
---

## Claim

One sentence.

## Result

The numbers. Compare them to the d12 seed-variation reference (F0001). State
the effect size and the spread. Say if the effect is larger than the spread.

## Limitations

What could make this wrong. Check the caveats in `DATASET.md`. State which
ones apply.
