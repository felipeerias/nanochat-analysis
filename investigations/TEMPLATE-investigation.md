---
id: I0000
kind: question              # question | hypothesis
status: open                # open | running | closed
data: <collection id; exact segments>
selection: <planned filters: defined rows, arm, direction, verdict>
universe: <how many channels you plan to test>
allowed inputs: <what an analyst may read, with commits>
---

## Question or claim

One sentence.

## Test

How the data answers it. For a hypothesis, state the decision rule before you
look: what result would support the claim, and what result would refute it.

**If the rule touches more than one channel, name the ONE channel that
decides, and say what the others are for.** Three investigations have been
weakened by rules that listed several channels without saying how to combine
them; analysts then reported different verdicts from the same numbers, all
defensible. Say "the verdict is decided by X; Y and Z are reported as
context".

**Write the selection so it still works when nothing passes.** A filter that
empties the data changes the question rather than answering it. If a
restriction could remove every record, say what to do in that case.

## Output

What an answer looks like. A number, a ranking, a plot.
