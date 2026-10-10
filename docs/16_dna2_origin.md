# DNA-2 origin statistics: self-replicators arising from random material (2026-10-10)

docs/15 found one unseeded DNA-2 world in which a self-sustaining replicator arose by itself. This note measures how
often that happens, how long it takes, what the founders look like, and how the code's degeneracy changes it: 96
unseeded worlds, 48 seeds under the standard genetic code and 48 under the uniform code. Every number is a count from
`results/dna2_origin/` (`report.json` holds every run); nothing is estimated except where a sentence says so.

## 1. Setup

Every world: 99,999,744 cells (97,656 slots of 1,024 cells), DNA-2 (`--dna2`), random soup, **no seeds**, mutation
0.001 per copied cell, double strand, 95 damage hits and 23 refilled slots per tick (the defaults), lifetime 64 ticks,
16 instructions per tick, 200,000 ticks, census every 100 ticks with the ten most common genes and their codons.
Seeds 1 to 48 under each code. Run on knecht24's three RTX 3060 and adler40's RTX 4090 by a work queue
(`tools/dna2_origin_queue.py`), 19:55 to 23:02 local; analysis `tools/dna2_origin_report.py`, pictures
`tools/dna2_origin_plot.py`.

**Definitions.** *Takeover*: the first census at which more than half of the slots (48,828) are alive; this is the
counted event, because a lineage that fills half the world has clearly established itself. *First faithful birth*:
the first census with a faithful birth (a child whose gene equals its parent's); it can come from a one-shot copier
that founds nothing. *Founder*: the most common gene at the takeover census. *Leader*: the most common gene at the
last census. *Persists*: more than 10,000 alive at the end.

**Rate.** Waiting times are treated as exponential with censoring: rate = takeovers / total exposure (ticks until
takeover, or 200,000 if none), mean waiting time = 1 / rate, 95 % interval from the exact Poisson limits.

**Check.** Standard seed 1 reproduces the unseeded pilot of docs/15 exactly: first faithful birth in the census of tick
43,399, the same 11-codon founder.

## 2. Result

| | standard code | uniform code |
|---|---:|---:|
| worlds | 48 | 48 |
| **taken over by a self-made lineage** | **48** | **6** |
| of these still alive at tick 200,000 | 48 | 5 |
| worlds with a faithful birth but no takeover | 0 | 9 |
| total exposure | 1,910,552 ticks | 8,782,994 ticks |
| **mean waiting time to takeover** | **39,803 ticks** (95 %: 30,020 to 53,988) | **1,463,832 ticks** (95 %: 672,524 to 4,008,695) |
| the same in refilled slots (23 per tick) | about 915,000 | about 33.7 million |
| takeover ticks | 799 to 149,199 | 6,799, 16,499, 33,399, 57,199, 85,999, 183,099 |

**Under the standard code every world produced life**, after about 40,000 ticks on average; under the uniform code 6 of
48 did, about **37 times** more slowly. That matches the exhaustive sweeps of docs/15, where the standard code has 60
to 130 times more replicators among short genes: the synonyms let random strings spell a working copy loop in many
more ways.

The spread of takeover times (799 to 149,199 ticks in the standard code) is what a waiting process with a constant
rate produces; there is no sign of a hidden preparation phase. In 35 of 48 standard worlds the takeover census came
within 500 ticks of the first faithful birth: once a real replicator appears, it fills the world in a few hundred
ticks. In the other 13, one-shot copiers made a faithful copy 1,900 to 96,100 ticks before the lineage that took over.

A rough consistency check (an estimate, not a measurement): a refilled slot is a gene only if its first codon is ATG,
1 in 64, so about 14,000 random genes enter per standard-code origin, and one in 14,000 random genes founding a
lineage is of the order of the replicator density the sweep measured for 6- and 7-codon genes (1.1 to 1.6 x 10^-4)
reduced by the chance that a random gene's slot is otherwise viable.

## 3. The founders: every world invents its own

| | standard code | uniform code |
|---|---|---|
| distinct founder operation sequences | **46 of 48** | 6 of 6 |
| founder length (codons incl. START) | 5 to 17, median 8.5 | 7 to 13, median 12 |
| founders that are the minimal five operations in some order | 4 (`START COPY ALLOC TERM DIVIDE` twice) | 0 |

Examples of standard-code founders: `START SKNZ TERM DIVIDE R3 SKNZ L1 COPY R1 ALLOC L3` (seed 1),
`START TERM DIVIDE R1 COPY ALLOC R1`, `START ALLOC COPY SKZ SKZ TERM DIVIDE`,
`START SEEK L1 COPY R1 COPY COPY TERM INC TERM DIVIDE ALLOC`. They use skips, head moves, the accumulator, SEEK. The
uniform-code founders are padded with NOP codons (42 of the 64 codons are NOP in that code), e.g.
`START NOP TERM DIVIDE COPY NOP START NOP ALLOC NOP NOP NOP`.

A working copy loop in DNA-2 needs ALLOC, COPY, TERM and DIVIDE in an order that executes; everything else in a
founder is slack. Random material finds dozens of different arrangements.

## 4. After the origin: streamlining, and a few that learn to search

| | standard code | uniform code |
|---|---|---|
| leader at the end shorter than the founder | 35 of 48 | 4 of 6 |
| leader length at the end | median 6 | median 6 |
| leader is exactly START, ALLOC, COPY, TERM, DIVIDE in some order | **17 of 48** | 1 of 6 |
| most common leader at the end | `START ALLOC COPY TERM DIVIDE` (7 worlds) | |
| leaders containing SEEK | 5 (seeds 8, 21, 25, 26, 38) | 1 |

Evolution shortens the founders: a shorter gene copies in fewer ticks and leaves more children in its 64-tick life.
In 17 of 48 standard worlds it reached exactly the five essential operations, the minimum the exhaustive sweep
found, and in 7 worlds the very sequence of the hand-written seed of docs/15, `START ALLOC COPY TERM DIVIDE`. Different
origins, the same end point.

In 5 standard worlds the winner instead kept or acquired SEEK (e.g. `START SEEK SETQ COPY COPY COPY COPY ALLOC TERM
DIVIDE R1`). SEEK scans up to 3,072 cells for a template each time it runs, so these worlds ran 15 to 20 times slower
(standard seed 22: 21.8 ticks per second on a 3060, 9,166 s for the run, against about 380 ticks per second in a
world of streamlined copiers). Why SEEK pays in these lineages is not yet understood; a trace of one of them is the
next step.

The pictures show it happening (`results/dna2_origin/*.png`, top: population and faithful births over the run;
bottom: the ten most common operation sequences around the takeover):

| file | what it shows |
|---|---|
| `standard_s001.png` | the docs/15 world: founder (11 codons) sweeps at tick 43,899, then four mutant sequences replace each other within 3,000 ticks |
| `standard_s017.png` | the earliest origin: first faithful birth by tick 699, takeover at 799 |
| `standard_s005.png` | the latest: a one-shot faithful copy at tick 53,099, the real takeover only at 149,199 |
| `standard_s026.png` | a SEEK lineage |
| `uniform_s025.png` | the earliest uniform-code origin (6,799) |
| `uniform_s038.png` | the one collapse (below) |

## 5. The one collapse

Uniform seed 38: a lineage took over at tick 16,499 and held about 89,000 slots, with mutants replacing one another
as in the standard worlds (`START TERM DIVIDE NOP NOP ALLOC COPY L1 NOP NOP SKZ TERM` and variants in which the
second codon mutated to DIVIDE, NOP, SKZ, ALLOC, STA, SKNZ). From about tick 33,000 its cumulative faithful births stop
growing: the lineage no longer copies itself faithfully. The population declines over 12,000 ticks and is extinct by
about tick 45,000; the world returns to the random state of before (about 20 organisms) for the remaining 155,000
ticks. Of 54 takeovers in both codes this is the only one that did not persist. Which mutation broke faithful copying
is visible in the top-gene file but not yet traced.

## 6. Caveats

- One configuration: one world size, one mutation rate, one inflow rate, one slot size, 200,000 ticks. The waiting
  time depends on all of them; the inflow rate sets how fast random genes are tried.
- The uniform-code rate rests on 6 events; its interval spans a factor of 6.
- The founder is the most common gene at the takeover census (100-tick resolution), which in fast takeovers may
  already be a mutant of the gene that made the first faithful copy.
- **Run incident.** The queue's one-hour SSH timeout fired on two slow SEEK worlds (standard seeds 22 and 26) and
  requeued them while the original processes kept running. Seed 26's rerun finished first and is the one recorded
  (the worlds are deterministic, so both runs are identical). For seed 22 the duplicate truncated the original's census
  and top-gene files while both were running, which removed the censuses from tick 59,299 to 113,199 and left NUL bytes
  that the report skips. The takeover (before tick 58,600) and the end state (200,000 ticks, 88,745 alive; summary in
  `standard_s022.err`) are intact, so seed 22 counts. The duplicate was stopped at 23:20. The queue's timeout should
  scale with the run length before it is used again.

## 7. Reading

1. **In DNA-2 under the standard code, life is not a rare accident but the expected outcome**: 48 of 48 worlds, after
   about a million random slots (14,000 random genes) on average.
2. **The genetic code's redundancy speeds up the origin about 37-fold** (uniform code: 6 of 48), in line with the
   60- to 130-fold difference in replicator density.
3. **The origin is contingent, the outcome is convergent.** 46 different founder designs in 48 worlds; after
   200,000 ticks a third of the worlds have converged on the same five-operation minimum, the one an exhaustive search
   predicts and a human designed by hand. A minority evolves template search instead.
4. **Established lineages almost always last** (53 of 54); the one exception collapsed when faithful copying broke,
   an error catastrophe in a code without synonyms to buffer mutations.
5. What these worlds still lack is anything to do besides copying: no space, no energy, no interaction, no tasks.
   That is DNA-3 (docs/17).
