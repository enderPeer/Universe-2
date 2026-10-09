# DNA-1 built, verified and run at 3 x 10^9 cells (2026-10-09)

The machine of [docs/09_dna1_spec.md](09_dna1_spec.md) exists in two implementations that agree bit for bit, and
its world has run on a single GPU at the size of a genome. This note records how it was built, how it was checked,
how fast it runs, and what the first worlds did. Nothing here is estimated; every number is from a log in
`results/dna1/` or from the engines' own output.

## 1. What was built

| file | what | size |
|---|---|---|
| [`sim/dna_ref.py`](../sim/dna_ref.py) | the reference: pure Python 3.11, standard library, sequential, every rule of the spec | 650 lines |
| [`gpu/dna_world.cu`](../gpu/dna_world.cu) | the engine: single-file CUDA C++17, one thread per slot, 64-bit indexing, packed 2-bit strands, tick-start snapshot, deterministic ALLOC by atomicMin, host-side state hash | 751 lines |
| `docs/09_dna1_spec.md` section 13 | clarifications fixed during the build (tick labels, hash and census schedule, births file format, default N, D/F arguments, inflow atomics, ALLOC sentinel) | |

Both take the same command line (`--N --G --ticks --seed --I --T --mu --D --F --no-double --uniform-code --blank
--genome --n-seed --hash-every --hash-init --no-hash --census-every --births --dump --out`) and print the same
lines: `tick t hash <16 hex>` for the state after tick t, `census {json}` every `--census-every` ticks, a final
census JSON to `--out`, and a births file `#tick parent child parent_hash child_hash faithful`.

Build method: two agents wrote the reference and the engine independently from the spec; three reviewers checked
reference against spec, engine against spec, and the engine for races, overflow and memory at 3 x 10^9; one agent
fixed the confirmed findings and ran the comparison; one agent ran the scaling tests and the world. 7 agents,
63 minutes, 257 tool calls.

## 2. Review findings (21, of which 5 blockers), all fixed before the comparison

- Reference: tick labels off by one against the agreed convention and the initial state hashed by default; census
  labelled with completed ticks; no `--hash-init`; births file header, parent hash of spontaneous births and row order
  not as specified.
- Engine: a data race in the inflow refill when the slot size is not a multiple of 4 (two blocks refilling adjacent
  slots share a byte; now word atomics); `--D 0` and `--F 0` launched kernels with an empty grid and crashed;
  `--genome` parsing stricter than specified; the ALLOC lowest-requester scratch compared as signed; the engine's own
  default N = 3,000,000,000 is not a multiple of G = 1,024 and was rejected by its own check.
- Both: default N is now 2,929,687 x 1,024 = **2,999,999,488 cells** (the S of the spec); `--ticks` must be at least 1;
  D and F accept `auto` or a non-negative integer.

## 3. Verification: identical state hashes on 17 configurations

The six test worlds of spec section 10 and eleven extra edge cases (G = 6, 10 and 255 with large inflow, D = F = 0,
two duplicate-inflow seeds, a dividing genome with mu = 0.01 and 143 births, lower-case and tab-separated genomes,
`--hash-every 0` and `7`), every tick hashed (`--hash-every 1`): **every hash identical** between the reference
(sequential Python) and the engine (parallel CUDA, RTX 4090), census lines identical, birth logs identical row for
row. One fix round was needed (the findings above); the comparison itself found no further disagreement.
Reproduction: the commands are in the cross-check record in `results/dna1/` and in spec section 13; the outputs of
the agreeing runs are kept on adler40 in `~/universe2/dna/agree/`.

Caveat the comparison agent recorded: test t4 (the hand-written organism with mu = 0.01) never copies a cell, so
it equals t3 and does not exercise mutation; mutation agreement rests on the stress configuration (143 births) and
one duplicate-inflow case. The hand-written organism of spec section 11 indeed never divides: its SETQ copies the
head position instead of the child start and its loop tests a cell that is always 0, exactly as the spec predicted
(trace in the reference's self-test).

## 4. Speed and memory on the RTX 4090 (adler40, GPU 0, driver 595.91, CUDA 12.4)

| cells N | slots | ticks | main-loop seconds | ticks per second | device memory |
|---:|---:|---:|---:|---:|---:|
| 99,999,744 | 97,656 | 100 | 0.011 | 8,709 (launch-latency bound) | 88 MB |
| 999,999,488 | 976,562 | 30 | 0.025 | 1,210 | 866 MB |
| **2,999,999,488** | **2,929,687** | 200 | 0.494 | **405** | **2,586 MB** |

At 3 x 10^9 cells a tick is 2.5 ms: two strands plus the snapshot are 2.25 GB of memory traffic, 2.9 million slot
threads, 2,861 damage hits, 715 refilled slots. The reference does the same world at 1,048,576 cells in 0.4 s per
50 ticks; the engine does it in 5 ms. A 25-minute GPU run is about 600,000 ticks of a 3-billion-cell world.

## 5. The worlds

### Comparison at 10^8 cells, 200 ticks, seed 1 (`results/dna1/cmp_*`)

| run | births | of which spontaneous | faithful | deaths | live at the end | distinct genes | max lineage depth |
|---|---:|---:|---:|---:|---:|---:|---:|
| default (mu 0.001, double strand) | 69 | 69 | 0 | 46 | 23 | 23 | 0 |
| `--no-double` | 69 | 69 | 0 | 46 | 23 | 23 | 0 |
| `--mu 0` | 69 | 69 | 0 | 46 | 23 | 23 | 0 |

All three are byte-identical: in 200 ticks at 10^8 cells nothing copied anything, so neither mutation nor the second
strand had anything to act on. Every birth came from inflow (random slots that happen to start with ATG and contain
a stop codon: probability about 1/64 x 1 per refilled slot).

### The 3 x 10^9-cell world (`results/dna1/big_3e9_*`)

Launched 2026-10-09 16:47 local on GPU 0: `dna_world --N 2999999488 --G 1024 --ticks 600000 --seed 1 --no-hash
--census-every 10` (I 16, T 64, mu 0.001, D 2,861, F 715, double strand, standard code, random soup). Measured rate
in the run: 396 ticks per second at 94 % GPU utilisation and 2,952 MiB.

Census at tick 88,249 (223 s into the run, the point at which the run agent had to report): live 503; births
968,487 of which 938,719 spontaneous (inflow) and **29,768 by DIVIDE**; faithful births 0; deaths 967,984; distinct
genes among the living 496; lineage depth histogram {0: 487, 1: 16}. The final census is in section 7.

## 6. Reading

1. **The machine replicates partially but not faithfully, yet.** About 3 % of births come from DIVIDE: an organism
   claimed a slot, copied some cells and divided, and the child happened to be expressible. None of these children
   was a faithful copy, and none of them ever divided in turn (every lineage stops at depth 1). The reason is visible
   in the hand-written organism's trace: COPY advances P and Q by one cell per execution, the loop condition (JNZB)
   reads the cell under the head H, which COPY does not move, so a random gene either copies the same cell over and
   over (REWIND in the loop) or copies a fixed small number of cells. A faithful copy of an n-codon gene needs 3n
   COPY executions with P and Q aligned, which needs a loop whose exit depends on something the copy changes. Such
   genes exist in the design (H can be placed in the child slot ahead of Q, where COPY will eventually write a 0),
   but random code at 715 new slots per tick has not produced one in 90,000 ticks.
2. **This is the predicted first result, not a failure of the build.** Universe-1's crawler needed an exhaustive
   sweep of 2^32 programs to find 141 copiers and 2^40 to find 3 intact ones; NANO's 316 replicators came from
   2^40. The DNA-1 world is sampling random genes at about 7 x 10^5 per second, which is 2^40 in 18 days. The
   copier sweep of spec section 12 (all genes of 2 to 7 codons, each alone in a small world, scored by fecundity) is
   the direct route and needs one more engine mode: many small worlds per GPU launch.
3. **What the double strand and mutation will measure once copying exists**: in these runs they were never
   consulted, which the byte-identical comparison runs prove rather than assume.
4. **Energy is still the slot lifetime.** T = 64 ticks and I = 16 instructions is Universe-1's soup accounting; the
   EAT/WASTE budget of docs/07 section 5 is not in version 1.

## 7. Final census of the 600,000-tick run

Finished 2026-10-09 17:12 local: 600,000 ticks in 1,512.5 s (396.7 ticks per second), 2.68 GB of device memory.
Files: `results/dna1/big_3e9_final.json`, `big_3e9.err`, `big_3e9_census.txt.gz` (60,000 census lines, one per 10
ticks), `big_3e9.births.sampled` (the engine kept 1 birth row in 1,000: 6,705 rows), `big_3e9_births_stats.txt`.

| tick | live | births so far | by DIVIDE | faithful | deaths so far | distinct genes | lineage depths |
|---:|---:|---:|---:|---:|---:|---:|---|
| 9 | 86 | 112 | 0 | 0 | 26 | 86 | {0: 86} |
| 99 | 494 | 1,084 | 13 | 0 | 590 | 489 | {0: 483, 1: 11} |
| 999 | 552 | 11,252 | 340 | 0 | 10,700 | 545 | {0: 531, 1: 21} |
| 9,999 | 528 | 110,097 | 3,511 | 0 | 109,569 | 517 | {0: 492, 1: 36} |
| 599,999 | 505 | 6,585,887 | 204,881 | **3** | 6,585,382 | 494 | {0: 499, 1: 6} |

Read: the population is a steady state of about 500 living processors, almost all of them spontaneous (the inflow
refills 715 slots per tick, about 11 of which are expressible), each living its 64 ticks. 3.1 % of all births are
DIVIDE births, children of organisms that claimed a slot and copied something, and in the whole run **three** of
those 204,881 children were faithful copies of their parent (parent and child gene hashes equal), about 1 in
70,000 DIVIDE births or 1 in 2.2 million births. None of the three divided in turn: the lineage-depth histogram
never shows depth 2. The 1-in-1,000 birth sample did not catch any of the three rows, so their genes are not
recorded; only the counter is. A future run should log every faithful birth unsampled.

The recurring "top genes" of the census (hashes beginning `0846..07b4fc..` or `0847..`, counts 2 to 3) are not
clones: they are the two-codon genes `ATG xxx STOP`, whose FNV-1a hashes differ only in the low bits because they
differ only in the last codon, and the inflow creates them again and again.

So the first DNA-1 world is where Universe-1's crawler soup and NANO's alife census were before their sweeps:
copying exists, faithful copying is a one-in-a-million event among random genes, and no lineage persists. The
copier sweep (spec section 12, proposed as its own session) is the way to find the genes that do it on purpose.
