# Does the best 2-bit machine have a multiply-accumulate? Every program of 15 and 16 symbols on the cluster

Date: 2026-10-09. The question came from the BitNet experiment in `Ghost/bitnet_u2` (outside this repo): BitNet
b1.58-2B-4T is 2,084,044,800 ternary weights, and a token costs that many multiply-accumulates acc + w * x with
w in {-1, 0, +1}. The best W = 2 machine of docs/07, table 50098 `INC_R1, INC_L1, FLIP_R2, JNZB` (ring of 256
two-bit cells, head at 0, PC wraps, 64 steps), has COPY at 5 symbols, a ternary multiplier at 10 symbols, an adder
mod 4 at 11 and a carry at 11, but the numpy search there found **no multiply-accumulate in any program of 1 to 14
symbols** (357,913,940 programs per encoding, 1,263 s on 24 local cores). 15 and 16 symbols are 4^15 + 4^16 =
5,368,709,120 more programs per encoding: cluster work. This note covers them.

**Result: no program of exactly 15 or exactly 16 symbols computes acc + w * x mod 4 either, in either weight
encoding.** With the earlier runs, every one of the 5,726,623,060 programs of 1 to 16 symbols has been tried at
64 steps, and none is a multiply-accumulate under the definition below. No program is consistent with more than
17 of the 48 input triples. Section 9, added later the same day, repeats 13 to 16 symbols with a 256-step budget
on the cluster: still none, so with the numpy 256-step run up to 12 symbols every program of 1 to 16 symbols is
covered at both budgets (the best 256-step program reaches 22 triples).

## 1. The primitive

Inputs seeded into cells 0, 1, 2 = (acc, w, x), everything else blank, head at 0. After the run the result
(acc + w * x) mod 4 must stand in one fixed cell (the same cell for all inputs), or in the cell under the final
head position (the same rule for all inputs). Two weight encodings, each 3 x 4 x 4 = 48 triples:

| name | cell 1 holds | triples | target |
|---|---|---|---|
| MAC_T | w in {1, 3, 0} (3 = -1 mod 4) | w outer, acc, x inner | (acc + w * x) mod 4 |
| MAC_B | the BitNet checkpoint's code c in {2, 0, 1}, meaning c - 1 | c outer, acc, x inner | (acc + (c - 1) * x) mod 4 |

The triples are ordered nonzero target first (a stable sort of the enumeration order), exactly as
`Ghost/bitnet_u2/prim_search.py` orders them, and a program stops at its first failing triple. The number of
programs still alive after each triple is recorded; a program alive after all 48 is a multiply-accumulate.

## 2. The tool: `prim` mode of the C sweeper

`tools/u2_tables.c` got a mode `prim PRIMS W L steps OPS start count threads`: one explicit machine, every
program of exactly L symbols whose base-4 index lies in [start, start + count), a comma-separated list of
primitives (COPY, ADD, MUL_T, MAC_T, MAC_B, CARRY with the definitions of `prim_search.py`). Candidate output cells
are a 256-bit mask that only shrinks, plus a flag for the head cell; a program whose mask is empty and whose head
flag is clear stops. MAXL went from 12 to 16 for this mode (the table modes keep their L <= 12 bound). Built under
a new name, `u2_tables_mac`, on each node (the running `u2_tables*` binaries cannot be overwritten). Driver:
`tools/u2_mac.py` (build, validate, launch, status, collect; found programs are re-run with `tools/u2_ops.py`).

Output per primitive: the survivor count after every input and each found program with its output cells and
head flag. Cost: a run is up to 64 instructions, about 1 ns each; a program averages 2.4 runs per encoding because
the second triple already removes 78 % and the fourth 97 %.

## 3. Validation (adler40, 24 threads, 10.4 s in all)

All six primitives over every program of 1 to 12 symbols (22,369,620 programs per primitive), one invocation per
length, summed, against `prim_INC_R1-INC_L1-FLIP_R2-JNZB_W2_L12_s64.json` of the numpy search (which was itself
validated against `u2_ops.run` on 600 random states). File: `results/u2_mac/validation_L12_s64.json`.

| primitive | survivors after each input | found (C) | found (numpy) | re-verified with `u2_ops.run` |
|---|---|---|---|---|
| MAC_T (48 inputs) | identical, all 48 | 0 | 0 | |
| MAC_B (48) | identical | 0 | 0 | |
| ADD (16) | identical | 5 | 5, the same programs | 5 / 5 |
| MUL_T (12) | identical | 11 | 11, the same | 11 / 11 |
| COPY (4) | identical | 31,506 | 31,506, the same | 31,506 / 31,506 |
| CARRY (16) | identical | 26 | 26, the same | 26 / 26 |

Every found program's cells and head flag were checked on every input of its primitive with the Python reference
interpreter. The build on each node also runs a self-check (COPY at 5 symbols must find exactly `01302` with output
cell 1). The 12-symbol length alone took 6.9 s for all six primitives on 24 threads, against 370.5 s for the numpy
search on 24 processes. (The validation was rerun from the committed source after a one-line fix to the table
modes' length check; both runs were identical.)

## 4. Cluster at launch (2026-10-09, 20:37 workstation time; the nodes' clocks read 18:37)

| node | threads used, nice 10 | load before | other CPU work at launch |
|---|---|---|---|
| adler40 (i9-12900K) | 24 | 0.42 | one single-thread `dna_world_v3` run of the DNA-1 seeded-world session (docs/12) |
| falke64 (Ryzen 5 7600X) | 12 | 0.20 | none |
| knecht24 (Threadripper 1920X) | 24 | 0.68 | one single-thread `dna_world_v3` run, the Windows VM |
| specht32 (i7-8700K) | 12 | 0.00 | none |

CPU only; no GPU was touched. Shards proportional to the speeds measured in docs/06 (24 : 15 : 8 : 8), one detached
chain per node (`results/u2_mac/plan_mac_L15_L16.json`), plus an overlap range of 2^26 = 67,108,864 programs per
length run on both adler40 (Intel) and knecht24 (AMD):

| node | 15 symbols: start + count | 16 symbols: start + count | overlap |
|---|---|---|---|
| adler40 | 0 + 468,541,887 | 0 + 1,874,167,547 | 123,456,789 + 2^26 (L = 15), 3,000,000,000 + 2^26 (L = 16) |
| falke64 | 468,541,887 + 292,838,679 | 1,874,167,547 + 1,171,354,717 | |
| knecht24 | 761,380,566 + 156,180,629 | 3,045,522,264 + 624,722,516 | the same two ranges |
| specht32 | 917,561,195 + 156,180,629 | 3,670,244,780 + 624,722,516 | |

## 5. Results (collected 2026-10-09 20:43; `results/u2_mac/mac_L15.json`, `mac_L16.json`, raw shard outputs in `raw/`)

| | 15 symbols, MAC_T | 15 symbols, MAC_B | 16 symbols, MAC_T | 16 symbols, MAC_B |
|---|---:|---:|---:|---:|
| programs | 1,073,741,824 | 1,073,741,824 | 4,294,967,296 | 4,294,967,296 |
| alive after input 1 | 1,069,302,736 | 1,069,439,842 | 4,279,101,086 | 4,279,315,448 |
| after 2 | 238,563,214 | 237,587,354 | 952,884,104 | 947,742,770 |
| after 3 | 147,311,538 | 149,473,689 | 580,964,429 | 589,402,460 |
| after 4 | 27,140,084 | 27,674,272 | 108,799,225 | 110,819,568 |
| after 6 | 6,402,471 | 6,562,161 | 25,525,650 | 26,162,047 |
| after 9 | 16,520 | 19,249 | 60,170 | 72,356 |
| after 12 | 1,143 | 1,091 | 2,985 | 4,554 |
| after 13 | 161 | 97 | 516 | 508 |
| after 14 | 38 | 25 | 85 | 193 |
| after 15 | 24 | 5 | 30 | 35 |
| after 16 | 23 | 2 | 27 | 2 |
| after 17 | 18 | 2 | 10 | 2 |
| after 18 | **0** | **0** | **0** | **0** |
| **multiply-accumulates (alive after 48)** | **0** | **0** | **0** | **0** |

The first twelve triples are the positive weight (w = 1, code 2) with nonzero target; input 13 is the first
negative weight. Inputs 13 to 17 are (acc, x) = (0,1), (0,2), (0,3), (1,0), (1,2) with w = -1, and input 18 is
(acc, x) = (1, 3), target 2. In both encodings and at both lengths every program that gets through the positive
weights and the first five negative-weight triples fails on the sixth. In the numpy search up to 14 symbols the
best programs survived 16 inputs (MAC_T) and 14 (MAC_B); at 15 and 16 symbols both encodings reach 17, so two
more symbols move the frontier by one input for MAC_T and by three for MAC_B, and then it stops at the same
triple for both. Programs that survive 17 inputs: 18 and 10 (MAC_T at 15 and 16 symbols), 2 and 2 (MAC_B).

Total over 1 to 16 symbols, 64 steps, both encodings: 5,726,623,060 programs each, **0 multiply-accumulates**.
Runs executed on the cluster (first triple for every program, further triples for survivors): 2.57 billion per
encoding at 15 symbols, 10.27 billion at 16, 5 % of the 48 x 4^L a search without the early stop would need.

### Run record

| node | 15 symbols | 16 symbols | overlap 15 / 16 | node total |
|---|---|---|---|---|
| adler40, 24 threads | 468,541,887 programs, 54.0 s | 1,874,167,547, 216.1 s | 8.7 s / 7.6 s | 286.4 s |
| falke64, 12 threads | 292,838,679, 37.6 s | 1,171,354,717, 149.7 s | | 187.3 s |
| knecht24, 24 threads | 156,180,629, 27.5 s | 624,722,516, 110.0 s | 13.6 s / 11.3 s | 162.4 s |
| specht32, 12 threads | 156,180,629, 50.3 s | 624,722,516, 200.7 s | | 251.0 s |

Launched 20:37:38, all twelve runs done by 20:42:56 (workstation clock); 4.5 CPU-core-hours in all. Checks at
collection: every shard's stdout header and stderr summary carry the planned start, count, length, machine and
host name; the shards of each length sum to 4^L exactly; the 2^26-program overlap ranges are **identical between
adler40 and knecht24** at both lengths (all 96 survivor counts and the empty found lists). No found program to
re-verify; the verification path (`u2_ops.run` on every input) is the one exercised in section 3.

## 6. Reading

The machine multiplies (10 symbols) and adds (11 symbols) but no program of up to 16 symbols does both in one
pass, although 16 symbols is five more than the adder needs and six more than the multiplier. The adders found
earlier write their sum into cell 0 or 1, the multipliers their product into cell 1 or under the head (Ghost
report, section 4a); a multiply-accumulate has to bring the product and acc together in one cell under the fixed
layout, and nothing of this size does. Whether a longer program does is open. The survivor frontier does not move
with length (16 or 14 inputs up to 14 symbols, 17 at 15 and again 17 at 16), which suggests the layout rather
than the length is the obstacle, but that is a reading, not a result. (At 256 steps it does move with length,
17 to 22 inputs from 13 to 16 symbols, section 9; there the step budget looks like as much of an obstacle as the
layout.)

So the BitNet question is settled for this machine at this program size: the model's inner step is not a
primitive here, and a token would need a composite of the found adder and multiplier (two programs, a copy in
between, and the layout they expect), which is the "big program from blocks" question of docs/04, not a one-program
answer.

## 7. Caveats

- The definition is fixed: inputs in cells 0, 1, 2 in the order (acc, w, x), head at 0, output in one fixed cell or
  under the final head, mod 4 with no carry. A multiply-accumulate with another layout (inputs elsewhere, w and x
  swapped, output split over two cells, output defined relative to the head's start) is not excluded by this run.
- Two budgets only. Sections 4 and 5 are 64 steps; the numpy search also tried 256 steps up to 12 symbols (no
  multiply-accumulate; nine times as many adders), and section 9 extends 256 steps to 13 to 16 symbols (none
  either, 15.1 core-hours). Both are snapshots: a program with the right answer on the tape at step 100 that
  overwrites it by step 256 is invisible here. docs/14 removes that caveat by checking every budget 1 to 256 at
  once (still none). Budgets beyond 256 steps were not tried; the frontier moves with length (section 9, docs/14),
  so a longer program is not excluded by these runs.
- Exactly this machine. The other two W = 2 candidates (`INC, INC_R1, INC_L1, JNZB` and `L1, INC_L1, FLIP_R2, JNZB`)
  were searched up to 10 symbols only (Ghost report, section 4c).
- Only the two weight encodings above; the primitive list of the C mode and the numpy search is the same, so a
  third encoding is a one-line addition.

## 8. Files and commands

- `tools/u2_tables.c` (`prim` mode, built as `u2_tables_mac`), `tools/u2_mac.py` (driver; `--steps N` selects the
  budget, default 64, and `--L` accepts 13 to 16).
- 64 steps: `results/u2_mac/validation_L12_s64.json`, `plan_mac_L15_L16.json`, `mac_L15.json`, `mac_L16.json`.
- 256 steps (section 9): `validation_L12_s256.json`, `plan_mac_s256_L13_L14_L15_L16.json`, `mac_s256_L13.json` to
  `mac_s256_L16.json`.
- `raw/`: every shard's stdout and stderr, named by node, budget and range.
- Reference search and report: `C:\Users\end\Ghost\bitnet_u2\` (`prim_search.py`, `u2_vec.py`, `REPORT.md`,
  `results/prim_*_W2_L12_s64.json`, `..._L14_s64.json`, `..._L12_s256.json`), outside this repository.

```
python tools/u2_mac.py build
python tools/u2_mac.py validate
python tools/u2_mac.py launch --L 15 --L 16
python tools/u2_mac.py status --L 15 --L 16
python tools/u2_mac.py collect --L 15 --L 16
python tools/u2_mac.py validate --steps 256
python tools/u2_mac.py launch --steps 256 --L 13 --L 14 --L 15 --L 16
python tools/u2_mac.py status --steps 256 --L 13 --L 14 --L 15 --L 16
python tools/u2_mac.py collect --steps 256 --L 13 --L 14 --L 15 --L 16
```

## 9. 256 steps at 13 to 16 symbols (2026-10-09, 21:10 to 21:25 workstation time)

Section 7 left the longer budget open. This section closes it: the same binary (`u2_tables_mac`, source identical
on all four nodes to the committed `tools/u2_tables.c`), the same shards, the same checks, with 256 instead of 64
instructions per run (`tools/u2_mac.py --steps 256`), over every program of exactly 13, 14, 15 and 16 symbols in
both weight encodings. The numpy search had already done 256 steps for 1 to 12 symbols (Ghost report, section 4b).

**Result: no program of 13 to 16 symbols computes acc + w * x mod 4 at 256 steps either, in either encoding.**
Together with the numpy run, all 5,726,623,060 programs of 1 to 16 symbols are now covered at 256 steps as well as
at 64, with 0 multiply-accumulates. The longer budget does move the frontier: at 16 symbols three MAC_T programs
get through 22 of the 48 triples (17 at 64 steps), but no program at any length gets through the negative weight.

### Validation at 256 steps (adler40, 24 threads, 17.9 s of compute)

All four primitives of the numpy 256-step file (`prim_INC_R1-INC_L1-FLIP_R2-JNZB_W2_L12_s256.json`: ADD, MUL_T,
MAC_T, MAC_B; that run had no COPY and no CARRY), every program of 1 to 12 symbols, one invocation per length.
File: `results/u2_mac/validation_L12_s256.json`.

| primitive | survivors after each input | found (C) | found (numpy) | re-verified with `u2_ops.run` at 256 steps |
|---|---|---|---|---|
| MAC_T (48 inputs) | identical, all 48 | 0 | 0 | |
| MAC_B (48) | identical | 0 | 0 | |
| ADD (16) | identical | 46 | 46, the same programs | 46 / 46 |
| MUL_T (12) | identical | 16 | 16, the same | 16 / 16 |

The 12-symbol length took 11.6 s (6.9 s at 64 steps); the numpy run took 614.3 s on 24 processes.

### Cluster at launch (21:10:30 workstation time; the nodes' clocks read 19:10)

| node | threads, nice 10 | load before | other CPU work at launch |
|---|---|---|---|
| adler40 | 24 | 1.00 | the single-thread `dna_world_v3` run of docs/12, still going (its GPU work was not touched) |
| falke64 | 12 | 0.21 | none |
| knecht24 | 24 | 0.08 | the Windows VM |
| specht32 | 12 | 0.08 | none |

Shards by the same speed rule as section 4 (`results/u2_mac/plan_mac_s256_L13_L14_L15_L16.json`); 15 and 16
symbols exactly as in the table there, 13 and 14 as follows, plus Intel/AMD overlap ranges on adler40 and knecht24
at every length:

| node | 13 symbols: start + count | 14 symbols: start + count | overlap |
|---|---|---|---|
| adler40 | 0 + 29,283,868 | 0 + 117,135,472 | 10,000,000 + 2^24 (L = 13), 100,000,000 + 2^26 (L = 14), the two ranges of section 4 (L = 15, 16) |
| falke64 | 29,283,868 + 18,302,417 | 117,135,472 + 73,209,670 | |
| knecht24 | 47,586,285 + 9,761,289 | 190,345,142 + 39,045,157 | the same four ranges |
| specht32 | 57,347,574 + 9,761,290 | 229,390,299 + 39,045,157 | |

### Results (collected 21:26; `results/u2_mac/mac_s256_L13.json` to `mac_s256_L16.json`, raw shard outputs in `raw/`)

| 256 steps | 13, MAC_T | 13, MAC_B | 14, MAC_T | 14, MAC_B | 15, MAC_T | 15, MAC_B | 16, MAC_T | 16, MAC_B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| programs | 67,108,864 | 67,108,864 | 268,435,456 | 268,435,456 | 1,073,741,824 | 1,073,741,824 | 4,294,967,296 | 4,294,967,296 |
| alive after input 1 | 66,879,786 | 66,892,106 | 267,645,095 | 267,677,609 | 1,071,180,331 | 1,071,244,031 | 4,285,833,253 | 4,286,174,479 |
| after 2 | 18,447,360 | 18,419,241 | 73,814,354 | 73,643,821 | 299,880,736 | 298,816,032 | 1,192,877,481 | 1,188,618,984 |
| after 3 | 10,867,736 | 10,941,147 | 42,458,876 | 42,802,715 | 170,794,662 | 172,226,967 | 664,972,858 | 671,291,291 |
| after 4 | 2,032,318 | 2,065,277 | 8,138,710 | 8,276,940 | 33,059,349 | 33,564,312 | 134,726,055 | 136,935,906 |
| after 6 | 393,051 | 398,798 | 1,569,335 | 1,593,989 | 6,297,185 | 6,394,713 | 25,241,688 | 25,622,408 |
| after 9 | 1,901 | 1,514 | 5,954 | 5,322 | 21,959 | 19,965 | 80,850 | 72,311 |
| after 12 | 709 | 318 | 1,879 | 799 | 6,724 | 2,743 | 22,800 | 7,944 |
| after 13 | 188 | 113 | 375 | 217 | 1,427 | 906 | 2,872 | 2,317 |
| after 14 | 10 | 2 | 32 | 33 | 105 | 127 | 299 | 352 |
| after 15 | 3 | 2 | 3 | 13 | 37 | 61 | 99 | 145 |
| after 16 | 1 | 2 | 2 | 8 | 26 | 26 | 77 | 66 |
| after 17 | 1 | 2 | 2 | 8 | 24 | 20 | 58 | 65 |
| after 18 | **0** | 2 | 2 | 7 | 16 | 18 | 48 | 53 |
| after 19 | 0 | **0** | 2 | 3 | 16 | 3 | 48 | 3 |
| after 20 | 0 | 0 | **0** | **0** | **0** | **0** | 3 | **0** |
| after 21, 22 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 |
| after 23 | 0 | 0 | 0 | 0 | 0 | 0 | **0** | 0 |
| **multiply-accumulates (alive after 48)** | **0** | **0** | **0** | **0** | **0** | **0** | **0** | **0** |

The triple order is that of section 1: inputs 1 to 12 are the positive weight, 13 to 24 the negative one (acc
rising, x inner, zero targets deferred to the end), 25 to 36 the zero weight, 37 to 48 the twelve zero-target
triples. At 64 steps every survivor died on input 18, (acc, x) = (1, 3) with w = -1. At 256 steps the frontier
depends on length and encoding: MAC_B dies on input 19 at 13 symbols and on input 20, (acc, x) = (2, 1), at 14 to
16 symbols; MAC_T dies on input 18 at 13 symbols, on 20 at 14 and 15, and at 16 symbols three programs survive
inputs 20 to 22, (acc, x) = (2, 1), (2, 3), (3, 0), and die on input 23, (acc, x) = (3, 1), target 2. The numpy run
at 12 symbols and 256 steps had reached 16 (MAC_T, one program) and 18 (MAC_B, two). So four times the budget lets
the best programs handle two more values of acc under the negative weight, and length now matters (17, 19, 19, 22
inputs for MAC_T at 13, 14, 15, 16 symbols), but no program at any length completes the negative-weight block, and
none reaches the zero weight. Programs alive at the frontier: 1, 2, 16, 3 (MAC_T at 13 to 16 symbols) and 2, 3, 3,
3 (MAC_B). Nothing was found, so nothing had to be re-verified; the `u2_ops.run` path at 256 steps is the one that
re-verified the 46 adders and 16 multipliers of the validation above.

Runs executed per encoding (first triple for every program, further triples for survivors): 166 million at 13
symbols, 664 million at 14, 2.66 billion at 15, 10.63 billion at 16 (the 64-step run needed 2.57 and 10.27 billion
at 15 and 16: a few more programs survive the early triples at the longer budget).

### Run record

| node | 13 symbols | 14 symbols | 15 symbols | 16 symbols | overlaps 13 / 14 / 15 / 16 | node total |
|---|---|---|---|---|---|---|
| adler40, 24 threads | 29,283,868 programs, 9.9 s | 117,135,472, 39.6 s | 468,541,887, 157.3 s | 1,874,167,547, 608.6 s | 5.6 / 21.7 / 25.0 / 21.3 s | 889.0 s |
| falke64, 12 threads | 18,302,417, 7.1 s | 73,209,670, 27.9 s | 292,838,679, 111.5 s | 1,171,354,717, 442.8 s | | 589.3 s |
| knecht24, 24 threads | 9,761,289, 6.3 s | 39,045,157, 23.9 s | 156,180,629, 94.9 s | 624,722,516, 379.5 s | 10.1 / 39.6 / 46.4 / 38.1 s | 638.8 s |
| specht32, 12 threads | 9,761,290, 10.6 s | 39,045,157, 41.5 s | 156,180,629, 165.9 s | 624,722,516, 663.6 s | | 881.6 s |

Launched 21:10:30, the last run (the 16-symbol overlap on adler40) finished 21:25:20 workstation time; 15.1
CPU-core-hours in all, 3.4 times the 64-step run of section 5 (which had no 13 and 14; per program at 15 and 16
symbols the 256-step runs took 2.8 times as long on adler40 and 2.9 on falke64). CPU only, nice 10; the
`dna_world_v3` job on adler40 ran throughout. Checks at collection, as in section 5: every shard's header and
summary carry the planned start, count, length, machine and host; the shards of each length sum to 4^L; the
overlap ranges (2^24 programs at 13 symbols, 2^26 at 14, 15, 16) are **identical between adler40 and knecht24 at
all four lengths** (96 survivor counts and the empty found lists each).

### What is now covered

| budget | 1 to 12 symbols | 13, 14 symbols | 15, 16 symbols | multiply-accumulates |
|---|---|---|---|---|
| 64 steps | numpy (Ghost report, sections 3 and 4) | numpy (section 4 there) | this note, sections 4 and 5 | 0 |
| 256 steps | numpy (Ghost report, section 4b) | this section | this section | 0 |

Both budgets, both encodings, every program of 1 to 16 symbols, machine 50098 with the fixed layout of section 1.
