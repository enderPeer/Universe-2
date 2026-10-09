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
17 of the 48 input triples.

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

## 3. Validation (adler40, 24 threads, 10.3 s in all)

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
cell 1). The 12-symbol length alone took 6.8 s for all six primitives on 24 threads, against 370.5 s for the numpy
search on 24 processes.

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
than the length is the obstacle, but that is a reading, not a result.

So the BitNet question is settled for this machine at this program size: the model's inner step is not a
primitive here, and a token would need a composite of the found adder and multiplier (two programs, a copy in
between, and the layout they expect), which is the "big program from blocks" question of docs/04, not a one-program
answer.

## 7. Caveats

- The definition is fixed: inputs in cells 0, 1, 2 in the order (acc, w, x), head at 0, output in one fixed cell or
  under the final head, mod 4 with no carry. A multiply-accumulate with another layout (inputs elsewhere, w and x
  swapped, output split over two cells, output defined relative to the head's start) is not excluded by this run.
- 64 steps. The numpy search also tried 256 steps up to 12 symbols (no multiply-accumulate; nine times as many
  adders). 256 steps at 13 to 16 symbols would cost about four times this run and was not done.
- Exactly this machine. The other two W = 2 candidates (`INC, INC_R1, INC_L1, JNZB` and `L1, INC_L1, FLIP_R2, JNZB`)
  were searched up to 10 symbols only (Ghost report, section 4c).
- Only the two weight encodings above; the primitive list of the C mode and the numpy search is the same, so a
  third encoding is a one-line addition.

## 8. Files and commands

- `tools/u2_tables.c` (`prim` mode, built as `u2_tables_mac`), `tools/u2_mac.py` (driver).
- `results/u2_mac/validation_L12_s64.json`, `plan_mac_L15_L16.json`, `mac_L15.json`, `mac_L16.json`, `raw/`
  (every shard's stdout and stderr, named by node and range).
- Reference search and report: `C:\Users\end\Ghost\bitnet_u2\` (`prim_search.py`, `u2_vec.py`, `REPORT.md`,
  `results/prim_*_W2_L12_s64.json`, `..._L14_s64.json`), outside this repository.

```
python tools/u2_mac.py build
python tools/u2_mac.py validate
python tools/u2_mac.py launch --L 15 --L 16
python tools/u2_mac.py status --L 15 --L 16
python tools/u2_mac.py collect --L 15 --L 16
```
