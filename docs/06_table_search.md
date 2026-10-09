# The 2-bit table search: every one-symbol instruction, every four-instruction machine, scored on all short programs

Date: 2026-10-09. Method: Universe-7's (choose the law by exhaustive search), applied to a tape machine with
one 2-bit symbol per instruction. Tools: [`tools/u2_ops.py`](../tools/u2_ops.py) (catalog and Python reference),
[`tools/u2_tables.c`](../tools/u2_tables.c) (the sweeper, C + OpenMP), [`tools/u2_table_search.py`](../tools/u2_table_search.py)
(cluster driver: launch, status, collect, verify, rank). Results: `results/u2_tables/`.

## 1. All possible instructions for 2 bits

An instruction with no operand bits can only do five kinds of things. The catalog enumerates them with the strides
and values that a 64-step run can use (strides 1 to 3; larger strides would leave the 256-cell ring in 64 steps):

| # | name | group | effect |
|---|---|---|---|
| 0-5 | R1 L1 R2 L2 R3 L3 | move | head += 1, -1, 2, -2, 3, -3 |
| 6 | INC | write | cell += 1 (mod 2^W) |
| 7 | DEC | write | cell -= 1 |
| 8 | FLIP | write | cell ^= 1 (low bit) |
| 9 | NOT | write | cell ^= all ones |
| 10 | ZERO | write | cell = 0 |
| 11 | SET | write | cell = all ones |
| 12 | SHL | write | cell <<= 1 (mod 2^W) |
| 13 | INC_R1 | pen | cell += 1, head += 1 |
| 14 | INC_L1 | pen | cell += 1, head -= 1 (Bohm's P'' lambda) |
| 15 | FLIP_R1 | pen | cell ^= 1, head += 1 (Universe-7's flip pen) |
| 16 | FLIP_L1 | pen | cell ^= 1, head -= 1 |
| 17 | FLIP_R2 | pen | cell ^= 1, head += 2 |
| 18 | FLIP_R3 | pen | cell ^= 1, head += 3 |
| 19 | SET_R1 | pen | cell = all ones, head += 1 |
| 20 | SET_R3 | pen | cell = all ones, head += 3 |
| 21 | ZERO_R1 | pen | cell = 0, head += 1 |
| 22 | ZERO_R3 | pen | cell = 0, head += 3 |
| 23 | SKZ | control | skip the next instruction if cell == 0 |
| 24 | SKNZ | control | skip the next instruction if cell != 0 |
| 25 | JZS | control | if cell == 0: PC = 0 |
| 26 | JNZS | control | if cell != 0: PC = 0 |
| 27 | JNZB | control | if cell != 0: PC -= 2 (a two-instruction loop) |
| 28 | OPEN | control | `[`: if cell == 0 jump to after the matching `]` (unmatched: PC = 0) |
| 29 | CLOSE | control | `]`: if cell != 0 jump to after the matching `[` (unmatched: PC = 0) |
| 30 | HALT | control | stop |
| 31 | NOP | control | nothing |
| 32 | LDA | accumulator | A = cell |
| 33 | STA | accumulator | cell = A |
| 34 | SWAP | accumulator | exchange A and cell |
| 35 | ADDA | accumulator | cell += A (mod 2^W) |

What is deliberately not in the catalog: anything with an operand (there are no bits for one), strides above 3,
instructions that act on a second register (A is the only implicit register), and random access. Every
four-element subset of these 36 operations is a machine; a machine's four operations are assigned to the symbols
0, 1, 2, 3 in some order, but relabelling symbols only permutes the programs of each length, so the distinct-tape
score is the same for every order and only the **C(36, 4) = 58,905 subsets** need to be scored.

Machine: ring of 256 cells of W bits (W = 1 and W = 2 swept), head at cell 0, A = 0, PC wraps at the program
length (NANO's convention: the program repeats), 64 instructions executed, HALT stops early. The result of a program
is its final tape, in canonical form (first nonzero cell, then the values up to the last nonzero cell; the blank
tape is one result). The score of a machine is the number of **distinct final tapes over all programs of 1 to 10
symbols**: 4 + 16 + ... + 4^10 = **1,398,100 programs per machine**, 20 bits at the longest, which is 7 bits
beyond the depth of Universe-7's four-operation round.

## 2. Validation before the run

- **Reference agreement.** 403 random (W in {1, 2, 4}, step budget 64 or 200, table id, program of 1 to 12
  symbols) cases, including table 0 and table 58,904: the C sweeper on adler40 and the Python reference on the
  workstation give identical final tape, head, A, PC, halted flag and step count: **0 mismatches**. The
  combinatorial unranking of table ids was checked at the same time (the C binary prints the operation names).
- **Timing calibration** (L = 10, 64 steps, 24 tables): adler40 1.0 s at 24 threads (both W), knecht24 1.5 s for
  12 tables at 24 threads, specht32 1.5 s for 12 at 12 threads, falke64 0.8 s for 12 at 12 threads. About 1 ns
  per executed instruction per thread.
- **Cross-node agreement** (checked at collection): tables 20,000 to 20,199 are computed three times for each W,
  by the shard that owns them and separately on adler40 (Intel i9-12900K) and knecht24 (AMD Threadripper 1920X);
  all three lines (per-length union counts, halted and nonblank counts, and the order-independent checksum over
  all 1,398,100 program tape hashes) must be identical.
- **Accounting.** Every shard file must have exactly `count` lines with contiguous table ids; the merge must hold
  all 58,905 ids exactly once per W.

## 3. Cluster overview at launch (2026-10-09, 14:09 to 14:20 local time)

| node | CPU used by this run | GPUs at launch | RAM | disk free |
|---|---|---|---|---|
| adler40 | 24 threads (i9-12900K), nice 10 | RTX 4090 idle (31 W), RTX 4080 idle but holding 1.3 GB of an old homunculi process | 2 / 61 GB used | 126 GB |
| falke64 | 12 threads (Ryzen 5 7600X), nice 10 | both R9700 at 99 %: Universe-1 exp11, Avida length-9 sweep (`u1_avida_vk --len 9`) | 3 / 60 GB | 1.5 TB |
| knecht24 | 24 threads (Threadripper 1920X), nice 10 | 3x RTX 3060 idle | 11 / 60 GB | 38 GB |
| specht32 | 12 threads (i7-8700K), nice 10 | RX 9060 XT and RX 9070 XT at 97 to 99 %: the same Avida sweep | 2 / 44 GB | 222 GB |

The GPUs were not used and not touched; the Avida sweep kept its cards. The table search is CPU-only and ran
at `nice -n 10` so that the GPU jobs' host threads kept priority. The workstation had no part in the compute.

### Shards (proportional to measured speed; `results/u2_tables/plan.json`)

| node | W = 1 tables | W = 2 tables | extra |
|---|---|---|---|
| adler40 | 0 to 25,703 (25,704) | same range | overlap 20,000 to 20,199 for both W |
| falke64 | 25,704 to 41,768 (16,065) | same | |
| knecht24 | 41,769 to 50,336 (8,568) | same | overlap 20,000 to 20,199 for both W |
| specht32 | 50,337 to 58,904 (8,568) | same | |

Launch incident, recorded for honesty: the first launch command hung because the remote background job inherited
the SSH session's standard input, so only adler40 started at 14:11; falke64 followed at 14:15, knecht24 and specht32
at 14:20 after the launcher was changed to fork the job in a subshell with all three streams detached. Nothing was
computed twice as a result; the shards are disjoint by construction and the overlap is deliberate.

## 4. Results

*Filled in by `python tools/u2_table_search.py collect` when all shards report done; see `results/u2_tables/report.json`
and the merged `tables_W1_L10_s64.tsv`, `tables_W2_L10_s64.tsv` (one line per machine: union of distinct tapes
after each program length, halted and nonblank program counts, checksum).*
