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

Where they come from: the moves and pens from Universe-7 (whose search chose flip/set/clear pens with strides 3, 4
and 8), the writes from NANO (INC, DEC, NOT), Universe-1 (INC, DEC) and Brainfuck (+, -), the control operations
from Universe-1's champion (SKZ), NANO's PC wrap (restart), Bohm's P'' (the brackets) and HALT, the accumulator
operations from Universe-1's SWAP and ADD with the operand removed. Not in the catalog: anything with an operand
(there are no bits for one), strides above 3, a second register, random access, conditional moves, seek, and
two-cell operations (copy to the neighbour); those are the second round.

How many operand-free instructions exist in principle: a rule reads the local view (the cell, and A if present)
and chooses a new cell value, a head displacement and a PC action. For 1-bit cells without A that is 4 write
rules x 49 cell-dependent stride choices x 49 cell-dependent PC actions = 9,604 instructions; with a 1-bit A
about 1.5 billion. The 36 are the atoms of that space plus the write-and-move combinations known to matter; a
4-instruction search over them explores their combinations.

Every four-element subset of the 36 is a machine; a machine's four operations are assigned to the symbols 0, 1, 2, 3
in some order, but relabelling symbols only permutes the programs of each length, so the distinct-tape score is the
same for every order and only the **C(36, 4) = 58,905 subsets** need to be scored.

Machine: ring of 256 cells of W bits (W = 1 and W = 2 swept), head at cell 0, A = 0, PC wraps at the program
length (NANO's convention: the program repeats), 64 instructions executed, HALT stops early. The result of a program
is its final tape in canonical form (first nonzero cell, then the values up to the last nonzero cell; the blank tape
is one result). The score of a machine is the number of **distinct final tapes over all programs of 1 to 10
symbols**: 4 + 16 + ... + 4^10 = **1,398,100 programs per machine**, 20 bits at the longest, 7 bits beyond the
depth of Universe-7's four-operation round.

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

The GPUs were not used and not touched; the Avida sweep kept its cards at 95 to 99 % throughout (checked at 14:25:
its shard files kept appearing on both nodes). The table search is CPU-only and ran at `nice -n 10` so that the GPU
jobs' host threads kept priority. The workstation had no part in the compute.

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

## 4. Results (collected 2026-10-09 14:47)

Files: `results/u2_tables/report.json` (ranking, top 50 per width, operation frequencies, shard log, overlap
checks), `tables_W1_L10_s64.tsv` and `tables_W2_L10_s64.tsv` (one line per machine: union of distinct tapes after
each program length 1..10, halted and nonblank program counts, checksum).

### Run record

| node | W = 1 shard | W = 2 shard | overlap (200 tables) |
|---|---|---|---|
| adler40, 24 threads | 25,704 tables, 740.9 s | 755.2 s | 7.6 s / 7.4 s |
| falke64, 12 threads | 16,065 tables, 664.1 s | 674.2 s | |
| knecht24, 24 threads | 8,568 tables, 399.7 s | 401.6 s | 11.6 s / 11.9 s |
| specht32, 12 threads | 8,568 tables, 723.9 s | 726.0 s | |

Total: 2 x 58,905 machines x 1,398,100 programs = **164.7 billion program runs** of up to 64 instructions, about
24 minutes of wall time per node, 11 CPU-core-hours in all.

Verification: every shard has exactly its `count` lines with contiguous ids; the merge holds all 58,905 ids once
per width; the 200 overlap tables are **identical line for line** between adler40 (Intel), knecht24 (AMD) and the
owning shard for both widths (200 / 200 / 200 each).

### The ceiling, and the machines that reach it

Because the PC wraps, a program that is a repetition of a shorter word (`0`, `00`, `0000`, `0101`) is the same
program and must give the same tape. Of the 1,398,100 words of 1 to 10 symbols, 1,464 are such repetitions (a
Mobius count over the divisors of each length), so **1,396,636 distinct programs** exist and no machine can score
higher. **12 machines at W = 1 and 47 at W = 2 reach exactly 1,396,636**: every distinct program draws a
different tape, at every length (their per-length unions are 4, 16, 76, 316, 1,336, 5,356, 21,736, 87,016,
349,096, 1,396,636, which are the cumulative primitive-word counts). This is Universe-7's 100 % result again, now
on a 1-D tape, over 20-bit programs, and with the whole four-instruction space searched rather than a hand-picked
candidate set.

All perfect machines are made of **pens** (write and move in one step) with at most one pure move:

| W | perfect machines | composition | operations used (count among the perfect machines) |
|---|---:|---|---|
| 1 | 12 | 6 with four pens, 6 with three pens and one move | FLIP_R2 in all 12; R3, FLIP_R3, SET_R3, ZERO_R3 in 6 each; INC_R1, FLIP_R1, SET_R1 in 4 each |
| 2 | 47 | 25 with four pens, 22 with three pens and one move | FLIP_R2 31, SET_R1 28, SET_R3 28, FLIP_R3 27, INC_R1 18, FLIP_R1 18, R3 16, ZERO_R3 16, R2 6 |

Example, W = 1, table 25,729: `R3, INC_R1, FLIP_R2, FLIP_R3`. Example, W = 2, table 16,275: `R2, INC_R1, FLIP_R2,
SET_R1`. The recipe is the one Universe-7 found by hand and confirmed by its own search: flip pens with *different*
strides (2 and 3 here, 4 and 8 there) so that paths rarely revisit a cell, plus a set or clear pen.

### What does not reach the top

| machine class | best W = 1 | best W = 2 | best example |
|---|---:|---:|---|
| any conditional (SKZ, SKNZ, JZS, JNZS, JNZB, brackets) | 263,919 (18.90 %), rank 768 | 377,923 (27.06 %), rank 1,750 | `L3, INC_L1, FLIP_R3, JNZB` |
| with the accumulator | 262,303 (18.78 %) | 408,842 (29.27 %) | `INC_R1, INC_L1, FLIP_R3, SWAP` |
| with HALT | 98,003 (7.02 %); 93.7 % of programs halt early | 107,844 (7.72 %) | `INC_R1, FLIP_R2, FLIP_R3, HALT` |
| no pens at all | 45,451 (3.25 %) | 162,279 (11.62 %) | `R2, R3, INC, SET` |
| P'' exactly (`R1, INC_L1, OPEN, CLOSE`) | 5,493 (0.39 %), rank 24,454 | 11,626 (0.83 %), rank 23,066 | |
| skip design S4 (`R1, L1, INC, SKZ`) | 7,527 (0.54 %) | 29,559 (2.12 %) | |
| Brainfuck without loops (`R1, L1, INC, DEC`) | 3,126 (0.22 %) | 45,795 (3.28 %) | |

Distribution over all 58,905 machines: quartiles 51 / 3,775 / 19,061 at W = 1 and 51 / 5,691 / 41,473 at W = 2;
8,856 machines (15 %) produce a single tape (no write that can ever fire). Going from 1-bit to 2-bit cells does not
raise the ceiling (it is already reached) but lifts the middle of the field by about 1.5x and the controlled
machines by 1.4 to 4x.

### Reading

1. **Variety is maximised by machines that cannot compute.** The perfect machines have no conditional and no
   register: they are drawing laws, exactly like Universe-7's. Every conditional costs variety because it makes
   programs collapse onto the same tape (a skipped instruction is a shorter program). The Turing-complete P''
   sits at rank 24,454 of 58,905.
2. **This is the right answer to the question asked, and the wrong question for a computing machine.** The score
   "distinct final tapes from a blank tape" measures expressiveness as drawing. A machine that computes functions
   of inputs needs a conditional, and the best machines *with* a conditional (the `JNZB` two-instruction loop plus
   pens, 19 to 27 %) are the candidates. The next scoring mode, functions on inputs placed on the tape (NANO's and
   Universe-1's measure, exhaustive over all input values), will rank those.
3. **The 1,464 repetitions are the only loss.** A perfect machine is injective on programs: the tape *is* the
   program, so reading a block back from the world (the genome-like soup of docs/05, section 6b) is lossless.
4. **Two bits per instruction are enough for perfect variety at 20 bits.** NANO's 40-bit space has 2^40 programs
   but only 219,846 probe behaviours and six classes; the measures differ, but the direction is the same as
   Universe-7's: the dense alphabet pays.

### Next

- Function scoring on inputs for the top 2,000 machines of each width (all pairs of inputs on the tape, result
  cells read after 64 steps; exhaustive over inputs, as the NANO adders were proven).
- The chosen drawing law and the best computing law to 2^40 (20 symbols) on the GPUs once the Avida sweep frees
  them: a CUDA/Vulkan port of `u2_tables.c` with the same per-chunk checksum discipline.
- Second catalog round (conditional moves, seek, copy-to-neighbour, two-bit rotates) and the self-modifying
  variant where the tape is the program.
