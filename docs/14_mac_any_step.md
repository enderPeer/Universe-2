# The multiply-accumulate search at every step budget: all 256 budgets at once, on the cluster's CPUs and GPUs

Date: 2026-10-10. docs/13 settled that no program of 1 to 16 symbols of machine 50098 `INC_R1, INC_L1, FLIP_R2, JNZB`
(W = 2, ring of 256 two-bit cells, head at 0, PC wraps) computes acc + w * x mod 4 when the tape is read after
exactly 64 or exactly 256 instructions. Those are two snapshots. A program that has the right answer on the tape
after, say, 100 instructions and overwrites it before 256 is invisible to both. This note closes that gap: every
program of 1 to 16 symbols is checked at **every budget s = 1, 2, ..., 256**, in both weight encodings. Section 11
repeats everything on a ring of 512 cells with budgets 1 to 512.

**Result: no program of 1 to 16 symbols of machine 50098 is a multiply-accumulate at any budget from 1 to 256
instructions, in either weight encoding.** 5,726,623,060 programs per encoding, each checked at all 256 budgets
(1 to 12 symbols by the Python reference and the C mode, 13 to 16 on the four nodes' CPUs and five GPUs, 7 min 39 s
wall). With the budget free the best programs get further than at a fixed budget, 24 of the 48 input triples
instead of 22, and one 16-symbol program of the BitNet encoding handles every triple with w = +1 and w = -1 at
some budget and fails only on w = 0; nothing survives the zero weight, and nothing comes near the 48.

## 1. The question, made exact

Inputs (acc, w, x) in cells 0, 1, 2, everything else blank, head at cell 0. A program P is a multiply-accumulate
**at budget s** when, for every one of the 48 input triples, the tape after exactly s instructions holds
(acc + w * x) mod 4 in one fixed cell c (the same c for all triples), or in the cell under the head after those s
instructions (the same rule for all triples). docs/13 asked this for s = 64 and s = 256; this note asks whether
there is any s in 1..256 for which P is a multiply-accumulate, and records every such s. Same two encodings as
docs/13 section 1 (MAC_T: w in {1, 3, 0}; MAC_B: BitNet code c in {2, 0, 1} meaning c - 1), same triple order
(nonzero targets first). The other primitives of the C mode (COPY, ADD, MUL_T, CARRY) get the same treatment on
small lengths, as validation material.

A program alive after input i means: some budget s still has a candidate cell or a valid head rule after inputs
1..i. The survivor counts below are therefore counts over programs, each program counted once however many
budgets it survives at.

## 2. Three implementations

- **CPU, `primany` mode of `tools/u2_tables.c`** (built as `u2_tables_any` on the four nodes). One candidate mask per
  step, cand[s] (256 cell bits and a head bit, s = 1..256), kept across the inputs and only ever shrinking; the
  program stops at the first input after which no step has a candidate. The set of cells holding the target changes
  by at most one bit per instruction (an instruction writes at most the cell under the head), so the per-step
  "cell == target" mask is maintained incrementally. The instruction switch is a verbatim copy of the one in `run()`,
  inserted by script. Output per primitive: survivors after each input, found programs with their steps, cells and
  head flag, and `found_by_step` (how many programs are the primitive at each budget).
- **GPU, `gpu/u2_mac_any.cu`** (`u2_mac_any_gpu`, `-arch=sm_89` on adler40's RTX 4090 and RTX 4080, `sm_86` on
  knecht24's three RTX 3060). One thread per program, deliberately a different algorithm: the first four inputs
  run in lockstep and only the per-step OR of their joint masks is accumulated, so a program that dies within four
  inputs (97 % of them) touches no per-step memory; a survivor is re-run in lockstep to fill cand[s] and then takes
  the remaining inputs one by one. The four tapes of a thread are packed two bits per cell (64 bytes each) in shared
  memory, interleaved across the block; the equality masks live in registers. (A first version with byte tapes in
  local memory was DRAM-bound: 6.3 M programs/s on both Ada cards alike and 0.55 M/s on a 3060; the packed version
  is 2.2 and 4.4 times faster.) F lines carry program and step ranges; the per-step cells of a found program are
  recovered by the CPU mode on that one program.
- **Python reference, `tools/u2_mac_any_ref.py`**: a copy of the loop of `u2_ops.run` that yields the state after
  every step, checked against `u2_ops.run` on 1,500 random (machine, program, input, budget) cases; per program the
  tape after every step of every input is a (256 x 256) numpy array and the candidate set is the intersection of the
  boolean "cell == target" arrays, head rule alongside. No bit tricks, nothing incremental, nothing shared with the
  two fast implementations beyond the input definitions.

Driver: `tools/u2_mac_any.py` (build, selfcheck, ref, validate, calibrate, launch, status, collect).

## 3. Validation

`results/u2_mac_any/validation_L12.json`, `ref_L1.json` to `ref_L12.json`. For every length the CPU mode on adler40
and the GPU program on adler40's RTX 4090 and knecht24's first RTX 3060 ran the full length; the reference ran on
the workstation (24 processes). Compared per primitive: survivors after every input, the set of found programs,
their per-step cells and head flags (CPU vs reference), `found_by_step`, and the GPU outputs against the CPU output.

| length | primitives | found programs (any budget 1..256) | reference | all identical |
|---|---|---|---|---|
| 1 to 3 | all six | none | 0.2 to 0.6 s | yes |
| 4 | all six | COPY 2 | 0.7 s | yes |
| 5 | all six | COPY 16 | 0.8 s | yes |
| 6 | all six | COPY 121 | 1.4 s | yes |
| 7 | all six | COPY 693, CARRY 4 | 3.9 s | yes |
| 8 | all six | COPY 3,266, CARRY 13, **ADD 1** | 13.6 s | yes |
| 9 | ADD, MUL_T, MAC_T, MAC_B, CARRY | ADD 1, CARRY 40 | 50.0 s | yes |
| 10 | ADD, MUL_T, MAC_T, MAC_B, CARRY | ADD 5, MUL_T 11, CARRY 124 | 271.7 s | yes |
| 11 | MAC_T, MAC_B | none | 333.0 s | yes |
| 12 | MAC_T, MAC_B | none | 1,379.8 s (28 processes) | yes |

MAC_T and MAC_B: no program of 1 to 12 symbols at any budget. The validation also exercised the found path
thoroughly (4,000 copiers, 181 carries, 7 adders, 11 multipliers with their exact steps and cells). The any-step
survivor counts summed over 1 to 12 symbols are at least the single-budget counts of docs/13 at every input, for both
budgets and both encodings, as they must be. At 5 symbols the self-check that every build runs expects COPY to find
16 programs and the 64-step copier `01302` of docs/13 among them with steps 57 to 64 in cell 1.

Side result of the validation (not the question of this note): with the budget free, the machine has an adder at
8 symbols (one program; the shortest at a fixed budget was 10 symbols at 256 steps and 11 at 64), a carry at 7 and
a copier at 4.

## 4. Calibration and shards (`calibration_L13.json`)

All 67,108,864 programs of 13 symbols, both encodings, on every device at once, nice 10; the nine outputs are
identical (survivors, found lists, found_by_step):

| device | hardware | seconds | M programs/s |
|---|---|---:|---:|
| adler40-cpu | i9-12900K, 24 threads | 50.7 | 1.32 |
| adler40-gpu0 | RTX 4090 | 4.8 | 13.98 |
| adler40-gpu1 | RTX 4080 | 7.5 | 8.95 |
| falke64-cpu | Ryzen 5 7600X, 12 threads | 106.7 | 0.63 |
| knecht24-cpu | Threadripper 1920X, 24 threads | 84.7 | 0.79 |
| knecht24-gpu0, 1, 2 | RTX 3060 (each) | 28.1, 27.4, 27.2 | 2.39, 2.45, 2.47 |
| specht32-cpu | i7-8700K, 12 threads | 161.7 | 0.42 |

33.4 M programs/s in all; the RTX 4090 alone does 42 % of the work, the five GPUs 89 %. The shards of each length are
proportional to these speeds, one detached chain per device (a node's CPU and GPUs run at the same time), plus an
overlap range per length run on four devices of different kinds: adler40-cpu (Intel), knecht24-cpu (AMD),
adler40-gpu0 (Ada) and knecht24-gpu0 (Ampere): 10,000,000 + 2^24 (13 symbols), 100,000,000 + 2^26 (14),
123,456,789 + 2^26 (15), 3,000,000,000 + 2^26 (16), the ranges of docs/13.

## 5. Cluster at launch (2026-10-10, 15:16:33 workstation time)

| node | used | load before | other work at launch |
|---|---|---|---|
| adler40 | 24 threads, RTX 4090, RTX 4080 | 0.00 | an idle `homunculi` process holding memory on the RTX 4080 since 2026-10-01 |
| falke64 | 12 threads | 0.1 | another user's `llama-server`, 13 % of a core (not touched) |
| knecht24 | 24 threads, 3 x RTX 3060 | 0.3 | the Windows VM |
| specht32 | 12 threads | 0.0 | none |

(nvidia-smi on adler40 reports an NVML driver/library mismatch and shows nothing; CUDA itself works, and the GPU
probe found the two cards idle.) Twenty seconds after launch every CPU shard ran at full thread count and the three
3060s reported 100 % utilization at about 105 W.

## 6. Results (collected 15:25; `results/u2_mac_any/any_L13.json` to `any_L16.json`, shard outputs in `raw/`)

Programs alive after each input, where alive means that some budget s in 1..256 still has a candidate cell or a
valid head rule. Inputs 1 to 12 are the positive weight, 13 to 24 the negative one, 25 to 36 the zero weight (all
with nonzero target), 37 to 48 the twelve zero-target triples.

| every budget 1..256 | 13, MAC_T | 13, MAC_B | 14, MAC_T | 14, MAC_B | 15, MAC_T | 15, MAC_B | 16, MAC_T | 16, MAC_B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| programs | 67,108,864 | 67,108,864 | 268,435,456 | 268,435,456 | 1,073,741,824 | 1,073,741,824 | 4,294,967,296 | 4,294,967,296 |
| alive after inputs 1 to 3 | all | all | all | all | all | all | all | all |
| after 4 | 16,344,142 | 14,651,655 | 65,588,359 | 58,768,474 | 265,097,056 | 237,781,525 | 1,062,835,796 | 953,713,848 |
| after 5 | 4,863,285 | 4,915,503 | 19,544,336 | 19,751,129 | 79,301,862 | 80,113,313 | 317,633,203 | 320,682,157 |
| after 6 | 3,188,210 | 3,222,537 | 12,743,137 | 12,861,607 | 51,291,595 | 51,698,577 | 205,340,685 | 206,833,636 |
| after 9 | 28,472 | 28,473 | 104,997 | 106,823 | 393,749 | 405,071 | 1,481,167 | 1,534,214 |
| after 12 | 3,073 | 2,859 | 9,125 | 8,679 | 28,517 | 27,977 | 91,007 | 92,365 |
| after 13 | 761 | 605 | 2,156 | 1,715 | 7,103 | 5,870 | 21,010 | 19,065 |
| after 14 | 178 | 85 | 527 | 306 | 1,683 | 1,109 | 5,187 | 3,924 |
| after 15 | 81 | 25 | 234 | 89 | 735 | 310 | 2,032 | 1,020 |
| after 16 | 43 | 11 | 125 | 34 | 382 | 126 | 1,069 | 415 |
| after 17 | 10 | 10 | 40 | 33 | 163 | 104 | 528 | 351 |
| after 18 | 2 | 3 | 6 | 9 | 35 | 30 | 130 | 83 |
| after 19 | **0** | **0** | 4 | 3 | 28 | 10 | 115 | 19 |
| after 20 | 0 | 0 | 1 | **0** | 7 | 6 | 27 | 8 |
| after 21 | 0 | 0 | **0** | 0 | **0** | 6 | 5 | 8 |
| after 22 | 0 | 0 | 0 | 0 | 0 | **0** | 5 | 2 |
| after 23 | 0 | 0 | 0 | 0 | 0 | 0 | **0** | 1 |
| after 24 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| after 25 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0** |
| **multiply-accumulates at any budget (alive after 48)** | **0** | **0** | **0** | **0** | **0** | **0** | **0** | **0** |

`found_by_step` is zero at every one of the 256 budgets for every length and encoding: no program is a
multiply-accumulate at any budget. Inputs 1 to 3 are survived by every program at every length, because at budget
s = 1 the tape still holds x in cell 2 and the first three triples are (0, +1, x) with target x; input 4,
(acc, x) = (1, 0) with target 1, is the first that needs acc and x combined, and it removes three quarters of the
programs.

**Where the best programs die.** With the budget free the frontier moves two to seven triples past the snapshot
results of docs/13 (17 at 64 steps, 22 at 256 steps for the best length):

| length | MAC_T: survives, then dies on | MAC_B: survives, then dies on |
|---|---|---|
| 13 | 18 inputs (2 programs); input 19, (acc, w, x) = (2, -1, 0), target 2 | 18 (3 programs); input 19, (2, -1, 0) |
| 14 | 20 (1 program); input 21, (2, -1, 3), target 3 | 19 (3); input 20, (2, -1, 1), target 1 |
| 15 | 20 (7); input 21, (2, -1, 3) | 21 (6); input 22, (3, -1, 0), target 3 |
| 16 | 22 (5); input 23, (3, -1, 1), target 2 | **24 (1 program)**; input 25, (1, 0, 0), target 1 |

So one 16-symbol program in the BitNet encoding computes acc + w * x mod 4 at some budget for every one of the 24
nonzero-target triples with w = +1 and w = -1, and fails only when the weight is 0 (code 1): it does not leave acc
alone when there is nothing to add. In the signed encoding (w = 0 is the symbol 0 in cell 1) nothing gets past the
negative weight. These frontier programs were recovered afterwards with a truncated input list (`maxin`, section 7).

Runs executed per encoding (first triple for every program, further triples for the survivors): 0.29 billion at
13 symbols, 1.17 at 14, 4.69 at 15, 18.78 at 16; each run is 256 instructions with the per-step bookkeeping.

### Run record

| device | 13 symbols | 14 | 15 | 16 | overlaps 13 / 14 / 15 / 16 | device total |
|---|---|---|---|---|---|---|
| adler40-cpu (24 thr) | 2,660,053 programs, 2.0 s | 10,640,211, 8.1 s | 42,560,842, 33.0 s | 170,243,370, 131.7 s | 12.6 / 50.9 / 50.0 / 47.9 s | 386.9 s |
| adler40-gpu0 (RTX 4090) | 28,096,806, 1.8 s | 112,387,225, 6.8 s | 449,548,898, 27.3 s | 1,798,195,593, 103.4 s | 1.0 / 4.1 / 3.6 / 3.4 s | 156.2 s |
| adler40-gpu1 (RTX 4080) | 17,981,956, 1.7 s | 71,927,824, 6.6 s | 287,711,295, 26.7 s | 1,150,845,180, 101.5 s | | 144.0 s |
| falke64-cpu (12 thr) | 1,263,961, 1.9 s | 5,055,845, 6.9 s | 20,223,381, 25.0 s | 80,893,522, 103.3 s | | 243.8 s |
| knecht24-cpu (24 thr) | 1,592,263, 2.4 s | 6,369,052, 9.0 s | 25,476,207, 36.0 s | 101,904,827, 145.0 s | 21.9 / 82.1 / 80.0 / 79.6 s | 540.7 s |
| knecht24-gpu0 (RTX 3060) | 4,799,454, 2.8 s | 19,197,818, 10.8 s | 76,791,271, 44.1 s | 307,165,084, 170.7 s | 6.2 / 24.9 / 21.9 / 20.6 s | 330.1 s |
| knecht24-gpu1 (RTX 3060) | 4,922,068, 2.7 s | 19,688,273, 10.6 s | 78,753,092, 43.7 s | 315,012,367, 171.7 s | | 256.1 s |
| knecht24-gpu2 (RTX 3060) | 4,958,260, 2.7 s | 19,833,040, 10.1 s | 79,332,159, 40.6 s | 317,328,634, 161.2 s | | 241.8 s |
| specht32-cpu (12 thr) | 834,043, 2.4 s | 3,336,168, 8.7 s | 13,344,679, 35.0 s | 53,378,719, 140.4 s | | 348.2 s |

Launched 15:16:33, the last shard (knecht24-cpu's 16-symbol overlap) finished 15:24:12 workstation time: 7 min 39 s
wall, 8.2 CPU-core-hours plus 0.31 GPU-hours (the nodes' clocks read 13:16 and 13:24). Nice 10 throughout, nothing
else of ours on the nodes. Checks at collection: every shard's header and summary carry the planned start, count,
length, budget, machine, host and device kind; the shards of each length sum to 4^L; the overlap ranges are
**identical on all four overlap devices at all four lengths** (Intel CPU, AMD CPU, Ada GPU, Ampere GPU: 96 survivor
counts, 512 found_by_step counts and the empty found lists each). Nothing was found, so nothing had to be
re-verified; the re-verification path (the Python reference on the program, then `u2_ops.run` at every claimed
budget) is the one that checked the 7,476 found programs of section 3.

## 7. The frontier programs

Recovered with a truncated input list (`u2_mac_any_gpu ... --maxin N` keeps the first N triples, so the programs
that survive N show up as found, with their budgets), then the per-step cells from the CPU mode on each program,
then re-verified with the Python reference under the same limit and with `u2_ops.run` at every claimed budget
on every kept triple. `results/u2_mac_any/exhibit/` (GPU outputs, `frontier_programs.json`). All 28 programs
of the frontier table of section 6 are here; none exists in both encodings.

| symbols | encoding | program | triples | budgets where it holds | result cell | first failing triple (acc, w or code, x) -> target, what the cell holds |
|---|---|---|---:|---|---|---|
| 16 | BitNet | `2111012110003111` | 24 | 115-116 | cell 0 | (1, 1, 0) -> 1, holds 3 |
| 16 | signed | `0112110113102000` | 22 | 116-256 | cell 0 | (3, 3, 1) -> 2, holds 0 |
| 16 | signed | `1210110113102000` | 22 | 116-256 | cell 0 | (3, 3, 1) -> 2, holds 0 |
| 16 | signed | `1211001113102000` | 22 | 116-256 | cell 0 | (3, 3, 1) -> 2, holds 0 |
| 16 | signed | `1311011000222011` | 22 | 119-120 | cell 0 | (3, 3, 1) -> 2, holds 0 |
| 16 | signed | `1311100100222011` | 22 | 119-120 | cell 0 | (3, 3, 1) -> 2, holds 0 |
| 15 | BitNet | `001011021310110` | 21 | 137-142 | cell 0 | (3, 0, 0) -> 3, holds 1 |
| 15 | BitNet | `001011021311001` | 21 | 137-142 | cell 0 | (3, 0, 0) -> 3, holds 1 |
| 15 | BitNet | `001100121310110` | 21 | 137-140 | cell 0 | (3, 0, 0) -> 3, holds 1 |
| 15 | BitNet | `001100121311001` | 21 | 137-140 | cell 0 | (3, 0, 0) -> 3, holds 1 |
| 15 | BitNet | `010010121310110` | 21 | 137-138 | cell 0 | (3, 0, 0) -> 3, holds 1 |
| 15 | BitNet | `010010121311001` | 21 | 137-138 | cell 0 | (3, 0, 0) -> 3, holds 1 |
| 14 | signed | `10020321013110` | 20 | 142-151 | cell 0 | (2, 3, 3) -> 3, holds 1 |
| 15 | signed | `013203002111131` | 20 | 185-197 | cell 2 | (2, 3, 3) -> 3, holds 0 |
| 15 | signed | `100200321013110` | 20 | 157-158 | cell 0 | (2, 3, 3) -> 3, holds 1 |
| 15 | signed | `100203021013110` | 20 | 169-178 | cell 0 | (2, 3, 3) -> 3, holds 1 |
| 15 | signed | `100203210113110` | 20 | 137-146 | cell 0 | (2, 3, 3) -> 3, holds 1 |
| 15 | signed | `100203210133110` | 20 | 147-156 | cell 0 | (2, 3, 3) -> 3, holds 1 |
| 15 | signed | `100203321013110` | 20 | 147-156 | cell 0 | (2, 3, 3) -> 3, holds 1 |
| 15 | signed | `111222101013110` | 20 | 186,220-235 | cell 0 | (2, 3, 3) -> 3, holds 1 |
| 14 | BitNet | `00111100300111` | 19 | 172-256 | cell 2 | (2, 0, 1) -> 1, holds 3 |
| 14 | BitNet | `00111110030110` | 19 | 179-256 | cell 2 | (2, 0, 1) -> 1, holds 3 |
| 14 | BitNet | `00111110031001` | 19 | 179-256 | cell 2 | (2, 0, 1) -> 1, holds 3 |
| 13 | BitNet | `0113102020111` | 18 | 87-256 | cell 0 | (2, 0, 0) -> 2, holds 0 |
| 13 | BitNet | `0113331020211` | 18 | 89-256 | cell 0 | (2, 0, 0) -> 2, holds 0 |
| 13 | BitNet | `1212110133201` | 18 | 98 | cell 2 | (2, 0, 0) -> 2, holds 3 |
| 13 | signed | `0210101311100` | 18 | 161-162,190-204 | cell 0 | (2, 3, 0) -> 2, holds 1 |
| 13 | signed | `3110112110221` | 18 | 223 | under the head | (2, 3, 0) -> 2, holds 0 |

The 24-triple program `2111012110003111` (BitNet encoding) leaves acc + w * x in cell 0, the accumulator's own cell,
after exactly 115 or 116 instructions for all 24 triples with code 2 (w = +1) and code 0 (w = -1); on the first
zero-weight triple, (acc, code, x) = (1, 1, 0) with target 1, cell 0 holds 3 at those budgets. The five 22-triple
programs of the signed encoding all die on (3, -1, 1), target 2, with 0 in their result cell; three of them hold
the result in cell 0 from budget 116 to the end of the run, two only at budgets 119 and 120. Most frontier programs
write cell 0 or cell 2; the 13-symbol program `3110112110221` is the only one whose result is read under the head.
The budgets at which these programs hold lie between 87 and 256, never 64: the 64-step snapshot of docs/13 could
not have seen any of them. The 256-step snapshot saw exactly the three 16-symbol programs whose range reaches 256
(section 9 of docs/13 counted three programs alive after 22 inputs); the other 25 were invisible to it.

## 8. Reading

For this machine and this layout, the budget was never the obstacle at a fixed 64 or 256 steps that docs/13
could not exclude: freeing it moves the frontier from 17 or 22 triples to at most 24 of 48, and no program at any
length is a multiply-accumulate at any budget up to 256. The frontier now grows with length in both encodings (18,
20, 20, 22 and 18, 19, 21, 24 triples from 13 to 16 symbols), and the one 24-triple program of the BitNet encoding
shows what 16 symbols can do: add x, subtract x, but not leave acc alone on a zero code. Whether 17 or 18 symbols
close the last block is open (4^17 = 17 billion programs per encoding, about half an hour at the rate of this run).

With docs/13 this settles the BitNet question for machine 50098 up to 16 symbols with no step-budget caveat left:
the inner step of the model is not a one-program primitive of the machine under the fixed layout; the composite
route (the found adder and multiplier, a copy between them) remains the only one.

## 9. Caveats

- The layout is fixed as in docs/13: inputs in cells 0, 1, 2 in the order (acc, w, x), head at 0, output in one
  fixed cell or under the head, mod 4 with no carry. Swapped inputs, split outputs or outputs defined relative to
  the head's start are not covered.
- Budgets 1 to 256 on the 256-cell ring (sections 6 and 7) and 1 to 512 on a 512-cell ring (section 11). The candidate
  mask has one slot per step, so a larger budget is a recompile (MAXSTEPS) and proportionally more time.
- Exactly this machine and these two encodings.
- The GPU program refuses machines with OPEN/CLOSE (no bracket matching) and widths other than 1, 2 and 4 bits (packed
  tapes); the CPU mode has neither limit.

## 10. Files and commands

- `tools/u2_tables.c` (`primany` mode, built as `u2_tables_any`), `gpu/u2_mac_any.cu` (`u2_mac_any_gpu`),
  `tools/u2_mac_any_ref.py` (reference), `tools/u2_mac_any.py` (driver).
- `results/u2_mac_any/`: `ref_L1.json` to `ref_L12.json`, `validation_L12.json`, `calibration_L13.json` (the shards; `calibration_L12.json` is the first, local-memory kernel at 12 symbols),
  `plan_any_L13_L14_L15_L16.json`, `any_L13.json` to `any_L16.json`, `raw/` (every shard's stdout and stderr by device
  and range), `exhibit/` (the frontier runs of section 7).

```
python tools/u2_mac_any.py build            # scp, gcc and nvcc on every node, 5-symbol self-check on all nine devices
python tools/u2_mac_any.py ref              # the Python reference, 1..12 symbols (local, 24 processes)
python tools/u2_mac_any.py validate         # CPU and two GPUs against the reference, 1..12
python tools/u2_mac_any.py calibrate        # 13 symbols on every device, speeds for the shards
python tools/u2_mac_any.py launch --L 13 --L 14 --L 15 --L 16
python tools/u2_mac_any.py status --L 13 --L 14 --L 15 --L 16
python tools/u2_mac_any.py collect --L 13 --L 14 --L 15 --L 16
```

## 11. The same on a ring of 512 cells with budgets 1 to 512

Asked on 2026-10-10 after section 6: the machine's ring is 256 cells and the budget 256 instructions; does a larger
ring with twice the budget change anything? The ring size matters even at 256 steps, because a program that walks
one way (FLIP_R2 moves the head two cells per instruction) wraps around a 256-cell ring within 128 instructions and
then overwrites what it left behind; on 512 cells it does not. So this is a different, larger machine variant, and
its results are not a superset of sections 6 and 7: a program can be a primitive at some budget on one ring and not on
the other.

**Result: no program of 1 to 16 symbols is a multiply-accumulate at any budget from 1 to 512 on a ring of 512
cells either, in either encoding.** Validated the same way as the small ring (reference, CPU and two GPUs identical for 1 to 11 symbols; nine devices identical at 13; four device kinds identical on 2^24-program overlaps at 13 to 16), run on the nine devices in 15 min 53 s. The frontier is exactly the one of the small ring: 18, 20, 20, 22 triples (signed) and 18, 19, 21, 24 (BitNet) at 13 to 16 symbols, and no budget between 257 and 512 adds a surviving program.

### What changed in the tools

Ring size and step limit are compile-time parameters of both programs (`-DNCELL=512 -DMAXSTEPS=512`; every head
move wraps with `& (NCELL - 1)`, the cell masks have NCELL / 64 words, the step masks MAXSTEPS / 64). The default
build is unchanged: with the parameters at 256 the C sweeper gives byte-identical output to the committed binary for
the any-step mode and for the snapshot `prim` mode of docs/13 (checked on adler40 before the variant was built). The
driver takes `--ncell 512 --steps 512`, builds and names the variant binaries (`u2_tables_any_n512_s512`,
`u2_mac_any_gpu_n512_s512`), keeps its results in `results/u2_mac_any_n512_s512/`, and sizes the overlap ranges at
2^24 programs per length (the AMD CPU runs the variant at 0.33 M programs/s). The reference and `u2_ops.run` got a
ring-size argument (`u2_ops.run(..., ncell=512)`, default unchanged); the reference self-check against `u2_ops.run`
on 1,500 random cases passes on both ring sizes. The GPU kernel uses 64 threads per block on 512 cells (four packed
128-byte tapes per thread, 32 KB of shared memory per block).

### Validation on the 512-cell ring

Same procedure as section 3: Python reference on the workstation (24 processes), CPU mode on adler40, GPU program on
the RTX 4090 and on a RTX 3060, compared per primitive in survivors, found programs with per-step cells and head
flags, and `found_by_step`. `results/u2_mac_any_n512_s512/validation_L11.json`, `ref_L*.json`.

| length | primitives | found programs (any budget 1..512, 512 cells) | reference | all identical |
|---|---|---|---|---|
| 1 to 3 | all six | none | 0.2 to 0.6 s | yes |
| 4 | all six | COPY 2 | 0.8 s | yes |
| 5 | all six | COPY 17 (16 on 256 cells) | 1.3 s | yes |
| 6 | all six | COPY 145 (121) | 3.0 s | yes |
| 7 | all six | COPY 831 (693), CARRY 4 | 9.8 s | yes |
| 8 | all six | COPY 4,000 (3,266), CARRY 13, ADD 1 | 35.2 s | yes |
| 9 | ADD, MUL_T, MAC_T, MAC_B, CARRY | ADD 1, CARRY 45 (40) | 133.0 s | yes |
| 10 | ADD, MUL_T, MAC_T, MAC_B, CARRY | ADD 7 (5), MUL_T 19 (11), CARRY 164 (124) | 649.8 s | yes |
| 11 | MAC_T, MAC_B | none | 694.9 s | yes |

The extra copiers and carries are programs that run off the 256-cell ring's far side: on 512 cells they keep going
and some of them land a copy at a budget the smaller ring never shows. The multiply-accumulate counts are zero on
both rings at every length.

### Calibration (13 symbols, both encodings, every device at once; `calibration_L13.json`)

| device | seconds | M programs/s | relative to 256 cells / 256 budgets |
|---|---:|---:|---|
| adler40-cpu | 107.3 | 0.63 | 0.48 x |
| adler40-gpu0 (RTX 4090) | 19.3 | 3.48 | 0.25 x |
| adler40-gpu1 (RTX 4080) | 30.6 | 2.19 | 0.24 x |
| falke64-cpu | 223.7 | 0.30 | 0.48 x |
| knecht24-cpu | 202.4 | 0.33 | 0.42 x |
| knecht24-gpu0, 1, 2 (RTX 3060) | 107.9, 106.2, 105.3 | 0.62, 0.63, 0.64 | 0.26 x |
| specht32-cpu | 333.8 | 0.20 | 0.48 x |

9.0 M programs/s in all (33.4 on the small ring); all nine outputs identical. Twice the budget and twice the mask
width cost the CPUs a factor of two; the GPUs lose another factor of two because a thread's four tapes now take 512
bytes of shared memory, which halves the resident warps.

### Results (launched 16:01:05, collected 16:17; `results/u2_mac_any_n512_s512/any_L13.json` to `any_L16.json`)

| 512 cells, every budget 1..512 | 13, MAC_T | 13, MAC_B | 14, MAC_T | 14, MAC_B | 15, MAC_T | 15, MAC_B | 16, MAC_T | 16, MAC_B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| programs | 67,108,864 | 67,108,864 | 268,435,456 | 268,435,456 | 1,073,741,824 | 1,073,741,824 | 4,294,967,296 | 4,294,967,296 |
| alive after inputs 1 to 3 | all | all | all | all | all | all | all | all |
| after 4 | 17,448,280 | 15,768,554 | 70,611,070 | 63,890,743 | 286,112,339 | 259,179,631 | 1,151,348,609 | 1,044,052,813 |
| after 5 | 6,162,467 | 6,194,611 | 25,047,935 | 25,206,521 | 102,862,325 | 103,520,498 | 414,329,982 | 417,062,676 |
| after 6 | 3,674,487 | 3,704,525 | 14,819,249 | 14,921,992 | 60,070,607 | 60,412,710 | 241,176,670 | 242,459,565 |
| after 9 | 38,426 | 37,896 | 146,368 | 147,183 | 568,717 | 575,490 | 2,192,266 | 2,228,716 |
| after 12 | 3,456 | 3,167 | 10,468 | 9,779 | 33,548 | 32,240 | 108,799 | 108,080 |
| after 13 | 872 | 687 | 2,523 | 2,066 | 9,066 | 7,634 | 27,438 | 25,105 |
| after 14 | 193 | 112 | 607 | 418 | 2,081 | 1,542 | 6,627 | 5,627 |
| after 15 | 87 | 37 | 255 | 121 | 890 | 428 | 2,566 | 1,462 |
| after 16 | 46 | 14 | 131 | 38 | 420 | 155 | 1,208 | 526 |
| after 17 | 11 | 10 | 41 | 33 | 172 | 109 | 574 | 390 |
| after 18 | 2 | 3 | 7 | 9 | 37 | 30 | 145 | 89 |
| after 19 | **0** | **0** | 4 | 3 | 30 | 10 | 120 | 19 |
| after 20 | 0 | 0 | 1 | **0** | 7 | 6 | 29 | 8 |
| after 21 | 0 | 0 | **0** | 0 | **0** | 6 | 7 | 8 |
| after 22 | 0 | 0 | 0 | 0 | 0 | **0** | 5 | 2 |
| after 23 | 0 | 0 | 0 | 0 | 0 | 0 | **0** | 1 |
| after 24 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| after 25 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0** |
| **multiply-accumulates at any budget 1..512 (alive after 48)** | **0** | **0** | **0** | **0** | **0** | **0** | **0** | **0** |

`found_by_step` is zero at every one of the 512 budgets. More programs survive the early inputs on the larger ring
(7 to 10 % more after input 4, 15 to 20 % more after input 12: programs whose head runs around the small ring and
spoils its own work are no longer spoiled), but the frontier is the same at every length and encoding as on 256
cells, down to the number of programs alive there: 18 (2 programs), 20 (1), 20 (7), 22 (5) triples for MAC_T and
18 (3), 19 (3), 21 (6), 24 (1) for MAC_B at 13 to 16 symbols, dying on the same triples. The frontier programs are
listed below; they are the same 28 programs as in section 7, with the same cells and the same budgets up to 256 (table below). Budgets 257 to 512 add nothing: no program holds a candidate at any of them that it does not
already hold at some budget below 257 for every input it survives. The twelve w = 0 triples and the twelve zero-target
triples are never reached.

Runs executed per encoding: 0.30 billion at 13 symbols, 1.19 at 14, 4.75 at 15, 19.0 at 16, each 512 instructions.

### Run record

| device | 13 | 14 | 15 | 16 symbols | overlaps 13 / 14 / 15 / 16 (2^24 each) | device total |
|---|---|---|---|---|---|---|
| adler40-cpu (24 thr) | 4,653,500 programs, 7 s | 18,613,998, 29 s | 74,455,993, 117 s | 297,823,972, 475 s | 26.6 / 26.7 / 28.1 / 24.5 s | 841.4 s |
| adler40-gpu0 (RTX 4090) | 25,871,529, 7 s | 103,486,115, 26 s | 413,944,459, 102 s | 1,655,777,835, 406 s | 4.0 / 3.7 / 5.0 / 2.9 s | 575.9 s |
| adler40-gpu1 (RTX 4080) | 16,317,664, 7 s | 65,270,654, 25 s | 261,082,616, 100 s | 1,044,330,465, 388 s | | 550.8 s |
| falke64-cpu (12 thr) | 2,232,099, 7 s | 8,928,395, 28 s | 35,713,581, 111 s | 142,854,324, 445 s | | 814.6 s |
| knecht24-cpu (24 thr) | 2,466,999, 8 s | 9,867,994, 30 s | 39,471,977, 120 s | 157,887,906, 483 s | 49.3 / 49.2 / 50.8 / 48.0 s | 1,040.9 s |
| knecht24-gpu0 (RTX 3060) | 4,627,623, 11 s | 18,510,491, 41 s | 74,041,965, 165 s | 296,167,861, 638 s | 23.8 / 21.6 / 28.2 / 17.7 s | 1,053.4 s |
| knecht24-gpu1 (RTX 3060) | 4,701,700, 10 s | 18,806,799, 40 s | 75,227,194, 165 s | 300,908,778, 644 s | | 966.2 s |
| knecht24-gpu2 (RTX 3060) | 4,741,885, 10 s | 18,967,540, 38 s | 75,870,162, 154 s | 303,480,648, 614 s | | 922.1 s |
| specht32-cpu (12 thr) | 1,495,865, 8 s | 5,983,470, 32 s | 23,933,877, 129 s | 95,735,507, 508 s | | 1,006.9 s |

Launched 16:01:05, last shard (knecht24-gpu0's 16-symbol overlap) done 16:16:58 workstation time: 15 min 53 s wall,
18.6 CPU-core-hours plus 1.13 GPU-hours, nice 10, nothing else of ours on the nodes. Checks at collection as in
section 6; the 2^24-program overlap ranges are identical on the Intel CPU, the AMD CPU, the RTX 4090 and a RTX 3060 at
all four lengths (96 survivor counts, 1,024 found_by_step counts and the empty found lists each).

### The frontier programs on the 512-cell ring

Recovered as in section 7 (`tools/u2_mac_any_exhibit.py --ncell 512 --steps 512`: GPU runs with the input list cut at
the frontier, per-step cells from the CPU mode, re-verification with the reference and `u2_ops.run` on the 512-cell
ring; `results/u2_mac_any_n512_s512/exhibit/`). **They are the same 28 programs as in section 7**, with the same result
cells and the same budgets up to 256; the larger ring and the longer run change only the ranges that continue past
the old end of the run, for 9 of the 28 programs:

| symbols | encoding | program | triples | budgets on 256 cells (1..256) | budgets on 512 cells (1..512) |
|---|---|---|---:|---|---|
| 13 | BitNet | `0113102020111` | 18 | 87-256 | 87-512 |
| 13 | BitNet | `0113331020211` | 18 | 89-256 | 89-512 |
| 13 | signed | `3110112110221` | 18 | 223 | 223,294,365,436,507 |
| 14 | BitNet | `00111100300111` | 19 | 172-256 | 172-512 |
| 14 | BitNet | `00111110030110` | 19 | 179-256 | 179-512 |
| 14 | BitNet | `00111110031001` | 19 | 179-256 | 179-512 |
| 16 | signed | `0112110113102000` | 22 | 116-256 | 116-512 |
| 16 | signed | `1210110113102000` | 22 | 116-256 | 116-512 |
| 16 | signed | `1211001113102000` | 22 | 116-256 | 116-512 |

Eight of these simply keep the result in place from the first budget to the end of the run (256 was the end of the
run, not of the program). The 13-symbol program `3110112110221`, whose 18 triples agree under the head at budget 223
only, agrees again every 71 instructions: at 294, 365, 436 and 507, a loop that passes the right cell under the head
once per period. The other 19 programs have the same ranges on both rings, so none of the 28 ever touched the far
side of the 256-cell ring within its range. The 24-triple program `2111012110003111` holds at budgets 115 and 116 on
both rings and fails on w = 0 at both.

### Files and commands of this section

- `results/u2_mac_any_n512_s512/`: `ref_L1.json` to `ref_L11.json`, `validation_L11.json`, `calibration_L13.json`,
  `plan_any_L13_L14_L15_L16.json`, `any_L13.json` to `any_L16.json`, `raw/`, `exhibit/`.

```
python tools/u2_mac_any.py build --ncell 512 --steps 512
python tools/u2_mac_any.py ref --ncell 512 --steps 512
python tools/u2_mac_any.py validate --ncell 512 --steps 512 --Lmax 11
python tools/u2_mac_any.py calibrate --ncell 512 --steps 512
python tools/u2_mac_any.py launch --ncell 512 --steps 512 --L 13 --L 14 --L 15 --L 16
python tools/u2_mac_any.py collect --ncell 512 --steps 512 --L 13 --L 14 --L 15 --L 16
python tools/u2_mac_any_exhibit.py --ncell 512 --steps 512
```
