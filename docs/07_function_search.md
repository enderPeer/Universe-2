# Which 2-bit machine computes? The function search, and the comparison with Universe-1's champion and JC

Date: 2026-10-09. The table search of docs/06 scored machines by *variety of drawings*. This note scores them the
way Universe-1 scores its instruction sets: by the number of distinct **functions of an input** that all programs
up to a length compute. Same catalog, same 58,905 four-instruction machines, same cluster, same validation.

## 1. The measure

Universe-1 puts a 4-bit x in the accumulator, runs every 8-instruction program (2^32) for 256 steps and reads the
accumulator; the vector of 16 outputs is the program's function; the ISA's score is the number of distinct
functions. Champion `SWAP, ADD, NAND, SKZ`: **1,829,051**; `SWAP, ADD, NAND, JC`: **8,533,818**; the 8-primitive
`SWAP, LDI, NAND, ADD, ROL, SKNZ, INC, HALT`: 1,463,824; the 16-primitive `LD ... HALT`: 977,928 (`results/exp03_summary.md`,
`exp05_summary.md`). The universe is the 16^16 = 2^64 unary 4-bit functions.

The 2-bit machine has no accumulator, so x goes on the tape: with 1-bit cells, bit i of x in cell i (cells 0..3);
with 2-bit cells, cells 0 and 1. The head starts at cell 0. After 64 instructions (HALT stops early) the same cells
are read back as the output nibble. 16 inputs per program, truth table packed in 64 bits, distinct tables counted
per machine. Same universe, 2^64. Also recorded per machine: how many of its functions are bijections, how many
constants, and which of 39 named functions it reaches (identity, x+1, x-1, ~x, -x, shifts, rotates, bit reverse,
nibble swap, x==0, x!=0, parity, popcount, x*x, 3x, 5x, x/2, x>=8, x<8, 2x+1, x+2, x+3, x+4, x+8, x^1, x^8, x&1, x&8,
x|1, three constants, x%3, x%5, min(x,7), 15-x, |x-8|).

Validation: the C implementation (`u2_tables fn`, `u2_tables_v2 fnall`) agrees with the Python reference
(`u2_ops.truth_table`) on 300 random (machine, program, W) truth tables: 0 mismatches. The 200-machine overlap
between adler40 (Intel) and knecht24 (AMD) is identical line for line for both widths.

## 2. Screening: all 58,905 machines, all programs of 1 to 8 symbols (87,380 programs, 16 inputs each)

Run 2026-10-09 14:52 to 15:08 on the four nodes' CPUs at nice 10 (adler40 377 s per width for 25,704 machines,
falke64 326 s, knecht24 251 s, specht32 415 s); 2 x 58,905 x 87,380 x 16 = 165 billion program runs.

| | W = 1 (bit per cell) | W = 2 |
|---|---:|---:|
| best machine | `L1, INC_L1, FLIP_R3, JNZB`: **19,439** functions | `INC_R1, INC_L1, FLIP_R2, JNZB`: **16,104** |
| its functions after 1..8 symbols | 3, 7, 26, 89, 289, 1,152, 4,809, 19,439 | 4, 7, 29, 86, 304, 1,102, 4,233, 16,104 |
| machines with >= 10,000 functions | 207 | 85 |
| machines with >= 1,000 | 6,299 | 3,842 |
| median machine | 24 | 16 |
| machines with <= 16 functions (constants and the identity at most) | 27,141 (46 %) | 32,290 (55 %) |
| machines with exactly 1 function | 2,276 | 2,276 |

**Every one of the 500 best machines has exactly one control operation**, and it is a two-instruction loop or a
skip: JNZB in 218 of the top 500 (W = 1), SKNZ 166, SKZ 126; the brackets appear twice, restart jumps twice. The
other three operations are pens and at most one move or write. Composition of the top 100 at W = 1: 66 machines
are {one control, one move, two pens}, 32 are {one control, three pens}.

| machine class | best W = 1 | rank | best W = 2 |
|---|---|---:|---|
| with a skip (SKZ / SKNZ) | `L1, INC_L1, FLIP_R3, SKNZ` 14,283 | 23 | `INC, INC_R1, FLIP_L1, SKZ` 14,333 (rank 6) |
| with the accumulator | `INC_L1, FLIP_R2, SKNZ, SWAP` 12,379 | 74 | 11,222 |
| with one bracket | `L1, INC_L1, FLIP_R3, CLOSE` 8,227 | 413 | 6,499 |
| with both brackets | `INC_L1, FLIP_R2, OPEN, CLOSE` 5,207 | 1,213 | 4,022 |
| P'' exactly (`R1, INC_L1, OPEN, CLOSE`) | 3,192 | 2,628 | 1,812 |
| skip design S4 (`R1, L1, INC, SKZ`) | 3,288 | 2,522 | 4,407 |
| with HALT | 3,276 | 2,532 | 3,175 |
| no control at all (best: accumulator machines) | `INC_L1, FLIP_R2, SWAP, ADDA` 2,438 | 3,617 | 1,525 |
| Brainfuck without loops (`R1, L1, INC, DEC`) | 16 | 31,765 | 16 |
| the perfect drawing machine of docs/06 (`R3, INC_R1, FLIP_R2, FLIP_R3`) | **10** | 38,791 | 5 |

Named functions: at W = 1 some machine reaches 32 of the 39 (never reached at 8 symbols: bit reverse, parity,
popcount, x*x, 5x, x%3, x%5, min(x,7), |x-8|); the machine with the most names is `R1, L3, FLIP_R2, SWAP` with 21
(all shifts and rotates, the nibble swap, x/2, x>=8, x<8, 2x+1, the constants). At W = 2 24 of 39 are reached and
5x appears, which W = 1 never produces. Multiplication by 3 appears at W = 1, multiplication by 5 at W = 2, squaring
and the popcount family nowhere.

### Reading

- **The variety winners and the function winners are disjoint.** The 59 machines that drew every program as a
  different tape compute 5 to 10 functions. The function winners draw few distinct tapes (their JNZB loop makes
  most programs collapse). Universe 2 therefore has two laws, not one: a drawing law for the world and a computing
  law for functions, and no four-instruction set is both.
- **One conditional, and a cheap one.** Turing completeness (P'') is not what makes a machine express many
  functions at short lengths; a two-instruction conditional loop is. This is Universe-1's finding ("the operand
  field, not the opcode count, is the bottleneck"; SKZ in the champion) in a machine that has no operands at all.
- **Growth is about 4x per symbol at the top** (4,809 to 19,439 from 7 to 8 symbols: 22 % of the 65,536 new programs
  are new functions). Universe-1's champion has 1,829,051 functions from 2^32 programs, 0.043 % of its programs. If
  the 2-bit growth continued at 22 % to 16 symbols it would exceed the champion; it will not continue, because the
  universe of reachable functions saturates. The run in section 4 measures where.

## 3. The top 644 machines rerun at 10 symbols (1,398,100 programs each)

*(collected from the cluster when the run finishes; results/u2_tables/report_fn_L10_s64_sub644.json)*

## 4. The best machine at 16 symbols = 32 bits: all 4^16 = 4,294,967,296 programs, the Universe-1 program size

Machine 11054, `L1, INC_L1, FLIP_R3, JNZB`, 1-bit cells, 64 steps, every program of exactly 16 symbols. Run
2026-10-09 15:11 to 15:35 on the four nodes' CPUs (`u2_tables_v2 fnall`; shards of 1.87 G, 1.17 G, 0.62 G and
0.62 G programs by node speed; falke64 315 s at 12 threads, knecht24 300 s at 24, specht32 486 s at 12, adler40
longer because it was also running two other 24-thread jobs). Each shard dumps its distinct truth tables; the
union is exact. Files: `results/u2_fnall/table11054_W1_L16.json` and the complete list of its functions,
`table11054_W1_L16.functions.txt` (one 64-bit table per line).

| | 2-bit machine 11054 | Universe-1 champion `SWAP, ADD, NAND, SKZ` | Universe-1 `SWAP, ADD, NAND, JC` |
|---|---:|---:|---:|
| programs | 4^16 = 2^32, 16 symbols of 2 bits | 2^32, 8 instructions of 4 bits | 2^32 |
| steps | 64 | 256 | 256 |
| state | a tape of 1-bit cells and a head, no register | 4-bit accumulator, 4 words, flags | same, with the carry flag |
| **distinct unary 4-bit functions** | **2,930,290** | 1,829,051 | 8,533,818 |
| as a share of all 2^32 programs | 0.068 % | 0.043 % | 0.199 % |
| bijections (permutations of 0..15) | 11,337 | 15,378 | |
| constants | 3 (of 16) | 16 | |
| functions depending on all four input bits | 2,927,492 (99.9 %) | 1.63 M (89 %) | |
| named functions reached (of 39 tested here) | 15: id, -x, 15-x, x+1, x+2, x+4, x/2, x==0, parity, x&8, x^1, x^8, x\|1, const 0, const 15 | all 16 constants, all 256 affine maps, 1,891 named | |
| image size (distinct outputs) | mode at 10 outputs (625,448 functions); 894 with 2 outputs, 3 constants | peak at 4 to 8 | |

**Reading.** At the same program size (32 bits) the best four-instruction 2-bit tape machine computes **1.6 times
as many distinct functions as the Universe-1 champion**, with a quarter of the step budget and no register at all,
and a third of JC's count. The 2-bit encoding is not a handicap for expressiveness; the operand field Universe-1
found to be the bottleneck is replaced by head motion, and sixteen operand-free symbols buy more than eight
4-bit instructions whose operands mostly address the same four words.

The shape of what it computes is different. Almost everything it produces depends on all four input bits (99.9 %
against the champion's 89 %) and the constants are nearly absent (3 of 16): a tape machine that has to walk to
each bit cannot easily ignore bits or erase them all. It has fewer bijections (11,337 against 15,378) and far fewer
named arithmetic functions: no x-1, no ~x, no shifts or rotates at 16 symbols, while the champion has every affine
map. So the champion's library is arithmetic-shaped and this one is Boolean-shaped. For the "big program from
blocks" question that is fine, since composition supplies the rest (docs/04), but a translator of named functions
would start from the champion, not from here.

Caveats: 64 steps here against 256 there (a longer budget can only add functions for programs that have not
settled, and most 16-symbol loops under JNZB have); different input placement (tape cells against accumulator);
the count is exact (merged 64-bit tables, no hashing of tables, every shard's line count checked against its
summary).

## 5. The biological machine: EAT, MATE, SLEEP, WASTE

A four-instruction machine proposed on 2026-10-09 from the four things every animal does. The accumulator A is the
stomach. The specification left MATE ambiguous ("increments the cell and writes the accumulator state"), so two
readings were swept. Implemented in `u2_ops.py` (operations 36 to 40) and `u2_tables.c` (binary `u2_tables_bio`,
`custom` mode); C and Python agree on 720 random cases (3 symbol orderings, W = 1 and 2, step budgets 64 and 200).

| symbol | name | mechanics |
|---|---|---|
| 00 | EAT | A = cell; cell = 0 (the cell is eaten into the stomach) |
| 01 | MATE | reading A: cell = cell + A + 1; reading B: cell = A + 1; then head += 1 (offspring written to the right) |
| 10 | SLEEP | if A == 0: skip the next instruction (no energy, nothing happens) |
| 11 | WASTE | cell = 11 (all ones); A = 0; head -= 1 (excrete, step back) |

Scores, same conditions as all other machines (256-cell ring, 64 steps, PC wraps), `results/u2_bio/`:

| measure | bio machine | rank among 58,905 | median machine | best machine |
|---|---:|---:|---:|---:|
| variety, W = 1, programs to 10 symbols | 5,446 tapes | 24,508 | 3,775 | 1,396,636 (perfect) |
| variety, W = 2 (reading A) | 10,809 tapes | 23,408 | 5,691 | 1,396,636 |
| variety, W = 2 (reading B) | 7,740 | | | |
| functions, W = 1, programs to 8 symbols | 1,346 functions | 5,496 | 24 | 19,439 |
| functions, W = 2 (reading A) | 586 | 5,187 | 16 | 16,104 |
| functions, W = 2 (reading B) | 385 | | | |
| functions, W = 1, to 10 symbols | 6,865 | | | |
| functions, W = 2, to 12 symbols | 11,174 (bijections 62, constants 16) | | | |

Growth per added symbol at the end of the range: 2.25x for the biological machine, 4.04x for the best function
machine, 4x being the ceiling (every new program a new function). Named functions it reaches at W = 1: identity,
x-1, ~x, 15-x, x^1, x&1, x&8, x|1 and the constants; at W = 2 only the identity and constants (its inputs occupy
two cells, and MATE's step right plus WASTE's step left mostly shuttle between them).

Reading: as a drawing law it is average (rank 23,000 to 24,500 of 58,905, below the top quartile); as a computing
law it is in the top 10 % (rank about 5,200 to 5,500), because SLEEP is a conditional skip, the one ingredient
every top machine has, but its condition is on the stomach rather than on the cell, and EAT and WASTE both destroy
information (EAT empties the cell, WASTE overwrites it with 11 and empties A). Machines that keep what they read
compute more. Its biological reading, however, is exactly what the best function machines lack: a resource (A)
that is consumed, so a soup of these organisms has an energy budget by construction. The 2^32 function map at
16 symbols (W = 2, reading A) is recorded in section 6 when it completes.
