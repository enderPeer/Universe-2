# Where this comes from: the three machines, their instruction sets, and the sweeps

Universe 2 continues three repositories by the same author. This note states, from their result files as of
2026-10-09, which machine each one swept, which instructions it had, and what the sweeps found. Numbers are
copied from the committed results; nothing here was recomputed except the NANO checks in `tools/`.

| | Dimension42 / NANO | Universe-1 | Universe-7 |
|---|---|---|---|
| repository | [enderPeer/Dimension42](https://github.com/enderPeer/Dimension42) (main, 76ed06a) | [enderPeer/Universe-1](https://github.com/enderPeer/Universe-1/tree/claude/charming-rubin-pi8nzs) (**branch claude/charming-rubin-pi8nzs**, 83b0539; `main` holds only the cluster inventory) | [enderPeer/universe7](https://github.com/enderPeer/universe7) (main, 7991565) |
| local clone | `C:\Users\end\Desktop\Dimension42` | `C:\Users\end\Desktop\u1` | `C:\Users\end\Desktop\universe7` |
| word | 8-bit byte | 4-bit word (W = 1, 2, 4, 5, 8 swept) | 1-bit pixel |
| memory | 16 cells, code and data shared, cells 8..15 are an 8x8 display | 4 words (layout L3) or 16 to 32 words shared with code (layout L2) | a 64x64 screen and a head position |
| instruction | 1 byte: opcode nibble + operand nibble, 16 fixed operations | 4 bits (5 for the 5-byte crawler): 2 or 3 opcode bits + operand bits; 37 instruction sets swept | 2 bits, 4 fixed drawing moves |
| largest exhaustive sweep | **all 2^48 six-byte programs** (class only) and all 2^40 five-byte programs (class, probes, adders, copiers) | 2^32 per ISA (37 ISAs), 2^32 x 256 steps (2 ISAs), **2^40 for the 5-byte crawler** | all programs of 1 to 22 bits (8.4 million) |

All local clones were compared with GitHub on 2026-10-09: Dimension42 and universe7 are identical to origin/main;
the Universe-1 clone is identical to its origin branch.

## 1. NANO: the instruction set behind the 5-byte and 6-byte maps

NANO is the Dimension42 CPU. One byte is one instruction: the high nibble is the operation, the low nibble `n` is
the argument. `A` is the 8-bit accumulator, `Mn` is memory cell n (0..15). The program of L bytes sits in M0..M(L-1);
the PC wraps at L, `JMP`/`JZ` go to `n mod L`. Every program runs **64 instructions** from the zero state
(A = 0, PC = 0, all non-code cells 0). Reference: [`tools/nano_ref.py`](../tools/nano_ref.py), checked against
Dimension42's `explorer/nano_ref.py` on 3,000 random programs (classes) and 10,000 input cases (first output): identical.

| opcode | instruction | effect |
|---|---|---|
| `0n` | NOP | nothing (in the NANO-256 variant this byte became BANK n) |
| `1n` | LDI n | A = n |
| `2n` | ADDI n | A = A + n (mod 256) |
| `3n` | SUBI n | A = A - n (mod 256) |
| `4n` | LD Mn | A = M[n] |
| `5n` | ST Mn | M[n] = A |
| `6n` | ADD Mn | A = A + M[n] |
| `7n` | JMP n | PC = n mod L |
| `8n` | JZ n | if A = 0: PC = n mod L |
| `9n` | INC Mn | M[n] += 1 |
| `An` | DEC Mn | M[n] -= 1 |
| `Bn` | ROL n | rotate A left by n mod 8 bits |
| `Cn` | XOR Mn | A = A xor M[n] |
| `Dn` | OUT Mn | output M[n] (the **first OUT is "the answer"** in every function result) |
| `En` | NOT Mn | M[n] = not M[n] |
| `Fn` | HLT | stop |

Because code and data share the 16 cells, `ST`, `INC`, `DEC` and `NOT` on cells 0..L-1 rewrite the program. That is
where almost every interesting 5-byte program comes from: 98.67 % of the 84,933 singleton behaviours self-modify.

Each program gets one of six classes by priority: DRAW (touched a display cell M8..MF) > OUTPUT (executed OUT) >
SELF-MOD (code differs at the end) > ACTIVE (A or a data cell ever nonzero) > HALT > IDLE.

### The 5-byte sweep (2^40 = 1,099,511,627,776 programs)

| experiment | what was computed | result |
|---|---|---|
| class map | every program, 64 steps, zero input | 1,024 chunks, 159.65 s on 9 GPUs; repeat run identical |
| behaviour catalog | 16 input pairs (a in M5, b in M6), hash of the first outputs | 219,846 signatures, 84,933 with exactly one program |
| 2 + 2 = 4 | every program with inputs 2, 2; candidates proven on all 65,536 pairs | 74,527 real 5-byte adders; shortest adder `45 66 50 D0` (4 bytes); **no program up to 5 bytes multiplies** |
| three-input adders | 10 cell layouts, proven on all 16,777,216 triples | 423 to 2,504 adders per layout; `67 A0 5C E2` adds three cells in 4 bytes by rewriting itself |
| alife toolkit | zero input, what a program does to memory | 316 replicators (exact copy of its 5 bytes elsewhere), none with its original intact afterwards; 52,261 mutating replicators; 3,666 two-position repairers |
| copiers 1..5 bytes | GPU search + Python re-check | lists in `explorer/results/copiers/` |

Example from the catalog, verified here again on all 65,536 pairs: `17 C5 91 50 71` computes a xor b xor 7.
`LDI 7 · XOR M5 · INC M1 · ST M0 · JMP 1`: `INC M1` walks the XOR operand from M5 to M6 and onward until the byte
`CF` increments into `D0`, which is `OUT M0`.

### The 6-byte sweep (2^48 = 281,474,976,710,656 programs)

What exists exhaustively is the **class map only**: 65,536 zstd chunks, 794 GB, computed 2026-10-06 19:13 to
2026-10-07 10:47 on seven to eight GPUs (about 54 GPU-hours), stored on adler40 (34,003 chunks), falke64 (13,860),
specht32 (10,942) and knecht24 (6,731). Classes: DRAW 40.32 %, ACTIVE 16.45 %, OUTPUT 13.82 %, SELF-MOD 12.27 %,
HALT 10.01 %, IDLE 7.11 % (`explorer/results/L6_phenotypes/full-map-census.json`).

The "six-byte formulas" are **sampled, not exhaustive**: a uniform sample of 8,388,608 programs and a 20,000,000
rarity sample were probed on 16 input pairs, and five singleton programs were dissected and proven on all 65,536
pairs (`formulas-first-two.json`, `formulas-last-three.json`). Two of them:

| program | verified formula | mechanism |
|---|---|---|
| `EC 6D CB 22 A2 A0` | x = 8; for v in [b, a, 160, 162, 34, 194]: x = ((x xor v) + 2) mod 256 | `DEC M2` walks the XOR operand down through the code; `NOT M3` turns `ADDI 2` into `OUT D` at instruction 64 |
| `A7 A1 CD BA E1 66` | x = 0; for v in [a, a, a, a-1, a-2, 101, 225, 224, 224]: x = (rol(x, 2) + v) mod 256 | nine rounds of rotate-and-add over its own bytes |

An exhaustive 6-byte input/output catalog does not exist. It would cost about 256 times the 5-byte catalog, which
Dimension42 estimates as days on the cluster, and it would answer whether multiplication appears at 6 bytes.

## 2. Universe-1: the crawler and the 5-byte crawler sweep

Universe-1 sweeps *instruction sets* as well as programs: the machine has W-bit words, 8 instructions per program,
and the ISA is a parameter (37 were swept exhaustively at 2^32 programs each). A program's result is the
accumulator after 256 steps for every input x (unary, 16 inputs) or x, y (binary, 256 inputs); the vector of
results is its function. Champion ISA `SWAP, ADD, NAND, SKZ`: 1,829,051 unary and 24,684,247 binary functions;
over every step 1..256 it reaches 142,263,973 functions and 267,132,154 over 512 steps.

The **crawler** is the instruction set `LDIND, STIND, INCM, JNZ` under the self-modifying layout L2 (fetch reads
memory, code is data):

| instruction | effect |
|---|---|
| `LDIND op` | A = M[M[op]] (load through a pointer word) |
| `STIND op` | M[M[op]] = A (store through a pointer word) |
| `INCM op` | M[op] += 1 |
| `JNZ op` | if A != 0: PC = op |

With 2 opcode bits the operand gets the remaining bits of the word. The crawler was the only one of seven L2
instruction sets whose programs copy themselves:

| configuration | programs | copiers (all 8 words copied) | persisting | original intact at the copy |
|---|---:|---:|---:|---:|
| 4-bit words, 16 words of memory (exp06b) | 2^32 | 141 | 16 | 0 |
| 4-bit words, 32 words | 2^32 | 141 | 16 | 0 |
| **5-bit words, 32 words = 8 x 5 bits = 5 bytes (exp06c)** | **2^40** | 21,818 | 7,482 | **3** (`0xa2c841f269`, `0xa9344f2e8d`, `0xd9b17359e8`) |

This 2^40 run is "the 5-byte sweep for the crawler". It found the first replicators with intact code in any of the
three projects. The follow-up (design C, a Tierra-style soup on the crawler, 16 soups over 60 million ticks) showed
that every copier has fecundity one by construction: its pointers are its own instruction words, and a 3-bit operand
survives only eight increments. The next machine change proposed there is W = 6 (4-bit operands).

Universe-1 also ran exp11, an Avida interpreter on the cluster: all 26^8 length-8 genomes (916 viable, the 914
published plus two), all 26^7 (none), and 26^9 was running on falke64's two R9700 cards on 2026-10-09.

## 3. Universe-7: the law chosen by search

Two-bit instructions `F4 F8 S3 C3` (flip / set / clear the pixel under the head, then move the head by 4, 8, 3
or 3), 127 steps, chosen out of 125,580 three-operation and 3,876 four-operation tables so that almost every short
program draws a distinct picture: 100.00 % distinct final screens for all 8,388,606 programs of 1 to 22 bits on
the 64x64 screen (99.17 % on 8x8). A 352-byte handwritten x86-64 ELF runs it.

## 4. What the three projects say together

- Self-modification is the source of rare behaviour (NANO singletons 98.7 % self-modifying; Universe-1 L2 multiplies
  function counts 43 to 184 times).
- Replication appears at 4 to 5 bytes in both machines, intact replication only at 5 bytes and only in the crawler,
  and never sustains itself because pointers are code.
- No program up to 5 NANO bytes multiplies; the 6-byte question is open.
- Composition closes the gap: chains of Universe-1 champion programs reach all 16^16 unary functions.
  [`tools/nano_compose.py`](../tools/nano_compose.py) brings the same operation to NANO (see
  [04_big_program.md](04_big_program.md)).
