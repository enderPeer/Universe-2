# Universe 2 is a 2-bit machine: what fits in two bits, and what the other widths decide

The genome is written in a 4-letter alphabet: one base is 2 bits. Universe 2 takes that as its design rule, one
2-bit symbol per instruction, and asks which machine makes those symbols into building blocks of bigger programs.
This note lays out the design space, what the predecessors already measured in it, and a first sweep of three
candidate four-instruction machines (`tools/u2_ref.py`).

## 1. Read the genome analogy correctly

| genome | machine term | consequence for the design |
|---|---|---|
| base (A, C, G, T): 2 bits | symbol | the alphabet has four letters; this is fixed |
| codon: 3 bases = 6 bits, 64 codes | instruction word | the unit of meaning need not be one symbol; 64 codes map to only 21 meanings (20 amino acids + stop), so **3 bits of every codon are redundancy**, which is what makes point mutations survivable |
| ribosome: reading frame, current position | processor: PC, head | the pointer lives in the machine, not in the genome; a genome never addresses itself by absolute position |
| protein: a chain whose function emerges from the sequence | the composed program | building blocks (exons, domains) are **relocatable**: the same block means the same thing anywhere, because nothing in it is an address |
| gene regulation (promoters, repressors) | control flow | control is structural (start/stop signals, templates), not a jump to a numbered line |

So "2 bits" does not force a 4-instruction machine. It forces a 4-letter alphabet. The choices left are the
instruction length in symbols (k), how the 2k bits split into opcode and operand, how wide a memory cell is (W),
where the pointers live, and how loops are expressed. Each of these has already been probed by one of the three
predecessors, which is why Universe 2 can start from measurements rather than taste.

## 2. The arithmetic: 2k bits = opcode bits + operand bits

| k (symbols per instruction) | bits | opcode / operand splits | who tried it | what happened |
|---|---|---|---|---|
| 1 | 2 | 4 operations, **0 operand bits** | Universe-7 (4 drawing moves); Universe-1 W = 2 (4 opcodes, LD/ST can only address the accumulator) | Universe-7: with the head as the only address, 100 % of 8.4 M short programs draw distinct pictures. Universe-1: with memory addressing but no operand bits, only 4 of 256 unary functions are reachable: **the operand field, not the opcode count, is the bottleneck** |
| 2 | 4 | 2 + 2 (crawler), 3 + 1 | Universe-1 L2 sweeps, 2^32 each | 2 + 2 is the only split that copies itself (141 copiers); every 3 + 1 ISA stops at 7 of 8 words copied |
| 2.5 | 5 | 2 + 3 | the 5-byte crawler, 2^40 | 21,818 copiers, 3 intact replicators; fecundity one because a 3-bit pointer survives 8 increments |
| 3 | 6 | 3 + 3, 4 + 2, or 64 codes with degeneracy (the codon) | not yet swept | the genome's choice; 64 codes leave room for synonyms |
| 4 | 8 | 4 + 4 | NANO (Dimension42), 2^40 and 2^48 | 16 operations, 16 addressable cells, rich self-modification, no multiplier at 5 bytes |

With k = 1 the operand is gone, so **the head is the operand**: an instruction can only act on the cell under the
head and move the head. Addressing becomes sequential (reach cell n in n moves), and the cost of that shows up as
program length, which an exhaustive sweep pays for anyway.

## 3. Where the pointers live decides composability

The crawler and NANO keep code and data in one memory. That is what makes their rare behaviours (98.7 % of NANO's
singletons self-modify) and what kills their replicators: a copy loop's pointers are its own instruction words and
wear out. A tape machine keeps the pointer (the head) in the processor. Two consequences:

- **No absolute addresses exist, so every program is position-independent.** Concatenation is composition. This is
  exactly the property the NANO catalog lacks (docs/04 section 2) and the reason "building blocks that execute
  bigger programs" is natural at 2 bits and impossible at NANO's 8.
- **Self-modification is gone**, unless the tape is also the program (a von Neumann 2-bit machine, where a cell
  holds one symbol: W = 2 and the head can write code). That variant is worth a sweep of its own, because it is the
  only way to get NANO-style self-modification and relocatable blocks at once.

## 4. Control flow with zero operand bits

A jump needs a target, and there are no bits for one. The options, with what we know about each:

| mechanism | example | cost in the machine | known results |
|---|---|---|---|
| none; the whole program repeats for a fixed number of steps | Universe-7 (127 steps), NANO's PC wrap, design M4 below | nothing | Universe-7: maximal picture variety; no halting, no conditionals, so no computation that depends on the data |
| skip the next instruction on a condition | Universe-1 champion `SWAP, ADD, NAND, SKZ`; design S4 below | nothing | the champion ISA had the most functions of 37 sets at W = 4; SKZ plus wrap gives conditional loops |
| structural brackets `[` `]` matched by the machine | Bohm's P'' (1964), Brainfuck's ancestor; design P4 below | the machine must find the matching bracket (a stack, or a scan) | **Turing complete with four instructions** on an unbounded tape, a 60-year-old theorem; it is the proof that two bits per instruction lose no computational power |
| fixed relative jump (back 2, to start) | - | nothing | not swept |
| templates (jump to the next occurrence of a pattern) | Tierra, Avida's heads, DNA promoters | pattern search in the machine | Avida's 26-instruction set had 914 replicators at length 8 (reproduced in Universe-1 exp11) |

The bracket machine is the genome-like one: a loop is delimited by start and stop symbols, nests, and can be cut
out and pasted elsewhere intact.

## 5. Cell width W

W is the tape's word width, independent of the 2-bit instruction width. Universe-1 swept W = 1, 2, 4, 8 and found
the function count grows steeply with W only when the operand field grows with it; here the operand field is zero,
so W only sets how many states a cell has and how long the `+` cycle is. W = 1 makes `+` a flip (its own inverse).
W = 2 lets a cell hold one instruction symbol, the precondition for the self-modifying variant in section 3.

## 6. First sweep: three candidate machines, every program of 1 to 8 symbols

`python tools/u2_ref.py --sweep 8 --W 1 --steps 64` and `--W 2`. Tape of 16 cells, head at 0, 64 steps.
87,380 programs per design and width. Score, as in Universe-7: distinct final tapes.

| design | instructions | W = 1: distinct tapes (of 87,380 programs) | W = 2 | notes |
|---|---|---:|---:|---|
| **P4** (P'') | `>` move right, `L` increment then move left, `[`, `]` | 223 (0.26 %) | 408 (0.47 %) | 92 % of 8-symbol programs halt before the budget; short programs mostly walk off the end |
| **S4** (skip) | `>`, `+`, `?` skip next if cell = 0, `<` | **1,906 (2.18 %)** | **5,688 (6.51 %)** | wraps; conditional loops via skip; highest variety at every length |
| **M4** (no control) | `>`, `+`, `<`, `-` | 455 (0.52 %) | 3,807 (4.36 %) | wraps; a straight line repeated; W = 2 already beats P4 |

(`results/u2_sweep/sweep_len8_W{1,2}_steps64.json` has the per-length rows: at length 8, S4 reaches 1,472 (W = 1)
and 4,058 (W = 2) distinct tapes of 65,536 programs; P4 223 and 408; M4 138 and 1,324.)

Reading: at 8 symbols (16 bits, the size of a NANO 2-byte program) variety is low for every design, as it was for
Universe-7 before its table search, because 16 bits are only 65,536 programs against 2^16 (W = 1) or 2^32 (W = 2)
possible tapes. The Turing-complete P4 scores worst on this measure because most of its short programs halt early:
expressiveness in the limit and variety at short lengths are different properties, and the sweep rewards the second.
S4 (skip + wrap) wins both widths. None of this is a decision yet: the next step is Universe-7's method, scoring
*every* four-instruction table over a candidate operation set (moves, increments, flips, skip, brackets, fixed jumps,
halt) on all programs up to 11 or 12 symbols, on the cluster CPUs, and then the chosen law to 2^40 on the GPUs.

## 6b. How big can the memory be? (asked: 93.75 to 100 million words)

Any size in that range is trivial for the hardware: 100,000,000 cells of 2 bits are 25 MB (12.5 MB at 1 bit,
50 MB at 4 bits); the workstation has 64 GB and every cluster GPU 12 to 32 GB. What matters is *what the memory is
for*:

- **For the sweep** (every program up to a length, run for a step budget), the tape only has to be as large as the
  head can reach: at most one cell per step, so 64 steps need 64 cells, 256 steps 256 cells. A larger tape changes
  nothing in the results and costs GPU memory per program instance (a GPU runs millions of programs at once, so the
  per-program tape is what limits parallelism; 25 MB per program would allow about 1,000 at a time on a 24 GB card
  instead of millions).
- **For a world** (a soup where many programs live in one shared memory and compose by being next to each other,
  the genome-like use), one tape of 10^8 cells shared by all heads is exactly right: 25 MB holds it on any device,
  a GPU can run millions of heads over it in parallel, and with W = 2 every cell is also an instruction, so code and
  data are one thing as in the crawler and NANO. For comparison, the human genome is 3.1 x 10^9 bases, so 10^8
  cells is about a thirtieth of it, or one large chromosome.

So: yes. The sweep gets a small tape sized to the step budget; the world gets the 10^8-cell tape. The two numbers
in the question differ by 1/16 (93.75 = 100 x 15/16); if that was meant as "2^k-ish", 2^26 = 67.1 M and
2^27 = 134.2 M cells are the neighbouring powers of two, and 100 M is fine as is.

## 7. What a 2-bit machine cannot do in one symbol, and why that is fine

No load-immediate (there are no value bits: constants are built by repeated `+`), no random access (the head
walks), no two-operand operations (one cell at a time). Every program is longer than its NANO equivalent. But the
objects we sweep are *all programs up to a length*, so length is the budget, and every symbol of the budget is a
real choice with no wasted encodings: NANO's `0n` wastes 15 of 256 byte values on NOP, AArch64 wastes 64.45 % of
its 4-byte words on undefined encodings (`results/isa4_census`), the codon wastes 43 of 64 codes on synonyms.
A 2-bit machine with four meaningful symbols wastes nothing, which is why Universe-7 could reach 100 % distinct
outputs. The price is that it has no redundancy either, so a mutation always changes the program; the genome's
synonym layer (k = 3 with degeneracy) is the natural second universe to compare against.

## 8. Decisions to take, in order

1. Fix the score: distinct final tapes (Universe-7's), or distinct functions on probe inputs (NANO's and
   Universe-1's, the tape pre-loaded with a and b), or both.
2. Table search over 4-instruction sets (CPU, hours), including the self-modifying W = 2 variant where the tape is
   the program.
3. Choose the law, write the C/CUDA/Vulkan sweeper with the Dimension42 validation discipline (reference agreement
   on samples, cross-device identical hashes, per-chunk accounting), sweep to 2^40 (20 symbols).
4. Then the composition question for real: which blocks, concatenated, compute more than their parts.
