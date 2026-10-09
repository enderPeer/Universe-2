# Universe 2

**What do all the short programs of a machine do, and what can be built from them?**

Universe 2 is the fourth repository in a line of exhaustive program sweeps, and it is a **2-bit machine**: one
instruction is one symbol of a 4-letter alphabet, as in the genome, and programs are built from relocatable blocks.
[docs/05_two_bit_isa.md](docs/05_two_bit_isa.md) lays out what fits in two bits, what the predecessors already
measured about opcode width, operand width, cell width and control flow, and the first sweep of three candidate
four-instruction machines. Two side questions came first and keep their tools here:

1. **Real instruction sets.** The NANO maps covered every 5-byte (2^40) and 6-byte (2^48) program of an 8-bit toy
   CPU. Can the same census be done for all 2^32 four-byte words of **x86-64** and **AArch64**, the instruction sets
   of every PC, phone and Mac? Yes: decoding all 2^32 takes minutes on one workstation, executing every decodable
   word from a fixed state takes hours. [docs/02_real_isa_sweep.md](docs/02_real_isa_sweep.md).
2. **Big programs from mapped parts.** The sweeps left catalogs of small programs with proven behaviour. How do
   they compose into something larger than any one of them? Host-side chains work today and are verified exactly;
   in-machine linking needs a new machine. [docs/04_big_program.md](docs/04_big_program.md).

## The three predecessors

| project | machine | largest exhaustive sweep | headline |
|---|---|---|---|
| [Dimension42](https://github.com/enderPeer/Dimension42) | NANO: 16 bytes of shared code/data memory, 16 one-byte instructions, 64 steps | **all 2^48 six-byte programs** (class map, 794 GB on the cluster); all 2^40 five-byte programs with input probes | 74,527 proven 5-byte adders, no multiplier up to 5 bytes, 316 replicators none intact, 98.7 % of the rare behaviours self-modify |
| [Universe-1](https://github.com/enderPeer/Universe-1/tree/claude/charming-rubin-pi8nzs) (work is on branch `claude/charming-rubin-pi8nzs`) | 4-bit words, 8 instructions per program, the instruction set is a parameter | 37 ISAs at 2^32; every step of every program for two ISAs; **the 5-byte crawler at 2^40** | 1.8 M to 876 M distinct functions per ISA, chains reach all 16^16 unary functions, the first 3 intact self-replicators (crawler ISA `LDIND, STIND, INCM, JNZ`), Tierra-style soups |
| [universe7](https://github.com/enderPeer/universe7) | 2-bit drawing moves on a 64x64 screen, 127 steps | all 8.4 M programs of 1 to 22 bits | the law itself was chosen by exhaustive search so that 100 % of short programs draw distinct pictures |

[docs/01_lineage.md](docs/01_lineage.md) gives the instruction tables, the exact sweep definitions and the numbers
with their sources. All three local clones were checked against GitHub on 2026-10-09 and are identical to their
origin branches.

## What is here

| path | what |
|---|---|
| [`tools/nano_ref.py`](tools/nano_ref.py) | NANO reference interpreter, identical to Dimension42's on 3,000 random programs and 10,000 input cases |
| [`tools/nano_compose.py`](tools/nano_compose.py) | chains of mapped NANO programs: exact 256x256 tables, composition, formula verification, depth-limited chain search over a basis |
| [`tools/u2_ref.py`](tools/u2_ref.py) | the three candidate 2-bit machines (P'' brackets, skip, no-control) and the exhaustive short-program sweep |
| `results/u2_sweep/` | distinct final tapes per design, length and cell width |
| [`tools/isa4_census.py`](tools/isa4_census.py) | decode census of 4-byte words (Capstone), sampled or exhaustive, x86-64 and AArch64 |
| [`tools/isa4_exec_probe.py`](tools/isa4_exec_probe.py) | execution census (Unicorn): NANO-style classes and state fingerprints for every decodable word |
| `results/isa4_census/`, `results/isa4_exec/` | JSON results with counts, percentages, throughput and host |
| [`sim/dna_ref.py`](sim/dna_ref.py), [`gpu/dna_world.cu`](gpu/dna_world.cu) | **DNA-1**: the DNA-like machine of [docs/09](docs/09_dna1_spec.md) (4 symbols, complement xor 3, two strands, 1,024-cell slots, the standard genetic code's 64 codons onto 20 operations + STOP, processors with machine-held copy pointers, deterministic ticks). Python reference and CUDA engine agree bit for bit on 17 test worlds; the engine runs 2,999,999,488 cells at 405 ticks/s in 2.6 GB on one RTX 4090. First worlds and reading in [docs/10](docs/10_dna1_results.md). |
| [`gpu/dna_sweep.cu`](gpu/dna_sweep.cu), [`tools/dna_sweep_verify.py`](tools/dna_sweep_verify.py), [`tools/dna_trace.py`](tools/dna_trace.py) | the DNA-1 **copier sweep**: thousands of single-gene 8 x 256-cell worlds per GPU launch (one thread per world), reduced on the GPU, verified tick by tick against the reference on 5,179 genes; exhaustive over every gene of 2 to 6 codons under both codon tables ([docs/11](docs/11_dna1_copier_sweep.md)); the tracer prints a gene's world tick by tick |
| [`docs/08_dna_machine.md`](docs/08_dna_machine.md) | how to build the machine like DNA: complement as repair, pointers in the polymerase, codon layer with degeneracy, regulation by templates, the world; six experiments in order |
| [`docs/03_resources.md`](docs/03_resources.md) | the workstation and the four cluster nodes as found on 2026-10-09, and how to bound CPU, memory and power on each |

## First results (2026-10-09)

**The 2-bit table search** ([docs/06](docs/06_table_search.md)): all 36 operand-free operations, all C(36,4) =
58,905 four-instruction machines, each run on all 1,398,100 programs of 1 to 10 symbols for 1-bit and 2-bit cells
(164.7 billion program runs, 11 CPU-core-hours on the four cluster nodes, Intel/AMD overlap identical). The ceiling
is 1,396,636 distinct programs (1,464 words are repetitions of shorter ones); **12 machines at W = 1 and 47 at
W = 2 reach it exactly**: every distinct program draws a different tape. All of them are pens (write and move in one
step) with flips of two different strides, Universe-7's recipe rediscovered by exhaustive search. No machine with a
conditional, an accumulator or HALT comes close (best 19 to 29 %); Bohm's Turing-complete P'' ranks 24,454th.
Variety from a blank tape rewards drawing laws; the function search ([docs/07](docs/07_function_search.md)) scores
the same machines by distinct functions of a 4-bit input, Universe-1's measure. Winner: `L1, INC_L1, FLIP_R3, JNZB`
(a two-instruction conditional loop plus pens). **Over all 4^16 = 2^32 sixteen-symbol programs it computes
2,930,290 distinct unary functions, 1.6 times the Universe-1 champion's 1,829,051 at the same program size**, with
64 instead of 256 steps and no register; JC (8,533,818) stays ahead. Every top machine has exactly one conditional;
the perfect drawing machines compute 5 to 10 functions; Böhm's P'' ranks 2,628th. A proposed biological machine
(EAT, MATE, SLEEP, WASTE) ranks in the top 10 % for functions but saturates: 16,855 functions over all 5.7 billion programs of up to 16 symbols.

**The multiply-accumulate search** ([docs/13](docs/13_mac_search.md)): can the best W = 2 machine `INC_R1, INC_L1,
FLIP_R2, JNZB` compute acc + w * x mod 4 for a ternary weight w (the BitNet b1.58 inner step) with the inputs in
cells 0, 1, 2 and the result in one fixed cell or under the head? **No program of 1 to 16 symbols does**, in either
weight encoding: 5,726,623,060 programs per encoding at 64 steps (1 to 14 symbols locally with numpy, 15 and 16 on the
four cluster nodes with a new `prim` mode of the C sweeper, validated against the numpy search at 12 symbols and
with an Intel/AMD overlap), while the same machine has copy at 5 symbols, a ternary multiplier at 10 and an adder
at 11. The furthest any program gets is 17 of the 48 input triples.

**The DNA-1 copier sweep** ([docs/11](docs/11_dna1_copier_sweep.md)): every gene of 2 to 7 codons (64 + 4,096 +
262,144 + 16,777,216 + 1,073,741,824 + 68,719,476,736 codon strings after START) alone in a blank 8-slot world with
mu = 0, under the standard genetic code and a uniform code, verified tick by tick against the reference on 6,219
genes; the 2^36 strings of length 7 took 28 minutes on four GPUs. **No gene of up to 5 codons copies itself; at 6
codons exactly 16 strings do**, all spellings of `START R1 ALLOC R3 START INC`, which has no COPY: it rewrites itself
with INC into `START R3 SKNZ DIVIDE COPY SEEK`, copies that once and stops. **At 7 codons, 4,520 strings copy
faithfully and 2,304 of them copy the seeded gene intact: every spelling of `START SKZ DIVIDE ALLOC R3 SKNZ COPY`
and `START ALLOC SKZ DIVIDE R3 SKNZ COPY`**, the first real self-replicators of DNA-1 (the head holds COPY back
while it reads nonzero cells, copies across the blank neighbouring slot, and divides when it reaches the copy it
wrote). They reproduce whenever the ALLOC draw gives them the slot two ahead (1 in 7), lineages of depth 3 appear
under 2 of 64 seeds, fecundity is always 1. Under the uniform code no string of up to 6 codons copies faithfully and
44 of 2^36 seven-codon strings do, none intact: the intact mechanism needs the gene's own codons to begin with
nonzero letters during the claim tick, which the standard code's spellings (TT?, GG?, GC?) provide and the uniform
code's 21 meaningful codons (all A or C) cannot. The standard code's degeneracy gives it 3 to 40 times more
claimers, copiers and dividers per string than the uniform code at every length.

**Real ISAs.** AArch64 exhaustive over all 2^32 words: **1,526,900,218 valid instructions (35.5509 %)**, 1,298
mnemonics, 15 minutes on the workstation. Encoding density of other ISAs (16.7 M-word samples, Capstone): ARM32
86.8 %, MIPS64 79.7 %, PowerPC64 76.9 %, SPARC 52.6 %, **x86-64 22.4648 % whole 4-byte words (exact, all 2^32,
940 mnemonics)**, RISC-V without compressed instructions 7.1 %; the designed machines (NANO, Universe-1, universe7, Universe 2) and the genome's codon are at
100 % (docs/02, section 5b). The execution rows below are 1 M-word samples.

| | AArch64 | x86-64 |
|---|---:|---:|
| 4-byte words that are a complete instruction (sequence) | 35.5509 % (exact) | 22.4648 % (exact) |
| of all words: trap or privileged | 5.96 % | 0.57 % |
| of all words: branch away | 4.76 % | 5.46 % |
| of all words: write memory | 2.64 % | 5.45 % |
| of all words: change registers or flags only | 9.37 % | 7.38 % |
| of all words: do nothing | 8.77 % | 1.25 % |

NANO chains, verified on all 65,536 input pairs: `17C5915071 -> 17C5915071` is the identity (xor 7 twice),
`456650D0 -> 17C5915071` is (a + b) xor b xor 7, three adders in a row give a + 3b.

## Quickstart

Python 3.11+, `pip install capstone unicorn numpy`. No compiler or GPU needed.

```bash
python tools/nano_ref.py "17 C5 91 50 71" -a 3 -b 5 --first-output
python tools/nano_compose.py 456650D0 17C5915071 --formula "(a+b)^b^7"
python tools/isa4_census.py --arch both --sample 1000000
python tools/isa4_exec_probe.py --arch aarch64 --sample 100000
python tools/u2_ref.py P4 "1 0 2 1 0 3" --W 1 --trace
python tools/u2_ref.py --sweep 8 --W 2
```

## Caveats

- The six-byte NANO "formulas" in the predecessor are from samples (8.4 M and 20 M programs); only the class map is
  exhaustive at 6 bytes.
- Execution classes come from Unicorn (QEMU's CPU model), from one fixed state, with at most 4 instructions per word.
  They describe the architecture as modelled, not a physical chip.
- Sample percentages carry about ±0.05 percentage points at 1 M words.

## License

MIT, see [LICENSE](LICENSE).
