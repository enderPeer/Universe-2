# Universe 2

**What do all the short programs of a machine do, and what can be built from them?**

Universe 2 is the fourth repository in a line of exhaustive program sweeps. The first three mapped tiny machines
of their own design; this one starts from their results and asks two new questions:

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
| [`tools/isa4_census.py`](tools/isa4_census.py) | decode census of 4-byte words (Capstone), sampled or exhaustive, x86-64 and AArch64 |
| [`tools/isa4_exec_probe.py`](tools/isa4_exec_probe.py) | execution census (Unicorn): NANO-style classes and state fingerprints for every decodable word |
| `results/isa4_census/`, `results/isa4_exec/` | JSON results with counts, percentages, throughput and host |
| [`docs/03_resources.md`](docs/03_resources.md) | the workstation and the four cluster nodes as found on 2026-10-09, and how to bound CPU, memory and power on each |

## First results (2026-10-09)

Random samples; the exhaustive 2^32 decode census was running when this was written and its numbers go into
docs/02 when it finishes.

| | AArch64 | x86-64 |
|---|---:|---:|
| 4-byte words that are a complete instruction (sequence) | 35.55 % | 22.46 % |
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
```

## Caveats

- The six-byte NANO "formulas" in the predecessor are from samples (8.4 M and 20 M programs); only the class map is
  exhaustive at 6 bytes.
- Execution classes come from Unicorn (QEMU's CPU model), from one fixed state, with at most 4 instructions per word.
  They describe the architecture as modelled, not a physical chip.
- Sample percentages carry about ±0.05 percentage points at 1 M words.

## License

MIT, see [LICENSE](LICENSE).
