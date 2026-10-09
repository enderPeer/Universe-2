# From the parts we found to a big program

The sweeps produced catalogs of *small* programs with known behaviour: 219,846 NANO behaviour signatures, 74,527
proven adders, five dissected six-byte functions, Universe-1's 1.8 M unary and 24.7 M binary champion functions
with witnesses, 141 + 21,818 crawler copiers. "A big program" means assembling these into something that computes
more than any one of them can. There are three different ways to do that, and only the first works today.

## 1. Chains on the host: composition (works now)

A mapped NANO program is a function f(a, b) = the first OUT within 64 steps, a in M5, b in M6. A **chain** runs
stage 1, takes its answer as the a of stage 2 (b is supplied again, memory is fresh), and so on. The chain's function
is f_k(... f_2(f_1(a, b), b) ..., b). This is exactly Universe-1's composition, for which it proved that chains of
champion programs reach every one of the 16^16 unary 4-bit functions.

[`tools/nano_compose.py`](../tools/nano_compose.py) does this for NANO with exact verification on all 65,536 pairs:

| chain | formula | verified |
|---|---|---|
| `17C5915071` -> `17C5915071` | a (xor-7 twice is the identity) | 65,536 / 65,536 |
| `456650D0` -> `17C5915071` | (a + b) xor b xor 7 | 65,536 / 65,536 |
| `456650D0` -> `456650D0` -> `456650D0` | a + 3b | 65,536 / 65,536 |

The search mode finds chains for a target over a basis file: `--basis basis.txt --target "(a*b)&255" --depth 2`.
Tables are 256x256 uint16 arrays; composing two is a numpy gather of about 50 µs, so 10^6 compositions take a
minute on the workstation. The limits are the basis size (depth-2 search is quadratic) and the Python table
computation (about 1 s per program; the 219,846 catalog representatives need the cluster's CUDA catalog kernel
extended to full tables: 219,846 x 64 KB = 14 GB).

**First target: multiplication.** No program up to 5 bytes multiplies. The question "is a*b mod 256 a chain of
two or three catalog functions?" is decidable: for every stage-1 function g that is a bijection in a for fixed b,
the required stage 2 is h = target o g^-1, one table lookup per g (Universe-1's trick). The same test covers the
nine functions Universe-1 still lacks at every champion step: multiply, max, gcd, lcm, divide, remainder, less-than,
carry, absolute difference.

## 2. One NANO program made of parts: does not work in 16 bytes

Concatenating two catalog programs inside one NANO memory does not compose them:

- the PC wraps at the program length L, so a 10-byte concatenation changes every `JMP n mod L` of both parts;
- the parts use absolute cells (M0..MF) and rewrite their own bytes at absolute positions; moved to another offset,
  a self-modifying part is a different program;
- the first `OUT` ends the evaluation, so the second part never gets to answer;
- 16 cells hold at most three 5-byte parts plus one data cell, and cells 8..15 double as the display.

So "a big NANO program" needs a bigger machine. Dimension42 already has one on paper: **NANO-256** (256 cells in 16
banks, programs of up to 16 bytes, 512-step budget, a Python reference and a C interpreter, no GPU port, no sweeps).
The honest status is that *no result of the 5-byte or 6-byte maps carries over to NANO-256* (bank selection
replaced NOP, and more steps change final states). A linking convention for NANO-256 would have to place each
part's data cells in its own bank and relocate its code, which the catalog's position-dependent programs do not
survive. The practical route is therefore section 1 (host chains) until a machine is designed whose parts are
relocatable, which is a new sweep, not a reuse.

## 3. Organisms made of parts: replicator + repairer

The alife toolkit shows what 5 bytes cannot combine: no program is both a replicator and a repairer, nothing protects
more than 2 of its 5 bytes, and every replicator wears out its own pointers (the same fecundity-one wall Universe-1
hit in the crawler soup). The open experiment ranked first across the three projects is whether an **intact,
repeatable replicator exists at 6 NANO bytes**. The 6-byte class map cannot answer it (classes only). A 6-byte run
of the copier kernel (`nano_copy.cu`, 482 s on 5 GPUs for 2^40) scales to roughly 256 x 482 s = 34 GPU-hours on five
cards, about 20 hours on all nine when falke64 is free. That is the cheapest big result still on the table, and the
one that decides whether "a big program by evolution" can start in NANO at all.

## 4. The same idea on a real CPU

The real-ISA census in [`02_real_isa_sweep.md`](02_real_isa_sweep.md) gives every 4-byte AArch64 or x86-64 word a
behaviour on the zero state. That is an instruction catalog by *behaviour* rather than by mnemonic, and a chain of
such words is simply a program. The composition tool's logic transfers unchanged: state in, state out, verified by
re-execution. The difference is scale, 2^32 parts instead of 2^40 programs, and that the parts were designed to
compose, so the "big program" question there is not whether chains work but which behaviours are reachable in how
many instructions.

## 5. Order of work

1. Exhaustive decode census, both ISAs (running on the workstation, about 15 minutes each).
2. Execution census of all 2^32 AArch64 words with Unicorn, split across the workstation and the idle cluster CPUs;
   fingerprints written per chunk; count distinct behaviours.
3. Full 256x256 tables for the 219,846 NANO catalog representatives on one cluster GPU (extend `nano_catalog.cu`),
   then the depth-2 membership test for multiply and the nine Universe-1 targets with `nano_compose.py`.
4. The 6-byte NANO copier sweep on the cluster GPUs once falke64's Avida run has finished.
