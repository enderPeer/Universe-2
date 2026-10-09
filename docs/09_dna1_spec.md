# DNA-1: exact specification of the first DNA-like machine and its world

Version 1, 2026-10-09. This is the contract between the Python reference (`sim/dna_ref.py`) and the GPU engine
(`gpu/dna_world.cu`): both implement exactly this, and they must produce identical world states on identical seeds.
Everything that is a design choice rather than a biological fact is marked *choice*.

## 1. Symbols and strands

- Alphabet: 4 symbols, 2 bits. Letters A = 0 (`00`), C = 1 (`01`), G = 2 (`10`), T = 3 (`11`).
- Complement: `comp(s) = s xor 3` (A-T, C-G).
- The world tape has N cells in a ring. Two strands: `fwd[i]` and `rev[i]`. The pairing invariant is
  `rev[i] == comp(fwd[i])`. The forward strand is the one that is read and executed; the reverse strand is the
  backup. (*choice*: one ring, two strands, forward strand executed.)
- Storage: 2 bits per cell, 4 cells per byte, cell i in byte `i >> 2`, bits `2*(i & 3)`..+1 (little-endian within
  the byte). Both engines use this layout so that memory dumps compare byte for byte.

## 2. Slots and genes

- N is a multiple of the slot size G. Slot s is cells `[s*G, (s+1)*G)`. S = N / G slots. (*choice*: fixed slots
  as in Universe-1's design C soup; the default G = 1,024 cells = 341 codons, "a gene of hundreds of codons".)
- A codon is 3 consecutive cells read in frame from the slot start: codon k of slot s is cells `s*G + 3k .. +2`,
  value `c = 16*cell0 + 4*cell1 + cell2` (0..63). Codons that would cross the slot end do not exist; the last
  usable codon index is `floor(G / 3) - 1`.
- The gene of a slot is the codon sequence from codon 0 up to and excluding the first STOP codon (or up to the last
  usable codon if there is no STOP). A slot is **expressible** if codon 0 is START (AUG, see section 3) and the
  gene has at least 2 codons (START plus one more). A gene has length `len` codons (START included), `1 < len <= 341`.

## 3. The codon table

64 codons map to 20 operations plus STOP with the degeneracy pattern of the standard genetic code (RNA U is T here).
Codon written as three letters, first letter the most significant.

| codons | amino acid | operation |
|---|---|---|
| ATG | Met | START (when executed inside a gene: NOP) |
| TAA, TAG, TGA | stop | STOP (ends the gene) |
| TTA, TTG, CTT, CTC, CTA, CTG | Leu | COPY |
| TCT, TCC, TCA, TCG, AGT, AGC | Ser | R1 |
| CGT, CGC, CGA, CGG, AGA, AGG | Arg | L1 |
| GTT, GTC, GTA, GTG | Val | JNZB |
| CCT, CCC, CCA, CCG | Pro | SETP |
| ACT, ACC, ACA, ACG | Thr | SETQ |
| GCT, GCC, GCA, GCG | Ala | ALLOC |
| GGT, GGC, GGA, GGG | Gly | DIVIDE |
| ATT, ATC, ATA | Ile | SKNZ |
| TTT, TTC | Phe | SKZ |
| TAT, TAC | Tyr | LDA |
| CAT, CAC | His | STA |
| CAA, CAG | Gln | INC |
| AAT, AAC | Asn | FLIP |
| AAA, AAG | Lys | R3 |
| GAT, GAC | Asp | L3 |
| GAA, GAG | Glu | SEEK |
| TGT, TGC | Cys | COPYR |
| TGG | Trp | REWIND |

Mapping rationale (*choice*): the most redundant amino acids carry the replication primitives (Leu: COPY, Ser and
Arg: the moves), so random code is more likely to contain copy loops; the 4-codon families carry the loop and
pointer operations; the rare codons carry the mutagenic and rarely useful ones. A `--uniform-code` option maps
codons 0..20 to the 21 meanings in the table order and leaves 21..63 as NOP, for the comparison in docs/08 E3.

## 4. The processor

A processor belongs to one slot (its gene). State:

| field | meaning | initial value at birth |
|---|---|---|
| `slot` | the slot it executes | the slot |
| `pc` | codon index within the gene, 0 .. len-1 | 1 (the codon after START) |
| `H` | head on the world tape, absolute cell index | `slot*G` |
| `P` | template pointer (absolute cell) | `slot*G` |
| `Q` | write pointer (absolute cell) | 0, invalid until ALLOC |
| `child` | claimed child slot or -1 | -1 |
| `A` | accumulator, 0..3 | 0 |
| `age` | ticks lived | 0 |
| `depth` | lineage depth | parent's depth + 1 (0 for seeded or spontaneous processors) |
| `copied` | cells written by COPY/COPYR into the child since the claim | 0 |

Per tick every live processor executes `I` instructions (default I = 16, *choice*: Universe-1's soup income). An
instruction reads the current codon `op = table[codon(slot, pc)]`, sets `pc = pc + 1`, then acts; afterwards if
`pc >= len`, `pc = 1` (the gene loops, skipping START). All cell indices are taken modulo N (ring). All writes to
`fwd[i]` also set `rev[i] = comp(new value)` (the pairing is restored by the write), except where stated.

**Snapshot rule for reads.** A processor reads cells of its own slot and of its claimed child slot live (it is the
only writer there). Every other cell it reads (LDA, SKZ, SKNZ, JNZB, SEEK, COPY's template through P when P points
elsewhere) is read from the **state at the start of the tick** (`fwd_old`, a copy taken before step 1). The gene
itself (codon fetch) is read live from the own slot. This makes the tick independent of the order in which
processors execute, so the reference (sequential) and the engine (parallel) agree. Costs one extra copy of `fwd`.

| operation | effect |
|---|---|
| NOP | nothing |
| R1 / L1 / R3 / L3 | `H += 1 / -1 / 3 / -3` |
| INC | `fwd[H] = (fwd[H] + 1) mod 4` (mutagenic; allowed only inside the own slot or the claimed child slot, otherwise NOP) |
| FLIP | `fwd[H] ^= 1`, same write rule as INC |
| LDA | `A = fwd[H]` |
| STA | `fwd[H] = A`, same write rule as INC |
| SKZ | if `fwd[H] == 0`: `pc += 1` |
| SKNZ | if `fwd[H] != 0`: `pc += 1` |
| JNZB | if `fwd[H] != 0`: `pc -= 2` (before the wrap rule; if the result is below 1 it becomes 1) |
| SETP | `P = H` |
| SETQ | `Q = H` (any cell; writes through Q still obey the write rule) |
| REWIND | `P = slot*G`; `Q = child*G` if a child slot is claimed, else unchanged |
| SEEK | the next codon of the gene is the template (3 cells); `pc += 1` to skip it; then H moves forward (H+1, H+2, ...) to the first position h within `3*G` cells where `fwd[h], fwd[h+1], fwd[h+2]` equal the **complement** of the template, cell by cell; if none, H is unchanged. The template codon is consumed, not executed. If the template would lie beyond the gene end, SEEK is NOP. |
| ALLOC | request a child slot (resolved at the end of the tick, section 5); NOP if a child is already claimed |
| COPY | if `child >= 0` and `Q` lies inside the child slot: `fwd[Q] = mutate(fwd[P])`, `rev[Q] = comp(fwd[Q])`, `P += 1`, `Q += 1`, `copied += 1`; otherwise `P += 1`, `Q += 1` only |
| COPYR | as COPY but writes `mutate(comp(fwd[P]))` and does `P -= 1`, `Q += 1` (inversion: a reverse-complement copy) |
| DIVIDE | if `child >= 0` and `copied >= 6` (at least two codons written): the child slot is marked for birth (section 5); then `child = -1`, `copied = 0`, `Q = 0`. Otherwise NOP |
| STOP | never executed (it ends the gene); if the gene has no STOP the wrap rule applies at the last usable codon |

Write rule: INC, FLIP, STA write only if H is inside the processor's own slot or inside its claimed child slot;
otherwise the instruction is a NOP. COPY and COPYR write only into the claimed child slot. So a slot's content is
changed only by its own processor, by the parent that claimed it (while it is dead), by the world's damage and
inflow (section 6), and never by anyone else. (*choice*: this is what makes the tick race-free and the engines
comparable.)

`mutate(v)`: with probability `mu` per written cell the value is replaced by a uniformly random *different* symbol
(the PRNG of section 7). Point mutations only in version 1; frame shifts, duplications and transposition arise from
the organisms' own pointer arithmetic, inversions from COPYR.

## 5. The tick

A tick t (t = 0, 1, 2, ...) does, in this order:

1. **Execute.** For every live processor, in increasing slot order in the reference and in parallel in the engine
   (the write rule makes the order irrelevant), run `I` instructions. ALLOC requests are recorded, not resolved.
   A processor whose slot stopped being expressible (its own writes destroyed START, or it has no second codon)
   finishes its instructions and dies at step 4.
2. **Resolve ALLOC.** Each requesting processor p proposes the candidate slot `cand = (slot_p + 1 + (rnd(t, slot_p, 0) mod (S - 1))) mod S`
   (never its own slot). A candidate that is dead (no processor, not claimed) and proposed by one or more
   processors is granted to the **lowest slot index** among them; the others fail silently. A granted child is
   claimed: `child = cand`, `Q = cand*G`, `copied = 0`.
3. **Births.** For every processor that executed DIVIDE this tick (step 1 marked its child slot): if the child slot
   is expressible now, a new processor is born in it with the initial values of section 4 and `depth = parent.depth + 1`;
   the birth is logged (tick, parent slot, child slot, parent gene hash, child gene hash, faithful = hashes equal).
   If not expressible, nothing is born (the slot keeps its content and is dead; the claim is released).
4. **Deaths.** `age += 1` for all processors. A processor dies if `age >= T` (lifetime, default T = 64 ticks,
   *choice*) or if its slot is not expressible. Its claimed child, if any, is released (the child's content stays).
5. **Damage.** `D` cells are hit (default D = max(1, N / 2^20) per tick, about one per million cells, *choice*): cell
   `i_k = rnd(t, k, 1) mod N` for k = 0..D-1 gets `fwd[i_k] = fwd[i_k] xor (1 + rnd(t, k, 2) mod 3)`; `rev` is not
   touched, so the pair now mismatches.
6. **Repair.** If the double strand is enabled: every cell hit in step 5 of this tick is restored, `fwd[i] = comp(rev[i])`.
   (In version 1 the repair pass only visits the cells hit this tick, which is equivalent to a full mismatch scan
   because writes always restore the pairing.) If the double strand is disabled, nothing happens and the damage
   persists as a mutation.
7. **Inflow.** `F` dead slots (default F = max(1, S / 2^12), *choice*) receive fresh random content: slot
   `d_k = rnd(t, k, 3) mod S`; if it is dead and unclaimed, all its G cells get `fwd = rnd` symbols (and `rev = comp`),
   using `rnd(t, k, 4 + j)` for cell j; if it is live or claimed, nothing happens. If the same dead slot is selected by
   several k in one tick, the refills are applied in increasing k, so the largest k wins. If the refilled slot is
   expressible, a processor with `depth = 0` is born in it immediately (a spontaneous organism) and logged as a birth
   with parent -1.
8. **Census** (every `census_every` ticks and at the end): live processors; births, faithful births, spontaneous
   births and deaths since the last census; distinct gene hashes among live processors and the 10 most common with
   counts; the lineage-depth histogram; the state hash of section 8.

## 6. Initial state

Seed `seed` (64-bit). Tick 0 state: every cell `fwd[i] = rnd(-1, i, 0) mod 4`, `rev = comp(fwd)` (random soup), or,
with `--blank`, all zero. Then `n_seed` seeded organisms (default 0): slots `k * (S / n_seed)` receive the genome
given by `--genome` (a codon string), padded with STOP, and a processor is born in each with depth 0. Processors
for other expressible random slots are **not** created at tick 0; organisms enter through births and inflow
(*choice*: so that every live processor has a known origin).

## 7. Random numbers

Deterministic and reproducible: `rnd(t, k, j) = splitmix64(seed xor mix(t, k, j))` where
`mix(t, k, j) = (t + 1) * 0x9E3779B97F4A7C15 + k * 0xBF58476D1CE4E5B9 + j * 0x94D049BB133111EB` (unsigned 64-bit
wraparound, t = -1 allowed for the initial state) and splitmix64 is the standard finaliser:
`z = x + 0x9E3779B97F4A7C15; z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9; z = (z ^ (z >> 27)) * 0x94D049BB133111EB; return z ^ (z >> 31)`.
The mutation decision for a COPY at tick t by processor in slot s writing its c-th cell since the claim uses
`r = rnd(t, s, 1000 + c)`: mutate if `(r >> 32) / 2^32 < mu`, new symbol `v + 1 + (r mod 3)` mod 4.

**Gene hash.** `genehash(slot)` = FNV-1a 64 (offset 14695981039346656037, prime 1099511628211) over the codon
values 0..63 of the gene as single bytes, START included, STOP excluded. Used in the birth log (faithful = parent
and child gene hashes equal) and in the census.

## 8. State hash and dumps

The state hash after a tick is FNV-1a 64 over: the packed `fwd` bytes, then the packed `rev` bytes, then for each
live processor in slot order the little-endian 64-bit fields `slot, pc, H, P, Q, child, A, age, depth, copied`, then
the little-endian 64-bit counters `births, faithful, spontaneous, deaths` (cumulative). A dump writes the same bytes
to a file. Reference and engine must agree on the hash after every tick of every test configuration. Hashing needs
the whole tape on the host; at N = 3 x 10^9 the engine computes it only when `--hash-every` asks for it (0 = only
at the end, or never with `--no-hash`), and the census of step 8 does not need it.

## 9. Parameters and defaults

| name | default | note |
|---|---|---|
| N | 3,000,000,000 (3 x 10^9) | cells; must be a multiple of G; the test worlds use N = 16,384 to 1,048,576 |
| G | 1,024 | slot size in cells |
| I | 16 | instructions per processor per tick |
| T | 64 | lifetime in ticks |
| mu | 0.001 | mutation probability per copied cell |
| D | max(1, N / 2^20) | damaged cells per tick (integer division) |
| F | max(1, S / 2^12) | refilled dead slots per tick (integer division) |
| double strand | on | repair of damage from the reverse strand |
| code | standard | or uniform |
| seed | 1 | |

Memory at N = 3 x 10^9: fwd 750 MB, rev 750 MB, processors at most S = 2,929,687 x 80 bytes = 234 MB, slot
state 3 MB: about 1.75 GB, fits any GPU of the cluster. The engine stores processors in a dense array indexed by
slot (a slot either has a processor or not), so births and deaths are flag changes.

## 10. Test configurations (both engines, hashes must match tick by tick)

| name | N | G | ticks | seed | notes |
|---|---:|---:|---:|---:|---|
| t1 | 16,384 | 256 | 200 | 1 | random soup, inflow on |
| t2 | 65,536 | 1,024 | 100 | 2 | random soup |
| t3 | 65,536 | 256 | 300 | 3 | seeded with the hand-written copier of section 11 in 4 slots, mu = 0 |
| t4 | 65,536 | 256 | 300 | 3 | same with mu = 0.01 |
| t5 | 1,048,576 | 1,024 | 50 | 5 | double strand off |
| t6 | 65,536 | 256 | 300 | 7 | uniform code |

## 11. A hand-written copier (seed organism)

In codons: `ATG GCT CCT ACT CTT GTT GGT TAA` = START, ALLOC, SETP, SETQ, COPY, JNZB, DIVIDE, STOP. It does not
work as written (SETQ copies H, not the child start; JNZB tests the tape under H, which never moves): the
reference run of t3 must report that. The first real copier for this machine is to be found by the sweep of
section 12, not written by hand; a corrected hand-written organism may be added once the engines agree.

## 12. Sweeps that this machine makes possible

- Copier sweep: every gene of 2 to 7 codons (4^3 to 4^18 codon strings with START prefix) alone in a small world
  with mu = 0: does it claim, copy itself faithfully and divide within T ticks, and how many times (fecundity)?
- Robustness: for the copiers found, the fraction of single-symbol mutations that preserve copying, under the
  standard code and the uniform code.
- The world at 3 x 10^9 cells: births per tick, lineage depth, faithful fraction, genome census, with and without
  the double strand and with and without mutation.
