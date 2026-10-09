# Building the machine like DNA: alphabet, complement, strand, codon, polymerase, ribosome

Question (2026-10-09): A, T, C, G; A pairs with T and C with G; a 3-billion-letter sequence. If the instruction set
and its working should mimic this, how do we construct it? This note maps each biological mechanism onto a machine
mechanism, says what the sweeps so far already tell us about each, and lays out the construction in layers with the
first experiments for each layer.

## 1. The mapping

| biology | machine | status in Universe 2 |
|---|---|---|
| four bases, 2 bits each | 4-symbol alphabet, 2 bits per symbol | done: the whole table search is on this alphabet |
| complementary pairing A-T, C-G | an involution on symbols, `comp(s) = s xor 3` with A=00, T=11, C=01, G=10; comp(comp(s)) = s | not yet used by any machine |
| double strand | two tapes locked by `other[i] = comp(this[i])`: one strand holds the information, both hold it redundantly | the repairer NANO never found: a damaged cell is restored from its partner |
| strand direction 5' to 3'; the partner is read the other way | a head on the complement strand runs the sequence backwards and complemented: every block encodes two programs, p and revcomp(p) | measurable now on the function machine (section 4, E1) |
| polymerase copies a strand symbol by symbol using the template | a copy head: read at p, write comp at q, advance both; **the two pointers live in the machine, not in the tape** | the decisive lesson of Universe-1's soup: when the copy loop's pointers are its own code words they wear out and fecundity is one; DNA keeps them in the enzyme |
| codon: 3 bases, 64 codes, 20 amino acids + stop, synonyms | instruction = 3 symbols = 6 bits; 64 codes mapped onto about 20 operations plus STOP with degeneracy | our instruction is 1 symbol = 4 operations; the codon layer is the next experiment (E3) |
| start codon, stop codons | a gene is the block between START and STOP; the ribosome scans for START | relocatable blocks with structural boundaries: exactly the composition property docs/04 wanted |
| promoter, repressor, transcription factor | template search ("seek the pattern") decides which block runs and when | SEEK is in the second catalog round |
| transcription to mRNA, translation to protein | the block is copied, then executed on a working tape; the protein is the block's **function**, not its sequence | the function measure of docs/07 is the protein: input in, output out |
| ATP | the biological machine's accumulator A, consumed by action | implemented (EAT, MATE, SLEEP, WASTE), saturating as a computer but it is the energy layer |
| mutation: point, insertion, deletion, duplication, inversion, transposon | symbol change; length change (frame shift); block copy; **reverse complement of a block**; self-copying blocks (replicators) | the alife toolkit and copier sweeps found replicators in NANO and the crawler, all fecundity one |
| 3 x 10^9 bases, about 1.5 % coding, about 20,000 genes | a world tape of 3 x 10^9 cells = 750 MB at 2 bits (1.5 GB double-stranded), fits one GPU; genes are blocks of hundreds of codons | the "100 million words" question of docs/05 section 6b, scaled up 30x: a world, not a sweep |

## 2. What the results so far say about each design choice

1. **Keep the dense alphabet.** 100 % of symbol strings are programs (docs/02 section 5b): every mutation is a program.
   Real ISAs waste 64 to 93 % of their encodings on undefined words; DNA wastes none at the base level and spends its
   redundancy at the codon level instead.
2. **One cheap conditional, nothing more, at the symbol level.** Every top function machine has exactly one
   conditional (a two-instruction loop or a skip), and Turing-complete brackets rank 2,628th (docs/07). The codon
   layer can afford more operations (20), but the sweep says the control part should stay minimal.
3. **Pointers in the machine.** Three projects found replicators (NANO 316, crawler 141 at 4 bytes, 3 intact at 5
   bytes) and every one has fecundity one because the copy loop's pointers are code. DNA does not have this problem:
   the polymerase holds the pointers. So the copy mechanism is a machine instruction with machine-held registers
   (COPY: read template at P, write comp at Q, P += 1, Q += 1), not a program.
4. **The complement is the repair mechanism.** No 5-byte NANO program is both a replicator and a repairer, and
   nothing protects more than 2 of its 5 bytes. A double strand protects every cell: a mismatch `other[i] !=
   comp(this[i])` identifies a damaged cell and either strand can restore the other. This is free once the pairing
   is a world rule.
5. **Degeneracy costs expressiveness per bit and buys robustness; measure both.** A 6-bit codon for 21 meanings
   spends 1.6 bits per instruction on synonyms. The sweep can quantify what that buys: the fraction of single-symbol
   mutations that leave a program's function unchanged (its neutral neighbourhood), against the number of
   functions per program bit.
6. **Two programs per block.** Because the partner strand is read backwards and complemented, revcomp(p) is a
   different program with the same information. Nothing in NANO or Universe-1 has this symmetry; it doubles what a
   block can express and is testable today.

## 3. The construction, in layers

**Layer 0, physics.** A ring of N cells of 2 bits; `comp(s) = s xor 3`. Optionally a second strand kept consistent by
the world (`other[i] = comp(this[i])` after every write), which is what makes repair possible.

**Layer 1, replication.** A polymerase head with two machine registers P (template) and Q (target) and one
instruction COPY (write comp(tape[P]) at Q, P += 1, Q += 1), plus a fidelity parameter: with probability mu the
written symbol is wrong (point mutation), and with smaller probabilities a symbol is skipped or doubled (frame
shift), a block is written reverse-complemented (inversion), or a block is written twice (duplication). Fecundity is
no longer bounded by the genome's pointer bits; it is bounded by energy (layer 4).

**Layer 2, expression.** A ribosome reads codons: three symbols, 64 codes, a table onto about 20 operations and
STOP, with a START pattern. The operations are the top of the function search: pens, moves, one or two
conditionals (JNZB, SKNZ), and the accumulator operations, chosen by a 20-of-36 search (too large for brute force:
C(36, 20) = 7.3 x 10^9 tables; use the frequency ranking of docs/07 and a greedy or evolutionary table search).
The ribosome executes the block on a working tape with an input; the block's function is its protein.
Degeneracy pattern as in the standard code: the first two symbols fix the operation for most codons, the third is
mostly free, so a mutation in a third position is usually silent.

**Layer 3, regulation.** SEEK (move the head to the next occurrence of a 3-symbol template) and the conditional
decide which blocks run. Promoters are templates before blocks; a block that writes a template elsewhere regulates
another block. This is also how the copy head finds its origin.

**Layer 4, the world.** One double-stranded tape of 10^8 to 3 x 10^9 cells on a GPU, many ribosomes and polymerases
with energy budgets (the EAT/WASTE accounting), random inflow of symbols, and the measurements of the Universe-1 soup:
births per generation, lineage depth, Bedau activity against a neutral shadow, novelty of born blocks, now with
block length and reach distributions since length is variable.

## 4. First experiments, cheapest first

- **E1, reverse-complement census** (minutes, existing data): for the function winner `L1, INC_L1, FLIP_R3, JNZB`
  at 8 and 10 symbols, how often is f(revcomp(p)) = f(p), its inverse, or a new function? Says how much a second
  strand is worth as a second program.
- **E2, neutral robustness** (hours on the cluster CPUs, new mode of the sweeper): for the top 100 function machines,
  the share of the 2L single-symbol mutations of each program that preserve its function; a machine's robustness
  curve against its function count. This is the number degeneracy will be compared against.
- **E3, the codon machine** (a day): instruction = 3 symbols, 20 operations + STOP from the function ranking,
  (a) with the standard code's degeneracy pattern and (b) with 21 codes used and 43 undefined, (c) with a uniform
  3-symbol code for 64 distinct operations. Sweep all programs of 1 to 5 codons (3 to 15 symbols, up to 4^15 = 1.07 G
  programs) with the function measure, and the robustness of E2. Decides what the genome's 1.6 redundant bits buy.
- **E4, the double strand as repairer** (a day): the alife toolkit's damage test (corrupt k cells before the run) on
  single- against double-stranded tapes for the top machines; count programs whose function survives.
- **E5, template copying with machine pointers** (the design C2 of Universe-1, here): a COPY instruction with
  registers P, Q; the copier sweep over all programs of 1 to 10 symbols for fecundity, i.e. how many faithful
  copies a block makes within the budget, which the crawler and NANO could never raise above one.
- **E6, the world** (after E3 to E5): layer 4 on one GPU with a 10^8-cell tape, then 3 x 10^9.

The order matters: E1 and E2 cost nothing new, E3 is the first machine that is codon-shaped, and E5 is the
experiment that decides whether DNA's separation of pointers from code is what the three predecessors were missing.
