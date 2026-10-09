# DNA-1 seeded worlds: the 7-codon copier in the big world (2026-10-09)

[docs/11](11_dna1_copier_sweep.md) found the first intact self-replicators of DNA-1 in a blank 8-slot world of
256-cell slots: every spelling of `START SKZ DIVIDE ALLOC R3 SKNZ COPY` and `START ALLOC SKZ DIVIDE R3 SKNZ COPY`,
heritable in 1 of 7 ALLOC draws, fecundity 1. This note puts the copier into the world engine of
[docs/10](10_dna1_results.md): first the configuration the plan named (10^8 cells, 1,024-cell slots, random soup,
4,096 seeds, mutation, single strand, the second sequence), then the configurations in which the copier can work at
all, then the 3 x 10^9-cell world with a million seeds for 40 minutes of the RTX 4090. Every number is a count from
a file in `results/dna1_seeded/` (summaries `*.summary.json` produced by `tools/dna_seeded_report.py` from the
engine's census series and birth logs); nothing is estimated. The engine changes are output only and are verified
against the reference on 3,600 tick hashes.

**Headline.** In the world as the plan specified it (1,024-cell slots, random soup, lifetime 64) the copier is
sterile: every one of 4,096 (and later 1,000,000) seeds dies at tick 63 with 2 cells copied, because its head walks
3 cells per loop pass and never leaves its own slot, and in random soup the first nonzero cell it meets after 6
copies forces an early, unfaithful DIVIDE. Mutation, the single strand and the second sequence change nothing there
(0 faithful births in all five variants). In the world the copier evolved for (256-cell slots, blank tape) it
reproduces: 1 million seeds in the 3 x 10^9-cell world made 96,404 faithful children at tick 53, those made 31,298,
then 11,739, 4,658, 1,948, 832, 361, ...: **a geometric decay with a ratio of 0.33 to 0.50 per generation, extinct
after 14 generations**, and the same curve at 10^8 cells with or without inflow, with or without mutation. The
lineage dies because heritability is below 1 and fecundity is exactly 1 (no parent ever had a second faithful child,
in 499,306 divisions of the 3 x 10^9 run and 400,000 of the pilots): the loop's exit is a nonzero cell in the
*neighbourhood* of the child's slot, which is there in 30 to
75 % of the placements, and after one division the template pointer points past the gene for good. The ALLOC draw
(the 1-in-7 of docs/11) is not what limits it: in the big world any slot serves as a child slot. Mutation produces
no better copier: the 23 (mu 0.001) and 81 (mu 0.01) faithful births by mutant parents are all synonymous
spellings of the same seven operations, which behave identically (5,675 against 5,679 faithful births in the same
world), a variant with one extra codon is 23 times less heritable, and the one non-synonymous variant that was
born often (DIVIDE to JNZB) is sterile. Unrepaired damage at ten times the default rate *helps* (8,875 against
5,675 faithful births): the hits put nonzero cells into blank slots, which is exactly the exit signal the loop
needs. After extinction the dead copier slots (about 18 % of the world) are a reservoir: random inflow organisms
revive the gene 92 times in 20,000 ticks at 10^8 cells and 2,104 times at 3 x 10^9 by writing a START and a SKZ
codon over the first six cells of such a slot, and each revival starts a short lineage, so the seed gene still has a
few live carriers at tick 13,000 and its last faithful birth in the big world is at tick 39,273. The design change for
DNA-2 that the measurements ask for is a copy loop that terminates on the template (a STOP under P) and a DIVIDE
that re-arms the template pointer, not a better ALLOC.

## 1. Engine additions (output only)

`gpu/dna_world.cu` gained, without any change to the world rules:

| option | what |
|---|---|
| `--log-births FILE` | every DIVIDE birth unsampled (docs/10 kept 1 row in 1,000), with the child's lineage depth, the parent's age at the division, the cells the parent had copied into the child since the claim, and the parent's template pointer P |
| `--log-filter divide/faithful/all`, `--log-every k` | which rows, which ticks |
| `--log-codons` | two more columns: the parent's and the child's gene as codon letters (read at the end of the tick) |
| `--depth-cap K` | the census `depth_hist` buckets every depth >= K into one entry and gains a `max_depth` field (K = 0, the default, leaves the census exactly as the reference prints it) |
| `--top-genes FILE` | at every census, for each of the 10 most common gene hashes among the living, one carrier's codons and operations |
| `--slot-map FILE` | one byte per slot per census (dead, claimed, live with depth), the frames of the video of section 9 |
| `--max-seconds s` | stop after the tick at which the main loop has run s seconds; that tick is the last one (final census, `--out`) |

Verification (`results/dna1_seeded/verify/`): eight small worlds through the unchanged `sim/dna_ref.py` and the new
build, every tick hashed, census lines and birth files compared byte for byte: six seeded worlds (the copier in 128
or 64 slots of 256, blank tape, mu 0, 0.01 and 0.001, with and without the double strand at D 10, with inflow F 4;
the second sequence; 64 seeds in 256 slots of 1,024 in soup), the spec's t1 soup world, and the same worlds with the
new flags on. **2,300 tick hashes, 230 census lines, 8 birth files identical**, state hashes unchanged with the new
flags, and the extended log checked row by row against the reference's birth file (same rows; depth = the parent's
depth + 1; age consistent with the parent's birth tick; copied >= 6). The second build (`--log-codons`,
`--max-seconds`) was checked on three of the worlds (1,000 hashes, birth files identical), the third (`parent_P`)
on one (300 hashes). The RTX 3060 of knecht24 gives the same counters as the RTX 4090 on the 20,000-tick pilot B3
(35,451 births, 5,675 faithful, 70,890 deaths).

## 2. Why the specified configuration is sterile (and the pilot that shows it)

The copier's loop is SKZ, DIVIDE, ALLOC, R3, SKNZ, COPY (docs/11 section 6): the head H moves 3 cells per pass and
the two tests read the cell under it. SKZ skips DIVIDE when that cell is 0, SKNZ skips COPY when it is not 0. So the
organism copies one cell per *zero* read and divides at the first *nonzero* read once 6 cells are copied. In its own
slot the stride-3 cells are the first letters of its codons and of the STOP padding (T of TAA): two zeros at cells
12 and 15, then nothing. The copy proper happens while the head walks the next slot, which must read zero, and the
division when it meets a nonzero cell, in docs/11 the copy it wrote itself two slots ahead. At about 9.6 cells per
tick the head needs 27 ticks to cross a 256-cell slot and 107 ticks to cross a 1,024-cell one.

Pilot A1, the configuration of the plan (`results/dna1_seeded/pilots/p0`): N = 99,999,744 cells, G = 1,024 (97,656
slots), random soup, 4,096 seeds of `ATG TTC GGA GCA AAA ATA CTC` at every 23rd slot, mu 0, double strand, D 95, F 23,
T 64, I 16, 20,000 ticks, 3 s on the RTX 4090:

| tick | live | DIVIDE births so far | faithful so far | depth histogram |
|---:|---:|---:|---:|---|
| 9 | 4,098 | 0 | 0 | {0: 4,098} |
| 59 | 4,110 | 0 | 0 | {0: 4,110} |
| 69 | 14 | 0 | 0 | {0: 14} |
| 999 | 17 | 2 | 0 | {0: 17} |
| 19,999 | 17 | 246 | 0 | {0: 17} |

All 4,096 seeds die at tick 63 (lifetime) without a single division; the 246 DIVIDE births of the run belong to the
random inflow organisms (as in docs/10) and none is faithful. The four variants of the plan give the same world to
within one birth: mu 0.001 (A2) and the second sequence (A5) are byte-identical in the counters (246 DIVIDE births,
0 faithful), mu 0.01 (A3) 245, the single strand with D 950 (A4) 269 and 0 faithful. With lifetime 256 instead of 64
(A6, soup) the seeds do reach the next slot, divide at the first nonzero stride cell after 6 copies, and produce
5,261 children of 6 to 10 cells plus random content: 2 faithful births in the run, both of random 2-codon genes. In
the 3 x 10^9-cell world with 1,000,000 seeds, mu 0.001 and the same defaults (section 7, `big/big_spec`) it is the
same: 1,000,161 live at tick 59, 206 at tick 69, 3 faithful births in 2,000 ticks, all by random organisms.

So the slot size the spec chose for the world (1,024 cells, "a gene of hundreds of codons") and the lifetime of 64
ticks together exclude the only copier the sweep found; the sweep's world had 256-cell slots. Two ways out were
measured: slot size 256 (the sweep's), or lifetime 256 with 1,024-cell slots (A8, A9 below: the mechanism works,
four times slower). Everything below uses G = 256 unless stated.

## 3. The runs at 10^8 cells

N = 99,999,744 cells, 20,000 ticks each, seed 1, I 16, T 64, double strand, D 95 (auto), F 95 (auto; 23 for G
1,024), standard code, `--depth-cap 10`, every DIVIDE birth logged. "Spacing" is S / n_seed: the seeds sit at every
k-th slot. The first sequence `ATG TTC GGA GCA AAA ATA CTC` unless stated. Files `results/dna1_seeded/pilots/`.

| run | G | seeds | tape | mu | T | DIVIDE births | faithful births | faithful in generation 1 | of the seeds | deepest generation | ticks/s |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A1 as specified | 1024 | 4,096 | soup | 0 | 64 | 246 | 0 | 0 | 0.0 % | 2 | 6,476 |
| A2 mu 0.001 | 1024 | 4,096 | soup | 0.001 | 64 | 246 | 0 | 0 | 0.0 % | 2 | 6,500 |
| A3 mu 0.01 | 1024 | 4,096 | soup | 0.01 | 64 | 245 | 0 | 0 | 0.0 % | 2 | 6,521 |
| A4 single strand, D 950 | 1024 | 4,096 | soup | 0 | 64 | 269 | 0 | 0 | 0.0 % | 2 | 6,586 |
| A5 second sequence | 1024 | 4,096 | soup | 0 | 64 | 246 | 0 | 0 | 0.0 % | 2 | 6,521 |
| A6 lifetime 256 | 1024 | 4,096 | soup | 0 | 256 | 5,261 | 2 | 0 | 0.0 % | 4 | 4,218 |
| A7 blank, spacing 2 | 1024 | 48,828 | blank | 0 | 64 | 270 | 0 | 0 | 0.0 % | 1 | 1,834 |
| A8 blank, spacing 2, lifetime 256 | 1024 | 48,828 | blank | 0 | 256 | 31,804 | 29,371 | 19,247 | 39.4 % | 27 | 518 |
| A9 blank, spacing 11, lifetime 256 | 1024 | 8,878 | blank | 0 | 256 | 4,302 | 2,730 | 1,372 | 15.5 % | 16 | 463 |
| B1 soup | 256 | 4,096 | soup | 0 | 64 | 5,107 | 0 | 0 | 0.0 % | 3 | 3,423 |
| B2 blank, spacing 95 | 256 | 4,096 | blank | 0 | 64 | 1,181 | 144 | 134 | 3.3 % | 3 | 1,502 |
| B3 blank, spacing 11 | 256 | 35,511 | blank | 0 | 64 | 7,076 | 5,675 | 3,624 | 10.2 % | 11 | 1,519 |
| B4 blank, spacing 4 | 256 | 97,656 | blank | 0 | 64 | 60,855 | 56,686 | 22,506 | 23.0 % | 30 | 1,531 |
| B5 blank, spacing 3 | 256 | 130,208 | blank | 0 | 64 | 85,440 | 79,832 | 37,618 | 28.9 % | 34 | 1,538 |
| B6 blank, spacing 2 | 256 | 195,312 | blank | 0 | 64 | 122,662 | 116,712 | 76,587 | 39.2 % | 39 | 1,507 |
| B7 as B3, no inflow | 256 | 35,511 | blank | 0 | 64 | 4,496 | 4,484 | 3,076 | 8.7 % | 9 | 8,076 |
| B8 as B6, no inflow | 256 | 195,312 | blank | 0 | 64 | 121,708 | 117,292 | 76,675 | 39.3 % | 35 | 6,274 |
| C1 as B3, mu 0.001 | 256 | 35,511 | blank | 0.001 | 64 | 6,970 | 5,424 | 3,522 | 9.9 % | 11 | 1,524 |
| C2 as B3, mu 0.01 | 256 | 35,511 | blank | 0.01 | 64 | 6,297 | 3,930 | 2,797 | 7.9 % | 7 | 1,522 |
| C3 as B3, single strand, D 950 | 256 | 35,511 | blank | 0 | 64 | 10,750 | 8,875 | 4,894 | 13.8 % | 17 | 1,562 |
| C4 as B3, second sequence | 256 | 35,511 | blank | 0 | 64 | 7,071 | 5,674 | 3,623 | 10.2 % | 11 | 1,525 |

"DIVIDE births" and "faithful births" are the whole run's, inflow organisms and mutants included (random organisms
add about 250 DIVIDE births per run, the A1 column, and 0 to 27 faithful ones; the summaries separate the copier's
own lineage, `generations_seed`, from the rest, `faithful_births_outside_seed_lineage`, and the generation tables
below are the copier's own). "Deepest generation" is the largest lineage depth that was ever
born. Every run ends with the same background as docs/10: 72 live inflow organisms at G 256, 17 at G 1,024, and no
carrier of the seed gene (except by the revivals of section 5).

**Soup kills the copy (B1).** In random soup with 256-cell slots the seeds divide (5,107 DIVIDE births, 4,332 of the
4,096 seeds are dividing parents) but every child is unfaithful: the head meets a nonzero cell in the neighbouring
slot after 6 to 28 copied cells (mean 6.4), the child holds START, SKZ and random codons, and the parent's second and
later children do not even start with a START (P has moved on). 0 faithful in 20,000 ticks.

**Blank tape, sparse seeds (B2, B3).** With a blank neighbour the copy runs through it; what the organism then needs
is a nonzero cell at a stride-3 position within reach before it dies (cells 322 to 544 from its own start: the
second half of the next slot or the first 32 cells of the one after). On a blank tape that cell comes only from
other organisms' slots (seeds, claimed children, inflow refills). At spacing 95 3.3 % of the seeds reproduce, at
spacing 11 10.2 %, at spacing 4 23 %, at spacing 2 39 %: the denser the world, the more exit signals. Spacing 2 is
not special here (in docs/11 the child had to sit exactly two slots ahead because the world was otherwise blank).

**Dense seeds (B4 to B6).** The seeds' children land in random free slots, and their own neighbourhoods are then
whatever the world holds: in B6 the second generation is 16,446 of 76,587 (0.21) because a child next to a seed
slot reads the seed's codons at offset 2 mod 3 (G, C, A, A, A, A, C) and divides unfaithfully at the C after 6
copies, but from the sixth generation on the ratio settles at 0.72 to 0.77, the highest heritability measured, and
the lineage lasts 39 generations and 4,900 ticks.

### The decay curve, generation by generation

B3 (spacing 11, mu 0), every faithful birth by the child's lineage depth (`b256_35511.summary.json`):

| generation | births | faithful | ratio to the previous generation | first tick | last tick | copied cells (mean) | parent age at the division (median) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4,893 | 3,624 |  | 27 | 19,917 | 66.2 | 53 |
| 2 | 1,227 | 1,184 | 0.327 | 60 | 19,485 | 92.6 | 27 |
| 3 | 516 | 485 | 0.410 | 87 | 12,999 | 95.4 | 27 |
| 4 | 248 | 215 | 0.443 | 115 | 11,149 | 84.7 | 27 |
| 5 | 107 | 94 | 0.437 | 142 | 7,450 | 83.6 | 27 |
| 6 | 48 | 40 | 0.426 | 171 | 4,283 | 77.8 | 27 |
| 7 | 23 | 21 | 0.525 | 213 | 3,110 | 71.1 | 27 |
| 8 | 9 | 7 | 0.333 | 269 | 3,137 | 76.7 | 27 |
| 9 | 3 | 3 | 0.429 | 296 | 423 | 97.3 | 27 |
| 10 | 1 | 1 | 0.333 | 439 | 439 | 33.0 | 16 |
| 11 | 1 | 1 | 1.000 | 493 | 493 | 162.0 | 54 |

("births" of generation 1 includes the 1,181 unfaithful children and the late revivals of section 5, which is why
its last tick is 19,917.) The seeds divide at age 53 (their slot carries 84 STOP codons of padding to walk), their
children at age 27 (a child slot holds the 87 copied cells and then whatever was there, blank in most cases, so the
copy starts after 29 cells of padding instead of 255). The live population: 35,525 at tick 9, 39,395 at tick 59
(the seeds plus their first children), 3,909 at tick 69 (the seeds are dead), 4,312 at 99, 1,380 at 119, 388 at 199,
127 at 299, 65 at 499; the depth histogram at tick 199 is {0: 69, 1: 2, 3: 112, 4: 160, 5: 39, 6: 6}, at tick 299
{0: 64, 2: 2, 5: 13, 6: 26, 7: 16, 8: 5, 9: 1}, at tick 499 {0: 60, 1: 3, 10+: 2}. Fecundity: of the 6,278 parents
that divided in B3, 603 had no faithful child and 5,675 had exactly one; none had two (`fecundity_hist`).

B7, the same without inflow (F 0): 3,076 faithful at generation 1 (all at tick 53), then 889, 319, 125, 51, 16, 5, 2,
1: ratios 0.29, 0.36, 0.39, 0.41, 0.31, 0.31, 0.40, 0.50, extinct at tick 349 with the world empty from tick 499
(`b256_35511_F0`). The inflow is therefore not what ends the lineage in the first 500 ticks: the decay is the
copier's own. What the inflow does later is fill the blank slots with soup (95 refills per tick, every slot once in
about 4,100 ticks), which removes the blank neighbours the copier needs; the revivals of section 5 stop around tick
13,000.

B6, spacing 2 (`b256_195312`, the copier's own lineage; 18 further faithful births in the run belong to random
organisms): 76,587, 16,445, 9,973, 4,126, 2,500, 1,833, 1,354, 990, 717, 539, 403, 297, 224, 172, 132, 101, ...
(ratios 0.21, 0.61, 0.41, 0.61, 0.73, 0.74, 0.73, 0.72, 0.75, 0.75, 0.74, 0.75, 0.77, 0.77, 0.77, 0.75): the deepest
lineage reached generation 39 at tick 4,909. B4 (spacing 4) and B5 (spacing 3) settle at the same 0.73 to 0.77 from
the fifth generation on (B4: 22,506, 9,215, 6,977, 4,727, 3,435, 2,544, 1,890, 1,417, 1,076, 794, ...; B5: 37,618,
11,858, 8,864, 5,534, 3,942, 2,926, 2,190, 1,675, 1,262, 971, ...). A8 (G 1,024, lifetime 256, spacing 2): 19,247, 4,317, 2,600, 978, 518, 374, 298,
237, 185, 146, 112, 81, 62, 65, ... ratios 0.72 to 0.80 after generation 6, the division at age 213 for seeds and
107 for children: the same mechanism four times slower, 27 generations.

## 4. The variants: mutation, single strand, second sequence

Runs C1 to C4 (spacing 11), and the mutant tests of `results/dna1_seeded/mutants/` (knecht24, RTX 3060).

**Mutation** (mu is the probability per copied cell; a child's 24 gene cells carry a mutation with probability 2.4 %
at mu 0.001 and 21 % at mu 0.01): the copier's own lineage falls from 3,624, 1,184, 485, 215, 94, 40 (B3) to 3,522,
1,126, 447, 191, 71, 26 (C1, ratios 0.32, 0.40, 0.43, 0.37, 0.37) and 2,797, 733, 217, 73, 20, 7 (C2, ratios 0.26,
0.30, 0.34, 0.27, 0.35), the deepest generation from 11 to 11 and 7. The mutants that were born and *copied
themselves faithfully* (faithful births whose parent is not the seed gene; `c256_35511_mu001.log.gz`,
`c256_35511_mu01.log.gz`, columns `parent_codons`):

| mu | faithful births by mutant parents | distinct mutant genes | the genes |
|---:|---:|---:|---|
| 0.001 | 23 | 11 | `ATG TTC GGA GCA AAA ATT CTC` (5), `ATG TTC GGG GCA AAA ATA CTC` (5), `ATG TTT GGA ...` (2), `... AAA ATC CTC` (2), `... ATA CTA` (2), `... CTC GAA` (2), `ATG TTC GGT ...` (1), `... CTC TAT` (1), ... |
| 0.01 | 81 | 16 | `ATG TTC GGA GCC AAA ATA CTC` (13), `ATG TTC GGA GCT AAA ATA CTC` (10), `... CTC TAC` (10), `... ATA CTG` (7), `... CTC GAA` (6), `ATG TTT ...` (5), `ATG TTC GGT ...` (5), `... ATA CTT` (5), ... |

Every one of them is either a synonymous spelling of the same seven operations (TTT for SKZ; GGG, GGC, GGT for
DIVIDE; GCC, GCT for ALLOC; ATT, ATC for SKNZ; CTA, CTG, CTT for COPY) or the original gene with its STOP codon
mutated into an eighth codon (GAA SEEK, TAC LDA, TAT LDA, CAA INC) and the next padding codon as the new STOP. No
mutant with a different operation sequence ever copied itself. The one non-synonymous variant that was born often,
`ATG TTC GTA GCA AAA ATA CTC` = START SKZ JNZB ALLOC R3 SKNZ COPY (DIVIDE to JNZB, 22 births as a child at mu 0.01),
never divided. Heritability of the mutants, measured by seeding 35,511 copies of each into the B3 world (mu 0;
`mutants/m_*.summary.json`):

| seeded gene | faithful births | generation 1 | generations | note |
|---|---:|---:|---:|---|
| `ATG TTC GGA GCA AAA ATA CTC` (original, on knecht24) | 5,675 | 3,624 | 11 | identical to B3 on the RTX 4090 |
| `ATG TTC GGG GCA AAA ATA CTC` (DIVIDE as GGG) | 5,679 | 3,628 | 11 | the stride-3 letters are the same (T, G, G, A, A, C), so the dynamics are the same; the 4 extra births are revivals that the different spelling makes or breaks |
| `ATG TTT GGA GCA AAA ATA CTC` (SKZ as TTT) | 5,675 | 3,624 | 11 | identical to the original |
| `ATG TTC GGA GCA AAA ATA CTC GAA` (STOP to SEEK) | 242 | 175 | 5 | 23 times less heritable: the loop is seven instructions long, the head moves 7 instead of 9.6 cells per tick and mostly dies before it reaches the exit cell |
| `ATG TTC GTA GCA AAA ATA CTC` (DIVIDE to JNZB) | 0 | 0 | 0 | sterile |

So no mutant with higher heritability or fecundity appeared: the synonymous ones are the same organism under another
name (and the head reads the same letters, since the degenerate code changes third letters, which the stride-3 walk
never reads in this gene), and every change of length or of an operation is worse. The best mutant *is* the original;
the trace of docs/11 section 6 (`python tools/dna_trace.py "ATG TTC GGA GCA AAA ATA CTC" --ticks 256`) applies to
all of them unchanged, and `python tools/dna_trace.py "ATG TTC GGA GCA AAA ATA CTC GAA" --ticks 256` shows the slow
variant: the SEEK's template would be its STOP, beyond the gene, so it is a NOP that costs one instruction per pass.

**Single strand with ten times the damage** (C3: `--no-double --D 950`, 950 random cells per tick xor-ed and never
repaired, against D 95 repaired): *more* faithful births, 8,875 against 5,675; the copier's lineage 4,885, 1,838,
926, 498, 270, 158, 106, 61, 36, 27, 19, 11, ... against 3,624, 1,184, 485, 215, 94, 40, 21, 7, 3, 1, 1, ratios 0.38,
0.50, 0.54, 0.54, 0.59, 0.67, 0.58, 0.59, 0.75, 0.70 instead of 0.33 to 0.44, 17 generations instead of 11, the last
faithful birth of generation 5 or deeper at tick 2,929 instead of 7,450 (a revival's descendants in B3; the main
lineage of B3 ends at tick 493). The damage puts nonzero cells into
blank slots (950 per tick is one hit per 411 slots per tick), and a nonzero cell at a stride-3 position in the slot
after the child's is the exit signal the loop needs; a hit in the gene itself (24 cells of 10^8, 2.3 x 10^-4 per tick
per organism) is rare. The double strand's repair is, for this copier, a cost: it removes the exit signals. The
damage also makes copies of genes whose STOP was hit: 33 of the 118 births of the seed gene from a parent with a
different gene in C3 have a parent that reads `ATG TTC GGA GCA AAA ATA` followed by a changed codon, a damaged copier
whose child is the undamaged gene because the copy had passed the hit cell.

**The second sequence** (C4, `ATG GCA TTC GGA AAA ATA CTC` = START ALLOC SKZ DIVIDE R3 SKNZ COPY): 5,674 faithful
births against 5,675, its lineage 3,623, 1,188, 482, 213, 92, 40, 19, 6, 3, 1, 1 against 3,624, 1,184, 485, 215, 94,
40, 21, 7, 3, 1, 1, the same 11 generations. The two sequences read the same
letters at the stride-3 positions (G, T, G instead of T, G, G at cells 3, 6, 9: all nonzero) and run the same loop
in a different order; the ALLOC request and the claim land in the same tick either way. The first sequence was used
for the big run; the choice does not matter.

## 5. Revivals: the dead slots are a reservoir

After the lineage of B3 is extinct (tick 493), the seed gene keeps appearing: 92 births of the exact seed gene whose
parent has a different gene (`c256_35511.log.gz` and `src_35511.log.gz`, rows with `child_hash d63dbd3ebc6a4cbb`
and another parent hash), between tick 175 and tick 19,454, with 6 (68 rows), 7 (18), 8 (5) or 9 (1) copied cells.
One is a copier whose own STOP had not been copied (a child of 21 to 23 cells, gene `ATG TTC GGA GCA AAA ATA CTC
AAA AAA ...`, 85 codons, which still runs the loop and whose 9-cell child, written over a dead copier slot, hashes as
the seed gene). The other 91, by 61 distinct random inflow organisms (for example `ATG CAT GTG CCA CGC ACA TGC GGA
ATC CCC GTG ATG AAT GCC TCG ...`), are 90 of the 1,055 divisions that random depth-0 organisms made in the run: a
random organism claims a slot, copies 6 to 8 cells and divides, as 3 % of random genes do (docs/10); when the child
slot is one of the dead slots that still hold the copier gene and the written cells read `ATG TTC`, the child is the
seed gene, born with lineage depth 1, and it reproduces with the usual ratio (the 92 revived organisms had 69
children within their lifetime, 49 of them faithful). 18 of the 61 parents did it more than once (up to 6 times, into
different slots), so the cells they write are not a coincidence of one tick; which cells of their surroundings they
copy was not traced (their template pointer at the division sits at their own slot start in most rows, reset by a
SETP or REWIND after the copy). The reservoir is large: 62 of the 92 revived slots had been born as the seed gene
before (a seed slot or a faithful child; a dead slot keeps its content until the inflow refills it), the other 30
are slots a seed had written its whole gene into without ever dividing (a seed that finds no exit still copies 87
cells and more into its child slot by tick 53; 31,673 of the 35,511 seeds of B3 never divided, and about nine in
ten of them had a claimed child slot to write into), so roughly 18 % of the world's slots hold the gene. The revived lineages are why generation 1 of B3 has births until tick 19,917 and why the census
lists the seed gene among the ten most common genes with 2 carriers at tick 13,019 (`b256_35511.top.gz`). The
revivals thin out as the inflow refills the dead copier slots with soup (every slot is refilled about once per
4,100 ticks; the last revival with a descendant is at tick 11,149).

## 6. The 3 x 10^9-cell world, as specified, 2,000 ticks

`big/big_spec`: N = 2,999,999,488, G = 1,024 (2,929,687 slots), random soup, 1,000,000 seeds of the first sequence at
every second slot, mu 0.001, double strand, D 2,861, F 715, T 64, 2,000 ticks, 355 ticks per second, 2.8 GB.

| tick | live | DIVIDE births so far | faithful so far | depth histogram |
|---:|---:|---:|---:|---|
| 59 | 1,000,161 | 0 | 0 | {0: 1,000,161} |
| 69 | 206 | 0 | 0 | {0: 206} |
| 199 | 565 | 60 | 3 | |
| 1,999 | 502 | 886 | 3 | |

The million seeds die at tick 63 without a division (0 DIVIDE births before tick 81); the 886 DIVIDE births and the
3 faithful ones (a random 2-codon lineage of depth 4 at ticks 173 to 188, 6 cells copied each) are the docs/10
background. The run was stopped at 2,000 ticks because its remaining 898,000 ticks would reproduce docs/10 section
7 with 1,000,000 extra dead slots; the 40 minutes went to the world in which the copier works.

## 7. The 3 x 10^9-cell world, G 256, blank, 1,000,000 seeds, 40 minutes

`big/big`: N = 2,999,999,488 cells, G = 256 (11,718,748 slots), blank tape, 1,000,000 seeds of the first sequence at
every 11th slot, mu 0.001, double strand, D 2,861 and F 2,861 (auto), T 64, I 16, census every 100 ticks, every DIVIDE
birth logged with codons, `--max-seconds 2400`. Device memory 4.07 GB. The 500-tick probe of the same world
(`big/big_probe`, 162.7 ticks per second while the million seeds and their children were alive) gave the generation
table; the 40-minute run continued it.

**The run.** 895,900 ticks in 2,400.3 s (373 ticks per second on average: 163 while the million seeds and their
children were alive, about 377 in the background phase), 8,959 census lines, 1,373,185 DIVIDE births logged with
codons (the full log is 300 MB and stays on adler40; `big_lineage.log.gz` keeps the 161,945 rows whose parent or
child is the seed gene or that are faithful). Final counters: 2,051 live, 39,527,135 births of which 38,153,950
spontaneous, 148,008 faithful, 40,525,084 deaths.

**The copier's lineage** (`big.summary.json`, `generations_seed`: births whose parent or child is the seed gene,
by the child's lineage depth):

| generation | births | faithful | ratio to the previous generation | first tick | last tick | copied cells (mean) | parent age at the division (median) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 106,673 | 96,404 |  | 27 | 39,245 | 80.8 | 53 |
| 2 | 32,447 | 31,298 | 0.325 | 59 | 39,273 | 95.2 | 27 |
| 3 | 12,989 | 11,739 | 0.375 | 68 | 39,283 | 93.3 | 27 |
| 4 | 5,380 | 4,658 | 0.397 | 96 | 21,006 | 86.1 | 27 |
| 5 | 2,215 | 1,948 | 0.418 | 126 | 11,673 | 85.8 | 27 |
| 6 | 964 | 832 | 0.427 | 159 | 7,419 | 85.2 | 27 |
| 7 | 426 | 361 | 0.434 | 198 | 4,657 | 80.3 | 27 |
| 8 | 218 | 179 | 0.496 | 223 | 3,878 | 77.4 | 27 |
| 9 | 103 | 75 | 0.419 | 268 | 3,338 | 74.8 | 27 |
| 10 | 44 | 37 | 0.493 | 315 | 2,887 | 80.3 | 27 |
| 11 | 19 | 14 | 0.378 | 372 | 2,915 | 68.7 | 27 |
| 12 | 10 | 9 | 0.643 | 430 | 2,969 | 94.6 | 54 |
| 13 | 6 | 5 | 0.556 | 457 | 2,985 | 81.3 | 27 |
| 14 | 3 | 2 | 0.400 | 511 | 2,997 | 56.7 | 27 |
| 15 | 1 | 0 |  | 538 | 538 | 17.0 | 9 |

(The 106,673 births of generation 1 are the 96,404 faithful children of the seeds, 8,165 unfaithful ones, and the
2,104 revivals below, whose child is the seed gene.) 9.6 % of the million seeds reproduced, then the ratio is 0.33,
0.38, 0.40, 0.42, 0.43, 0.43, 0.50, ...: the same curve as B3 at 10^8 cells (10.2 %, then 0.33, 0.41, 0.44, 0.44,
0.43), 30 times larger. Fecundity: 499,306 dividing parents in the whole run, 351,298 with no faithful child, 148,008
with exactly one, none with two.

| tick | live | DIVIDE births so far | faithful so far | depth histogram (10 = 10 or more) |
|---:|---:|---:|---:|---|
| 99 | 116,943 | 122,310 | 113,378 | {0: 1,772, 1: 97,604, 2: 17,276, 3: 289, 4: 2} |
| 199 | 9,610 | 154,026 | 143,276 | {0: 1,995, 1: 110, 2: 16, 3: 2,528, 4: 4,011, 5: 880, 6: 69, 7: 1} |
| 299 | 3,289 | 157,100 | 145,872 | {0: 1,941, 1: 82, 2: 16, 3: 6, 4: 2, 5: 363, 6: 528, 7: 274, 8: 68, 9: 9} |
| 499 | 2,086 | 157,855 | 146,224 | {0: 1,906, 1: 145, 2: 13, 3: 5, 5: 1, 6: 1, 10: 15} |
| 999 | 2,191 | 158,943 | 146,445 | {0: 2,049, 1: 106, 2: 17, 3: 7, 4: 5, 5: 2, 6: 3, 7: 2} |
| 2,999 | 2,150 | 163,627 | 147,224 | {0: 1,996, 1: 118, 2: 16, 3: 8, 4: 4, 5: 3, 6: 1, 7: 1, 8: 1, 10: 2} |
| 9,999 | 2,087 | 175,100 | 147,885 | {0: 2,000, 1: 80, 2: 5, 3: 2} |
| 19,999 | 2,123 | 188,937 | 147,961 | {0: 2,021, 1: 102} |
| 99,999 | 2,061 | 296,271 | 147,973 | {0: 2,000, 1: 60, 2: 1} |
| 399,999 | 2,092 | 703,121 | 147,988 | {0: 2,014, 1: 78} |
| 895,899 | 2,051 | 1,373,185 | 148,008 | {0: 1,985, 1: 64, 2: 2} |

The seeds die at tick 63 and their generation-1 children carry the population to 116,943 at tick 99; by tick 499 the
world is back to the inflow's background, about 2,050 live random organisms (2,861 refills per tick, about 1 in 90
expressible, each living 64 ticks), where it stays for the remaining 895,000 ticks. The seed gene is the most common
gene of the census from tick 99 (112,527 carriers) until tick 699 and is listed among the ten most common for the
last time at tick 14,199 (8 carriers); from tick 5,899 on the rank-1 genes are the two-codon `ATG xxx` genes of the
inflow, as in docs/10. The depth histogram after tick 10,000 (depth 1: 60 to 100, depth 2: 0 to 4) is the random
organisms' own children.

**Revivals at scale.** 2,104 births of the exact seed gene from a parent with another gene, by 1,464 distinct
parents (6 of them copier variants, the rest random inflow organisms), from tick 68 (five ticks after the seeds died,
over their dead slots) to tick 39,245, with 6 (1,676 rows), 7 (310), 8 (82), 9 (23) or 10 (8) copied cells. The
revived organisms and their descendants account for 1,664 faithful births by seed-gene parents after tick 600 and
350 after tick 5,000; the last faithful birth of the seed gene is at tick 39,273, generation 3 of a revival. Scaled by
world size this is the pilots' rate (92 per 20,000 ticks at 10^8 cells, 2,104 at 3 x 10^9, most of them in the first
10,000 ticks, until the inflow has refilled the dead copier slots).

**Mutants at scale** (mu 0.001): 447 faithful births whose parent is not the seed gene, by 63 genes: 13 synonymous
spellings (365 births: `ATG TTC GGC GCA AAA ATA CTC` 43, `... ATA CTG` 40, `... GCG AAA ...` 38, `... GCT AAA ...` 34,
`... AAG ATA ...` 30, `ATG TTT ...` 30, `... GCC AAA ...` 25, `... ATA CTA` 25, `ATG TTC GGT ...` 24, `... ATT CTC` 22,
`... ATA CTT` 21, `... ATC CTC` 19, `ATG TTC GGG ...` 14), 3 with extra codons after the gene (32 births), one
non-synonymous variant with 3 births (`ATG TAC GGA GCA AAA ATA CTC`, SKZ to LDA, whose 6-cell children can only be
faithful over a slot that already holds its gene), and 46 random inflow genes with 47 births (the docs/10 background:
47 faithful among about 1.2 million random divisions, two-codon genes copying their six cells). The census of what
evolved is therefore the seed gene, its synonyms, and nothing with a different mechanism.

## 8. Reading

1. **DNA-1 does not sustain a lineage, and the measurements say why.** Every seeded population decays geometrically
   and is extinct within 500 to 5,000 ticks, at 10^8 and at 3 x 10^9 cells, with and without inflow, with and
   without mutation. The expected number of faithful children per organism is h x f with heritability h = 0.3 to
   0.77 and fecundity f = 1, hence below 1 in every configuration; no stochastic rescue is possible because f never
   exceeds 1 (0 parents with two faithful children in 400,000 logged divisions).
2. **What limits it is the exit of the copy loop, not the ALLOC draw.** The 1-in-7 of docs/11 was the chance that
   the child's slot is the one the head would reach; in the big world the exit signal is any nonzero cell at a
   stride-3 position in the slot after the child's, and the child's slot can be anywhere. h is the fraction of
   placements whose next slot reads zero (blank or a dead slot with the right alignment) and whose slot after that
   has a nonzero cell within reach: 0.33 to 0.44 at 10 % occupancy, 0.72 to 0.77 at 50 %, higher with unrepaired
   damage (which adds nonzero cells) and lower in soup (where the next slot itself is nonzero and the copy stops after
   6 cells). An ALLOC that prefers the slot two ahead would make h = 1 only in a blank world and would change
   nothing in soup.
3. **Fecundity 1 is the second limit, and it is structural.** After DIVIDE the template pointer P points past the
   gene (at cell 87 or wherever the exit came), nothing rewinds it, and the organism's remaining ticks copy padding
   into a second child that never gets a START. The spec's REWIND would fix it at the cost of an eighth codon and of
   a loop structure that re-arms the exit test, which the sweep did not find at 7 codons.
4. **Lifetime matters only through geometry.** T = 64 is enough at G = 256 (division at age 53 for seeds, 27 for
   children) and not at G = 1,024 (A7: nothing; A8 with T = 256: the same curve at a quarter of the speed). A longer
   life does not raise h or f.
5. **Mutation finds nothing better** because the 7-codon copier is a local optimum of its own mechanism: every
   synonymous spelling is the same organism, every extra codon slows the head, every changed operation breaks the loop
   (the sweep of docs/11 says the same for all 2^36 strings: 30 faithful sequences, two of them intact, none with
   f > 1). A better copier is not one mutation away; it needs a different exit mechanism.
6. **The double strand is the wrong tool for this organism**: repair removes the nonzero cells its loop exits on. The
   design intent of docs/08 (damage as the mutation source, repair as the defence) will matter for organisms whose
   loop ends on their own template.

## 9. What to change for DNA-2

The measurements point at the copy loop's termination, not at placement:

1. **A copy loop that ends on the template, not on the neighbourhood.** Recommended: a codon-level test on the
   template pointer, `SKSTOP` (skip the next codon if the codon at P is STOP, or equivalently a COPY that copies
   three cells and sets a flag when it copied a STOP), so that `ALLOC; loop: COPY, SKSTOP, DIVIDE` copies any gene
   up to and including its STOP and divides, in any surroundings, with h limited only by the claim's success. The
   candidate "a loop that tests the cell under Q" does not do this: the cell under Q is the child slot's old content,
   which is blank or random, not a progress signal. "COPY also moves H" (H = P after a copy) lets SKZ and SKNZ read the
   template, but a 2-bit cell test cannot tell the gene's own A cells from its end; the end of a gene is a codon
   (STOP), and the test has to be a codon test. The polymerase analogy says the same: the terminator is a sequence,
   not a single base.
2. **DIVIDE re-arms the template**: P = own slot start after a division (the spec's *choice* in section 4 could simply
   do this inside DIVIDE instead of needing REWIND), so that the same loop claims again and copies again; fecundity
   is then bounded by lifetime over copy time (64 / 27 = 2 at G 256) instead of by 1, and h x f exceeds 1 as soon as
   h > 0.5.
3. **Not "ALLOC preferring the slot two ahead"**: it would raise h in a blank world only, costs the slot-independence
   that makes the engine race-free, and the big world's soup makes neighbourhood-based exits fail regardless.
4. Keep G = 256 as the world's default slot size, or raise T with G, since the 7-codon organisms of this machine
   need about 2 G / 9.6 ticks per generation.

With 1 and 2 the sweep of docs/11 should be rerun (the copier space changes), and the seeded-world experiment of
this note is the acceptance test: a lineage that holds or grows at 10^8 cells with inflow and mu 0.001.

## 10. Files and reproduction

`results/dna1_seeded/`: `pilots/` (every 10^8-cell run: `.err` configuration and summary, `.json` final census,
`.txt.gz` census series every 10 ticks, `.log.gz` every DIVIDE birth, `.top.gz` top genes per census,
`.summary.json`; the `c256_*` reruns with codons; `video.*` the run of the video), `mutants/` (the five mutant
heritability runs on knecht24), `big/` (`big_spec`: the specified 3 x 10^9 configuration, 2,000 ticks; `big_probe`:
500 ticks of the G 256 world; `big`: the 40-minute run, census every 100 ticks, every DIVIDE birth with codons),
`verify/` (the reference comparison script, its output and the log checker), `dna1_seeded_1e8.mp4` (the video:
run B3 with mu 0.001, one frame per 20 ticks for 20,000 ticks, one pixel per slot, white = seeded organism, colours
= lineage depth 1 to 6; rendered by `tools/dna_slot_video.py` from the engine's `--slot-map`). The raw stdout,
logs and the 390 MB slot map stay on adler40 in `~/universe2/dna/seeded/runs/`.

```bash
nvcc -O3 -std=c++17 -arch=sm_89 -o dna_world gpu/dna_world.cu
# A1, the specified pilot
./dna_world --N 99999744 --G 1024 --ticks 20000 --seed 1 --no-hash --census-every 10 --depth-cap 10 --n-seed 4096 \
    --genome "ATG TTC GGA GCA AAA ATA CTC" --mu 0 --log-births p0.log --top-genes p0.top --out p0.json > p0.txt 2> p0.err
# B3, the copier's world
./dna_world --N 99999744 --G 256 --blank --ticks 20000 --seed 1 --no-hash --census-every 10 --depth-cap 10 --n-seed 35511 \
    --genome "ATG TTC GGA GCA AAA ATA CTC" --mu 0 --log-births b3.log --log-codons --top-genes b3.top --out b3.json > b3.txt 2> b3.err
# the 40-minute 3e9 run
./dna_world --N 2999999488 --G 256 --blank --ticks 900000 --seed 1 --no-hash --census-every 100 --depth-cap 10 --n-seed 1000000 \
    --genome "ATG TTC GGA GCA AAA ATA CTC" --mu 0.001 --log-births big.log --log-codons --top-genes big.top --out big.json --max-seconds 2400 > big.txt 2> big.err
python tools/dna_seeded_report.py results/dna1_seeded/pilots/b256_35511 --md
python tools/dna_slot_video.py video.map --S 390624 --census video.txt --width 625 --out dna1_seeded_1e8.mp4
```
