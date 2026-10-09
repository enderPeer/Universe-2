# DNA-1 copier sweep: every gene of 2 to 7 codons alone in a small world (2026-10-09)

Spec section 12 of [docs/09](09_dna1_spec.md) names the experiment: every gene of 2 to 7 codons (START plus 1 to 6
codons, 4^3 to 4^18 codon strings) alone in a small world with mu = 0, scored by whether it claims a slot, copies
itself faithfully and divides within its lifetime, and how often. This note records the batch engine that makes the
sweep possible, its verification against the reference simulator, the exhaustive counts for lengths 2 to 7 under the
standard and the uniform codon table, and what the genes that copy actually do. Every number is from a file in
`results/dna1_sweep/` or from the engines' own output; nothing is estimated.

**Headline.** No gene of up to 5 codons copies itself faithfully (16,781,520 strings per code, exhaustive, both codes,
64 and 256 ticks). At 6 codons, under the standard code, exactly **16 of 1,073,741,824 strings** produce a faithful
child (child gene hash equal to the parent's at birth): all sixteen are codon variants of one sequence,
`START R1 ALLOC R3 START INC`, which contains no COPY at all; it rewrites itself with INC during its first four ticks
into `START R3 SKNZ DIVIDE COPY SEEK`, copies that gene once, and stops. None of the 16 is intact (the parent's gene
at birth is not the seeded gene), and under the uniform code no 6-codon string copies faithfully at all. At 7 codons,
exhaustively over all **68,719,476,736 strings** of the standard code, **4,520 strings in 30 operation sequences
copy faithfully, and 2,304 of them, every spelling of the two sequences `START SKZ DIVIDE ALLOC R3 SKNZ COPY` and
`START ALLOC SKZ DIVIDE R3 SKNZ COPY`, copy the seeded gene intact**: the first real self-replicators of DNA-1. Their
loop holds COPY back while the head reads nonzero cells, copies while it walks the blank neighbouring slot, and
divides when the head reaches the copy it has itself written in the child slot. With the sweep's fixed seed the
child draws a slot too far ahead to repeat this within its lifetime, so no world of the sweep shows a grandchild;
under other seeds the same gene reproduces whenever the ALLOC draw gives it the slot two ahead of its own
(9 of 64 seeds) and lineages of depth 3 appear (2 of 64). Fecundity is 1 in every case. Under the uniform code the
same 2^36 strings hold 44 faithful copiers, all of them INC self-modifiers, and no intact one: the intact mechanism
needs the gene's own codons to begin with nonzero letters, which the standard code's spellings of SKZ, DIVIDE and
ALLOC (T, G, G) provide and the uniform code's 21 meaningful codons (all A or C) cannot.

## 1. The experiment

One world per gene: W = 8 slots of G = 256 cells (N = 2,048 cells, 85 usable codons per slot), tape blank (every
cell A), the gene written into slot 0 followed by STOP codons (TAA) up to codon 84, the one trailing cell (256 mod 3
= 1) blank, one processor born in slot 0 with the initial values of spec section 4. I = 16 instructions per tick,
lifetime T = 64 ticks, mu = 0, no damage (D = 0), no inflow (F = 0), seed 1 (the seed fixes the ALLOC draw, so every
gene sees the same sequence of candidate child slots: slot 2 for a request in tick 0, then 7, 6, 5, 4, ...). This
is exactly the world `sim/dna_ref.py --N 2048 --G 256 --blank --n-seed 1 --genome ... --mu 0 --D 0 --F 0` builds.
Every world runs 64 ticks (the seed organism's whole life) and, in a second run, 256 ticks (its children's lives too).
Both codon tables of spec section 3 are swept: the standard code and `--uniform-code` (codons 0..20 are the 21
meanings in table order, 21..63 NOP; START is AAA and STOP is AAC there).

A **string** of length L is START followed by L-1 codons, index `g = sum c_k 64^(L-1-k)` (the last codon is the
least significant digit). A string is a **proper gene** of length L when none of its L-1 codons is a STOP
(61^(L-1) strings under the standard code, 63^(L-1) under the uniform code); the other strings are shorter genes
followed by codons that sit in the slot after the STOP, where the head can read them and INC, FLIP or STA can
rewrite them. Both kinds are swept and counted; the tables below count strings, and the per-gene-length buckets of
the JSON files separate them.

Per world the engine records: **claims** (ALLOC grants to the seed organism), **copied** (cells it wrote into child
slots with COPY or COPYR), **divides** (its DIVIDEs that marked a child), **children** (its children that were
expressible and born), **faithful** (children whose gene hash equals the parent's gene hash at the moment of birth,
spec step 3), **intact** (faithful children born while the parent's gene still hashed as seeded, the analogue of
Universe-1's "original intact at the copy"), **grandchildren** (births whose parent is itself a child: a child
divided in turn), the world totals of births, faithful births and deaths, the processors alive at the end, the
maximum lineage depth and the tick of the first birth. "Seed organism" is the processor of depth 0, which is unique
because inflow is off. The faithful count of the seed organism is its **fecundity**.

## 2. The engine: `gpu/dna_sweep.cu`

One GPU thread runs one world, sequentially, slot by slot, as the reference does: the 512-byte packed strand of the
world lives in shared memory (one bank offset per thread against conflicts), the tick-start snapshot of spec section
4 is kept per slot in local memory and only for the slots that can be written in the tick (slots live or claimed at
the tick start), which reads the same values as the reference's full copy; the gene length of a slot is cached and
recomputed only after a write into that slot. Damage and inflow are off, so the reverse strand is always the
complement of the forward strand and the state hash of spec section 8 is computed from one strand. The device code
for the codon table, the PRNG and the gene functions is copied verbatim from `gpu/dna_world.cu`.

Modes: `--L n --all` (every string of length n), `--L n --random m --rseed s` (m indices
`splitmix64(rseed + i * K1) mod 64^(n-1)`), `--genes FILE` (listed codon strings, the verification mode).
Results are reduced on the GPU into histograms per gene length (block-level shared-memory counters, then one atomic
add per block), every gene selected by `--log-mask` is logged unsampled with its full record (default: a faithful
child or a grandchild), `--records` writes every world's record (small runs), `--hash-ticks` writes the state hash
after every tick of every world (verification), `--chunk-log` writes one accounting line per launch (range, counts,
seconds), and `--chunk-start/--chunk-end` split a sweep over several GPUs (`tools/dna_sweep_merge.py` sums the parts
and checks that the ranges tile). Launches of 2^22 to 2^24 worlds; 32 or 64 threads per block (the shared-memory
tape limits the block to 192 threads for 8 x 256 worlds).

**Prefilter.** A gene whose codons before the first STOP contain neither ALLOC nor INC, FLIP or STA can never claim
a slot nor change its own gene, so its world is static: the record is written without running it. It is exact, and
it was checked: all 16 sweeps of lengths 2 to 5 (both codes, 64 and 256 ticks) were run with and without it and
every histogram counter is identical (`results/dna1_sweep/prefilter_check.tsv`). It removes 48 % of the standard-code
strings and 73 % of the uniform-code strings at length 6.

**SEEK by words.** The soup runs (section 9) were five times faster than the blank runs with the same prefilter,
which located the cost: SEEK scans up to 3G = 768 cells for the complement of its template, cell by cell under the
snapshot rule, and on a blank tape it finds nothing and scans them all; genes with SEEK in a loop do that 16 times
per tick. The engine now tests 16 cells per 32-bit word of the packed strand with lane-wise bit operations (the
same source array per word as the per-cell rule, since G is a multiple of 16 and a word never straddles a slot; the
two cells after the word come from the next word of the ring), and keeps the per-cell loop as `--slow-seek` and for
other slot sizes. Same result: identical histograms and identical logs row for row on the full length-6 standard
sweep and on lengths 5 (both codes, fast against slow), and the whole verification suite of section 3 again.

**Speed on the RTX 4090 (adler40, GPU 0):** with the per-cell SEEK 3.3 million worlds per second for the standard
code at 64 threads per block and 4.8 million at 32 (the one-warp blocks release their shared memory as soon as
their 32 worlds end); with SEEK by words 14 million per second on length 6 and about 20 million on length 7
(the length-6 standard sweep in 75 s instead of 253 s; the files in `results/dna1_sweep/` for lengths 2 to 6 are
from the per-cell build, the length-7 files from the word build). The uniform code runs about twice as fast (more
NOP genes, more prefiltered). Each RTX 3060 of knecht24 does a fifth to a quarter of the 4090's rate; all three
give the same counts as the 4090 on the full length-5 sweep with both builds. A random gene order (`--random`) is
7 times slower than the exhaustive order, because the 32 worlds of a warp then execute 32 unrelated genes; the
7-codon space was therefore swept exhaustively rather than sampled.

## 3. Verification against the reference

`tools/dna_sweep_verify.py` runs a sample of genes through `sim/dna_ref.py` (one process per gene, `--hash-every 1`)
and through the engine (`--genes --hash-ticks --records`) and compares, per gene, the state hash of spec section 8
after **every** tick and the reference's done-line counters (births, divide_births, faithful, deaths, live, and the
two counters added to the reference for this purpose, claims and copied_cells; spec section 13 "Reference-only
output") with the engine's record. The sample mixes uniform random genes of 2 to 7 codons, genes drawn from the
engine's own logs (claimers, dividers, every 5-codon gene that produced a child), the hand-written organism of spec
section 11, both codes, 64 and 256 ticks, the blank tape and the random soup of spec section 6, and two other world
shapes (4 x 128 cells; 16 x 64 cells with lifetime 20, where G = 64 is not a multiple of 3).

| set | code | ticks | world | genes | genes identical | tick hashes compared | mismatches | genes with claims / copies / divides / children |
|---|---|---:|---|---:|---:|---:|---:|---|
| rand64 | standard | 64 | 8x256 blank | 301 | 301 | 19,264 | 0 | 62 / 16 / 3 / 0 |
| std64_logs | standard | 64 | 8x256 blank | 1,807 | 1,807 | 115,648 | 0 | 1,724 / 1,630 / 1,618 / 1,538 |
| std256 | standard | 256 | 8x256 blank | 1,697 | 1,697 | 434,432 | 0 | 1,618 / 1,601 / 1,596 / 1,536 |
| uni64 | uniform | 64 | 8x256 blank | 365 | 365 | 23,360 | 0 | 224 / 175 / 174 / 95 |
| uni256 | uniform | 256 | 8x256 blank | 255 | 255 | 65,280 | 0 | 156 / 154 / 154 / 97 |
| soup64 | standard | 64 | 8x256 soup | 211 | 211 | 13,504 | 0 | 90 / 68 / 60 / 20 |
| soup256 | standard | 256 | 8x256 soup | 141 | 141 | 36,096 | 0 | 59 / 43 / 40 / 8 |
| w4g128 | standard | 64 | 4x128 blank | 201 | 201 | 12,864 | 0 | 88 / 60 / 51 / 1 |
| w16g64 | standard | 100 | 16x64 blank, T = 20 | 201 | 201 | 20,100 | 0 | 78 / 58 / 51 / 0 |
| **total** | | | | **5,179** | **5,179** | **740,548** | **0** | |

Every tick hash and every counter identical (`results/dna1_sweep/verify/verify_v_*.json`). The whole suite was run
again with the final build (SEEK by words, section 2), plus three sets it did not have: 300 random genes with one or
two SEEK codons forced in, on the blank tape and in the soup, and 357 genes drawn from all 30 faithful 7-codon
sequences (up to 12 spellings each) and the 6-codon copiers, 256 ticks:

| final build | sets | genes | tick hashes compared | mismatches |
|---|---:|---:|---:|---:|
| the nine sets above | 9 | 5,179 | 740,548 | 0 |
| seek_blank, seek_soup | 2 | 602 | 38,528 | 0 |
| L7all (every faithful 7-codon sequence) + L7faithful (first 60 found) | 2 | 438 | 112,128 | 0 |
| **total** | **13** | **6,219** | **891,204** | **0** |

(`verify/verify_v5_*.json`.) Beyond the reference comparison: the builds of the engine (before and after the record
gained the intact field, before and after SEEK by words) give identical histograms on the length-5 and length-6
sweeps and identical logs row for row, and the knecht24 RTX 3060s give the same histograms as the RTX 4090.

Caveat: a per-tick state hash proves that the engine's world is the reference's world for those genes; the sweep
runs 10^9 other genes through the same code path, and the prefilter and the early exit of a world without live
processors are the only places where the sweep does something the verified path does not (both argued exact above,
the prefilter also measured). One build error was caught by the timing, not by the tests: a block of fewer than 64
threads left half of the shared-memory codon table uninitialised, which made 32-thread runs 30 times "faster";
every published number comes from runs with 64 threads per block or from the fixed build.

## 4. Results

Blank 8 x 256 world, seed 1, T = 64, I = 16. The seed organism's own statistics (claims, copies, divides, children,
faithful) are identical between the 64-tick and the 256-tick runs, as they must be (it dies at the end of tick 63
either way); the 256-tick run adds what its children do.

### Standard code

| length (codons incl. START) | strings swept | proper genes | claimed a slot | copied a cell | divided | expressible child | faithful child | intact | grandchild | seconds (64-tick run) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 64 | 61 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 |
| 3 | 4,096 | 3,721 | 484 | 64 | 0 | 0 | 0 | 0 | 0 | 0.1 |
| 4 | 262,144 | 226,981 | 43,645 | 9,224 | 768 | 0 | 0 | 0 | 0 | 0.2 |
| 5 | 16,777,216 | 13,845,841 | 3,482,174 | 986,494 | 143,816 | 1,536 | 0 | 0 | 0 | 5.4 |
| 6 | 1,073,741,824 | 844,596,301 | 263,163,833 | 80,285,144 | 14,889,581 | 241,736 | **16** | **0** | 0 | 253.5 |
| 7 | 68,719,476,736 | 51,520,374,361 | 19,077,034,165 | 6,218,919,714 | 1,365,115,826 | 21,387,976 | **4,520** | **2,304** | 0 | 4,843 (sum over 4 GPUs, 28 min wall) |

The length-7 run is the 256-tick run only (the seed organism's columns do not depend on the tick count); it was
split into 13 parts over the RTX 4090 (10 parts, 1,719 s) and the three RTX 3060 of knecht24 (3 parts, 3,122 s) and
merged with `tools/dna_sweep_merge.py`, which checks that the parts tile the 2^36 strings exactly once
(`results/dna1_sweep/L7_standard_t256.json`, `merged_from`; per-launch accounting for all 4,096 launches in
`L7_standard_t256_chunks.tsv.gz`; every faithful gene in `L7_standard_t256_log_faithful.tsv.gz`).

### Uniform code

| length (codons incl. START) | strings swept | proper genes | claimed a slot | copied a cell | divided | expressible child | faithful child | intact | grandchild | seconds (64-tick run) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 64 | 63 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 |
| 3 | 4,096 | 3,969 | 124 | 4 | 0 | 0 | 0 | 0 | 0 | 0.1 |
| 4 | 262,144 | 250,047 | 11,700 | 700 | 12 | 10 | 0 | 0 | 0 | 0.1 |
| 5 | 16,777,216 | 15,752,961 | 983,958 | 87,436 | 2,820 | 94 | 0 | 0 | 0 | 2.1 |
| 6 | 1,073,741,824 | 992,436,543 | 77,729,504 | 8,387,572 | 395,420 | 24,081 | 0 | 0 | 0 | 76.0 |
| 7 | 68,719,476,736 | 62,523,502,209 | 5,894,162,844 | 755,457,189 | 46,232,358 | 5,773,121 | **44** | **0** | **3** | 1,156 (RTX 4090 alone, 4 parts) |

The length-2 to 5 runs used 64 threads per block, the length-6 runs 32 (the length-6 files in `results/dna1_sweep/`
are the reruns with the intact field; the first build's runs at 64 threads per block gave the same counts in 324 s
and 129 s).

Children per seed organism (256-tick runs, all children, faithful or not): standard code length 6: 240,912 genes with
one child, 824 with two, none with more; length 7: 18,999,798 with one, 170,680 with two, 19,378 with three,
170,636 with four, 253,398 with five and the rest of the 21,387,976 with more; uniform code length 6: 7,165 with
one, 114 with two, and the rest of the 24,081 up to seven (a blank slot is expressible under the uniform code, see
section 8, so a gene that claims, copies six cells anywhere and divides fills every slot it draws). Fecundity
(faithful children) is 1 for every one of the 16 + 4,520 standard-code copiers and the 44 uniform-code ones, and 0
for everything else. Maximum lineage depth is 1 in every blank-tape world of the standard code: in the 2 x 2 x 16.8
million worlds of up to 5 codons, the 2 x 2 x 1.07 billion worlds of 6 codons and the 68.7 billion worlds of 7
codons, no child ever divided (with this seed; section 6). The only depth-2 lineages on the blank tape are three
7-codon strings of the uniform code (section 8).

By effective gene length inside the length-7 sweep: the 4,520 faithful strings are 4,376 proper 7-codon genes, 96
strings whose gene is 6 codons followed by one codon after the STOP, and 48 whose gene is 4 codons followed by two
(`START R3 STA COPY STOP ALLOC REWIND`: STA writes the accumulator, 0, over its own STOP and grows into a copier);
all 2,304 intact ones are proper 7-codon genes.

## 5. What the genes that produce children do

**Length 5, standard code: 1,536 strings, one child each, born in tick 2, none faithful.** They are exactly the
strings `START X Y Z W` with X one of the four codons `AAT CAT GAT TAT` (FLIP, STA, L3, LDA: any codon whose second
and third letters are A and T), and Y Z W one of the four orders of ALLOC, COPY, DIVIDE in which the codon after X
begins with G (ALLOC `GC?` or DIVIDE `GG?`, never COPY): 4 x 4 orders x (4 ALLOC x 6 COPY x 4 DIVIDE codons) = 1,536.
The mechanism is a frame shift. In tick 0 the ALLOC is only a request; the four COPYs of the tick run without a
child and advance the template pointer P by 4 (COPY always moves P and Q, spec section 4). The claim is granted at
the end of tick 0, and from tick 1 the organism copies the parent from cell 4 onward: cells 4, 5, 6 are the A and T
of X and the G of the next codon, so the child's codon 0 is ATG = START, and after 6 copied cells the DIVIDE of tick
2 finds an expressible child. Its gene is START, one accidental codon, then 83 R3 codons (AAA, the blank slot has no
STOP), so it never hashes like its parent and, having neither ALLOC nor a gene that ends, does nothing in its 64
ticks (trace: `python tools/dna_trace.py "ATG AAT GCA CTA GGA" --quiet-static`).

**Length 4 and 6, same story at other alignments.** The 768 four-codon dividers (`START {ALLOC, COPY, DIVIDE}` in
every order, and the COPYR variants) divide but never produce an expressible child, because no cell triple in a
3-codon gene reads ATG at the offset the frame shift selects. At length 6 the 241,736 child producers are dominated by
R1 and SKNZ loops (`START ALLOC R1 SKNZ COPY DIVIDE` and its permutations, 1,728 codon variants each), and 824 of
them manage two children; all children are shifted or truncated copies.

**The 16 faithful copiers of length 6.** `ATG AGC GCA AAA ATG CAA` and 15 codon variants (R1 as AGC or AGT, ALLOC as
GCA, GCG or GCT, R3 as AAA or AAG, INC as CAA or CAG: the 16 combinations that survive the self-modification below;
list in `results/dna1_sweep/L6_standard_t256_log_faithful_or_multi.tsv`). The sequence is `START R1 ALLOC R3 START
INC`: it has no COPY and no DIVIDE. Trace (`python tools/dna_trace.py "ATG AGC GCA AAA ATG CAA"`), gene of slot 0
after each tick:

| tick | gene of slot 0 after the tick | what happened |
|---|---|---|
| seeded | `ATG AGC GCA AAA ATG CAA` = START R1 ALLOC R3 START INC | H = 0, P = 0 |
| 0 | `ATG ATC GGA ACA ATG CAA` = START SKNZ DIVIDE SETQ START INC | R1 moves H to cell 1, ALLOC requests a slot, R3 moves H to cell 4, INC turns the G of AGC into T: the R1 codon is now ATC = SKNZ. The loop continues with SKNZ (cell 4 is T, so it skips the ALLOC codon), R3, INC on cell 7 (GCA to GGA = DIVIDE), SKNZ, R3, INC on cell 10 (AAA to ACA = SETQ). The claim of slot 2 is granted at the end of the tick; the ALLOC codon no longer exists. |
| 1 | `ATG ATC GTC CGA ATG CAA` = START SKNZ JNZB L1 START INC | the loop keeps INC-ing cells of its own gene while SKNZ skips over the codon under the head |
| 2 | `ATG AAG ATC CGA ATG CAA` = START R3 SKNZ L1 START INC | |
| 3 | `ATG AAG ATC GGA CTG GAA` = START R3 SKNZ DIVIDE COPY SEEK | the last INC turned CAA into GAA = SEEK; the gene is now stable. H = 18, 2 cells copied (P = 2, Q = 514). |
| 4 to 22 | unchanged | each tick: 4 passes of R3, SKNZ, COPY, SEEK (SEEK's template would be codon 6, beyond the gene, so it is a NOP). H advances 3 cells per pass over the STOP padding of the own slot, whose cells 3k are T, so SKNZ always skips the DIVIDE; 4 cells are copied per tick (P, Q and copied grow by 4). |
| 23 | unchanged | after passes 1 and 2 (H = 249, 252, both T) pass 3 moves H to cell 255, the one trailing cell of the slot, which is blank (A = 0): SKNZ does not skip, DIVIDE runs with copied = 80 >= 6 and marks slot 2. Birth at step 3: slot 2 holds cells 0..79 of the parent, the 6 codons, the STOP and 19 padding codons; parent and child hash `ccdb8c577eab3922`, faithful. |
| 24 to 63 | unchanged | the parent has no ALLOC, so it never claims again (fecundity 1); the child has the same gene, hence no ALLOC either: it executes COPY without a child until it dies at age 64. Both die; the world is empty from tick 87 on. |

So the loop terminates by geometry, not by a test on the copy: the head walks the own slot at stride 3 reading the
first cell of every STOP codon (T, nonzero), and the first zero it meets is the trailing cell of the slot. The copy
is faithful because copying began only after the gene had finished rewriting itself (P was still 0 when the first
real COPY ran, tick 3, since COPY did not exist in the gene before), which is exactly what the frame-shifted
5-codon genes cannot arrange: any gene that contains COPY from the start runs it in the claim tick and misaligns P.
That is the structural reason the first faithful copier is a self-modifier: in this machine a gene needs either a
conditional that holds COPY back until the claim is granted (at least ALLOC, a test, COPY, DIVIDE, a terminating
test, and something to move the head: 6 codons after START) or a gene that creates its COPY after the claim, as
these 16 do in 5 codons after START. It is a copier of its rewritten self, not of the seeded string: 0 intact.

**Absence proof for lengths 2 to 5.** The sweeps are exhaustive over all 64 + 4,096 + 262,144 + 16,777,216 strings per
code, run for 64 and for 256 ticks, with and without the prefilter, and the faithful column is 0 in every bucket of
every run (`results/dna1_sweep/L[2-5]_*.json`). Under the standard code no 5-codon string even produces a child
that hashes like its parent; under the uniform code no string of up to 6 codons does.

**The 4,520 faithful copiers of length 7** fall into 30 operation sequences (`L7_standard_t256_log_faithful.tsv.gz`,
column `ops`). 28 of them, 2,216 strings, are self-modifiers like the 6-codon one: every one contains INC, FLIP or
STA and copies the gene it has rewritten (`START R3 STA COPY REWIND R1 L1`, 192 spellings, writes the accumulator's 0
over its own codons; `START R1 ALLOC R3 INC SKNZ COPY`, 156; `START INC FLIP R3 L1 SKZ ALLOC`, 128; the 6-codon
sequence with a seventh codon appended, 48 + 48 + 16; and so on). The other two sequences are the intact copiers of
section 6. Every one of the 4,520 has fecundity 1 and no grandchild in the sweep.

## 6. The shortest copier

- **Shortest faithful copier (child equals parent at birth): 6 codons**, `ATG AGC GCA AAA ATG CAA` and 15 codon
  variants, standard code, fecundity 1, parent rewritten, child sterile. Found exhaustively: 16 of 2^30 strings,
  none shorter.
- **Shortest intact copier (child equals the seeded gene): 7 codons**, `START SKZ DIVIDE ALLOC R3 SKNZ COPY` and
  `START ALLOC SKZ DIVIDE R3 SKNZ COPY`, in every one of their 1,152 + 1,152 spellings (SKZ 2 codons x DIVIDE 4 x
  ALLOC 4 x R3 2 x SKNZ 3 x COPY 6), standard code, fecundity 1. Found exhaustively: 2,304 of 2^36 strings, none
  shorter (0 of all strings of up to 6 codons under both codes).
- **Shortest heritable copier (a child that divides in turn): the same 7-codon gene**, under the right ALLOC draw
  (below). In the sweep itself, with seed 1, no lineage reached depth 2 up to and including length 7.

**How the intact copier works.** Trace of `ATG TTC GGA GCA AAA ATA CTC` (`python tools/dna_trace.py "ATG TTC GGA
GCA AAA ATA CTC" --ticks 256`); the loop body after START is SKZ, DIVIDE, ALLOC, R3, SKNZ, COPY, 6 codons, so a
tick of 16 instructions runs it about three times. The two tests read the cell under the head H, which R3 moves
3 cells per pass, and they guard the two actions: SKZ skips DIVIDE when the cell is 0, SKNZ skips COPY when it is
not 0. P and Q are never touched except by COPY.

| tick | state after the tick | what happened |
|---|---|---|
| 0 | H = 9, P = 0, child = slot 2, copied 0 | SKZ reads cell 0 (A = 0) and skips DIVIDE; ALLOC requests a slot; R3 moves H to cell 3, the T of TTC: SKNZ skips COPY. Passes 2 and 3 land on cell 6 (G) and 9 (G): COPY skipped again, P stays 0. The claim of slot 2 is granted at the end of the tick. |
| 1 | H = 18, P = 2, copied 2 | H reaches cells 12 (AAA: A = 0) and 15 (ATA: A = 0): two COPYs, cells 0 and 1 go to the child; then cell 18 (TAA: T): skipped. |
| 2 to 25 | H walks 9 cells per tick through the STOP padding of slot 0; P stays 2 | every cell 3k of the padding is the T of a TAA, so SKNZ skips every COPY; SKZ does not skip, DIVIDE runs every pass but is a NOP because copied = 2 < 6. |
| 26 to 52 | H walks slot 1 (blank); P and copied grow by about 3 per tick: 4, 7, 10, ..., 87 | every cell read is 0: SKZ skips DIVIDE, SKNZ does not skip, COPY runs every pass and writes cells 2, 3, 4, ... of the parent into the child slot. |
| 53 | birth in slot 2, hashes equal, `d63dbd3ebc6a4cbb` | pass 1 moves H to cell 513, the second cell of the child slot, which holds the T of the ATG the parent copied there at tick 1: SKZ does not skip, DIVIDE runs with copied = 87 >= 6. The child holds cells 0..86 of the parent: the 7 codons, the STOP, 21 more STOP codons. The parent immediately claims slot 4 and keeps copying from P = 88, STOP padding only. |
| 54 to 63 | parent dies at age 64 | its second child never gets a START. |
| 53 to 116 | the child runs the same program in slot 2 | it claims slot 7 at tick 55 (the ALLOC draw for slot 2 at tick 55); its head would reach slot 7 after walking slots 2 to 6, about 130 ticks, and it dies at tick 116 with 3 cells copied. No grandchild. |

The loop terminates on something the copy changed: the head meets the copy's own first codon. That is the design
docs/10 section 6 asked for ("H can be placed in the child slot ahead of Q, where COPY will eventually write a 0"),
found here in the opposite polarity (the head arrives at a cell the copy made nonzero), and it is why the copier
needs exactly the slot two ahead of its own: one slot of padding to wait out the claim and let the alignment hold,
one blank slot to copy across, and the child slot as the stop signal. The 6-codon one-shot copier exits on its own
slot's trailing cell instead and needs no particular child slot, which is why it works under every seed.

**Heritability depends on the draw.** The sweep uses seed 1 for every gene, so the seed organism always draws slot 2
in tick 0 (the draw depends only on tick and slot), which is the one slot that makes this gene reproduce, and its
child in slot 2 always draws slot 7, which is out of reach. Re-running the three genes under seeds 1 to 64 for 512
ticks (`dna_sweep --genes --seed s`, `results/dna1_sweep/L7_intact_seeds_W8.tsv` and `_W16.tsv`):

| gene | 8 slots: seeds with a faithful child / with a grandchild / max depth | 16 slots: the same |
|---|---|---|
| `ATG TTC GGA GCA AAA ATA CTC` (intact, SKZ DIVIDE ALLOC R3 SKNZ COPY) | 9 / 2 / 3 | 5 / 2 / 3 |
| `ATG GCA TTC GGA AAA ATA CTC` (intact, ALLOC SKZ DIVIDE R3 SKNZ COPY) | 9 / 2 / 3 | 5 / 2 / 3 |
| `ATG AGC GCA AAA ATG CAA` (one-shot, 6 codons) | 64 / 0 / 1 | 64 / 0 / 1 |

9 of 64 is 1 in 7, the chance that a draw among the 7 other slots is the slot two ahead; a grandchild needs the
child's draw to be its slot two ahead as well, and in 2 of the 64 seeds the lineage reached depth 3. So the 7-codon
gene is a heritable replicator with a 1-in-7 chance per generation in this world, and fecundity 1 because after the
division P points into the padding and nothing rewinds it; a REWIND would cost an eighth codon.

## 7. Comparison with the predecessors

| machine | space | copiers (copy equals the program as it is at the copy) | original intact at the copy | fecundity | source |
|---|---|---|---|---|---|
| NANO, 5 bytes | 2^40 | 316 replicators (2.9 x 10^-10) | 0 | 1 | docs/01, Dimension42 alife toolkit |
| Universe-1 crawler, 4-bit words | 2^32 | 141 (3.3 x 10^-8) | 0 | 1 | docs/01, exp06b |
| Universe-1 crawler, 5-bit words (5 bytes) | 2^40 | 21,818 (2.0 x 10^-8) | **3** (2.7 x 10^-12) | 1 | docs/01, exp06c |
| DNA-1, 7 codons (36 bits after START), standard code | 2^36 | **4,520** (6.6 x 10^-8) | **2,304** (3.4 x 10^-8) | 1 | this note |
| DNA-1, 7 codons, uniform code | 2^36 | 44 (6.4 x 10^-10) | 0 | 1 | this note |
| DNA-1, 6 codons (30 bits after START), standard code | 2^30 | **16** (1.5 x 10^-8) | 0 | 1 | this note |
| DNA-1, up to 5 codons | 2^24 + 2^18 + 2^12 + 2^6 | 0 | 0 | | this note |
| DNA-1 random world, 3 x 10^9 cells, 600,000 ticks | 6,585,887 births, 204,881 by DIVIDE, genes of up to 341 random codons | 3 faithful births (1 in 70,000 DIVIDE births, 1 in 2.2 million births) | not recorded | none divided again | docs/10 section 7 |

Read: at comparable program sizes (30 to 40 bits) all three machines put copiers at 10^-8 to 10^-10 per program,
and every copier found has fecundity one. Where DNA-1 differs is the intact copier: the crawler needed 2^40
programs for 3 intact ones (2.7 x 10^-12), NANO had none in 2^40, DNA-1 has 2,304 in 2^36 (3.4 x 10^-8), four
orders of magnitude denser, and they are heritable under the right draw. DNA-1 was built so that the copy pointers
live in the machine (P, Q, H) instead of in the program, which was the predecessors' reason for fecundity one
(docs/08 section 2); the sweep shows what that buys and what it does not. It buys intact copying: the 7-codon gene
never touches its own code, and its child is the same gene. It does not buy fecundity: after one division P points
past the gene and nothing short of an eighth codon (REWIND) brings it back, and the claim's timing (granted at the
end of the tick, so COPY must be held back for a tick) and the exit test cost the other codons. The random world's
3 faithful births in 204,881 DIVIDE births (docs/10) are consistent with this: with random genes of tens of codons
the faithful rate per division is 1.5 x 10^-5, and no faithful child divided again there either.

## 8. Standard code against uniform code

At every length the standard code has more strings that claim (length 6: 263 million against 78 million), copy
(80 million against 8.4 million), divide (14.9 million against 0.4 million) and produce a child (241,736 against
24,081), and it has the 16 faithful copiers where the uniform code has none. The reason is the degeneracy the code
was designed with (spec section 3): ALLOC, COPY and DIVIDE together own 14 of the 64 codons (22 %) under the standard
code and 3 of 64 (4.7 %) under the uniform code, where 43 codons are NOP, so a random standard-code string is far
more likely to contain the replication primitives, and each operation sequence appears in many codon spellings (the
16 copiers are one sequence in 16 spellings; the 1,536 frame-shift children are 16 sequences in 96 spellings each).
Per *operation sequence* the picture inverts in one respect: under the uniform code more distinct operation
sequences of length 5 produce a child (the 94 strings are about 80 sequences, the 1,536 standard-code strings are
16), because of the blank-slot quirk: a blank slot reads AAA AAA ..., which under the uniform code is START with no
STOP, so every slot is expressible before anything is written into it, and any gene that claims, copies 6 cells and
divides has a child. Under the standard code the child needs an ATG at its cell 0, which only a copy aligned on a
START (or a lucky frame shift) provides. The soup runs (section 9) remove this quirk.

So degeneracy helps copier density per string, by about an order of magnitude at every length, and it is also what
makes the self-modifying route possible: the 16 copiers exist because INC moves a codon from one 4- or 6-codon family
to a neighbouring one (AGC to ATC, GCA to GGA, CAA to GAA), and such neighbours are what a degenerate code arranges.

**At 7 codons the two codes part ways** (`L7_uniform_t256.json`, `L7_uniform_t256_log_faithful.tsv.gz`): the uniform
code has 44 faithful copiers in 2^36 strings against the standard code's 4,520, and **no intact copier against
2,304**. The 44 are all INC self-modifiers (`START R3 NOP NOP NOP NOP INC` and relatives: R3 walks the head over
the gene, INC rewrites codons into ALLOC, COPY and DIVIDE, 150 to 200 cells get copied, the child is the rewritten
gene). The intact mechanism of section 6 is not available to the uniform code at this length for a reason of
spelling, not of operations: during the claim tick the head reads the first cell of codons 1, 2 and 3 of the gene
and must find them nonzero so that SKNZ holds COPY back until the claim is granted. The standard code spells SKZ as
TT?, DIVIDE as GG? and ALLOC as GC?: first letters T, G, G. The uniform code's 21 meaningful codons are the values 0
to 20, all beginning with A (0 to 15) or C (16 to 20); the sequence `START SKZ DIVIDE ALLOC R3 SKNZ COPY` is spelled
`AAA AGT AGC AGA CAA AGG AAG` there, the head reads A = 0 at cells 3, 6 and 9, COPY runs in the claim tick, and P is
misaligned by three cells for good. A uniform-code gene would need three nonzero-leading codons (NOPs from the
unused range, or the C-leading R3, L3, SEEK, COPYR, REWIND) in front of the six-codon loop: at least 9 codons after
START. The letters the code assigns to its operations are part of the program, because the head reads them.

The uniform code does have the **only depth-2 lineages on the blank tape**: `AAA AAA AGA AAG CAA AGT AGC` =
`START START ALLOC COPY R3 SKZ DIVIDE` and two permutations of its loop (`ALLOC R3 SKZ DIVIDE COPY`, `ALLOC R3 COPY
SKZ DIVIDE`), each with one child, 0 faithful, 1 grandchild. The trace (`python tools/dna_trace.py "AAA AAA AGA AAG
CAA AGT AGC" --uniform-code --ticks 256`) shows a shifted copy that happens to be the same program: the three COPYs
of the claim tick advance P to cell 3, so the child receives the parent from its second codon on, and because that
second codon is a START (a NOP inside the gene) the child's gene is `START ALLOC COPY R3 SKZ DIVIDE STOP`, the same
loop minus one NOP, a different hash. It divides in turn at tick 99 (into slot 7, with yet another gene), and the
lineage ends there. Functionally heritable for one generation, formally unfaithful: a doubled START is a one-codon
buffer against the claim-tick shift, and the uniform code, where START is the blank cell's codon AAA, is the code in
which doubled STARTs are cheap.

## 9. Soup background

The same sweeps with the random soup of spec section 6 (seed 1) instead of the blank tape: the other seven slots and
the trailing cell of slot 0 hold random symbols, the gene and its STOP padding are written as before. Lengths 2 to 6,
both codes, 256 ticks (`results/dna1_sweep/L*_t256_soup.json`).

| code | length | strings | claimed | copied a cell | divided | expressible child | faithful | intact | grandchild | children per gene (1 / 2 / 3 / 4-7 / 8+) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| standard | 4 | 262,144 | 43,827 | 9,225 | 768 | 0 | 0 | 0 | 0 | |
| standard | 5 | 16,777,216 | 3,490,534 | 980,102 | 143,816 | 37,632 | 0 | 0 | 0 | 37,056 / 576 / 0 / 0 / 0 |
| standard | 6 | 1,073,741,824 | 263,072,256 | 78,380,143 | 14,650,872 | 1,601,252 | **16** | 0 | 0 | 1,540,306 / 57,074 / 3,872 / 0 / 0 |
| uniform | 4 | 262,144 | 11,706 | 700 | 12 | 7 | 0 | 0 | 0 | 7 / 0 / 0 / 0 / 0 |
| uniform | 5 | 16,777,216 | 984,679 | 87,427 | 2,820 | 66 | 0 | 0 | 0 | 42 / 0 / 0 / 0 / 24 |
| uniform | 6 | 1,073,741,824 | 77,775,757 | 8,371,505 | 395,032 | 95,567 | 0 | 0 | **1** | 88,977 / 378 / 14 / 1,307 / 4,891 |

(Lengths 2 and 3: no child under either code, as on the blank tape.) What changes against the blank tape:

- **The 16 faithful copiers are the same 16 strings** of `START R1 ALLOC R3 START INC`, now copying 86 instead of 80
  cells before the division at tick 25 (the trailing cell 255 is a random symbol, so the head's exit comes two
  passes later, at the first zero cell of the neighbouring slot), still fecundity 1, still 0 intact. No new copier
  appears when the world is random, and none disappears.
- **Children are 7 times more frequent under the standard code** (1,601,252 against 241,736 at length 6) and 4 times
  under the uniform code: a copied fragment lands in random content that supplies the STOP a blank slot lacks, so
  more shifted copies are expressible. None of them hashes like its parent.
- **The uniform code's blank-slot advantage disappears**: on the blank tape it had 24,081 child producers at length
  6 against the standard code's 241,736 (1 : 10); in the soup it has 95,567 against 1,601,252 (1 : 17). The
  children-per-gene tail of the uniform code (6,198 genes with 4 or more children, up to 11) comes from genes
  that divide into every slot they draw, each child a different random gene that soon dies and frees its slot.
- **The only lineage of depth 2 in all the sweeps** is in the uniform-code soup: `AAA TGA CAA TAG TGC ATG` = START
  NOP R3 NOP NOP INC, a gene with neither ALLOC nor COPY that rewrites itself with INC, has 2 children of its own
  and 1 grandchild, none faithful (`tools/dna_trace.py "AAA TGA CAA TAG TGC ATG" --uniform-code --soup --ticks 256`).
  It is the counterpart of the standard code's one-shot copier: self-modification is the only route to replication
  that 6 codons afford, under either code and on either background.

## 10. Caveats

- The world is small and specific: 8 slots of 256 cells, the gene padded with STOP codons, blank elsewhere, seed 1
  for the ALLOC draw. A copier that needs a particular candidate slot, a longer slot or a different padding is not
  found by this sweep; the 16 copiers depend on the trailing blank cell of a 256-cell slot (G mod 3 = 1) for their
  exit.
- Fecundity counts the seed organism's faithful children within its 64-tick life. A gene that would copy itself
  more slowly than that is not a copier here.
- Faithful is defined by the gene hash at the moment of birth, as the world engine defines it; a gene that rewrites
  itself before copying counts as faithful but not intact. Both numbers are reported.
- The uniform code's blank slots are expressible (AAA = START, no STOP), which inflates its child counts on the blank
  tape; the soup runs are the fair comparison for children, the faithful counts are unaffected (0 in both settings).
- Every count is exhaustive; length 7 was run once (256 ticks) and split over four GPUs of two kinds, with the parts
  merged and their coverage checked. The length-7 standard-code sweep was not repeated without the prefilter
  (lengths 2 to 5 were) and not repeated at 64 ticks (the seed organism's columns are tick-independent by
  construction and were so for lengths 2 to 6).
- One seed. Every gene sees the same ALLOC draws; for the 7-codon intact copier that draw happens to be the one
  that lets it reproduce once and then denies its child. The counts are exact for this world, and the seed test of
  section 6 shows how the heritability depends on the draw; a sweep over seeds, or a world rule that lets a gene
  choose or find its child slot, is the next experiment.
- Verification covers 5,862 genes tick by tick with the final build (and 5,179 with the first), including 357
  genes drawn from all 30 faithful 7-codon sequences; the prefilter and the empty-world exit are argued exact and
  the prefilter was measured on 16 full sweeps.

## 11. Files and reproduction

`results/dna1_sweep/`: `L{2..6}_{standard,uniform}_t{64,256}.json` and `L7_{standard,uniform}_t256.json`
(summary and histograms per gene length; the length-7 files are merges of 13 and 4 parts, `merged_from`),
`L{2..6}_*_t256_soup.json` (soup background), `*_chunks.tsv[.gz]` (per-launch accounting, 4,096 launches for each
length-7 sweep), `*_nopf.json` (prefilter off), compact logs (`L2`, `L3` full; `L4` dividers; `L5` child producers;
`L6` faithful copiers and multi-child genes; `L7_*_log_faithful.tsv.gz` every faithful gene of length 7),
`L7_intact_seeds_W{8,16}.tsv` (the seed test of section 6), `prefilter_check.tsv`, `verify/verify_v_*.json` (first
build) and `verify/verify_v5_*.json` (final build). The full logs of every claimer and child producer (up to 240 MB)
stay on adler40 in `~/universe2/dna/sweep/res*/` and the length-7 parts on adler40 and knecht24 in `res7/`.

```bash
nvcc -O3 -std=c++17 -arch=sm_89 -o dna_sweep gpu/dna_sweep.cu
./dna_sweep --L 6 --all --ticks 256 --tpb 32 --chunk 16777216 --log-mask 0x07 --out L6.json --log L6_log.tsv --chunk-log L6_chunks.tsv
./dna_sweep --L 6 --all --ticks 256 --uniform-code --out L6u.json --log L6u_log.tsv
./dna_sweep --L 7 --all --ticks 256 --tpb 32 --chunk 16777216 --chunk-start 0 --chunk-end 4815060992 --out L7_part0.json --log L7_part0_log.tsv   # one of 13 parts
python tools/dna_sweep_merge.py L7_standard_t256.json L7_part*.json
python3 tools/dna_sweep_verify.py --ref sim/dna_ref.py --sweep ./dna_sweep --workdir v --n-random 300 --rseed 11 --ticks 64
python tools/dna_trace.py "ATG TTC GGA GCA AAA ATA CTC" --ticks 256
python tools/dna_sweep_report.py results/dna1_sweep
```
