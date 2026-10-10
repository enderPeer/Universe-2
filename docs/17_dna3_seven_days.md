# DNA-3: the seven days (2026-10-10)

DNA-2 ([docs/15](15_dna2.md)) is a chemistry in which replicators are common (one 7-codon gene in 6,400 founds a
lineage), populations persist for 17,923 generations, and an unseeded world makes its own replicator. What it lacks
is a world: children land in random slots, so there is no space; nothing costs anything, so there is no economy;
the only interaction is competition for empty slots, and the only thing rewarded is copying, so evolution drifts
through synonyms. DNA-3 builds that world in the order of Genesis 1, one layer per day, each layer a delta to DNA-2
behind a flag, each with an acceptance test.

Days 1 to 4 are specified here at the precision of [docs/09](09_dna1_spec.md) (bit-exact: a second implementation
can be written from this text and must agree hash for hash). **Days 1, 2 and 3 are implemented** in the reference
(`sim/dna_ref.py`) and the engine (`gpu/dna_world.cu`) behind `--days` / `--dna3` and verified tick by tick (section
10); their first worlds are in section 11. Day 4 is specified exactly but not implemented. Days 5, 6 and 7 are
specified at design level with predictions. Section 12 lists the decisions that need the user.

## 1. The mapping

| day | Genesis 1 | machine layer | what it adds that DNA-2 cannot have | state |
|---|---|---|---|---|
| 1 | light, separated from darkness; evening and morning | a clock and a light cycle; **energy**: light fills a per-organism store, every instruction burns one unit, the store sets how fast an organism runs | time with a phase; a resource; rate of living set by income | implemented, verified |
| 2 | the firmament, dividing the waters above from the waters below | a **2D toroidal grid** of slots; a child is placed in a free neighbouring slot; two rows of barrier slots (the firmament) cut the torus into two zones that no lineage can cross | space; lineage patches; isolated populations (allopatry) | implemented, verified |
| 3 | waters gathered, dry land appears; the earth brings forth plants | **terrain** (sea, land) with different light and inflow; **HARVEST** (photosynthesis) as a new operation; costs per copied cell; maintenance per tick; death by starvation | autotrophs; selection on something other than copying speed; a land/sea ecology | implemented, verified |
| 4 | lights in the firmament for signs, seasons, days and years | sun height by latitude, **seasons** with opposite phase in the two zones (hemispheres), a moon, stars; **SENSE** (branch on a sky signal) and **WAIT** (sleep) | timing behaviour, dormancy, seasonal strategies | implemented, verified (section 14) |
| 5 | water creatures and birds | **predation** (energy taken from a neighbour), **motion** (swimming; flying across the firmament), **foreign execution** (a processor may run a neighbour's gene, so parasites can arise) | heterotrophs, trophic levels, migration between hemispheres, parasites | implemented, verified (section 15) |
| 6 | land animals; then humans | **tasks** on a working tape rewarded with energy (stepping stones from the docs/07 function library); **signals** between neighbours; the **interface** for a learned controller (the "human" layer), specified, not built | computation selected by the environment, communication, the slot for learning minds | design |
| 7 | the work completed; rest | **no design change**: the frozen rule set runs long and unattended with open-endedness measurements (evolutionary activity against a neutral shadow, novelty, complexity) and stated criteria for open-ended against plateau | the test | design |

**Changes against the suggested mapping, and why.** (a) Energy enters on day 1, not day 3: day 1 is "light", and
light that does nothing is not a layer; with day 1 alone the light already sets the rate of living (organisms run
fast by day and slow down at night), which is testable by itself. Day 3 then changes *who* gets the light (only the
genes that harvest it) and adds the costs of matter (copied cells) and of staying alive (maintenance), which is what
makes plants a selected phenotype. (b) The firmament is two barrier bands, not one: on a torus one band does not
separate anything. The two zones become the two hemispheres on day 4, with opposite seasons ("waters above and
waters below"), and on day 5 flying is the only way across ("fowl that fly above the earth in the open firmament").
(c) Day 4's signals act through a branch (SENSE skips the next codon when a sky signal is nonzero) and a sleep
(WAIT), because DNA-1's accumulator has no conditional: a signal loaded into A could not change behaviour.
(d) Swimming and flying are day 5, walking on land is day 6, as in the text; the parasite rule (foreign execution)
goes to day 5 with the other interactions.

## 2. Conventions

- **Base.** DNA-3 day k is DNA-2 (docs/15: TERM, DIVIDE re-arms P, COPY/COPYR NOP without a child) plus the days
  switched on. Every day implies `--dna2`; day 3 needs days 1 and 2; day 4 needs days 1, 2 and 3. Without
  `--days` every program behaves exactly as before (verified: section 10). Everything in docs/09 and docs/15 holds
  unless a day below changes it.
- **Flags.** `--days 1`, `--days 2`, `--days 1,2`, `--days 1,2,3`; `--dna3` = `--days 1,2,3`.
- **New random streams** (`rnd(t, k, j)` of docs/09 section 7): `rnd(t, s, 2000)` (day 2 ALLOC and day 5 MOVE
  priority), `rnd(t, e, 3000)` (day 5 order of eaters) and `rnd(-2, i, 9)` (day 3 terrain lattice). DNA-1/2 use j = 0 (ALLOC), 1, 2 (damage), 3, 4 + j (inflow), 1000 + c
  (mutation); 2000 is above every inflow cell index for G < 1996 and t = -2 is used by nothing else. Day 4 uses no
  random numbers.
- **Integer arithmetic.** All energy quantities are non-negative integers; `floor` of non-negative operands, so
  C's truncating division and Python's `//` agree.
- **Tick labels, hash schedule, census schedule, births file**: unchanged (docs/09 section 13).

## 3. Day 1: light and darkness (implemented)

**State.** Every processor gains a field `E` (energy, integer, 0 <= E <= emax). The world has a clock: tick t.

**Light.** `lambda(t) = lmax` if `t mod day_len < light_len`, else 0 (a square wave: day, then night).

**Parameters** (defaults): `day_len` 32, `light_len` day_len / 2 (integer division) = 16, `lmax` 16, `emax` 1,024,
`eshift` 4, `e_seed` 512, `e_spont` 128, `g_pass` 16 (passive gain in sixteenths). `T` keeps its meaning; `T = 0`
is allowed on day 1 and means no age limit. Constraints: day_len >= 1, 0 <= light_len <= day_len, emax >= 1,
0 <= e_seed, e_spont <= emax, all gains and costs >= 0.

**Step 1 (execute)** for each live processor, before its first instruction:

1. Passive light: `g = floor(lambda(t) * gp / 16)` with gp = g_pass (day 3: gp of the slot's terrain);
   `add = min(g, emax - E)`; `E += add` (the rest, g - add, is wasted).
2. Budget: `b = min(I, floor(E / 2^eshift))`.
3. The instruction loop of docs/09 runs at most b times; at the top of each iteration: if `E < 1` the tick ends for
   this processor; otherwise `E -= 1`, then the codon is fetched and executed as in DNA-2.
4. A successful DIVIDE (docs/09 and docs/15) also moves half of the store to the child: `x = floor(E / 2)`,
   `E -= x`, and x is the child's endowment.

**Step 3 (births).** A child born from a DIVIDE mark gets `E = x` (the endowment of that DIVIDE). If the marked slot
is not expressible, nothing is born and x is lost.

**Step 4 (deaths).** As in docs/09, except that `T = 0` means no age limit.

**Step 7 (inflow).** A spontaneous processor gets `E = e_spont`.

**Initial state.** Seeded processors get `E = e_seed`.

**Hash.** Each live processor's record gains `E` as an eleventh little-endian u64 after `copied`.

**Census.** After `depth_hist` (and the engine's optional `max_depth`), in this order: `"light"` (lambda of the
census tick), `"energy_total"` (sum of E over the living), `"energy_init"` (seeded energy), `"ledger"`: nine
cumulative counters `[passive, harvest, spontaneous, instructions, copy, maintenance, dead, lost_endowments,
wasted]`. The ledger closes exactly: energy_total = energy_init + passive + harvest + spontaneous - instructions -
copy - maintenance - dead - lost_endowments (the endowments of children that are born are internal transfers;
wasted light is outside the balance).

**Why this form.** The budget `E >> eshift` is the simplest rule under which an organism cannot run itself to zero:
without income its store decays geometrically (by a sixteenth per tick at eshift 4) and it slows down, which is how
a cell without light behaves. A fixed cost per instruction makes the gene's length and loop structure matter for
the first time: every executed codon costs.

**Acceptance test (day 1).** (i) Reference and engine identical on every tick (section 10). (ii) The ledger closes
in every census line. (iii) In a seeded world births follow the light: births per light half-day exceed births per
dark half-day (section 11.1).

## 4. Day 2: the firmament (implemented)

**Grid.** Parameter `W` (columns); `R = S / W` rows (W >= 3, R >= 3, W divides S). Slot s is at column
`x = s mod W`, row `y = s div W`. The tape is unchanged: one ring of N cells, slot s = cells s*G .. s*G + G - 1; so
a head that leaves its slot to the right reads the slot east of it (at the end of a row, the first slot of the next
row). Space acts through placement; reading stays one-dimensional (*choice*, see section 12).

**Firmament.** Parameter `firm` (rows per band, default 1; 0 = none). Row y is firmament if
`firm > 0 and y mod (R / 2) < firm`: two bands, rows 0 .. firm-1 and R/2 .. R/2+firm-1, which cut the torus into
zone 0 (rows firm .. R/2-1, "the waters above") and zone 1 (rows R/2+firm .. R-1, "the waters below"). With
firm > 0, R must be even and firm <= R/2 - 1. A firmament slot is never granted as a child, never refilled by the
inflow and never seeded, so it never holds a processor; its cells are read like any others.

**Step 2 (ALLOC), replaced.** On the state after step 1 (no grant of this tick visible yet): every requesting
processor s computes `r = rnd(t, s, 0) mod 8` and scans the eight neighbours in the order
`d = (r + i) mod 8, i = 0..7`, with offsets (dx, dy) = N (0,-1), NE (1,-1), E (1,0), SE (1,1), S (0,1), SW (-1,1),
W (-1,0), NW (-1,-1), coordinates modulo W and R. It proposes the first neighbour c that is not firmament, has no
processor and is not claimed; if there is none it proposes nothing. Each proposed slot is granted to the proposer
with the lowest key `(rnd(t, s, 2000) >> 32) * 2^32 + s` (a random priority, ties impossible because s is in the low
bits); losers fail silently. A grant sets `child`, `Q`, `copied` as in docs/09. Random priority instead of docs/09's
"lowest slot index" keeps the grid isotropic: with the lowest index, every contested slot would go to the requester
above or to the left.

**Step 7 (inflow).** A selected slot that is firmament is skipped.

**Seeding.** Seed k goes to slot `k * (S / n_seed)`, advanced (s = s + 1 mod S) to the first slot that is not
firmament, not already seeded and (with `--seed-terrain sea|land`, day 3) of that terrain. `--seed-slots a,b,...`
gives the slots explicitly (n_seed of them, distinct, none in the firmament).

**Census.** With firm > 0: `"zone_live": [live in rows y < R/2, live in rows y >= R/2]`. No new hashed state.

**Acceptance test (day 2).** (i) Identical on every tick. (ii) A single seed in zone 0 never produces a processor in
zone 1 (zone_live[1] = 0 for the whole run; section 11.2). (iii) Lineage patches: the spatial assortment of clades
(fraction of neighbouring live pairs from the same founder over the fraction expected for random placement) is
about 1 in DNA-2 and much larger on day 2 (section 11.2).

## 5. Day 3: dry land, sea and plants (implemented)

**Terrain.** Every slot that is not firmament is sea (0) or land (1), from value noise on a coarse lattice:
parameter `land_cell` L (default 32; must divide W and R), lattice `nx = W / L` by `ny = R / L`, lattice value
`v(i) = rnd(-2, i, 9) >> 48` (16 bits) for i = j * nx + i_x. For slot (x, y): `i0 = x div L, fx = x mod L,
i1 = (i0 + 1) mod nx`, likewise j0, fy, j1 from y; `V = (v(j0,i0)(L-fx) + v(j0,i1) fx)(L-fy) + (v(j1,i0)(L-fx) +
v(j1,i1) fx) fy` (exact integers); land if `V >= land_thr * L^2` (default land_thr 32,768 of 65,536), else sea.
The terrain is fixed for the run.

**Codon table.** HARVEST takes AGA and AGG from Arg (L1 keeps CGT, CGC, CGA, CGG): the two codons that vertebrate
mitochondria reassign. Under `--uniform-code` HARVEST is codon value 22 (CCG), the first NOP after TERM's 21.

**Light by terrain.** The passive gain (day 1, item 1) uses `gp_sea` (default 8) or `gp_land` (default 0) for the
slot's terrain: the sea gives a little energy to everything in it, land gives nothing to a gene that cannot use light.

| operation | effect |
|---|---|
| HARVEST | if the processor has not harvested yet in this tick: `g = floor(lambda(t) * gh / 16)` with gh = `gh_sea` (default 16) or `gh_land` (default 32) for its slot; `add = min(g, emax - E)`; `E += add`. Further HARVESTs in the same tick are NOP. |

**Cost of matter.** A COPY or COPYR that writes a cell (child claimed and Q inside the child slot) first needs
`c_copy` units (default 1): if `E < c_copy` the instruction does nothing (no write, P and Q unchanged), `pc` is set
back to the COPY (pc -= 1) and the processor's tick ends (before the wrap rule). Otherwise `E -= c_copy` and the
write proceeds as in DNA-2.

**Step 4 (deaths), extended.** After the age and expressibility test, a processor that survived it pays maintenance
`maint` (default 1): if `E < maint` it dies of starvation (counted in `deaths` and in `starved`), else `E -= maint`.
The store of every processor that dies (any cause) leaves the balance as "dead".

**Step 7 (inflow).** A selected land slot is skipped unless `--inflow-land`: new material arrives through the sea.

**Hash.** The cumulative `starved` count is appended as a fifth counter after `deaths`.

**Census.** After the day-1 and day-2 keys: `"starved_total"`, `"live_land"` (live processors on land),
`"harvesters"` (live processors whose gene, codons 1 .. len-1, contains a HARVEST codon).

**Calibration of the defaults** (reference, 32 x 32 slots of 256 cells, 16 seeds in the sea, 512 to 1,024 ticks):
with eshift 3 or 64-tick days every population starves in its first night; with eshift 4 and 32-tick days the
plant `ATG AGA GCT CTT TTA GGT` (START HARVEST ALLOC COPY TERM DIVIDE) fills 768 to 813 of the 960 habitable slots
and the DNA-2 replicator (no HARVEST) dies out at gp_sea 2 or 4, persists at about 190 at gp_sea 8 and spreads at
gp_sea 16. gp_sea 8 is the default: a non-plant can live in the sea, but poorly, so a plant is a real improvement
that selection can find. emax never bound in these worlds.

**Acceptance test (day 3).** (i) Identical on every tick, including the uniform code, G not a multiple of 4 and the
single strand. (ii) The ledger closes. (iii) A plant seeded in the sea colonises the land; the non-plant seed cannot
live on land (gp_land 0); whether plants evolve from the non-plant seed is the experiment of section 11.3.

## 6. Day 4: sun, moon and stars (implemented after the first version of this note, section 14)

Day 4 needs days 1, 2 and 3. It replaces lambda(t) by a light lambda(t, s) that depends on the slot, and adds two
operations.

**Calendar.** `d = t div day_len` (the day number), `phi = t mod day_len`. Triangle wave on [0, P):
`tri(u, P) = P - |4u - 2P|`, an integer in [-P, P] (P at u = P/2).

**Parameters** (defaults): `year_days` Py 16 (even), `moon_days` Pm 8, `season_amp` As day_len / 4 = 8 (ticks),
`lat_amp` 8 (sixteenths), `lmoon` 1. Constraints: As <= light_len and light_len + As <= day_len.

**Seasons, opposite in the two zones.** `sy = tri(d mod Py, Py) + Py` in [0, 2Py]. The zone of slot s is
z = 0 if y < R/2, else 1; its season value is `sz = 2Py - sy` for z = 0 and `sz = sy` for z = 1 (2Py = midsummer,
0 = midwinter). Day length of the zone: `Lz = light_len - As + floor(As * sz / Py)`; the sun is up when `phi < Lz`.

**Latitude.** The subsolar row `ys = floor(R/4) + floor(floor(R/2) * sy / (2 Py))` moves from the middle of zone 0
(midsummer there) to the middle of zone 1 and back once per year. `dy = |y - ys|`, `dist = min(dy, R - dy)`,
`fac = 16 - floor(lat_amp * dist / floor(R/2))` in [16 - lat_amp, 16].

**Moon.** `ml = floor(((tri(d mod Pm, Pm) + Pm) * 3 + Pm) / (2 Pm))` in [0, 3] (full moon 3).

**Light.** `lambda(t, s) = floor(lmax * fac / 16)` while the sun is up, else `lmoon * ml` (moonlight).
lambda(t, s) replaces lambda(t) in the passive gain and in HARVEST.

**Sky channels** (2-bit values, for SENSE): channel 0 sun = 0 if down, else 3 if fac >= 14, 2 if fac >= 11, else 1;
channel 1 moon = ml; channel 2 season = `floor(4 sz / (2 Py + 1))` (0 midwinter .. 3 midsummer, per zone);
channel 3 stars = 0 while the sun is up, else `1 + floor(3 y / R)` (the pole star's height: position by night).

**Codon table.** SENSE takes AGT and WAIT takes AGC from Ser (R1 keeps TCT, TCC, TCA, TCG); uniform code: 23 and 24.

| operation | effect |
|---|---|
| SENSE | the next codon is the operand (consumed, like SEEK's template); if it lies beyond the gene end (pc >= len) SENSE is NOP. Otherwise `c = codon(pc)`, `pc += 1`, and if channel `c mod 4` at (t, slot) is nonzero, `pc += 1` (the codon after the operand is skipped). |
| WAIT | the processor's tick ends after this instruction (its unit is paid; the wrap rule applies). |

So `SENSE <sun> WAIT` sleeps in the dark and `SENSE <sun> WAIT X` does X only in the dark; both polarities need
no further operation.

**Census.** After the day-3 keys: `"sensers"` (live genes containing SENSE), `"sleepers"` (containing WAIT). On day
4 the census key `"light"` becomes a list of two: lambda at column 0 of row R/4 and of row R/2 + R/4 (the middle of
each zone). No new hashed state (the calendar is a function of t).

**Acceptance test (day 4).** (i) Identical on every tick, with day lengths and lights checked against a table of
(t, s) values printed by the reference; (ii) a hand-written sleeper `START HARVEST SENSE <AAA> WAIT ALLOC COPY TERM
DIVIDE` against the plant without the SENSE-WAIT pair in one world: **prediction** the sleeper wins where nights
are long (each zone's winter) and loses in summer, so its share oscillates with the season and with opposite phase
in the two zones; (iii) from the plant seed alone, with mutation, a SENSE-WAIT motif arises and rises in frequency
first at high latitude (prediction, not a requirement).

## 7. Day 5: creatures of the water and birds of the air (implemented, section 15)

Day 5 needs days 1 to 4. It adds three operations, a processor field, two steps between execution and births, and
three ledger entries. The design version of this section proposed a blank source after a move, a separate FLY
operation and resolution after ALLOC; the implemented rules below differ where noted (*changed*).

**Codon table.** EAT takes CCA and CCG from Pro (SETP keeps CCT, CCC), MOVE takes ACA and ACG from Thr (SETQ keeps
ACT, ACC), EXEC takes GTA and GTG from Val (JNZB keeps GTT, GTC). Uniform code: values 25, 26, 27.

**Parameters** (defaults): `c_move` 2 (energy for a one-slot move), `c_fly` 8 (energy per slot of a flight),
`eat_shift` 1 (an eater takes floor(rest / 2^eat_shift)), `eat_eff` 8 (it keeps floor(taken * eat_eff / 16)).

**The code being executed.** Every processor gains a field `xs`, the slot whose gene it runs (its own slot at birth;
hashed as a twelfth little-endian u64 after E). While `xs` is the own slot the gene is fetched live, as before; while
it is another slot, every codon (instructions and operands, SEEK's template and SENSE's operand included) is read from
the tick-start snapshot `fwd_old` of that slot, and `len` is that gene's length there. The wrap rule becomes: if
`pc >= len`, then if `xs` is foreign, `xs` = own slot and `len` = the own gene's length; and `pc = 1`. At the start
of a processor's tick, if `xs` is foreign: if that slot is not expressible in the snapshot, `xs` = own, `pc = 1`;
otherwise `len` is its length there, and if `pc >= len` the processor returns home (`xs` = own, `pc = 1`). A write
of the processor into its own slot (INC, FLIP, STA) recomputes `len` only while it runs its own gene. P, Q, H, A,
the claimed child and the write rule are always the processor's own.

**Operations.** Each takes the next codon of the code being run as its operand (consumed like SEEK's template; if
it lies at or beyond `len` the operation is NOP). With `c` the operand and (dx, dy) the neighbour offset number
`c mod 8` of day 2:

| operation | effect in step 1 |
|---|---|
| EAT | if no EAT of this processor has counted yet in this tick and the neighbour slot is live (live flags of the tick start, which step 1 does not change): it becomes this tick's prey of the processor. |
| MOVE | if `child < 0`: a move request with operand c; an ALLOC request of this processor in this tick is withdrawn; the wrap rule applies and the processor's tick ends. With a claimed child, MOVE does nothing beyond consuming its operand. |
| EXEC | if the neighbour slot is expressible in the snapshot: `xs` = that slot, `pc = 1`, `len` = its gene's length (the processor continues with that gene's codon 1 in the same tick). |

**Step 2a (new): predation**, on the stores after step 1. For every prey (in any order: the result does not depend on
it) its eaters are ordered by the key `(rnd(t, e, 3000) >> 32) * 2^32 + e`; starting from the prey's store `rest`,
each eater in turn takes `x = floor(rest / 2^eat_shift)`, `rest -= x`, and gains `floor(x * eat_eff / 16)`. Then
every processor's store becomes `min(emax, E - lost as prey + gained as eater)`. Ledger: `eaten` += x, `trophic loss`
+= x minus the gain, and the excess over emax also counts as trophic loss. A prey left with less than `maint` starves
at step 4.

**Step 2 (ALLOC), extended.** A move request of processor s with operand c has the distance `dist = 1 + ((c >> 3)
mod 4)` (1 to 4 slots) and the target `t` = s shifted by (dx * dist, dy * dist) on the torus; its cost is `c_move`
for dist 1 (swimming) and `c_fly * dist` otherwise (flying). The request proposes t if t is not firmament, has no
processor, is not claimed, the store (after step 2a) holds the cost, and for dist 1 both s and t are sea. Move
proposals and ALLOC proposals compete for the same slots under day 2's key `(rnd(t, s, 2000) >> 32) * 2^32 + s`
(a processor makes at most one of the two). Flying is the only way across the firmament.

**Step 2c (new): moves.** Every granted move copies the G cells of s (both strands) to t and moves the processor:
slot = t, H and P shifted by (t - s) * G (mod N), Q = 0, child = -1, copied = 0, E -= cost (ledger `movement`),
xs = t if it was s, otherwise unchanged; the clade label moves with it. The source slot is dead and keeps its cells
(a shell; *changed*: not blanked, so the old gene remains material for read-through and the inflow). Births of step
3 marked by a processor that moved later in the same tick use the parent's values as they were (its depth, its
endowment, its gene in the old slot).

**Ledger.** On day 5 the census ledger has twelve entries: the nine of day 1, then `eaten` (moved from prey to eater,
not part of the balance), `trophic loss` and `movement` (both subtracted in the balance):
energy_total = energy_init + passive + harvest + spontaneous - instructions - copy - maintenance - dead -
lost_endowments - trophic_loss - movement.

**Census.** After the day-4 keys: `"eaters"`, `"movers"`, `"execers"` (live genes containing EAT, MOVE, EXEC) and
`"foreign"` (live processors running a neighbour's gene at the census).

**Predictions, written before the open run of section 15.** (i) Heterotrophs arise from plants by mutation (one EAT
codon in the loop) and persist at a low share, because a ninth-shaded plant's store is small; (ii) parasites (genes
that EXEC a neighbour and carry no complete copy loop of their own) arise within 10^5 ticks in dense plant mats;
(iii) fliers are rare without a reason to cross, but where one zone is empty (O1) a lineage that flies across has a
free hemisphere.

## 8. Day 6: land animals, and humans (design; the human layer is an interface only)

- **Land animals.** `MOVE` on land (walking) becomes possible; whether a gene swims, walks or flies is fixed by a
  "body" codon read once at birth (codon 1 after START mod 3), so body plans are heritable and mutable.
- **Tasks.** The last cells of each slot (beyond the codons; G = 256 gives 85 codons and 1 spare cell, so day 6 needs
  G with a 16-cell I/O area, e.g. G = 271) are a working tape. At the start of each tick the world writes an input
  x (4 bits, two cells, `rnd(t, s, 4000)`) into the I/O area; the gene computes; `OUT` submits the two cells under H
  as y. If y = f(x) for a function f of the task list, the processor gains `bonus(f)` units once per tick and
  function. The task list comes from the docs/07 library of 2-bit machines, ordered by how many programs of the
  winning machine compute each function (the rarer, the harder): stepping stones from identity and NOT through
  x+1, 15-x, parity, to x*x mod 16, with bonus doubling per difficulty rank (Avida's EQU at the top).
  Prediction: the first tasks appear within 10^5 ticks at 10^8 cells (the identity costs about four codons);
  harder tasks only if the easy ones are rewarded (Lenski et al. 2003 stepping stones).
- **Senses and signals.** `SIGNAL` writes A into a 2-bit per-slot signal register (visible to neighbours from the next
  tick, through the snapshot); `LISTEN`: operand direction, A = that neighbour's register. Prediction: signals are
  first used as kin markers (cooperation within patches), later as alarm calls if predators exist.
- **The human layer: an interface for a learned controller (not built).** A processor may be *controlled*: a flag
  set at birth for genes that carry a `MIND` codon. A controlled processor does not fetch codons in step 1; it
  executes one action per tick from a fixed set {NOP, HARVEST, ALLOC+COPY loop as a macro, DIVIDE, MOVE d, EAT d,
  SIGNAL v, WAIT} supplied by an external policy. Observation (fixed layout, 32 bytes): own E (8 bits), age, gene
  length, the four sky channels, terrain and occupancy of the eight neighbours, their signals and an own-or-other
  bit, the last reward. Reward: the change in E plus a constant per faithful child. Timing: the observation is taken
  from the tick-start state, the action is applied in step 1 in place of the codons, so the snapshot rule and the
  bit-exactness hold provided the policy is deterministic. Determinism: the policy runs on the host between ticks
  (or as a GPU kernel), its parameters and RNG are part of the checkpoint, and every action is logged so that the
  reference can replay a run from the log. The policy's parameters may be inherited at DIVIDE (genetic) and updated
  during life (learning), which allows the Baldwin effect to be measured. The learner and the observation encoding
  should be the ones of enderPeer/matrix (factored Q learning over local features, rewards from energy only, no
  named-task bonus), so that a matrix controller can be dropped in; matrix's JSONL experience export is the dataset
  format. Building it is the next project after days 4 to 6, not part of DNA-3.

## 9. Day 7: rest (design)

No rule changes after day 6: the rule set is frozen (a version string in the run header) and runs long and
unattended: 2,999,999,488 cells on the RTX 4090 for 10^7 ticks (about 11 h at the 1,024-cell slot size, 40 h at
256), census every 1,000 ticks, checkpoints every 10^6, and the same world at 10^8 cells on the three RTX 3060 with
three seeds.

**Measurements.**
- *Evolutionary activity* (Bedau and Packard) with a *neutral shadow* (as in Universe-1's design C,
  `docs/12_design_c_soup.md`): the component is the gene's **operation sequence** (synonymous codons collapsed, so
  drift through synonyms, which DNA-2 showed is free, does not count as activity). The shadow is a population of
  operation sequences with exactly the world's demography (a random shadow genome is copied with the same mutation
  per birth, dies per death, enters per spontaneous birth) and no selection; its activity distribution gives the
  threshold (99th percentile) above which a component counts as adaptive.
- *New activity* (components crossing the threshold per interval), *cumulative and mean activity*, *diversity*
  (distinct adaptive components).
- *Novelty*: operation sequences never seen before that cross the threshold.
- *Complexity*: per adaptive component the number of distinct operations used, the number of tasks performed, the
  trophic level, the use of SENSE, SIGNAL, MOVE; the maximum and mean over time.

**Criteria (stated before the run).** The run is called **open-ended** if all three hold over its second half:
(1) the rate of new adaptive components in the last quarter is at least half of that in the second quarter;
(2) at least one complexity measure reaches a new maximum in the last quarter; (3) diversity of adaptive components
does not fall below half its mean over the run. It is a **plateau** if (1) fails (new activity decays towards the
shadow's rate), and **bounded** (Bedau class 2) if (1) holds but (2) fails (turnover without growth). Prediction: a
plateau without tasks (days 1 to 5), bounded or open-ended only with the task ladder of day 6; the honest prior is
plateau, since no artificial-life system has yet passed criterion (2) for long.

## 10. Verification of days 1 to 3

`tools/dna3_verify.py` runs the reference locally and the engine on an RTX 3060 of knecht24 on the same arguments,
hashes the state after every tick and compares every stdout line (the tick hashes, and the census lines with the
energy ledger, zone counts, starvation, land and harvester counts) and the birth files byte for byte. Results and
both programs' outputs in `results/dna3/verify/` (`verify.json`).

| configuration | what it covers | ticks hashed | births | result |
|---|---|---:|---:|---|
| dna1_t1 | DNA-1 t1 of docs/09 (hash `a8dc45022d3a32d1` as in docs/10) | 200 | 3 | identical |
| dna2_seeded | DNA-2, 4 replicator seeds, mu 0.001 | 300 | 1,099 | identical |
| dna2_uniform | DNA-2, uniform code | 200 | 763 | identical |
| day1_seeded | day 1, 4 seeds | 300 | 984 | identical |
| day1_soup_edge | day 1, eshift 0, emax 200, 25-tick days with 10 light ticks, T = 0, mu 0.01, F 4 | 300 | 251 | identical |
| day2_firm | day 2, 16 x 16 grid, firmament | 300 | 971 | identical |
| day2_open_blank | day 2, 32 x 8 grid, no firmament, blank tape, mu 0.005 | 300 | 1,070 | identical |
| day12_slots | days 1 and 2, explicit seed slots in both zones | 300 | 783 | identical |
| dna3_plant | days 1 to 3, 32 x 32 grid, 16 plant seeds in the sea | 400 | 3,761 | identical |
| dna3_nonplant_mu | days 1 to 3, the non-plant seed, mu 0.01 | 400 | 327 | identical |
| dna3_uniform | days 1 to 3, uniform code (HARVEST = codon 22) | 300 | 2,556 | identical |
| dna3_g255_single | days 1 to 3, G = 255 (word-atomic writes), single strand with 20 hits per tick, inflow on land, c_copy 3, maint 2, gp_land 4 | 300 | 776 | identical |
| dna3_soup | days 1 to 3, unseeded, F 8, e_spont 300 | 300 | 29 | identical |

**13 of 13 configurations identical on every tick: 3,900 tick hashes, 390 census lines, 13,373 birth rows.** The
reference's DNA-1 and DNA-2 outputs are also byte-identical to files recorded before the edit (six configurations,
1,664 lines). The energy ledger closes in every census line of every world of section 11 (checked by
`tools/dna3_report.py`). One edge case shows why day 3 adds maintenance: with T = 0 and eshift 0 (`day1_soup_edge`)
organisms that spend their store to 0 never die, and the world fills with 250 dormant processors holding no energy.

## 11. First worlds

Series A (`tools/dna3_runs_a.sh`, one RTX 3060 of knecht24): 104,857,600 cells = 640 x 640 slots of 256 cells,
seed 1, mu 0.001, double strand, damage and inflow at their defaults (100 hits, 100 refills per tick), 4,096 seeds
of the DNA-2 replicator `ATG GCT CTT TTA GGT` (or the plant `ATG AGA GCT CTT TTA GGT`), census every 100 ticks (200 in
A5), slot maps of founder lineage (clade), gene, energy and plant status. Day-3 terrain with land_cell 64: 241,977
land slots (59.1 %), 166,343 sea, 1,280 firmament. Summaries in `results/dna3/runs_a/summary.json`, spatial
statistics in `*.spatial.json` (`tools/dna3_maps.py stats`), pictures and two videos in the same directory.

| world | days | ticks | alive (second half) | births | faithful | deepest lineage | distinct genes at the end | ticks/s |
|---|---|---:|---|---:|---:|---:|---:|---:|
| A0 DNA-2 control | - | 20,000 | 370,185 to 375,891 | 118,493,394 | 98.39 % | 642 | 2,220 | 428 |
| A1 | 1 | 20,000 | 338,829 to 344,338 | 108,391,729 | 98.41 % | 582 | 1,993 | 774 |
| A2 | 2 | 20,000 | 374,602 to 378,656 | 119,557,785 | 98.40 % | 637 | 2,634 | 328 |
| A3 | 1, 2 | 20,000 | 345,739 to 351,810 | 110,470,580 | 98.41 % | 581 | 2,175 | 629 |
| A4 plant seed | 1, 2, 3 | 20,000 | 342,432 to 349,050 | 108,229,484 | 98.13 % | 610 | 2,879 | 470 |
| A5 non-plant seed | 1, 2, 3 | 100,000 | 343,204 to 350,486 | 530,913,175 | 97.52 % | 2,836 | 7,185 | 264 |
| A7 one seed in zone 0 | 1, 2 | 20,000 | 173,043 to 176,272 | 46,337,986 | 98.37 % | 778 | 1,425 | 1,082 |

### 11.1 Day 1: the light sets the pace

- The population runs at 92 % of DNA-2's on the same seeds (338,829 to 344,338 against 370,185 to 375,891): at full
  light (16 units per tick) a full-speed organism just breaks even, and every division halves its store.
- **Births follow the light** (acceptance test iii; a 4,000-tick run of the same world with a census every 16 ticks,
  i.e. every half-day, second half): **103,935 births per light half-day against 69,704 per dark half-day**; the
  living hold 50.0 million units at dusk and 16.0 million at dawn.
- Energy budget over 20,000 ticks: 54.3 G units of passive light; 76.4 % spent on instructions, 23.5 % lost with
  organisms that die at age 64 holding a store, 0.04 % in endowments of children that were not born, none wasted on
  full stores. The ledger closes in all 200 census lines.
- Without space the founders' lineages stay mixed: clade assortment 0.99 to 1.01 (DNA-2: 0.99 to 1.02).

### 11.2 Day 2: space, patches and the firmament

- **Lineage patches.** Spatial assortment of founder lineages (the fraction of neighbouring live pairs descended from
  the same founder, divided by the fraction expected if the same population were placed at random): **DNA-2 1.01 at
  every census; day 2 171 at tick 99, 124.7 at tick 4,999, 112.1 at tick 19,999**: 61 % of neighbouring pairs share a
  founder where 0.5 % would by chance. At tick 19,999 the day-2 clade map has 62,181 patches, mean 6.1 slots, largest
  969; DNA-2's largest patch never exceeds 4 slots (368,772 patches for 373,650 organisms). Days 1 and 2 together:
  106.1; days 1 to 3 with the plant: 92.7. Synonymous spellings form patches too (gene assortment 11.0 at tick
  19,999, from 1.01 at tick 99 when the seed spelling still held nearly everything). Pictures: `A0_dna2_clade_t19999.png`
  (salt and pepper), `A2_day2_clade_t19999.png` and a 4x crop, video `A2_day2_clade.mp4` (a 320 x 320 corner, ticks
  0 to 19,999).
- Space costs no population: day 2 holds 374,602 to 378,656 (DNA-2: 370,185 to 375,891). Births next door replace
  the random draw, and where organisms die at age 64 a free neighbour is almost always there.
- **The firmament holds** (acceptance test ii). One seed in zone 0 (slot 6,720, row 10), inflow off (A7b): its
  lineage fills zone 0 to 174,639 of its 204,160 slots by tick 16,099, and **zone 1 holds 0 processors in all 200
  censuses**. With the inflow on (A7), zone 1 carries 29 to 55 spontaneous organisms at any time and none of them
  founds a lineage in 20,000 ticks. From one seed the lineage spreads as a front of constant speed: in the clade map
  of A7 its half-width along the row is 68, 138, 208 and 277 slots at ticks 999, 1,999, 2,999 and 3,999, and its
  extent down from the firmament 76, 144, 212 and 280 rows: **one slot per 14.5 ticks in both directions** (Fisher's
  wave of advance), until it meets itself around the torus at tick ~4,600.
- The fronts are squares, not circles: a child may go to any of the eight neighbours, so distance on this grid is the
  chessboard distance (visible in the pictures; see the decision on the neighbourhood in section 12).

### 11.3 Day 3: plants

- **The plant seed** (A4, `START HARVEST ALLOC COPY TERM DIVIDE`, 4,096 seeds in the sea) colonises the land within
  1,000 ticks and holds 342,432 to 349,050 slots: at tick 19,999, 209,855 on land (86.7 % of the land) and 139,042
  in the sea (83.6 %); 347,941 of the 348,897 organisms carry HARVEST. Energy over 20,000 ticks: income 95.5 G units,
  **88.6 % from HARVEST** and 11.4 % passive (sea only); spent on instructions 66.2 %, copied cells 2.5 %,
  maintenance 7.1 %, lost with the dead 24.0 %. Starvation is rare: 259,581 of 107,884,683 deaths (0.24 %). The
  lifetime of 64 ticks ends most lives; energy sets how fast they copy.
- **Plants evolve from the non-plant** (A5: the DNA-2 replicator, which cannot live on land since gp_land = 0,
  seeded in the sea). Harvesting genes are present at the first census (tick 199: 98 carriers, all in the sea); the
  first plants live on land at tick 799 (69 of 491 land organisms; non-plants reach the coast on their parents'
  endowment and starve there); harvesters are the majority from tick 2,999 (84,498 of 159,253); by tick 9,999 they
  are 322,780 of 323,984, 195,573 of them on land, and they stay (348,514 of 349,582 at tick 99,999). In the sea, where
  the non-plant can live on passive light, the plants displace it as well: harvesting triples a sea organism's income
  by day (8 + 16 units against 8).

| tick | alive | with HARVEST | on land | plants on land | in the sea | plants in the sea |
|---:|---:|---:|---:|---:|---:|---:|
| 199 | 42,825 | 98 | 2 | 0 | 42,823 | 98 |
| 799 | 86,531 | 1,248 | 491 | 69 | 86,040 | 1,179 |
| 1,999 | 111,459 | 23,623 | 13,557 | 13,462 | 97,902 | 10,161 |
| 2,999 | 159,253 | 84,498 | 54,768 | 54,138 | 104,485 | 30,360 |
| 3,999 | 239,930 | 188,123 | 126,543 | 126,043 | 113,387 | 62,080 |
| 9,999 | 323,984 | 322,780 | 196,078 | 195,573 | 127,906 | 127,207 |
| 99,999 | 349,582 | 348,514 | 209,268 | 208,896 | 140,314 | 139,618 |

- **How HARVEST got into the gene.** The seed has no spare codon. The first carriers in the top-gene lists are the
  seed with codons appended after DIVIDE: `... GGT TCA AGG` (R1 HARVEST, the most common carrier at ticks 999 and
  4,999), `... GGT TTA ATT AGA`, `... GGT TAT AGA`, `... GGT GAA ACG AGG AGT`. A copy ends once the template has passed
  its STOP, so a mutation that turns the seed's STOP (TAA) into a sense codon makes the child's gene run on into
  whatever its slot held before (soup or an earlier occupant), up to the next STOP there, and the child copies the
  longer gene faithfully. Stop-codon read-through into old material supplied the new operation; the soup supplies
  AGA or AGG at 2 of 64 codons. Selection then streamlined the genes: at ticks 29,999 and 99,999 the ten most common
  genes are all synonyms of **`START ALLOC COPY TERM DIVIDE HARVEST`** (six codons), in which TERM's skip over DIVIDE
  lands on HARVEST, so the copy loop is ALLOC COPY TERM HARVEST and harvests once per pass, as the hand-written plant
  does with HARVEST in front.
- **A selective sweep, visible in space.** The plants descend from few founders: the number of distinct founder
  colours in the clade map falls from 253 to 51 by tick 9,999 and to 26 by tick 99,999 (largest patch 82,344 slots),
  while in the world seeded with the plant itself (A4) 250 remain at tick 19,999. Pictures
  `A5_nonplant_plant_f*.png` (purple non-plants filling the sea, green plant patches spreading onto the land from the
  coasts) and the video `A5_nonplant_plant.mp4` (ticks 0 to 19,999).

### 11.4 Replicates of the plant origin

The A5 world with seeds 2 to 5 (each seed also draws its own terrain), 20,000 ticks, census every 100:

| seed | first harvesters (census) | harvesters > 50 % | > 90 % | alive at tick 19,999 | with HARVEST | on land |
|---:|---:|---:|---:|---:|---:|---:|
| 1 (A5) | 199 (census every 200) | 2,999 | 5,999 | 342,572 | 341,445 | 204,625 |
| 2 | 99 | 3,299 | 5,199 | 346,089 | 344,922 | 172,809 |
| 3 | 99 | 3,499 | 5,399 | 342,269 | 341,083 | 173,359 |
| 4 | 99 | 3,499 | 5,499 | 341,607 | 340,372 | 153,873 |
| 5 | 99 | 3,199 | 5,599 | 343,742 | 342,565 | 178,508 |

Five of five worlds: harvesters at the first census, the majority after 3,000 to 3,500 ticks, 90 % after 5,200 to
6,000, and every world's most common gene at tick 19,999 is a spelling of `START ALLOC COPY TERM DIVIDE HARVEST`. The
land count differs because each seed draws its own terrain.

### 11.5 Bands behind the fronts: lifetime cohorts, not day rings

Plant patches on land show concentric bands (pictures of A5 and of the big world). They looked like growth rings
of the day/night cycle; they are not. Three 3,000-tick variants of A5 (2D Fourier peak of the plant map in every
48 x 48 land window that plants cover, at tick 2,999; `bands_daynight_vs_constant_light.png`):

| light | lifetime T | windows | band period (median, slots) |
|---|---:|---:|---:|
| 32-tick days, 16 light | 64 | 23 | 3.43 |
| constant light | 64 | 63 | 6.00 |
| constant light | 128 | 56 | 12.00 |

Without a night the bands are sharper, and doubling the lifetime doubles their spacing: the organisms born together
at the advancing front die together at age T and their slots are refilled together, so the spacing is the front
speed times the lifetime (about 6 slots per 64 ticks in constant light, half that when the light is on half the
time). The bands are an artifact of the fixed lifetime; section 14 tests worlds without it.

### 11.6 Origin without a seed under day-3 rules

| world (10^8 cells, 256-cell slots, 200,000 ticks) | inflow | random genes before the first lineage of 10,000 | first lineage of 10,000 at tick | founder | at tick 199,999 |
|---|---|---:|---:|---|---|
| A6, days 1 to 3 | F 100 (sea only) | none in 120,997 | none | (46 faithful births, no lineage) | 23 alive |
| A6b, days 1 to 3 | F 4,000 (sea only) | 151,549 | 6,399 | `ATG TTG GGG CTA GCG AGG AGA` = START TERM DIVIDE COPY ALLOC HARVEST HARVEST: **already a plant** | 347,927 alive, 346,901 plants, 209,237 on land; leader START COPY ALLOC TERM DIVIDE HARVEST |
| A6c, DNA-2 control | F 4,000 (everywhere) | 33,276 | 599 | 12 codons, START TERM DIVIDE STA R3 SKNZ R3 R1 L3 ALLOC COPY LDA | 373,349 alive; leader START COPY ALLOC TERM DIVIDE |

With the default inflow the day-3 soup saw too few random genes for an origin (DNA-2's origin
statistics, docs/16: a self-made lineage after about 40,000 ticks at 23 refilled 1,024-cell slots per tick). With forty times the inflow a replicator arose after about 150,000 random genes (DNA-2 in the
same world: 33,000; one world each, so the ratio is a single sample), and it arose as a plant: in a world where
energy comes from light, the first self-sustaining gene already carried two HARVEST codons. Its lineage took the
sea and then the land, and by tick 199,999 its descendants had rearranged the operations into
START COPY ALLOC TERM DIVIDE HARVEST.

### 11.7 The 3 x 10^9-cell world

3,001,286,656 cells = 3,424 x 3,424 slots of 256 cells (land_cell 214: 6,796,810 land slots, 4,920,118 sea, 6,848
firmament), 1,000,000 seeds of the non-plant DNA-2 replicator in the sea, days 1 to 3 at their defaults, mu 0.001,
D 2,862 and F 2,862 per tick, 40,000 ticks in 10 min 34 s on the RTX 4090 (**63 ticks/s** with censuses and two slot
maps every 200 ticks; 202 ticks/s while the world was sparse), 3.99 GB. Files in `results/dna3/big/` (summary,
pictures; the census series and top genes there too; the 4.7 GB of slot maps stay on adler40).

| tick | alive | with HARVEST | on land | plants on land | in the sea | plants in the sea |
|---:|---:|---:|---:|---:|---:|---:|
| 199 | 2,356,722 | 1,543 | 143 | 66 | 2,356,579 | 1,477 |
| 999 | 2,548,137 | 26,035 | 6,506 | 6,385 | 2,541,631 | 19,650 |
| 1,999 | 2,766,486 | 278,241 | 88,682 | 88,027 | 2,677,804 | 190,214 |
| 2,999 | 3,383,537 | 1,159,020 | 455,590 | 450,559 | 2,927,947 | 708,461 |
| 4,999 | 6,174,548 | 5,273,967 | 2,745,938 | 2,740,080 | 3,428,610 | 2,533,887 |
| 9,999 | 9,551,841 | 9,515,831 | 5,649,845 | 5,636,323 | 3,901,996 | 3,879,508 |
| 39,999 | 10,003,577 | 9,972,546 | 5,886,366 | 5,876,046 | 4,117,211 | 4,096,500 |

The 10^8-cell result holds at 30 times the size: harvesters at the first census, the majority from tick 3,599, and
at the end 99.7 % of 10.0 million organisms are plants, 86.6 % of the land is covered. 5,678,780,280 births (97.82 %
faithful), lineages 1,368 generations deep, 41,776 distinct genes; the ten most common are all spellings of
`START ALLOC COPY TERM DIVIDE HARVEST` (375,351 carriers of `ATG GCT CTT TTA GGT AGA`). At this size the direct
spelling (the seed's STOP replaced by AGA) already led the plants at tick 2,999 (214,755 carriers), ahead of the
read-through variants (`... TCA AGA`, `... AAA AGG`, `... TAC AGA`). Energy over 40,000 ticks: 87.3 % of the income
from HARVEST, 65.9 % spent on instructions, 2.7 % on copying, 7.4 % on maintenance, 24.0 % lost with the dead;
21,282,249 starvations (0.38 % of deaths); the ledger closes in all 200 census lines. Pictures:
`big_plant_t{1999,2999,4999,9999,39999}_overview.png` (every fourth slot), `big_plant_t2999_crop.png` (856 x 856
slots at full resolution: round plant patches growing inside the purple non-plant sea, square banded plant fronts on
land) and `big_clade_t39999_crop.png`.

### 11.8 Reading

1. **Each day does what it was built to do, and the acceptance tests pass**: the light sets the pace of life
   (births 1.5 times faster by day), space makes lineages into patches (assortment 1 to 112) and the firmament keeps
   two populations apart, and on day 3 the energy economy selects for a new operation: plants arise from a non-plant
   within a few hundred ticks, take the land that the non-plant cannot use and then the sea as well.
2. **This is the first selection in these worlds for something other than copying speed.** DNA-2's big world drifted
   through synonyms of one operation sequence for 600,000 ticks; in the day-3 world a new operation entered the gene
   by read-through of a mutated STOP into old material, spread, and was moved by selection to the one position in
   the loop where it pays once per pass.
3. **It is still a world of one answer.** Once HARVEST is in the loop nothing more is rewarded; the end state is again
   drift among synonyms of one six-codon gene. Seasons and signals (day 4), predators and parasites (day 5) and tasks
   (day 6) are what can keep selection going; day 7 measures whether they do.
4. Caveats: one world per configuration at 10^8 cells (section 11.4 replicates the plant origin); the energy
   defaults are calibrated on small worlds (section 5), and other values change rates (plants need gp_sea < 16 to be
   an advantage worth finding); reading stays one-dimensional on the two-dimensional grid; the clade statistics hash
   founders into 253 colours, so they are lower bounds for patchiness.

## 12. Open decisions for days 4 to 7 (need the user)

1. **Day 4 calendar**: year of 16 days, moon of 8, seasons of opposite phase in the two zones (the firmament as the
   equator band between hemispheres), latitude light falling to half at the poles. Other periods, or one global
   season instead of hemispheres?
2. **Day 4 codons**: SENSE and WAIT from Ser's AGT and AGC (R1 keeps four codons). Alternatively from Leu's COPY family
   (CTA, CTG), or a fourth STOP reassignment (TGA, as in mitochondria)?
3. **Day 5 predation**: partial transfer (half the prey's store, half of that lost) as proposed, or kill-and-take-all?
   Energy-only, or does eating also copy material (the prey's gene as food that can be read)?
4. **Day 5 motion**: does a mover leave its slot blank or leave the old content behind (a shell that the inflow and
   read-through can use)? Fliers as the only way across the firmament, at what cost?
5. **Day 5 parasites**: foreign execution of a neighbour's gene (Tierra-style) as proposed, or only foreign reading?
6. **Day 6 tasks**: which functions of the docs/07 library, which reward ladder, and the I/O layout (the slot needs
   room beyond the codons: G = 271, or registers outside the tape)?
7. **Day 6 human layer**: should a learned controller replace a gene's control or sit beside it (e.g. choose among the
   gene's own subroutines)? Is matrix's factored Q learner the intended mind, and should learned parameters be
   inherited (Baldwin and Lamarck variants)?
8. **Day 7 budget and criteria**: 10^7 ticks at 3 x 10^9 cells (11 to 40 h on the 4090) plus three 10^8 worlds; are the
   three criteria of section 9 the ones to commit to before the run?
9. **Reading in two dimensions**: the head still reads the tape in row order, so a gene can read its east neighbour
   but not its north one. Keep it (simple, verified) or give each processor a second head direction?
10. **Neighbourhood**: eight neighbours give square fronts (chessboard metric); four give diamonds; a hexagonal grid
    would be closest to isotropic. Keep eight?
11. *(Answered by the user on 2026-10-10: no fixed lifetime; see section 14.)* **Energy defaults** (eshift 4, 32-tick days, gp_sea 8, gp_land 0, gh 16/32, c_copy 1, maint 1, T 64): keep the
    age limit alongside energy, or let energy alone end lives (T = 0 with maintenance)?

## 13. Reproduction

```bash
python sim/dna_ref.py --dna3 --W 32 --land-cell 8 --N 262144 --G 256 --ticks 400 --seed 3 --n-seed 16 --seed-terrain sea --genome "ATG AGA GCT CTT TTA GGT"
nvcc -O3 -std=c++17 -arch=sm_86 -o dna_world3 gpu/dna_world.cu
python tools/dna3_verify.py --host knecht24 --bin '~/universe2/dna3/dna_world3' --gpu 0 --out results/dna3/verify
sh tools/dna3_runs_a.sh 0 A0 A1 A2 A3 A4 A5 A6 A7
python tools/dna3_report.py RUNS/A0_dna2 RUNS/A5_nonplant --out summary.json
python tools/dna3_maps.py stats RUNS/A2_day2 --W 640 --mode clade --out A2.spatial.json
python tools/dna3_maps.py video RUNS/A5_nonplant --W 640 --mode plant --frames 0:100 --out A5.mp4
```

Binaries and raw runs (slot maps are not in the repository: 82 to 205 MB per mode at 10^8 cells, 11.7 MB per frame
at 3 x 10^9): knecht24 `~/universe2/dna3/` (dna_world3 for sm_86, `runs/`), adler40 `~/universe2/dna3/` (dna_world3
for sm_89, `big/`).


## 14. Without a fixed lifetime: shading, day 4, open-ended worlds

After the first version of this note the user asked for worlds without a fixed lifetime (energy alone ends lives),
every tick checked, open-ended runs, day 4 built next, and pictures checked for emergent behaviour.

### 14.1 Without the age limit the day-3 world freezes

The A5 world (non-plant seed, days 1 to 3) with `--T 0` (`results/dna3/no_lifetime/C1_T0.*`, 20,000 ticks): the
non-plants fill the sea by tick ~2,000 and, fed by the passive light, never starve; from then on 165,722 to 170,847
organisms hold the world (the sea has 166,343 slots), births fall from 55,591 per 100 ticks to **about 17,600 per 100
ticks, all of them replacements of starved organisms at the coast**, lineages stop at depth 21 and the 300 plant
mutants never spread (with T = 64 the plants took 99.7 % of the world, section 11.3: 5,309 births per tick against 176
here). The same happens with day 4 switched on (reference, 32 x 32 slots: 956 alive and 0 to 3 deaths per 128
ticks after tick 900): stores carry plants through the longest winter night. A full world has no turnover, and
without turnover evolution stops. For energy to end lives, a crowded slot must earn less.

### 14.2 Shading (a day-3 option, exact)

`--shade` (needs day 3): before its first instruction of tick t, a processor counts the live processors n (0 .. 8)
among its eight grid neighbours, read from the live flags at the start of the tick (before step 1, so the count is
the same in every execution order); every light gain of this processor in this tick, the passive gain and HARVEST,
becomes `floor(g * (9 - n) / 9)` (g as in sections 3 and 5). Light is shared with the neighbours' canopy: an
organism alone keeps all of it, one in a full patch a ninth. No new state; hash and census unchanged.

Reference, 32 x 32 slots, T = 0: with shading the plant world of days 1 to 3 holds 562 to 629 of 960 slots with
1,094 to 1,411 births per 256 ticks, every death a starvation; with day 4 the populations of the two hemispheres swing
with their seasons in opposite phase (zone 0: 334 alive in its summer, 93 in its winter).

### 14.3 Day 4 implemented; verification; checkpoints

Day 4 is implemented as specified in section 6 (`World.sky` in the reference, `sky()` in the engine, SENSE on AGT,
WAIT on AGC, census keys `sensers`, `sleepers` and the two-zone `light`), together with shading. Six configurations
were added to `tools/dna3_verify.py`: shading with T = 0 on the non-plant (`shade_T0_nonplant`), the plant on days 1
to 4 (`day4_plant`), the hand-written sleeper with an odd calendar (4-day years, 3-day moons, season amplitude 6,
latitude amplitude 12, moonlight 2, mu 0.01), day 4 without shading and with T = 64, day 4 under the uniform code
(`AAA CCG CCT AAA CGA AGA AAG CCC AGC`), and an unseeded single-strand G = 255 world with 20-tick days. **19 of 19
configurations identical on every tick, 7,300 tick hashes, 730 census lines, 18,828 birth rows**, on an RTX 3060
(`results/dna3/verify/`) and again on the RTX 4090 (`results/dna3/verify_4090/`): the sm_86 and sm_89 builds give
the same states. The reference's outputs of the 13 earlier configurations are byte-identical to the committed ones.

Open-ended runs need **checkpoints**. `--checkpoint FILE --checkpoint-every K` writes, every K ticks (a multiple of
the census interval) and at the last tick, both strands, the live flags, the processors, the claims, the counters,
the ledger, the clade map and the seeded energy; `--resume FILE` continues from it (N, G, seed, code and days must
match; the census, top-gene and slot-map files are appended to). Test: 600 ticks straight against 300 ticks, a
checkpoint and 300 resumed ticks of a day-4 world, every tick hashed: **all 612 output lines identical**.
`tools/dna3_open.sh` runs a world with no planned end (10^12 ticks) and resumes from its checkpoint whenever it is
started again. A new output-only slot-map mode `traits` (1 + HARVEST + 2 SENSE + 4 WAIT per organism) shows sleepers
in the pictures. One bug was found on the way: the engine's operation-name table lacked SENSE and WAIT, so the first
day-4 runs with `--top-genes` crashed on the first gene containing them (output only; fixed before the runs below).

### 14.4 Open-ended worlds

Four worlds were started on 2026-10-10 at about 21:41 UTC with no planned end (`tools/dna3_open.sh`; all with
T = 0 and shading, mu 0.001, 256-cell slots, census and slot-map frames every 10,240 ticks = 20 years of 512 ticks,
so every census sees the same season; checkpoints every 102,400 ticks). They keep running after this note;
`tools/dna3_watch.py` pulls their census series, top genes and latest frames and lists emergence events into
`results/dna3/open/<name>/` (timeline.png, plant/traits/clade pictures, events.json). Numbers below are from the
watch at the tick given.

| world | where | days | start | tick | alive | zone 0 / zone 1 | plants | with WAIT | with SENSE | on land |
|---|---|---|---|---:|---:|---|---:|---:|---:|---:|
| O1_seasons | 10^8 cells, RTX 3060 | 1-4 | 4,096 non-plant seeds in the sea | 204,799 | 115,846 | 4,676 / 111,170 | 114,559 | 88,138 | 18,695 | 76,000 |
| O2_soup | 10^8, RTX 3060 | 1-4, F 4,000 | unseeded | 40,959 | 42,549 | 17,833 / 24,716 | 39,902 | 7,043 | 4,906 | 3,799 |
| O3_noseasons | 10^8, RTX 3060 | 1-3 | 4,096 non-plant seeds | 20,479 | 235,114 | 113,396 / 121,718 | 232,552 | - | - | 126,180 |
| O4_big_seasons | 3 x 10^9, RTX 4090 | 1-4 | 1,000,000 non-plant seeds | 81,919 | 1,963,951 | 1,339,312 / 624,639 | 1,940,912 | 234,147 | 21,065 | 271,949 |

**What emerged (seen in the pictures and the series; one world each, so each is an observation, not a rate):**

1. **A crash, then slow adaptive growth instead of an equilibrium.** With energy as the only killer the seeded
   population crashes in its first year (O1: 12,208 alive at tick 127, 360 at tick 511, with 38,950 starvations; the
   non-plants cannot live on a ninth of the sea's passive light in a full patch) and then grows for 200,000 ticks
   as plants and new designs spread: 1,864 at tick 10,239, 43,620 at 102,399, 115,846 at 204,799 (O1); 205,472 to
   1,963,951 between ticks 10,239 and 81,919 at 3 x 10^9 (O4). With T = 64 (section 11) every world reached its
   ceiling within a few thousand ticks and then drifted.
2. **WAIT spreads, and genes get longer.** In both seasonal worlds the share of organisms carrying WAIT rises after
   about 70,000 ticks: O1 9 % at tick 61,439, 46 % at 102,399, 76 % at 204,799; O4 2 % at 61,439, 12 % at 81,919
   (234,147 carriers). The leading genes grow from 6 to 15 to 19 codons with repeated `HARVEST WAIT` pairs, e.g.
   `START ALLOC COPY TERM DIVIDE LDA HARVEST ALLOC SKNZ SKZ STA WAIT ALLOC HARVEST WAIT HARVEST WAIT HARVEST WAIT`
   (O1, tick 204,799) and `START ALLOC COPY TERM DIVIDE WAIT HARVEST WAIT HARVEST` (O4, tick 81,919, the most common
   gene, 27,999 carriers). WAIT here is mostly not used as a night switch (SENSE stays at 16 % in O1 and 1 % in O4);
   the instructions spent per organism and tick stay at 3.8 to 4.1 in O1 while WAIT rises from 9 % to 76 %, so it does
   not simply save energy either. A hypothesis, not yet tested: a tick ends at each WAIT, so `HARVEST WAIT` pairs
   spread over the loop make every tick contain exactly one HARVEST however few instructions the store allows,
   i.e. WAIT aligns the gene with the clock. Diversity stays high: 19,884 distinct genes in O1 at tick 204,799,
   120,279 in O4 at 81,919; the most common gene holds 4 % of O1 and 1.4 % of O4.
3. **A hemisphere died and stays empty.** In O1 the lineages of zone 0 died out in an early winter (before tick
   10,239); since nothing crosses the firmament, zone 0 has held only depth-0 organisms (random inflow genes that
   never reproduce, 4,676 at tick 204,799) ever since, while zone 1 fills up (`O1_seasons/traits_t204799.png`: the
   upper half dark, the lower half a mosaic of plant territories). At 3 x 10^9 cells both zones survived. Day 5's
   fliers are the designed way back.
4. **Territories.** Without seasons (O3) founder lineages hold large solid territories with sharp borders
   (`O3_noseasons/clade_t20479.png`), unlike the fine patches of the T = 64 worlds (section 11.2): an organism keeps
   its slot until it starves, so a lineage that holds ground keeps it, and borders move only by starvation.
5. **Life begins as a plant again.** The unseeded seasonal soup (O2, F 4,000) has self-made plant lineages by tick
   10,239 (13,840 plants of 18,285 alive) and WAIT and SENSE carriers among them (`O2_soup/traits_t40959.png`:
   yellow plant+WAIT and white plant+SENSE+WAIT clusters).
6. **An artifact to know about:** with T = 0 a random inflow gene that cannot copy but can live on the sea's
   passive light sits in its slot forever. These depth-0 sitters accumulate (4,634 in O1, all in the sea).

Next steps for these worlds: test the WAIT hypothesis by tracing a `HARVEST WAIT` gene against the same gene
without WAIT in one world (the trace tools of docs/11 apply), Bedau activity against a neutral shadow (day 7), and
day 5 (fliers to cross the firmament, predators, parasites).

## 15. Day 5 implemented; the open worlds later

### 15.1 Verification

Day 5 as specified in section 7 is in both programs (`--days 1,2,3,4,5`; the engine's predation runs one thread per
prey and applies losses and gains in a second kernel, moves run one thread per mover). Six configurations were added:
plants with mu 0.02 (eaters, movers and EXEC arise by mutation), a world of hand-written eaters (`START HARVEST EAT
<N> ALLOC COPY TERM DIVIDE`), swimmers (`MOVE <AGA>`: one slot east), fliers without shading (`MOVE <CTA>`: four slots
south, c_fly 3), EXEC carriers with eat_shift 2, eat_eff 12, c_move 1, and the uniform code at G = 255. They exercise
the new paths (in the eater world 135,232 units changed hands by predation, the swimmers spent 19,952 units on moves,
up to 75 processors ran a neighbour's gene at once). **25 of 25 configurations identical on every tick, 11,500 tick
hashes, 1,150 census lines, 21,634 birth rows, on an RTX 3060 and again on the RTX 4090.** The reference's outputs of
the 19 earlier configurations did not change. The ledger with the three new entries closes in every census line
(reference check over 3,072 ticks and all engine census lines of the worlds below).

### 15.2 O5: the day-5 world

`O5_day5`, started 2026-10-10 22:01 UTC on an RTX 3060 with the same world as O1 (10^8 cells, shading, T = 0,
days 1 to 5) but seeded with 4,096 *plants* (`ATG AGA GCT CTT TTA GGT`) in the sea; census every 10,240 ticks.

| tick | alive | zone 0 / 1 | with EAT | with MOVE | with EXEC | running a neighbour's gene | with WAIT |
|---:|---:|---|---:|---:|---:|---:|---:|
| 10,239 | 64,934 | 45,218 / 19,716 | 163 | 140 | 380 | 242 | 99 |
| 30,719 | 162,649 | 100,926 / 61,723 | 3,664 | **146,500** | 51,218 | 200 | 387 |
| 61,439 | 142,142 | 80,247 / 61,895 | 14,408 | 54,586 | 30,167 | | 5,771 |
| 102,399 | 199,845 | 114,378 / 85,467 | 78,253 | 128,856 | 6,432 | | 58,241 |
| 153,599 | 179,441 | 89,060 / 90,381 | **145,687** | 64,892 | 2,920 | 172 | 101,271 |

1. **Dispersal came first.** By tick 30,719, 90 % of the organisms carry MOVE. The leading form appended
   `HARVEST MOVE JNZB` to the plant: the operand GTC (or GTT) decodes to a flight of two slots south-west (south), so
   every pass of the loop throws the organism two slots on, like a seed carried by the wind. In the pictures the
   movers form diagonal streaks along their flight directions (`animals_t51199.png`). The share then fell back to
   36 %.
2. **Then predation took over.** Eaters rose from 163 (tick 10,239) to 14,433 (51,199) and 145,687 (153,599): 81 % of
   the organisms. They are all still plants (179,420 of 179,441 carry HARVEST): mixotrophs that harvest light and eat
   their neighbours, e.g. `START HARVEST ALLOC COPY TERM DIVIDE HARVEST EAT L1 HARVEST EAT R1 HARVEST`. In the last
   census interval an organism gained on average 0.74 units of passive light and 5.42 of HARVEST per tick and took
   2.28 from its neighbours (1.19 of which were lost as trophic loss): about 37 % of the light income now passes
   through predation. Pure heterotrophs (EAT without HARVEST) did not appear.
3. **Hunters.** Organisms with EAT and MOVE (magenta in `animals_t153599.png`) hold large regions; plain plants
   (green) survive in a few refuges.
4. **Parasites, so far, no.** EXEC carriers rose to 51,218 (tick 30,719) but at most a few hundred processors run a
   neighbour's gene at any census: most EXEC codons sit at the end of a gene where their operand lies beyond the gene
   and they do nothing; the carriers fell to 2,920. Prediction (ii) of section 7 (parasites within 10^5 ticks) has not
   come true in this world by tick 153,599.
5. **Crossing the firmament: not measured yet.** Both zones were seeded, so a crossing does not show in the zone
   counts; whether any lineage crossed by flight needs the clade map of both sides compared over time.

### 15.3 The day-4 worlds later

| world | tick | alive | zone 0 / zone 1 | with WAIT | with SENSE | on land | distinct genes | deepest lineage |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| O1_seasons | 573,439 | 271,790 | 135,909 / 135,881 | 210,909 | 23,426 | 158,629 | 27,108 | 1,968 |
| O2_soup | 614,399 | 258,185 | 145,890 / 112,295 | 189,073 | 13,284 | 162,963 | 19,907 | 2,929 |
| O4_big_seasons (3 x 10^9) | 184,319 | 5,829,121 | 3,642,111 / 2,187,010 | 3,498,777 | 115,415 | 3,364,666 | 255,739 | 2,382 |

- **A second origin of life in the empty hemisphere (O1).** Zone 0, barren since an early winter (section 14.4),
  held 5,616 depth-0 organisms at tick 296,959; then 9,003 at 307,199, 31,849 at 337,919, 55,849 at 358,399, and
  135,909 at 573,439, while the depth-0 sitters fell from 5,555 to 633 as the new lineage displaced them. Nothing can
  cross the firmament on day 4, so the lineage arose in zone 0 itself from the inflow; its genes have a design of their
  own (`START HARVEST ALLOC TERM DIVIDE JNZB COPY`, `ATG AGG GCT TTG GGC GTG CTG`, in the top 10 from tick 307,199),
  unlike any gene of zone 1. At tick 573,439 each hemisphere is held by a single founder lineage
  (`O1_seasons/clade_t573439.png`: green above, pink below): two biospheres with independent origins. The founding
  event has not been traced instruction by instruction.
- **WAIT stays and grows**: 78 % of O1 (210,909), 73 % of O2, 60 % of O4. The populations still grow (O1 115,846 at
  tick 204,799, 271,790 at 573,439), now in both zones.
