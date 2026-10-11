# Day 7 run log (D7_big)

Rules frozen at commit 85eed02 (binary dna_world3h): days 1 to 6, `--shade --T 0`, mu 0.001, 3,001,286,656 cells
(3,424 x 3,424 slots of 256 cells, land-cell 214, seed 7), 1,000,000 plant seeds (`ATG AGA GCT CTT TTA GGT`) in the
sea, census and slot maps (plant, traits, animals) every 10,240 ticks, checkpoint every 102,400 ticks, on adler40's
RTX 4090 from 2026-10-10 22:27 UTC. First planned to end Monday 2026-10-12 09:50 Berlin; on 2026-10-11 the user asked
to let it run open-ended. At its first checkpoint (tick 102,400) it switches to dna_world3i with `--op-census D7_big.ops`
(output only, built by the day-7 measurement session) and from then on has no end.
Picture and emergence checks every 2 hours; entries below give only measured numbers.


## 2026-10-11 01:50 UTC: tick 51,199 (first scheduled check)

- 11,421,120 alive (zone 0 5,676,852 / zone 1 5,744,268; 5,999,572 on land); at tick 30,719: 11,310,241.
- Task code in 11,418,519 (99.98 %). Rewarded answers so far: x^1 137.7 G, x+1 603.8 G, x^3 126.7 G, x-1 2.19 G
  (x-1 grew 1.03 G -> 1.61 G -> 2.19 G per 10,240 ticks, about 0.58 G per interval).
- Eaters 587,841 (falling: 691,219 at tick 30,719), movers 3,469,833 (flat), WAIT 29,060 (flat), SIGNAL 3,820,769,
  LISTEN 1,525,865; 105,121 distinct genes (106,548 at 30,719); deepest lineage 755 (also 755 at 40,959).
- Most common gene: `START HARVEST ALLOC COPY TERM DIVIDE LDA L3 OUT LISTEN OUT IN INC` (424,278; its LISTEN value is
  overwritten by IN, section 16 signal-flow reading). No new operation sequence entered the top 10 since tick 20,479.
- Pictures (`D7_big/traits_t51199.png`, `animals_t51199.png`, every fourth slot): almost uniformly plants with task
  code; a contiguous patch without HARVEST (purple, traits map) at the same columns on both sides of the firmament
  (around x 2,700 to 3,200), not yet checked for a crossing; movers form large blue regions with straight diagonal
  edges (flight directions); eaters in small red patches; one larger EXEC-carrier patch (yellow) in zone 0.
- Processes: dna_world3h (D7) running, switch script waiting for the first checkpoint (tick 102,400); O5 at tick
  4,710,399, O6 at 2,867,199 running. Disk: adler40 113 GB free, knecht24 34 GB.

## 2026-10-11 03:50 UTC: tick 81,919

- 11,488,127 alive (zone 0 5,719,688 / zone 1 5,768,439; 6,021,444 on land): +67,000 since tick 51,199.
- Task code in 11,485,911. Rewarded answers: x^1 255.8 G, x+1 1,123.2 G, x^3 236.2 G, x-1 3.95 G (x-1 now +0.58 G per
  10,240 ticks, unchanged rate).
- Eaters 508,446 (still falling: 587,841 at tick 51,199), movers 3,457,886, WAIT 29,434, SIGNAL 3,860,966, LISTEN
  1,512,636; 103,069 distinct genes (slowly falling); deepest lineage 761.
- Top genes unchanged in order (leader 439,323 carriers); no new operation sequence in the top 10.
- Pictures: the traits map at tick 81,919 is nearly identical to tick 51,199, including the non-HARVEST patch on both
  sides of the firmament: the landscape is static over these 30,000 ticks (organisms keep their slots; T = 0).
- Processes: D7 (dna_world3h) running, switch waiting for tick 102,400 (next census); O5 tick 6,973,439, O6 tick
  4,986,879 running. Disk: adler40 112 GB, knecht24 34 GB free.

## 2026-10-11 05:50 UTC: tick 112,639; switched to the op-census build

- Switch done: at tick 102,400 the checkpoint (3.1 GB) was written at 03:43 UTC, the dna_world3h process was stopped
  at 03:44:28 and D7 resumed at 03:44:32 with dna_world3i `--op-census D7_big.ops`, open-ended. First op census at
  tick 112,639 (D7_big.ops 4.6 MB). About 4.3 ticks per second after the resume (tick 102,400 -> 112,639 in 38 min).
- 11,519,771 alive (zone 0 5,741,120 / zone 1 5,778,651; 6,032,961 on land); +31,644 since tick 81,919.
- Task code in 11,517,840. Rewarded answers: x^1 374.8 G, x+1 1,646.7 G, x^3 346.6 G, x-1 5.70 G (x-1 +0.58 G per
  interval, unchanged).
- Eaters 471,174 (falling), movers 3,450,253, WAIT 29,057, SIGNAL 3,882,548, LISTEN 1,501,939; 101,404 distinct
  genes (falling); deepest lineage 766. Top-10 order unchanged (leader 444,191 carriers).
- Processes: D7 on dna_world3i running; O5 tick 8,581,119, O6 tick 7,065,599 running. Disk: adler40 109 GB,
  knecht24 33 GB free.

## 2026-10-11 07:50 UTC: tick 143,359

- 11,538,962 alive (zone 0 5,754,134 / zone 1 5,784,828; 6,041,015 on land); +19,191 since tick 112,639.
- Task code in 11,537,125. Rewarded answers: x^1 494.3 G, x+1 2,172.4 G, x^3 457.6 G, x-1 7.46 G (x-1 +0.585 G per
  interval, still constant).
- Eaters 448,703 (falling), movers 3,444,198, WAIT 28,797, SIGNAL 3,895,599, LISTEN 1,495,738; 100,245 distinct
  genes (falling); deepest lineage 769. Top-10 order unchanged; no new operation sequence in the top 10 since
  tick 20,479 (9 entries in events.json, all before).
- Rate about 4.5 ticks per second; D7_big.ops 8.3 MB after 4 op censuses. O5 tick 10,045,439, O6 tick 9,103,359.
  Disk: adler40 108 GB, knecht24 32 GB free.
