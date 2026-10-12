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

## 2026-10-11 09:50 UTC: tick 174,079

- 11,551,344 alive (zone 0 5,762,406 / zone 1 5,788,938; 6,046,660 on land); +12,382 since tick 143,359.
- Task code in 11,549,627. Rewarded answers: x^1 614.1 G, x+1 2,699.7 G, x^3 569.0 G, x-1 9.21 G (+0.585 G per
  interval, constant).
- Eaters 433,285 (falling), movers 3,438,034, WAIT 28,706, SIGNAL 3,903,745, LISTEN 1,491,287; 99,101 distinct genes
  (falling); deepest lineage 797. Top-10 order unchanged; still no new operation sequence since tick 20,479.
- About 4.4 ticks per second. O5 tick 11,335,679, O6 tick 11,130,879. Disk: adler40 108 GB, knecht24 32 GB free.

## 2026-10-11 13:40 UTC: tick 235,519

- 11,567,708 alive (zone 0 5,773,225 / zone 1 5,794,483): +6,000 per 10,240 ticks, nearly flat. Task code in
  11,566,147; x-1 answers 12.72 G (+0.585 G per interval, constant). Eaters 413,249, movers 3,430,922, WAIT 28,574,
  SIGNAL 3,915,419, LISTEN 1,486,896; 97,605 distinct genes. Top 10 unchanged. Running (dna_world3i); 107 GB free.

## 2026-10-11 15:40 UTC: tick 266,239

- 11,573,153 alive (zone 0 5,776,819 / zone 1 5,796,334), nearly flat. x-1 answers 14.48 G (+0.585 G per 10,240
  ticks, constant). Eaters 406,865, movers 3,428,204, WAIT 28,492, SIGNAL 3,918,991, LISTEN 1,485,986; 96,949
  distinct genes. Top 10 unchanged. Running; 106 GB free.

## 2026-10-11 17:40 UTC: tick 296,959

- 11,577,962 alive (zone 0 5,779,851 / zone 1 5,798,111). x-1 answers 16.24 G (+0.585 G per 10,240 ticks, constant).
  Eaters 401,280, movers 3,426,571, WAIT 28,466, SIGNAL 3,922,366, LISTEN 1,484,770; 96,361 distinct genes (falling).
  Top 10 unchanged. Running; 106 GB free.

## 2026-10-11 19:40 UTC: tick 327,679

- 11,581,595 alive (zone 0 5,782,453 / zone 1 5,799,142). x-1 answers 17.99 G (+0.585 G per 10,240 ticks). Eaters
  396,592, movers 3,425,067, WAIT 28,364, SIGNAL 3,924,866, LISTEN 1,484,195; 95,857 distinct genes (falling). Top 10
  unchanged. Running; 105 GB free.

## 2026-10-11 21:40 UTC: tick 358,399

- 11,584,400 alive (zone 0 5,784,054 / zone 1 5,800,346). x-1 answers 19.75 G (+0.585 G per 10,240 ticks). Eaters
  392,807, movers 3,423,832, WAIT 28,317, SIGNAL 3,926,761, LISTEN 1,483,749; 95,554 distinct genes. Top 10
  unchanged. Running; 105 GB free.

## 2026-10-11 23:40 UTC: tick 389,119

- 11,587,076 alive (zone 0 5,785,613 / zone 1 5,801,463). x-1 answers 21.51 G (+0.585 G per 10,240 ticks). Eaters
  389,999, movers 3,422,513, WAIT 28,272, SIGNAL 3,928,755, LISTEN 1,483,409; 95,284 distinct genes. Top 10
  unchanged. Running; 104 GB free.

## 2026-10-12 01:40 UTC: tick 419,839

- 11,588,805 alive (zone 0 5,786,803 / zone 1 5,802,002). x-1 answers 23.28 G (+0.585 G per 10,240 ticks). Eaters
  387,541, movers 3,421,455, WAIT 28,229, SIGNAL 3,929,841, LISTEN 1,483,104; 94,954 distinct genes. Top 10
  unchanged. Running; 104 GB free.

## 2026-10-12 03:40 UTC: tick 450,559

- 11,590,810 alive (zone 0 5,787,874 / zone 1 5,802,936). x-1 answers 25.04 G (+0.585 G per 10,240 ticks). Eaters
  385,356, movers 3,420,339, WAIT 28,186, SIGNAL 3,931,127, LISTEN 1,482,815; 94,823 distinct genes. Top 10
  unchanged. Running; 103 GB free.

## 2026-10-12 05:40 UTC: tick 481,279

- 11,592,306 alive (zone 0 5,788,690 / zone 1 5,803,616). x-1 answers 26.80 G (+0.585 G per 10,240 ticks). Eaters
  383,451, movers 3,419,703, WAIT 28,164, SIGNAL 3,932,315, LISTEN 1,482,703; 94,601 distinct genes. Top 10
  unchanged. Running; 103 GB free.

## 2026-10-12 07:40 UTC: tick 511,999

- 11,593,697 alive (zone 0 5,789,522 / zone 1 5,804,175). x-1 answers 28.57 G (+0.585 G per 10,240 ticks). Eaters
  381,838, movers 3,418,858, WAIT 28,178, SIGNAL 3,933,137, LISTEN 1,482,521; 94,382 distinct genes. Top 10
  unchanged. Running; 102 GB free.

## 2026-10-12 09:40 UTC: tick 552,959

- 11,594,941 alive (zone 0 5,790,098 / zone 1 5,804,843). x-1 answers 30.92 G (+0.589 G per 10,240 ticks). Task
  income 1,869.0 G per 10,240 ticks. Eaters 380,068, movers 3,417,709, WAIT 28,174, SIGNAL 3,934,396, LISTEN
  1,482,155; 94,264 distinct genes (94,382 at tick 511,999). Top 10 unchanged; the animals map (green plants, blue
  movers, red eater patches, one yellow patch in the north) looks the same as at tick 522,239. Running; adler40 101 GB
  free (new CUDA processes on adler40 fail with a driver/library mismatch since the 2026-10-10 package update; the
  running D7 and O7 are not affected).

## 2026-10-12 11:00 UTC: D7 stopped at tick 583,679 (the user's decision)

- Last census tick 583,679; checkpoint at tick 512,000 (`~/universe2/dna3/day7/D7_big.ckpt`, adler40). Reason: the
  plateau is established (section 18.5) and D7's recorded series already covers the ticks D8 and D9 will reach for the
  next ~250,000 ticks; the RTX 4090 goes to D9. D7 can be resumed with its `dna3_open.sh` command (deterministic: it
  recomputes the same states from tick 512,000). adler40 rebooted at 11:00 UTC to clear the NVIDIA driver/library
  mismatch.
