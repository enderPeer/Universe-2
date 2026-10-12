# D8 run log (the fork of D7 with the long ladder)

D7's checkpoint at tick 204,800 (11.56 M organisms, rules of days 1 to 6 frozen at 85eed02) continued with `--ladder`
(18 ranked tasks on 4-bit inputs, the social task, depletable task pools of 32 x 32-slot regions, streak 7) and
`--eat-eff 12`, everything else unchanged (docs/17 section 18). Binary dna_world3k (sm_86) on knecht24's RTX 3060,
GPU 0 (shared with another of the user's jobs), from 2026-10-11 10:35 UTC, open-ended; census every 5,120 ticks,
checkpoint every 51,200, op census in D8_ladder.ops. D7 continues unchanged on the RTX 4090 as the control.
Entries below give only measured numbers.

## 2026-10-11 ~11:20 UTC: tick 209,919 (first census, 5,120 ticks after the fork)

- 5,617,252 alive (zone 0 3,379,561 / zone 1 2,237,691): **down 51 %** from 11,558,000 at the fork. D7 at its tick
  204,799 held 11.56 M; the old task income vanished (D7's `... OUT IN INC` genes now add 1 without carry).
- Task carriers 5,555,105. Rewarded answers since the fork, by task (rank): x^1 (0) 794 M, x^4 (0) 18,747, x+4 (0)
  3,575, x&12 (0) 40.5 M, x^5 (1) 5,100, x+8 (1) 151, x&3 (1) 18,584, 4x (1) 92,514, a (1) 18.0 M, **x+1 (4) 4,144 M**,
  x|3 (2) 19.0 M, x/4 (2) 93,596, 15-x (3) 17, x-1 (4) 39.7 M, 2x (4) 43,219, x/2 (4) 52,640, parity (4) 42,419,
  a xor b (4) 18.0 M, **social (3) 3.83 M**.
- Pools (sum over 11,449 regions, full = 46,895,104): x+1 at 12.4 M (26 %), x^1 at 45.2 M, x&12 at 46.4 M, x-1 at
  46.3 M, the rest full.
- Eaters 1,045,118 (D7: 425,579 at tick 194,559; eat_eff 12), SIGNAL 911,778, LISTEN 785,782; **167,940 distinct
  genes** (D7 at the fork: 98,555).
- Engine identical to the reference on the RTX 4090 too (5 of 5 ladder configurations). About 2 ticks per second.

## 2026-10-11 13:40 UTC: tick 215,039 (10,240 ticks after the fork)

- 5,400,645 alive (zone 0 3,254,167 / zone 1 2,146,478), still falling (5,617,252 at tick 209,919). D7 at tick
  215,039 (by interpolation of its censuses at 204,799 and 225,279): about 11.56 M.
- Rewarded answers since the fork (rank): x^1 (0) 1,863 M, x^4 (0) 0.61 M, x+4 (0) 33,165, x&12 (0) 215 M, x^5 (1)
  5,899, x+8 (1) 0.41 M, x&3 (1) 1.05 M, 4x (1) 0.44 M, a (1) 30.2 M, x+1 (4) 8,535 M, x|3 (2) 34.6 M, x/4 (2)
  0.44 M, 15-x (3) 40, x-1 (4) 306 M (7.7x in one interval), 2x (4) 79,381, x/2 (4) 0.38 M, parity (4) 85,992,
  a xor b (4) 31.2 M, social (3) 6.29 M (+2.46 M in the interval).
- Pools (full = 46,895,104): x+1 8.78 M (19 %), x^1 41.5 M, x&12 44.4 M, x-1 43.9 M; the others full or nearly.
  Task income since the fork: 34,575.3 G - 34,468.3 G at tick 209,919 = 107 G in the interval.
- Eaters 1,578,858 (+51 %), movers 2,510,331, WAIT 29,766, SIGNAL 339,360 (falling from 911,778), LISTEN 957,998
  (rising): fewer signallers, more listeners. 172,038 distinct genes. New in the top 10: `... LDA SKZ MOVE LISTEN OUT
  IN INC` (102,051), `... LDA SKZ MOVE EAT OUT IN INC` (97,351), `... SKZ EAT LISTEN OUT IN INC`, `... SKNZ SEEK MOVE OUT
  IN FLIP ALLOC` (a FLIP after IN: x^1 computers).
- Pictures (`D8_ladder/animals_t215039.png`): a churning mosaic of hunters (EAT + MOVE, magenta), eaters (red) and
  movers (blue) with many empty slots in zone 1, against D7's static green-blue map.
- Running (dna_world3k, GPU 0 shared); knecht24 28 GB free.

## 2026-10-11 15:40 UTC: ticks 220,159 and 225,279 (15,360 and 20,480 ticks after the fork)

- Population recovering: 5,400,645 (215,039) -> 5,578,272 -> 5,926,389 (zone 0 3,411,271 / zone 1 2,515,118).
  D7 at tick 225,279: 11,565,052.
- Rewarded answers since the fork at tick 225,279 (rank): x^1 (0) 5,979 M, x^4 (0) 7.26 M, x+4 (0) 2.94 M, x&12 (0)
  2,179 M, x^5 (1) 0.22 M, x+8 (1) 8.49 M, x&3 (1) 37.0 M, 4x (1) 3.61 M, a (1) 47.5 M, x+1 (4) 18,206 M, x|3 (2)
  62.2 M, x/4 (2) 3.62 M, 15-x (3) 186, **x-1 (4) 3,064 M**, 2x (4) 0.17 M, x/2 (4) 3.57 M, parity (4) 0.87 M,
  a xor b (4) 81.1 M, social (3) 13.75 M.
- x-1 per 5,120 ticks: 266 M, 870 M, 1,888 M (accelerating). D7's x-1 rate: 0.585 G per 10,240 ticks in a population
  twice as large. 
- Pools (full 46,895,104): x+1 4.39 M (9 %), x^1 25.3 M (54 %), x-1 29.4 M (63 %), x&12 33.5 M (71 %); the rest
  nearly full. Task income in the last interval: 34,812.0 G - 34,679.1 G = 132.9 G.
- Eaters 2,689,391 (45 % of the living), movers 3,180,648, WAIT 59,927, SIGNAL 196,874 (falling), LISTEN 1,263,481
  (rising); 204,674 distinct genes (rising). Top 10 led by `... LDA SKZ EAT LISTEN OUT IN INC` (70,092), with x^1
  computers (`... OUT IN FLIP ...`) and x-1 computers (`... OUT IN INC INC INC`) among them.
- Running (dna_world3k); knecht24 27 GB free.

## 2026-10-11 17:40 UTC: tick 230,399 (25,600 ticks after the fork)

- 6,138,467 alive (zone 0 3,399,938 / zone 1 2,738,529), growing again. D7 at tick 235,519: 11,567,708.
- Rewarded answers since the fork (rank): x^1 (0) 9,179 M, x^4 (0) 13.6 M, x+4 (0) 6.38 M, x&12 (0) 4,482 M,
  x^5 (1) 2.53 M, x+8 (1) 20.6 M, x&3 (1) 91.8 M, 4x (1) 7.05 M, a (1) 56.0 M, x+1 (4) 22,715 M, x|3 (2) 74.7 M,
  x/4 (2) 7.06 M, 15-x (3) 280, x-1 (4) 5,921 M, 2x (4) 0.25 M, x/2 (4) 7.03 M, parity (4) 2.20 M, a xor b (4)
  138.9 M, social (3) 19.76 M.
- x-1 per 5,120 ticks: 266 M, 870 M, 1,888 M, 2,857 M. a xor b per interval: 13 M, 18 M, 32 M, 58 M.
- Pools (full 46,895,104): x+1 3.93 M (8 %), x^1 16.4 M (35 %), x-1 22.9 M (49 %), x&12 27.7 M (59 %), x&3 46.5 M,
  a xor b 46.5 M. Task income in the interval: 34,981.8 G - 34,812.0 G = 169.9 G (rising: 107 G, 133 G, 170 G).
- Eaters 3,237,766 (53 % of the living), movers 3,257,491, WAIT 63,736, SIGNAL 206,219, LISTEN 1,263,699; 223,392
  distinct genes (rising). Top 10: `... SKZ EAT LISTEN OUT IN INC` leads (52,396); x^1 and x-1 computers with SEEK MOVE
  EAT among them.
- Slower now: one census per ~2 h (about 0.7 ticks per second; GPU 0 shared). knecht24 26 GB free.

## 2026-10-11 19:40 UTC: tick 240,639 (35,840 ticks after the fork)

- 6,080,894 alive (zone 0 3,332,572 / zone 1 2,748,322); 6,138,467 at tick 230,399. D7 at tick 235,519: 11,567,708.
- Rewarded answers since the fork (rank): x^1 (0) 16,831 M, x^4 (0) 30.2 M, x+4 (0) 18.3 M, x&12 (0) 10,899 M,
  x^5 (1) 12.7 M, x+8 (1) 82.0 M, x&3 (1) 380.7 M, 4x (1) 16.4 M, a (1) 73.2 M, x+1 (4) 31,124 M, x|3 (2) 97.9 M,
  x/4 (2) 16.4 M, 15-x (3) 384, x-1 (4) 12,963 M, 2x (4) 0.60 M, x/2 (4) 16.4 M, parity (4) 9.26 M, a xor b (4)
  417.2 M, social (3) 30.3 M.
- Over the last two intervals (10,240 ticks): x-1 +7,042 M, a xor b +278 M, x&3 +289 M, social +10.5 M.
- Pools (full 46,895,104): x^1 7.60 M (16 %), x+1 4.52 M (10 %), x-1 18.2 M (39 %), x&12 20.9 M (45 %), x&3 46.0 M,
  a xor b 45.8 M. Task income over the two intervals: 35,377.5 G - 34,981.8 G = 395.7 G.
- Eaters 3,969,591 (65 % of the living), movers 2,680,416, WAIT 63,742, SIGNAL 158,376, LISTEN 949,822 (now falling
  too); 248,690 distinct genes (rising). The top 5 are now x^1 computers (`... OUT IN FLIP`), most with EAT and MOVE;
  the x-1 genes left the top 5.
- Running; knecht24 26 GB free. About 0.7 ticks per second.

## 2026-10-11 21:40 UTC: tick 250,879 (46,080 ticks after the fork)

- 6,046,376 alive (zone 0 3,325,065 / zone 1 2,721,311). D7 at tick 245,759 (interpolated) about 11.57 M.
- Rewarded answers since the fork (rank): x^1 (0) 24,955 M, x^4 (0) 48.4 M, x+4 (0) 33.3 M, x&12 (0) 18,147 M,
  x^5 (1) 38.5 M, x+8 (1) 205.7 M, x&3 (1) 1,025 M, 4x (1) 27.4 M, a (1) 90.6 M, x+1 (4) 39,168 M, x|3 (2) 120.6 M,
  x/4 (2) 27.3 M, 15-x (3) 591, x-1 (4) 20,376 M, 2x (4) 1.33 M, x/2 (4) 27.3 M, parity (4) 25.3 M, a xor b (4)
  1,027 M, social (3) 37.6 M.
- Last 10,240 ticks: x-1 +7,412 M, a xor b +610 M (was +278 M), x&3 +645 M, parity +16.0 M, x+8 +124 M, social
  +7.3 M (slowing: +10.5 M before).
- Pools (full 46,895,104): x^1 5.54 M (12 %), x+1 5.71 M (12 %), x-1 17.6 M (38 %), x&12 18.4 M (39 %), x&3 45.3 M,
  a xor b 44.9 M. Task income over 10,240 ticks: 35,809.5 G - 35,377.5 G = 431.9 G.
- Eaters 4,276,746 (71 %), movers 2,048,471 (falling), WAIT 51,569, SIGNAL 109,050 (falling), LISTEN 819,219
  (falling); 271,785 distinct genes (rising). Top 5: x^1 computers (`... OUT IN FLIP`), four of them with EAT.
- Running; knecht24 25 GB free.

## 2026-10-11 23:40 UTC: tick 261,119 (56,320 ticks after the fork)

- 6,057,579 alive (zone 0 3,339,470 / zone 1 2,718,109). D7 at tick 255,999 (interpolated) about 11.57 M.
- Rewarded answers since the fork (rank): x^1 (0) 33,134 M, x^4 (0) 67.6 M, x+4 (0) 55.9 M, x&12 (0) 25,539 M,
  x^5 (1) 85.5 M, x+8 (1) 397.8 M, x&3 (1) 2,093 M, 4x (1) 39.1 M, a (1) 110.8 M, x+1 (4) 46,996 M, x|3 (2) 143.7 M,
  x/4 (2) 38.8 M, 15-x (3) 743, x-1 (4) 27,794 M, 2x (4) 2.52 M, x/2 (4) 38.7 M, parity (4) 52.3 M, a xor b (4)
  2,030 M, social (3) 43.5 M.
- Last 10,240 ticks: x-1 +7,418 M, a xor b +1,003 M (+610 M before), x&3 +1,068 M, x+8 +192 M, x^5 +47 M, parity
  +27 M, social +5.9 M (slowing further).
- Pools (full 46,895,104): x^1 5.25 M (11 %), x+1 6.80 M (14 %), x&12 17.7 M (38 %), x-1 18.2 M (39 %), x&3 44.6 M,
  a xor b 43.9 M. Task income over 10,240 ticks: 36,278.6 G - 35,809.5 G = 469.1 G.
- Eaters 4,491,091 (74 %), movers 1,566,863 (falling), WAIT 47,284, SIGNAL 79,787, LISTEN 804,831; 291,198 distinct
  genes. New in the top 5: `START HARVEST ALLOC COPY TERM DIVIDE IN INC SKZ OUT HARVEST EAT HARVEST EAT EAT` (9,943):
  after IN INC the low cell is 0 exactly when the 4-bit x+1 needs a carry, and SKZ then skips OUT, so the gene answers
  only when its carry-less x+1 is right and its streak is never reset (reading of the gene, not traced).
- Running; knecht24 21 GB free (falling about 1 GB per 2 h with O5, O6 and D8 maps).

## 2026-10-12 01:40 UTC: tick 271,359 (66,560 ticks after the fork)

- 6,089,851 alive (zone 0 3,354,454 / zone 1 2,735,397). D7 at tick 266,239: 11,573,153.
- Rewarded answers since the fork (rank): x^1 (0) 41,291 M, x^4 (0) 93.5 M, x+4 (0) 90.9 M, x&12 (0) 32,917 M,
  x^5 (1) 142.8 M, x+8 (1) 634.7 M, x&3 (1) 3,621 M, 4x (1) 51.4 M, a (1) 132.7 M, x+1 (4) 54,686 M, x|3 (2) 167.7 M,
  x/4 (2) 50.7 M, 15-x (3) 934, x-1 (4) 35,177 M, 2x (4) 4.24 M, x/2 (4) 50.6 M, parity (4) 91.9 M, a xor b (4)
  3,457 M, social (3) 48.6 M.
- Last 10,240 ticks: x-1 +7,383 M, a xor b +1,427 M (+1,003 M before), x&3 +1,528 M, x+8 +237 M, x^5 +57 M,
  parity +40 M, social +5.1 M.
- Pools (full 46,895,104): x^1 5.49 M, x+1 7.43 M, x&12 17.8 M, x-1 19.1 M, x&3 43.9 M, a xor b 42.8 M.
  Task income over 10,240 ticks: 36,793.9 G - 36,278.6 G = 515.3 G.
- Eaters 4,683,774 (77 %), movers 1,216,001 (falling), WAIT 45,332, SIGNAL 69,528, LISTEN 818,281; 304,881 distinct
  genes. Top 5: x^1 computers (`... OUT IN FLIP`), the answer-only-when-right x+1 gene (`IN INC SKZ OUT ...`, 10,325),
  and new `... TERM SEEK HARVEST SKNZ OUT IN FLIP` (8,991 and 8,177).
- Running; knecht24 21 GB free.

## 2026-10-12 03:40 UTC: tick 286,719 (81,920 ticks after the fork)

- 6,176,964 alive (zone 0 3,393,200 / zone 1 2,783,764). D7 at tick 286,719 (interpolated) about 11.58 M.
- Rewarded answers since the fork (rank): x^1 (0) 53,458 M, x^4 (0) 147.0 M, x+4 (0) 173.6 M, x&12 (0) 43,913 M,
  x^5 (1) 242.5 M, x+8 (1) 1,076 M, x&3 (1) 6,851 M, 4x (1) 71.3 M, a (1) 167.7 M, x+1 (4) 66,018 M, x|3 (2)
  205.8 M, x/4 (2) 69.8 M, 15-x (3) 1,768, x-1 (4) 46,171 M, 2x (4) 8.07 M, x/2 (4) 69.5 M, parity (4) 180.0 M,
  a xor b (4) 6,518 M, social (3) 55.8 M.
- Per 10,240 ticks over the last 15,360: x-1 +7.3 G, a xor b +2.0 G (1.4 G before), x&3 +2.2 G, x+8 +0.29 G,
  parity +0.06 G, social +4.8 M.
- Pools (full 46,895,104): x^1 6.02 M, x+1 7.76 M, x&12 18.2 M, x-1 20.6 M, x&3 42.8 M, a xor b 40.9 M. Task income:
  37,664.4 G - 36,793.9 G = 870.5 G over 15,360 ticks (580 G per 10,240).
- Eaters 4,934,973 (80 %), movers 907,902 (falling), WAIT 55,769, SIGNAL 70,101, LISTEN 903,251; 325,179 distinct
  genes. Top 5: the answer-only-when-right x+1 gene leads (8,692); x^1 computers with EAT; `... SKZ SEEK MOVE LISTEN
  OUT IN L1` (a gene ending in L1 after IN: the head moves to the high cell).
- Faster: three censuses in 2 h (about 2 ticks per second). knecht24 20 GB free.

## 2026-10-12 05:40 UTC: tick 296,959 (92,160 ticks after the fork)

- 6,234,751 alive (zone 0 3,415,813 / zone 1 2,818,938). D7 at tick 296,959: 11,577,962.
- Rewarded answers since the fork (rank): x^1 (0) 61,518 M, x^4 (0) 192.7 M, x+4 (0) 235.9 M, x&12 (0) 51,184 M,
  x^5 (1) 318.1 M, x+8 (1) 1,414 M, x&3 (1) 9,644 M, 4x (1) 85.9 M, a (1) 193.0 M, x+1 (4) 73,488 M, x|3 (2)
  233.1 M, x/4 (2) 83.6 M, 15-x (3) 2,037, x-1 (4) 53,435 M, 2x (4) 11.6 M, x/2 (4) 83.3 M, parity (4) 260.5 M,
  a xor b (4) 9,212 M, social (3) 60.9 M.
- Last 10,240 ticks: x-1 +7.26 G, a xor b +2.69 G (2.0 G before), x&3 +2.79 G, x+8 +0.34 G, parity +0.08 G,
  x^4 +0.046 G, x+4 +0.062 G, social +5.0 M.
- Pools (full 46,895,104): x^1 6.50 M, x+1 7.85 M, x&12 18.8 M, x-1 21.8 M, a xor b 39.7 M, x&3 42.1 M. Task income
  over 10,240 ticks: 38,313.5 G - 37,664.4 G = 649.1 G (580 G before).
- Eaters 5,080,966 (81 %), movers 784,666 (falling), WAIT 63,680, SIGNAL 64,003, LISTEN 947,409; 336,055 distinct
  genes. Top 5 now led by `... SKZ SEEK MOVE LISTEN OUT IN L1` (7,595); new `... LDA L1 HARVEST SKNZ OUT IN SKNZ`.
- Running; knecht24 19 GB free.

## 2026-10-12 07:40 UTC: tick 307,199 (102,400 ticks after the fork)

- 6,307,733 alive (zone 0 3,448,194 / zone 1 2,859,539). D7 at tick 307,199 (interpolated) about 11.58 M.
- Rewarded answers since the fork (rank): x^1 (0) 69,535 M, x^4 (0) 244.7 M, x+4 (0) 289.1 M, x&12 (0) 58,401 M,
  x^5 (1) 401.2 M, x+8 (1) 1,784 M, x&3 (1) 12,929 M, 4x (1) 101.6 M, a (1) 220.0 M, x+1 (4) 80,914 M, x|3 (2)
  262.3 M, x/4 (2) 98.3 M, 15-x (3) 2,112, x-1 (4) 60,646 M, 2x (4) 15.9 M, x/2 (4) 98.0 M, parity (4) 359.1 M,
  a xor b (4) 12,426 M, social (3) 66.0 M.
- Last 10,240 ticks: x-1 +7.21 G, a xor b +3.21 G (2.69 G before), x&3 +3.29 G, x+8 +0.37 G, parity +0.10 G,
  social +5.1 M.
- Pools (full 46,895,104): x^1 7.01 M, x+1 7.64 M, x&12 19.9 M, x-1 23.2 M, a xor b 38.4 M, x&3 41.3 M. Task income
  over 10,240 ticks: 39,017.7 G - 38,313.5 G = 704.2 G (649 G before).
- Eaters 5,216,798 (83 %), movers 690,922 (falling), WAIT 72,306, SIGNAL 61,502, LISTEN 971,672; 344,564 distinct
  genes. Top 5 stable since the last check.
- Running; knecht24 18 GB free.

## 2026-10-12 ~08:30 UTC: day-7 statistics (tools/dna3_activity.py, 2 shadow replicates)

- D8, 21 op censuses (ticks 209,919 to 312,319), 830,423 distinct operation sequences ever: **open-ended by the three
  criteria of section 9 in both variants.** All components: new adaptive per census Q2 1,470.6, Q4 1,408.4 (criterion
  1 holds), new maximum in Q4 of mean trophic level (2 holds), adaptive diversity in the second half >= 8,560
  against a mean of 8,371 (3 holds). Established components: Q2 994.4, Q4 1,121.6; new maxima in Q4: max tasks and
  mean trophic level; adaptive diversity >= 4,853 (mean 4,964). Caveats: only 102,400 ticks after the fork (the
  verdict can turn with the horizon, section 17.5); the op census's task columns follow the day-6 rules and are
  wrong under the ladder, so "max tasks" does not count; the mean trophic level rises because predation spreads.
- D7, 41 op censuses (ticks 112,639 to 522,239), 56,858 sequences: plateau in both variants (new adaptive per census
  Q2 15.9 -> Q4 5.0; established 4.6 -> 1.2; last census 1 new against the shadow's 75.5).
