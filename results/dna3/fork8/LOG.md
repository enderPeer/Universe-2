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
