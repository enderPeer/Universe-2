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
