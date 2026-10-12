# D9_answer: D8 with the abstention loophole closed (docs/17 section 18.7)

Measured numbers only. D9 = D8's checkpoint at tick 307,200 continued with D8's arguments plus `--answer-all` (an IN
while the last input is unanswered resets every streak). Engine `~/universe2/dna3/dna_world3l` (sm_86), directory
`~/universe2/dna3/fork9` on knecht24, RTX 3060 GPU 2 (shared with O5_day5). Task counts in the census are cumulative
since D8's fork at tick 204,800; D9's own rewards are the increase after tick 307,200 (values at tick 307,199 in
`results/dna3/fork8/LOG.md`).

Command: `MODES=plant,traits,animals sh ../dna3_open.sh D9_answer ../dna_world3l 2 5120 51200 "--resume
D8_t307200.ckpt" --days 1,2,3,4,5,6 --N 3001286656 --G 256 --W 3424 --land-cell 214 --seed 7 --mu 0.001 --shade --T 0
--ladder --answer-all --eat-eff 12 --op-census D9_answer.ops` (`D8_t307200.ckpt` is a hard link to D8's checkpoint of
tick 307,200, kept when D8 writes its next one).

## 2026-10-12 08:59 UTC: start

- Verified first: 10 of 10 configurations identical engine vs reference (`results/dna3/verify_aa/`).
- Resumed from tick 307,200 (6,307,733 alive); 5.4 GB on GPU 2. knecht24 17 GB free before D9's first checkpoint.

## 2026-10-12 11:15 UTC: moved to adler40's RTX 4090

- On knecht24 GPU 2 (RTX 3060, alone after O5 stopped): 5,120 ticks from 10:40:31 to 11:08:46 UTC, **3.0 ticks/s**.
- adler40 rebooted (driver/library mismatch cleared); D9 restarted there from the same checkpoint (tick 307,200,
  `dna_world3l` built for sm_89 from the same source, md5 6234fdb5...), started 10:58:43 UTC, reached tick 322,559 at
  11:11:38: **about 20 ticks/s** including the load (6.6 times the 3060).
- Census lines of ticks 312,319, 317,439 and 322,559 on the 4090 and on the 3060: **3 of 3 identical**. The knecht24
  copy was stopped at tick 327,679+ (its files in `knecht24:~/universe2/dna3/fork9/knecht24_partial/`); D9 continues
  on adler40 GPU 0, `~/universe2/dna3/fork9`.

## 2026-10-12 ~12:20 UTC: first checkpoint, tick 358,400 (docs/17 section 18.8)

- 5,764,450 alive. Real x+1 with carry: `SKZ SEEK L1 INC OUT IN INC`, 99.6 % correct, 25,581 of 25,864 carry inputs;
  39,314 carriers (126 in D8 at tick 307,200). Carry skipper `IN INC SKZ OUT` 168,883; x-1 without borrow
  `IN FLIP INC FLIP` 195,046 (75 % correct, 0 of the borrow inputs).
- No signal channel: social target matched 25.20 % (listened neighbour) against 25.18 % (control); signallers 132,199.

## 2026-10-12 12:22 UTC: tick 389,119

- 6,160,363 alive (zone 0 3,352,257 / zone 1 2,808,106), rising every census (5,764,450 at the checkpoint, tick
  358,400). D7 at tick 389,119: 11,587,076 alive, 389,999 eaters, 95,284 distinct genes. D8 at its last census (tick
  343,039, recorded): 6.55 M alive.
- Rewarded answers since D8's fork (rank): x^1 (0) 121.9 G, x^4 (0) 2.59 G, x+4 (0) 3.31 G, x&12 (0) 60.3 G, x^5 (1)
  4.86 G, x+8 (1) 12.3 G, x&3 (1) 18.8 G, 4x (1) 107 M, a (1) 581 M, x+1 (4) 140.3 G, x|3 (2) 1.08 G, x/4 (2) 101 M,
  15-x (3) 42,710, x-1 (4) 74.8 G, 2x (4) 18.1 M, x/2 (4) 100 M, parity (4) 361 M, a xor b (4) 12.8 G, social (3) 323 M.
- Last 10,240 ticks: x^1 +7.90 G, x+1 +7.45 G, x-1 +3.33 G (+1.82 G one window before), x+8 +2.10 G (+1.33 G),
  x&3 +1.22 G, x^5 +1.08 G, x+4 +0.59 G, x^4 +0.45 G, x&12 +0.44 G, x|3 +0.16 G, a xor b +0.071 G, social +45.1 M,
  15-x +11,100. Task income 558.4 G per 10,240 ticks (460 G at tick 353,279, 524 G at 373,759).
- Pools (full 46,895,104): x^1 28.1 M, x+1 20.2 M, x-1 40.3 M, x+8 42.9 M, x^5 43.7 M, x&3 44.7 M; others full.
- Eaters 4,103,896 (66.6 %, falling from 85 % at tick 312,319), movers 824,201 (falling), WAIT 90,752, SIGNAL 159,561
  (rising every census), LISTEN 1,305,926; 429,066 distinct genes (422,125 at tick 373,759).
- New in the top 10 since tick 358,400: x-1-without-borrow programs (`IN FLIP INC FLIP OUT` and variants), the plain
  x^1 program `IN FLIP OUT` (3,213), `START LISTEN ALLOC COPY TERM DIVIDE IN FLIP OUT ALLOC` (3,900) and
  `... SKZ EAT EAT HARVEST OUT IN INC` (2,998).
- Pictures: the animals map is more mixed than D8's (red eaters with many green and magenta areas, magenta and dark
  empty areas in the south of zone 1); the traits map shows more cyan patches and several yellow ones than D8's at the
  same stage. No new checkpoint since tick 358,400 (next 409,600). Running (adler40 GPU 0, 100 %); 92 GB free.
- Predictions of section 18.7: (i) income fell (786 G -> 179 G per 10,240 in D8's terms right after the fork) and is
  recovering; (ii) a xor b / x&3 / x&12 collapsed, x^1 kept rising: as predicted; (iii) honest rank-0/1 tasks
  spread (x+8, x^5, x+4, x^4): as predicted; (iv) a real x+1 with carry appeared within 51,200 ticks: earlier than
  predicted; (v) predation did not rise: the eater share fell from 85 % to 67 %, contrary to the prediction.

## 2026-10-12 ~13:20 UTC: checkpoint tick 409,600 (profiles)

- 6,312,744 alive. Screen of every profiled program that earns a task (>= 5 % of its rewards), answers recorded in the
  reference with `--answer-all`: **x-1: 666 programs (412,964 organisms), none solves the borrow inputs** (all < 10 %
  correct on b = 0); x+1: 30,102 profiled organisms solve the carry (>= 90 %), the carry family
  `SKZ SEEK L1 INC OUT IN INC` holds 56,549 in the whole world (39,314 at tick 358,400); a xor b: no program.
- Whole world: x-1 without borrow (`FLIP INC FLIP`) 373,793 (195,046 at tick 358,400); carry skipper `IN INC SKZ OUT`
  140,332 (168,883).
- Signals: 183,257 signallers, 1,007,844 listeners with a direction; the signal carries 0.087 bits about the sender's
  b (0.145 before); social target matched 24.99 % with the listened neighbour, 24.86 % with the control (with a
  signalling partner 24.1 % vs 25.3 %, 1,830 cases): no channel.
