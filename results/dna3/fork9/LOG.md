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

## 2026-10-12 14:21 UTC: tick 491,519

- 6,633,486 alive (zone 0 3,523,064 / zone 1 3,110,422); +473,000 since tick 389,119. D7 at tick 491,519: 11,592,809
  alive, 382,793 eaters, 94,569 distinct genes.
- Rewarded answers since D8's fork (rank): x^1 (0) 213.3 G, x^4 (0) 10.0 G, x+4 (0) 14.0 G, x&12 (0) 68.1 G, x^5 (1)
  27.0 G, x+8 (1) 44.8 G, x&3 (1) 38.8 G, 4x (1) 145 M, a (1) 1.75 G, x+1 (4) 214.2 G, x|3 (2) 3.39 G, x/4 (2) 116 M,
  15-x (3) 267,600, x-1 (4) 117.0 G, 2x (4) 29.9 M, x/2 (4) 112 M, parity (4) 370 M, a xor b (4) 13.9 G, social (3)
  868 M.
- Last 10,240 ticks: x^1 +9.85 G, x+1 +7.32 G, x-1 +4.56 G, x+8 +4.21 G (2.10 G at tick 389,119), x^5 +3.14 G
  (1.08 G), x&3 +2.65 G, x+4 +1.51 G, x&12 +1.06 G, x^4 +1.02 G, x|3 +0.27 G, a +0.15 G, a xor b +0.13 G, social
  +57.3 M, 15-x +32,800. Task income 647.5 G per 10,240 ticks (558.4 G at tick 389,119).
- Pools (fraction of full): x+1 0.47, x^1 0.58, x-1 0.81, x+8 0.81, x^5 0.82, x&3 0.89, x+4 0.97, x&12 0.98.
- Eaters 4,007,699 (60.4 %), movers 658,562, WAIT 123,473, SENSE 863,895 (above 10 % since tick 440,319), SIGNAL
  251,571, LISTEN 1,606,236; harvesters 6,507,873; 468,625 distinct genes.
- New in the top 10: more x-1-without-borrow variants (`... OUT IN FLIP INC FLIP`, `TERM HARVEST INC FLIP OUT IN
  FLIP`), `START SKNZ ALLOC COPY TERM DIVIDE IN FLIP OUT`, and `START SENSE R1 COPY TERM DIVIDE IN FLIP OUT ALLOC`
  (2,054; no HARVEST: lives on task income and its SENSE); `... SKNZ MOVE EAT HARVEST OUT IN EAT IN INC`.
- Pictures: traits map with many more cyan areas (plant + SENSE) than at tick 389,119, mostly in zone 1, yellow
  plant + WAIT patches in zone 0, purple (no HARVEST) patches in the south; animals map: red eaters, green plain plants
  and magenta hunters mixed, dark empty bands in the south of zone 1. Running (adler40 GPU 0, 100 %); 89 GB free.
  Checkpoint tick 460,800 being profiled.
- Checkpoint tick 460,800 (6,556,131 alive): x-1: 739 programs (403,739 organisms) screened, none solves the borrow
  inputs; x+1: 29,362 profiled organisms solve the carry (`SKZ SEEK L1 INC OUT IN INC`, 98.8 to 99.3 % correct);
  a xor b: no program. Signals: 234,660 signallers, 1,159,963 listeners with a direction, 0.074 bits about the
  sender's b; social target 25.02 % (listened) vs 25.15 % (control), with a signalling partner 25.69 % vs 25.69 %
  (2,585 cases): no channel.

## 2026-10-13 01:55 UTC: tick 1,279,999 (overnight; D9 has run 15 h without interruption, about 20 ticks/s)

- 6,164,069 alive (zone 0 3,239,203 / zone 1 2,924,866). Series: 6.63 M at tick 604,159, 6.51 M at 701,439, 6.37 M
  at 906,239, 6.25 M at 1,105,919: slowly falling since tick ~600,000.
- Distinct genes 465,903; maximum 499,245 at tick 701,439, then falling (493,515 at 906,239, 478,562 at 1,105,919).
- Eaters 4,763,288 (77 %, rising again from 64 % at tick 604,159), movers 430,582 (falling), WAIT 280,656 (rising),
  SENSE 430,752 (halved since tick 604,159), SIGNAL 409,826 (rising), LISTEN 1,393,090; harvesters 6,105,691.
- Rewarded answers since D8's fork (rank): x^1 (0) 1,289 G, x^4 (0) 110 G, x+4 (0) 198 G, x&12 (0) 172 G, x^5 (1)
  469 G, x+8 (1) 549 G, x&3 (1) 428 G, 4x (1) 2.2 G, a (1) 21.6 G, x+1 (4) 739 G, x|3 (2) 29.3 G, x/4 (2) 494 M,
  15-x (3) 3.84 M, x-1 (4) 413 G, 2x (4) 298 M, x/2 (4) 337 M, parity (4) 576 M, a xor b (4) 30.2 G, social (3)
  6.58 G.
- Last 10,240 ticks: x^1 +17.6 G, x+8 +6.94 G, x+1 +6.72 G, x^5 +6.58 G, x&3 +6.01 G, x-1 +3.58 G, x+4 +2.11 G,
  x&12 +1.24 G, x^4 +1.13 G, x|3 +0.36 G, a +0.29 G, a xor b +0.25 G, social +86.4 M. Task income 577.1 G per 10,240
  ticks (656 G at tick 604,159).
- Pools (fraction of full): x+1 0.42, x^1 0.58, x+8 0.70, x^5 0.71, x&3 0.76, x-1 0.83.
- New in the top 10 overnight: mostly x^1 variants (`OUT IN FLIP ... OUT IN FLIP`, one with SIGNAL between),
  `IN INC OUT` (1,573, plain x+1 without carry), `IN INC SKNZ STA OUT`, eater variants; the largest top gene holds
  fewer than 3,000.
- Pictures: animals map almost uniformly red eaters with green plants, magenta hunters scattered; the larger
  green/magenta areas of the evening are gone. Traits map: cyan (plant + SENSE) reduced to a few patches in the
  corners, yellow plant + WAIT areas in the north-east and south-east.
- Reading: after tick ~700,000 D9 drifts back towards predation (eaters 64 % -> 77 %) with slowly falling population,
  diversity and income; x^1 and the rank-1 tasks still rise, x+1 and x-1 fall slightly.
- Running (adler40 GPU 0, 100 %); **70 GB free** (the copy of dimension42-explorer/L6 to the PC's D: is verified; its
  deletion on adler40, which would free 428 GB, waits for the user). Checkpoint tick 1,280,000 being profiled.
- Checkpoint tick 1,280,000: **a second carry program for x+1** has evolved from the carry skipper:
  `IN INC SKZ OUT HARVEST EAT SKNZ EAT L1 COPY L1 INC OUT` (617 carriers, 100.0 % correct, 2,088 of 2,088 carry inputs
  in the reference; a variant with SETQ / L3 also 100 %). Without a carry SKZ lets the first OUT answer; with a carry
  it skips it, the program runs on, moves the head to the high cell (L1), adds the carry (INC) and answers with the
  second OUT. Pattern counts in the whole world: carry A (`SKZ SEEK L1 INC OUT IN ...`) 54,610, carry B
  (`IN INC SKZ OUT ... L1 INC OUT`) 31,936, plain carry skipper 60,030, x-1 without borrow 339,601.
- x-1: 1,007 programs (475,913 organisms) screened, none solves the borrow. a xor b: no program.
- Signals: 409,826 signallers; the signal now carries 0.29 bits about the sender's b (0.07 at tick 460,800) and
  0.15 bits about its low I/O cell; listeners mostly echo (p_echo_b 0.62). Social target 25.21 % (listened) vs
  25.48 % (control), with a signalling partner 23.7 % vs 25.5 % (2,695 cases): still no channel.

## 2026-10-13 02:22 UTC: tick 1,315,839

- 6,161,296 alive (zone 0 3,235,142 / zone 1 2,926,154); eaters 4,784,418, movers 420,851, WAIT 286,947, SENSE
  422,639, SIGNAL 397,995, LISTEN 1,389,650; 465,606 distinct genes; deepest lineage 10,517. Task income 576.9 G per
  10,240 ticks. Last 10,240 ticks: x^1 +18.2 G, x+8 +6.90 G, x+1 +6.75 G, x^5 +6.60 G, x&3 +6.09 G, x-1 +3.61 G,
  x+4 +2.07 G, x&12 +1.25 G, x^4 +1.12 G, x|3 +0.37 G, a +0.30 G, a xor b +0.26 G. Nearly unchanged since tick
  1,279,999 (no new picture in between, no new top-10 sequence). Top genes are x^1 programs (`OUT IN FLIP ... OUT IN
  FLIP`, 5,249 the largest).
- Disk: 69 GB free. D9's own files grow by about 0.9 GB per hour (op census 3.06 GB, three slot maps 2.31 GB each):
  the 30 GB warning line is about 40 hours away unless the verified L6 copy is deleted on adler40 (428 GB).

## 2026-10-13 04:22 UTC: tick 1,474,559

- 6,158,361 alive (zone 0 3,238,312 / zone 1 2,920,049), nearly flat since tick 1,279,999 (6,164,069). Eaters
  4,787,981 (77.7 %), movers 390,387, WAIT 301,890, SENSE 344,109 (falling), SIGNAL 384,625, LISTEN 1,367,841;
  466,994 distinct genes (465,903 at tick 1,279,999).
- Rewarded answers since D8's fork (rank): x^1 (0) 1,651 G, x^4 (0) 132 G, x+4 (0) 238 G, x&12 (0) 195 G, x^5 (1)
  595 G, x+8 (1) 679 G, x&3 (1) 546 G, 4x (1) 3.5 G, a (1) 27.5 G, x+1 (4) 868 G, x|3 (2) 37.6 G, x/4 (2) 621 M,
  15-x (3) 4.86 M, x-1 (4) 483 G, 2x (4) 390 M, x/2 (4) 422 M, parity (4) 633 M, a xor b (4) 35.3 G, social (3)
  8.26 G.
- Last 10,240 ticks: x^1 +20.4 G, x+1 +6.87 G, x+8 +6.79 G, x^5 +6.67 G, x&3 +6.27 G, x-1 +3.80 G, x+4 +2.03 G,
  x^4 +1.21 G, x&12 +1.20 G, x|3 +0.54 G, a +0.33 G, a xor b +0.29 G, social +89.9 M. Task income 563.6 G per 10,240
  ticks (577.1 G at tick 1,279,999).
- Pools (fraction of full): x+1 0.35, x^1 0.55, x^5 0.70, x+8 0.72, x&3 0.75, x-1 0.80.
- New in the top 10: two more x^1 programs with SIGNAL (`START COPY ALLOC ... OUT IN FLIP SIGNAL OUT IN FLIP`,
  `START SIGNAL ALLOC ... OUT IN FLIP HARVEST OUT IN FLIP`).
- Pictures: as at tick 1,279,999: eaters (red) almost everywhere with green plants and scattered magenta hunters; traits
  map green with cyan (plant + SENSE) and yellow (plant + WAIT) patches in the corners.
- Watcher: the 04:21 and 06:22 local runs failed to write `D9_answer.raw.txt.gz` (Windows OSError 22, a brief file
  lock) and kept the old series; reran successfully, and `tools/dna3_watch.py` now retries such writes.
- Running (adler40 GPU 0, 100 %); 68 GB free (L6 still on adler40). Checkpoint tick 1,433,600 being profiled.
- Checkpoint tick 1,433,600 (6,154,069 alive): x+1 with carry in 11,081 profiled organisms; besides carry B
  (`IN INC SKZ OUT ... L1 INC OUT`, 100 %) a third form: `START COPY ALLOC COPY TERM DIVIDE SKNZ L3 EAT R3 R1 R1 INC OUT
  HARVEST IN INC EAT IN` (435 carriers, 100.0 % correct, 1,228 of 1,228 carry inputs). x-1: 1,058 programs (527,301
  organisms), none solves the borrow. a xor b: no program. Signals: 367,835 signallers; the signal carries 0.39 bits
  about the sender's b (0.29 at tick 1,280,000), 0.21 bits about its low I/O cell; listeners echo (0.69). Social target
  24.73 % (listened) vs 25.26 % (control), with a signalling partner 23.7 % vs 24.8 % (1,725 cases): no channel.

## 2026-10-13 06:22 UTC: tick 1,633,279

- 6,151,694 alive (zone 0 3,236,840 / zone 1 2,914,854): flat (6,158,361 at tick 1,474,559). Eaters 4,777,449
  (77.7 %), movers 398,004, WAIT 296,088, SENSE 325,154 (still falling), SIGNAL 373,022, LISTEN 1,403,819; 468,650
  distinct genes.
- Rewarded answers since D8's fork (rank): x^1 (0) 1,979 G, x^4 (0) 151 G, x+4 (0) 268 G, x&12 (0) 214 G, x^5 (1)
  699 G, x+8 (1) 784 G, x&3 (1) 644 G, 4x (1) 4.8 G, a (1) 32.9 G, x+1 (4) 975 G, x|3 (2) 47.6 G, x/4 (2) 734 M,
  15-x (3) 5.67 M, x-1 (4) 543 G, 2x (4) 477 M, x/2 (4) 500 M, parity (4) 680 M, a xor b (4) 39.9 G, social (3)
  9.69 G.
- Last 10,240 ticks: x^1 +21.6 G, x+1 +6.99 G, x+8 +6.79 G, x^5 +6.75 G, x&3 +6.40 G, x-1 +3.90 G, x+4 +1.97 G,
  x&12 +1.23 G, x^4 +1.22 G, x|3 +0.71 G (+0.36 G at tick 1,279,999), a +0.36 G, a xor b +0.31 G, social +94.1 M.
  Task income 564.3 G per 10,240 ticks (563.6 G at tick 1,474,559).
- Pools (fraction of full): x+1 0.32, x^1 0.54, x^5 0.71, x+8 0.73, x&3 0.74, x-1 0.79.
- New in the top 10: more x^1 programs carrying SIGNAL in the copy loop (`START COPY ALLOC SIGNAL TERM DIVIDE OUT IN
  FLIP HARVEST OUT IN FLIP`, `START HARVEST ALLOC SIGNAL TERM DIVIDE OUT IN FLIP COPY OUT IN FLIP`) and an x-1
  without borrow (`TERM DIVIDE INC FLIP OUT IN FLIP`).
- Pictures unchanged in character (traits: green with cyan and yellow patches at the corners; animals: red eaters with
  green and magenta). D9 is in a steady state since about tick 1,200,000: population, income and diversity flat,
  the x^1 and x|3 counts still rising.
- Running (adler40 GPU 0, 100 %); 66 GB free (L6 still on adler40). Checkpoint tick 1,587,200 being profiled.
- Checkpoint tick 1,587,200 (6,155,104 alive): x+1 with carry in 8,384 profiled organisms (carry A and B forms,
  98.9 to 100 %); x-1: 1,150 programs (576,804 organisms), none solves the borrow; a xor b: no program. Signals:
  367,745 signallers; the signal carries 0.55 bits about the sender's b (0.39 at tick 1,433,600; 0.07 at 460,800) and
  0.30 bits about its low I/O cell; listeners echo (0.71). Social target 25.17 % (listened) vs 24.89 % (control),
  with a signalling partner 23.5 % vs 24.0 % (1,847 cases): no channel.

## 2026-10-13 08:22 UTC: tick 1,791,999

- 6,142,268 alive (zone 0 3,234,798 / zone 1 2,907,470); eaters 4,763,247 (77.5 %), movers 381,546, WAIT 282,858,
  SENSE 336,912, SIGNAL 373,757, LISTEN 1,420,562; 474,413 distinct genes (468,650 at tick 1,633,279).
- Rewarded answers since D8's fork (rank): x^1 (0) 2,315 G, x^4 (0) 170 G, x+4 (0) 299 G, x&12 (0) 233 G, x^5 (1)
  803 G, x+8 (1) 890 G, x&3 (1) 743 G, 4x (1) 6.4 G, a (1) 38.4 G, x+1 (4) 1,084 G, x|3 (2) 60.2 G, x/4 (2) 851 M,
  15-x (3) 6.41 M, x-1 (4) 604 G, 2x (4) 568 M, x/2 (4) 582 M, parity (4) 729 M, a xor b (4) 44.7 G, social (3)
  11.1 G.
- Last 10,240 ticks: x^1 +21.9 G, x+1 +7.07 G, x+8 +6.92 G, x^5 +6.66 G, x&3 +6.38 G, x-1 +3.98 G, x+4 +2.01 G,
  x&12 +1.15 G, x^4 +1.10 G, x|3 +0.96 G (rising: 0.71 G at tick 1,633,279), a +0.37 G, a xor b +0.31 G, 4x +0.11 G,
  social +92.7 M. Task income 567.0 G per 10,240 ticks.
- Pools (fraction of full): x+1 0.31, x^1 0.54, x^5 0.72, x+8 0.72, x&3 0.75, x-1 0.78, x|3 0.97 (first drawn down).
- No new sequence in the top 10. Animals map unchanged in character (red eaters, green plants, magenta hunters).
  Steady state continues; x|3 (rank 2) is the one task still rising clearly.
- Running (adler40 GPU 0, 100 %); 65 GB free (L6 still on adler40). Checkpoint tick 1,740,800 being profiled.
- Checkpoint tick 1,792,000 (copied after it overwrote 1,740,800; 6,142,268 alive): a **shorter carry program** for
  x+1: `START HARVEST ALLOC COPY TERM DIVIDE SKZ L3 L1 INC OUT IN INC` (472 carriers, 100.0 % correct, 8,135 of 8,135
  carry inputs; without a carry L3 moves the head 3 cells left, off the I/O cells, so L1 INC changes nothing answered;
  with a carry SKZ skips L3 and L1 INC adds the carry), next to carry B (100 %) and the SKNZ form (100 %); 7,629
  profiled organisms solve the carry. x-1: 1,163 programs (578,195 organisms), none solves the borrow; a xor b: no
  program. Signals: 0.65 bits about the sender's b (0.55 at tick 1,587,200), 0.37 bits about its low I/O cell; social
  target 24.90 % (listened) vs 24.74 % (control), with a signalling partner 23.4 % vs 23.5 % (1,713 cases): no channel.

## 2026-10-13 10:22 UTC: tick 1,950,719

- 6,120,798 alive (zone 0 3,219,848 / zone 1 2,900,950); eaters 4,782,109 (78.1 %), movers 340,654 (falling), WAIT
  282,687, SENSE 370,541, SIGNAL 371,985, LISTEN 1,463,157; 480,371 distinct genes (rising slowly: 474,413 at tick
  1,791,999).
- Rewarded answers since D8's fork (rank): x^1 (0) 2,649 G, x^4 (0) 186 G, x+4 (0) 330 G, x&12 (0) 250 G, x^5 (1)
  906 G, x+8 (1) 997 G, x&3 (1) 840 G, 4x (1) 8.1 G, a (1) 44.0 G, x+1 (4) 1,194 G, x|3 (2) 79.4 G, x/4 (2) 966 M,
  15-x (3) 7.12 M, x-1 (4) 665 G, 2x (4) 653 M, x/2 (4) 667 M, parity (4) 775 M, a xor b (4) 49.4 G, social (3)
  12.6 G.
- Last 10,240 ticks: x^1 +21.3 G, x+1 +7.06 G, x+8 +6.86 G, x^5 +6.61 G, x&3 +6.26 G, x-1 +3.95 G, x+4 +1.86 G,
  x|3 +1.49 G (0.96 G at tick 1,791,999, still rising), x&12 +1.19 G, x^4 +1.04 G, a +0.35 G, a xor b +0.30 G,
  4x +0.10 G, social +90.0 M. Task income 574.6 G per 10,240 ticks.
- Pools (fraction of full): x+1 0.30, x^1 0.56, x^5 0.73, x+8 0.74, x&3 0.75, x-1 0.78, x|3 0.96.
- New in the top 10: another x^1 program with SIGNAL (`START COPY ALLOC HARVEST TERM DIVIDE OUT IN FLIP SIGNAL OUT IN
  FLIP`). Traits map: the cyan (plant + SENSE) patches are spread more evenly than before, the yellow (plant + WAIT)
  corner patches are smaller.
- Running (adler40 GPU 0, 100 %); 63 GB free (L6 still on adler40). Latest checkpoint being profiled.
- Checkpoint tick 1,945,600 (6,119,631 alive): x+1 with carry in 9,274 profiled organisms (carry A, the short
  `SKZ L3 L1 INC OUT IN INC` at 100 %); x-1: 1,185 programs (606,800 organisms), none solves the borrow; a xor b: no
  program. Signals: 0.76 bits about the sender's b (0.65 at tick 1,792,000), 0.44 bits about its low I/O cell;
  social target 24.90 % (listened) vs 25.08 % (control), with a signalling partner 23.5 % vs 23.6 % (1,600 cases):
  no channel.

## 2026-10-13 12:22 UTC: tick 2,109,439

- 6,123,212 alive (zone 0 3,206,570 / zone 1 2,916,642); eaters 4,771,924 (77.9 %), movers 304,063 (falling), WAIT
  293,882, SENSE 359,336, SIGNAL 371,004, LISTEN 1,543,209 (rising); 487,181 distinct genes (rising: 480,371 at
  tick 1,950,719).
- Rewarded answers since D8's fork (rank): x^1 (0) 2,982 G, x^4 (0) 200 G, x+4 (0) 357 G, x&12 (0) 268 G, x^5 (1)
  1,008 G, x+8 (1) 1,103 G, x&3 (1) 937 G, 4x (1) 9.7 G, a (1) 49.5 G, x+1 (4) 1,303 G, x|3 (2) 107 G, x/4 (2)
  1.08 G, 15-x (3) 7.79 M, x-1 (4) 727 G, 2x (4) 735 M, x/2 (4) 752 M, parity (4) 819 M, a xor b (4) 54.1 G, social
  (3) 13.9 G.
- Last 10,240 ticks: x^1 +21.6 G, x+1 +7.09 G, x+8 +6.78 G, x^5 +6.59 G, x&3 +6.21 G, x-1 +4.02 G, x|3 +2.16 G
  (1.49 G at tick 1,950,719: still rising), x+4 +1.65 G, x&12 +1.12 G, x^4 +0.89 G, a +0.36 G, a xor b +0.30 G,
  4x +0.11 G, social +87.8 M. Task income 588.8 G per 10,240 ticks (574.6 G).
- Pools (fraction of full): x+1 0.30, x^1 0.56, x^5 0.74, x+8 0.74, x&3 0.76, x-1 0.77, x|3 0.93.
- No new top-10 sequence. Animals map unchanged in character. Steady state with slowly rising diversity, income and
  x|3.
- Running (adler40 GPU 0, 100 %); 62 GB free (L6 still on adler40). Latest checkpoint being profiled.
- Checkpoint tick 2,099,200 (6,126,467 alive): x+1 with carry in 5,258 profiled organisms (the short
  `SKZ L3 L1 INC OUT IN INC` in two spellings, 100 %; a carry-B form, 100 %); x-1: 1,220 programs (629,971
  organisms), none solves the borrow; a xor b: no program. Signals: 0.74 bits about the sender's b, 0.42 bits about
  its low I/O cell; social target 24.86 % (listened) vs 25.40 % (control); with a signalling partner 25.7 % vs 23.5 %
  (2,113 cases, a difference of about 1.7 standard errors, not significant): no channel shown.

## 2026-10-13 14:22 UTC: tick 2,268,159

- 6,132,163 alive (zone 0 3,205,875 / zone 1 2,926,288); eaters 4,768,379 (77.8 %), movers 319,330, WAIT 282,445,
  SENSE 380,898, SIGNAL 415,229, LISTEN 1,655,592 (rising); 493,260 distinct genes (rising).
- Rewarded answers since D8's fork (rank): x^1 (0) 3,314 G, x^4 (0) 212 G, x+4 (0) 381 G, x&12 (0) 285 G, x^5 (1)
  1,109 G, x+8 (1) 1,205 G, x&3 (1) 1,033 G, 4x (1) 11.6 G, a (1) 55.2 G, x+1 (4) 1,414 G, x|3 (2) 148 G, x/4 (2)
  1.20 G, 15-x (3) 8.40 M, x-1 (4) 789 G, 2x (4) 816 M, x/2 (4) 841 M, parity (4) 862 M, a xor b (4) 58.8 G, social
  (3) 15.3 G.
- Last 10,240 ticks: x^1 +21.6 G, x+1 +7.13 G, x+8 +6.55 G, x^5 +6.51 G, x&3 +6.22 G, x-1 +3.96 G, x|3 +3.09 G
  (2.16 G at tick 2,109,439), x+4 +1.42 G, x&12 +1.05 G, x^4 +0.70 G, a +0.38 G, a xor b +0.31 G, 4x +0.13 G, social
  +87.5 M. Task income 600.5 G per 10,240 ticks (588.8 G).
- Pools (fraction of full): x+1 0.29, x^1 0.56, x^5 0.74, x+8 0.75, x&3 0.77, x-1 0.77, x|3 0.89.
- New in the top 10: three more x^1 programs (`OUT IN FLIP OUT IN FLIP` and copy-loop variants).
- Running (adler40 GPU 0, 100 %); 60 GB free (L6 still on adler40). adler40's second card (RTX 4080, GPU 1) no
  longer answers nvidia-smi ("Unable to determine the device handle ... Unknown Error"); D9 on the 4090 is not
  affected.

## 2026-10-13 16:22 UTC: tick 2,431,999

- 6,149,792 alive (zone 0 3,198,091 / zone 1 2,951,701); eaters 4,786,598 (77.8 %), movers 308,871, WAIT 254,894,
  SENSE 407,924, SIGNAL 482,077 (rising), LISTEN 1,743,762 (rising); 492,767 distinct genes.
- Rewarded answers since D8's fork (rank): x^1 (0) 3,668 G, x^4 (0) 222 G, x+4 (0) 404 G, x&12 (0) 302 G, x^5 (1)
  1,213 G, x+8 (1) 1,310 G, x&3 (1) 1,133 G, 4x (1) 13.7 G, a (1) 61.5 G, x+1 (4) 1,528 G, x|3 (2) 204 G, x/4 (2)
  1.32 G, 15-x (3) 8.95 M, x-1 (4) 853 G, 2x (4) 906 M, x/2 (4) 936 M, parity (4) 907 M, a xor b (4) 63.9 G, social
  (3) 16.7 G.
- Last 10,240 ticks: x^1 +22.3 G, x+1 +7.15 G, x+8 +6.52 G, x^5 +6.40 G, x&3 +6.22 G, x-1 +3.98 G, x|3 +3.84 G
  (3.09 G at tick 2,268,159), x+4 +1.43 G, x&12 +1.06 G, x^4 +0.53 G, a +0.40 G, a xor b +0.32 G, 4x +0.15 G, social
  +88.0 M. Task income 610.9 G per 10,240 ticks (600.5 G).
- Pools (fraction of full): x+1 0.29, x^1 0.56, x+8 0.75, x^5 0.76, x-1 0.76, x&3 0.77, x|3 0.86.
- No new top-10 sequence. Traits map: green with evenly scattered cyan (plant + SENSE) and yellow (plant + WAIT);
  the corner patches of the morning have dissolved.
- Running (adler40 GPU 0, 100 %); 486 GB free after the L6 deletion. The RTX 4080 (GPU 1) still does not answer
  nvidia-smi. Latest checkpoint being profiled (with a screen of the x|3 programs).
- Checkpoint tick 2,406,400 (6,149,886 alive, 270,553 operation sequences): **x|3 is computed by programs that use
  LISTEN as a source of zero**: `... IN HARVEST LISTEN <operand> STA INC INC INC` (348, 308, 299 carriers in the
  profile, 100 % correct in the reference, where the neighbours are silent clones): LISTEN loads a silent neighbour's
  signal (0) into A, STA writes it into the low cell, three INCs make it 3, the high cell stays: x|3. In D9 this works
  only next to neighbours that do not signal a non-zero value. x+1 with carry: 6,288 profiled organisms (three forms,
  100 %); x-1: 1,244 programs, none solves the borrow; a xor b: no program. Signals: social target 24.59 % (listened)
  vs 25.20 % (control), with a signalling partner 25.3 % vs 24.4 % (2,889 cases): no channel.
- Energy over the last 10,240 ticks (ledger, G): income light 43.5, harvest 254.7, tasks 610.8; spent on
  instructions 616.8, copying 37.3, maintenance 64.2; overflow (energy above emax, lost) 302.9; moved by predation
  713.9 with 189.7 trophic loss.
- Gene length (START to the codon before STOP): mean 16.9 codons, median 16; the most common gene (358,648 organisms)
  is `START HARVEST ALLOC COPY TERM DIVIDE IN FLIP INC FLIP OUT` (11 codons), x-1 without borrow.

## 2026-10-13 18:22 UTC: tick 2,595,839

- 6,171,019 alive (zone 0 3,201,816 / zone 1 2,969,203); eaters 4,783,781 (77.5 %), movers 302,933, WAIT 228,860
  (falling), SENSE 423,847, SIGNAL 504,009 (rising), LISTEN 1,845,992 (rising); 490,936 distinct genes.
- Rewarded answers since D8's fork (rank): x^1 (0) 4,026 G, x^4 (0) 230 G, x+4 (0) 425 G, x&12 (0) 319 G, x^5 (1)
  1,316 G, x+8 (1) 1,413 G, x&3 (1) 1,231 G, 4x (1) 16.1 G, a (1) 68.2 G, x+1 (4) 1,642 G, x|3 (2) 272 G, x/4 (2)
  1.45 G, 15-x (3) 9.43 M, x-1 (4) 916 G, 2x (4) 1.00 G, x/2 (4) 1.04 G, parity (4) 953 M, a xor b (4) 69.1 G, social
  (3) 18.1 G.
- Last 10,240 ticks: x^1 +22.8 G, x+1 +7.16 G, x^5 +6.49 G, x+8 +6.37 G, x&3 +6.13 G, x|3 +4.52 G (3.84 G at tick
  2,431,999), x-1 +3.99 G, x+4 +1.29 G, x&12 +1.07 G, x^4 +0.48 G, a +0.44 G, a xor b +0.34 G, 4x +0.15 G, social
  +93.0 M. Task income 620.6 G per 10,240 ticks (610.9 G).
- Pools (fraction of full): x+1 0.28, x^1 0.54, x^5 0.75, x+8 0.77, x-1 0.77, x&3 0.78, x|3 0.83.
- New in the top 10: `START SIGNAL ALLOC HARVEST TERM DIVIDE OUT IN FLIP COPY OUT IN FLIP` (x^1). Animals map:
  red eaters with green plants and magenta hunters, a few small white/magenta clusters in zone 1.
- Running (adler40 GPU 0, 100 %); 484 GB free; the RTX 4080 still offline. Latest checkpoint being profiled.
