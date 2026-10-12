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
