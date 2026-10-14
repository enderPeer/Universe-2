# D10_listen: D9 with an easy social task (docs/17 section 18.9)

Measured numbers only. D10 = D9's checkpoint at tick 3,891,200 continued with D9's arguments plus `--social-copy` (the
social task asks for the listened neighbour's b itself). Engine `~/universe2/dna3/dna_world3n` (sm_89, source md5
a3b860a9...), directory `~/universe2/dna3/fork10` on adler40, RTX 4090 (GPU 0). D9 runs unchanged on the RTX 5090
from the same checkpoint. Task counts are cumulative since D8's fork; D10's own rewards are the increase after tick
3,891,200 (D9's values at that tick are the baseline).

Command: `MODES=plant,traits,animals sh ../dna3_open.sh D10_listen ../dna_world3n 0 5120 51200 "--resume
D9_t3891200.ckpt" --days 1,2,3,4,5,6 --N 3001286656 --G 256 --W 3424 --land-cell 214 --seed 7 --mu 0.001 --shade --T 0
--ladder --answer-all --social-copy --eat-eff 12 --op-census D10_listen.ops`

## 2026-10-14 14:33 UTC: start

- Verified first: 12 of 12 configurations identical engine vs reference (`results/dna3/verify_sc/`).
- Resumed from tick 3,891,200 (D9's state); RTX 4090 at 100 %, 5.8 GB.
