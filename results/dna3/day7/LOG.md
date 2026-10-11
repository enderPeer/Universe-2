# Day 7 run log (D7_big)

Rules frozen at commit 85eed02 (binary dna_world3h): days 1 to 6, `--shade --T 0`, mu 0.001, 3,001,286,656 cells
(3,424 x 3,424 slots of 256 cells, land-cell 214, seed 7), 1,000,000 plant seeds (`ATG AGA GCT CTT TTA GGT`) in the
sea, census and slot maps (plant, traits, animals) every 10,240 ticks, checkpoint every 102,400 ticks, on adler40's
RTX 4090 from 2026-10-10 22:27 UTC. First planned to end Monday 2026-10-12 09:50 Berlin; on 2026-10-11 the user asked
to let it run open-ended. At its first checkpoint (tick 102,400) it switches to dna_world3i with `--op-census D7_big.ops`
(output only, built by the day-7 measurement session) and from then on has no end.
Picture and emergence checks every 2 hours; entries below give only measured numbers.

