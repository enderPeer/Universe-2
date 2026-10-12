# Stopped open-ended worlds

## 2026-10-12 09:57 UTC: O5_day5, O6_day6, O7_day7 stopped (the user's decision, to free GPUs)

All three can be continued with their original `dna3_open.sh` commands (each resumes from `NAME.ckpt`).

| world | node, card | last census tick | checkpoint (UTC) | directory |
|---|---|---:|---|---|
| O5_day5 | knecht24 RTX 3060 GPU 2 (`dna_world3g`) | 45,875,199 | 2026-10-12 09:52 | `~/universe2/dna3/open` |
| O6_day6 | knecht24 RTX 3060 GPU 1 (`dna_world3h`) | 29,450,239 | 2026-10-12 09:48 | `~/universe2/dna3/open` |
| O7_day7 | adler40 RTX 4080 GPU 1 (`dna_world3i`, `--op-census`) | 36,362,239 | 2026-10-12 09:52 | `~/universe2/dna3/open` (adler40) |

After the stop: knecht24 GPU 1 free, GPU 2 runs D9_answer alone; adler40's 4080 free (new CUDA processes on adler40
need the reboot planned after D7's checkpoint at tick 614,400).
