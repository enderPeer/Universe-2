# Runbook: the A/B/C experiment of docs/17 section 19.3 (macros against the plateau)

Everything below is prepared on the branch; nothing has run on the cluster yet. Steps 1 and 2 are the gate: the
engine must compile and pass the verification before any world is started. D7 (RTX 4090, adler40) and D8 (knecht24
GPU 0) are not touched. Record what happened in `LOG.md` next to this file, measured numbers only, as in
`results/dna3/day7/LOG.md`.

## 0. Check out and build (both nodes)

```sh
cd ~/universe2 && git fetch origin && git checkout claude/beautiful-rubin-xb2wn3 && git pull   # or main once merged
nvcc -O3 -std=c++17 -arch=sm_86 -o ~/universe2/dna3/dna_world3m gpu/dna_world.cu      # knecht24 (RTX 3060)
nvcc -O3 -std=c++17 -arch=sm_89 -o ~/universe2/dna3/dna_world3m gpu/dna_world.cu      # adler40 (RTX 4090 / 4080)
```

The new binary is `dna_world3m` (macros). It reproduces every earlier configuration (step 1 checks that), reads
version-3, 4 and 5 checkpoints, writes version 5 only when `--macros` is on. Two new flags: `--macros FILE` and
`--in-fresh` (docs/17 section 19.2).

## 1. Verification (about 20 minutes, one GPU)

From the repository on a machine that can ssh to the node (as before), or on the node itself with `--local`:

```sh
python tools/dna3_verify.py --host knecht24 --bin '~/universe2/dna3/dna_world3m' --gpu 1 --out results/dna3/verify
python tools/dna3_verify.py --local --bin ~/universe2/dna3/dna_world3m --gpu 1 --out results/dna3/verify   # on the node
```

Expected: **44 of 44 configurations identical** (36 earlier ones, 5 macro worlds, 3 checkpoint tests). The same
suite passed here against a CPU emulation of the engine (`results/dna3/verify_cpu/verify.json`); the GPU run is the
real one. If anything differs, stop: do not start the worlds. Commit `results/dna3/verify/verify.json`.

## 2. The echo loophole check (optional, 2 minutes)

`--in-fresh` closes a loophole found while testing (section 19.4): without it every IN of a tick draws the same
input. The verification covers the rule; this just shows it on a GPU world:

```sh
~/universe2/dna3/dna_world3m --days 1,2,3,4,5,6 --ladder --shade --T 0 --W 32 --land-cell 8 --N 262144 --G 256 --ticks 2000 \
    --seed 3 --n-seed 16 --seed-terrain sea --genome "ATG AGA GCT CTT TTA GGT TCA TCG TCA TCG TCA TCG TCA TCG" --census-every 1000 --gpu 1
```

The `tasks` list of the last census shows what an `IN OUT IN OUT ...` echo earns per 1,000 ticks; with `--in-fresh`
added the same world should show zeros or single digits.

## 3. Epoch 1: three fresh ladder worlds (about 1.5 h each on the 4080, 3 to 4 h on a 3060)

In a directory for the runs, e.g. `~/universe2/dna3/ab`:

```sh
REPO=~/universe2 sh ~/universe2/tools/dna3_ab.sh epoch1 ~/universe2/dna3/dna_world3m 1 11    # adler40 GPU 1 (4080)
REPO=~/universe2 sh ~/universe2/tools/dna3_ab.sh epoch1 ~/universe2/dna3/dna_world3m 1 12    # knecht24 GPU 1
REPO=~/universe2 sh ~/universe2/tools/dna3_ab.sh epoch1 ~/universe2/dna3/dna_world3m 2 13    # knecht24 GPU 2
```

Each writes `E1_s<seed>.txt` (census), `.top`, `.ops`, `.ckpt`, `.json`, `.log`. Watch with
`tools/dna3_watch.py` or `tail -1 E1_s11.txt`. Which GPUs are free is yours to decide; the script only needs a GPU
index.

## 4. Harvest and promote (minutes, CPU; needs numpy)

```sh
REPO=~/universe2 sh ~/universe2/tools/dna3_ab.sh harvest 11     # and 12, 13
```

Writes `E1_s11_solutions.json` (the library), `E1_s11_macros.json` + `E1_s11_macros.txt` (the table arm A uses) and
`E1_s11_macros_C.txt` (the same codons, empty bodies: arm C). Read the printed table: only verified bodies of
length >= 2 are promoted, hardest tasks first. If the harvest holds no body of length >= 2 (every solution a single
instruction), the world has nothing to promote: note it and let that world run longer before forking.

## 5. Epoch 2: nine forks (3 waves of 3, about 12 h in total)

```sh
for ARM in A B C; do REPO=~/universe2 sh ~/universe2/tools/dna3_ab.sh epoch2 ~/universe2/dna3/dna_world3m 1 11 $ARM; done   # one GPU, sequential
```

or one arm per GPU in parallel. Each arm copies the epoch-1 checkpoint to its own `E2_s<seed>_<ARM>.ckpt` and
continues for 2 x 10^6 ticks. The run header (`.err`) prints the macro table as loaded; check it once.

## 6. Report

```sh
REPO=~/universe2 sh ~/universe2/tools/dna3_ab.sh report 11     # and 12, 13
```

Harvests every arm at its end and runs the day-7 activity measure. Then evaluate the three criteria of section 19.3
from the census series (`tasks`, `macro_users`, `macro_exec`, distinct genes) and the activity output, per pair, and
write the numbers into docs/17 section 19.4. Copy to the repository: the census series (`gzip`ped `.txt` and `.top`),
the `_solutions.json`, `_macros.txt`, `_activity.json` files and `LOG.md`; not the checkpoints or `.ops` (large).

## What was not possible from here

- No GPU and no nvcc in the cloud session: the engine was type-checked with g++ and verified against the reference
  through a CPU emulation (kernel launches as loops, warp sums as plain adds), which exercises every rule but not the
  CUDA build itself. Step 1 is therefore mandatory.
- The D8 harvest (`python tools/dna3_extract_solutions.py --mode ladder --ckpt D8_ladder.ckpt --verify ...` on
  knecht24) was not run; its result belongs in section 19.4.
