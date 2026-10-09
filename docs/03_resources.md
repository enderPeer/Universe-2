# Resources: what we have, what it was doing on 2026-10-09, and how to bound CPU, memory and power

## 1. Inventory (checked over SSH on 2026-10-09, 11:21 node time)

| machine | CPU (threads) | RAM | GPUs | disk free | state at the check |
|---|---|---:|---|---:|---|
| **this workstation** (Windows 11) | i9-13900K (32) | 64 GB | *none usable*: the RTX 4070 Ti SUPER is listed by Windows with problem code 45, "not connected" (unplugged or dead); only the Intel UHD 770 is active | 857 GB | idle; Python 3.11 with capstone, unicorn, numpy, torch-cpu; **no gcc, nvcc, make or WSL** |
| adler40 (192.168.178.171) | i9-12900K (24) | 61 GB | RTX 4090 24 GB, RTX 4080 16 GB | 126 GB (86 % used) | both GPUs 0 %; the 4080 holds 1.3 GB of an old homunculi process; gcc 15.2, CUDA 12.4, glslc, rustc, Python 3.14 |
| knecht24 (192.168.178.200) | Threadripper 1920X (24) | 60 GB | 3x RTX 3060 12 GB (power limits 170 / 110 / 110 W) | 38 GB (83 % used) | idle |
| specht32 (192.168.178.187) | i7-8700K (12) | 44 GB | RX 9070 XT 16 GB, RX 9060 XT 16 GB (Vulkan only) | 222 GB | idle |
| falke64 (192.168.178.188) | Ryzen 5 7600X (12) | 60 GB | 2x Radeon AI PRO R9700 32 GB (Vulkan only) | 1.5 TB | **both GPUs 100 % busy**: Universe-1 exp11, `u1_avida_vk --len 9` (the 26^9 Avida sweep) |

Cluster total: 72 logical CPUs, 9 discrete GPUs, 172 GB VRAM. Access: `ssh adler40` etc. (aliases exist, user
`ender`, key-only). The cluster's full inventory is in Universe-1's `docs/cluster-inventory.md`.

The rules from the admin's access note apply to everything below: **no sudo**, GPUs are shared (look before you
start, never kill a process that is not yours), `df -h` before large writes, no reboots, services bind to the LAN
or localhost only, no keys in commits.

## 2. Bounding CPU use

**Workstation.** The tools here take `--procs N` (default: all 32 logical CPUs). The decode census at 24 processes
measured 4.8 M words/s on AArch64 and 4.0 M words/s on x86-64 (Python + Capstone); fewer processes scale down
nearly linearly. To pin a run to specific cores from PowerShell:

```powershell
$p = Start-Process python -ArgumentList "tools/isa4_census.py --arch aarch64 --range 0 4294967296 --procs 8" -PassThru
$p.ProcessorAffinity = 0x00FF   # cores 0-7
```

**Cluster.** `taskset -c 0-7 ./program` pins to cores, `nice -n 19` yields to other users, `OMP_NUM_THREADS=8` bounds
the OpenMP tools (Dimension42's `nano_cpu` takes the thread count as an argument).

## 3. Bounding memory

All tools here stream fixed-size chunks (`--chunk`, default 2^18 words for decoding, 2^14 for execution) and keep
only counters and hash sets, so memory stays below a few hundred MB per process regardless of range size; the
execution census keeps one fingerprint set per run (the 2^32 run will need the set written out per chunk instead,
see `02_real_isa_sweep.md`).

On the cluster, per-process limits without sudo: `ulimit -v 16000000` (KB, in the shell that starts the job) or
`prlimit --as=16000000000 ./program`. `systemd-run --user -p MemoryMax=16G ...` only works if the admin enabled
cgroup delegation for user sessions; this has not been tested on these nodes. Storage is the real memory limit:
Dimension42's `cluster6.py` kept a 20 % reserve per node and stopped writing to a node below it; the same guard
belongs in every big run here. knecht24 is the tightest (38 GB free).

## 4. Bounding power

**Workstation.** The active plan is *Balanced* with maximum processor state 100 % and minimum 5 %. To cap package
power coarsely, cap the processor state (an administrator shell):

```powershell
powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX 70   # 70 % max processor state
powercfg /setactive SCHEME_CURRENT
```

A real wattage cap (PL1/PL2) is a BIOS or Intel XTU setting, not a Windows one. With no working GPU the
workstation's compute power is the CPU's; the i9-13900K draws up to roughly 250 W at full load on all cores.

**Cluster.** Without sudo, `nvidia-smi -pl` (power limit) and `-lgc` (clocks) are not available, and the AMD sysfs
`power1_cap` files are root-only. The 3060s already have admin-set limits (170 / 110 / 110 W). Power is therefore
controlled by **how much work is submitted**: which GPUs a run uses, the chunk size per submission (Dimension42 moved
the AMD cards to short compute-queue submissions after the amdgpu watchdog silently dropped 92 million programs), and
pauses between chunks. Observed peaks in the 6-byte run: RX 9060 XT hotspot 97 °C (limit 110 °C), NVIDIA up to 74 °C,
no throttling; `nvidia-smi --query-gpu=power.draw,temperature.gpu --format=csv -l 5` and
`cat /sys/class/drm/card*/device/gpu_busy_percent` monitor it.

## 5. What fits where

| job | where | why |
|---|---|---|
| decode census of all 2^32 four-byte words (x86-64, AArch64) | workstation CPU | 15 to 20 minutes per ISA at 24 processes; pure Python + Capstone |
| execution census of all 2^32 words (Unicorn) | workstation CPU + cluster CPUs | about 65 k words/s per process; see `02_real_isa_sweep.md` for the measured total |
| anything NANO at 2^40 or 2^48 | cluster GPUs | the existing CUDA / Vulkan kernels do 17 G programs/s on the whole cluster |
| a GPU interpreter for a real ISA subset | cluster GPUs, later | would have to be written from scratch, as NANO's was |
| composition searches (`nano_compose.py`) | workstation | numpy, 65,536-entry tables, about 50 µs per composition |
