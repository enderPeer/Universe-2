"""Dynamic work queue for the exhaustive 7-codon DNA-2 copier sweep (docs/15): 64 ranges of 2^30 strings, pulled by
whichever GPU is free. knecht24's three RTX 3060 start when their pilot worlds are done (pilots/gpuN.done), adler40's
RTX 4090 when the big world is done (big.done). A failed range goes back into the queue. Afterwards the 4090 runs the
uniform-code random sample and the parts are merged with tools/dna_sweep_merge.py on adler40.

  python tools/dna2_l7_queue.py            (runs on the workstation; ~2.5 h; log on stdout)
"""
import queue
import subprocess
import sys
import threading
import time

L, TICKS, NR = 7, 64, 64
RANGE = 1 << 30                      # 64 ranges x 2^30 = 2^36 strings
CHUNK = 1 << 24
WORKERS = [("knecht24", 0, "pilots/gpu0.done"), ("knecht24", 1, "pilots/gpu1.done"),
           ("knecht24", 2, "pilots/gpu2.done"), ("adler40", 0, "big.done")]
todo = queue.Queue()
for i in range(NR):
    todo.put(i)
done, lock, t0 = {}, threading.Lock(), time.time()


def ssh(node, cmd, timeout=None):
    try:
        return subprocess.run(["ssh", "-n", "-o", "BatchMode=yes", node, cmd], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess([], 124, "", "timeout")


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def worker(node, gpu, ready):
    while ssh(node, f"test -f ~/universe2/dna2/{ready} && echo y", timeout=60).stdout.strip() != "y":
        time.sleep(30)
    log(f"{node} gpu{gpu} starts")
    while True:
        try:
            i = todo.get_nowait()
        except queue.Empty:
            return
        a, b = i * RANGE, (i + 1) * RANGE
        tag = f"L7_standard_t{TICKS}_r{i:02d}"
        env = "CUDA_VISIBLE_DEVICES=0 " if node == "adler40" else ""
        cmd = (f"cd ~/universe2/dna2 && mkdir -p res7 && {env}./dna_sweep2 --L {L} --all --dna2 --ticks {TICKS} --tpb 32 "
               f"--chunk {CHUNK} --chunk-start {a} --chunk-end {b} --gpu {0 if node == 'adler40' else gpu} --log-mask 0 "
               f"--out res7/{tag}.json --chunk-log res7/{tag}_chunks.tsv > res7/{tag}.out 2>&1; tail -1 res7/{tag}.out")
        s = time.time()
        r = ssh(node, cmd, timeout=7200)
        line = r.stdout.strip()
        if r.returncode != 0 or "summary worlds=" not in line:
            log(f"!! {node} gpu{gpu} range {i} failed ({r.returncode}): {line[:120]} {r.stderr[:120]}; requeued")
            todo.put(i)
            time.sleep(60)
            continue
        with lock:
            done[i] = (node, gpu, time.time() - s)
            log(f"{node} gpu{gpu} range {i} in {time.time() - s:.0f} s; {len(done)}/{NR} done, elapsed {(time.time() - t0) / 60:.0f} min")


threads = [threading.Thread(target=worker, args=w) for w in WORKERS]
[t.start() for t in threads]
[t.join() for t in threads]
if len(done) != NR:
    sys.exit(f"only {len(done)} of {NR} ranges done")
log("all 64 ranges done; fetching the parts and merging on the workstation")
import os
outdir = "results/dna2_sweep/L7_parts"
os.makedirs(outdir, exist_ok=True)
for i, (node, gpu, secs) in sorted(done.items()):
    tag = f"L7_standard_t{TICKS}_r{i:02d}"
    for suffix in (".json", "_chunks.tsv", ".out"):
        subprocess.run(["scp", "-q", "-o", "BatchMode=yes", f"{node}:universe2/dna2/res7/{tag}{suffix}", outdir], check=False)
parts = [f"{outdir}/L7_standard_t{TICKS}_r{i:02d}.json" for i in range(NR)]
r = subprocess.run([sys.executable, "tools/dna_sweep_merge.py", f"results/dna2_sweep/L7_standard_t{TICKS}.json"] + parts,
                   capture_output=True, text=True)
log("merge: " + (r.stdout + r.stderr).strip()[-300:])
with open(f"{outdir}/ranges.tsv", "w") as f:
    f.write("#range\tnode\tgpu\tseconds\n")
    for i, (node, gpu, secs) in sorted(done.items()):
        f.write(f"{i}\t{node}\t{gpu}\t{secs:.1f}\n")
log("uniform-code random sample, 2^28 strings, on the 4090")
r = ssh("adler40", "cd ~/universe2/dna2 && CUDA_VISIBLE_DEVICES=0 ./dna_sweep2 --L 7 --random 268435456 --rseed 7 --dna2 --uniform-code "
                   "--ticks 64 --tpb 32 --chunk 16777216 --gpu 0 --log-mask 0x03 --out res7/L7_uniform_t64_rand.json "
                   "--log res7/L7_uniform_t64_rand_log.tsv --chunk-log res7/L7_uniform_t64_rand_chunks.tsv > res7/L7_uniform_t64_rand.out 2>&1; "
                   "tail -1 res7/L7_uniform_t64_rand.out; date +%s > l7.done")
log("uniform: " + r.stdout.strip()[:300])
log(f"L7 QUEUE FINISHED after {(time.time() - t0) / 60:.0f} min")
