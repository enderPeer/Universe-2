"""Origin statistics for DNA-2 (docs/16): many unseeded worlds, standard against uniform code, on all free GPUs.

Every run: 99,999,744 cells, 1,024-cell slots, random soup, no seeds, mu 0.001, double strand, D and F at their
defaults (95 damage hits, 23 refilled slots per tick), up to 200,000 ticks, census every 100 ticks with the top 10
genes and their codons. Seed s = 1 of the standard code repeats the unseeded pilot of docs/15 (origin near tick
43,350), which checks the setup. Jobs are pulled from one queue by knecht24's three RTX 3060 and adler40's RTX 4090;
outputs land in ~/universe2/dna2/orig/ on the node and are fetched to results/dna2_origin/.

  python tools/dna2_origin_queue.py --seeds 48 --ticks 200000
"""
import argparse
import os
import queue
import subprocess
import threading
import time

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=48)
ap.add_argument("--ticks", type=int, default=200000)
ap.add_argument("--codes", default="standard,uniform")
a = ap.parse_args()

WORKERS = [("knecht24", 0), ("knecht24", 1), ("knecht24", 2), ("adler40", 0)]
OUT = "results/dna2_origin"
os.makedirs(OUT, exist_ok=True)
jobs = queue.Queue()
for s in range(1, a.seeds + 1):            # interleave the codes so both finish at about the same time
    for code in a.codes.split(","):
        jobs.put((code, s))
done, lock, t0 = [], threading.Lock(), time.time()


def ssh(node, cmd, timeout=None):
    try:
        return subprocess.run(["ssh", "-n", "-o", "BatchMode=yes", node, cmd], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess([], 124, "", "timeout")


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def worker(node, gpu):
    while True:
        try:
            code, s = jobs.get_nowait()
        except queue.Empty:
            return
        tag = f"{code}_s{s:03d}"
        cf = "--uniform-code" if code == "uniform" else ""
        env = "CUDA_VISIBLE_DEVICES=0 " if node == "adler40" else ""
        g = 0 if node == "adler40" else gpu
        cmd = (f"cd ~/universe2/dna2 && mkdir -p orig && {env}./dna_world2 --dna2 {cf} --N 99999744 --G 1024 --ticks {a.ticks} "
               f"--seed {s} --mu 0.001 --no-hash --census-every 100 --depth-cap 50 --top-genes orig/{tag}.top "
               f"--out orig/{tag}.json --gpu {g} > orig/{tag}.txt 2> orig/{tag}.err; "
               f"gzip -f orig/{tag}.txt orig/{tag}.top; tail -1 orig/{tag}.err")
        st = time.time()
        r = ssh(node, cmd, timeout=3600)
        if r.returncode != 0 or "summary" not in r.stdout:
            log(f"!! {node} gpu{gpu} {tag} failed: {r.stdout[-150:]} {r.stderr[-150:]}; requeued")
            jobs.put((code, s))
            time.sleep(60)
            continue
        for suffix in (".txt.gz", ".top.gz", ".json", ".err"):
            subprocess.run(["scp", "-q", "-o", "BatchMode=yes", f"{node}:universe2/dna2/orig/{tag}{suffix}", OUT], check=False)
        with lock:
            done.append(tag)
            log(f"{node} gpu{gpu} {tag} in {time.time() - st:.0f} s; {len(done)} done, elapsed {(time.time() - t0) / 60:.1f} min")


threads = [threading.Thread(target=worker, args=w) for w in WORKERS]
[t.start() for t in threads]
[t.join() for t in threads]
log(f"ORIGIN QUEUE FINISHED: {len(done)} runs in {(time.time() - t0) / 60:.1f} min")
