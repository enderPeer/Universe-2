"""Watch an open-ended DNA-3 world (docs/17 section 14): fetch its census series, top genes and the latest slot-map
frames from the cluster node, render pictures and a timeline, and list emergence events.

  python tools/dna3_watch.py --host knecht24 --dir universe2/dna3/open --name O1_seasons --W 640 --out results/dna3/open
        [--frames-back 0,5,20] [--modes plant,traits,clade] [--no-fetch]

Writes into OUT/NAME/: NAME.txt.gz and NAME.top.gz (copies, census lines deduplicated by tick, the last one kept, since
a resumed run repeats the ticks after its checkpoint), timeline.png (live, plants, organisms with SENSE and WAIT, the
two zones, births per census), <mode>_t<tick>.png for the latest frame and frames-back frames before it, events.json
and a printed summary. Emergence events, from the top-10 gene lists and the census:
- novelty: an operation sequence (synonymous codons collapsed) enters the top 10 for the first time;
- thresholds: the first census at which organisms carrying HARVEST, SENSE or WAIT exceed 1 %, 10 %, 50 % of the living;
- the longest operation sequence in the top 10 reaching a new maximum.
"""
import argparse
import gzip
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dna3_maps  # noqa: E402


def gzopen(path, mode, tries=10):
    """gzip.open that retries for a few seconds when Windows briefly locks the file (e.g. a virus scan: OSError 22/13)."""
    import time
    for k in range(tries):
        try:
            return gzip.open(path, mode)
        except OSError:
            if k == tries - 1:
                raise
            time.sleep(2)


def ssh(host, cmd):
    return subprocess.run(["ssh", "-n", "-o", "BatchMode=yes", host, cmd], capture_output=True, text=True).stdout


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="knecht24")
    ap.add_argument("--dir", default="universe2/dna3/open")
    ap.add_argument("--name", required=True)
    ap.add_argument("--W", type=int, required=True)
    ap.add_argument("--out", default="results/dna3/open")
    ap.add_argument("--modes", default="plant,traits,clade")
    ap.add_argument("--frames-back", default="0")
    ap.add_argument("--shrink", type=int, default=1)
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--census-every", type=int, default=0,
                    help="keep only census lines with (tick + 1) mod this = 0 (drops lines of aborted earlier starts)")
    a = ap.parse_args()
    od = os.path.join(a.out, a.name)
    os.makedirs(od, exist_ok=True)
    base = "%s/%s" % (a.dir, a.name)
    S = a.W * a.W
    if not a.no_fetch:
        txt = ssh(a.host, "cat %s.txt" % base)
        top = ssh(a.host, "cat %s.top" % base)
        with gzopen(os.path.join(od, a.name + ".raw.txt.gz"), "wt") as f:
            f.write(txt)
        with gzopen(os.path.join(od, a.name + ".top.gz"), "wt") as f:
            f.write(top)
        subprocess.run(["scp", "-q", "-o", "BatchMode=yes", "%s:%s.map.terrain" % (a.host, base),
                        os.path.join(od, "frames.map.terrain")])
        S = os.path.getsize(os.path.join(od, "frames.map.terrain"))
        backs = [int(x) for x in a.frames_back.split(",")]
        for m in a.modes.split(","):
            # the last frame and some before it, cut out on the node with dd (no numpy needed there)
            n = int(ssh(a.host, "stat -c %%s %s.map.%s" % (base, m)).strip() or 0) // S
            # a census written between the two fetches adds a frame without its line: use only frames with a line
            n = min(n, sum(1 for l in txt.splitlines() if l.startswith("census ")))
            idx = sorted(set(n - 1 - b for b in backs if n - 1 - b >= 0))
            cmd = "; ".join("dd if=%s.map.%s bs=%d skip=%d count=1 status=none" % (base, m, S, i) for i in idx)
            data = subprocess.run(["ssh", "-n", "-o", "BatchMode=yes", a.host, cmd], capture_output=True).stdout
            with open(os.path.join(od, "frames.map." + m), "wb") as f:
                f.write(data)
            with open(os.path.join(od, "frames.%s.idx" % m), "w") as f:
                json.dump(idx, f)
    with gzopen(os.path.join(od, a.name + ".raw.txt.gz"), "rt") as f:
        lines = [l for l in f if l.startswith("census ")]
    if a.census_every:
        lines = [l for l in lines if (json.loads(l[7:])["tick"] + 1) % a.census_every == 0]
    by_tick = {}
    order = []
    for l in lines:
        d = json.loads(l[7:])
        if d["tick"] not in by_tick:
            order.append(d["tick"])
        by_tick[d["tick"]] = d
    cens = [by_tick[t] for t in sorted(order)]
    with gzopen(os.path.join(od, a.name + ".txt.gz"), "wt") as f:
        for c in cens:
            f.write("census " + json.dumps(c) + "\n")
    # all census lines in file order map one to one onto the map frames
    frame_ticks = [json.loads(l[7:])["tick"] for l in lines]
    tops = {}
    with gzopen(os.path.join(od, a.name + ".top.gz"), "rt") as f:
        for l in f:
            if l.startswith("#"):
                continue
            r = l.rstrip("\n").split("\t")
            tops.setdefault(int(r[0]), []).append((int(r[2]), r[5], r[6]))
    # events
    events, seen, longest = [], set(), 0
    thresholds = {k: [0.01, 0.1, 0.5] for k in ("harvesters", "sensers", "sleepers")}
    for c in cens:
        t = c["tick"]
        for k, ths in thresholds.items():
            if k in c and c["live"]:
                while ths and c[k] > ths[0] * c["live"]:
                    events.append({"tick": t, "event": "%s above %d %%" % (k, round(100 * ths[0])), "count": c[k],
                                   "live": c["live"]})
                    ths.pop(0)
        for cnt, codons, ops in tops.get(t, []):
            if ops not in seen:
                seen.add(ops)
                events.append({"tick": t, "event": "new operation sequence in the top 10", "ops": ops, "codons": codons,
                               "count": cnt})
            nops = len(ops.split())
            if nops > longest:
                longest = nops
                events.append({"tick": t, "event": "longest top-10 gene so far", "length": nops, "ops": ops})
    with open(os.path.join(od, "events.json"), "w") as f:
        json.dump(events, f, indent=1)
    # timeline
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    T = np.array([c["tick"] for c in cens])
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for k, lab in (("live", "alive"), ("harvesters", "with HARVEST"), ("sensers", "with SENSE"), ("sleepers", "with WAIT"),
                   ("live_land", "on land")):
        if k in cens[-1]:
            ax[0].plot(T, [c.get(k, 0) for c in cens], label=lab, lw=1)
    if "zone_live" in cens[-1]:
        ax[0].plot(T, [c["zone_live"][0] for c in cens], label="zone 0", lw=0.6, ls="--")
        ax[0].plot(T, [c["zone_live"][1] for c in cens], label="zone 1", lw=0.6, ls="--")
    ax[0].legend(fontsize=8, ncol=4)
    ax[0].set_ylabel("organisms")
    ax[1].plot(T, [c["births"] for c in cens], label="births since last census", lw=1)
    ax[1].plot(T, [c.get("distinct_genes", 0) for c in cens], label="distinct genes", lw=1)
    ax[1].set_yscale("log")
    ax[1].legend(fontsize=8)
    ax[1].set_xlabel("tick")
    fig.suptitle("%s, %d censuses to tick %d" % (a.name, len(cens), T[-1]))
    fig.tight_layout()
    fig.savefig(os.path.join(od, "timeline.png"), dpi=110)
    # pictures
    ter = np.fromfile(os.path.join(od, "frames.map.terrain"), dtype=np.uint8)
    S = ter.size
    for m in a.modes.split(","):
        p = os.path.join(od, "frames.map." + m)
        if not os.path.exists(p):
            continue
        fr = np.fromfile(p, dtype=np.uint8).reshape(-1, S)
        idx = json.load(open(os.path.join(od, "frames.%s.idx" % m)))
        for k, i in enumerate(idx):
            tk = frame_ticks[i] if i < len(frame_ticks) else i
            dna3_maps.render(fr[k], a.W, m, ter, None, 1, a.shrink).save(os.path.join(od, "%s_t%d.png" % (m, tk)))
    last = cens[-1]
    print(json.dumps({k: last.get(k) for k in ("tick", "live", "harvesters", "sensers", "sleepers", "live_land", "zone_live",
                                                 "starved_total", "births_total", "distinct_genes", "max_depth")}))
    for e in events[-12:]:
        print(json.dumps(e))
    for cnt, codons, ops in tops.get(last["tick"], [])[:5]:
        print(cnt, codons, "|", ops)


if __name__ == "__main__":
    main()
