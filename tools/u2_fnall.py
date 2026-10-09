"""Exhaustive unary function map of ONE 2-bit machine over all 4^L programs of exactly L symbols, on the cluster.

This is the direct comparison with Universe-1: its champion SWAP,ADD,NAND,SKZ reaches 1,829,051 distinct unary
4-bit functions from all 2^32 eight-instruction programs (256 steps); JC reaches 8,533,818. L = 16 symbols of
2 bits is exactly 2^32 programs. The machine's distinct truth tables are dumped by every node's shard
(`u2_tables_v2 fnall`), merged here exactly (sets of 64-bit tables), and described: count, bijections,
constants, named vocabulary, per-bit input dependence.

  python tools/u2_fnall.py launch TABLE_ID --W 1 --L 16
  python tools/u2_fnall.py status TABLE_ID --W 1 --L 16
  python tools/u2_fnall.py collect TABLE_ID --W 1 --L 16      -> results/u2_fnall/table<ID>_W<W>_L<L>.json
"""
import argparse
import collections
import json
import pathlib
import subprocess
import time

NODES = {"adler40": (24, 24.0), "falke64": (12, 15.0), "knecht24": (24, 8.0), "specht32": (12, 8.0)}
STEPS = 64
OUT = pathlib.Path("results/u2_fnall")
VOCAB = None


def ssh(node, cmd, **kw):
    try:
        return subprocess.run(["ssh", "-n", "-o", "BatchMode=yes", node, cmd], capture_output=True, text=True, **kw)
    except subprocess.TimeoutExpired as e:
        return subprocess.CompletedProcess(e.cmd, -1, "", "timeout")


def plan(L):
    total = 4 ** L
    tw = sum(w for _, w in NODES.values())
    shards, start = {}, 0
    names = list(NODES)
    for i, n in enumerate(names):
        count = total - start if i == len(names) - 1 else round(total * NODES[n][1] / tw)
        shards[n] = (start, count)
        start += count
    return shards


def tag(tid, W, L):
    return f"fnall_t{tid}_W{W}_L{L}_s{STEPS}"


def launch(tid, W, L):
    for n, (start, count) in plan(L).items():
        name = tag(tid, W, L)
        cmd = f"nice -n 10 ./u2_tables_v2 fnall {W} {L} {STEPS} {tid} {start} {count} {NODES[n][0]} > results/{name}.txt 2> results/{name}.err; echo done >> results/{name}.err"
        r = ssh(n, f"cd ~/universe2 && mkdir -p results && (setsid nohup sh -c '{cmd}' < /dev/null > /dev/null 2>&1 &); echo launched", timeout=30)
        print(n, r.stdout.strip() or r.stderr.strip(), f"programs {start:,} + {count:,}")


def status(tid, W, L):
    done = True
    for n in NODES:
        r = ssh(n, f"cat ~/universe2/results/{tag(tid, W, L)}.err 2>/dev/null | tr '\\n' ' '; uptime | sed 's/.*load/load/'")
        print(f"{n}: {r.stdout.strip()[:220]}")
        done &= "done" in r.stdout
    print("ALL DONE" if done else "running")
    return done


def nib(t, x):
    return (t >> (4 * x)) & 15


def vocab():
    global VOCAB
    if VOCAB is None:
        txt = ssh("adler40", "~/universe2/u2_tables_v2 --vocab").stdout
        VOCAB = {}
        for line in txt.strip().splitlines():
            parts = line.split()
            t = 0
            for x, v in enumerate(parts[1:]):
                t |= int(v) << (4 * x)
            VOCAB[t] = parts[0]
    return VOCAB


def collect(tid, W, L):
    OUT.mkdir(parents=True, exist_ok=True)
    tables = set()
    log = []
    for n, (start, count) in plan(L).items():
        name = tag(tid, W, L)
        r = ssh(n, f"cat ~/universe2/results/{name}.txt; echo ===ERR===; cat ~/universe2/results/{name}.err")
        txt, err = r.stdout.split("===ERR===\n")
        lines = txt.split()
        summary = [ln for ln in err.splitlines() if "mode=fnall" in ln]
        assert summary, f"{n}: no summary"
        kv = dict(kvp.split("=", 1) for kvp in summary[0].split() if "=" in kvp)
        assert int(kv["programs"]) == count and int(kv["start"]) == start and int(kv["lines"]) == len(lines), (n, kv, len(lines))
        node_tables = set(int(x, 16) for x in lines)
        log.append({"node": n, "start": start, "count": count, "distinct_in_shard": len(node_tables), "secs": float(kv["secs"]), "ops": kv["ops"]})
        tables |= node_tables
    V = vocab()
    bij = sum(1 for t in tables if len({nib(t, x) for x in range(16)}) == 16)
    consts = sum(1 for t in tables if len({nib(t, x) for x in range(16)}) == 1)
    named = sorted(V[t] for t in tables if t in V)
    image_hist = collections.Counter(len({nib(t, x) for x in range(16)}) for t in tables)
    # input-bit dependence: which of the 4 input bits ever change the output
    dep = collections.Counter()
    for t in tables:
        mask = 0
        for b in range(4):
            if any(nib(t, x) != nib(t, x ^ (1 << b)) for x in range(16)):
                mask |= 1 << b
        dep[mask] += 1
    res = {"table_id": tid, "ops": log[0]["ops"], "W": W, "L": L, "steps": STEPS, "programs": 4 ** L,
           "distinct_functions": len(tables), "bijections": bij, "constants": consts,
           "named_functions": named, "named_count": len(named), "vocabulary_size": len(V),
           "image_size_histogram": {str(k): v for k, v in sorted(image_hist.items())},
           "input_bits_that_matter": {f"{m:04b}": c for m, c in sorted(dep.items())},
           "depends_on_all_four_bits": dep[15], "shards": log,
           "universe1_reference": {"champion SWAP,ADD,NAND,SKZ (2^32 programs, 256 steps)": 1829051,
                                   "JC SWAP,ADD,NAND,JC": 8533818, "o3 SWAP,LDI,NAND,ADD,ROL,SKNZ,INC,HALT": 1463824,
                                   "full 16-op LD..HALT": 977928}}
    path = OUT / f"table{tid}_W{W}_L{L}.json"
    path.write_text(json.dumps(res, indent=1))
    (OUT / f"table{tid}_W{W}_L{L}.functions.txt").write_text("\n".join(f"{t:016x}" for t in sorted(tables)) + "\n")
    print(f"table {tid} {res['ops']} W={W} L={L}: {len(tables):,} distinct unary functions from {4 ** L:,} programs; "
          f"bijections {bij}, constants {consts}, named {len(named)}/{len(V)}: {', '.join(named)}")
    print("image-size histogram:", dict(sorted(image_hist.items())))
    print("input bits that matter (mask: count):", {f'{m:04b}': c for m, c in sorted(dep.items())})
    print("written", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["launch", "status", "collect"])
    ap.add_argument("table_id", type=int)
    ap.add_argument("--W", type=int, default=1)
    ap.add_argument("--L", type=int, default=16)
    a = ap.parse_args()
    {"launch": launch, "status": status, "collect": collect}[a.action](a.table_id, a.W, a.L)


if __name__ == "__main__":
    main()
