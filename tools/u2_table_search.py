"""Drive the Universe 2 table search on the cluster CPUs: launch shards, watch them, collect, verify, rank.

Every table (4-element subset of the 36-operation catalog, C(36,4) = 58,905) is scored on all programs of 1..L
symbols (L = 10: 1,398,100 programs) for W = 1 and W = 2, 64 steps, by tools/u2_tables.c (compiled on each node in
~/universe2). Shards are proportional to measured node speed; an OVERLAP range is computed on both adler40 (Intel)
and knecht24 (AMD) and must agree table by table (union counts and checksums).

  python tools/u2_table_search.py launch [node ...]
  python tools/u2_table_search.py status
  python tools/u2_table_search.py collect       # fetch, verify, merge, rank -> results/u2_tables/
"""
import collections
import json
import pathlib
import subprocess
import sys
import time
from math import comb

NOPS, NTABLES = 36, comb(36, 4)
L, STEPS = 10, 64
WIDTHS = (1, 2)
# node: (threads, measured tables/s at L=10 with 24 tables... relative weight)
NODES = {"adler40": (24, 24.0), "falke64": (12, 15.0), "knecht24": (24, 8.0), "specht32": (12, 8.0)}
OVERLAP = (20000, 200)  # computed on both adler40 and knecht24 in addition to their shards
OUT = pathlib.Path("results/u2_tables")
RAW = OUT / "raw"


def ssh(node, cmd, **kw):
    try:
        return subprocess.run(["ssh", "-n", "-o", "BatchMode=yes", node, cmd], capture_output=True, text=True, **kw)
    except subprocess.TimeoutExpired as e:
        return subprocess.CompletedProcess(e.cmd, -1, (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or ""), "timeout")


def plan():
    total_w = sum(w for _, w in NODES.values())
    shards = {n: [] for n in NODES}
    for W in WIDTHS:
        start = 0
        names = list(NODES)
        for i, n in enumerate(names):
            count = NTABLES - start if i == len(names) - 1 else round(NTABLES * NODES[n][1] / total_w)
            shards[n].append((W, start, count))
            start += count
        assert start == NTABLES
    for n in ("adler40", "knecht24"):
        for W in WIDTHS:
            shards[n].append((W, OVERLAP[0], OVERLAP[1]))
    return shards


def shard_name(W, start, count):
    return f"W{W}_L{L}_s{STEPS}_{start}_{count}"


def launch(only=None):
    shards = plan()
    for n, lst in shards.items():
        if only and n not in only:
            continue
        threads = NODES[n][0]
        cmds = []
        for W, start, count in lst:
            name = shard_name(W, start, count)
            cmds.append(f"nice -n 10 ./u2_tables {W} {L} {STEPS} {start} {count} {threads} > results/{name}.tsv 2> results/{name}.err; echo done >> results/{name}.err")
        chain = "; ".join(cmds)
        # setsid + all three streams detached, otherwise ssh waits for the background job
        r = ssh(n, f"cd ~/universe2 && mkdir -p results && (setsid nohup sh -c '{chain}' < /dev/null > /dev/null 2>&1 &); echo launched", timeout=30)
        print(n, r.stdout.strip() or r.stderr.strip(), [shard_name(*s) for s in lst])
    (OUT / "plan.json").parent.mkdir(parents=True, exist_ok=True)
    (OUT / "plan.json").write_text(json.dumps({"launched": time.strftime("%Y-%m-%dT%H:%M:%S"), "L": L, "steps": STEPS,
                                               "ntables": NTABLES, "shards": shards, "overlap": OVERLAP}, indent=1))


def status():
    all_done = True
    for n, lst in plan().items():
        r = ssh(n, "cd ~/universe2/results 2>/dev/null && for f in *.err; do echo \"$f: $(tr '\\n' ' ' < $f)\"; done; uptime | sed 's/.*load/load/'")
        print(f"=== {n} ===\n{r.stdout.strip()}")
        for s in lst:
            if f"{shard_name(*s)}.err: " in r.stdout and "done" not in r.stdout.split(f"{shard_name(*s)}.err: ")[1].split("\n")[0]:
                all_done = False
            if f"{shard_name(*s)}.err" not in r.stdout:
                all_done = False
    print("ALL DONE" if all_done else "running")
    return all_done


def collect():
    RAW.mkdir(parents=True, exist_ok=True)
    shards = plan()
    rows = {W: {} for W in WIDTHS}
    overlap = {W: {} for W in WIDTHS}
    log = {W: [] for W in WIDTHS}
    for n, lst in shards.items():
        for W, start, count in lst:
            name = shard_name(W, start, count)
            r = ssh(n, f"cat ~/universe2/results/{name}.tsv; echo ===ERR===; cat ~/universe2/results/{name}.err")
            tsv, err = r.stdout.split("===ERR===\n")
            (RAW / f"{n}_{name}.tsv").write_text(tsv)
            lines = [ln for ln in tsv.splitlines() if ln.strip()]
            assert len(lines) == count, f"{n} {name}: {len(lines)} lines, expected {count}"
            ids = [int(ln.split("\t")[0]) for ln in lines]
            assert ids == list(range(start, start + count)), f"{n} {name}: ids not contiguous"
            summary = [ln for ln in err.splitlines() if ln.startswith("tables=")]
            log[W].append({"node": n, "start": start, "count": count, "stderr": summary[0] if summary else err.strip()})
            is_overlap = (start, count) == OVERLAP and n in ("adler40", "knecht24")
            target = overlap[W].setdefault(n, {}) if is_overlap else rows[W]
            for ln in lines:
                target[int(ln.split("\t")[0])] = ln
    # verify overlaps: both nodes' lines must be identical, and identical to the shard that owns the range
    checks = {}
    for W in WIDTHS:
        a, k = overlap[W]["adler40"], overlap[W]["knecht24"]
        same = sum(a[i] == k[i] for i in a)
        owner_same = sum(rows[W][i] == a[i] for i in a)
        checks[W] = {"overlap_tables": len(a), "adler40_vs_knecht24_identical": same, "overlap_vs_owning_shard_identical": owner_same}
        assert same == len(a) == owner_same, checks[W]
        assert len(rows[W]) == NTABLES and set(rows[W]) == set(range(NTABLES)), f"W={W}: {len(rows[W])} tables"
    # merge, rank
    header = ["table_id", "op0", "op1", "op2", "op3"] + [f"union_L{l}" for l in range(1, L + 1)] + ["halted", "nonblank", "checksum"]
    report = {"L": L, "steps": STEPS, "programs_per_table": sum(4 ** l for l in range(1, L + 1)), "ntables": NTABLES,
              "widths": {}, "shards": log, "overlap_checks": checks}
    for W in WIDTHS:
        merged = OUT / f"tables_W{W}_L{L}_s{STEPS}.tsv"
        with merged.open("w") as f:
            f.write("\t".join(header) + "\n")
            for i in range(NTABLES):
                f.write(rows[W][i] + "\n")
        parsed = []
        for i in range(NTABLES):
            p = rows[W][i].split("\t")
            parsed.append({"id": i, "ops": p[1:5], "union": [int(x) for x in p[5:5 + L]], "halted": int(p[5 + L]), "nonblank": int(p[6 + L])})
        parsed.sort(key=lambda r: -r["union"][-1])
        top = parsed[:50]
        op_freq = collections.Counter(op for r in parsed[:500] for op in r["ops"])
        unions = [r["union"][-1] for r in parsed]
        report["widths"][W] = {
            "best_union": unions[0], "median_union": unions[NTABLES // 2], "worst_union": unions[-1],
            "tables_with_union_1": sum(u == 1 for u in unions),
            "top50": [{"id": r["id"], "ops": r["ops"], "union": r["union"][-1], "union_per_length": r["union"],
                       "halted_percent": round(100 * r["halted"] / report["programs_per_table"], 2),
                       "nonblank_percent": round(100 * r["nonblank"] / report["programs_per_table"], 2)} for r in top],
            "op_frequency_in_top500": op_freq.most_common(),
        }
        print(f"W={W}: best {unions[0]:,} distinct tapes of {report['programs_per_table']:,} programs ({100 * unions[0] / report['programs_per_table']:.2f} %), "
              f"table {top[0]['id']} {top[0]['ops']}; median {unions[NTABLES // 2]:,}")
        for r in top[:10]:
            print(f"   {r['id']:6d} {','.join(r['ops']):32s} {r['union'][-1]:>9,}  halted {100 * r['halted'] / report['programs_per_table']:5.1f} %")
    (OUT / "report.json").write_text(json.dumps(report, indent=1))
    print("written", OUT / "report.json")


if __name__ == "__main__":
    if sys.argv[1] == "launch":
        launch(sys.argv[2:] or None)
    else:
        {"status": status, "collect": collect}[sys.argv[1]]()
