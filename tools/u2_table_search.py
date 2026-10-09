"""Drive the Universe 2 table search on the cluster CPUs: launch shards, watch them, collect, verify, rank.

Two scoring modes of tools/u2_tables.c (compiled on each node in ~/universe2):
  tapes  distinct final tapes from the blank tape over all programs of 1..L symbols (Universe-7's measure)
  fn     distinct unary 4-bit functions (input x in the first cells, 16 inputs, output read from the same cells)
         over all programs of 1..L symbols (Universe-1's measure), plus bijections, constants and a named vocabulary

Shards are proportional to measured node speed; an OVERLAP range is computed on both adler40 (Intel) and knecht24
(AMD) and must agree table by table. A table subset can be given for the fn mode (e.g. the top machines of a
cheaper screening) with --ids FILE (one table id per line).

  python tools/u2_table_search.py tapes launch [node ...]      python tools/u2_table_search.py tapes status
  python tools/u2_table_search.py fn launch --L 8              python tools/u2_table_search.py fn collect --L 8
  python tools/u2_table_search.py fn launch --L 10 --ids results/u2_tables/top_ids.txt
"""
import argparse
import collections
import json
import pathlib
import subprocess
import sys
import time
from math import comb

NOPS, NTABLES = 36, comb(36, 4)
STEPS = 64
WIDTHS = (1, 2)
# node: (threads, relative speed)
NODES = {"adler40": (24, 24.0), "falke64": (12, 15.0), "knecht24": (24, 8.0), "specht32": (12, 8.0)}
OVERLAP = (20000, 200)
OUT = pathlib.Path("results/u2_tables")


def ssh(node, cmd, **kw):
    try:
        return subprocess.run(["ssh", "-n", "-o", "BatchMode=yes", node, cmd], capture_output=True, text=True, **kw)
    except subprocess.TimeoutExpired as e:
        return subprocess.CompletedProcess(e.cmd, -1, "", "timeout")


class Job:
    def __init__(self, mode, L, ids=None):
        self.mode, self.L = mode, L
        self.ids = ids  # None = all tables (ranges); list = explicit subset (fn only)
        self.tag = f"{mode}_L{L}_s{STEPS}" + (f"_sub{len(ids)}" if ids else "")

    def plan(self):
        total_w = sum(w for _, w in NODES.values())
        shards = {n: [] for n in NODES}
        n_items = NTABLES if self.ids is None else len(self.ids)
        for W in WIDTHS:
            start = 0
            names = list(NODES)
            for i, n in enumerate(names):
                count = n_items - start if i == len(names) - 1 else round(n_items * NODES[n][1] / total_w)
                shards[n].append((W, start, count))
                start += count
        if self.ids is None:
            for n in ("adler40", "knecht24"):
                for W in WIDTHS:
                    shards[n].append((W, OVERLAP[0], OVERLAP[1]))
        return shards

    def name(self, W, start, count):
        return f"{self.tag}_W{W}_{start}_{count}"

    def launch(self, only=None):
        shards = self.plan()
        for n, lst in shards.items():
            if only and n not in only:
                continue
            threads = NODES[n][0]
            cmds = []
            for W, start, count in lst:
                name = self.name(W, start, count)
                if self.ids is None:
                    prefix = "fn " if self.mode == "fn" else ""
                    cmds.append(f"nice -n 10 ./u2_tables {prefix}{W} {self.L} {STEPS} {start} {count} {threads} > results/{name}.tsv 2> results/{name}.err; echo done >> results/{name}.err")
                else:
                    ids = " ".join(str(i) for i in self.ids[start:start + count])   # whitespace-separated, read by fscanf
                    cmds.append(f"echo {ids} > results/{name}.ids; nice -n 10 ./u2_tables_v3 fn {W} {self.L} {STEPS} @results/{name}.ids {count} {threads} > results/{name}.tsv 2> results/{name}.err; echo done >> results/{name}.err")
            chain = "; ".join(cmds)
            r = ssh(n, f"cd ~/universe2 && mkdir -p results && (setsid nohup sh -c '{chain}' < /dev/null > /dev/null 2>&1 &); echo launched", timeout=30)
            print(n, r.stdout.strip() or r.stderr.strip(), [self.name(*s) for s in lst])
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"plan_{self.tag}.json").write_text(json.dumps({"launched": time.strftime("%Y-%m-%dT%H:%M:%S"), "mode": self.mode, "L": self.L,
                                                               "steps": STEPS, "ntables": NTABLES, "ids": self.ids, "shards": shards,
                                                               "overlap": OVERLAP if self.ids is None else None}, indent=1))

    def status(self):
        all_done = True
        for n, lst in self.plan().items():
            r = ssh(n, f"cd ~/universe2/results 2>/dev/null && for f in {self.tag}_W*.err; do echo \"$f: $(tail -c 200 $f | tr '\\n' ' ')\"; done; uptime | sed 's/.*load/load/'")
            print(f"=== {n} ===\n{r.stdout.strip()}")
            for s in lst:
                nm = f"{self.name(*s)}.err: "
                if nm not in r.stdout or "done" not in r.stdout.split(nm)[1].split("\n")[0]:
                    all_done = False
        print("ALL DONE" if all_done else "running")
        return all_done

    def collect(self):
        raw = OUT / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        shards = self.plan()
        rows = {W: {} for W in WIDTHS}
        overlap = {W: {} for W in WIDTHS}
        log = {W: [] for W in WIDTHS}
        for n, lst in shards.items():
            for W, start, count in lst:
                name = self.name(W, start, count)
                r = ssh(n, f"cat ~/universe2/results/{name}.tsv; echo ===ERR===; cat ~/universe2/results/{name}.err")
                tsv, err = r.stdout.split("===ERR===\n")
                (raw / f"{n}_{name}.tsv").write_text(tsv)
                lines = [ln for ln in tsv.splitlines() if ln.strip()]
                assert len(lines) == count, f"{n} {name}: {len(lines)} lines, expected {count}"
                ids = [int(ln.split("\t")[0]) for ln in lines]
                expect = list(range(start, start + count)) if self.ids is None else self.ids[start:start + count]
                assert ids == expect, f"{n} {name}: ids differ from plan"
                summary = [ln for ln in err.splitlines() if "secs=" in ln]
                secs = sum(float(ln.split("secs=")[1].split()[0]) for ln in summary)
                log[W].append({"node": n, "start": start, "count": count, "secs": round(secs, 1), "runs": len(summary), "binary": "u2_tables_v3" if self.ids else "u2_tables"})
                is_overlap = self.ids is None and (start, count) == OVERLAP and n in ("adler40", "knecht24")
                target = overlap[W].setdefault(n, {}) if is_overlap else rows[W]
                for ln in lines:
                    target[int(ln.split("\t")[0])] = ln
        checks = {}
        for W in WIDTHS:
            if self.ids is None:
                a, k = overlap[W]["adler40"], overlap[W]["knecht24"]
                same = sum(a[i] == k[i] for i in a)
                owner_same = sum(rows[W][i] == a[i] for i in a)
                checks[W] = {"overlap_tables": len(a), "adler40_vs_knecht24_identical": same, "overlap_vs_owning_shard_identical": owner_same}
                assert same == len(a) == owner_same, checks[W]
                assert len(rows[W]) == NTABLES, f"W={W}: {len(rows[W])} tables"
            else:
                assert len(rows[W]) == len(self.ids)
        L = self.L
        per_table = sum(4 ** l for l in range(1, L + 1))
        report = {"mode": self.mode, "L": L, "steps": STEPS, "programs_per_table": per_table, "tables": len(rows[1]),
                  "widths": {}, "shards": log, "overlap_checks": checks}
        if self.mode == "tapes":
            header = ["table_id", "op0", "op1", "op2", "op3"] + [f"union_L{l}" for l in range(1, L + 1)] + ["halted", "nonblank", "checksum"]
        else:
            header = ["table_id", "op0", "op1", "op2", "op3"] + [f"functions_L{l}" for l in range(1, L + 1)] + ["bijections", "constants", "vocab_hits", "checksum", "vocab"]
        for W in WIDTHS:
            merged = OUT / f"{self.tag}_W{W}.tsv"
            with merged.open("w") as f:
                f.write("\t".join(header) + "\n")
                for i in sorted(rows[W]):
                    f.write(rows[W][i] + "\n")
            parsed = []
            for i in sorted(rows[W]):
                p = rows[W][i].split("\t")
                d = {"id": i, "ops": p[1:5], "score": [int(x) for x in p[5:5 + L]]}
                if self.mode == "tapes":
                    d.update(halted=int(p[5 + L]), nonblank=int(p[6 + L]))
                else:
                    voc = p[9 + L].rstrip(",") if len(p) > 9 + L else ""
                    d.update(bijections=int(p[5 + L]), constants=int(p[6 + L]), vocab_hits=int(p[7 + L]), vocab=voc.split(",") if voc else [])
                parsed.append(d)
            parsed.sort(key=lambda r: (-r["score"][-1], r["id"]))
            top = parsed[:50]
            scores = [r["score"][-1] for r in parsed]
            op_freq = collections.Counter(op for r in parsed[:500] for op in r["ops"])
            report["widths"][W] = {"best": scores[0], "median": scores[len(scores) // 2], "worst": scores[-1],
                                   "top50": top, "op_frequency_in_top500": op_freq.most_common()}
            if self.mode == "fn":
                vocab_union = collections.Counter(v for r in parsed for v in r["vocab"])
                report["widths"][W]["vocab_reached_by_any_machine"] = vocab_union.most_common()
                report["widths"][W]["best_by_vocab_hits"] = sorted(parsed, key=lambda r: (-r["vocab_hits"], -r["score"][-1]))[:20]
            unit = "distinct tapes" if self.mode == "tapes" else "distinct functions"
            print(f"W={W}: best {scores[0]:,} {unit} of {per_table:,} programs, table {top[0]['id']} {top[0]['ops']}; median {scores[len(scores) // 2]:,}")
            for r in top[:12]:
                extra = f"bij {r['bijections']} const {r['constants']} vocab {r['vocab_hits']}: {','.join(r['vocab'][:8])}" if self.mode == "fn" else f"halted {r['halted']}"
                print(f"   {r['id']:6d} {','.join(r['ops']):32s} {r['score'][-1]:>9,}  {extra}")
        (OUT / f"report_{self.tag}.json").write_text(json.dumps(report, indent=1))
        print("written", OUT / f"report_{self.tag}.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["tapes", "fn"])
    ap.add_argument("action", choices=["launch", "status", "collect"])
    ap.add_argument("nodes", nargs="*")
    ap.add_argument("--L", type=int, default=10)
    ap.add_argument("--ids", help="file with table ids (fn subset)")
    args = ap.parse_args()
    ids = [int(x) for x in pathlib.Path(args.ids).read_text().split()] if args.ids else None
    job = Job(args.mode, args.L, ids)
    if args.action == "launch":
        job.launch(args.nodes or None)
    elif args.action == "status":
        job.status()
    else:
        job.collect()


if __name__ == "__main__":
    main()
