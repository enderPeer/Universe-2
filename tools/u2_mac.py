"""Multiply-accumulate search for ONE 2-bit machine over every program of exactly L symbols, on the cluster CPUs.

Question (docs/12): does machine 50098 `INC_R1,INC_L1,FLIP_R2,JNZB` (W = 2, 64 steps) have a program that computes
acc + w * x mod 4 for every acc, x in 0..3 and every ternary weight w, with (acc, w, x) seeded in cells 0, 1, 2
and the result in one fixed cell or under the final head? Two weight encodings: MAC_T has w in {1, 3, 0} (3 = -1
mod 4), MAC_B has the BitNet code c in {2, 0, 1} meaning c - 1. The local numpy search (Ghost/bitnet_u2/
prim_search.py) covered 1..14 symbols; this driver covers 15 and 16 with the `prim` mode of tools/u2_tables.c,
built under the name u2_tables_mac on each node.

  python tools/u2_mac.py build                 # scp tools/u2_tables.c to every node, gcc -> u2_tables_mac, self-check
  python tools/u2_mac.py validate              # L = 1..12 on adler40 against Ghost/bitnet_u2 prim_search results
  python tools/u2_mac.py launch --L 15 --L 16  # shards by node speed, detached (plus Intel/AMD overlap ranges)
  python tools/u2_mac.py status --L 15 --L 16
  python tools/u2_mac.py collect --L 15 --L 16 # -> results/u2_mac/mac_L15.json, mac_L16.json (found programs re-verified)
"""
import argparse
import json
import pathlib
import socket
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import u2_ops  # noqa: E402

MACHINE = "INC_R1,INC_L1,FLIP_R2,JNZB"
TABLE_ID = 50098
W, STEPS = 2, 64
PRIMS = "MAC_T,MAC_B"
ALL_PRIMS = "MAC_T,MAC_B,ADD,MUL_T,COPY,CARRY"
BINARY = "u2_tables_mac"
# node: (threads, relative speed) as in tools/u2_table_search.py
NODES = {"adler40": (24, 24.0), "falke64": (12, 15.0), "knecht24": (24, 8.0), "specht32": (12, 8.0)}
OVERLAP_NODES = ("adler40", "knecht24")
OVERLAP = {15: (123456789, 1 << 26), 16: (3000000000, 1 << 26)}   # (start, count): same range on an Intel and an AMD node
OUT = pathlib.Path("results/u2_mac")
REF = pathlib.Path(r"C:\Users\end\Ghost\bitnet_u2\results\prim_INC_R1-INC_L1-FLIP_R2-JNZB_W2_L12_s64.json")


def ssh(node, cmd, timeout=120):
    try:
        return subprocess.run(["ssh", "-n", "-o", "BatchMode=yes", node, cmd], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        return subprocess.CompletedProcess(e.cmd, -1, "", "timeout")


def plan(L):
    """Shards proportional to node speed over the 4^L programs of exactly L symbols (same rule as u2_fnall.plan)."""
    total = 4 ** L
    tw = sum(w for _, w in NODES.values())
    shards, start = {}, 0
    names = list(NODES)
    for i, n in enumerate(names):
        count = total - start if i == len(names) - 1 else round(total * NODES[n][1] / tw)
        shards[n] = (start, count)
        start += count
    return shards


def name(L, start, count, prims=PRIMS):
    return f"mac_{prims.replace(',', '-')}_W{W}_L{L}_s{STEPS}_{start}_{count}"


def cmd_line(L, start, count, threads, prims=PRIMS):
    nm = name(L, start, count, prims)
    return (f"nice -n 10 ./{BINARY} prim {prims} {W} {L} {STEPS} {MACHINE} {start} {count} {threads} "
            f"> results/{nm}.txt 2> results/{nm}.err; echo done >> results/{nm}.err")


def parse_output(txt, err):
    """The prim mode's stdout -> dict; the stderr summary line -> dict."""
    head, prims, cur = {}, {}, None
    for line in txt.splitlines():
        p = line.split("\t")
        if p[0] == "F":
            cells = [] if p[3] == "-" else [int(c) for c in p[3].split(",")]
            prims[p[1]]["found"].append([p[2], cells, p[4] == "1"])
        elif p[0] == "prim":
            cur = p[1]
            prims[cur] = {"found": []}
        elif cur is None:
            head[p[0]] = p[1]
        elif p[0] == "survivors":
            prims[cur]["survivors"] = [int(x) for x in p[1:]]
        elif p[0] == "order":
            prims[cur]["order"] = p[1:]
        elif p[0] in ("inputs", "excluded_cell0", "found"):
            prims[cur][p[0] + ("_count" if p[0] == "found" else "")] = int(p[1])
    summary = [ln for ln in err.splitlines() if "mode=prim" in ln]
    kv = dict(kvp.split("=", 1) for kvp in summary[0].split() if "=" in kvp) if summary else {}
    return head, prims, kv, "done" in err


def inputs_of(prim):
    """(init cells, target) in the order the C mode and prim_search.py use: nonzero targets first."""
    mask, nv, neg = (1 << W) - 1, 1 << W, (1 << W) - 1
    vals = range(nv)
    if prim == "COPY":
        lst = [((a,), a) for a in vals if a] + [((0,), 0)]
    elif prim == "ADD":
        lst = [((a, b), (a + b) & mask) for a in vals for b in vals]
    elif prim == "MUL_T":
        lst = [((w, x), (w * x) & mask) for w in (1, neg, 0) for x in vals]
    elif prim == "MAC_T":
        lst = [((acc, w, x), (acc + w * x) & mask) for w in (1, neg, 0) for acc in vals for x in vals]
    elif prim == "MAC_B":
        lst = [((acc, c, x), (acc + (c - 1) * x) & mask) for c in (2, 0, 1) for acc in vals for x in vals]
    elif prim == "CARRY":
        lst = [((a, b), int(a + b >= nv)) for a in vals for b in vals]
    else:
        raise ValueError(prim)
    lst.sort(key=lambda t: t[1] == 0)
    return lst


def verify(prim, program, cells, head_flag, L=None):
    """Re-run a found program with the reference interpreter tools/u2_ops.py on every input of the primitive."""
    table = u2_ops.parse_table(MACHINE)
    prog = [int(c) for c in program]
    excl = {0} if prim == "COPY" else set()
    ok_cells = {c: True for c in cells}
    ok_head = bool(head_flag)
    for init, target in inputs_of(prim):
        st = u2_ops.run(table, prog, W, STEPS, init=list(init))
        tape = st["tape_nonzero"]
        for c in cells:
            if c in excl or tape.get(c, 0) != target:
                ok_cells[c] = False
        if head_flag and (st["head"] in excl or tape.get(st["head"], 0) != target):
            ok_head = False
    return bool(all(ok_cells.values()) and (ok_head or not head_flag) and (cells or head_flag)), ok_cells, ok_head


def build(nodes=None):
    src = pathlib.Path("tools/u2_tables.c")
    for n in nodes or NODES:
        r = subprocess.run(["scp", "-q", str(src), f"{n}:~/universe2/{BINARY}.c"], capture_output=True, text=True)
        if r.returncode:
            print(n, "scp failed:", r.stderr.strip()); continue
        r = ssh(n, f"cd ~/universe2 && gcc -O3 -march=native -fopenmp -std=c11 -Wall -Wextra {BINARY}.c -o {BINARY} 2>&1 | grep -c warning; "
                   f"ls -la {BINARY} | cut -d' ' -f5-; ./{BINARY} prim COPY {W} 5 {STEPS} {MACHINE} 0 1024 2 | grep -E '^(found|F)'", timeout=300)
        out = r.stdout.strip().replace("\n", " | ")
        ok = "found\t1\nF\tCOPY\t01302\t1\t0" in r.stdout
        print(f"{n}: {out}  {'SELF-CHECK OK' if ok else 'SELF-CHECK FAILED'}")


def validate(L=12, node="adler40", threads=24):
    """All six primitives, every program of 1..L symbols, on one node; compared with prim_search.py's L12 results."""
    OUT.mkdir(parents=True, exist_ok=True)
    ref = json.loads(REF.read_text())
    assert ref["machine"] == MACHINE and ref["W"] == W and ref["steps"] == STEPS and ref["L"] == L
    per_prim = {}
    log = []
    for l in range(1, L + 1):
        nm = f"mac_val_L{l}_s{STEPS}"
        r = ssh(node, f"cd ~/universe2 && grep -q done results/{nm}.err 2>/dev/null || (nice -n 10 ./{BINARY} prim {ALL_PRIMS} {W} {l} {STEPS} {MACHINE} 0 {4 ** l} {threads} "
                      f"> results/{nm}.txt 2> results/{nm}.err; echo done >> results/{nm}.err); cat results/{nm}.txt; echo ===ERR===; cat results/{nm}.err", timeout=900)
        txt, err = r.stdout.split("===ERR===\n")
        head, prims, kv, done = parse_output(txt, err)
        assert done and int(head["programs"]) == 4 ** l and int(head["L"]) == l, (l, head, err)
        log.append({"L": l, "programs": 4 ** l, "secs": float(kv["secs"]), "threads": int(kv["threads"]), "host": kv["host"]})
        for p, d in prims.items():
            acc = per_prim.setdefault(p, {"survivors": [0] * d["inputs"], "found": [], "programs": 0})
            acc["programs"] += 4 ** l
            for i, s in enumerate(d["survivors"]):
                acc["survivors"][i] += s
            acc["found"] += d["found"]
    report = {"machine": MACHINE, "table_id": TABLE_ID, "W": W, "steps": STEPS, "L": L, "node": node, "binary": BINARY,
              "reference": str(REF), "reference_host": ref["host"], "reference_secs": ref["secs"], "runs": log, "primitives": {}}
    all_ok = True
    for p, acc in per_prim.items():
        rp = ref["primitives"][p]
        same_surv = acc["survivors"] == rp["survivors_after_input"]
        mine = sorted((f[0], tuple(f[1]), bool(f[2])) for f in acc["found"])
        theirs = sorted((f[0], tuple(f[1]), bool(f[2])) for f in rp["found"])
        same_found = mine == theirs
        ver = [verify(p, f[0], f[1], f[2])[0] for f in acc["found"]]
        ok = same_surv and same_found and all(ver) and acc["programs"] == rp["programs"]
        all_ok &= ok
        report["primitives"][p] = {"programs": acc["programs"], "survivors_after_input": acc["survivors"],
                                   "reference_survivors": rp["survivors_after_input"], "survivors_identical": same_surv,
                                   "found": len(acc["found"]), "reference_found": len(rp["found"]), "found_identical": same_found,
                                   "found_reverified_with_u2_ops": sum(ver), "ok": ok,
                                   "found_programs": [f for f in acc["found"]] if len(acc["found"]) <= 64 else acc["found"][:64]}
        print(f"{p:6s} programs {acc['programs']:,} survivors {'identical' if same_surv else 'DIFFER'} found {len(acc['found'])} "
              f"(reference {len(rp['found'])}) {'identical' if same_found else 'DIFFER'}, re-verified {sum(ver)}/{len(ver)}")
    report["all_identical"] = all_ok
    path = OUT / f"validation_L{L}_s{STEPS}.json"
    path.write_text(json.dumps(report, indent=1))
    print("ALL IDENTICAL" if all_ok else "MISMATCH", "written", path)
    return all_ok


def shard_list(Ls):
    """(node, L, start, count, is_overlap) for every run of the job."""
    runs = []
    for n in NODES:
        for L in Ls:
            start, count = plan(L)[n]
            runs.append((n, L, start, count, False))
        if n in OVERLAP_NODES:
            for L in Ls:
                if L in OVERLAP:
                    runs.append((n, L, OVERLAP[L][0], OVERLAP[L][1], True))
    return runs


def launch(Ls, only=None):
    OUT.mkdir(parents=True, exist_ok=True)
    runs = shard_list(Ls)
    for n in NODES:
        if only and n not in only:
            continue
        threads = NODES[n][0]
        chain = "; ".join(cmd_line(L, s, c, threads) for (nn, L, s, c, _) in runs if nn == n)
        r = ssh(n, f"cd ~/universe2 && mkdir -p results && (setsid nohup sh -c '{chain}' < /dev/null > /dev/null 2>&1 &); echo launched", timeout=30)
        print(n, r.stdout.strip() or r.stderr.strip(), [(L, f"{s:,} + {c:,}", "overlap" if o else "") for (nn, L, s, c, o) in runs if nn == n])
    tag = "_".join(f"L{L}" for L in Ls)
    (OUT / f"plan_mac_{tag}.json").write_text(json.dumps({
        "launched": time.strftime("%Y-%m-%dT%H:%M:%S"), "machine": MACHINE, "table_id": TABLE_ID, "W": W, "steps": STEPS,
        "prims": PRIMS, "binary": BINARY, "lengths": Ls, "programs": {L: 4 ** L for L in Ls}, "nodes": NODES,
        "runs": [{"node": n, "L": L, "start": s, "count": c, "overlap": o} for (n, L, s, c, o) in runs]}, indent=1))


def status(Ls):
    done = True
    for n in NODES:
        names = [name(L, s, c) for (nn, L, s, c, _) in shard_list(Ls) if nn == n]
        r = ssh(n, "cd ~/universe2/results 2>/dev/null; " + "; ".join(f"echo \"{nm}: $(tail -c 160 {nm}.err 2>/dev/null | tr '\\n' ' ')\"" for nm in names)
                + "; uptime | sed 's/.*load/load/'")
        print(f"=== {n} ===\n{r.stdout.strip()}")
        for nm in names:
            line = [ln for ln in r.stdout.splitlines() if ln.startswith(nm + ":")]
            if not line or "done" not in line[0]:
                done = False
    print("ALL DONE" if done else "running")
    return done


def collect(Ls):
    raw = OUT / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    runs = shard_list(Ls)
    fetched = {}
    for (n, L, s, c, o) in runs:
        nm = name(L, s, c)
        r = ssh(n, f"cat ~/universe2/results/{nm}.txt; echo ===ERR===; cat ~/universe2/results/{nm}.err", timeout=300)
        assert "===ERR===\n" in r.stdout, (n, nm, r.stderr)
        txt, err = r.stdout.split("===ERR===\n")
        (raw / f"{n}_{nm}.txt").write_text(txt)
        (raw / f"{n}_{nm}.err").write_text(err)
        head, prims, kv, done = parse_output(txt, err)
        assert done, f"{n} {nm}: not done"
        assert int(head["programs"]) == c and int(head["start"]) == s and int(head["L"]) == L and head["machine"] == MACHINE, (n, nm, head)
        assert int(kv["programs"]) == c and int(kv["start"]) == s and kv["host"] == n, (n, nm, kv)
        for p in PRIMS.split(","):
            assert prims[p]["found_count"] == len(prims[p]["found"]), (n, nm, p)
        fetched[(n, L, s, c)] = (prims, kv)
    for L in Ls:
        shards, overlap = [], {}
        merged = {p: {"survivors": None, "found": []} for p in PRIMS.split(",")}
        total = 0
        for (n, LL, s, c, o) in runs:
            if LL != L:
                continue
            prims, kv = fetched[(n, L, s, c)]
            rec = {"node": n, "start": s, "count": c, "secs": float(kv["secs"]), "threads": int(kv["threads"]), "host": kv["host"],
                   "survivors": {p: prims[p]["survivors"] for p in merged}, "found": {p: prims[p]["found_count"] for p in merged}}
            if o:
                overlap[n] = rec
                continue
            shards.append(rec)
            total += c
            for p, m in merged.items():
                m["survivors"] = prims[p]["survivors"] if m["survivors"] is None else [a + b for a, b in zip(m["survivors"], prims[p]["survivors"])]
                m["found"] += prims[p]["found"]
        assert total == 4 ** L, (L, total)
        # Intel/AMD overlap: the same program range must give identical survivor counts and found lists
        ov_ok = None
        if len(overlap) == 2:
            a, k = (overlap[n] for n in OVERLAP_NODES)
            pa, _ = fetched[(OVERLAP_NODES[0], L, a["start"], a["count"])]
            pk, _ = fetched[(OVERLAP_NODES[1], L, k["start"], k["count"])]
            ov_ok = all(pa[p]["survivors"] == pk[p]["survivors"] and sorted(map(str, pa[p]["found"])) == sorted(map(str, pk[p]["found"])) for p in merged)
            assert ov_ok, ("overlap differs", L, a, k)
        verified = {}
        for p, m in merged.items():
            ver = []
            for prog, cells, hflag in m["found"]:
                ok, ok_cells, ok_head = verify(p, prog, cells, hflag)
                ver.append({"program": prog, "cells": cells, "head": hflag, "u2_ops_verified": ok})
            verified[p] = ver
        res = {"machine": MACHINE, "table_id": TABLE_ID, "W": W, "L": L, "steps": STEPS, "programs": 4 ** L,
               "input_cells": "cells 0, 1, 2 = acc, w (or code), x; output in one fixed cell or under the final head",
               "primitives": {p: {"description": {"MAC_T": "acc + w * x mod 4, w in {1, 3, 0} (3 = -1)",
                                                  "MAC_B": "acc + (c - 1) * x mod 4, BitNet code c in {2, 0, 1}"}[p],
                                  "inputs": len(m["survivors"]), "order": inputs_of(p),
                                  "survivors_after_input": m["survivors"], "found": len(m["found"]),
                                  "found_programs": verified[p],
                                  "all_found_verified_with_u2_ops": all(v["u2_ops_verified"] for v in verified[p])}
                              for p, m in merged.items()},
               "shards": shards, "overlap": {"range": OVERLAP.get(L), "nodes": list(overlap), "identical": ov_ok, "runs": overlap},
               "collected": time.strftime("%Y-%m-%dT%H:%M:%S"), "collected_on": socket.gethostname()}
        path = OUT / f"mac_L{L}.json"
        path.write_text(json.dumps(res, indent=1))
        for p, m in merged.items():
            print(f"L={L} {p}: {4 ** L:,} programs, survivors {m['survivors'][:8]}..., found {len(m['found'])}"
                  + (f" {[v['program'] for v in verified[p][:5]]}" if m["found"] else ""))
        print(f"   shards: " + ", ".join(f"{s['node']} {s['count']:,} in {s['secs']:.0f} s" for s in shards)
              + (f"; overlap {OVERLAP[L][1]:,} programs on {'/'.join(overlap)}: {'identical' if ov_ok else 'DIFFER'}" if ov_ok is not None else ""))
        print("written", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["build", "validate", "launch", "status", "collect"])
    ap.add_argument("nodes", nargs="*")
    ap.add_argument("--L", type=int, action="append", help="program length(s), exactly L symbols")
    a = ap.parse_args()
    Ls = a.L or [15, 16]
    if a.action == "build":
        build(a.nodes or None)
    elif a.action == "validate":
        sys.exit(0 if validate() else 1)
    elif a.action == "launch":
        launch(Ls, a.nodes or None)
    elif a.action == "status":
        status(Ls)
    else:
        collect(Ls)


if __name__ == "__main__":
    main()
