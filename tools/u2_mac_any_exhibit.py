"""Gather the frontier programs of docs/14 (sections 7 and 11): the programs that survive the most triples of the
any-step search. From the collected any_L13.json .. any_L16.json the script reads the frontier of every (length,
encoding), runs the GPU program with that many inputs kept (`--maxin N`, so the programs that survive N show up as
found, with their budgets), recovers the per-step cells and head flag of every program with the CPU mode
(u2_tables_any primany ... count 1 maxin), re-verifies each with the Python reference (eval_any with the same input
limit) and with u2_ops.run at every claimed budget, writes <results dir>/exhibit/frontier_programs.json and prints
a Markdown table.

  python tools/u2_mac_any_exhibit.py                       # the 256-cell, 256-budget results of section 7
  python tools/u2_mac_any_exhibit.py --ncell 512 --steps 512   # the variant of section 11
"""
import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import u2_ops  # noqa: E402
from u2_mac import MACHINE, W, inputs_of, ssh  # noqa: E402
import u2_mac_any_ref as ref  # noqa: E402
import u2_mac_any as any_  # noqa: E402

# the GPU each (length) runs on: 16 symbols on the two Ada cards, the rest on the 3060s
GPU_FOR = {(16, "MAC_B"): "adler40-gpu0", (16, "MAC_T"): "adler40-gpu1", (15, "MAC_T"): "knecht24-gpu1", (15, "MAC_B"): "knecht24-gpu2",
           (14, "MAC_T"): "knecht24-gpu0", (14, "MAC_B"): "knecht24-gpu0", (13, "MAC_T"): "knecht24-gpu0", (13, "MAC_B"): "knecht24-gpu0"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ncell", type=int, default=256)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--L", type=int, action="append")
    a = ap.parse_args()
    any_.set_variant(a.ncell, a.steps)
    OUT = any_.OUT / "exhibit"
    OUT.mkdir(parents=True, exist_ok=True)
    table = u2_ops.parse_table(MACHINE)
    # frontiers from the collected results
    runs = []
    for L in a.L or (13, 14, 15, 16):
        d = json.loads((any_.OUT / f"any_L{L}.json").read_text())
        for p in any_.PRIMS.split(","):
            s = d["primitives"][p]["survivors_after_input"]
            assert d["primitives"][p]["found"] == 0
            maxin = max(i for i, v in enumerate(s) if v) + 1
            runs.append((L, p, maxin, s[maxin - 1], GPU_FOR[(L, p)]))
    # launch, one chain per GPU
    stems = {}
    by_dev = {}
    for L, p, maxin, n, dev in runs:
        stem = f"exhibit{any_.variant()}_L{L}_{p}_in{maxin}"
        stems[(L, p)] = stem
        node, _, g = any_.DEVICES[dev]
        by_dev.setdefault(dev, []).append(f"nice -n 10 ./{any_.GPU_BIN} {p} {W} {L} {any_.STEPS} {MACHINE} 0 {4 ** L} --gpu {g} --maxin {maxin} > results/{stem}.txt 2> results/{stem}.err; echo done >> results/{stem}.err")
    for dev, cmds in by_dev.items():
        node = any_.DEVICES[dev][0]
        chain = "; ".join(cmds)
        r = ssh(node, f"cd ~/universe2 && (setsid nohup sh -c '{chain}' < /dev/null > /dev/null 2>&1 &); echo launched", timeout=30)
        assert "launched" in r.stdout, (dev, r.stdout, r.stderr)
    print("launched", len(runs), "frontier runs on", len(by_dev), "GPUs:", [(L, p, maxin, n) for L, p, maxin, n, _ in runs])
    pending = {(L, p): dev for L, p, _, _, dev in runs}
    while pending:
        time.sleep(20)
        for key, dev in list(pending.items()):
            r = ssh(any_.DEVICES[dev][0], f"tail -c 12 ~/universe2/results/{stems[key]}.err 2>/dev/null")
            if "done" in r.stdout:
                del pending[key]
    rows = []
    for L, p, maxin, n_alive, dev in runs:
        node = any_.DEVICES[dev][0]
        stem = stems[(L, p)]
        r = ssh(node, f"cat ~/universe2/results/{stem}.txt; echo ===ERR===; cat ~/universe2/results/{stem}.err", timeout=300)
        txt, err = r.stdout.split("===ERR===\n")
        (OUT / f"{stem}.txt").write_text(txt)
        (OUT / f"{stem}.err").write_text(err)
        head, prs, kv, done = any_.parse_output(txt, err)
        assert done and int(head["L"]) == L and int(head["programs"]) == 4 ** L
        d = prs[p]
        assert d["inputs"] == maxin and d["found_count"] == len(d["found"]) == n_alive and d["survivors"][-1] == n_alive, (L, p, d["found_count"], n_alive)
        for prog, rg, _ in d["found"]:
            q = int(prog, 4)
            c = ssh("adler40", f"cd ~/universe2 && ./{any_.CPU_BIN} primany {p} {W} {L} {any_.STEPS} {MACHINE} {q} 1 1 {maxin} 2>/dev/null | grep '^F'", timeout=120)
            _, prs1, _, _ = any_.parse_output(f"prim\t{p}\n" + c.stdout, "done")
            f1 = prs1[p]["found"][0]
            assert f1[0] == prog and f1[1] == rg, (prog, rg, c.stdout[:200])
            detail = f1[2]
            ns, ref_detail = ref.eval_any(table, [int(ch) for ch in prog], W, any_.STEPS, p, maxin, any_.NCELL)
            ok = ns == maxin and ref_detail is not None and [[x, y, int(h)] for x, y, h in ref_detail] == detail
            if ok:   # and u2_ops.run at every claimed budget, on the kept inputs
                for s, cells, h in detail:
                    for init, target in inputs_of(p)[:maxin]:
                        st = u2_ops.run(table, [int(ch) for ch in prog], W, s, init=list(init), ncell=any_.NCELL)
                        tape = st["tape_nonzero"]
                        if any(tape.get(cc, 0) != target for cc in cells) or (h and tape.get(st["head"], 0) != target):
                            ok = False
            nxt = inputs_of(p)[maxin] if maxin < len(inputs_of(p)) else None
            s0, cells0, h0 = detail[0]
            st = u2_ops.run(table, [int(ch) for ch in prog], W, s0, init=list(nxt[0]), ncell=any_.NCELL) if nxt else None
            got = (st["tape_nonzero"].get(cells0[0], 0) if cells0 else st["tape_nonzero"].get(st["head"], 0)) if st else None
            rows.append({"L": L, "prim": p, "program": prog, "triples_survived": maxin, "steps": rg, "per_step": detail,
                         "fails_on": {"input": maxin + 1, "triple": list(nxt[0]), "target": nxt[1], "leaves": got, "at_budget": s0} if nxt else None,
                         "reverified": ok, "gpu_device": kv["device"], "gpu_secs": float(kv["secs"])})
            cells_txt = ",".join(map(str, cells0)) if cells0 else "head"
            print(f"| {L} | {p} | `{prog}` | {maxin} | {rg} | cell {cells_txt} (budget {s0}) | ({nxt[0][0]}, {nxt[0][1]}, {nxt[0][2]}) -> {nxt[1]}, leaves {got} | {'yes' if ok else 'NO'} |")
    (OUT / "frontier_programs.json").write_text(json.dumps({"machine": MACHINE, "W": W, "steps": any_.STEPS, "ncell": any_.NCELL, "programs": rows}, indent=1))
    print("written", OUT / "frontier_programs.json", "programs:", len(rows), "all reverified:", all(r["reverified"] for r in rows))


if __name__ == "__main__":
    main()
