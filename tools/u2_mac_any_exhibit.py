"""Gather the frontier programs of docs/14 section 7: the GPU runs with a truncated input list (--maxin) name the
programs that survive the most triples; this script fetches those outputs from the nodes, recovers the per-step
cells and head flag of every program with the CPU mode (u2_tables_any primany ... count 1 maxin), re-verifies each
with the Python reference (eval_any with the same input limit) and with u2_ops.run at every claimed budget, writes
results/u2_mac_any/exhibit/frontier_programs.json and prints a Markdown table.

  python tools/u2_mac_any_exhibit.py
"""
import json
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import u2_ops  # noqa: E402
from u2_mac import MACHINE, W, inputs_of, ssh  # noqa: E402
import u2_mac_any_ref as ref  # noqa: E402
from u2_mac_any import CPU_BIN, STEPS, parse_output  # noqa: E402

RUNS = [("adler40", "exhibit_L16_MAC_B_in24", 16, 24), ("adler40", "exhibit_L16_MAC_T_in22", 16, 22),
        ("knecht24", "exhibit_L13_in18", 13, 18), ("knecht24", "exhibit_L14_MAC_T_in20", 14, 20), ("knecht24", "exhibit_L14_MAC_B_in19", 14, 19),
        ("knecht24", "exhibit_L15_MAC_T_in20", 15, 20), ("knecht24", "exhibit_L15_MAC_B_in21", 15, 21)]
OUT = pathlib.Path("results/u2_mac_any/exhibit")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    table = u2_ops.parse_table(MACHINE)
    rows = []
    for node, stem, L, maxin in RUNS:
        r = ssh(node, f"cat ~/universe2/results/{stem}.txt; echo ===ERR===; cat ~/universe2/results/{stem}.err", timeout=300)
        assert "===ERR===\n" in r.stdout and "done" in r.stdout.split("===ERR===\n")[1], (stem, r.stdout[-200:])
        txt, err = r.stdout.split("===ERR===\n")
        (OUT / f"{stem}.txt").write_text(txt)
        (OUT / f"{stem}.err").write_text(err)
        head, prs, kv, done = parse_output(txt, err)
        assert int(head["L"]) == L and int(head["programs"]) == 4 ** L
        for p, d in prs.items():
            assert d["inputs"] == maxin and d["found_count"] == len(d["found"]) and d["survivors"][-1] == d["found_count"]
            for prog, rg, _ in d["found"]:
                q = int(prog, 4)
                c = ssh("adler40", f"cd ~/universe2 && ./{CPU_BIN} primany {p} {W} {L} {STEPS} {MACHINE} {q} 1 1 {maxin} 2>/dev/null | grep '^F'", timeout=120)
                _, prs1, _, _ = parse_output(f"prim\t{p}\n" + c.stdout, "done")
                f1 = prs1[p]["found"][0]
                assert f1[0] == prog and f1[1] == rg, (prog, rg, c.stdout[:200])
                detail = f1[2]
                ns, ref_detail = ref.eval_any(table, [int(ch) for ch in prog], W, STEPS, p, maxin)
                ok = ns == maxin and ref_detail is not None and [[a, b, int(h)] for a, b, h in ref_detail] == detail
                if ok:   # and u2_ops.run at every claimed budget, on the kept inputs
                    for s, cells, h in detail:
                        for init, target in inputs_of(p)[:maxin]:
                            st = u2_ops.run(table, [int(ch) for ch in prog], W, s, init=list(init))
                            tape = st["tape_nonzero"]
                            if any(tape.get(cc, 0) != target for cc in cells) or (h and tape.get(st["head"], 0) != target):
                                ok = False
                # the next triple (the one it fails) and what the program leaves there at its first claimed budget
                nxt = inputs_of(p)[maxin] if maxin < len(inputs_of(p)) else None
                s0, cells0, h0 = detail[0]
                st = u2_ops.run(table, [int(ch) for ch in prog], W, s0, init=list(nxt[0])) if nxt else None
                got = (st["tape_nonzero"].get(cells0[0], 0) if cells0 else st["tape_nonzero"].get(st["head"], 0)) if st else None
                rows.append({"L": L, "prim": p, "program": prog, "triples_survived": maxin, "steps": rg, "per_step": detail,
                             "fails_on": {"input": maxin + 1, "triple": list(nxt[0]), "target": nxt[1], "leaves": got, "at_budget": s0} if nxt else None,
                             "reverified": ok, "gpu_device": kv["device"], "gpu_secs": float(kv["secs"])})
                cells_txt = ",".join(map(str, cells0)) if cells0 else "head"
                print(f"| {L} | {p} | `{prog}` | {maxin} | {rg} | cell {cells_txt} (budget {s0}) | ({nxt[0][0]}, {nxt[0][1]}, {nxt[0][2]}) -> {nxt[1]}, leaves {got} | {'yes' if ok else 'NO'} |")
    (OUT / "frontier_programs.json").write_text(json.dumps({"machine": MACHINE, "W": W, "steps": STEPS, "programs": rows}, indent=1))
    print("written", OUT / "frontier_programs.json", "all reverified:", all(r["reverified"] for r in rows))


if __name__ == "__main__":
    main()
