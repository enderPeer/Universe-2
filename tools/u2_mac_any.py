"""Any-step multiply-accumulate search for machine 50098 on the cluster's CPUs and GPUs (docs/14).

docs/13 asked whether a program of machine 50098 `INC_R1,INC_L1,FLIP_R2,JNZB` (W = 2) computes acc + w * x mod 4
(inputs in cells 0, 1, 2, result in one fixed cell or under the final head) after exactly 64 or exactly 256 steps.
This driver asks it for EVERY budget 1..256 at once: the `primany` mode of tools/u2_tables.c (built as
u2_tables_any on the four nodes) and its GPU twin gpu/u2_mac_any.cu (u2_mac_any_gpu on adler40's RTX 4090 and
RTX 4080 and knecht24's three RTX 3060s) keep one candidate mask per step. Both are checked against the Python
reference tools/u2_mac_any_ref.py on small lengths, against each other on every length, and Intel/AMD and CPU/GPU
overlap ranges are run twice.

  python tools/u2_mac_any.py build                       # scp + gcc / nvcc on every node, self-check at 5 symbols
  python tools/u2_mac_any.py ref                         # the Python reference for 1..12 symbols (results/u2_mac_any/ref_L*.json)
  python tools/u2_mac_any.py validate                    # CPU (adler40) and GPUs (adler40 gpu0, knecht24 gpu0) vs the reference, 1..12
  python tools/u2_mac_any.py calibrate                   # 13 symbols on every device: speeds for the shards, outputs compared
  python tools/u2_mac_any.py launch --L 13 --L 14 --L 15 --L 16
  python tools/u2_mac_any.py status --L 13 --L 14 --L 15 --L 16
  python tools/u2_mac_any.py collect --L 13 --L 14 --L 15 --L 16   # -> results/u2_mac_any/any_L13.json .. any_L16.json
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
from u2_mac import MACHINE, TABLE_ID, W, inputs_of, ssh  # noqa: E402
import u2_mac_any_ref as ref  # noqa: E402

STEPS = 256
PRIMS = "MAC_T,MAC_B"
CPU_BIN, GPU_BIN = "u2_tables_any", "u2_mac_any_gpu"
NODES = {"adler40": {"threads": 24, "gpus": {0: "RTX 4090", 1: "RTX 4080"}, "arch": "sm_89"},
         "falke64": {"threads": 12, "gpus": {}},
         "knecht24": {"threads": 24, "gpus": {0: "RTX 3060", 1: "RTX 3060", 2: "RTX 3060"}, "arch": "sm_86"},
         "specht32": {"threads": 12, "gpus": {}}}
# device id -> (node, kind, threads or gpu index); the order is the shard order
DEVICES = {}
for _n, _d in NODES.items():
    DEVICES[f"{_n}-cpu"] = (_n, "cpu", _d["threads"])
    for _g in _d["gpus"]:
        DEVICES[f"{_n}-gpu{_g}"] = (_n, "gpu", _g)
OVERLAP_DEVICES = ("adler40-cpu", "knecht24-cpu", "adler40-gpu0", "knecht24-gpu0")   # Intel, AMD, Ada GPU, Ampere GPU
OVERLAP = {13: (10000000, 1 << 24), 14: (100000000, 1 << 26), 15: (123456789, 1 << 26), 16: (3000000000, 1 << 26)}
OUT = pathlib.Path("results/u2_mac_any")
CAL_L = 13                                  # the calibration length whose speeds shard the run (--calL)
CALIBRATION = OUT / f"calibration_L{CAL_L}.json"


def set_cal(L):
    global CAL_L, CALIBRATION
    CAL_L = L
    CALIBRATION = OUT / f"calibration_L{L}.json"
VALIDATE_DEVICES = ("adler40-cpu", "adler40-gpu0", "knecht24-gpu0")


def name(L, start, count, prims=PRIMS):
    return f"any_{prims.replace(',', '-')}_W{W}_L{L}_s{STEPS}_{start}_{count}"


def out_stem(dev, L, start, count, prims=PRIMS):
    return f"{dev}_{name(L, start, count, prims)}"


def cmd_line(dev, L, start, count, prims=PRIMS):
    node, kind, x = DEVICES[dev]
    stem = out_stem(dev, L, start, count, prims)
    if kind == "cpu":
        prog = f"./{CPU_BIN} primany {prims} {W} {L} {STEPS} {MACHINE} {start} {count} {x}"
    else:
        prog = f"./{GPU_BIN} {prims} {W} {L} {STEPS} {MACHINE} {start} {count} --gpu {x}"
    return f"nice -n 10 {prog} > results/{stem}.txt 2> results/{stem}.err; echo done >> results/{stem}.err"


def parse_output(txt, err):
    """stdout of primany (CPU or GPU) -> (header, prims, summary kv, done). F lines -> (program, ranges, detail or None)."""
    head, prims, cur = {}, {}, None
    for line in txt.splitlines():
        p = line.split("\t")
        if p[0] == "F":
            detail = None
            if len(p) > 4:
                detail = []
                for ent in p[4].split(";"):
                    s, cells, h = ent.split(":")
                    detail.append([int(s), [] if cells == "-" else [int(c) for c in cells.split(",")], int(h)])
            prims[p[1]]["found"].append((p[2], p[3], detail))
        elif p[0] == "prim":
            cur = p[1]
            prims[cur] = {"found": []}
        elif p[0] == "found_overflow":
            head["found_overflow"] = int(p[1])
        elif cur is None:
            head[p[0]] = p[1]
        elif p[0] in ("survivors", "found_by_step"):
            prims[cur][p[0]] = [int(x) for x in p[1:]]
        elif p[0] == "order":
            prims[cur]["order"] = p[1:]
        elif p[0] in ("inputs", "excluded_cell0", "found"):
            prims[cur][p[0] + ("_count" if p[0] == "found" else "")] = int(p[1])
    summary = [ln for ln in err.splitlines() if "mode=primany" in ln]
    kv = dict(kvp.split("=", 1) for kvp in summary[0].split() if "=" in kvp) if summary else {}
    return head, prims, kv, "done" in err


def fetch(dev, L, start, count, prims=PRIMS, save_raw=True):
    node = DEVICES[dev][0]
    stem = out_stem(dev, L, start, count, prims)
    r = ssh(node, f"cat ~/universe2/results/{stem}.txt; echo ===ERR===; cat ~/universe2/results/{stem}.err", timeout=600)
    assert "===ERR===\n" in r.stdout, (dev, stem, r.stderr)
    txt, err = r.stdout.split("===ERR===\n")
    if save_raw:
        raw = OUT / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        (raw / f"{stem}.txt").write_text(txt)
        (raw / f"{stem}.err").write_text(err)
    head, prs, kv, done = parse_output(txt, err)
    assert done, f"{dev} {stem}: not done"
    assert "found_overflow" not in head, (dev, stem, "GPU found buffer overflow")
    assert int(head["programs"]) == count and int(head["start"]) == start and int(head["L"]) == L and head["machine"] == MACHINE and int(head["steps"]) == STEPS, (dev, head)
    assert int(kv["programs"]) == count and int(kv["start"]) == start and kv["host"] == node, (dev, kv)
    assert kv["device"].startswith("gpu" if DEVICES[dev][1] == "gpu" else "cpu"), (dev, kv["device"])
    for p in prims.split(","):
        assert prs[p]["found_count"] == len(prs[p]["found"]), (dev, stem, p)
        assert len(prs[p]["found_by_step"]) == STEPS and len(prs[p]["survivors"]) == len(inputs_of(p)), (dev, stem, p)
    return head, prs, kv


def found_set(prs, p):
    return sorted((f[0], f[1]) for f in prs[p]["found"])


def same_results(a, b, prims):
    """survivors, found (program, ranges) and found_by_step identical for every primitive."""
    return all(a[p]["survivors"] == b[p]["survivors"] and found_set(a, p) == found_set(b, p) and a[p]["found_by_step"] == b[p]["found_by_step"] for p in prims.split(","))


def run_jobs(jobs, poll=15, timeout=36000):
    """jobs: list of (dev, L, start, count, prims); one detached chain per device; waits for all; returns {job: (head, prims, kv)}."""
    by_dev = {}
    for j in jobs:
        by_dev.setdefault(j[0], []).append(j)
    for dev, js in by_dev.items():
        node = DEVICES[dev][0]
        chain = "; ".join(cmd_line(*j) for j in js)
        r = ssh(node, f"cd ~/universe2 && mkdir -p results && (setsid nohup sh -c '{chain}' < /dev/null > /dev/null 2>&1 &); echo launched", timeout=30)
        assert "launched" in r.stdout, (dev, r.stdout, r.stderr)
    t0 = time.time()
    pending = set(jobs)
    while pending and time.time() - t0 < timeout:
        time.sleep(poll)
        for j in list(pending):
            dev, L, s, c, prims = j
            r = ssh(DEVICES[dev][0], f"tail -c 20 ~/universe2/results/{out_stem(dev, L, s, c, prims)}.err 2>/dev/null")
            if "done" in r.stdout:
                pending.discard(j)
    assert not pending, ("timeout", pending)
    return {j: fetch(*j) for j in jobs}


def build(nodes=None):
    for n in nodes or NODES:
        d = NODES[n]
        r = subprocess.run(["scp", "-q", "tools/u2_tables.c", f"{n}:~/universe2/{CPU_BIN}.c"], capture_output=True, text=True)
        assert r.returncode == 0, (n, r.stderr)
        cmd = f"cd ~/universe2 && gcc -O3 -march=native -fopenmp -std=c11 -Wall -Wextra {CPU_BIN}.c -o {CPU_BIN} 2>&1 | grep -c warning; ls -la {CPU_BIN} | cut -d' ' -f5-"
        if d["gpus"]:
            r = subprocess.run(["scp", "-q", "gpu/u2_mac_any.cu", f"{n}:~/universe2/u2_mac_any.cu"], capture_output=True, text=True)
            assert r.returncode == 0, (n, r.stderr)
            cmd += f"; nvcc -O3 -std=c++17 -arch={d['arch']} -o {GPU_BIN} u2_mac_any.cu 2>&1 | grep -c -i warning; ls -la {GPU_BIN} | cut -d' ' -f5-"
        r = ssh(n, cmd, timeout=900)
        print(n, r.stdout.strip().replace("\n", " | "), r.stderr.strip())
    selfcheck()


def selfcheck():
    """COPY and MAC_T over all 1,024 programs of 5 symbols on every device: identical, and the 64-step copier 01302 shows steps 57-64."""
    jobs = [(dev, 5, 0, 1024, "COPY,MAC_T") for dev in DEVICES]
    res = run_jobs(jobs, poll=5)
    base = res[jobs[0]][1]
    ok = True
    for j in jobs:
        prs = res[j][1]
        same = same_results(base, prs, "COPY,MAC_T")
        copier = [f for f in prs["COPY"]["found"] if f[0] == "01302"]
        good = same and prs["COPY"]["survivors"] == [1023, 117, 34, 16] and prs["COPY"]["found_count"] == 16 and copier and copier[0][1] == "57-64" and prs["MAC_T"]["found_count"] == 0
        ok &= bool(good)
        print(f"{j[0]:14s} {res[j][2]['device']:36s} {res[j][2]['secs']} s  {'SELF-CHECK OK' if good else 'SELF-CHECK FAILED'}")
    print("ALL DEVICES AGREE" if ok else "MISMATCH")
    return ok


def run_ref(procs=24):
    OUT.mkdir(parents=True, exist_ok=True)
    plan = [(l, "COPY,ADD,MUL_T,MAC_T,MAC_B,CARRY") for l in range(1, 9)] + [(l, "ADD,MUL_T,MAC_T,MAC_B,CARRY") for l in (9, 10)] + [(l, PRIMS) for l in (11, 12)]
    for l, prims in plan:
        path = OUT / f"ref_L{l}.json"
        if path.exists():
            continue
        subprocess.run([sys.executable, "tools/u2_mac_any_ref.py", "--L", str(l), "--prims", prims, "--procs", str(procs), "--out", str(path)], check=True)


def validate(L_max=12):
    """Every length 1..L_max: CPU on adler40, GPUs on adler40 and knecht24, the Python reference. Everything must agree."""
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"machine": MACHINE, "table_id": TABLE_ID, "W": W, "steps": STEPS, "devices": list(VALIDATE_DEVICES), "lengths": [], "ok": True}
    sums = {}
    for l in range(1, L_max + 1):
        rp = OUT / f"ref_L{l}.json"
        rj = json.loads(rp.read_text())
        prims = ",".join(p for p in ("MAC_T", "MAC_B", "ADD", "MUL_T", "COPY", "CARRY") if p in rj["primitives"])
        jobs = [(dev, l, 0, 4 ** l, prims) for dev in VALIDATE_DEVICES]
        # reuse finished runs
        res = {}
        todo = []
        for j in jobs:
            dev, L, s, c, pr = j
            r = ssh(DEVICES[dev][0], f"tail -c 20 ~/universe2/results/{out_stem(dev, L, s, c, pr)}.err 2>/dev/null")
            if "done" in r.stdout:
                res[j] = fetch(*j)
            else:
                todo.append(j)
        if todo:
            res.update(run_jobs(todo, poll=5))
        cpu = res[jobs[0]][1]
        rec = {"L": l, "programs": 4 ** l, "primitives": prims, "reference_secs": rj["secs"], "runs": {j[0]: {"secs": float(res[j][2]["secs"]), "device": res[j][2]["device"]} for j in jobs}, "per_prim": {}}
        for p in prims.split(","):
            r = rj["primitives"][p]
            ref_found = sorted((f[0], f[1]) for f in r["found_programs"])
            ref_detail = {f[0]: f[2] for f in r["found_programs"]}
            cpu_detail = {f[0]: f[2] for f in cpu[p]["found"]}
            surv_ok = cpu[p]["survivors"] == r["survivors_after_input"]
            found_ok = found_set(cpu, p) == ref_found
            detail_ok = all(cpu_detail[k] == ref_detail[k] for k in ref_detail) if found_ok else False
            fbs_ok = cpu[p]["found_by_step"] == r["found_by_step"]
            gpu_ok = all(same_results(cpu, res[j][1], p) for j in jobs[1:])
            ok = surv_ok and found_ok and detail_ok and fbs_ok and gpu_ok
            report["ok"] &= ok
            rec["per_prim"][p] = {"found": len(ref_found), "survivors_identical": surv_ok, "found_identical": found_ok, "cells_and_head_identical": detail_ok,
                                  "found_by_step_identical": fbs_ok, "gpus_identical_to_cpu": gpu_ok, "ok": ok, "survivors": cpu[p]["survivors"]}
            acc = sums.setdefault(p, [0] * len(cpu[p]["survivors"]))
            for i, v in enumerate(cpu[p]["survivors"]):
                acc[i] += v
            print(f"L={l:2d} {p:6s} found {len(ref_found):6d}  survivors {'ok' if surv_ok else 'DIFFER'}  found {'ok' if found_ok else 'DIFFER'}  cells/head {'ok' if detail_ok else 'DIFFER'}  "
                  f"by_step {'ok' if fbs_ok else 'DIFFER'}  gpus {'ok' if gpu_ok else 'DIFFER'}   cpu {float(res[jobs[0]][2]['secs']):.1f} s, ref {rj['secs']} s")
        report["lengths"].append(rec)
    # the any-step survivors over 1..12 must dominate the single-budget survivors of docs/13 at every input
    dom = {}
    for st in (64, 256):
        vp = pathlib.Path(f"results/u2_mac/validation_L12_s{st}.json")
        if L_max == 12 and vp.exists():
            v = json.loads(vp.read_text())
            for p in PRIMS.split(","):
                snap = v["primitives"][p]["survivors_after_input"]
                dom[f"{p}_s{st}"] = all(a >= b for a, b in zip(sums[p], snap))
    report["sum_1_to_12"] = sums
    report["dominates_single_budget_survivors"] = dom
    report["ok"] &= all(dom.values())
    print("dominance over the 64- and 256-step snapshot survivors (sum 1..12):", dom)
    path = OUT / f"validation_L{L_max}.json"
    path.write_text(json.dumps(report, indent=1))
    print("ALL IDENTICAL" if report["ok"] else "MISMATCH", "written", path)
    return report["ok"]


def calibrate(L=CAL_L):
    """MAC_T, MAC_B over all programs of L symbols on every device at once: speeds for the shards; outputs must agree."""
    set_cal(L)
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(dev, L, 0, 4 ** L, PRIMS) for dev in DEVICES]
    res = run_jobs(jobs, poll=10)
    base = res[jobs[0]][1]
    speeds, ok = {}, True
    for j in jobs:
        secs = float(res[j][2]["secs"])
        speeds[j[0]] = 4 ** L / secs
        same = same_results(base, res[j][1], PRIMS)
        ok &= same
        print(f"{j[0]:14s} {res[j][2]['device']:36s} {secs:8.1f} s  {4 ** L / secs / 1e6:8.2f} M programs/s  {'identical' if same else 'DIFFER'}")
    doc = {"L": L, "programs": 4 ** L, "prims": PRIMS, "steps": STEPS, "secs": {j[0]: float(res[j][2]["secs"]) for j in jobs}, "programs_per_s": speeds,
           "devices": {j[0]: res[j][2]["device"] for j in jobs}, "all_identical": ok,
           "survivors": {p: base[p]["survivors"] for p in PRIMS.split(",")}, "found": {p: base[p]["found_count"] for p in PRIMS.split(",")},
           "found_by_step": {p: base[p]["found_by_step"] for p in PRIMS.split(",")}, "when": time.strftime("%Y-%m-%dT%H:%M:%S")}
    CALIBRATION.write_text(json.dumps(doc, indent=1))
    print("ALL DEVICES IDENTICAL" if ok else "MISMATCH", "written", CALIBRATION)
    return ok


def plan(L, speeds):
    total = 4 ** L
    tw = sum(speeds[d] for d in DEVICES)
    shards, start = {}, 0
    devs = list(DEVICES)
    for i, d in enumerate(devs):
        count = total - start if i == len(devs) - 1 else round(total * speeds[d] / tw)
        shards[d] = (start, count)
        start += count
    return shards


def shard_list(Ls, speeds):
    runs = []
    for d in DEVICES:
        for L in Ls:
            s, c = plan(L, speeds)[d]
            runs.append((d, L, s, c, False))
        if d in OVERLAP_DEVICES:
            for L in Ls:
                runs.append((d, L, OVERLAP[L][0], OVERLAP[L][1], True))
    return runs


def launch(Ls):
    OUT.mkdir(parents=True, exist_ok=True)
    speeds = json.loads(CALIBRATION.read_text())["programs_per_s"]
    runs = shard_list(Ls, speeds)
    for d in DEVICES:
        node = DEVICES[d][0]
        chain = "; ".join(cmd_line(dd, L, s, c) for (dd, L, s, c, _) in runs if dd == d)
        r = ssh(node, f"cd ~/universe2 && mkdir -p results && (setsid nohup sh -c '{chain}' < /dev/null > /dev/null 2>&1 &); echo launched", timeout=30)
        print(d, r.stdout.strip() or r.stderr.strip(), [(L, f"{s:,} + {c:,}", "overlap" if o else "") for (dd, L, s, c, o) in runs if dd == d])
    tag = "_".join(f"L{L}" for L in Ls)
    (OUT / f"plan_any_{tag}.json").write_text(json.dumps({
        "launched": time.strftime("%Y-%m-%dT%H:%M:%S"), "machine": MACHINE, "table_id": TABLE_ID, "W": W, "steps": STEPS, "prims": PRIMS,
        "binaries": {"cpu": CPU_BIN, "gpu": GPU_BIN}, "lengths": Ls, "programs": {L: 4 ** L for L in Ls}, "devices": DEVICES, "speeds_programs_per_s": speeds,
        "overlap": {L: OVERLAP[L] for L in Ls}, "overlap_devices": OVERLAP_DEVICES,
        "runs": [{"device": d, "L": L, "start": s, "count": c, "overlap": o} for (d, L, s, c, o) in runs]}, indent=1))


def status(Ls):
    speeds = json.loads(CALIBRATION.read_text())["programs_per_s"]
    runs = shard_list(Ls, speeds)
    done = True
    for n in NODES:
        stems = [out_stem(d, L, s, c) for (d, L, s, c, _) in runs if DEVICES[d][0] == n]
        r = ssh(n, "cd ~/universe2/results 2>/dev/null; " + "; ".join(f"echo \"{st}: $(tail -c 120 {st}.err 2>/dev/null | tr '\\n' ' ')\"" for st in stems) + "; uptime | sed 's/.*load/load/'", timeout=120)
        print(f"=== {n} ===")
        for ln in r.stdout.splitlines():
            print("  " + ln.replace(f"_{PRIMS.replace(',', '-')}_W{W}", "").replace("mode=primany prims=MAC_T,MAC_B ops=INC_R1,INC_L1,FLIP_R2,JNZB W=2 ", "")[:170])
        for st in stems:
            line = [ln for ln in r.stdout.splitlines() if ln.startswith(st + ":")]
            if not line or "done" not in line[0]:
                done = False
    print("ALL DONE" if done else "running")
    return done


def reverify(p, program, detail):
    """A found program: the Python reference must give the same inputs-survived count and the same per-step cells and head flags."""
    table = u2_ops.parse_table(MACHINE)
    ns, ref_detail = ref.eval_any(table, [int(c) for c in program], W, STEPS, p)
    ok = ns == len(inputs_of(p)) and ref_detail is not None and [[d[0], d[1], int(d[2])] for d in ref_detail] == detail
    # and, independently, u2_ops.run at each claimed budget
    if ok:
        for s, cells, h in detail:
            for init, target in inputs_of(p):
                st = u2_ops.run(table, [int(c) for c in program], W, s, init=list(init))
                tape = st["tape_nonzero"]
                if any(tape.get(c, 0) != target for c in cells) or (h and tape.get(st["head"], 0) != target):
                    ok = False
    return ok


def collect(Ls):
    speeds = json.loads(CALIBRATION.read_text())["programs_per_s"]
    runs = shard_list(Ls, speeds)
    fetched = {(d, L, s, c): fetch(d, L, s, c) for (d, L, s, c, o) in runs}
    for L in Ls:
        shards, overlap = [], {}
        merged = {p: {"survivors": None, "found": {}, "found_by_step": [0] * STEPS} for p in PRIMS.split(",")}
        total = 0
        for (d, LL, s, c, o) in runs:
            if LL != L:
                continue
            head, prs, kv = fetched[(d, L, s, c)]
            rec = {"device": d, "node": DEVICES[d][0], "hardware": kv["device"], "start": s, "count": c, "secs": float(kv["secs"]), "threads": int(kv["threads"]),
                   "survivors": {p: prs[p]["survivors"] for p in merged}, "found": {p: prs[p]["found_count"] for p in merged}}
            if o:
                overlap[d] = rec
                continue
            shards.append(rec)
            total += c
            for p, m in merged.items():
                m["survivors"] = prs[p]["survivors"] if m["survivors"] is None else [a + b for a, b in zip(m["survivors"], prs[p]["survivors"])]
                m["found_by_step"] = [a + b for a, b in zip(m["found_by_step"], prs[p]["found_by_step"])]
                for prog, rg, detail in prs[p]["found"]:
                    m["found"][prog] = (rg, detail)
        assert total == 4 ** L, (L, total)
        ov_ok = None
        if len(overlap) == len(OVERLAP_DEVICES):
            base = fetched[(OVERLAP_DEVICES[0], L, OVERLAP[L][0], OVERLAP[L][1])][1]
            ov_ok = all(same_results(base, fetched[(d, L, OVERLAP[L][0], OVERLAP[L][1])][1], PRIMS) for d in OVERLAP_DEVICES[1:])
            assert ov_ok, ("overlap differs", L)
        found_out = {}
        for p, m in merged.items():
            lst = []
            for prog in sorted(m["found"]):
                rg, detail = m["found"][prog]
                if detail is None:   # found on a GPU: the per-step cells come from the CPU mode on that one program
                    q = int(prog, 4)
                    r = ssh("adler40", f"cd ~/universe2 && ./{CPU_BIN} primany {p} {W} {L} {STEPS} {MACHINE} {q} 1 1 2>/dev/null | grep '^F'", timeout=120)
                    _, prs1, _, _ = parse_output(f"prim\t{p}\n" + r.stdout, "done")
                    assert prs1[p]["found"] and prs1[p]["found"][0][0] == prog and prs1[p]["found"][0][1] == rg, (p, prog, rg, r.stdout[:200])
                    detail = prs1[p]["found"][0][2]
                lst.append({"program": prog, "steps": rg, "per_step": detail, "reverified": reverify(p, prog, detail)})
            found_out[p] = lst
        res = {"machine": MACHINE, "table_id": TABLE_ID, "W": W, "L": L, "steps": f"every budget 1..{STEPS}", "programs": 4 ** L,
               "definition": "the primitive at budget s: one fixed cell, or the cell under the head after exactly s instructions, holds the target after exactly s instructions for every input",
               "input_cells": "cells 0, 1, 2 = acc, w (or code), x",
               "primitives": {p: {"description": {"MAC_T": "acc + w * x mod 4, w in {1, 3, 0} (3 = -1)", "MAC_B": "acc + (c - 1) * x mod 4, BitNet code c in {2, 0, 1}"}[p],
                                  "inputs": len(m["survivors"]), "order": inputs_of(p),
                                  "survivors_after_input": m["survivors"], "found": len(m["found"]), "found_by_step": m["found_by_step"],
                                  "found_programs": found_out[p], "all_found_reverified": all(f["reverified"] for f in found_out[p])}
                              for p, m in merged.items()},
               "shards": shards, "overlap": {"range": OVERLAP[L], "devices": list(overlap), "identical": ov_ok, "runs": overlap},
               "collected": time.strftime("%Y-%m-%dT%H:%M:%S"), "collected_on": socket.gethostname()}
        path = OUT / f"any_L{L}.json"
        path.write_text(json.dumps(res, indent=1))
        for p, m in merged.items():
            last = max((i for i, v in enumerate(m["survivors"]) if v), default=-1) + 1
            print(f"L={L} {p}: {4 ** L:,} programs, survivors {m['survivors'][:6]}..., frontier input {last}, found {len(m['found'])}"
                  + (f" {[f['program'] + ' @' + f['steps'] for f in found_out[p][:5]]}" if m["found"] else ""))
        print("   shards: " + ", ".join(f"{s['device']} {s['count']:,} in {s['secs']:.0f} s" for s in shards)
              + (f"; overlap {OVERLAP[L][1]:,} programs on {len(overlap)} devices: {'identical' if ov_ok else 'DIFFER'}" if ov_ok is not None else ""))
        print("written", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["build", "selfcheck", "ref", "validate", "calibrate", "launch", "status", "collect"])
    ap.add_argument("nodes", nargs="*")
    ap.add_argument("--L", type=int, action="append", choices=sorted(OVERLAP))
    ap.add_argument("--Lmax", type=int, default=12, help="validate: lengths 1..Lmax")
    ap.add_argument("--procs", type=int, default=24, help="ref: local processes")
    ap.add_argument("--calL", type=int, default=CAL_L, help="calibration length (calibrate runs it; launch/status/collect read its speeds)")
    a = ap.parse_args()
    set_cal(a.calL)
    Ls = a.L or [13, 14, 15, 16]
    if a.action == "build":
        build(a.nodes or None)
    elif a.action == "selfcheck":
        sys.exit(0 if selfcheck() else 1)
    elif a.action == "ref":
        run_ref(a.procs)
    elif a.action == "validate":
        sys.exit(0 if validate(a.Lmax) else 1)
    elif a.action == "calibrate":
        sys.exit(0 if calibrate(a.calL) else 1)
    elif a.action == "launch":
        launch(Ls)
    elif a.action == "status":
        status(Ls)
    else:
        collect(Ls)


if __name__ == "__main__":
    main()
