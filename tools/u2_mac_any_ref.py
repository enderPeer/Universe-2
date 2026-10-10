"""Python reference for the any-step primitive search (docs/14): the primitives of tools/u2_mac.py at every step
budget 1..steps at once, over every program of exactly L symbols, written for checking the C mode `primany` of
tools/u2_tables.c and the GPU program gpu/u2_mac_any.cu on small L.

A program is the primitive "at budget s" when one fixed cell, or the cell under the head after exactly s
instructions, holds the target after exactly s instructions for every input (inputs and order of u2_mac.inputs_of,
i.e. of Ghost/bitnet_u2/prim_search.py; COPY excludes cell 0 and a head at cell 0). Per program and primitive the
reference records the tape after every step of every input as a (steps x 256) array and intersects the boolean
"cell == target" arrays over the inputs with numpy; no bit tricks, nothing incremental, so it is independent of the
two implementations it checks. step_states() is a copy of the loop of u2_ops.run that yields the state after every
step; --selfcheck compares it with u2_ops.run on random programs, inputs and budgets.

  python tools/u2_mac_any_ref.py --selfcheck 2000
  python tools/u2_mac_any_ref.py --L 6 --prims COPY,ADD,MUL_T,MAC_T,MAC_B,CARRY --out results/u2_mac_any/ref_L6.json
  python tools/u2_mac_any_ref.py --L 8 --prims MAC_T,MAC_B --procs 16 --out results/u2_mac_any/ref_L8.json
  python tools/u2_mac_any_ref.py --program 01302 --prims COPY            # one program, prints its steps and cells

Output JSON per primitive: survivors_after_input (programs alive after each input: some step still has a candidate),
found (count), found_programs [[program, ranges, [[s, cells, head_flag], ...]], ...], found_by_step (steps counts).
"""
import argparse
import json
import pathlib
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import u2_ops  # noqa: E402
from u2_mac import MACHINE, W as W_DEFAULT, inputs_of  # noqa: E402

N = 256


def step_states(table, program, W, steps, init):
    """The state after each of the first `steps` instructions: a list of (tape bytes, head). After HALT the state repeats."""
    mask = (1 << W) - 1
    tape = bytearray(N)
    for i, v in enumerate(init):
        tape[i] = v
    head = pc = A = 0
    L = len(program)
    halted = False
    m = u2_ops.matching(program, table)
    out = []
    for _step in range(steps):
        if not halted:
            s = program[pc]
            op = u2_ops.NAMES[table[s]]
            at = pc
            pc = (pc + 1) % L
            v = tape[head]
            if op == "R1": head = (head + 1) % N
            elif op == "L1": head = (head - 1) % N
            elif op == "R2": head = (head + 2) % N
            elif op == "L2": head = (head - 2) % N
            elif op == "R3": head = (head + 3) % N
            elif op == "L3": head = (head - 3) % N
            elif op == "INC": tape[head] = (v + 1) & mask
            elif op == "DEC": tape[head] = (v - 1) & mask
            elif op == "FLIP": tape[head] = v ^ 1
            elif op == "NOT": tape[head] = v ^ mask
            elif op == "ZERO": tape[head] = 0
            elif op == "SET": tape[head] = mask
            elif op == "SHL": tape[head] = (v << 1) & mask
            elif op == "INC_R1": tape[head] = (v + 1) & mask; head = (head + 1) % N
            elif op == "INC_L1": tape[head] = (v + 1) & mask; head = (head - 1) % N
            elif op == "FLIP_R1": tape[head] = v ^ 1; head = (head + 1) % N
            elif op == "FLIP_L1": tape[head] = v ^ 1; head = (head - 1) % N
            elif op == "FLIP_R2": tape[head] = v ^ 1; head = (head + 2) % N
            elif op == "FLIP_R3": tape[head] = v ^ 1; head = (head + 3) % N
            elif op == "SET_R1": tape[head] = mask; head = (head + 1) % N
            elif op == "SET_R3": tape[head] = mask; head = (head + 3) % N
            elif op == "ZERO_R1": tape[head] = 0; head = (head + 1) % N
            elif op == "ZERO_R3": tape[head] = 0; head = (head + 3) % N
            elif op == "SKZ":
                if v == 0: pc = (pc + 1) % L
            elif op == "SKNZ":
                if v != 0: pc = (pc + 1) % L
            elif op == "JZS":
                if v == 0: pc = 0
            elif op == "JNZS":
                if v != 0: pc = 0
            elif op == "JNZB":
                if v != 0: pc = (pc - 2) % L
            elif op == "OPEN":
                if v == 0: pc = (m[at] + 1) % L if m[at] >= 0 else 0
            elif op == "CLOSE":
                if v != 0: pc = (m[at] + 1) % L if m[at] >= 0 else 0
            elif op == "HALT":
                halted = True
            elif op == "NOP": pass
            elif op == "LDA": A = v
            elif op == "STA": tape[head] = A
            elif op == "SWAP": tape[head], A = A, v
            elif op == "ADDA": tape[head] = (v + A) & mask
            elif op == "EAT": A = v; tape[head] = 0
            elif op == "MATE_A": tape[head] = (v + A + 1) & mask; head = (head + 1) % N
            elif op == "MATE_B": tape[head] = (A + 1) & mask; head = (head + 1) % N
            elif op == "SLEEP":
                if A == 0: pc = (pc + 1) % L
            elif op == "WASTE": tape[head] = mask; A = 0; head = (head - 1) % N
            else:
                raise ValueError(op)
        out.append((bytes(tape), head))
    return out


def tapes_array(table, program, W, steps, init):
    """(steps x 256) uint8 array of the tape after each step, and the head after each step (steps,)."""
    st = step_states(table, program, W, steps, init)
    T = np.frombuffer(b"".join(t for t, _ in st), dtype=np.uint8).reshape(steps, N)
    H = np.fromiter((h for _, h in st), dtype=np.int64, count=steps)
    return T, H


def eval_any(table, program, W, steps, prim, maxin=0):
    """(inputs survived, per-step detail) for one program and primitive; detail = [(s, cells, head_flag)] when found.
    maxin > 0 keeps only the first maxin inputs (the frontier exhibits of docs/14 section 7)."""
    inputs = inputs_of(prim)
    if maxin > 0:
        inputs = inputs[:maxin]
    excl0 = prim == "COPY"
    cand = None
    hok = None
    for i, (init, target) in enumerate(inputs):
        T, H = tapes_array(table, program, W, steps, list(init))
        eq = T == target
        if excl0:
            eq[:, 0] = False
        h = T[np.arange(steps), H] == target
        if excl0:
            h &= H != 0
        if cand is None:
            cand, hok = eq, h
        else:
            cand &= eq
            hok &= h
        if not (cand.any() or hok.any()):
            return i, None
    detail = []
    for s in range(steps):
        cells = np.flatnonzero(cand[s]).tolist()
        if cells or hok[s]:
            detail.append((s + 1, cells, bool(hok[s])))
    return len(inputs), detail


def ranges(steps_list):
    out, start, prev = [], None, None
    for s in steps_list + [None]:
        if start is None:
            start = prev = s
        elif s is not None and s == prev + 1:
            prev = s
        else:
            out.append(f"{start}-{prev}" if prev > start else f"{start}")
            start = prev = s
    return ",".join(out)


def chunk_job(args):
    table, L, W, steps, prims, q0, q1 = args
    res = {p: {"hist": [0] * (len(inputs_of(p)) + 1), "found": [], "fbs": [0] * steps} for p in prims}
    for q in range(q0, q1):
        program = [(q >> (2 * (L - 1 - i))) & 3 for i in range(L)]
        for p in prims:
            ns, detail = eval_any(table, program, W, steps, p)
            res[p]["hist"][ns] += 1
            if detail is not None:
                res[p]["found"].append(["".join(map(str, program)), ranges([d[0] for d in detail]), [[d[0], d[1], int(d[2])] for d in detail]])
                for d in detail:
                    res[p]["fbs"][d[0] - 1] += 1
    return res


def search(table, L, W, steps, prims, procs=1):
    total = 4 ** L
    nchunks = max(1, min(total, procs * 8))
    bounds = [(total * i // nchunks, total * (i + 1) // nchunks) for i in range(nchunks)]
    jobs = [(table, L, W, steps, prims, a, b) for a, b in bounds if b > a]
    if procs > 1:
        with ProcessPoolExecutor(procs) as ex:
            parts = list(ex.map(chunk_job, jobs))
    else:
        parts = [chunk_job(j) for j in jobs]
    out = {}
    for p in prims:
        nin = len(inputs_of(p))
        hist = [sum(part[p]["hist"][i] for part in parts) for i in range(nin + 1)]
        surv = [sum(hist[i + 1:]) for i in range(nin)]
        found = sorted((f for part in parts for f in part[p]["found"]), key=lambda f: f[0])
        fbs = [sum(part[p]["fbs"][s] for part in parts) for s in range(steps)]
        out[p] = {"inputs": nin, "order": inputs_of(p), "programs": total, "survivors_after_input": surv,
                  "found": len(found), "found_programs": found, "found_by_step": fbs}
    return out


def selfcheck(n, seed=1):
    """step_states against u2_ops.run on random machines, programs, inputs and budgets."""
    rnd = random.Random(seed)
    machines = [MACHINE, "INC,INC_R1,INC_L1,JNZB", "L1,INC_L1,FLIP_R2,JNZB", "R1,INC_L1,OPEN,CLOSE", "INC_R1,HALT,SKZ,L1", "LDA,STA,R1,JNZS"]
    for _ in range(n):
        table = u2_ops.parse_table(rnd.choice(machines))
        L = rnd.randint(1, 8)
        program = [rnd.randrange(4) for _ in range(L)]
        W = rnd.choice([1, 2, 2, 2, 3])
        init = [rnd.randrange(1 << W) for _ in range(rnd.randint(0, 3))]
        steps = rnd.randint(1, 256)
        st = step_states(table, program, W, steps, init)
        for s in sorted(set([steps, rnd.randint(1, steps), rnd.randint(1, steps)])):
            ref = u2_ops.run(table, program, W, s, init=init)
            tape, head = st[s - 1]
            nz = {i: tape[i] for i in range(N) if tape[i]}
            assert nz == ref["tape_nonzero"] and head == ref["head"], (table, program, W, init, s)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--machine", default=MACHINE)
    ap.add_argument("--W", type=int, default=W_DEFAULT)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--L", type=int)
    ap.add_argument("--prims", default="MAC_T,MAC_B")
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--out")
    ap.add_argument("--program", help="evaluate one program instead of all of length L")
    ap.add_argument("--selfcheck", type=int, metavar="N")
    a = ap.parse_args()
    table = u2_ops.parse_table(a.machine)
    prims = a.prims.split(",")
    if a.selfcheck:
        t0 = time.time()
        selfcheck(a.selfcheck)
        print(f"selfcheck ok: {a.selfcheck} random (machine, program, input, budget) cases agree with u2_ops.run, {time.time() - t0:.1f} s")
        return
    if a.program:
        program = [int(c) for c in a.program]
        for p in prims:
            ns, detail = eval_any(table, program, a.W, a.steps, p)
            print(p, "inputs survived", ns, "of", len(inputs_of(p)), "" if detail is None else ranges([d[0] for d in detail]))
            if detail:
                for s, cells, h in detail:
                    print(f"   step {s}: cells {cells} head {h}")
        return
    t0 = time.time()
    res = search(table, a.L, a.W, a.steps, prims, a.procs)
    secs = time.time() - t0
    for p in prims:
        r = res[p]
        print(f"L={a.L} {p}: {r['programs']:,} programs, survivors {r['survivors_after_input'][:6]}..., found {r['found']}")
    doc = {"machine": a.machine, "W": a.W, "L": a.L, "steps": a.steps, "programs": 4 ** a.L, "secs": round(secs, 1), "procs": a.procs,
           "reference": "tools/u2_mac_any_ref.py", "primitives": res}
    if a.out:
        pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(a.out).write_text(json.dumps(doc))
        print("written", a.out, f"{secs:.1f} s")


if __name__ == "__main__":
    main()
