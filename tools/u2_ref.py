"""Universe 2 candidate machines: one 2-bit symbol per instruction, four instructions, a tape with a head.

With two bits there is no room for an operand, so every instruction acts on the cell under the head, and the
head's position is the only "address". The designs differ in how they loop, which is the real question at 2 bits.

  P4  Bohm's P'' (1964), the ancestor of Brainfuck, proven Turing complete with a long enough tape:
        0 >   head += 1
        1 L   cell += 1 (mod 2^W), then head -= 1
        2 [   if cell == 0: jump past the matching ]   (no match: stop)
        3 ]   if cell != 0: jump back to after the matching [   (no match: jump to the start)
      A program stops when the PC runs off the end, or at the step budget.
  S4  skip-based, the program wraps around (implicit outer loop, like NANO's PC wrap), runs for the budget:
        0 >   head += 1
        1 +   cell += 1 (mod 2^W)
        2 ?   skip the next instruction if cell == 0
        3 <   head -= 1
  M4  no control flow at all (baseline), wraps, runs for the budget:
        0 >   head += 1      1 +   cell += 1      2 <   head -= 1      3 -   cell -= 1

Tape: N cells of W bits, head starts at cell 0, wraps around. Programs are strings over {0,1,2,3}.

  python tools/u2_ref.py P4 "2 1 0 3" --W 1 --trace
  python tools/u2_ref.py --sweep 8 --W 1 --steps 64      # every program of 1..8 symbols, all three designs
"""
import argparse
import json
import sys
import time
from collections import Counter

N = 16
DESIGNS = ("P4", "S4", "M4")


def matching(program):
    """For P4: index of the matching bracket for every [ and ], or -1."""
    m = [-1] * len(program)
    stack = []
    for i, s in enumerate(program):
        if s == 2:
            stack.append(i)
        elif s == 3 and stack:
            j = stack.pop()
            m[i], m[j] = j, i
    return m


def run(design, program, W=1, steps=64, trace=False):
    mask = (1 << W) - 1
    tape = [0] * N
    head = pc = 0
    L = len(program)
    halted = False
    m = matching(program) if design == "P4" else None
    hist = []
    for step in range(1, steps + 1):
        if pc >= L:
            halted = True
            break
        s = program[pc]
        if trace:
            hist.append((step, pc, s, head, tape[head]))
        pc += 1
        if design == "P4":
            if s == 0:
                head = (head + 1) % N
            elif s == 1:
                tape[head] = (tape[head] + 1) & mask
                head = (head - 1) % N
            elif s == 2:
                if tape[head] == 0:
                    pc = m[pc - 1] + 1 if m[pc - 1] >= 0 else L
            else:
                if tape[head] != 0:
                    pc = m[pc - 1] + 1 if m[pc - 1] >= 0 else 0
        elif design == "S4":
            if s == 0:
                head = (head + 1) % N
            elif s == 1:
                tape[head] = (tape[head] + 1) & mask
            elif s == 2:
                if tape[head] == 0:
                    pc += 1
            else:
                head = (head - 1) % N
            pc %= L
        else:
            if s == 0:
                head = (head + 1) % N
            elif s == 1:
                tape[head] = (tape[head] + 1) & mask
            elif s == 2:
                head = (head - 1) % N
            else:
                tape[head] = (tape[head] - 1) & mask
            pc %= L
    else:
        step = steps
    return {"design": design, "program": "".join(map(str, program)), "W": W, "tape": tape, "head": head,
            "halted": halted, "steps": step if halted else steps, "trace": hist if trace else None}


def sweep(max_len, W, steps):
    out = {}
    for design in DESIGNS:
        t0 = time.perf_counter()
        per_len = []
        union = set()
        for L in range(1, max_len + 1):
            tapes = set()
            halts = 0
            touched = 0
            for p in range(4 ** L):
                program = [(p >> (2 * (L - 1 - i))) & 3 for i in range(L)]
                r = run(design, program, W, steps)
                key = bytes(r["tape"])
                tapes.add(key)
                halts += r["halted"]
                touched += any(r["tape"])
            union |= tapes
            per_len.append({"length": L, "programs": 4 ** L, "distinct_tapes": len(tapes),
                            "distinct_percent": round(100 * len(tapes) / 4 ** L, 2),
                            "halted_percent": round(100 * halts / 4 ** L, 2),
                            "nonblank_percent": round(100 * touched / 4 ** L, 2)})
        total = sum(4 ** L for L in range(1, max_len + 1))
        out[design] = {"per_length": per_len, "programs_total": total, "distinct_tapes_union": len(union),
                       "union_percent": round(100 * len(union) / total, 2),
                       "tape_states_possible": (1 << W) ** N, "seconds": round(time.perf_counter() - t0, 1)}
        print(f"{design} W={W}: {total:,} programs, {len(union):,} distinct final tapes ({out[design]['union_percent']} %), "
              f"{out[design]['seconds']} s", file=sys.stderr)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("design", nargs="?", choices=DESIGNS)
    ap.add_argument("program", nargs="?", help="symbols 0..3, spaces optional")
    ap.add_argument("--W", type=int, default=1, help="bits per tape cell")
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--sweep", type=int, metavar="MAXLEN", help="run every program of 1..MAXLEN symbols, all designs")
    ap.add_argument("--out", default="results/u2_sweep")
    args = ap.parse_args()
    if args.sweep:
        import pathlib
        res = sweep(args.sweep, args.W, args.steps)
        res["_meta"] = {"max_len": args.sweep, "W": args.W, "steps": args.steps, "tape_cells": N}
        out = pathlib.Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"sweep_len{args.sweep}_W{args.W}_steps{args.steps}.json"
        path.write_text(json.dumps(res, indent=1))
        print(path)
        return
    if not (args.design and args.program):
        ap.error("design and program, or --sweep")
    program = [int(c) for c in args.program.replace(" ", "")]
    r = run(args.design, program, args.W, args.steps, args.trace)
    if r["trace"]:
        for step, pc, s, head, cell in r["trace"]:
            print(f"step {step:3d} pc {pc:2d} sym {s} head {head:2d} cell {cell}")
    r.pop("trace")
    print(json.dumps(r))


if __name__ == "__main__":
    main()
