"""Rank candidate tasks for the DNA-3 long ladder (docs/17 section 18) by how hard they are on the organisms' own
machine: the length of the shortest instruction sequence that computes the task on the two I/O cells.

  python tools/dna3_ladder_rank.py [--max-len 7] [--json results/dna3/ladder_rank.json]

Model (exactly the DNA-3 semantics restricted to the I/O cells): after IN the high half of the 4-bit input x is in
cell G-2 and the low half in cell G-1, the head H is on the low cell and A = low half. Instructions: FLIP (cell ^= 1),
INC (cell = cell + 1 mod 4), LDA (A = cell), STA (cell = A), L1 (low -> high cell; from the high cell the head would
leave the I/O cells, which is excluded), R1 (high -> low), SKZ / SKNZ (skip the next instruction if the cell is / is
not 0), JNZB (if the cell is not 0, jump back to the instruction before it). A program runs once and must finish within 64
instructions (a loop that never ends would trap an organism); its answer is 4 * high + low. A task is a target function of x, compared on all four bits (unary) or on
the low cell only (two-input tasks on the halves a = high, b = low). Every sequence of 1 to max-len instructions is
tried; the rank of a task is its shortest length. Candidates where some answer value covers more than half of the
inputs are dropped (a constant answer would match too often), as are those not found within max-len.
"""
import argparse
import itertools
import json

OPS = ("FLIP", "INC", "LDA", "STA", "L1", "R1", "SKZ", "SKNZ", "JNZB")


def run(prog, x):
    """The answer of prog to the 4-bit input x, or None if the program hangs or reads or writes a cell outside the
    two I/O cells (the head may move outside and back; an access there is excluded). Besides OPS, a program may
    contain NOP (nothing) and R3 / L3 (moves of three cells); these are for the bodies that
    tools/dna3_extract_solutions.py cuts out of evolved genes (docs/17 section 19), the sweep of this tool uses OPS
    only. Shortest lengths are the same under either reading of a move outside: a program that moves out and back
    without an access there has a shorter equivalent."""
    cells = [x >> 2, x & 3]          # index 0 = high (G-2), 1 = low (G-1)
    h, a, pc, steps = 1, x & 3, 0, 0
    n = len(prog)
    while pc < n and steps < 64:
        op = prog[pc]
        pc += 1
        steps += 1
        if op == "NOP":
            continue
        if op == "L1":
            h -= 1
            continue
        if op == "R1":
            h += 1
            continue
        if op == "L3":
            h -= 3
            continue
        if op == "R3":
            h += 3
            continue
        if h not in (0, 1):              # an access outside the I/O cells
            return None
        if op == "FLIP":
            cells[h] ^= 1
        elif op == "INC":
            cells[h] = (cells[h] + 1) & 3
        elif op == "LDA":
            a = cells[h]
        elif op == "STA":
            cells[h] = a
        elif op == "SKZ":
            if cells[h] == 0:
                pc += 1
        elif op == "SKNZ":
            if cells[h] != 0:
                pc += 1
        elif op == "JNZB":
            if cells[h] != 0:
                pc = max(0, pc - 2)
    if pc < n:                       # still running after 64 steps: an organism would hang in this loop
        return None
    return 4 * cells[0] + cells[1]


# candidates: (name, function of x, mask) ; mask 15 = all four bits, 3 = low cell only
def candidates():
    c = [
        ("x^1", lambda x: x ^ 1), ("x^4", lambda x: x ^ 4), ("x^5", lambda x: x ^ 5), ("15-x", lambda x: 15 - x),
        ("x+1", lambda x: (x + 1) & 15), ("x-1", lambda x: (x - 1) & 15), ("x+2", lambda x: (x + 2) & 15),
        ("x+4", lambda x: (x + 4) & 15), ("x+8", lambda x: (x + 8) & 15), ("-x", lambda x: (-x) & 15),
        ("x^8", lambda x: x ^ 8), ("2x", lambda x: (2 * x) & 15), ("x/2", lambda x: x >> 1), ("4x", lambda x: (4 * x) & 15),
        ("x/4", lambda x: x >> 2), ("swap", lambda x: ((x & 3) << 2) | (x >> 2)), ("x|1", lambda x: x | 1),
        ("x&14", lambda x: x & 14), ("x&3", lambda x: x & 3), ("x&12", lambda x: x & 12), ("x|3", lambda x: x | 3),
        ("parity", lambda x: bin(x).count("1") & 1), ("x>=8", lambda x: int(x >= 8)), ("x<8", lambda x: int(x < 8)),
        ("2x+1", lambda x: (2 * x + 1) & 15), ("3x", lambda x: (3 * x) & 15), ("5x", lambda x: (5 * x) & 15),
        ("rotl", lambda x: ((x << 1) | (x >> 3)) & 15), ("rotr", lambda x: ((x >> 1) | (x << 3)) & 15),
        ("bitrev", lambda x: int("{:04b}".format(x)[::-1], 2)), ("x%3", lambda x: x % 3),
    ]
    out = [(n, f, 15) for n, f in c]
    two = [
        ("a xor b", lambda a, b: a ^ b), ("a+b", lambda a, b: (a + b) & 3), ("b-a", lambda a, b: (b - a) & 3),
        ("a-b", lambda a, b: (a - b) & 3), ("a and b", lambda a, b: a & b), ("a or b", lambda a, b: a | b),
        ("max", lambda a, b: max(a, b)), ("min", lambda a, b: min(a, b)), ("a==b", lambda a, b: int(a == b)),
        ("a", lambda a, b: a), ("b xor 3a", lambda a, b: b ^ (3 * a & 3)),
    ]
    out += [(n, (lambda f: lambda x: f(x >> 2, x & 3))(f), 3) for n, f in two]
    return out


def balanced(f, mask):
    counts = {}
    for x in range(16):
        v = f(x) & mask
        counts[v] = counts.get(v, 0) + 1
    return max(counts.values()) <= 8


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-len", type=int, default=7)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    cands = [(n, f, m) for n, f, m in candidates()]
    targets = {n: (tuple(f(x) & m for x in range(16)), m) for n, f, m in cands}
    best = {}
    for L in range(1, a.max_len + 1):
        for prog in itertools.product(OPS, repeat=L):
            outs = []
            ok = True
            for x in range(16):
                r = run(prog, x)
                if r is None:
                    ok = False
                    break
                outs.append(r)
            if not ok:
                continue
            for n, (tt, m) in targets.items():
                if n in best:
                    continue
                if all((outs[x] & m) == tt[x] for x in range(16)):
                    best[n] = (L, " ".join(prog))
        print("length %d done, %d tasks found" % (L, len(best)), flush=True)
    rows = []
    for n, f, m in cands:
        bal = balanced(f, m)
        L, p = best.get(n, (None, None))
        rows.append({"task": n, "mask": m, "balanced": bal, "shortest": L, "program": p})
        print("%-10s mask %2d balanced %-5s shortest %s  %s" % (n, m, bal, L, p or "-"))
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
