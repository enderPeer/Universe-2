"""Build bigger functions out of mapped NANO programs: chains (composition) with exact verification.

A mapped program is a function f(a, b) = first OUT within 64 steps, a in M5, b in M6 (the definition behind
the Dimension42 catalog, the adder search and the six-byte formulas). A chain is a pipeline: stage k+1 gets
stage k's answer as its a, b is supplied again, memory is fresh for every stage. If any stage gives no
answer the chain gives none. Tables use 256 as "no answer".

  python tools/nano_compose.py 456650D0 17C5915071 --formula "(a+b)^b^7"       # verify a chain
  python tools/nano_compose.py 17C5915071 17C5915071 --formula a                # XOR-7 twice is the identity
  python tools/nano_compose.py --basis basis.txt --target "(a*b)&255" --depth 2  # search chains over a basis
  python tools/nano_compose.py 456650D0 --table                                 # print the 256x256 table summary

Basis file: one hex program per line, '#' comments allowed. Formulas are Python over integers a, b.
"""
import argparse
import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import nano_ref  # noqa: E402

NONE = 256


def table(program):
    """256x256 uint16 array t[a, b] = first OUT, or 256 for no answer."""
    t = np.full((256, 256), NONE, dtype=np.uint16)
    for a in range(256):
        for b in range(256):
            v = nano_ref.first_out(program, a, b)
            if v is not None:
                t[a, b] = v
    return t


def compose_tables(t1, t2):
    """Table of stage2(stage1(a,b), b). Rows of t2 are indexed by stage-1 output; row 256 = no answer."""
    t2x = np.vstack([t2, np.full((1, 256), NONE, dtype=np.uint16)])
    bcol = np.broadcast_to(np.arange(256, dtype=np.intp), (256, 256))
    return t2x[t1.astype(np.intp), bcol]


def chain_table(tables):
    t = tables[0]
    for nxt in tables[1:]:
        t = compose_tables(t, nxt)
    return t


def formula_table(formula):
    a = np.arange(256).reshape(256, 1).astype(np.int64)
    b = np.arange(256).reshape(1, 256).astype(np.int64)
    env = {"a": a, "b": b, "np": np, "__builtins__": {}}
    v = eval(formula, env)  # user-supplied expression over integer arrays
    return (np.broadcast_to(v, (256, 256)) & 255).astype(np.uint16)


def describe(t):
    answered = int((t != NONE).sum())
    vals = t[t != NONE]
    distinct = len(np.unique(vals)) if answered else 0
    dep_a = bool((t != t[0:1, :]).any())
    dep_b = bool((t != t[:, 0:1]).any())
    return {"answered_pairs": answered, "distinct_outputs": distinct, "depends_a": dep_a, "depends_b": dep_b}


def load_basis(path):
    progs = []
    for line in pathlib.Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip().replace(" ", "")
        if line:
            progs.append(bytes.fromhex(line))
    return progs


def search(basis, target, depth):
    """Exhaustive chain search over the basis up to the given depth. Returns matching chains."""
    tabs = []
    for i, p in enumerate(basis):
        tabs.append(table(p))
        print(f"\rtables {i + 1}/{len(basis)}", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    hits = []
    n = len(basis)
    # depth 1
    for i in range(n):
        if np.array_equal(tabs[i], target):
            hits.append([basis[i]])
    frontier = [([i], tabs[i]) for i in range(n)]
    for d in range(2, depth + 1):
        new = []
        for idx, t in frontier:
            for j in range(n):
                c = compose_tables(t, tabs[j])
                if np.array_equal(c, target):
                    hits.append([basis[k] for k in idx + [j]])
                new.append((idx + [j], c))
        frontier = new
        print(f"depth {d}: {len(frontier):,} chains examined, {len(hits)} hits so far", file=sys.stderr)
    return hits


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("programs", nargs="*", help="hex programs forming a chain, stage 1 first")
    ap.add_argument("--formula", help="expected f(a,b), Python over integers a, b, taken mod 256")
    ap.add_argument("--table", action="store_true", help="describe the chain's table")
    ap.add_argument("--basis", help="file of hex programs to search chains over")
    ap.add_argument("--target", help="formula to search for (with --basis)")
    ap.add_argument("--depth", type=int, default=2)
    args = ap.parse_args()

    if args.basis:
        basis = load_basis(args.basis)
        target = formula_table(args.target)
        hits = search(basis, target, args.depth)
        print(f"target {args.target!r}: {len(hits)} chain(s) of depth <= {args.depth} over {len(basis)} programs")
        for h in hits:
            print("  " + " -> ".join(p.hex().upper() for p in h))
        return

    progs = [nano_ref.parse(p) for p in args.programs]
    if not progs:
        ap.error("give programs or --basis")
    for p in progs:
        print(f"{p.hex().upper()}: " + " ; ".join(x.split(" ", 1)[1] for x in nano_ref.disasm(p)))
    t = chain_table([table(p) for p in progs])
    print("chain:", describe(t))
    if args.formula:
        want = formula_table(args.formula)
        wrong = int((t != want).sum())
        missing = int((t == NONE).sum())
        print(f"formula {args.formula!r}: {65536 - wrong} / 65536 pairs correct, {wrong} wrong ({missing} without answer)")
        if wrong:
            a, b = np.argwhere(t != want)[0]
            print(f"  first failure a={a} b={b}: chain {'none' if t[a, b] == NONE else int(t[a, b])}, formula {int(want[a, b])}")
            sys.exit(1)


if __name__ == "__main__":
    main()
