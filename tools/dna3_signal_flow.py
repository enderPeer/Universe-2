"""Do the organisms of a DNA-3 day-6 world communicate? A static data-flow reading of every living gene of a checkpoint
(docs/17 section 16).

  python tools/dna3_signal_flow.py RUN.ckpt [--days 1,2,3,4,5,6] [--json out.json]

For each gene (codon 0 up to its first STOP, read with the day-6 codon table) the codons are parsed in execution order
from codon 1, cyclically, and every operation that takes an operand (SEEK, SENSE, EAT, MOVE, EXEC, LISTEN) marks the
next codon as its operand, which is not executed as an instruction. This is the straight-line reading of the loop:
skips (TERM, SENSE, SKZ, SKNZ) and jumps (JNZB) can make a different path run, so the result is an approximation of
what executes, not a trace. Then:
- an executed LISTEN loads a neighbour's signal into A; its value is followed forward (cyclically) to the first
  operation that uses A (OUT: answers a task with it, SIGNAL: relays it, STA: writes it to the tape) or overwrites A
  (IN, LDA, LISTEN). Only "used" LISTENs can carry information into behaviour;
- an executed SIGNAL broadcasts A; its value is followed backward to the operation that last set A (IN: the input,
  LDA: what is on the tape under the head, typically the computed answer, LISTEN: a relayed signal).
Prints the counts over all living organisms and the most common genes with a used LISTEN.
"""
import argparse
import collections
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sim"))
import dna_ref as R  # noqa: E402
import dna3_ckpt_genes as K  # noqa: E402

TAKES_OPERAND = {R.SEEK, R.SENSE, R.EAT, R.MOVE, R.EXEC, R.LISTEN}
SETS_A = {R.IN, R.LDA, R.LISTEN}
USES_A = {R.OUT, R.SIGNAL, R.STA}


def analyse(ops):
    """ops: operation codes of a gene, START at 0. Returns (listen outcomes, signal sources)."""
    n = len(ops)
    if n < 2:
        return [], []
    executed = [False] * n
    k = 1
    # one pass of the straight-line loop over codons 1 .. n-1
    while k < n:
        executed[k] = True
        k += 2 if ops[k] in TAKES_OPERAND else 1
    order = [k for k in range(1, n) if executed[k]]
    m = len(order)
    listens, signals = [], []
    for i, k in enumerate(order):
        if ops[k] == R.LISTEN:
            out = "overwritten"
            for j in range(1, m + 1):
                o = ops[order[(i + j) % m]]
                if o in USES_A:
                    out = R.OP_NAMES[o]
                    break
                if o in SETS_A:
                    break
            listens.append(out)
        elif ops[k] == R.SIGNAL:
            src = "nothing sets A"
            for j in range(1, m + 1):
                o = ops[order[(i - j) % m]]
                if o in SETS_A:
                    src = R.OP_NAMES[o]
                    break
            signals.append(src)
    return listens, signals


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt")
    ap.add_argument("--days", default="1,2,3,4,5,6")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    table = np.array(K.table_for([int(x) for x in a.days.split(",")], False), dtype=np.int16)
    with open(a.ckpt, "rb") as f:
        f.read(8)
        ver, N, G, S, seed, flags, t_next = struct.unpack("<7Q", f.read(56))
        nb = (N + 3) // 4
        fw = np.frombuffer(f.read(nb), dtype=np.uint8)
        f.seek(nb, 1)
        live = np.frombuffer(f.read(S), dtype=np.uint8)
    maxc = G // 3
    slots = np.nonzero(live)[0]
    idx = slots[:, None] * G + np.arange(3 * maxc)[None, :]
    c = (fw[idx >> 2] >> ((idx & 3) * 2).astype(np.uint8)) & 3
    ops = table[c[:, 0::3] * 16 + c[:, 1::3] * 4 + c[:, 2::3]]
    genes = collections.Counter()
    for row in ops:
        stop = np.nonzero(row == R.STOP)[0]
        genes[tuple(int(x) for x in row[:stop[0] if len(stop) else maxc])] += 1
    tot = collections.Counter()
    used_genes = collections.Counter()
    for g, cnt in genes.items():
        has_l = R.LISTEN in g[1:]
        has_s = R.SIGNAL in g[1:]
        listens, signals = analyse(list(g))
        tot["organisms"] += cnt
        tot["carry LISTEN codon"] += cnt * has_l
        tot["carry SIGNAL codon"] += cnt * has_s
        tot["execute LISTEN"] += cnt * bool(listens)
        tot["execute SIGNAL"] += cnt * bool(signals)
        for o in set(listens):
            tot["LISTEN value -> " + o] += cnt
        for o in set(signals):
            tot["SIGNAL sends value of " + o] += cnt
        if any(o in ("OUT", "SIGNAL", "STA") for o in listens):
            used_genes[" ".join(R.OP_NAMES[x] for x in g)] += cnt
    print("checkpoint tick %d" % t_next)
    for k, v in sorted(tot.items(), key=lambda kv: -kv[1]):
        print("%10d  %s" % (v, k))
    print("most common genes whose LISTEN value is used:")
    for g, cnt in used_genes.most_common(8):
        print("%8d  %s" % (cnt, g))
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"tick": t_next, "counts": dict(tot), "used_listen_genes": used_genes.most_common(30)}, f, indent=1)


if __name__ == "__main__":
    main()
