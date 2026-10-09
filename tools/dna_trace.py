"""Tick-by-tick trace of one gene alone in a small DNA-1 world, using the reference simulator unchanged.

  python tools/dna_trace.py "ATG GCT CCT ACT CTT GTT GGT TAA" [--W 8 --G 256 --ticks 64 --T 64 --I 16 --seed 1
                                                          --uniform-code --soup --slots 0,3]

Runs sim/dna_ref.py's World (W slots of G cells, mu = 0, D = F = 0, blank tape unless --soup, the gene seeded in
slot 0) one tick at a time and prints, after every tick, every live processor (slot, depth, age, pc and the codon it
points at, H and the cell under H, P, Q, child, copied) and the gene text of every slot that is live, claimed or
was ever written; births (with the faithful flag), claims and deaths are listed as events. This is the view that
explains how a copier's loop runs and terminates; it is a diagnostic, the engine's result is what counts.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sim"))
import dna_ref  # noqa: E402


def gene_text(w, s, maxcod=12):
    base = s * w.G
    n = dna_ref.gene_len(w.fwd, w.is_stop, base, w.maxc)
    cods = []
    for k in range(min(n + 1, maxcod)):
        c = w.fwd[base + 3 * k] * 16 + w.fwd[base + 3 * k + 1] * 4 + w.fwd[base + 3 * k + 2]
        cods.append(dna_ref.codon_text(c) + "/" + dna_ref.OP_NAMES[w.table[c]])
    return "%d codons: %s%s" % (n, " ".join(cods), " ..." if n + 1 > maxcod else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("genome")
    ap.add_argument("--W", type=int, default=8)
    ap.add_argument("--G", type=int, default=256)
    ap.add_argument("--ticks", type=int, default=64)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--I", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--uniform-code", action="store_true")
    ap.add_argument("--soup", action="store_true")
    ap.add_argument("--quiet-static", action="store_true", help="print only ticks with an event or a change of child/copied")
    a = ap.parse_args()
    genome = dna_ref.parse_genome(a.genome)
    w = dna_ref.World(a.W * a.G, a.G, I=a.I, T=a.T, mu=0.0, D=0, F=0, double=True, uniform=a.uniform_code,
                      blank=not a.soup, seed=a.seed, genome=genome, n_seed=1)
    print("gene: %s = %s" % (" ".join(dna_ref.codon_text(c) for c in genome),
                             " ".join(dna_ref.OP_NAMES[w.table[c]] for c in genome)))
    print("world: W=%d G=%d N=%d T=%d I=%d seed=%d code=%s %s" % (a.W, a.G, w.N, a.T, a.I, a.seed,
          "uniform" if a.uniform_code else "standard", "soup" if a.soup else "blank"))
    written = set([0])
    prev_births = 0
    prev_claims = 0
    prev_deaths = 0
    prev_state = None
    for t in range(a.ticks):
        w.tick()
        events = []
        if w.claims > prev_claims:
            for s in sorted(w.live):
                if w.child[s] >= 0 and w.copied[s] == 0:
                    events.append("slot %d claims slot %d" % (s, w.child[s]))
            prev_claims = w.claims
        for (tt, p, ch, ph, chh, f) in w.birth_log[prev_births:]:
            events.append("BIRTH in slot %d from slot %d, faithful=%d (parent %016x child %016x)" % (ch, p, f, ph, chh))
        prev_births = len(w.birth_log)
        if w.deaths > prev_deaths:
            events.append("%d death(s)" % (w.deaths - prev_deaths))
            prev_deaths = w.deaths
        for s in w.live:
            if w.child[s] >= 0:
                written.add(w.child[s])
        state = tuple((s, w.child[s], w.copied[s]) for s in sorted(w.live))
        if a.quiet_static and not events and state == prev_state:
            continue
        prev_state = state
        print("tick %d: live=%s births=%d faithful=%d deaths=%d%s" % (t, sorted(w.live), w.births, w.faithful, w.deaths,
              ("  EVENTS: " + "; ".join(events)) if events else ""))
        for s in sorted(w.live):
            base = s * w.G
            pc = w.pc[s]
            cb = base + 3 * pc
            cod = w.fwd[cb] * 16 + w.fwd[cb + 1] * 4 + w.fwd[cb + 2]
            print("   proc slot %d depth %d age %d: pc=%d (%s) H=%d [cell %d] P=%d Q=%d child=%d copied=%d A=%d"
                  % (s, w.depth[s], w.age[s], pc, dna_ref.OP_NAMES[w.xtable[cod]], w.H[s], w.fwd[w.H[s]], w.P[s], w.Q[s],
                     w.child[s], w.copied[s], w.A[s]))
        for s in sorted(written | set(w.live)):
            print("   slot %d: %s" % (s, gene_text(w, s)))
    print("done: births=%d faithful=%d deaths=%d claims=%d copied_cells=%d live=%d hash=%016x"
          % (w.births, w.faithful, w.deaths, w.claims, w.copied_cells, len(w.live), w.state_hash()))


if __name__ == "__main__":
    main()
