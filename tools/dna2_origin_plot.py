"""Picture of one origin event (docs/16): population over the whole run, and the genotype shares around the origin.

  python tools/dna2_origin_plot.py results/dna2_origin/standard_s001 --out results/dna2_origin/standard_s001.png

Panel 1: organisms alive and cumulative faithful births per census (log scale), the whole run.
Panel 2: a Muller-style stack of the most common genes (top 10 per census, by operation sequence) from 1,000 ticks
before the takeover to 3,000 after; the rest of the living population is grey. Data: the run's .txt.gz census and
.top.gz top-gene files.
"""
import argparse
import collections
import gzip
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--out", required=True)
ap.add_argument("--before", type=int, default=1000)
ap.add_argument("--after", type=int, default=3000)
a = ap.parse_args()

cs = [json.loads(l[7:]) for l in gzip.open(a.run + ".txt.gz", "rt") if l.startswith("census")]
top = collections.defaultdict(list)
for l in gzip.open(a.run + ".top.gz", "rt"):
    if l.startswith("#"):
        continue
    r = l.rstrip("\n").split("\t")
    top[int(r[0])].append(r)
SLOTS = 97656
take = next((c["tick"] for c in cs if c["live"] > SLOTS // 2), None)
first = next((c["tick"] for c in cs if c["faithful_total"] > 0), None)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), gridspec_kw={"height_ratios": [1, 1.4]})
t = [c["tick"] for c in cs]
ax1.semilogy(t, [max(1, c["live"]) for c in cs], color="#2a6fdb", lw=1.2, label="organisms alive")
ax1.semilogy(t, [max(1, c["faithful_total"]) for c in cs], color="#d1495b", lw=1.2, label="faithful births so far")
if first is not None:
    ax1.axvline(first, color="#999", ls=":", lw=1)
    ax1.text(first, 1.5, " first faithful copy", color="#666", fontsize=8, rotation=90, va="bottom")
if take is not None:
    ax1.axvline(take, color="#333", ls="--", lw=1)
    ax1.text(take, 1.5, " takeover", color="#333", fontsize=8, rotation=90, va="bottom")
ax1.set_xlabel("tick")
ax1.set_ylabel("count (log)")
ax1.set_title(f"{a.run.split('/')[-1]}: unseeded world, 99,999,744 cells, 97,656 slots")
ax1.legend(loc="center right", fontsize=8)
ax1.grid(alpha=0.25)

if take is not None:
    lo, hi = take - a.before, take + a.after
    window = [c for c in cs if lo <= c["tick"] <= hi]
    totals = collections.Counter()
    for c in window:
        for r in top.get(c["tick"], []):
            totals[r[6]] += int(r[2])
    seqs = [s for s, _ in totals.most_common(8)]
    colors = plt.cm.tab10.colors
    xs = [c["tick"] for c in window]
    layers = []
    for s in seqs:
        layers.append([sum(int(r[2]) for r in top.get(c["tick"], []) if r[6] == s) for c in window])
    rest = [max(0, c["live"] - sum(L[i] for L in layers)) for i, c in enumerate(window)]
    ax2.stackplot(xs, *layers, rest, colors=list(colors[:len(seqs)]) + ["#dddddd"],
                  labels=[s if len(s) < 70 else s[:67] + "..." for s in seqs] + ["all other genes"])
    ax2.axvline(take, color="#333", ls="--", lw=1)
    ax2.set_xlim(lo, hi)
    ax2.set_xlabel("tick")
    ax2.set_ylabel("organisms alive")
    ax2.set_title("genotypes around the takeover (operation sequences of the 10 most common genes per census)")
    ax2.legend(loc="upper left", fontsize=7, framealpha=0.9)
    ax2.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(a.out, dpi=130)
print("written", a.out, "first faithful", first, "takeover", take)
