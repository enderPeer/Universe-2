"""Tables for docs/11 from the sweep summaries in results/dna1_sweep/ (JSON written by gpu/dna_sweep.cu).

  python tools/dna_sweep_report.py results/dna1_sweep

Prints Markdown: per code and tick count one table over the gene lengths (strings swept, proper genes = strings whose
first STOP is the padding, claimers, copiers, dividers, child producers, faithful copiers, worlds with a grandchild,
seconds), the fecundity histograms where they are not all zero, and the verification and prefilter summaries.
"""
import glob
import json
import os
import re
import sys


def fmt(n):
    return "{:,}".format(n)


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "results/dna1_sweep"
    runs = {}
    for p in sorted(glob.glob(os.path.join(d, "L*_t*.json"))):
        name = os.path.basename(p)[:-5]
        if name.endswith("_nopf"):
            continue
        m = re.match(r"L(\d)_(standard|uniform)_t(\d+)(?:_(\w+))?$", name)
        if not m:
            continue
        L, code, ticks, extra = int(m.group(1)), m.group(2), int(m.group(3)), m.group(4) or ""
        runs[(code, ticks, extra, L)] = json.load(open(p))
    keys = sorted(set((c, t, e) for c, t, e, _ in runs))
    for code, ticks, extra in keys:
        print("\n### %s code, %d ticks%s\n" % (code, ticks, (" (" + extra + ")") if extra else ""))
        print("| gene length (codons incl. START) | strings swept | proper genes | claimed a slot | copied a cell | divided | expressible child | faithful child | intact | grandchild | seconds |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for L in sorted(l for c, t, e, l in runs if (c, t, e) == (code, ticks, extra)):
            r = runs[(code, ticks, extra, L)]
            tot = r["total"]
            bl = r["by_gene_length"]
            proper = bl.get(str(L), bl.get("%d+" % L, {})).get("strings", 0) if L < 7 else bl.get("7+", {}).get("strings", 0)
            mode = r["params"]["mode"]
            label = "%d%s" % (L, "" if mode == "all" else " (sample of %s)" % fmt(r["worlds"]))
            intact = fmt(tot["intact_any"]) if "intact_any" in tot else "n/a"
            print("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %.1f |" % (
                label, fmt(r["worlds"]), fmt(proper), fmt(tot["claimed"]), fmt(tot["copied_any"]), fmt(tot["divided"]),
                fmt(tot["children_any"]), fmt(tot["faithful_any"]), intact, fmt(tot["grandchildren_any"]), r["seconds"]))
        for L in sorted(l for c, t, e, l in runs if (c, t, e) == (code, ticks, extra)):
            r = runs[(code, ticks, extra, L)]
            tot = r["total"]
            if tot["faithful_any"]:
                fh = tot["fecundity_hist"]
                print("\nfecundity histogram (faithful children of the seed organism) for length %d: %s" % (
                    L, ", ".join("%d: %s" % (i if i < 16 else 16, fmt(fh[i])) for i in range(17) if fh[i] and i > 0) + " (16 = 16 or more)"))
                ch = tot["children_hist"]
                print("children histogram (all expressible children) for length %d: %s" % (
                    L, ", ".join("%d: %s" % (i if i < 16 else 16, fmt(ch[i])) for i in range(17) if ch[i] and i > 0)))
                dh = tot["maxdepth_hist"]
                print("max lineage depth for length %d: %s" % (L, ", ".join("%d: %s" % (i, fmt(dh[i])) for i in range(7) if dh[i])))
    # verification
    vs = sorted(glob.glob(os.path.join(d, "verify", "verify_*.json")))
    if vs:
        print("\n### Verification against sim/dna_ref.py\n")
        print("| set | code | ticks | world | genes | genes identical | tick hashes compared | mismatches | genes with claims / copies / divides / children |")
        print("|---|---|---:|---|---:|---:|---:|---:|---|")
        tg = th = tm = 0
        for p in vs:
            v = json.load(open(p))
            es = v["engine_summary"]
            world = "%dx%d %s" % (v["W"], v["G"], "soup" if v["soup"] else "blank") + (" T=%d" % v["T"] if v["T"] != 64 else "")
            print("| %s | %s | %d | %s | %d | %d | %s | %d | %d / %d / %d / %d |" % (
                os.path.basename(p)[9:-5], v["code"], v["ticks"], world, v["genes"], v["genes_identical"], fmt(v["hash_ticks_identical"]),
                len(v["mismatches"]), es["claims"], es["copied"], es["divides"], es["children"]))
            tg += v["genes"]; th += v["hash_ticks_identical"]; tm += len(v["mismatches"])
        print("| **total** | | | | **%d** | | **%s** | **%d** | |" % (tg, fmt(th), tm))
    pf = os.path.join(d, "prefilter_check.tsv")
    if os.path.exists(pf):
        rows = [l.rstrip("\n").split("\t") for l in open(pf) if not l.startswith("#") and "\t" in l]
        print("\nprefilter check: %d runs compared with and without the prefilter, %d identical" % (len(rows), sum(1 for r in rows if r[-1] == "identical")))


if __name__ == "__main__":
    main()
