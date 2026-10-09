"""Summarise a seeded DNA-1 world run of gpu/dna_world.cu (docs/12): census series, extended birth log, top genes.

  python tools/dna_seeded_report.py RUN [--out RUN.summary.json] [--md]

Reads RUN.err (configuration and summary lines of the engine), RUN.txt or RUN.txt.gz (the "census {json}" lines),
RUN.log or RUN.log.gz (the --log-births file: every DIVIDE birth with the child's depth, the parent's age and the
copied cells) and RUN.top or RUN.top.gz (the --top-genes file: codons of the most common genes at every census),
whichever exist, and writes one JSON summary: the parameters, the final counters, the live population and its
depth histogram at a fixed list of ticks (the decay or growth curve as numbers), the births per lineage depth
(generation) with their faithful fraction, first and last tick, mean copied cells and the parent's age at the
division, the fecundity distribution (faithful children per parent life), and the most common child genes with
their codons where the top-genes file names them. Standard library only. Nothing is estimated: every number is a
count over the files.
"""
import argparse
import gzip
import json
import os
import re
import statistics
import sys

TICKS = [9, 19, 49, 59, 69, 99, 119, 149, 199, 299, 499, 999, 1999, 2999, 4999, 9999, 19999, 49999, 99999, 199999,
         399999, 599999, 799999, 999999]


def open_any(path):
    for p in (path, path + ".gz"):
        if os.path.exists(p):
            return gzip.open(p, "rt") if p.endswith(".gz") else open(p)
    return None


def read_err(run):
    f = open_any(run + ".err")
    if f is None:
        return {}, {}
    params, summary = {}, {}
    for line in f:
        line = line.strip()
        if line.startswith("dna_world "):
            for kv in line.split()[1:]:
                k, v = kv.split("=", 1)
                params[k] = v
        elif line.startswith("summary "):
            for kv in line.split()[1:]:
                k, v = kv.split("=", 1)
                summary[k] = v
        elif line.startswith("seeded "):
            m = re.search(r"seeded (\d+) slots with (\d+) codons, gene hash ([0-9a-f]+)", line)
            if m:
                params["seed_hash"] = m.group(3)
                params["seed_codons"] = int(m.group(2))
    f.close()
    return params, summary


def read_census(run):
    f = open_any(run + ".txt")
    if f is None:
        return []
    rows = []
    for line in f:
        if line.startswith("census "):
            rows.append(json.loads(line[7:]))
    f.close()
    return rows


def read_log(run):
    f = open_any(run + ".log")
    if f is None:
        return []
    rows = []
    for line in f:
        if line.startswith("#"):
            continue
        p = line.rstrip("\n").split("\t")
        rows.append((int(p[0]), int(p[1]), int(p[2]), p[3], p[4], int(p[5]), int(p[6]), int(p[7]), int(p[8])))
    f.close()
    return rows


def read_top(run):
    f = open_any(run + ".top")
    if f is None:
        return {}, []
    names = {}
    rows = []
    for line in f:
        if line.startswith("#"):
            continue
        p = line.rstrip("\n").split("\t")
        rows.append((int(p[0]), p[1], int(p[2]), int(p[3]), int(p[4]), p[5], p[6]))
        names[p[1]] = (p[5], p[6])
    f.close()
    return names, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="path prefix: RUN.err, RUN.txt[.gz], RUN.log[.gz], RUN.top[.gz]")
    ap.add_argument("--out", default=None, help="summary JSON (default RUN.summary.json)")
    ap.add_argument("--md", action="store_true", help="also print markdown tables")
    ap.add_argument("--max-genes", type=int, default=12)
    a = ap.parse_args()
    run = a.run
    params, summary = read_err(run)
    census = read_census(run)
    log = read_log(run)
    names, top = read_top(run)
    out = {"run": os.path.basename(run), "params": params, "summary": summary}

    # the census series at the fixed ticks (the last census line of each tick <= T, and the final one)
    series = []
    by_tick = {c["tick"]: c for c in census}
    want = sorted(set(t for t in TICKS if census and t <= census[-1]["tick"]) | ({census[-1]["tick"]} if census else set()))
    for t in want:
        c = by_tick.get(t)
        if c is None:                            # the nearest census at or before t
            cands = [x for x in census if x["tick"] <= t]
            if not cands:
                continue
            c = cands[-1]
        series.append({"tick": c["tick"], "live": c["live"], "births_total": c["births_total"],
                       "divide_total": c["births_total"] - c["spontaneous_total"], "faithful_total": c["faithful_total"],
                       "spontaneous_total": c["spontaneous_total"], "deaths_total": c["deaths_total"],
                       "distinct_genes": c["distinct_genes"], "depth_hist": c["depth_hist"],
                       "max_depth": c.get("max_depth"), "top": c["top_genes"][:3]})
    out["series"] = series
    if census:
        peak = max(census, key=lambda c: c["live"])
        out["peak_live"] = {"tick": peak["tick"], "live": peak["live"]}
        md = [c.get("max_depth") for c in census if c.get("max_depth") is not None]
        out["max_depth_ever"] = max(md) if md else None
        # last tick at which a processor of depth >= 1 was alive (the lineage's end)
        last_lineage = None
        for c in census:
            if any(d >= 1 and n > 0 for d, n in c["depth_hist"]):
                last_lineage = c["tick"]
        out["last_tick_with_depth_ge_1"] = last_lineage
        seed_hash = params.get("seed_hash")
        if seed_hash:
            last_seed_gene = None
            carriers_peak = 0
            for c in census:
                n = dict((h, k) for h, k in c["top_genes"]).get(seed_hash, 0)
                if n:
                    last_seed_gene = c["tick"]
                    carriers_peak = max(carriers_peak, n)
            out["seed_gene"] = {"hash": seed_hash, "last_census_tick_among_top10": last_seed_gene,
                                "peak_count_in_top10": carriers_peak}

    # births per generation from the log; "seed lineage" = rows whose parent or child is the seed gene (random inflow
    # organisms that copy themselves faithfully are counted in "generations" too, but not in "generations_seed")
    gens = {}
    gens_seed = {}
    fec = {}                                     # (parent slot, parent birth tick) -> faithful children
    child_hash = {}
    child_hash_unf = {}
    seed_hash = params.get("seed_hash")
    for (t, p, c, ph, ch, f, d, age, cop) in log:
        if p < 0:
            continue
        for gd in ((gens,) if seed_hash is None or (ph != seed_hash and ch != seed_hash) else (gens, gens_seed)):
            g = gd.setdefault(d, {"births": 0, "faithful": 0, "first_tick": t, "last_tick": t, "copied": [], "parent_age": []})
            g["births"] += 1
            g["faithful"] += f
            g["last_tick"] = t
            g["copied"].append(cop)
            g["parent_age"].append(age)
        key = (p, t - age - 1)
        fec[key] = fec.get(key, 0) + f
        child_hash[ch] = child_hash.get(ch, 0) + 1
        if not f:
            child_hash_unf[ch] = child_hash_unf.get(ch, 0) + 1
    gen_rows = []
    for d in sorted(gens):
        g = gens[d]
        gen_rows.append({"depth": d, "births": g["births"], "faithful": g["faithful"],
                         "faithful_frac": round(g["faithful"] / g["births"], 4),
                         "first_tick": g["first_tick"], "last_tick": g["last_tick"],
                         "copied_mean": round(statistics.fmean(g["copied"]), 2), "copied_min": min(g["copied"]),
                         "copied_max": max(g["copied"]),
                         "parent_age_median": statistics.median(g["parent_age"]), "parent_age_min": min(g["parent_age"]),
                         "parent_age_max": max(g["parent_age"])})
    out["generations"] = gen_rows
    seed_rows = []
    for d in sorted(gens_seed):
        g = gens_seed[d]
        seed_rows.append({"depth": d, "births": g["births"], "faithful": g["faithful"], "first_tick": g["first_tick"],
                          "last_tick": g["last_tick"], "copied_mean": round(statistics.fmean(g["copied"]), 2),
                          "parent_age_median": statistics.median(g["parent_age"])})
    out["generations_seed"] = seed_rows
    out["faithful_births_outside_seed_lineage"] = sum(g["faithful"] for g in gen_rows) - sum(g["faithful"] for g in seed_rows)
    ratios = []
    for i in range(1, len(gen_rows)):
        prev = gen_rows[i - 1]["faithful"]
        ratios.append({"from_depth": gen_rows[i - 1]["depth"], "ratio": round(gen_rows[i]["faithful"] / prev, 4) if prev else None})
    out["faithful_ratio_per_generation"] = ratios
    fdist = {}
    for v in fec.values():
        fdist[v] = fdist.get(v, 0) + 1
    out["fecundity_hist_of_dividing_parents"] = [[k, fdist[k]] for k in sorted(fdist)]
    out["log_rows"] = len(log)
    out["log_divide_rows"] = sum(1 for r in log if r[1] >= 0)
    top_children = sorted(child_hash.items(), key=lambda kv: (-kv[1], kv[0]))[:a.max_genes]
    out["most_common_child_genes"] = [{"hash": h, "births": n, "faithful_to_parent": h == params.get("seed_hash"),
                                       "codons": names.get(h, (None, None))[0], "ops": names.get(h, (None, None))[1]}
                                      for h, n in top_children]
    top_unf = sorted(child_hash_unf.items(), key=lambda kv: (-kv[1], kv[0]))[:a.max_genes]
    out["most_common_unfaithful_child_genes"] = [{"hash": h, "births": n, "codons": names.get(h, (None, None))[0],
                                                  "ops": names.get(h, (None, None))[1]} for h, n in top_unf]
    # the top genes at the last census and the genes that ever reached rank 1
    if top:
        last_t = top[-1][0]
        out["top_genes_last_census"] = [{"hash": r[1], "count": r[2], "depth": r[4], "codons": r[5], "ops": r[6]}
                                        for r in top if r[0] == last_t]
        rank1 = {}
        for r in top:
            if r[1] not in rank1 and all(x[2] <= r[2] for x in top if x[0] == r[0]):
                rank1[r[1]] = (r[0], r[2], r[5], r[6])
        out["genes_ever_rank1"] = [{"hash": h, "first_tick": v[0], "count_then": v[1], "codons": v[2], "ops": v[3]}
                                   for h, v in sorted(rank1.items(), key=lambda kv: kv[1][0])][:a.max_genes]
    path = a.out or run + ".summary.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    if a.md:
        print("| tick | live | DIVIDE births so far | faithful so far | depth histogram |")
        print("|---:|---:|---:|---:|---|")
        for s in series:
            print("| %s | %s | %s | %s | %s |" % (s["tick"], s["live"], s["divide_total"], s["faithful_total"],
                                              "{" + ", ".join("%d: %d" % (d, n) for d, n in s["depth_hist"]) + "}"))
        print()
        print("| depth | births | faithful | first tick | last tick | copied (mean, min..max) | parent age at division (median, min..max) |")
        print("|---:|---:|---:|---:|---:|---|---|")
        for g in gen_rows:
            print("| %d | %d | %d | %d | %d | %.1f, %d..%d | %s, %d..%d |" % (g["depth"], g["births"], g["faithful"], g["first_tick"],
                  g["last_tick"], g["copied_mean"], g["copied_min"], g["copied_max"], g["parent_age_median"], g["parent_age_min"], g["parent_age_max"]))
    print("wrote", path, "census lines", len(census), "log rows", len(log), "generations", len(gen_rows),
          "faithful by depth", [(g["depth"], g["faithful"]) for g in gen_rows][:12], file=sys.stderr)


if __name__ == "__main__":
    main()
