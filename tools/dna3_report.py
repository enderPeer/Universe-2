"""Summaries of DNA-3 world runs (docs/17 section 11) from the engine's stdout (census lines) and --top-genes file.

  python tools/dna3_report.py DIR/RUN1 DIR/RUN2 ... --out summary.json

For each run prefix (RUN.txt[.gz], RUN.top[.gz]): the final census without the gene list; the second-half range of
the live count; the faithful share of all births; the energy ledger as shares of the total income (passive light,
HARVEST, spontaneous) spent on instructions, copying, maintenance and lost with the dead; the first census with a
harvester and with an organism on land; a sampled series (about 50 points) of tick, live, births, faithful,
energy_total, harvesters, live_land, zone_live and starved_total; and the top genes (codons, operations, count) at
the last census. Prints one Markdown table row per run.
"""
import argparse
import gzip
import json
import os


def opener(prefix, ext):
    for p in (prefix + ext + ".gz", prefix + ext):
        if os.path.exists(p):
            return (gzip.open if p.endswith(".gz") else open)(p, "rt")
    return None


def summarise(prefix):
    f = opener(prefix, ".txt")
    cens = [json.loads(l[7:]) for l in f if l.startswith("census ")]
    f.close()
    last = cens[-1]
    half = [c["live"] for c in cens if c["tick"] >= last["tick"] / 2]
    out = {"run": os.path.basename(prefix), "censuses": len(cens),
           "final": {k: v for k, v in last.items() if k not in ("top_genes", "depth_hist")},
           "live_second_half": [min(half), max(half)],
           "faithful_share": last["faithful_total"] / last["births_total"] if last["births_total"] else None}
    if "ledger" in last:
        g = last["ledger"]
        income = g[0] + g[1] + g[2]
        out["ledger_named"] = dict(zip(["passive", "harvest", "spontaneous", "instructions", "copy", "maintenance", "dead",
                                        "lost_endowments", "wasted"], g))
        out["ledger_shares_of_income"] = {k: round(v / income, 4) for k, v in zip(
            ["passive", "harvest", "spontaneous", "instructions", "copy", "maintenance", "dead", "lost_endowments"], g)}
        out["ledger_closes"] = all(c["energy_init"] + c["ledger"][0] + c["ledger"][1] + c["ledger"][2] - sum(c["ledger"][3:8])
                                   == c["energy_total"] for c in cens)
    if "harvesters" in last:
        out["first_harvester_tick"] = next((c["tick"] for c in cens if c["harvesters"] > 0), None)
        out["first_land_tick"] = next((c["tick"] for c in cens if c["live_land"] > 0), None)
        out["harvesters_half_tick"] = next((c["tick"] for c in cens if c["live"] and c["harvesters"] * 2 > c["live"]), None)
    step = max(1, len(cens) // 50)
    keys = ("tick", "live", "births", "faithful", "spontaneous", "energy_total", "harvesters", "live_land", "zone_live",
            "starved_total", "max_depth", "distinct_genes")
    out["series"] = [{k: c[k] for k in keys if k in c} for c in cens[::step]]
    t = opener(prefix, ".top")
    if t:
        rows = [l.rstrip("\n").split("\t") for l in t if not l.startswith("#")]
        t.close()
        lt = rows[-1][0]
        out["top_genes_last"] = [{"count": int(r[2]), "codons": r[5], "ops": r[6]} for r in rows if r[0] == lt]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = [summarise(p) for p in a.runs]
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    print("| run | ticks | live (2nd half) | births | faithful | max depth | distinct genes | harvesters | on land |")
    print("|---|---:|---|---:|---:|---:|---:|---:|---:|")
    for r in res:
        fi = r["final"]
        print("| %s | %d | %d to %d | %d | %.2f %% | %s | %d | %s | %s |" % (
            r["run"], fi["tick"] + 1, r["live_second_half"][0], r["live_second_half"][1], fi["births_total"],
            100 * r["faithful_share"], fi.get("max_depth"), fi["distinct_genes"], fi.get("harvesters", "-"),
            fi.get("live_land", "-")))


if __name__ == "__main__":
    main()
