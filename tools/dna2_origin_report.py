"""Analyse the origin-statistics runs of tools/dna2_origin_queue.py (docs/16).

For every run (results/dna2_origin/<code>_sNNN.{txt.gz,top.gz,err}):
  first_faithful  first census with faithful_total > 0 (can be a one-shot copier that founds no lineage)
  takeover        first census with more than half of the 97,656 slots alive: the event that is counted
  persists        alive > 10,000 at the last census
  founder         the most common gene at the takeover census (the new lineage dominates by then)
  leader          the most common gene at the last census
Rate per code: takeovers / total exposure (ticks until takeover, or the full run if none), an exponential
waiting-time estimate with censoring; 95 % interval from the exact Poisson limits (chi-square, Wilson-Hilferty);
mean waiting time = 1 / rate, also in refilled slots (23 per tick).

  python tools/dna2_origin_report.py [--md]
"""
import collections
import glob
import gzip
import json
import math
import os
import sys

D = "results/dna2_origin"
SLOTS, F = 97656, 23
rows = []
for f in sorted(glob.glob(os.path.join(D, "*.txt.gz"))):
    tag = os.path.basename(f)[:-7]
    code, s = tag.rsplit("_s", 1)
    # strip NUL bytes: standard_s022's files were partly overwritten by a duplicate run (docs/16, caveats)
    cs = [json.loads(l.strip("\0")[7:]) for l in gzip.open(f, "rt") if l.strip("\0").startswith("census")]
    if not cs:
        continue
    top = collections.defaultdict(list)
    tp = os.path.join(D, tag + ".top.gz")
    if os.path.exists(tp):
        for l in gzip.open(tp, "rt"):
            l = l.strip("\0")   # standard_s022's files were partly overwritten by a duplicate run (docs/16)
            if not l or not l[0].isdigit():
                continue
            r = l.rstrip("\n").split("\t")
            if len(r) < 7:
                continue
            top[int(r[0])].append(r)
    last = cs[-1]
    first = next((c for c in cs if c["faithful_total"] > 0), None)
    take = next((c for c in cs if c["live"] > SLOTS // 2), None)
    origin = take                      # the counted event
    founder = leader = None
    if take and top.get(take["tick"]):
        founder = top[take["tick"]][0]
    if top.get(last["tick"]):
        leader = top[last["tick"]][0]
    rows.append({"code": code, "seed": int(s), "ticks": last["tick"] + 1,
                 "first_faithful": first["tick"] if first else None,
                 "origin": origin["tick"] if origin else None, "takeover": take["tick"] if take else None,
                 "live_end": last["live"], "persists": last["live"] > 10000,
                 "births": last["births_total"], "faithful": last["faithful_total"],
                 "max_depth": last.get("max_depth"), "distinct_end": last["distinct_genes"],
                 "founder_codons": founder[5] if founder else None, "founder_ops": founder[6] if founder else None,
                 "leader_codons": leader[5] if leader else None, "leader_ops": leader[6] if leader else None,
                 "leader_count": int(leader[2]) if leader else None})

def chi2q(nu, z):
    """Wilson-Hilferty approximation of the chi-square quantile with nu degrees of freedom at normal quantile z."""
    return nu * (1 - 2 / (9 * nu) + z * math.sqrt(2 / (9 * nu))) ** 3


def wait_ci(k, expo):
    """95 % interval of the mean waiting time expo / k from the exact Poisson limits of k events."""
    if expo == 0:
        return None
    hi_rate = chi2q(2 * k + 2, 1.959964) / (2 * expo)
    lo_rate = chi2q(2 * k, -1.959964) / (2 * expo) if k > 0 else 0.0
    return (round(1 / hi_rate), round(1 / lo_rate) if lo_rate > 0 else None)


summary = {}
for code in sorted({r["code"] for r in rows}):
    rr = [r for r in rows if r["code"] == code]
    orig = [r for r in rr if r["origin"] is not None]
    expo = sum(r["origin"] if r["origin"] is not None else r["ticks"] for r in rr)
    rate = len(orig) / expo if expo else 0.0
    ts = sorted(r["origin"] for r in orig)
    summary[code] = {
        "runs": len(rr), "with_origin": len(orig), "persisting": sum(r["persists"] for r in rr),
        "exposure_ticks": expo, "rate_per_tick": rate,
        "mean_wait_ticks": (1 / rate) if rate else None,
        "mean_wait_ci95": wait_ci(len(orig), expo),
        "mean_wait_refilled_slots": (F / rate) if rate else None,
        "origin_ticks_sorted": ts,
        "median_origin_tick": ts[len(ts) // 2] if ts else None,
        "takeover_minus_first_faithful": sorted(r["takeover"] - r["first_faithful"] for r in orig if r["first_faithful"] is not None),
        "first_faithful_without_takeover": sum(1 for r in rr if r["first_faithful"] is not None and r["takeover"] is None),
        "founder_lengths": collections.Counter(len(r["founder_codons"].split()) for r in orig if r["founder_codons"]).most_common(),
        "founder_ops": collections.Counter(r["founder_ops"] for r in orig if r["founder_ops"]).most_common(10),
        "leader_ops_end": collections.Counter(r["leader_ops"] for r in orig if r["leader_ops"]).most_common(10),
    }

json.dump({"runs": rows, "summary": summary}, open(os.path.join(D, "report.json"), "w"), indent=1)
for code, s in summary.items():
    print(f"== {code}: {s['with_origin']} of {s['runs']} runs were taken over by a lineage, {s['persisting']} persist; "
          f"exposure {s['exposure_ticks']:,} ticks; mean wait "
          f"{format(round(s['mean_wait_ticks']), ',') if s['mean_wait_ticks'] else 'more than the exposure'} ticks "
          f"(95 % interval {s['mean_wait_ci95']}), = "
          f"{format(round(s['mean_wait_refilled_slots']), ',') if s['mean_wait_refilled_slots'] else '-'} refilled slots")
    print("   origin ticks:", s["origin_ticks_sorted"])
    print("   takeover - first faithful birth:", s["takeover_minus_first_faithful"])
    print("   runs with a faithful birth but no takeover:", s["first_faithful_without_takeover"])
    print("   founder lengths:", s["founder_lengths"])
    for ops, n in s["founder_ops"]:
        print(f"   founder x{n}: {ops}")
    for ops, n in s["leader_ops_end"][:5]:
        print(f"   leader at end x{n}: {ops}")
if "--md" in sys.argv:
    print("\n| code | seed | origin tick | takeover tick | alive at end | max depth | founder | leader at end |")
    print("|---|---:|---:|---:|---:|---:|---|---|")
    for r in rows:
        print(f"| {r['code']} | {r['seed']} | {r['origin'] if r['origin'] is not None else '-'} | {r['takeover'] if r['takeover'] is not None else '-'} | "
              f"{r['live_end']:,} | {r['max_depth']} | {r['founder_codons'] or '-'} | {r['leader_ops'] or '-'} |")
