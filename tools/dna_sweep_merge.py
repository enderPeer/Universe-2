"""Merge the summaries of a sweep that was split into chunk ranges (gpu/dna_sweep.cu --chunk-start/--chunk-end on
several GPUs): sums all histogram counters, checks that the ranges tile [0, range) without overlap and writes one
summary JSON of the same shape with "merged_from" listing the parts.

  python tools/dna_sweep_merge.py out.json part1.json part2.json ...
"""
import json
import sys


def add_into(dst, src):
    for k, v in src.items():
        if isinstance(v, dict):
            add_into(dst.setdefault(k, {}), v)
        elif isinstance(v, list):
            if k not in dst:
                dst[k] = list(v)
            else:
                dst[k] = [a + b for a, b in zip(dst[k], v)]
        elif isinstance(v, int):
            dst[k] = dst.get(k, 0) + v


def main():
    out_path, parts = sys.argv[1], sys.argv[2:]
    docs = [json.load(open(p)) for p in parts]
    p0 = docs[0]["params"]
    for d in docs[1:]:
        for k in ("mode", "L", "W", "G", "ticks", "T", "I", "seed", "code", "soup", "range", "rseed"):
            if d["params"][k] != p0[k]:
                sys.exit("parameter %s differs between parts" % k)
    ranges = sorted((d["params"]["chunk_start"], d["params"]["chunk_end"]) for d in docs)
    total = p0["range"] if p0["mode"] == "all" else None
    pos = 0
    for a, b in ranges:
        if a != pos:
            sys.exit("ranges do not tile: gap or overlap at %d (next part starts at %d)" % (pos, a))
        pos = b
    if total is not None and pos != total:
        sys.exit("ranges cover [0, %d) but the full range is %d" % (pos, total))
    merged = {"params": dict(p0), "worlds": 0, "seconds": 0.0, "chunks": 0, "logged": 0, "log_overflow": 0, "total": {}, "by_gene_length": {}}
    merged["params"]["chunk_start"] = 0
    merged["params"]["chunk_end"] = pos
    for d in docs:
        merged["worlds"] += d["worlds"]
        merged["seconds"] += d["seconds"]
        merged["chunks"] += d["chunks"]
        merged["logged"] += d["logged"]
        merged["log_overflow"] += d["log_overflow"]
        add_into(merged["total"], d["total"])
        add_into(merged["by_gene_length"], d["by_gene_length"])
    merged["worlds_per_second_summed_parts"] = merged["worlds"] / merged["seconds"] if merged["seconds"] else None
    merged["merged_from"] = [{"file": p, "chunk_start": d["params"]["chunk_start"], "chunk_end": d["params"]["chunk_end"], "seconds": d["seconds"],
                              "tpb": d["params"]["tpb"]} for p, d in zip(parts, docs)]
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=1)
    t = merged["total"]
    print("merged %d parts, %d worlds, faithful_any=%d intact_any=%s grandchildren_any=%d children_any=%d" % (
        len(parts), merged["worlds"], t["faithful_any"], t.get("intact_any"), t["grandchildren_any"], t["children_any"]))


if __name__ == "__main__":
    main()
