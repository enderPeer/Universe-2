"""Verify gpu/dna_sweep.cu against sim/dna_ref.py on a sample of single-gene worlds.

For every gene of the sample the reference runs the world the sweep engine runs (W slots of G cells, blank tape or
the section-6 soup, the gene seeded in slot 0, mu = 0, D = F = 0) with --hash-every 1, and the engine runs the same
genes in --genes mode with --hash-ticks and --records. The comparison is exact: the state hash of docs/09 section 8
after every tick, and the done-line counters of the reference (births, divide_births, faithful, deaths, live, claims,
copied_cells) against the engine's record (births_world, faithful_world, deaths, live_end, claims_world, copied_world).

  python3 dna_sweep_verify.py --ref dna_ref.py --sweep ./dna_sweep --workdir v1 --n-random 200 --rseed 11 \
      --from-log L5_log.tsv:60 --ticks 64 [--uniform-code] [--soup] [--W 8 --G 256 --T 64 --I 16 --seed 1] --jobs 24

The sample: --n-random genes with a uniform length 2..7 (START + uniform codons), plus up to k rows drawn uniformly
from each --from-log TSV (the engine's log of interesting genes), plus the hand-written organism of docs/09 section 11.
Writes genes.txt, recs.tsv, hashes.tsv, ref_*.txt and verify.json (counts and every mismatch) into --workdir.
"""
import argparse
import json
import os
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

LETTERS = "ACGT"


def codon_text(c):
    return LETTERS[c >> 4] + LETTERS[(c >> 2) & 3] + LETTERS[c & 3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--n-random", type=int, default=200)
    ap.add_argument("--rseed", type=int, default=1)
    ap.add_argument("--from-log", action="append", default=[], help="FILE:k rows drawn from the engine's log TSV")
    ap.add_argument("--genes-extra", default=None, help="file with extra codon strings, one per line")
    ap.add_argument("--W", type=int, default=8)
    ap.add_argument("--G", type=int, default=256)
    ap.add_argument("--ticks", type=int, default=64)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--I", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--uniform-code", action="store_true")
    ap.add_argument("--soup", action="store_true")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(a.workdir, exist_ok=True)
    rng = random.Random(a.rseed)
    start = "AAA" if a.uniform_code else "ATG"
    genes = ["ATG GCT CCT ACT CTT GTT GGT TAA"]
    for _ in range(a.n_random):
        L = rng.randint(2, 7)
        genes.append(" ".join([start] + [codon_text(rng.randrange(64)) for _ in range(L - 1)]))
    for spec in a.from_log:
        path, k = spec.rsplit(":", 1)
        rows = []
        with open(path) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                cols = line.rstrip("\n").split("\t")
                rows.append(cols[4])
        if rows:
            genes.extend(rng.sample(rows, min(int(k), len(rows))))
    if a.genes_extra:
        with open(a.genes_extra) as f:
            genes.extend(l.strip() for l in f if l.strip() and not l.startswith("#"))
    genes_path = os.path.join(a.workdir, "genes.txt")
    with open(genes_path, "w") as f:
        f.write("\n".join(genes) + "\n")
    print("sample: %d genes" % len(genes), file=sys.stderr)

    # the engine
    recs_path = os.path.join(a.workdir, "recs.tsv")
    hashes_path = os.path.join(a.workdir, "hashes.tsv")
    cmd = [a.sweep, "--genes", genes_path, "--records", recs_path, "--hash-ticks", hashes_path, "--W", str(a.W), "--G", str(a.G),
           "--ticks", str(a.ticks), "--T", str(a.T), "--I", str(a.I), "--seed", str(a.seed), "--gpu", str(a.gpu)]
    if a.uniform_code:
        cmd.append("--uniform-code")
    if a.soup:
        cmd.append("--soup")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        sys.exit("engine failed")
    recs = {}
    with open(recs_path) as f:
        header = None
        for i, line in enumerate(f):
            cols = line.rstrip("\n").split("\t")
            if line.startswith("#"):
                header = cols
                header[0] = header[0][1:]
                continue
            recs[int(cols[3])] = dict(zip(header, cols))
    eng_hash = {}
    with open(hashes_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            g, t, h = line.split()
            eng_hash.setdefault(int(g), {})[int(t)] = h

    # the reference, one process per gene
    def run_ref(idx):
        g = genes[idx]
        cmd = [sys.executable, a.ref, "--N", str(a.W * a.G), "--G", str(a.G), "--ticks", str(a.ticks), "--seed", str(a.seed), "--I", str(a.I),
               "--T", str(a.T), "--mu", "0", "--D", "0", "--F", "0", "--n-seed", "1", "--genome", g, "--hash-every", "1", "--census-every", "0"]
        if not a.soup:
            cmd.append("--blank")
        if a.uniform_code:
            cmd.append("--uniform-code")
        p = subprocess.run(cmd, capture_output=True, text=True)
        hashes = {}
        for line in p.stdout.splitlines():
            if line.startswith("tick "):
                parts = line.split()
                hashes[int(parts[1])] = parts[3]
        done = {}
        for line in p.stderr.splitlines():
            if line.startswith("done "):
                for kv in line.split()[1:]:
                    k, v = kv.split("=")
                    done[k] = v
        return idx, hashes, done, p.returncode, p.stderr[-500:]

    results = {}
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        for idx, hashes, done, rc, err in ex.map(run_ref, range(len(genes))):
            results[idx] = (hashes, done, rc, err)

    # compare
    mismatches = []
    n_hash_ticks = 0
    n_genes_ok = 0
    count_pairs = [("births", "births_world"), ("divide_births", "births_world"), ("faithful", "faithful_world"), ("deaths", "deaths"),
                   ("live", "live_end"), ("claims", "claims_world"), ("copied_cells", "copied_world")]
    with open(os.path.join(a.workdir, "ref_done.tsv"), "w") as fd:
        fd.write("#idx\tgene\t" + "\t".join(k for k, _ in count_pairs) + "\n")
        for idx in range(len(genes)):
            hashes, done, rc, err = results[idx]
            fd.write("%d\t%s\t%s\n" % (idx, genes[idx], "\t".join(done.get(k, "?") for k, _ in count_pairs)))
            problems = []
            if rc != 0:
                problems.append("reference exit %d: %s" % (rc, err.strip()[-200:]))
            eh = eng_hash.get(idx, {})
            for t in range(a.ticks):
                if hashes.get(t) != eh.get(t):
                    problems.append("tick %d hash ref %s engine %s" % (t, hashes.get(t), eh.get(t)))
                    break
                n_hash_ticks += 1
            rec = recs.get(idx)
            if rec is None:
                problems.append("no engine record")
            else:
                for rk, ek in count_pairs:
                    if done.get(rk) != rec.get(ek):
                        problems.append("%s ref %s engine %s=%s" % (rk, done.get(rk), ek, rec.get(ek)))
            if problems:
                mismatches.append({"idx": idx, "gene": genes[idx], "problems": problems})
            else:
                n_genes_ok += 1
    summary = {
        "genes": len(genes), "ticks": a.ticks, "code": "uniform" if a.uniform_code else "standard", "soup": a.soup,
        "W": a.W, "G": a.G, "T": a.T, "I": a.I, "seed": a.seed,
        "genes_identical": n_genes_ok, "hash_ticks_identical": n_hash_ticks, "hash_ticks_expected": len(genes) * a.ticks,
        "mismatches": mismatches,
        "engine_summary": {k: sum(1 for r in recs.values() if int(r[k]) > 0) for k in ("claims", "copied", "divides", "children", "faithful", "grandchildren")},
    }
    with open(os.path.join(a.workdir, "verify.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "mismatches"}), file=sys.stderr)
    print("mismatches: %d" % len(mismatches), file=sys.stderr)
    for m in mismatches[:20]:
        print(m, file=sys.stderr)


if __name__ == "__main__":
    main()
