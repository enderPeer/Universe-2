"""DNA-3 (docs/17): run sim/dna_ref.py locally and gpu/dna_world.cu on a cluster node on the same configurations and
compare every stdout line (the state hash after every tick and the census lines) and the birth files byte for byte.

  python tools/dna3_verify.py --host knecht24 --bin '~/universe2/dna3/dna_world3b' --gpu 0 --out results/dna3/verify

The configurations are listed in CONFIGS below: DNA-1 and DNA-2 regressions (the new engine must still reproduce
them) and DNA-3 worlds that switch on days 1, 2 and 3 alone and together, with edge cases (G not a multiple of 4,
single strand, T = 0, eshift 0, uniform code, unseeded soup). Writes one JSON summary per configuration and
verify.json with all of them.
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import time

REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sim", "dna_ref.py")
REPL = "ATG GCT CTT TTA GGT"            # the DNA-2 5-codon replicator (docs/15)
PLANT = "ATG AGA GCT CTT TTA GGT"       # the same with HARVEST after START (docs/17 day 3)
SLEEPER = "ATG AGA AGT AAA AGC GCT CTT TTA GGT"   # START HARVEST SENSE <sun> WAIT ALLOC COPY TERM DIVIDE (day 4)

CONFIGS = [
    ("dna1_t1", "--N 16384 --G 256 --ticks 200 --seed 1"),
    ("dna2_seeded", "--dna2 --N 65536 --G 256 --ticks 300 --seed 3 --mu 0.001 --n-seed 4 --genome '%s'" % REPL),
    ("dna2_uniform", "--dna2 --uniform-code --N 65536 --G 256 --ticks 200 --seed 9 --n-seed 4 --genome 'AAA AGA AAG CCC AGC'"),
    ("day1_seeded", "--days 1 --N 65536 --G 256 --ticks 300 --seed 3 --n-seed 4 --genome '%s'" % REPL),
    ("day1_soup_edge", "--days 1 --N 65536 --G 256 --ticks 300 --seed 11 --eshift 0 --emax 200 --day-len 25 --light-len 10 "
                       "--T 0 --mu 0.01 --e-spont 50 --e-seed 150 --F 4 --n-seed 2 --genome '%s'" % REPL),
    ("day2_firm", "--days 2 --W 16 --N 65536 --G 256 --ticks 300 --seed 3 --n-seed 4 --genome '%s'" % REPL),
    ("day2_open_blank", "--days 2 --W 32 --firm 0 --N 65536 --G 256 --ticks 300 --seed 5 --mu 0.005 --blank --n-seed 8 "
                        "--genome '%s'" % REPL),
    ("day12_slots", "--days 1,2 --W 16 --N 65536 --G 256 --ticks 300 --seed 13 --n-seed 2 --seed-slots 20,200 "
                    "--genome '%s'" % REPL),
    ("dna3_plant", "--dna3 --W 32 --land-cell 8 --N 262144 --G 256 --ticks 400 --seed 3 --n-seed 16 --seed-terrain sea "
                   "--genome '%s'" % PLANT),
    ("dna3_nonplant_mu", "--dna3 --W 32 --land-cell 8 --N 262144 --G 256 --ticks 400 --seed 17 --mu 0.01 --n-seed 16 "
                         "--seed-terrain sea --genome '%s'" % REPL),
    ("dna3_uniform", "--dna3 --uniform-code --W 32 --land-cell 8 --N 262144 --G 256 --ticks 300 --seed 19 --n-seed 16 "
                     "--genome 'AAA CCG AGA AAG CCC AGC'"),
    ("dna3_g255_single", "--dna3 --W 16 --land-cell 8 --N 65280 --G 255 --ticks 300 --seed 23 --no-double --D 20 "
                         "--inflow-land --c-copy 3 --maint 2 --gp-land 4 --F 3 --n-seed 8 --genome '%s'" % PLANT),
    ("dna3_soup", "--dna3 --W 32 --land-cell 16 --N 262144 --G 256 --ticks 300 --seed 29 --F 8 --e-spont 300"),
    # docs/17 section 14: no fixed lifetime (T = 0), shading, day 4
    ("shade_T0_nonplant", "--dna3 --shade --T 0 --W 32 --land-cell 8 --N 262144 --G 256 --ticks 600 --seed 31 --mu 0.005 "
                          "--n-seed 16 --seed-terrain sea --genome '%s'" % REPL),
    ("day4_plant", "--days 1,2,3,4 --shade --T 0 --W 32 --land-cell 8 --N 262144 --G 256 --ticks 600 --seed 3 --n-seed 16 "
                   "--seed-terrain sea --genome '%s'" % PLANT),
    ("day4_sleeper_odd", "--days 1,2,3,4 --shade --T 0 --W 32 --land-cell 8 --N 262144 --G 256 --ticks 600 --seed 37 --mu 0.01 "
                         "--year-days 4 --moon-days 3 --season-amp 6 --lat-amp 12 --lmoon 2 --n-seed 16 --genome '%s'" % SLEEPER),
    ("day4_noshade_T64", "--days 1,2,3,4 --W 32 --land-cell 8 --N 262144 --G 256 --ticks 600 --seed 41 --n-seed 16 "
                         "--genome '%s'" % SLEEPER),
    ("day4_uniform", "--days 1,2,3,4 --uniform-code --shade --T 0 --W 32 --land-cell 8 --N 262144 --G 256 --ticks 400 --seed 43 "
                     "--n-seed 16 --genome 'AAA CCG CCT AAA CGA AGA AAG CCC AGC'"),
    ("day4_g255_single_soup", "--days 1,2,3,4 --shade --T 0 --W 16 --land-cell 8 --N 65280 --G 255 --ticks 600 --seed 47 "
                              "--no-double --D 20 --F 6 --e-spont 300 --inflow-land --day-len 20 --light-len 12 --year-days 6"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="knecht24")
    ap.add_argument("--bin", default="~/universe2/dna3/dna_world3")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default="results/dna3/verify")
    ap.add_argument("--only", default=None, help="comma-separated configuration names")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    only = set(a.only.split(",")) if a.only else None
    summary = []
    for name, args in CONFIGS:
        if only and name not in only:
            continue
        common = args + " --hash-every 1 --census-every 10"
        t0 = time.time()
        ref = subprocess.run([sys.executable, REF] + shlex.split(common) + ["--births", os.path.join(a.out, name + ".ref.births")],
                             capture_output=True, text=True)
        t_ref = time.time() - t0
        if ref.returncode:
            print(name, "reference failed:", ref.stderr[-500:])
            summary.append({"name": name, "ok": False, "error": "reference"})
            continue
        rb = "/tmp/dna3v_%s.births" % name
        cmd = "%s %s --gpu %d --births %s && cat %s 1>&2" % (a.bin, common, a.gpu, rb, rb)
        t0 = time.time()
        eng = subprocess.run(["ssh", "-n", "-o", "BatchMode=yes", a.host, cmd], capture_output=True, text=True)
        t_eng = time.time() - t0
        if eng.returncode:
            print(name, "engine failed:", eng.stderr[-500:])
            summary.append({"name": name, "ok": False, "error": "engine"})
            continue
        # the engine's stderr holds its header lines, then the births file (cat), then nothing else
        err = eng.stderr.splitlines(True)
        bstart = next(i for i, l in enumerate(err) if l.startswith("#tick"))
        bend = next((i for i in range(bstart + 1, len(err)) if not err[i][:1].isdigit()), len(err))
        eng_births = "".join(err[bstart:bend])
        with open(os.path.join(a.out, name + ".ref.births")) as f:
            ref_births = f.read()
        rl, el = ref.stdout.splitlines(), eng.stdout.splitlines()
        hashes = sum(1 for l in rl if l.startswith("tick "))
        census = sum(1 for l in rl if l.startswith("census "))
        first_diff = next((i for i in range(max(len(rl), len(el))) if i >= len(rl) or i >= len(el) or rl[i] != el[i]), None)
        ok = first_diff is None and ref_births == eng_births
        last = json.loads(rl[-1][7:]) if rl and rl[-1].startswith("census ") else {}
        rec = {"name": name, "args": args, "ok": ok, "tick_hashes": hashes, "census_lines": census,
               "births_rows": ref_births.count("\n") - 1, "births_identical": ref_births == eng_births,
               "first_diff_line": first_diff, "ref_secs": round(t_ref, 1), "engine_secs": round(t_eng, 1),
               "final": {k: last.get(k) for k in ("tick", "live", "births_total", "faithful_total", "energy_total",
                                                   "zone_live", "live_land", "harvesters", "starved_total", "hash")}}
        if first_diff is not None:
            rec["ref_line"] = rl[first_diff] if first_diff < len(rl) else None
            rec["eng_line"] = el[first_diff] if first_diff < len(el) else None
        with open(os.path.join(a.out, name + ".ref.txt"), "w") as f:
            f.write(ref.stdout)
        with open(os.path.join(a.out, name + ".eng.txt"), "w") as f:
            f.write(eng.stdout)
        summary.append(rec)
        print("%-18s %s  %d tick hashes, %d census lines, %d births rows%s" % (
            name, "IDENTICAL" if ok else "DIFFERENT", hashes, census, rec["births_rows"],
            "" if ok else "  first diff at line %s" % first_diff), flush=True)
    with open(os.path.join(a.out, "verify.json"), "w") as f:
        json.dump(summary, f, indent=1)
    n_ok = sum(1 for r in summary if r.get("ok"))
    print("%d of %d configurations identical" % (n_ok, len(summary)))


if __name__ == "__main__":
    main()
