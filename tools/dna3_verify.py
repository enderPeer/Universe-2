"""DNA-3 (docs/17): run sim/dna_ref.py locally and gpu/dna_world.cu on a cluster node on the same configurations and
compare every stdout line (the state hash after every tick and the census lines) and the birth files byte for byte.

  python tools/dna3_verify.py --host knecht24 --bin '~/universe2/dna3/dna_world3b' --gpu 0 --out results/dna3/verify
  python tools/dna3_verify.py --local --bin ./dna_world3m --out results/dna3/verify      (the engine on this machine)

The configurations are listed in CONFIGS below: DNA-1 and DNA-2 regressions (the new engine must still reproduce
them) and DNA-3 worlds that switch on days 1, 2 and 3 alone and together, with edge cases (G not a multiple of 4,
single strand, T = 0, eshift 0, uniform code, unseeded soup), the ladder of section 18 and the macros of section 19.
CKPT_TESTS continue an engine checkpoint in both programs (a resume, and forks that add --ladder, --macros,
--in-fresh). Writes one JSON summary per configuration and verify.json with all of them. The macro tables the
configurations use are written into --out (and copied to the host); the placeholder {MACROS:name} in an argument
string names one of MACRO_TABLES.
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
EATER = "ATG AGA CCA AAA GCT CTT TTA GGT"        # START HARVEST EAT <N> ALLOC COPY TERM DIVIDE (day 5)
SWIMMER = "ATG AGA ACA AGA GCT CTT TTA GGT"      # START HARVEST MOVE <AGA: 1 slot east> ALLOC COPY TERM DIVIDE
FLIER = "ATG AGA ACA CTA GCT CTT TTA GGT"        # START HARVEST MOVE <CTA: 4 slots south> ALLOC COPY TERM DIVIDE
EXECER = "ATG AGA GTA AAG GCT CTT TTA GGT"       # START HARVEST EXEC <AAG: the east neighbour> ALLOC COPY TERM DIVIDE
TASKER1 = "ATG AGA TCA AAT TAT TCG GCT CTT TTA GGT"   # START HARVEST IN FLIP LDA OUT ALLOC COPY TERM DIVIDE: x^1 (day 6)
TASKERM1 = "ATG AGA TCA CAA CAA CAA TAT TCG GCT CTT TTA GGT"   # IN INC INC INC LDA OUT: x-1
SIGNALER = "ATG AGA TCA CGA CGG AAG TCG GCT CTT TTA GGT"  # START HARVEST IN SIGNAL LISTEN <E> OUT ALLOC COPY TERM DIVIDE
WALKER = "ATG AGA ACA AGA GCT CTT TTA GGT"       # the swimmer of day 5; on day 6 it walks on land too
LX1 = "ATG AGA TCA AAT TCG GCT CTT TTA GGT"      # ladder (section 18): IN FLIP OUT computes x^1 on the I/O cells
LSOC = "ATG AGA TCA CGA CGG AAA TCG GCT CTT TTA GGT"  # ladder: IN SIGNAL LISTEN <N> OUT (social task partner)
LCARRY = "ATG AGA TCA CAA TAT GTT CGT CAA TCG GCT CTT TTA GGT"  # ladder: IN INC LDA JNZB L1 INC OUT (hangs in LDA JNZB unless the carry is needed; exercises the loop)

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
    # docs/17 day 5
    ("day5_plant_mu", "--days 1,2,3,4,5 --shade --T 0 --W 32 --land-cell 8 --N 262144 --G 256 --ticks 1200 --seed 5 --mu 0.02 "
                      "--n-seed 16 --seed-terrain sea --genome '%s'" % PLANT),
    ("day5_eater", "--days 1,2,3,4,5 --shade --T 0 --W 32 --land-cell 8 --N 262144 --G 256 --ticks 600 --seed 53 --mu 0.005 "
                   "--n-seed 64 --genome '%s'" % EATER),
    ("day5_swimmer", "--days 1,2,3,4,5 --shade --T 0 --W 32 --land-cell 8 --N 262144 --G 256 --ticks 600 --seed 59 --n-seed 32 "
                     "--seed-terrain sea --genome '%s'" % SWIMMER),
    ("day5_flier_noshade", "--days 1,2,3,4,5 --W 32 --land-cell 8 --N 262144 --G 256 --ticks 600 --seed 61 --n-seed 32 --c-fly 3 "
                           "--genome '%s'" % FLIER),
    ("day5_execer_eat2", "--days 1,2,3,4,5 --shade --T 0 --W 32 --land-cell 8 --N 262144 --G 256 --ticks 600 --seed 67 --mu 0.01 "
                         "--eat-shift 2 --eat-eff 12 --c-move 1 --n-seed 64 --genome '%s'" % EXECER),
    ("day5_uniform_g255", "--days 1,2,3,4,5 --uniform-code --shade --T 0 --W 16 --land-cell 8 --N 65280 --G 255 --ticks 600 "
                          "--seed 71 --mu 0.02 --F 4 --n-seed 16 --genome 'AAA CCG CTT AAA AGA AAG CCC AGC'"),
    # docs/17 day 6
    ("day6_tasker_x1", "--days 1,2,3,4,5,6 --shade --T 0 --W 32 --land-cell 8 --N 262144 --G 256 --ticks 600 --seed 7 --mu 0.01 "
                       "--n-seed 16 --seed-terrain sea --genome '%s'" % TASKER1),
    ("day6_tasker_xm1", "--days 1,2,3,4,5,6 --shade --T 0 --W 32 --land-cell 8 --N 262144 --G 256 --ticks 600 --seed 73 "
                        "--task-bonus 3 --c-walk 2 --n-seed 16 --genome '%s'" % TASKERM1),
    ("day6_signaler", "--days 1,2,3,4,5,6 --shade --T 0 --W 32 --land-cell 8 --N 262144 --G 256 --ticks 600 --seed 79 --mu 0.005 "
                      "--n-seed 64 --genome '%s'" % SIGNALER),
    ("day6_walker_land", "--days 1,2,3,4,5,6 --shade --T 0 --W 32 --land-cell 8 --N 262144 --G 256 --ticks 600 --seed 83 "
                         "--n-seed 32 --seed-terrain land --genome '%s'" % WALKER),
    ("day6_uniform_g255", "--days 1,2,3,4,5,6 --uniform-code --shade --T 0 --W 16 --land-cell 8 --N 65280 --G 255 --ticks 600 "
                          "--seed 89 --mu 0.01 --F 4 --n-seed 16 --genome 'AAA CCG CTG ATT ATA CTT AGA AAG CCC AGC'"),
    ("day6_plant_mu", "--days 1,2,3,4,5,6 --shade --T 0 --W 32 --land-cell 8 --N 262144 --G 256 --ticks 1000 --seed 97 --mu 0.03 "
                      "--n-seed 16 --seed-terrain sea --genome '%s'" % PLANT),
    # docs/17 section 18: the long ladder
    ("ladder_x1_pools", "--days 1,2,3,4,5,6 --ladder --shade --T 0 --W 32 --land-cell 8 --N 262144 --G 256 --ticks 400 --seed 7 "
                        "--pool-cell 8 --pool-max 256 --pool-in 4 --n-seed 16 --seed-terrain sea --genome '%s'" % LX1),
    ("ladder_social", "--days 1,2,3,4,5,6 --ladder --shade --T 0 --W 32 --land-cell 8 --N 262144 --G 256 --ticks 400 --seed 101 "
                      "--streak-len 4 --n-seed 64 --genome '%s'" % LSOC),
    ("ladder_carry_nopool", "--days 1,2,3,4,5,6 --ladder --shade --T 0 --W 32 --land-cell 8 --N 262144 --G 256 --ticks 400 "
                            "--seed 103 --pool-cell 0 --streak-len 5 --eat-eff 12 --n-seed 16 --genome '%s'" % LCARRY),
    ("ladder_plant_mu", "--days 1,2,3,4,5,6 --ladder --shade --T 0 --W 32 --land-cell 8 --N 262144 --G 256 --ticks 600 --seed 107 "
                        "--mu 0.03 --pool-cell 16 --eat-eff 12 --n-seed 16 --seed-terrain sea --genome '%s'" % PLANT),
    ("ladder_uniform_g255", "--days 1,2,3,4,5,6 --ladder --uniform-code --shade --T 0 --W 16 --land-cell 8 --N 65280 --G 255 "
                            "--ticks 400 --seed 109 --mu 0.01 --F 4 --pool-cell 8 --pool-max 100 --pool-in 3 --n-seed 16 "
                            "--genome 'AAA CCG CTG ATT CTT AGA AAG CCC AGC'"),
]



# docs/17 section 19: macro tables for the verification (hand-written bodies: they test the mechanism, not the loop)
MACRO_TABLES = {
    "mtab_a": "M0 CTA FLIP INC INC L1 FLIP INC INC   # 15-x\nM1 GCA INC JNZB   # x&12\nM2 ATA L1 LDA R1 STA   # a\n"
              "M3 TGC NOP\nM4 GGA   # empty body: a NOP that costs its unit (arm C)\n",
    "mtab_c": "M0 CTA\nM1 GCA\nM2 ATA\nM3 TGC\nM4 GGA\n",
    "mtab_u": "M0 x FLIP\nM1 x INC JNZB L1 STA   # uniform code: codon values 32, 33 whatever the column says\n",
}
M15 = "ATG AGA GCT CTT TTA GGT TCA CTA TCG"              # START HARVEST ALLOC COPY TERM DIVIDE IN M0 OUT (15-x by macro)
M4X = "ATG AGA GCT CTT TTA GGT TCA GCA CGT CAT TCG"      # IN M1 L1 STA OUT: 4x with a looping body
CONFIGS += [
    # docs/17 section 19: macros and --in-fresh
    ("macro_15mx", "--days 1,2,3,4,5,6 --ladder --shade --T 0 --W 32 --land-cell 8 --N 262144 --G 256 --ticks 400 --seed 113 "
                   "--mu 0.005 --pool-cell 8 --macros {MACROS:mtab_a} --n-seed 16 --seed-terrain sea --genome '%s'" % M15),
    ("macro_loop_I2_infresh", "--days 1,2,3,4,5,6 --ladder --shade --T 0 --W 32 --land-cell 8 --N 262144 --G 256 --ticks 400 "
                              "--seed 127 --I 2 --eshift 0 --emax 300 --e-seed 300 --pool-cell 0 --in-fresh --macros {MACROS:mtab_a} "
                              "--n-seed 16 --genome '%s'" % M4X),
    ("macro_plant_mu", "--days 1,2,3,4,5,6 --ladder --shade --T 0 --W 32 --land-cell 8 --N 262144 --G 256 --ticks 600 --seed 131 "
                       "--mu 0.03 --pool-cell 16 --eat-eff 12 --in-fresh --macros {MACROS:mtab_a} --n-seed 16 --seed-terrain sea "
                       "--genome '%s'" % PLANT),
    ("macro_empty_day6", "--days 1,2,3,4,5,6 --shade --T 0 --W 32 --land-cell 8 --N 262144 --G 256 --ticks 400 --seed 137 --mu 0.01 "
                         "--in-fresh --macros {MACROS:mtab_c} --n-seed 16 --seed-terrain sea --genome '%s'" % TASKER1),
    ("macro_uniform_g255", "--days 1,2,3,4,5,6 --ladder --uniform-code --shade --T 0 --W 16 --land-cell 8 --N 65280 --G 255 "
                           "--ticks 400 --seed 139 --mu 0.01 --F 4 --pool-cell 8 --pool-max 100 --pool-in 3 --in-fresh "
                           "--macros {MACROS:mtab_u} --n-seed 16 --genome 'AAA CCG CTG GAA CTT AGA AAG CCC AGC'"),
]
# (name, arguments of the first run, its ticks, arguments of the continuation, its further ticks). The engine writes
# the checkpoint at the end of the first run; both programs then continue it with the second arguments.
CKPT_TESTS = [
    ("ckpt_fork_macros", "--days 1,2,3,4,5,6 --ladder --shade --T 0 --W 32 --land-cell 8 --N 262144 --G 256 --seed 149 --mu 0.01 "
                         "--pool-cell 8 --n-seed 16 --seed-terrain sea --genome '%s'" % LX1, 300,
                         "--days 1,2,3,4,5,6 --ladder --shade --T 0 --W 32 --land-cell 8 --N 262144 --G 256 --seed 149 --mu 0.01 "
                         "--pool-cell 8 --fork --in-fresh --macros {MACROS:mtab_a}", 300),
    ("ckpt_fork_day6_all", "--days 1,2,3,4,5,6 --shade --T 0 --W 32 --land-cell 8 --N 262144 --G 256 --seed 151 --mu 0.01 "
                           "--n-seed 16 --seed-terrain sea --genome '%s'" % TASKER1, 300,
                           "--days 1,2,3,4,5,6 --ladder --shade --T 0 --W 32 --land-cell 8 --N 262144 --G 256 --seed 151 --mu 0.01 "
                           "--pool-cell 8 --fork --in-fresh --macros {MACROS:mtab_a}", 300),
    ("ckpt_resume_v5", "--days 1,2,3,4,5,6 --ladder --shade --T 0 --W 32 --land-cell 8 --N 262144 --G 256 --seed 157 --mu 0.01 "
                       "--pool-cell 8 --in-fresh --macros {MACROS:mtab_a} --n-seed 16 --seed-terrain sea --genome '%s'" % M15, 300,
                       "--days 1,2,3,4,5,6 --ladder --shade --T 0 --W 32 --land-cell 8 --N 262144 --G 256 --seed 157 --mu 0.01 "
                       "--pool-cell 8 --in-fresh --macros {MACROS:mtab_a}", 300),
]


class Runner:
    """Runs the reference locally and the engine locally (--local) or on a host over ssh."""

    def __init__(self, a):
        self.a = a
        self.remote_dir = "/tmp"
        self.local_tables = {}
        for name, text in MACRO_TABLES.items():
            path = os.path.abspath(os.path.join(a.out, name + ".txt"))
            with open(path, "w") as f:
                f.write(text)
            self.local_tables[name] = path
        if not a.local:
            for name, path in self.local_tables.items():
                subprocess.run(["scp", "-q", "-o", "BatchMode=yes", path, "%s:%s/dna3v_%s.txt" % (a.host, self.remote_dir, name)], check=True)

    def args_for(self, args, remote):
        for name, path in self.local_tables.items():
            args = args.replace("{MACROS:%s}" % name, path if (self.a.local or not remote) else "%s/dna3v_%s.txt" % (self.remote_dir, name))
        return args

    def ref(self, args, extra):
        return subprocess.run([sys.executable, REF] + shlex.split(self.args_for(args, False)) + extra, capture_output=True, text=True)

    def engine(self, args, extra, cat=None):
        """Runs the engine; `cat` names a remote file whose content is appended to stderr (the births file)."""
        if self.a.local:
            r = subprocess.run([self.a.bin] + shlex.split(self.args_for(args, True)) + extra, capture_output=True, text=True)
            if cat and r.returncode == 0:
                with open(cat) as f:
                    r = subprocess.CompletedProcess(r.args, r.returncode, r.stdout, r.stderr + f.read())
            return r
        cmd = "%s %s %s" % (self.a.bin, self.args_for(args, True), " ".join(shlex.quote(x) for x in extra))
        if cat:
            cmd += " && cat %s 1>&2" % cat
        return subprocess.run(["ssh", "-n", "-o", "BatchMode=yes", self.a.host, cmd], capture_output=True, text=True)

    def fetch(self, remote_path, local_path):
        if self.a.local:
            if os.path.abspath(remote_path) != os.path.abspath(local_path):
                subprocess.run(["cp", remote_path, local_path], check=True)
            return
        subprocess.run(["scp", "-q", "-o", "BatchMode=yes", "%s:%s" % (self.a.host, remote_path), local_path], check=True)

    def scratch(self, name):
        return os.path.join(self.remote_dir, "dna3v_" + name)


def compare(runner, a, name, args, ref_extra=(), eng_extra=()):
    """Both programs on the same arguments; every stdout line and the births files must agree."""
    common = args + " --hash-every 1 --census-every 10"
    t0 = time.time()
    ref = runner.ref(common, list(ref_extra) + ["--births", os.path.join(a.out, name + ".ref.births")])
    t_ref = time.time() - t0
    if ref.returncode:
        print(name, "reference failed:", ref.stderr[-500:])
        return {"name": name, "ok": False, "error": "reference"}
    rb = runner.scratch(name + ".births")
    t0 = time.time()
    eng = runner.engine(common, list(eng_extra) + ["--births", rb], cat=rb)
    t_eng = time.time() - t0
    if eng.returncode:
        print(name, "engine failed:", eng.stderr[-500:])
        return {"name": name, "ok": False, "error": "engine"}
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
                                               "zone_live", "live_land", "harvesters", "starved_total", "tasks",
                                               "macro_users", "macro_exec", "hash")}}
    if first_diff is not None:
        rec["ref_line"] = rl[first_diff] if first_diff < len(rl) else None
        rec["eng_line"] = el[first_diff] if first_diff < len(el) else None
    with open(os.path.join(a.out, name + ".ref.txt"), "w") as f:
        f.write(ref.stdout)
    with open(os.path.join(a.out, name + ".eng.txt"), "w") as f:
        f.write(eng.stdout)
    print("%-22s %s  %d tick hashes, %d census lines, %d births rows%s" % (
        name, "IDENTICAL" if ok else "DIFFERENT", hashes, census, rec["births_rows"],
        "" if ok else "  first diff at line %s" % first_diff), flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="knecht24")
    ap.add_argument("--bin", default="~/universe2/dna3/dna_world3")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--local", action="store_true", help="run the engine binary on this machine (no ssh)")
    ap.add_argument("--out", default="results/dna3/verify")
    ap.add_argument("--only", default=None, help="comma-separated configuration names")
    ap.add_argument("--no-ckpt", action="store_true", help="skip the checkpoint tests")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    only = set(a.only.split(",")) if a.only else None
    runner = Runner(a)
    gpu = ["--gpu", str(a.gpu)]
    summary = []
    for name, args in CONFIGS:
        if only and name not in only:
            continue
        summary.append(compare(runner, a, name, args, eng_extra=gpu))
    if not a.no_ckpt:
        for name, args1, t1, args2, t2 in CKPT_TESTS:
            if only and name not in only:
                continue
            ck = runner.scratch(name + ".ckpt")
            r = runner.engine(args1 + " --ticks %d --no-hash --census-every 0 --checkpoint %s" % (t1, ck), gpu)
            if r.returncode:
                print(name, "engine (checkpoint run) failed:", r.stderr[-500:])
                summary.append({"name": name, "ok": False, "error": "engine checkpoint run"})
                continue
            local_ck = os.path.abspath(os.path.join(a.out, name + ".ckpt"))
            runner.fetch(ck, local_ck)
            rec = compare(runner, a, name, args2 + " --ticks %d" % (t1 + t2), ref_extra=["--resume", local_ck],
                          eng_extra=gpu + ["--resume", ck if not a.local else local_ck])
            rec["checkpoint_tick"] = t1
            summary.append(rec)
            os.remove(local_ck)
    with open(os.path.join(a.out, "verify.json"), "w") as f:
        json.dump(summary, f, indent=1)
    n_ok = sum(1 for r in summary if r.get("ok"))
    print("%d of %d configurations identical" % (n_ok, len(summary)))


if __name__ == "__main__":
    main()
