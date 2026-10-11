"""Harvest the computations that evolved in a DNA-3 world (docs/17 section 19): cut the IN .. OUT data path out of
every living gene, reduce it to a minimal program on the organisms' own machine, match it against the task ladder,
and write the library of evolved functions and, optionally, the promotion file that turns the best solutions into
new instructions (macros, `--macros FILE` of section 19.2).

  python tools/dna3_extract_solutions.py --mode ladder --ckpt RUN.ckpt [--days 1,2,3,4,5,6] [--uniform-code]
  python tools/dna3_extract_solutions.py --mode day6 --top results/dna3/open/O7_day7/O7_day7.top.gz [--tick T]
  python tools/dna3_extract_solutions.py --mode day6 --genes results/dna3/open/O6_day6/taskers_t102400.json
  python tools/dna3_extract_solutions.py --ops "START IN INC JNZB L1 STA OUT" --verify
  ... [--json library.json] [--promote macros.json --max-macros 8] [--verify [--verify-ticks 400]] [--min-carriers 1]

Sources: an engine checkpoint (`--ckpt`, every living gene with its spellings, needs numpy), the JSON of
tools/dna3_ckpt_genes.py (`--genes`), the top-genes TSV of tools/dna3_open.sh (`--top`, one census tick) or genes given
on the command line (`--ops`, repeatable). `--mode day6` is the 2-bit task world of day 6 (input in cell G-1, the
answer is the accumulator A at OUT); `--mode ladder` the 4-bit world of section 18 (input halves in cells G-2 and G-1,
the answer is the two cells).

Model. A gene is run from each of its IN codons on an abstract copy of the processor: the pc, the head H as an offset
from the I/O cell, the accumulator A and the I/O cells are tracked exactly; every other cell is unknown. Operations that
do not touch H, A or the I/O cells are no-ops for the data path (copying, HARVEST, ALLOC, DIVIDE, SIGNAL, WAIT, MOVE, EAT);
SEEK, SENSE, EAT, MOVE, EXEC and LISTEN consume their operand codon as in the world. Branches whose outcome the model
cannot know (SENSE on the sky, TERM on the copy state, SKZ / SKNZ / JNZB on an unknown cell) are followed both ways;
a slice is accepted only if every path reaches OUT with the same answer for every input. A path is rejected when it
hangs (256 steps), when EXEC would run a neighbour's gene, when the answer depends on an unknown value (LISTEN, LDA
off the I/O cells, SEEK), or when the gene writes into its own codons between IN and OUT (self-modification, which
the model does not follow). The accepted slice's static body (its codons between IN and OUT in loop order, with every
non-data operation and operand replaced by NOP so that skips and JNZB keep their targets) is checked to reproduce the
truth table on the body machine of tools/dna3_ladder_rank.py and then minimised by greedy deletion: an instruction is
removed whenever the truth table stays the same and no input hangs. The result is the evolved program in the units in
which section 18 ranks the tasks.

--verify seeds the gene (its recorded spelling) alone in a 4 x 4-slot world of the unchanged reference (sim/dna_ref.py,
days 1 to 6, mu 0, no inflow or damage, streak 1, no pools, a large energy store) and counts, over --verify-ticks ticks,
the answers it and its clones give and how many were correct per task: a task is verified when every answer was correct
and every input value was drawn. This runs the real VM with its sky, copying and seasons, not the model.
"""
import argparse
import collections
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sim"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dna_ref as R                           # noqa: E402
from dna3_ladder_rank import run as run_ladder_body   # noqa: E402

BODY_OPS = ("NOP", "FLIP", "INC", "LDA", "STA", "L1", "R1", "R3", "L3", "SKZ", "SKNZ", "JNZB")   # section 19.2
OPERAND_OPS = ("SEEK", "SENSE", "EAT", "MOVE", "EXEC", "LISTEN")
MOVES = {"R1": 1, "L1": -1, "R3": 3, "L3": -3}
MAX_STEPS = 256
MAX_PATHS = 4096
# donor codons for macros under the standard code (section 19.2): third-position variants of the operations that keep
# at least two synonyms (COPY keeps CTT, CTC; ALLOC GCT, GCC; DIVIDE GGT, GGC; SKNZ ATT, ATC; COPYR TGT)
DONORS = ("CTA", "CTG", "GCA", "GCG", "GGA", "GGG", "ATA", "TGC")


class Mode:
    def __init__(self, name):
        self.name = name
        if name == "ladder":
            self.n_in = 16
            self.tasks = [(n, f, m, r) for (n, f, m, r) in R.LADDER_TASKS]
        else:
            self.n_in = 4
            self.tasks = [(n, f, 3, r) for (n, f, r) in R.DNA3_TASKS]

    def start(self, x):
        """cells (offset -> value), head offset, accumulator after IN"""
        if self.name == "ladder":
            return {-1: x >> 2, 0: x & 3}, 0, x & 3
        return {0: x}, 0, x

    def answer(self, cells, a):
        if self.name == "ladder":
            hi, lo = cells.get(-1), cells.get(0)
            return None if hi is None or lo is None else 4 * hi + lo
        return a

    def run_body(self, prog, x):
        if self.name == "ladder":
            return run_ladder_body(prog, x)
        return run_day6_body(prog, x)


def run_day6_body(prog, x):
    """The day-6 body machine: one I/O cell holding x, the head on it, A = x; the answer is A. None if the program
    hangs or accesses a cell other than the I/O cell (the conventions of tools/dna3_ladder_rank.py)."""
    cell, a, pc, steps, h = x, x, 0, 0, 0
    n = len(prog)
    while pc < n and steps < 64:
        op = prog[pc]
        pc += 1
        steps += 1
        if op in MOVES:
            h += MOVES[op]
            continue
        if op == "NOP":
            continue
        if h != 0:
            return None
        if op == "FLIP":
            cell ^= 1
        elif op == "INC":
            cell = (cell + 1) & 3
        elif op == "LDA":
            a = cell
        elif op == "STA":
            cell = a
        elif op == "SKZ":
            if cell == 0:
                pc += 1
        elif op == "SKNZ":
            if cell != 0:
                pc += 1
        elif op == "JNZB":
            if cell != 0:
                pc = max(0, pc - 2)
    if pc < n:
        return None
    return a


def simulate(ops, i_in, x, mode, G, macros):
    """Run the gene from the IN at index i_in with input x. Returns (answers, rejects, soft): the set of answers over
    all explored paths, a Counter of fatal rejection reasons and one of paths that give no answer (a second IN before
    OUT: harmless, the next input is answered instead)."""
    n = len(ops)
    cells, h, a = mode.start(x)
    own_lo = -(G - 1)                                   # the own slot spans offsets own_lo .. 0
    answers, rejects, soft = set(), collections.Counter(), collections.Counter()
    # a path: (pc, cells, h, a, steps); branches are pushed on a stack
    stack = [(i_in + 1 if i_in + 1 < n else 1, dict(cells), h, a, 0)]
    paths = 0
    while stack:
        pc, cells, h, a, steps = stack.pop()
        paths += 1
        if paths > MAX_PATHS:
            rejects["branches"] += 1
            break
        while True:
            if steps >= MAX_STEPS:
                rejects["hangs"] += 1
                break
            steps += 1
            op = ops[pc]
            pc += 1
            if op in macros:                            # a macro codon of an earlier epoch: its body, inline
                res = run_body_on(macros[op], cells, h, a, own_lo, mode)
                if res is None:
                    rejects["macro"] += 1
                    break
                cells, h, a = res
            elif op == "IN":
                soft["superseded"] += 1                 # no answer on this path (the next IN's slice answers)
                break
            elif op == "OUT":
                v = mode.answer(cells, a)
                if v is None:
                    rejects["unknown answer"] += 1
                else:
                    answers.add(v)
                break
            elif op in OPERAND_OPS:
                has_operand = pc < n
                if has_operand:
                    pc += 1
                if op == "EXEC" and has_operand:
                    rejects["exec"] += 1               # a neighbour's gene would run: unknowable
                    break
                if op == "LISTEN" and has_operand:
                    a = None
                if op == "SEEK" and has_operand:
                    h = None
                if op == "SENSE" and has_operand:       # the sky decides: both ways
                    stack.append((pc + 1 if pc + 1 < n else 1, dict(cells), h, a, steps))
            elif op == "TERM":                          # the copy state decides: both ways
                stack.append((pc + 1 if pc + 1 < n else 1, dict(cells), h, a, steps))
            elif op in MOVES:
                if h is not None:
                    h += MOVES[op]
            elif op in ("INC", "FLIP", "STA"):
                if h is None:
                    rejects["unknown head"] += 1
                    break
                if h in cells:
                    v = cells[h]
                    if op == "STA":
                        cells[h] = a
                    elif v is not None:
                        cells[h] = (v + 1) & 3 if op == "INC" else v ^ 1
                elif own_lo <= h <= 0:
                    rejects["self-modifying"] += 1    # writes into the own codons: not followed
                    break
                # else: outside the own slot, no effect on the I/O cells
            elif op == "LDA":
                if h is None:
                    rejects["unknown head"] += 1
                    break
                a = cells.get(h)                        # None when the cell is unknown
            elif op in ("SKZ", "SKNZ", "JNZB"):
                v = cells.get(h) if h is not None else None
                if v is None:                           # unknown cell: both ways
                    if op == "JNZB":
                        stack.append((max(1, pc - 2), dict(cells), h, a, steps))
                    else:
                        stack.append((pc + 1 if pc + 1 < n else 1, dict(cells), h, a, steps))
                elif op == "JNZB":
                    if v != 0:
                        pc = max(1, pc - 2)
                elif (v == 0) == (op == "SKZ"):
                    pc += 1
            # every other operation leaves H, A and the I/O cells alone
            if pc >= n:
                pc = 1
    return answers, rejects, soft


def run_body_on(body, cells, h, a, own_lo, mode):
    """A macro body on the abstract state (used only for genes of a later epoch that carry macro codons)."""
    pc, steps, n = 0, 0, len(body)
    cells = dict(cells)
    while pc < n and steps < 64:
        op = body[pc]
        pc += 1
        steps += 1
        if op in MOVES:
            if h is not None:
                h += MOVES[op]
        elif op in ("INC", "FLIP", "STA"):
            if h is None:
                return None
            if h in cells:
                v = cells[h]
                if op == "STA":
                    cells[h] = a
                elif v is not None:
                    cells[h] = (v + 1) & 3 if op == "INC" else v ^ 1
            elif own_lo <= h <= 0:
                return None
        elif op == "LDA":
            if h is None:
                return None
            a = cells.get(h)
        elif op in ("SKZ", "SKNZ", "JNZB"):
            v = cells.get(h) if h is not None else None
            if v is None:
                return None
            if op == "JNZB":
                if v != 0:
                    pc = max(0, pc - 2)
            elif (v == 0) == (op == "SKZ"):
                pc += 1
    if pc < n:
        return None
    return cells, h, a


def static_body(ops, i_in, macros):
    """The codons between IN and OUT in loop order, data operations kept, everything else (and every operand) NOP.
    An OUT in an operand position ends the body too (it is executed on the paths where the operation before it is
    skipped); whether the body is a program for the function is settled by the truth-table check that follows."""
    n = len(ops)
    body = []
    pc = i_in + 1 if i_in + 1 < n else 1
    for _ in range(n):
        op = ops[pc]
        if op == "OUT":
            return body
        if op in macros:
            body.extend(macros[op])
        elif op in BODY_OPS:
            body.append(op)
        else:
            body.append("NOP")
        pc += 1
        if op in OPERAND_OPS and pc < n and ops[pc] != "OUT":
            body.append("NOP")
            pc += 1
        if pc >= n:
            pc = 1
    return None


def truth(body, mode):
    tt = []
    for x in range(mode.n_in):
        v = mode.run_body(tuple(body), x)
        if v is None:
            return None
        tt.append(v)
    return tt


def minimise(body, target, mode):
    """Greedy deletion: drop any instruction whose removal keeps the truth table (and never hangs)."""
    body = list(body)
    changed = True
    while changed:
        changed = False
        for i in range(len(body)):
            cand = body[:i] + body[i + 1:]
            if truth(cand, mode) == target:
                body = cand
                changed = True
                break
    return body


def match_tasks(tt, mode):
    out = []
    for k, (name, f, mask, rank) in enumerate(mode.tasks):
        if all((tt[x] & mask) == (f(x) & mask) for x in range(mode.n_in)):
            out.append((k, name, rank))
    return out


def spell_for(ops, table):
    """A spelling for an operation sequence: the lowest codon of every operation (used when no spelling is known)."""
    first = {}
    for c in range(64):
        first.setdefault(table[c], c)
    return " ".join(R.codon_text(first[R.OP_INDEX[o]]) for o in ops)


def verify(spelling, mode, G, ticks, uniform, macro_table=None):
    """Seed the gene alone in a 4 x 4-slot world of the reference and count its answers per task."""
    days = (1, 2, 3, 4, 5, 6)
    w = R.World(16 * G, G, I=16, T=0, mu=0.0, D=0, F=0, double=True, uniform=uniform, blank=True, seed=7,
                genome=R.parse_genome(spelling), n_seed=1, dna2=True, days=days, seed_slots=[0], W=4, firm=0,
                land_cell=1, land_thr=65536, emax=1 << 20, e_seed=1 << 20, eshift=0, ladder=mode.name == "ladder",
                streak_len=1, pool_cell=0, macros=macro_table, in_fresh=True)
    for _ in range(ticks):
        w.tick()
    seen = all(w.inputs_seen[x] > 0 for x in range(mode.n_in))
    ok = [k for k in range(len(mode.tasks)) if w.answers > 0 and w.correct[k] == w.answers]
    return {"answers": w.answers, "all_inputs": seen, "live": len(w.live), "tasks": ok}


def load_source(a, table):
    """-> (label, tick, genes) with genes a list of (ops tuple, carriers, spelling or None)"""
    genes = []
    if a.ckpt:
        from dna3_ckpt_genes import read_genes
        tick, n_live, by_ops, spell = read_genes(a.ckpt, table)
        for key, cnt in by_ops.most_common():
            genes.append((tuple(key.split()), cnt, spell[key]))
        return a.ckpt, tick, genes
    if a.genes:
        with open(a.genes) as f:
            d = json.load(f)
        for g in d["top"]:
            genes.append((tuple(g["ops"].split()), g["count"], collections.Counter({g["spelling"]: g.get("spelling_count", g["count"])})))
        return a.genes, d.get("tick"), genes
    if a.top:
        opener = gzip.open if a.top.endswith(".gz") else open
        rows = []
        with opener(a.top, "rt") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                p = line.rstrip("\n").split("\t")
                rows.append((int(p[0]), int(p[2]), p[5], p[6]))
        tick = a.tick if a.tick is not None else max(r[0] for r in rows)
        by = collections.OrderedDict()
        for t, cnt, codons, ops in rows:
            if t != tick:
                continue
            e = by.setdefault(ops, [0, collections.Counter()])
            e[0] += cnt
            e[1][codons] += cnt
        for ops, (cnt, sp) in by.items():
            genes.append((tuple(ops.split()), cnt, sp))
        return a.top, tick, genes
    for o in a.ops:
        genes.append((tuple(o.split()), 1, None))
    return "command line", None, genes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt", help="engine checkpoint (version 3 or 4)")
    src.add_argument("--genes", help="JSON written by tools/dna3_ckpt_genes.py --json")
    src.add_argument("--top", help="top-genes TSV of a run (tick hash count slot depth codons ops), optionally .gz")
    src.add_argument("--ops", action="append", help="a gene as operation names, START first (repeatable)")
    ap.add_argument("--tick", type=int, default=None, help="--top: the census tick to read (default: the last)")
    ap.add_argument("--mode", choices=("ladder", "day6"), default="ladder")
    ap.add_argument("--days", default="1,2,3,4,5,6", help="--ckpt: the days of the run (its codon table)")
    ap.add_argument("--uniform-code", action="store_true")
    ap.add_argument("--G", type=int, default=256, help="slot size of the run (for the own-slot bounds and --verify)")
    ap.add_argument("--macros", default=None, help="macro file of the run (genes may carry macro codons)")
    ap.add_argument("--min-carriers", type=int, default=1, help="skip genes with fewer organisms")
    ap.add_argument("--verify", action="store_true", help="run every accepted gene in the reference VM")
    ap.add_argument("--verify-ticks", type=int, default=400)
    ap.add_argument("--json", default=None, help="write the library of evolved functions")
    ap.add_argument("--promote", default=None, help="write the macro file for the next epoch (section 19.2)")
    ap.add_argument("--max-macros", type=int, default=8)
    ap.add_argument("--codons", default=",".join(DONORS), help="donor codons for --promote under the standard code")
    ap.add_argument("--rank-json", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "dna3", "ladder_rank.json"))
    a = ap.parse_args()

    mode = Mode(a.mode)
    days = [int(x) for x in a.days.split(",")] if a.days else []
    from dna3_ckpt_genes import table_for
    macro_table = R.parse_macros(a.macros) if a.macros else None   # the run's macro table (section 19.2 text format)
    table = table_for(days, a.uniform_code, macro_table)
    macros = {}
    if macro_table:
        for i, (_name, _cod, body) in enumerate(macro_table):
            macros["M%d" % i] = tuple(R.OP_NAMES[o] for o in body)
    known = {}
    if os.path.exists(a.rank_json):
        with open(a.rank_json) as f:
            for row in json.load(f):
                if row["shortest"]:
                    known[row["task"]] = (row["shortest"], row["program"])

    label, tick, genes = load_source(a, table)
    genes = [g for g in genes if g[1] >= a.min_carriers]
    total_carriers = sum(g[1] for g in genes)
    print("source %s tick %s: %d operation sequences, %d organisms, mode %s" % (label, tick, len(genes), total_carriers, mode.name))

    functions = collections.OrderedDict()           # truth tuple -> record
    rejects = collections.Counter()
    n_slices = n_accepted = 0
    verified_cache = {}
    for ops, carriers, spell in genes:
        if "IN" not in ops or "OUT" not in ops:
            continue
        for i_in in [i for i, o in enumerate(ops) if o == "IN"]:
            n_slices += 1
            per_x, bad = [], collections.Counter()
            for x in range(mode.n_in):
                ans, rej, _soft = simulate(ops, i_in, x, mode, a.G, macros)
                bad.update(rej)
                if rej or len(ans) != 1:
                    per_x = None
                    if not rej and not ans:
                        bad["never answers"] += 1
                    break
                per_x.append(next(iter(ans)))
            if per_x is None:
                if bad:
                    rejects[max(bad, key=bad.get)] += carriers
                else:
                    rejects["ambiguous"] += carriers
                continue
            sb = static_body(ops, i_in, macros)
            if sb is None or truth(sb, mode) != per_x:
                rejects["body mismatch"] += carriers
                continue
            mb = minimise(sb, per_x, mode)
            n_accepted += 1
            tt = tuple(per_x)
            rec = functions.setdefault(tt, {"truth": list(tt), "tasks": [n for _k, n, _r in match_tasks(tt, mode)],
                                            "best_body": mb, "best_len": len(mb), "carriers": 0, "genes": []})
            rec["carriers"] += carriers
            if len(mb) < rec["best_len"]:
                rec["best_body"], rec["best_len"] = mb, len(mb)
            g = {"ops": " ".join(ops), "carriers": carriers, "in_index": i_in, "static_body": sb, "minimal_body": mb}
            if spell:
                g["spelling"] = spell.most_common(1)[0][0]
            if a.verify and rec["tasks"]:
                sp = g.get("spelling") or spell_for(ops, table)
                if sp not in verified_cache:
                    verified_cache[sp] = verify(sp, mode, a.G, a.verify_ticks, a.uniform_code, macro_table)
                v = verified_cache[sp]
                g["verify"] = v
                g["verified_tasks"] = [mode.tasks[k][0] for k in v["tasks"]]
            rec["genes"].append(g)

    # report
    print("%d IN slices, %d accepted, %d distinct functions; rejected organisms by reason: %s"
          % (n_slices, n_accepted, len(functions), dict(rejects.most_common())))
    order = sorted(functions.values(), key=lambda r: -r["carriers"])
    for rec in order:
        tasks = ", ".join(rec["tasks"]) or "no ladder task"
        print("\n%9d organisms  f = %s  [%s]  minimal body (%d): %s" % (rec["carriers"], rec["truth"], tasks, rec["best_len"], " ".join(rec["best_body"]) or "-"))
        for name in rec["tasks"]:
            if name in known:
                print("           shortest known for %s: %d (%s)" % (name, known[name][0], known[name][1]))
        for g in sorted(rec["genes"], key=lambda g: -g["carriers"])[:3]:
            line = "  %9d  IN@%-2d %s" % (g["carriers"], g["in_index"], g["ops"])
            if "verify" in g:
                v = g["verify"]
                line += "  | verified: %s (%d answers, all inputs %s, %d live)" % (", ".join(g["verified_tasks"]) or "none", v["answers"], v["all_inputs"], v["live"])
            print(line)

    if a.json:
        with open(a.json, "w") as f:
            json.dump({"schema": "dna3-solutions-v1", "mode": mode.name, "source": label, "tick": tick,
                       "sequences": len(genes), "organisms": total_carriers, "slices": n_slices, "accepted": n_accepted,
                       "rejects": dict(rejects), "functions": order}, f, indent=1)
        print("\nlibrary written to %s" % a.json)

    if a.promote:
        # one macro per task: the shortest evolved body of length >= 2 (a one-instruction solution is already an
        # instruction), one body promoted once under its hardest task; hardest tasks first, then the most carried;
        # donor codons by rising use among the living
        best = {}
        for rec in order:
            if rec["best_len"] < 2:
                continue
            for k, name, rank in match_tasks(tuple(rec["truth"]), mode):
                cur = best.get(name)
                if cur is None or rec["best_len"] < cur["len"] or (rec["best_len"] == cur["len"] and rec["carriers"] > cur["carriers"]):
                    verified = any(name in g.get("verified_tasks", []) for g in rec["genes"])
                    best[name] = {"name": name, "task": k, "rank": rank, "body": rec["best_body"], "len": rec["best_len"],
                                  "truth": rec["truth"], "tasks": rec["tasks"], "carriers": rec["carriers"],
                                  "gene": max(rec["genes"], key=lambda g: g["carriers"])["ops"],
                                  "verified": verified if a.verify else None}
        chosen, bodies = [], set()
        for m in sorted(best.values(), key=lambda m: (-m["rank"], -m["carriers"], m["len"])):
            if tuple(m["body"]) not in bodies:
                bodies.add(tuple(m["body"]))
                chosen.append(m)
        chosen = chosen[:a.max_macros]
        usage = collections.Counter()
        for ops, carriers, spell in genes:
            if not spell:
                continue
            for sp, cnt in spell.items():
                for cod in set(sp.split()[1:]):
                    usage[cod] += cnt
        if a.uniform_code:
            codons = ["%d" % (32 + i) for i in range(len(chosen))]
        else:
            donors = sorted(a.codons.split(","), key=lambda c: (usage[c], DONORS.index(c) if c in DONORS else 99))
            if len(chosen) > len(donors):
                chosen = chosen[:len(donors)]
            codons = donors[:len(chosen)]
        for i, (m, cod) in enumerate(zip(chosen, codons)):
            m["codon"] = cod
            m["macro"] = "M%d" % i
        with open(a.promote, "w") as f:
            json.dump({"schema": "dna3-macros-v1", "mode": mode.name, "source": label, "tick": tick,
                       "donor_usage": {c: usage[c] for c in (a.codons.split(",") if not a.uniform_code else [])},
                       "macros": chosen}, f, indent=1)
        # the table the engines read (--macros FILE, section 19.2): "NAME CODON OP OP ...", one macro per line
        txt = a.promote[:-5] + ".txt" if a.promote.endswith(".json") else a.promote + ".txt"
        with open(txt, "w") as f:
            f.write("# macro table of docs/17 section 19.2, promoted from %s (tick %s, mode %s)\n" % (label, tick, mode.name))
            for m in chosen:
                f.write("%s %s %s   # %s, rank %d, %d carriers\n" % (m["macro"], m["codon"], " ".join(m["body"]), m["name"], m["rank"], m["carriers"]))
        print("\npromotion file %s (+ %s): %d macros" % (a.promote, txt, len(chosen)))
        for m in chosen:
            print("  %s %s  %-8s rank %d  body (%d): %s  carriers %d%s" % (m["macro"], m["codon"], m["name"], m["rank"], m["len"], " ".join(m["body"]), m["carriers"],
                                                                      "" if m["verified"] is None else ("  verified" if m["verified"] else "  NOT verified")))


if __name__ == "__main__":
    main()
