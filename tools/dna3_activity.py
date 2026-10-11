"""Day 7 of DNA-3 (docs/17 sections 9 and 17): Bedau-Packard evolutionary activity of operation sequences against a
neutral shadow, new activity, diversity, novelty and complexity, and the three open-endedness criteria of section 9.

  python tools/dna3_activity.py RUN.ops --days 1,2,3,4,5,6 --seed-genome "ATG AGA GCT CTT TTA GGT" \
      [--shadow genome|alleles] [--replicates 3] [--pct 99] [--cloud-factor 1] [--sub-frac 0.02] [--out DIR] [--max-census K]

Input: the engine's --op-census file (gpu/dna_world.cu): per census a line "C tick live births divide_births
op_faithful spontaneous deaths changed n_seq", "S ophash ops" the first time a sequence is printed, and "H ophash count
zone0 t0 t1 t2 t3" per operation sequence of the living. A resumed run appends to the file; census blocks that a
later block (same or earlier tick) supersedes are dropped.

Component = operation sequence (codons 0 .. first STOP mapped to operations, synonymous codons collapsed).
Activity of a component at census k = the sum of its census counts up to k (Bedau and Packard's component activity,
as in Universe-1 docs/12); a component that disappears and comes back keeps its activity.

The neutral shadow is a population without selection driven by the world's census-level demography: in the interval
before census k it gets exactly the world's DIVIDE births (each copies a uniformly random shadow member), spontaneous
births (a new random expressible gene, as the inflow makes them) and deaths (uniformly random members), applied in
substeps of at most --sub-frac of the population (births choose their parents from the population at the start of
the substep, and the substep's deaths are refilled by its births), so the shadow's population equals the world's at
every census. A birth changes the operation sequence with the world's measured probability for that interval,
u = 1 - op_faithful / divide_births. --shadow genome (default): shadow members carry codon genomes; a changed birth
gets one uniformly random point mutation among the single-cell changes of the copied cells (codons 1 .. len - 1 and
the STOP) that change the operation sequence and leave the child expressible (START at codon 0, length >= 2: other
children are not births in the world either); a mutated STOP reads through into random codons up to the next STOP
(the world reads into the slot's old content instead). So recurrent mutation, which can re-create a sequence and keep
it present without selection, exists in the shadow as in the world. --shadow alleles: every changed birth and every
spontaneous birth makes a new sequence (infinite alleles), a lower bar. What the shadow does not have: the timing of
events inside an interval (the world's births and deaths are spread evenly over the substeps), in-life changes of a
gene (writes into a living gene; the census reports them as "changed"), and spatial structure.

Threshold theta(k) = the --pct percentile of the activities of the shadow components present at census k (mean over
--replicates shadows, each in its own process). A world component present at census k is adaptive at k if its
activity exceeds theta(k). Second, stricter variant ("est_" keys, "established"): an adaptive component that also has
more carriers than --cloud-factor times what mutation from its present one-step neighbours would maintain if it were
exactly as fit as they are (cloud_supply below), so that the mutational cloud of the dominant sequences, which a
neutral shadow cannot have, does not count as adaptive. Per census: diversity (distinct sequences), adaptive diversity, new adaptive components (adaptive for the first time),
novel ones among them (whose set of operations no earlier adaptive component had), cumulative and mean activity of the
adaptive components, and complexity over the adaptive components: distinct operations, length, tasks performed (task k
counts for a component when at least half of its carriers are at streak 3 for k), trophic level (1 plant, 2 mixotroph:
HARVEST and EAT, 3 heterotroph: EAT only), and the share of adaptive carriers using SENSE, SIGNAL, MOVE (maxima and
carrier-weighted means). Criteria over the analysed censuses split into quarters: (1) rate of new adaptive components
in Q4 >= half that in Q2; (2) some complexity measure reaches a new maximum in Q4; (3) adaptive diversity in the second
half never below half its mean over the run. Open-ended if all three hold, plateau if (1) fails, bounded if (1) holds
and (2) fails; evaluated for both variants. Writes activity.json (per census rows, quarters, criteria) and activity.png
into --out. Without --seed-genome (e.g. an op census that starts at a resume) the shadow starts as a copy of the first
census, each sequence spelled with the lowest codon of each operation.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sim"))
sys.path.insert(0, HERE)
import dna_ref as R  # noqa: E402
from dna3_ckpt_genes import table_for  # noqa: E402

# operation numbers of the engine (enum Op in gpu/dna_world.cu): the op hash is FNV-1a 64 over these
ENG_OPS = ("NOP START STOP COPY R1 L1 R3 L3 JNZB SETP SETQ ALLOC DIVIDE SKNZ SKZ LDA STA INC FLIP SEEK COPYR REWIND TERM "
           "HARVEST SENSE WAIT EAT MOVE EXEC SIGNAL LISTEN IN OUT").split()
ENG_INDEX = {n: i for i, n in enumerate(ENG_OPS)}


# ------------------------------------------------------------------ op census file
def scan(path):
    """first pass: names, and the byte offset and tick of every census block; drop superseded blocks"""
    names, blocks = {}, []
    with open(path, "rb") as f:
        off = 0
        for line in f:
            c = line[:1]
            if c == b"C":
                blocks.append((int(line.split()[1]), off))
            elif c == b"S":
                p = line.split(None, 2)
                names[int(p[1], 16)] = p[2].decode().split() if len(p) > 2 else []
            off += len(line)
    keep = []
    for tick, off in blocks:                 # a resumed run rewrites the ticks after its checkpoint
        while keep and keep[-1][0] >= tick:
            keep.pop()
        keep.append((tick, off))
    return names, keep


def read_block(f, off):
    f.seek(off)
    p = f.readline().split()
    hdr = dict(zip(("tick", "live", "births", "divide", "op_faithful", "spont", "deaths", "changed", "n_seq"),
                   (int(x) for x in p[1:10])))
    n = hdr["n_seq"]
    keys = np.empty(n, dtype=np.uint64)
    val = np.empty((n, 6), dtype=np.int64)
    i = 0
    while i < n:
        line = f.readline()
        if not line:
            raise SystemExit("truncated census block at tick %d" % hdr["tick"])
        if line[:1] != b"H":
            continue
        q = line.split()
        keys[i] = int(q[1], 16)
        val[i] = [int(x) for x in q[2:8]]
        i += 1
    order = np.argsort(keys)
    return hdr, keys[order], val[order]


# ------------------------------------------------------------------ component properties
class Props:
    """static properties of an operation sequence, from its name"""
    def __init__(self, names):
        self.names, self.cache = names, {}

    def get(self, h):
        r = self.cache.get(h)
        if r is None:
            ops = self.names.get(h, [])
            s = set(ops)
            harvest, eat = "HARVEST" in s, "EAT" in s
            troph = 2 if harvest and eat else 3 if eat else 1 if harvest else 0
            r = (len(set(ops[1:])), len(ops), troph, "SENSE" in s, "SIGNAL" in s, "MOVE" in s, frozenset(ops[1:]))
            self.cache[h] = r
        return r


# ------------------------------------------------------------------ activity bookkeeping (sorted keys)
class Ledger:
    """activity, first census and first adaptive census per component key (uint64), merged census by census"""
    def __init__(self):
        self.keys = np.empty(0, dtype=np.uint64)
        self.act = np.empty(0, dtype=np.float64)
        self.first = np.empty(0, dtype=np.int32)
        self.crossed = np.empty(0, dtype=np.int32)

    def add(self, keys, counts, k):
        """keys sorted, unique; returns the indices of keys (after inserting new ones)"""
        pos = np.searchsorted(self.keys, keys)
        hit = pos < len(self.keys)
        hit[hit] = self.keys[pos[hit]] == keys[hit]
        new = keys[~hit]
        if len(new):
            self.keys = np.concatenate([self.keys, new])
            self.act = np.concatenate([self.act, np.zeros(len(new))])
            self.first = np.concatenate([self.first, np.full(len(new), k, dtype=np.int32)])
            self.crossed = np.concatenate([self.crossed, np.full(len(new), -1, dtype=np.int32)])
            o = np.argsort(self.keys, kind="stable")
            self.keys, self.act, self.first, self.crossed = self.keys[o], self.act[o], self.first[o], self.crossed[o]
        idx = np.searchsorted(self.keys, keys)
        self.act[idx] += counts
        return idx


# ------------------------------------------------------------------ the neutral shadow
class Shadow:
    """genotypes are bytes of codon values (START .. last codon before the STOP); the component key of a genotype is
    a 64-bit digest of its operation sequence (the shadow's own keys; they are never compared with the world's)"""
    def __init__(self, mode, table, maxk, rng, init):
        self.mode, self.table, self.maxk, self.rng = mode, table, maxk, rng
        self.trans = bytes([ENG_INDEX[R.OP_NAMES[table[c]]] for c in range(64)] + [0] * 192)
        self.stop = [table[c] == R.STOP for c in range(64)]
        self.start_codon = next(c for c in range(64) if table[c] == R.START)
        self.gen, self.gop, self.index = [], [], {}
        self.lost_births = self.lost_deaths = 0
        self.realized = [0, 0]                   # changed births: op sequence changed / asked to change
        self.fbuf, self.fi, self.cbuf, self.ci = [], 0, [], 0
        pop = []
        for codons, n in init:
            pop += [self.register(bytes(codons))] * n
        self.pop = np.array(pop, dtype=np.int64)

    def rand(self):
        if self.fi >= len(self.fbuf):
            self.fbuf, self.fi = self.rng.random(1 << 16).tolist(), 0
        self.fi += 1
        return self.fbuf[self.fi - 1]

    def codon(self):
        if self.ci >= len(self.cbuf):
            self.cbuf, self.ci = self.rng.integers(64, size=1 << 16).tolist(), 0
        self.ci += 1
        return self.cbuf[self.ci - 1]

    def key(self, c):
        return int.from_bytes(hashlib.blake2b(c.translate(self.trans), digest_size=8).digest(), "little")

    def register(self, c):
        g = self.index.get(c)
        if g is None:
            g = len(self.gen)
            self.gen.append(c)
            self.gop.append(self.key(c))
            self.index[c] = g
        return g

    def unique(self):
        """a component nobody else has (alleles mode)"""
        self.gen.append(b"")
        self.gop.append(int(self.rng.integers(1 << 62)) + (1 << 63))
        return len(self.gen) - 1

    def random_tail(self, room):
        out = []
        while len(out) < room:
            c = self.codon()
            if self.stop[c]:
                break
            out.append(c)
        return bytes(out)

    def spontaneous(self):
        if self.mode == "alleles":
            return self.unique()
        while True:
            tail = self.random_tail(self.maxk)
            if tail:                              # START plus at least one codon: expressible
                return self.register(bytes((self.start_codon,)) + tail)

    def mutate_one(self, g):
        """one uniformly random single-cell change of codons 1 .. len - 1 or of the STOP that changes the operation
        sequence and leaves the child expressible (rejection sampling over the cells and the three other bases)"""
        if self.mode == "alleles":
            self.realized[0] += 1
            return self.unique()
        c = self.gen[g]
        L = len(c)
        npos = L - 1 + (L <= self.maxk)           # codons 1 .. L - 1, and the STOP if there is one in the slot
        table, stop = self.table, self.stop
        for _ in range(1000):
            r = int(self.rand() * npos * 9)
            p, j, bi = 1 + r // 9, (r % 9) // 3, r % 3
            old = c[p] if p < L else 48           # TAA stands for the STOP (the copied STOP's spelling is unknown)
            sh = 2 * (2 - j)
            ob = (old >> sh) & 3
            nb = bi if bi < ob else bi + 1
            new = (old & ~(3 << sh)) | (nb << sh)
            if p < L:
                if stop[new]:
                    if p < 2:
                        continue
                    child = c[:p]
                elif table[new] != table[old]:
                    child = c[:p] + bytes((new,)) + c[p + 1:]
                else:
                    continue
            else:
                if stop[new]:
                    continue
                child = c + bytes((new,)) + self.random_tail(self.maxk - L)   # read-through into random codons
            self.realized[0] += 1
            return self.register(child)
        return g

    def interval(self, divide, spont, deaths, u, live_end, sub_frac):
        L0 = len(self.pop)
        ref = max(1, min(L0, live_end))
        n = int(min(50000, max(1, math.ceil(max(divide + spont, deaths) / (sub_frac * ref)))))
        cuts = lambda x: np.diff(np.round(np.linspace(0, x, n + 1)).astype(np.int64))
        bs, ss, ds = cuts(divide), cuts(spont), cuts(deaths)
        rng = self.rng
        for b, s, d in zip(bs.tolist(), ss.tolist(), ds.tolist()):
            L = len(self.pop)
            if L == 0:
                self.lost_births += b
                b = 0
            par = self.pop[rng.integers(L, size=b)] if b else np.empty(0, dtype=np.int64)
            if b and u > 0:
                m = np.nonzero(rng.random(b) < u)[0]
                if len(m):
                    self.realized[1] += len(m)
                    par[m] = [self.mutate_one(g) for g in par[m].tolist()]
            if s:
                par = np.concatenate([par, np.array([self.spontaneous() for _ in range(s)], dtype=np.int64)])
            if d > L:
                self.lost_deaths += d - L
                d = L
            dead = rng.choice(L, size=d, replace=False) if d else np.empty(0, dtype=np.int64)
            k = min(d, len(par))
            self.pop[dead[:k]] = par[:k]
            if d > k:
                self.pop = np.delete(self.pop, dead[k:])
            if len(par) > k:
                self.pop = np.concatenate([self.pop, par[k:]])

    def census(self):
        if len(self.pop) == 0:
            return np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.int64)
        ops = np.array(self.gop, dtype=np.uint64)[self.pop]
        keys, cnt = np.unique(ops, return_counts=True)
        self.gc()
        return keys, cnt

    def gc(self):
        """forget genotypes nobody carries (they are registered again if mutation makes them again)"""
        if len(self.gen) < 4 * max(1, len(self.pop)):
            return
        alive = np.unique(self.pop)
        remap = np.full(len(self.gen), -1, dtype=np.int64)
        remap[alive] = np.arange(len(alive))
        self.gen = [self.gen[g] for g in alive.tolist()]
        self.gop = [self.gop[g] for g in alive.tolist()]
        self.index = {c: i for i, c in enumerate(self.gen)} if self.mode == "genome" else {}
        self.pop = remap[self.pop]


def shadow_worker(conn, mode, table, maxk, seed, init, pct, sub_frac):
    """one shadow replicate in its own process: per census interval it gets the demography and returns the
    threshold, its number of new adaptive components, its diversity and its population"""
    sh, led = Shadow(mode, table, maxk, np.random.default_rng(seed), init), Ledger()
    k = 0
    while True:
        msg = conn.recv()
        if msg is None:
            conn.send({"lost_births": sh.lost_births, "lost_deaths": sh.lost_deaths, "changed_births_realized": sh.realized})
            return
        if msg:
            sh.interval(*msg, sub_frac)
        keys, cnt = sh.census()
        idx = led.add(keys, cnt, k)
        act = led.act[idx]
        theta = float(np.percentile(act, pct)) if len(act) else 0.0
        nw = (act > theta) & (led.crossed[idx] < 0)
        led.crossed[idx[nw]] = k
        conn.send((theta, int(nw.sum()), len(keys), len(sh.pop)))
        k += 1


# ------------------------------------------------------------------ per-census statistics and criteria
def cloud_supply(seqs, cnt, which):
    """the carriers of sequence i that its present one-step neighbours j would keep present by mutation if i were
    exactly as fit as they are: sum_j c_j / (9 len_j) over j that differ from i in one operation (same length) or
    that i truncates (i a proper prefix of j). In mutation-selection balance a neighbour with disadvantage s holds
    c_j u q / (u + s) carriers (q = its share of j's changed births); for s = 0 that is c_j q, independent of u.
    q = 1 / (9 len_j) assumes one of the 9 len_j single-cell changes of j's copied cells (codons 1 .. len - 1 and the
    STOP) per neighbour relation. A component above this count is more common than a neutral mutant of its
    neighbours would be: favoured by selection, or amplified by drift."""
    W, T = {}, {}
    for s, c in zip(seqs, cnt):
        w = c / (9.0 * len(s)) if s else 0.0
        for p in range(1, len(s)):
            key = s[:p] + b"\xff" + s[p + 1:]
            W[key] = W.get(key, 0.0) + w
        for q in range(2, len(s)):
            T[s[:q]] = T.get(s[:q], 0.0) + w
    out = {}
    for i in which:
        s = seqs[i]
        if not s:
            out[i] = 0.0
            continue
        x = sum(W.get(s[:p] + b"\xff" + s[p + 1:], 0.0) for p in range(1, len(s)))
        x -= (len(s) - 1) * cnt[i] / (9.0 * len(s))       # i matches its own masks
        out[i] = max(x, 0.0) + T.get(s, 0.0)
    return out


def metrics(tag, mask, new, keys, cnt, val, act, props, seen_sets):
    ai = np.nonzero(mask)[0]
    novel = 0
    for i in np.nonzero(new)[0].tolist():
        if props.get(int(keys[i]))[6] not in seen_sets:
            novel += 1
    for i in ai.tolist():
        seen_sets.add(props.get(int(keys[i]))[6])
    P = np.array([props.get(int(keys[i]))[:6] for i in ai.tolist()], dtype=np.float64).reshape(-1, 6)
    c = cnt[ai].astype(np.float64)
    tasks = (val[ai, 2:6] * 2 >= val[ai, 0:1]).sum(axis=1).astype(np.float64)
    wsum = c.sum()
    wm = (lambda x: float((x * c).sum() / wsum)) if wsum else (lambda x: 0.0)
    mx = (lambda x: float(x.max())) if len(ai) else (lambda x: 0.0)
    r = {"adaptive": int(len(ai)), "adaptive_carriers": int(c.sum()), "new_adaptive": int(new.sum()), "novel": novel,
         "cum_activity": float(act[ai].sum()), "mean_activity": float(act[ai].mean()) if len(ai) else 0.0,
         "max_distinct_ops": mx(P[:, 0]), "mean_distinct_ops": wm(P[:, 0]),
         "max_length": mx(P[:, 1]), "mean_length": wm(P[:, 1]),
         "max_tasks": mx(tasks), "mean_tasks": wm(tasks),
         "max_trophic": mx(P[:, 2]), "mean_trophic": wm(P[:, 2]),
         "share_sense": wm(P[:, 3]), "share_signal": wm(P[:, 4]), "share_move": wm(P[:, 5])}
    return {tag + k: v for k, v in r.items()}


def quarter_slices(K):
    b = [round(K * i / 4) for i in range(5)]
    return [slice(b[i], b[i + 1]) for i in range(4)]


MEASURES = ["max_distinct_ops", "max_tasks", "max_trophic", "mean_distinct_ops", "mean_tasks", "mean_trophic",
            "share_sense", "share_signal", "share_move"]


def quarters(rows, tag):
    col = lambda name: np.array([r[name] for r in rows], dtype=np.float64)
    out = []
    for i, q in enumerate(quarter_slices(len(rows))):
        n = q.stop - q.start
        m = lambda name: float(col(name)[q].mean()) if n else 0.0
        d = {"quarter": i + 1, "censuses": n, "ticks": [rows[q.start]["tick"], rows[q.stop - 1]["tick"]] if n else None,
             "new_adaptive_per_census": m(tag + "new_adaptive"), "novel_per_census": m(tag + "novel"),
             "adaptive_mean": m(tag + "adaptive"), "diversity_mean": m("diversity"),
             "shadow_new_adaptive_per_census": m("shadow_new_adaptive"), "shadow_diversity_mean": m("shadow_diversity")}
        for name in MEASURES + ["max_length", "mean_length"]:
            d[name + "_max"] = float(col(tag + name)[q].max()) if n else 0.0
        out.append(d)
    return out


def criteria(rows, tag):
    K = len(rows)
    qs = quarter_slices(K)
    col = lambda name: np.array([r[tag + name] for r in rows], dtype=np.float64)
    new = col("new_adaptive")
    r2 = float(new[qs[1]].mean()) if qs[1].stop > qs[1].start else 0.0
    r4 = float(new[qs[3]].mean()) if qs[3].stop > qs[3].start else 0.0
    c1 = r4 >= 0.5 * r2 and r4 > 0
    early = slice(0, qs[3].start)
    ok = qs[3].stop > qs[3].start and early.stop > 0
    by = {}
    for m in MEASURES:
        x = col(m)
        by[m] = {"Q1_Q3_max": float(x[early].max()) if ok else None, "Q4_max": float(x[qs[3]].max()) if ok else None,
                 "new_maximum": bool(ok and x[qs[3]].max() > x[early].max())}
    c2 = any(v["new_maximum"] for v in by.values())
    c2_maxima = any(by[m]["new_maximum"] for m in MEASURES[:3])
    ad = col("adaptive")
    half = slice(K // 2, K)
    c3 = bool(K and ad[half].min() >= 0.5 * ad.mean())
    verdict = ("open-ended" if c1 and c2 and c3 else "plateau" if not c1 else "bounded" if not c2
               else "criteria (1) and (2) hold, (3) fails")
    return {"c1_new_activity": {"Q2_per_census": r2, "Q4_per_census": r4, "holds": bool(c1)},
            "c2_new_complexity_maximum": {"by_measure": by, "holds": bool(c2), "holds_with_maxima_only": bool(c2_maxima)},
            "c3_diversity": {"mean": float(ad.mean()) if K else 0.0, "min_second_half": float(ad[half].min()) if K else 0.0,
                             "holds": c3},
            "verdict": verdict}


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ops")
    ap.add_argument("--days", default="1,2,3,4,5,6")
    ap.add_argument("--uniform-code", action="store_true")
    ap.add_argument("--G", type=int, default=256)
    ap.add_argument("--seed-genome", default=None, help="codons of the seed (fresh runs); else the first census seeds the shadow")
    ap.add_argument("--shadow", choices=("genome", "alleles"), default="genome")
    ap.add_argument("--replicates", type=int, default=3)
    ap.add_argument("--pct", type=float, default=99.0)
    ap.add_argument("--sub-frac", type=float, default=0.02)
    ap.add_argument("--rng-seed", type=int, default=1)
    ap.add_argument("--cloud-factor", type=float, default=1.0)
    ap.add_argument("--max-census", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    days = [int(x) for x in a.days.split(",")]
    table = table_for(days, a.uniform_code)
    maxk = a.G // 3 - 1
    out = a.out or os.path.splitext(a.ops)[0] + "_activity"
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    names, blocks = scan(a.ops)
    if a.max_census:
        blocks = blocks[: a.max_census]
    K = len(blocks)
    print("%s: %d census blocks (ticks %d .. %d), %d named sequences, scanned in %.0f s"
          % (a.ops, K, blocks[0][0], blocks[-1][0], len(names), time.time() - t0))
    props = Props(names)
    f = open(a.ops, "rb")
    hdr0, keys0, val0 = read_block(f, blocks[0][1])
    fresh = a.seed_genome is not None
    if fresh:
        sg = [R.codon_value(x) for x in a.seed_genome.replace(",", " ").split()]
        L0 = hdr0["live"] - hdr0["births"] + hdr0["deaths"]
        init = [(sg, L0)]
        start_k = 0
    else:   # seed the shadow with the first census, each sequence spelled with the lowest codon of each operation
        spell = {}
        for c in range(63, -1, -1):
            spell[R.OP_NAMES[table[c]]] = c
        init = [([spell[o] for o in names[int(h)]], int(v[0])) for h, v in zip(keys0, val0)]
        start_k = 1
    workers = []
    for r in range(a.replicates):
        here, there = mp.Pipe()
        proc = mp.Process(target=shadow_worker, args=(there, a.shadow, table, maxk, [a.rng_seed, r], init, a.pct, a.sub_frac),
                          daemon=True)
        proc.start()
        workers.append((proc, here))
    world = Ledger()
    crossed_est = {}                         # first census at which a component was adaptive and established
    seen = {"all": set(), "est": set()}      # operation sets of earlier adaptive components (novelty)
    opb = {}                                 # op sequence as bytes of engine numbers, per hash

    def opbytes(h):
        b = opb.get(h)
        if b is None:
            b = opb[h] = bytes(ENG_INDEX[o] for o in names.get(h, []))
        return b

    rows = []
    for k, (tick, off) in enumerate(blocks):
        hdr, keys, val = (hdr0, keys0, val0) if k == 0 else read_block(f, off)
        cnt = val[:, 0]
        u = 1.0 - hdr["op_faithful"] / hdr["divide"] if hdr["divide"] else 0.0
        msg = (hdr["divide"], hdr["spont"], hdr["deaths"], u, hdr["live"]) if k >= start_k else ()
        for proc, c in workers:
            c.send(msg)
        res = [c.recv() for proc, c in workers]
        th, s_new, s_div = [x[0] for x in res], [x[1] for x in res], [x[2] for x in res]
        if res[0][3] != hdr["live"]:
            print("warning: shadow population %d, world %d at tick %d" % (res[0][3], hdr["live"], tick))
        theta = float(np.mean(th))
        idx = world.add(keys, cnt, k)
        act = world.act[idx]
        ad = act > theta
        nw = ad & (world.crossed[idx] < 0)
        world.crossed[idx[nw]] = k
        # established: more carriers than --cloud-factor times a neutral one-step mutant of the present neighbours
        hk = keys.tolist()
        seqs = [opbytes(h) for h in hk]
        sup = cloud_supply(seqs, cnt.tolist(), np.nonzero(ad)[0].tolist())
        est = np.zeros(len(keys), dtype=bool)
        for i, x in sup.items():
            est[i] = cnt[i] > a.cloud_factor * x
        nw_est = np.zeros(len(keys), dtype=bool)
        for i in np.nonzero(est)[0].tolist():
            if hk[i] not in crossed_est:
                crossed_est[hk[i]] = k
                nw_est[i] = True
        row = {"k": k, "tick": tick, "live": hdr["live"], "divide_births": hdr["divide"], "spontaneous": hdr["spont"],
               "deaths": hdr["deaths"], "u_op": round(u, 6), "changed_in_life": hdr["changed"],
               "diversity": int(len(keys)), "theta": theta, "theta_min": float(min(th)), "theta_max": float(max(th)),
               "shadow_diversity": float(np.mean(s_div)), "shadow_new_adaptive": float(np.mean(s_new)),
               "top": [[" ".join(names.get(hk[i], [])), int(cnt[i]), float(act[i])] for i in np.argsort(-cnt)[:3].tolist()]}
        for tag, mask, new in (("", ad, nw), ("est_", est, nw_est)):
            row.update(metrics(tag, mask, new, keys, cnt, val, act, props, seen["est" if tag else "all"]))
        rows.append(row)
        if k % 20 == 0 or k == K - 1:
            print("k %4d tick %9d live %7d div %6d adaptive %5d (est %4d) new %4d (est %3d, shadow %6.1f) theta %.3g u %.4f  [%.0f s]"
                  % (k, tick, hdr["live"], len(keys), row["adaptive"], row["est_adaptive"], row["new_adaptive"],
                     row["est_new_adaptive"], row["shadow_new_adaptive"], theta, u, time.time() - t0), flush=True)
    crit = {"all": criteria(rows, ""), "established": criteria(rows, "est_")}
    quart = {"all": quarters(rows, ""), "established": quarters(rows, "est_")}
    ends = []
    for proc, c in workers:
        c.send(None)
        ends.append(c.recv())
        proc.join()
    meta = {"ops": a.ops, "shadow": a.shadow, "replicates": a.replicates, "pct": a.pct, "sub_frac": a.sub_frac,
            "censuses": K, "fresh": fresh, "seed_genome": a.seed_genome,
            "shadow_end": ends, "world_components_ever": int(len(world.keys)),
            "world_adaptive_ever": int((world.crossed >= 0).sum()), "world_established_ever": len(crossed_est),
            "cloud_factor": a.cloud_factor, "seconds": round(time.time() - t0, 1)}
    with open(os.path.join(out, "activity.json"), "w") as fo:
        json.dump({"meta": meta, "criteria": crit, "quarters": quart, "rows": rows}, fo, indent=1)
    for v in ("all", "established"):
        c = crit[v]
        print("%-11s %-12s (1) new adaptive per census Q2 %.1f, Q4 %.1f: %s; (2) new maximum in Q4: %s; (3) adaptive diversity "
              "min %.0f in the second half, mean %.0f: %s" % (v, c["verdict"], c["c1_new_activity"]["Q2_per_census"],
              c["c1_new_activity"]["Q4_per_census"], c["c1_new_activity"]["holds"],
              [m for m, x in c["c2_new_complexity_maximum"]["by_measure"].items() if x["new_maximum"]] or "none",
              c["c3_diversity"]["min_second_half"], c["c3_diversity"]["mean"], c["c3_diversity"]["holds"]))
    plot(rows, os.path.join(out, "activity.png"), a.ops, crit)


def plot(rows, path, title, crit):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    t = np.array([r["tick"] for r in rows]) / 1e6
    g = lambda name: np.array([r[name] for r in rows], dtype=np.float64)
    fig, ax = plt.subplots(2, 3, figsize=(16, 8.5), sharex=True)
    a = ax[0, 0]
    a.plot(t, g("diversity"), label="distinct sequences")
    a.plot(t, g("shadow_diversity"), label="shadow: distinct sequences", ls="--")
    a.plot(t, g("adaptive"), label="adaptive")
    a.plot(t, g("est_adaptive"), label="adaptive and established")
    a.set_yscale("log"); a.legend(fontsize=8); a.set_title("diversity")
    a = ax[0, 1]
    a.plot(t, g("new_adaptive"), label="new adaptive")
    a.plot(t, g("shadow_new_adaptive"), label="shadow: new adaptive", ls="--")
    a.plot(t, g("est_new_adaptive"), label="new established")
    a.plot(t, g("est_novel"), label="new established with a new operation set")
    a.set_yscale("symlog"); a.legend(fontsize=8); a.set_title("new evolutionary activity per census")
    a = ax[0, 2]
    a.plot(t, g("cum_activity"), label="cumulative activity, adaptive")
    a.plot(t, g("est_cum_activity"), label="cumulative activity, established")
    a.plot(t, g("theta"), label="threshold (shadow percentile)", ls="--")
    a.plot(t, g("mean_activity"), label="mean activity, adaptive")
    a.set_yscale("log"); a.legend(fontsize=8); a.set_title("activity")
    a = ax[1, 0]
    for m in ("est_max_distinct_ops", "est_mean_distinct_ops", "est_max_length", "est_mean_length"):
        a.plot(t, g(m), label=m[4:])
    a.legend(fontsize=8); a.set_xlabel("tick (millions)"); a.set_title("operations of established sequences")
    a = ax[1, 1]
    for m in ("est_max_tasks", "est_mean_tasks", "est_max_trophic", "est_mean_trophic"):
        a.plot(t, g(m), label=m[4:])
    a.legend(fontsize=8); a.set_xlabel("tick (millions)"); a.set_title("tasks and trophic level (established)")
    a = ax[1, 2]
    for m in ("est_share_sense", "est_share_signal", "est_share_move"):
        a.plot(t, g(m), label=m[4:])
    a.plot(t, g("u_op"), label="u (op change per birth)", ls=":")
    a.legend(fontsize=8); a.set_xlabel("tick (millions)"); a.set_title("shares of established carriers")
    for q in (0.25, 0.5, 0.75):
        for b in ax.flat:
            b.axvline(t[0] + q * (t[-1] - t[0]), color="grey", lw=0.5)
    fig.suptitle("%s: all components %s, established components %s" % (os.path.basename(title), crit["all"]["verdict"],
                                                                         crit["established"]["verdict"]))
    fig.tight_layout()
    fig.savefig(path, dpi=110)


if __name__ == "__main__":
    main()
