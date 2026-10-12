"""Which programs earn which ladder tasks (docs/17 section 18): the most common exact gene spellings of a checkpoint,
each run alone in a small reference world, with the ladder's reward counts per task.

  python tools/dna3_task_profile.py RUN.ckpt [--top 1500] [--ticks 1500] [--with OUT] [--json out.json] [--workers 24]

For each of the --top most common spellings (codons 0 .. first STOP) among the living processors of the checkpoint
(version 3 or 4; strands and live flags), a fresh sim/dna_ref.py world is built: days 1 to 6 with --ladder, shading,
T = 0, 16 x 16 slots of 256 cells, no mutation, no inflow, no damage, no task pools (so every correct streak pays the
full bonus), one seed of the spelling in the sea; after --ticks ticks the ladder's task counts (rewarded answers per
task, of the seed and its descendants, all with the same gene) and the census are read. A task counts as "earned" by a
program when it has at least 1 % of the program's rewarded answers. The world is not D8: no neighbours of other
kinds, no pools, no competition; the profile shows what the program computes, not what it earns in the big world.
"""
import argparse
import collections
import json
import multiprocessing as mp
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sim"))
sys.path.insert(0, HERE)
import dna_ref as R  # noqa: E402
import dna3_ckpt_genes as K  # noqa: E402

TASK_NAMES = [t[0] for t in R.LADDER_TASKS] + ["social"]
TASK_RANKS = [t[3] for t in R.LADDER_TASKS] + [R.LADDER_SOCIAL_RANK]
TABLE = K.table_for([1, 2, 3, 4, 5, 6], False)


def spellings(path, top, with_op):
    table = np.array(TABLE, dtype=np.int16)
    with open(path, "rb") as f:
        f.read(8)
        ver, N, G, S, seed, flags, t_next = struct.unpack("<7Q", f.read(56))
        nb = (N + 3) // 4
        fw = np.frombuffer(f.read(nb), dtype=np.uint8)
        f.seek(nb, 1)
        live = np.frombuffer(f.read(S), dtype=np.uint8)
    maxc = G // 3
    slots = np.nonzero(live)[0]
    cnt = collections.Counter()
    want = R.OP_INDEX[with_op] if with_op else None
    for lo in range(0, len(slots), 100000):
        sl = slots[lo:lo + 100000]
        idx = sl[:, None] * G + np.arange(3 * maxc)[None, :]
        c = (fw[idx >> 2] >> ((idx & 3) * 2).astype(np.uint8)) & 3
        cod = (c[:, 0::3] * 16 + c[:, 1::3] * 4 + c[:, 2::3]).astype(np.uint8)
        ops = table[cod]
        isstop = ops == R.STOP
        first = np.where(isstop.any(1), isstop.argmax(1), maxc)
        for row, n in zip(cod, first):
            g = bytes(row[:n])
            cnt[g] += 1
    out = []
    for g, n in cnt.most_common():
        ops = [int(table[x]) for x in g]
        if want is not None and want not in ops[1:]:
            continue
        out.append((g, n))
        if len(out) >= top:
            break
    return t_next, len(slots), out


def profile(args):
    g, ticks = args
    genome = [int(x) for x in g]
    try:
        w = R.World(16 * 16 * 256, 256, T=0, mu=0.0, D=0, F=0, seed=11, genome=genome, n_seed=1, days=(1, 2, 3, 4, 5, 6),
                    shade=True, W=16, land_cell=8, seed_terrain="sea", ladder=True, pool_cell=0)
        for _ in range(ticks):
            w.tick()
            if not w.live:
                break
        return list(w.task_count), len(w.live), w.births
    except Exception as e:                  # a gene that cannot be seeded (e.g. too long)
        return None, 0, str(e)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt")
    ap.add_argument("--top", type=int, default=1500)
    ap.add_argument("--ticks", type=int, default=1500)
    ap.add_argument("--with", dest="with_op", default="OUT")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    tick, nlive, genes = spellings(a.ckpt, a.top, a.with_op)
    covered = sum(n for _, n in genes)
    print("checkpoint tick %d, %d live; %d spellings with %s cover %d organisms" % (tick, nlive, len(genes), a.with_op, covered),
          flush=True)
    with mp.Pool(a.workers) as pool:
        res = pool.map(profile, [(g, a.ticks) for g, _ in genes], chunksize=4)
    rows = []
    per_task = collections.Counter()
    for (g, n), (tc, live, births) in zip(genes, res):
        if tc is None:
            continue
        total = sum(tc)
        energy = [tc[k] * (8 << TASK_RANKS[k]) for k in range(len(tc))]
        earned = [k for k in range(len(tc)) if total and tc[k] >= 0.01 * total]
        for k in earned:
            per_task[k] += n
        rows.append({"carriers": n, "codons": " ".join(R.codon_text(x) for x in g),
                     "ops": " ".join(R.OP_NAMES[TABLE[x]] for x in g),
                     "rewards": tc, "energy": energy, "total": total, "earned": [TASK_NAMES[k] for k in earned],
                     "bonus_ranks": [TASK_RANKS[k] for k in earned], "live_end": live, "births": births})
    print("organisms (of the profiled ones) whose program earns each task:")
    for k in range(len(TASK_NAMES)):
        print("  %-8s rank %d  %10d" % (TASK_NAMES[k], TASK_RANKS[k], per_task[k]))
    multi = collections.Counter()
    for r in rows:
        multi[len(r["earned"])] += r["carriers"]
    print("organisms by number of tasks their program earns:", dict(sorted(multi.items())))
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"tick": tick, "live": nlive, "profiled": len(rows), "covered": covered,
                       "per_task": {TASK_NAMES[k]: per_task[k] for k in range(len(TASK_NAMES))},
                       "by_number_of_tasks": dict(multi), "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
