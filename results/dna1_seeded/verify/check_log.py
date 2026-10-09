"""Check dna_world_v2's --log-births (filter all) against the reference births file: same rows (tick, parent,
child, hashes, faithful), depth = parent's depth + 1 (seeds 0, spontaneous 0), parent_age consistent with the
parent's birth tick (age = tick - birth_tick: a child born at tick 53 has age 27 at tick 80, spec 13 'Age'; seeds have age t at tick t), copied >= 6."""
import sys
ref, log = sys.argv[1], sys.argv[2]
rrows = [l.rstrip("\n").split("\t") for l in open(ref) if not l.startswith("#")]
lrows = [l.rstrip("\n").split("\t") for l in open(log) if not l.startswith("#")]
assert len(rrows) == len(lrows), (len(rrows), len(lrows))
depth = {}      # slot -> depth of the processor born there (latest)
born = {}       # slot -> tick of birth
bad = 0
for r, l in zip(rrows, lrows):
    assert r == l[:6], (r, l)
    t, p, c = int(l[0]), int(l[1]), int(l[2])
    d, age, cop = int(l[6]), int(l[7]), int(l[8])
    if p < 0:
        ok = d == 0 and age == -1 and cop == -1
    else:
        pd = depth.get(p, 0)                     # a parent not in the log is a seed (depth 0)
        pb = born.get(p, 0)                      # seeds: age t at tick t (age 0 at the start of tick 0)
        ok = d == pd + 1 and age == t - pb and cop >= 6
    if not ok:
        bad += 1
        if bad < 5: print("bad row", l)
    depth[c] = d; born[c] = t
print("rows", len(lrows), "divide", sum(1 for l in lrows if int(l[1]) >= 0), "bad", bad, "max_depth", max([int(l[6]) for l in lrows] + [0]))
