"""DNA-3 slot maps (docs/17): spatial statistics, pictures and videos from the --slot-map files of gpu/dna_world.cu.

The engine writes, at every census, one byte per slot to RUN.map.<mode> (modes gene, clade, energy, plant) and, on
day 2, the terrain once to RUN.map.terrain (0 sea, 1 land, 2 firmament). Map bytes: 0 dead, 255 dead but claimed as
a child, 1 .. 254 live: gene = 1 + gene hash mod 253, clade = 1 + splitmix64(founder slot) mod 253 (the founder is
the depth-0 ancestor: a seed or a spontaneous organism), energy = 1 + E * 253 / (emax + 1), plant = 2 for a gene
with a HARVEST codon, 1 otherwise, traits = 1 + HARVEST + 2 SENSE + 4 WAIT (white: a plant that senses and waits). Slot s is at column s mod W, row s div W.

  python tools/dna3_maps.py stats RUN --W 640 [--mode clade] [--every 1] > RUN.spatial.json
  python tools/dna3_maps.py png RUN --W 640 --mode clade --frame 199 --out pic.png [--scale 1] [--crop x0,y0,w,h]
  python tools/dna3_maps.py video RUN --W 640 --mode plant --out run.mp4 [--step 1] [--fps 30] [--crop ...]

stats: per frame, the live count; for clade and gene the spatial assortment (the fraction of live neighbour pairs,
right and down neighbours on the torus, that carry the same byte, divided by the fraction expected if the same live
population were placed at random, sum of squared byte frequencies; about 1 for random placement, larger for
patches), the number of 4-connected same-byte patches, their mean and largest size and the number of distinct bytes;
for plant the harvester fraction overall, on land and on sea. Byte collisions (two clades that hash to the same of
253 values) merge patches and raise both fractions; with thousands of clades the assortment is a lower bound of the
patchiness, not an exact clade statistic.
"""
import argparse
import gzip
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

try:
    from scipy import ndimage
except ImportError:            # patches are then not counted
    ndimage = None

TER_COL = np.array([[12, 24, 56], [52, 40, 22], [96, 96, 96]], dtype=np.uint8)   # dead: sea, land, firmament


def load_frames(path, S):
    size = os.path.getsize(path)
    if size % S:
        raise SystemExit("%s: size %d is not a multiple of S = %d" % (path, size, S))
    return np.memmap(path, dtype=np.uint8, mode="r", shape=(size // S, S))


def census_ticks(run):
    for p in (run + ".txt.gz", run + ".txt"):
        if os.path.exists(p):
            op = gzip.open if p.endswith(".gz") else open
            with op(p, "rt") as f:
                return [json.loads(l[7:])["tick"] for l in f if l.startswith("census ")]
    return None


def palette_hash():
    rng = np.random.default_rng(12345)
    pal = np.zeros((256, 3), dtype=np.uint8)
    h = rng.permutation(253) / 253.0
    for b in range(1, 254 + 1):
        hue = h[(b - 1) % 253]
        x = hue * 6.0
        i = int(x) % 6
        f = x - int(x)
        v, s = 0.95, 0.75
        p_, q_, t_ = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
        r, g, bl = [(v, t_, p_), (q_, v, p_), (p_, v, t_), (p_, q_, v), (t_, p_, v), (v, p_, q_)][i]
        pal[b] = (int(255 * r), int(255 * g), int(255 * bl))
    pal[255] = (70, 70, 70)
    return pal


def palette(mode):
    if mode in ("clade", "gene"):
        return palette_hash()
    pal = np.zeros((256, 3), dtype=np.uint8)
    pal[255] = (70, 70, 70)
    if mode == "plant":
        pal[1] = (200, 90, 220)    # no HARVEST codon
        pal[2] = (70, 230, 70)     # plant
    elif mode == "traits":                          # 1 + HARVEST + 2 SENSE + 4 WAIT
        pal[1] = (200, 90, 220)    # none
        pal[2] = (70, 230, 70)     # plant
        pal[3] = (90, 160, 255)    # SENSE only
        pal[4] = (40, 200, 255)    # plant + SENSE
        pal[5] = (255, 140, 0)     # WAIT only
        pal[6] = (255, 230, 0)     # plant + WAIT
        pal[7] = (255, 60, 60)     # SENSE + WAIT
        pal[8] = (255, 255, 255)   # plant + SENSE + WAIT: a sleeper plant
    elif mode == "energy":
        for b in range(1, 255):
            f = (b - 1) / 252.0
            pal[b] = (int(60 + 195 * f), int(30 + 225 * f ** 0.8), int(20 + 60 * f))
    return pal


def render(frame, W, mode, terrain, crop=None, scale=1, shrink=1):
    R = frame.size // W
    img = palette(mode)[frame].reshape(R, W, 3)
    if terrain is not None:
        bg = TER_COL[np.minimum(terrain, 2)].reshape(R, W, 3)
        dead = (frame == 0).reshape(R, W)
        img[dead] = bg[dead]
    else:
        img[(frame == 0).reshape(R, W)] = (0, 0, 0)
    if crop:
        x0, y0, w, h = crop
        img = img[y0:y0 + h, x0:x0 + w]
    if shrink > 1:                                   # every shrink-th slot in both directions (an overview)
        img = img[::shrink, ::shrink]
    im = Image.fromarray(np.ascontiguousarray(img))
    if scale != 1:
        im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    return im


def frame_stats(fr, W, mode, terrain):
    R = fr.size // W
    a = np.asarray(fr).reshape(R, W)
    live = (a >= 1) & (a <= 254)
    n = int(live.sum())
    out = {"live": n}
    if n == 0:
        return out
    if mode in ("clade", "gene"):
        same = pairs = 0
        for b in (np.roll(a, -1, axis=1), np.roll(a, -1, axis=0)):
            both = live & (b >= 1) & (b <= 254)
            pairs += int(both.sum())
            same += int((both & (a == b)).sum())
        counts = np.bincount(a[live], minlength=256).astype(np.float64)
        f_exp = float(((counts / n) ** 2).sum())
        out.update(pairs=pairs, same_frac=same / pairs if pairs else None, expected=f_exp,
                   assortment=(same / pairs) / f_exp if pairs and f_exp else None, distinct=int((counts > 0).sum()))
        if ndimage is not None:
            npatch, largest = 0, 0
            for b in np.nonzero(counts)[0]:
                lab, k = ndimage.label(a == b)
                npatch += k
                if k:
                    largest = max(largest, int(np.bincount(lab.ravel())[1:].max()))
            out.update(patches=npatch, mean_patch=n / npatch if npatch else None, largest_patch=largest)
    elif mode == "plant":
        harv = a == 2
        out["harvesters"] = int(harv.sum())
        if terrain is not None:
            t = terrain.reshape(R, W)
            for name, code in (("land", 1), ("sea", 0)):
                m = live & (t == code)
                out["live_" + name] = int(m.sum())
                out["harv_" + name] = int((harv & (t == code)).sum())
    elif mode == "energy":
        out["mean_byte"] = float(a[live].mean())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("stats", "png", "video"))
    ap.add_argument("run", help="run prefix: RUN.map.<mode>, RUN.map.terrain, RUN.txt.gz")
    ap.add_argument("--W", type=int, required=True)
    ap.add_argument("--S", type=int, default=0, help="slots (default: the terrain file's size, else W * W)")
    ap.add_argument("--mode", default="clade")
    ap.add_argument("--frame", type=int, default=-1)
    ap.add_argument("--frames", default=None, help="stats: comma list of frames (default: all, see --every); video: a:b range")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--scale", type=int, default=1)
    ap.add_argument("--crop", default=None)
    ap.add_argument("--crf", type=int, default=28)
    ap.add_argument("--shrink", type=int, default=1, help="png/video: keep every k-th slot in both directions")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    tpath = a.run + ".map.terrain"
    terrain = np.fromfile(tpath, dtype=np.uint8) if os.path.exists(tpath) else None
    S = a.S or (terrain.size if terrain is not None else a.W * a.W)
    frames = load_frames(a.run + ".map." + a.mode if a.mode != "depth" else a.run + ".map", S)
    ticks = census_ticks(a.run)
    crop = tuple(int(x) for x in a.crop.split(",")) if a.crop else None
    if a.cmd == "stats":
        idx = [int(x) for x in a.frames.split(",")] if a.frames else list(range(0, len(frames), a.every))
        if len(frames) - 1 not in idx and not a.frames:
            idx.append(len(frames) - 1)
        res = []
        for i in idx:
            st = frame_stats(frames[i], a.W, a.mode, terrain)
            st = dict(frame=i, tick=ticks[i] if ticks and i < len(ticks) else None, **st)
            res.append(st)
            print(json.dumps(st), file=sys.stderr, flush=True)
        out = {"run": os.path.basename(a.run), "mode": a.mode, "W": a.W, "S": S, "frames": len(frames),
               "land": int((terrain == 1).sum()) if terrain is not None else None,
               "sea": int((terrain == 0).sum()) if terrain is not None else None, "series": res}
        text = json.dumps(out, indent=1)
        if a.out:
            with open(a.out, "w") as f:
                f.write(text + "\n")
        else:
            print(text)
    elif a.cmd == "png":
        i = a.frame if a.frame >= 0 else len(frames) + a.frame
        render(frames[i], a.W, a.mode, terrain, crop, a.scale, a.shrink).save(a.out)
        print("frame %d (tick %s) -> %s" % (i, ticks[i] if ticks else "?", a.out))
    else:
        with tempfile.TemporaryDirectory() as d:
            n = 0
            lo, hi = 0, len(frames)
            if a.frames and ":" in a.frames:            # video: a frame range a:b
                lo, hi = (int(x) for x in a.frames.split(":"))
            for i in range(lo, min(hi, len(frames)), a.step):
                render(frames[i], a.W, a.mode, terrain, crop, a.scale, a.shrink).save(os.path.join(d, "f%06d.png" % n))
                n += 1
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", os.path.join(d, "f%06d.png"),
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(a.crf), "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", a.out],
                           check=True)
        print("%d frames -> %s" % (n, a.out))


if __name__ == "__main__":
    main()
