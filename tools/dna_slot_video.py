"""Render the --slot-map frames of gpu/dna_world.cu (docs/12) into a video: one pixel per slot, one frame per census.

  python tools/dna_slot_video.py RUN.map --S 390624 [--census RUN.txt] [--width 625] [--scale 2] [--fps 30]
                                 [--frames 0:2000] [--out RUN.mp4]

The map file holds S bytes per census: 0 = dead slot, 255 = dead but claimed as a child (being written), 1 .. 254 =
live processor of lineage depth byte - 1 (254 = depth >= 253). Colours: dead black, claimed dark grey, depth 0 (the
seeded organisms and spontaneous organisms from the inflow) white, depth 1, 2, 3, ... a warm-to-cold ramp (red,
orange, yellow, green, cyan, blue, violet, then grey-blue for deeper), so a lineage's generations are visible as
colours. Slots are laid out row by row (slot s at column s mod width, row s div width), so the two neighbours a
copier needs (s + 1 blank, s + 2 nonzero) are the next two pixels to the right. A caption gives the tick and, when
the census file is given, the live count, the faithful births so far and the depth histogram. Needs numpy, Pillow
and ffmpeg (libx264). Frames are written to a temporary directory and encoded with -pix_fmt yuv420p.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw


def depth_palette():
    pal = np.zeros((256, 3), dtype=np.uint8)
    pal[0] = (0, 0, 0)                     # dead
    pal[255] = (70, 70, 70)                # claimed
    pal[1] = (255, 255, 255)               # depth 0
    ramp = [(255, 40, 40), (255, 140, 0), (255, 230, 0), (60, 220, 60), (0, 210, 210), (60, 100, 255), (180, 60, 255),
            (255, 80, 200), (200, 200, 120), (120, 200, 200)]
    for i, c in enumerate(ramp):
        pal[2 + i] = c                     # depth 1 .. 10
    for b in range(12, 255):
        f = (b - 12) / 243.0
        pal[b] = (int(120 + 100 * f), int(140 - 60 * f), int(220 - 80 * f))
    return pal


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("map")
    ap.add_argument("--S", type=int, required=True, help="slots per frame")
    ap.add_argument("--census", default=None, help="the run's stdout with the census lines (same order as the frames)")
    ap.add_argument("--width", type=int, default=0, help="pixels per row (default: about sqrt(S))")
    ap.add_argument("--scale", type=int, default=1)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--frames", default=None, help="a:b frame range (default all)")
    ap.add_argument("--step", type=int, default=1, help="use every k-th frame")
    ap.add_argument("--out", default=None)
    ap.add_argument("--keep-frames", default=None, help="directory to keep the PNG frames in")
    a = ap.parse_args()
    S = a.S
    width = a.width or int(np.ceil(np.sqrt(S)))
    rows = (S + width - 1) // width
    size = os.path.getsize(a.map)
    nframes = size // S
    if size % S:
        print("warning: map size %d is not a multiple of S = %d" % (size, S), file=sys.stderr)
    f0, f1 = 0, nframes
    if a.frames:
        p = a.frames.split(":")
        f0 = int(p[0]) if p[0] else 0
        f1 = min(nframes, int(p[1])) if len(p) > 1 and p[1] else nframes
    census = []
    if a.census:
        with open(a.census) as f:
            for line in f:
                if line.startswith("census "):
                    census.append(json.loads(line[7:]))
    pal = depth_palette()
    frames_dir = a.keep_frames or tempfile.mkdtemp(prefix="dna_slot_video_")
    os.makedirs(frames_dir, exist_ok=True)
    out = a.out or os.path.splitext(a.map)[0] + ".mp4"
    cap_h = 40
    n_written = 0
    with open(a.map, "rb") as f:
        f.seek(f0 * S)
        for fi in range(f0, f1):
            raw = f.read(S)
            if len(raw) < S:
                break
            if (fi - f0) % a.step:
                continue
            m = np.frombuffer(raw, dtype=np.uint8)
            grid = np.zeros(rows * width, dtype=np.uint8)
            grid[:S] = m
            img = pal[grid.reshape(rows, width)]
            im = Image.fromarray(img, "RGB")
            if a.scale > 1:
                im = im.resize((width * a.scale, rows * a.scale), Image.NEAREST)
            W, H = im.size
            W2 = W + (W & 1)
            H2 = H + cap_h + ((H + cap_h) & 1)       # libx264 with yuv420p needs even dimensions
            canvas = Image.new("RGB", (W2, H2), (20, 20, 20))
            canvas.paste(im, (0, cap_h))
            d = ImageDraw.Draw(canvas)
            if fi < len(census):
                c = census[fi]
                dh = ", ".join("%d:%d" % (k, v) for k, v in c["depth_hist"][:8])
                txt = "tick %d  live %d  faithful births %d  DIVIDE births %d  depth {%s}" % (
                    c["tick"], c["live"], c["faithful_total"], c["births_total"] - c["spontaneous_total"], dh)
            else:
                txt = "frame %d" % fi
            d.text((6, 4), txt, fill=(230, 230, 230))
            d.text((6, 20), "white: depth 0 (seeded)   red/orange/yellow/green/cyan/blue: depth 1..6   grey: claimed child slot   black: dead",
                   fill=(160, 160, 160))
            canvas.save(os.path.join(frames_dir, "f%06d.png" % n_written))
            n_written += 1
    if n_written == 0:
        print("no frames", file=sys.stderr)
        return 1
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", os.path.join(frames_dir, "f%06d.png"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out]
    subprocess.run(cmd, check=True)
    print("wrote %s: %d frames (%d x %d slots, scale %d), %.1f s at %d fps" % (out, n_written, width, rows, a.scale, n_written / a.fps, a.fps),
          file=sys.stderr)
    if not a.keep_frames:
        shutil.rmtree(frames_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
