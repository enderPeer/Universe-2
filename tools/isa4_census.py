"""Decode census of 4-byte words on a real ISA (AArch64, x86-64) with Capstone.

The NANO maps asked "what does every 5- or 6-byte program do?". The same question for a real CPU starts
with "which of the 2^32 four-byte words are instructions at all?". AArch64 has fixed 4-byte instructions,
so one word is exactly one instruction slot; so do ARM32, RISC-V without the C extension, MIPS, PowerPC and
SPARC. x86-64 (and RISC-V with compressed instructions) is variable length, so one 4-byte word holds 0 to 4
whole instructions (and usually a trailing fragment).

  python tools/isa4_census.py --arch aarch64 --sample 16777216          # uniform random sample
  python tools/isa4_census.py --arch x86_64 --range 0 4294967296         # exhaustive (all 2^32 words)
  python tools/isa4_census.py --arch both --sample 1000000 --out results/isa4_census

Outputs JSON: counts, the mnemonic histogram, throughput. Decoding only; nothing is executed.
"""
import argparse
import json
import os
import pathlib
import platform
import random
import struct
import sys
import time
from collections import Counter
from multiprocessing import Pool, cpu_count

import capstone as cs

# name: (capstone arch, mode, fixed 4-byte instructions?)
ARCHES = {
    "aarch64": (cs.CS_ARCH_ARM64, cs.CS_MODE_ARM, True),
    "x86_64": (cs.CS_ARCH_X86, cs.CS_MODE_64, False),
    "arm32": (cs.CS_ARCH_ARM, cs.CS_MODE_ARM, True),
    "riscv64": (cs.CS_ARCH_RISCV, cs.CS_MODE_RISCV64, True),                        # without the C extension
    "riscv64c": (cs.CS_ARCH_RISCV, cs.CS_MODE_RISCV64 | cs.CS_MODE_RISCVC, False),  # 2- and 4-byte instructions
    "mips64": (cs.CS_ARCH_MIPS, cs.CS_MODE_MIPS64 | cs.CS_MODE_LITTLE_ENDIAN, True),
    "ppc64": (cs.CS_ARCH_PPC, cs.CS_MODE_64 | cs.CS_MODE_LITTLE_ENDIAN, True),
    "sparc": (cs.CS_ARCH_SPARC, cs.CS_MODE_BIG_ENDIAN, True),
}


def _words_sample(seed, count):
    rng = random.Random(seed)
    return [rng.getrandbits(32) for _ in range(count)]


def _words_range(start, count):
    return range(start, start + count)


def census(job):
    arch, kind, p0, p1 = job
    words = _words_sample(p0, p1) if kind == "sample" else _words_range(p0, p1)
    a, m, fixed = ARCHES[arch]
    md = cs.Cs(a, m)
    md.detail = False
    c = Counter()
    mnem = Counter()
    nins_hist = Counter()
    for w in words:
        b = struct.pack("<I", w)
        n = used = 0
        first = None
        for _addr, size, m, _ops in md.disasm_lite(b, 0):
            if first is None:
                first = m
            n += 1
            used += size
        c["words"] += 1
        if fixed:
            if n == 1 and used == 4:
                c["valid"] += 1
                mnem[first] += 1
            else:
                c["undefined"] += 1
        else:
            nins_hist[n] += 1
            if n == 0:
                c["no_instruction"] += 1
            elif used == 4:
                c["whole"] += 1
            else:
                c["partial"] += 1
            if first is not None:
                mnem[first] += 1
    return c, mnem, nins_hist


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", choices=list(ARCHES) + ["both", "all"], default="both",
                    help="both = aarch64 + x86_64; all = every architecture")
    ap.add_argument("--sample", type=int, help="number of uniformly random words")
    ap.add_argument("--range", nargs=2, type=int, metavar=("START", "COUNT"), help="exhaustive word range")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--procs", type=int, default=cpu_count())
    ap.add_argument("--chunk", type=int, default=1 << 18)
    ap.add_argument("--out", default="results/isa4_census")
    args = ap.parse_args()
    if not args.sample and not args.range:
        ap.error("--sample N or --range START COUNT")
    arches = list(ARCHES) if args.arch == "all" else ["aarch64", "x86_64"] if args.arch == "both" else [args.arch]
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for arch in arches:
        if args.sample:
            total = args.sample
            jobs = [(arch, "sample", args.seed * 1_000_003 + i, min(args.chunk, total - i * args.chunk))
                    for i in range((total + args.chunk - 1) // args.chunk)]
        else:
            start, total = args.range
            jobs = [(arch, "range", start + i * args.chunk, min(args.chunk, total - i * args.chunk))
                    for i in range((total + args.chunk - 1) // args.chunk)]
        t0 = time.perf_counter()
        c, mnem, hist = Counter(), Counter(), Counter()
        with Pool(args.procs) as pool:
            for ci, mi, hi in pool.imap_unordered(census, jobs):
                c += ci
                mnem += mi
                hist += hi
        dt = time.perf_counter() - t0
        assert c["words"] == total, (c["words"], total)
        res = {
            "arch": arch,
            "mode": "sample" if args.sample else "range",
            "seed": args.seed if args.sample else None,
            "range": None if args.sample else args.range,
            "words": total,
            "counts": {k: v for k, v in sorted(c.items()) if k != "words"},
            "percent": {k: round(100 * v / total, 4) for k, v in sorted(c.items()) if k != "words"},
            "instructions_per_word": {str(k): v for k, v in sorted(hist.items())} if hist else None,
            "distinct_mnemonics": len(mnem),
            "top_mnemonics": mnem.most_common(40),
            "seconds": round(dt, 2),
            "words_per_second": round(total / dt),
            "processes": args.procs,
            "capstone": cs.__version__,
            "host": f"{platform.node()} {platform.machine()} {os.cpu_count()} logical CPUs, Python {platform.python_version()}",
        }
        name = f"{arch}_{'sample' if args.sample else 'range'}_{total}.json"
        (out / name).write_text(json.dumps(res, indent=1))
        print(f"{arch}: {total:,} words in {dt:.1f} s ({total / dt:,.0f}/s): " + ", ".join(f"{k} {res['percent'][k]}%" for k in res["percent"]) + f"; {len(mnem)} distinct mnemonics -> {out / name}")


if __name__ == "__main__":
    main()
