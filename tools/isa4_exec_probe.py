"""Execution census of 4-byte words on a real ISA with the Unicorn emulator (AArch64, x86-64).

Analogue of the NANO class map: run every 4-byte word from a fixed zero state and classify what it did.
State: all general registers 0, flags 0, stack pointer at 0x8000 inside a zeroed 64 KiB data page
(mapped at address 0), the word at code address 0x100000. The word runs until the PC leaves the 4 bytes or
an instruction limit of 4 is reached (an x86-64 word can hold up to four one-byte instructions).

Every word is first decoded with Capstone. Invalid encodings are UNDEF without being executed (Unicorn's
AArch64 backend reports them as a generic exception, so decoding is what separates them from real traps). An
x86-64 word whose last instruction needs bytes beyond the four is OVERRUN and is not executed either: its
behaviour would depend on the padding, not on the word. Only words that decode to whole instructions run.

Classes (priority order):
  UNDEF       invalid encoding (Capstone decodes nothing / not exactly one AArch64 instruction)
  OVERRUN     x86-64 only: the decoded instructions do not end exactly at byte 4
  EXCEPTION   trap, privileged instruction, system call, breakpoint (Unicorn UC_ERR_EXCEPTION or other error)
  FAULT_MEM   data access outside the mapped page (unmapped or misaligned read or write)
  BRANCH_OUT  control left the word to an address other than the next word (jumps, calls, fetch faults)
  MEM_WRITE   fell through and wrote data memory
  REG         fell through, registers or flags changed
  IDLE        fell through, nothing changed (nop-like)

Every executed word also gets a 32-bit FNV-1a fingerprint of its final registers, flags, PC and memory
writes, so distinct behaviours can be counted later exactly as the NANO state hashes were.

  python tools/isa4_exec_probe.py --arch both --sample 1048576
  python tools/isa4_exec_probe.py --arch aarch64 --range 0 16777216 --procs 24
"""
import argparse
import json
import os
import pathlib
import platform
import random
import struct
import time
from collections import Counter
from multiprocessing import Pool, cpu_count

import capstone as cs
import unicorn as uc
from unicorn import arm64_const as A64
from unicorn import x86_const as X86

CODE = 0x100000
DATA, DATA_SIZE = 0x0, 0x10000
SP = 0x8000
CLASSES = ["UNDEF", "OVERRUN", "EXCEPTION", "FAULT_MEM", "BRANCH_OUT", "MEM_WRITE", "REG", "IDLE"]

ARCH = {
    "aarch64": dict(
        cs=(cs.CS_ARCH_ARM64, cs.CS_MODE_ARM),
        arch=uc.UC_ARCH_ARM64, mode=uc.UC_MODE_ARM,
        regs=[getattr(A64, f"UC_ARM64_REG_X{i}") for i in range(31)] + [A64.UC_ARM64_REG_SP, A64.UC_ARM64_REG_NZCV],
        sp=A64.UC_ARM64_REG_SP, pc=A64.UC_ARM64_REG_PC),
    "x86_64": dict(
        cs=(cs.CS_ARCH_X86, cs.CS_MODE_64),
        arch=uc.UC_ARCH_X86, mode=uc.UC_MODE_64,
        regs=[X86.UC_X86_REG_RAX, X86.UC_X86_REG_RBX, X86.UC_X86_REG_RCX, X86.UC_X86_REG_RDX,
              X86.UC_X86_REG_RSI, X86.UC_X86_REG_RDI, X86.UC_X86_REG_RBP, X86.UC_X86_REG_RSP,
              X86.UC_X86_REG_R8, X86.UC_X86_REG_R9, X86.UC_X86_REG_R10, X86.UC_X86_REG_R11,
              X86.UC_X86_REG_R12, X86.UC_X86_REG_R13, X86.UC_X86_REG_R14, X86.UC_X86_REG_R15,
              X86.UC_X86_REG_EFLAGS],
        sp=X86.UC_X86_REG_RSP, pc=X86.UC_X86_REG_RIP),
}
MEM_ERRORS = {uc.UC_ERR_READ_UNMAPPED, uc.UC_ERR_WRITE_UNMAPPED, uc.UC_ERR_READ_PROT, uc.UC_ERR_WRITE_PROT,
              uc.UC_ERR_READ_UNALIGNED, uc.UC_ERR_WRITE_UNALIGNED}
FETCH_ERRORS = {uc.UC_ERR_FETCH_UNMAPPED, uc.UC_ERR_FETCH_PROT, uc.UC_ERR_FETCH_UNALIGNED}


def fnv(h, v):
    for b in v.to_bytes(8, "little", signed=False):
        h = ((h ^ b) * 16777619) & 0xFFFFFFFF
    return h


class Probe:
    def __init__(self, arch):
        a = ARCH[arch]
        self.a = a
        self.mu = uc.Uc(a["arch"], a["mode"])
        self.mu.mem_map(DATA, DATA_SIZE, uc.UC_PROT_READ | uc.UC_PROT_WRITE)
        self.mu.mem_map(CODE, 0x1000, uc.UC_PROT_ALL)
        self.writes = []
        self.mu.hook_add(uc.UC_HOOK_MEM_WRITE, self._on_write)
        self.zero_page = bytes(DATA_SIZE)
        self.writes = []
        self.reset()
        self.zero_state = [self.mu.reg_read(r) for r in a["regs"]]  # measured: x86-64 EFLAGS bit 1 is always set
        self.has_cache_ctl = hasattr(self.mu, "ctl_remove_cache")
        self.md = cs.Cs(*a["cs"])
        self.md.detail = False
        self.x86 = arch == "x86_64"

    def decode(self, code):
        """UNDEF / OVERRUN / None (whole word, run it)."""
        n = used = 0
        for _addr, size, _m, _ops in self.md.disasm_lite(code, 0):
            n += 1
            used += size
        if n == 0 or (not self.x86 and n != 1):
            return "UNDEF"
        if self.x86 and used != 4:
            return "OVERRUN"
        return None

    def _on_write(self, mu, access, address, size, value, user):
        self.writes.append((address, size, value & ((1 << (8 * size)) - 1)))

    def reset(self):
        for r in self.a["regs"]:
            self.mu.reg_write(r, 0)
        self.mu.reg_write(self.a["sp"], SP)
        if self.writes:
            self.mu.mem_write(DATA, self.zero_page)
        self.writes = []

    def run(self, word):
        mu = self.mu
        code = struct.pack("<I", word)
        pre = self.decode(code)
        if pre is not None:
            return pre, 0, None
        self.reset()
        mu.mem_write(CODE, code + bytes(12))
        if self.has_cache_ctl:
            mu.ctl_remove_cache(CODE, CODE + 16)
        err = None
        try:
            mu.emu_start(CODE, CODE + 4, count=4)
        except uc.UcError as e:
            err = e.errno
        pc = mu.reg_read(self.a["pc"])
        regs = [mu.reg_read(r) for r in self.a["regs"]]
        if err == uc.UC_ERR_INSN_INVALID:
            cls = "UNDEF"  # Capstone accepted it, Unicorn did not: decoder disagreement, counted in unicorn_errors
        elif err == uc.UC_ERR_EXCEPTION:
            cls = "EXCEPTION"
        elif err in MEM_ERRORS:
            cls = "FAULT_MEM"
        elif err in FETCH_ERRORS or (err is None and pc != CODE + 4):
            cls = "BRANCH_OUT"
        elif err is not None:
            cls = "EXCEPTION"
        elif self.writes:
            cls = "MEM_WRITE"
        elif regs != self.zero_state:
            cls = "REG"
        else:
            cls = "IDLE"
        h = 2166136261
        for v in regs:
            h = fnv(h, v)
        h = fnv(h, pc)
        for w in self.writes:
            for v in w:
                h = fnv(h, v)
        return cls, h, err


def work(job):
    arch, kind, p0, p1 = job
    if kind == "sample":
        rng = random.Random(p0)
        words = [rng.getrandbits(32) for _ in range(p1)]
    else:
        words = range(p0, p0 + p1)
    pr = Probe(arch)
    c = Counter()
    hashes = set()
    errs = Counter()
    for w in words:
        cls, h, err = pr.run(w)
        c[cls] += 1
        if cls not in ("UNDEF", "OVERRUN"):
            hashes.add(h)
        if err is not None:
            errs[err] += 1
    return c, hashes, errs


ERR_NAMES = {getattr(uc, n): n for n in dir(uc) if n.startswith("UC_ERR_")}


def errname(code):
    return ERR_NAMES.get(code, str(code))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", choices=list(ARCH) + ["both"], default="both")
    ap.add_argument("--sample", type=int)
    ap.add_argument("--range", nargs=2, type=int, metavar=("START", "COUNT"))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--procs", type=int, default=cpu_count())
    ap.add_argument("--chunk", type=int, default=1 << 14)
    ap.add_argument("--out", default="results/isa4_exec")
    args = ap.parse_args()
    if not args.sample and not args.range:
        ap.error("--sample N or --range START COUNT")
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for arch in (list(ARCH) if args.arch == "both" else [args.arch]):
        if args.sample:
            total = args.sample
            jobs = [(arch, "sample", args.seed * 1_000_003 + i, min(args.chunk, total - i * args.chunk))
                    for i in range((total + args.chunk - 1) // args.chunk)]
        else:
            start, total = args.range
            jobs = [(arch, "range", start + i * args.chunk, min(args.chunk, total - i * args.chunk))
                    for i in range((total + args.chunk - 1) // args.chunk)]
        t0 = time.perf_counter()
        c, hashes, errs = Counter(), set(), Counter()
        with Pool(args.procs) as pool:
            for ci, hi, ei in pool.imap_unordered(work, jobs):
                c += ci
                hashes |= hi
                errs += ei
        dt = time.perf_counter() - t0
        assert sum(c.values()) == total
        res = {
            "arch": arch,
            "mode": "sample" if args.sample else "range",
            "seed": args.seed if args.sample else None,
            "range": None if args.sample else args.range,
            "words": total,
            "classes": {k: c.get(k, 0) for k in CLASSES},
            "percent": {k: round(100 * c.get(k, 0) / total, 4) for k in CLASSES},
            "distinct_state_fingerprints": len(hashes),
            "fingerprints_note": "over executed words only (not UNDEF / OVERRUN)",
            "unicorn_errors": {errname(k): v for k, v in errs.items()},
            "seconds": round(dt, 2),
            "words_per_second": round(total / dt),
            "processes": args.procs,
            "unicorn": uc.__version__, "capstone": cs.__version__,
            "state": "regs 0, flags 0, SP 0x8000, data page 0..0xFFFF zeroed, code at 0x100000, limit 4 instructions",
            "host": f"{platform.node()} {platform.machine()} {os.cpu_count()} logical CPUs, Python {platform.python_version()}",
        }
        name = f"{arch}_{'sample' if args.sample else 'range'}_{total}.json"
        (out / name).write_text(json.dumps(res, indent=1))
        print(f"{arch}: {total:,} words in {dt:.1f} s ({total / dt:,.0f}/s); "
              + ", ".join(f"{k} {res['percent'][k]}%" for k in CLASSES)
              + f"; {len(hashes):,} distinct fingerprints -> {out / name}")


if __name__ == "__main__":
    main()
