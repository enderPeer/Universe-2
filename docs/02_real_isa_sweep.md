# Can we sweep all 4-byte combinations of a real instruction set?

Yes, and it is cheap compared with the NANO maps. 2^32 = 4,294,967,296 words is the size of NANO's 4-byte space,
which the cluster finished in 0.37 s. A real CPU is slower to emulate than NANO by three to four orders of
magnitude, but 2^32 is still hours on CPUs we already have. This note fixes the definitions, gives the measured
speeds, and records the first results. Everything here runs on the workstation with Python, Capstone and Unicorn;
no compiler, GPU or cluster is needed for the first pass.

## 1. "Microsoft or Linux" is not the instruction set

Windows and Linux PCs run the same **x86-64** instruction set (Intel and AMD); Windows-on-ARM, Apple silicon and
most phones run **AArch64** (ARMv8-A 64-bit, the manual downloaded on 2026-10-06 describes it). The operating system
only decides what happens on a system call or a fault. A sweep of instruction words is a sweep of the ISA, and the
OS enters only as the sandbox we run it in. We use the Unicorn emulator (QEMU's CPU core without devices or an OS),
so a word that would call the OS is simply recorded as an exception.

## 2. What "4 bytes" means on each ISA

| | AArch64 | x86-64 |
|---|---|---|
| instruction length | always 4 bytes | 1 to 15 bytes |
| a 4-byte word is | exactly one instruction slot; valid or undefined | 0 to 4 whole instructions, usually followed by a fragment |
| the space 2^32 | *the complete single-instruction space*: every instruction the architecture has is in it exactly once | a slice of the program space: all programs of exactly 4 bytes |

For x86-64 a word whose last instruction needs more bytes than the four is classified **OVERRUN** and is not
executed, because its behaviour would be a property of whatever bytes follow, not of the word. This mirrors NANO,
where the program length L defines the universe.

## 3. Two censuses

**Decode census** ([`tools/isa4_census.py`](../tools/isa4_census.py), Capstone 5.0.9): is the word an instruction,
and which one? Exhaustive over 2^32 is a few hundred million decodes per minute on the workstation.

**Execution census** ([`tools/isa4_exec_probe.py`](../tools/isa4_exec_probe.py), Unicorn 2 + Capstone): run every
decodable word from one fixed state, NANO-style. State: all general registers 0, flags at their reset value, SP =
0x8000 inside a zeroed 64 KiB page mapped at address 0 (so loads and stores through zero registers hit real memory),
code at 0x100000, at most 4 instructions, stop when the PC leaves the word. Classes by priority:

| class | meaning | NANO analogue |
|---|---|---|
| UNDEF | not an instruction | invalid byte (NANO has none) |
| OVERRUN | x86-64: the instructions do not end at byte 4 | a program that depends on bytes outside L |
| EXCEPTION | trap, privileged, system call, breakpoint | HALT |
| FAULT_MEM | load or store outside the mapped 64 KiB | - |
| BRANCH_OUT | control left to some other address | JMP |
| MEM_WRITE | fell through and wrote memory | ST / INC / DEC / NOT |
| REG | fell through, registers or flags changed | ACTIVE |
| IDLE | fell through, nothing changed | IDLE |

Every executed word also gets a 32-bit FNV-1a fingerprint of its final registers, flags, PC and memory writes, the
same device NANO used to count distinct final states.

## 4. Measured speed (workstation, i9-13900K, 24 Python processes)

| step | throughput | all 2^32 words |
|---|---|---|
| decode, AArch64 | 4.8 M words/s | about 15 min |
| decode, x86-64 | 4.0 M words/s | about 18 min |
| execute (decodable words only), AArch64 | about 65 k emulations/s per process; 35.6 % of words execute | roughly 1.5 G emulations: 1 to 2 hours on 24 processes, less with the cluster's 72 CPUs |
| execute, x86-64 | similar per emulation; 22.5 % of words execute | under 1 hour |

GPUs are not usable for this directly: there is no GPU build of QEMU or Unicorn. A GPU interpreter for the executed
subset would have to be hand-written, as NANO's CUDA and Vulkan kernels were. For 2^32 it is not needed; it becomes
the question at 2^40 (five bytes of x86-64, or two AArch64 instructions).

## 5. First results (uniform random samples, seed 1)

Decode, 16,777,216 words per ISA (`results/isa4_census/*_sample_16777216.json`):

| | AArch64 | x86-64 |
|---|---:|---:|
| valid single instruction / whole decode | **35.55 %** | **22.46 %** (whole) |
| undefined / no instruction | 64.45 % | 32.53 % |
| partial (fragment at the end) | - | 45.01 % |
| distinct mnemonics seen | 1,219 | 766 |
| most frequent | ldr 2.88 %, adr 1.58 %, bl 1.57 %, adrp 1.56 %, b 1.56 %, stp 1.44 %, ldp 1.40 % | mov 5.22 %, push 3.95 %, xchg 3.67 %, pop 3.52 %; whole instructions per word: 0: 32.5 %, 1: 32.8 %, 2: 27.6 %, 3: 6.7 %, 4: 0.4 % |

Execution, 1,048,576 words per ISA (`results/isa4_exec/*_sample_1048576.json`):

| class | AArch64 | x86-64 |
|---|---:|---:|
| UNDEF | 64.41 % | 32.56 % |
| OVERRUN | - | 44.93 % |
| EXCEPTION | 5.96 % | 0.57 % |
| FAULT_MEM | 4.08 % | 2.40 % |
| BRANCH_OUT | 4.76 % | 5.46 % |
| MEM_WRITE | 2.64 % | 5.45 % |
| REG | 9.37 % | 7.38 % |
| IDLE | 8.77 % | 1.25 % |
| distinct fingerprints among executed words | 135,309 of 373,000 | 51,844 of 236,000 |

The Unicorn undefined share on AArch64 (64.41 %) matches Capstone's (64.45 %) within sampling error, which is
the cross-check between two independent decoders that the NANO work would demand. The 5.96 % AArch64 exceptions
are the system-register, barrier, SVC/HVC/SMC and privileged encodings that are legal instructions but trap in user
state. BRANCH_OUT on AArch64 is the branch family (B, BL, CBZ, TBZ, RET, ...) plus every word whose target lies
outside the mapped code.

**Exhaustive decode census, AArch64, all 2^32 words** (`results/isa4_census/aarch64_range_4294967296.json`,
902.87 s at 24 processes, 4.76 M words/s): **1,526,900,218 valid instructions = 35.5509 %**, 2,768,067,078
undefined = 64.4491 %, **1,298 distinct mnemonics**. The 16.7 M sample was within 0.0004 percentage points of
the exact figure. Most frequent: ldr 2.867 %, adr 1.575 %, b 1.562 %, adrp 1.562 %, bl 1.562 %, stp 1.442 %,
ldp 1.396 %, str 0.916 %, cbz / cbnz / tbz / tbnz 0.781 % each. The x86-64 exhaustive run was still in progress
when this was written.

## 6. What this is good for, and what it is not

- It gives the behaviour-based catalog of a real ISA on one state, the raw material for the composition idea in
  [04_big_program.md](04_big_program.md) and for comparing "how much of the encoding space does something" across
  NANO (100 % of bytes are instructions), AArch64 (35.6 %) and x86-64 (22.5 % whole words).
- Prior art exists for the decode side (exhaustive instruction-space scans of x86 for undocumented opcodes, around
  2017) and should be cited when this is written up; the execution census from a fixed zero state with NANO-style
  classes and fingerprints is the part these projects add.
- One state is one probe. The NANO catalog used 16 input states; the same applies here (registers preloaded with
  probe values) once the zero-state pass exists.
- Unicorn's model is QEMU's: it is faithful for the architectural state but not a real chip. Undocumented or
  model-specific behaviour is outside what this measures.

## 7. Next steps

1. Finish and commit the exhaustive decode census (in progress).
2. Execution census over all 2^32 AArch64 words: fingerprints written per chunk to disk (the current tool keeps one
   set in memory, fine for samples, not for 1.5 G entries), split by `--range` across the workstation and the idle
   cluster CPUs (adler40 has Python 3.14; Capstone and Unicorn need `pip install --user`).
3. Count distinct behaviours exactly and list the mnemonics per class, then the same for x86-64 whole words.
4. 16-probe catalog as in NANO, and the two-instruction AArch64 space (2^64, sampled only).
