"""Reference interpreter of the original NANO machine (Dimension42), unchanged semantics.

16 byte cells shared by code and data, 8-bit accumulator A, program of L bytes at M0..M(L-1),
PC wraps at L, JMP/JZ targets are n mod L, 64 executed instructions, M8..MF double as an 8x8 display.
This is the machine the exhaustive 5-byte (2^40) and 6-byte (2^48) maps were computed on.
Same instruction table and class rules as Dimension42/explorer/nano_cpu.c and nano_ref.py.

Opcode nibble / argument nibble n:
  0n NOP        1n LDI n      2n ADDI n     3n SUBI n
  4n LD Mn      5n ST Mn      6n ADD Mn     7n JMP n
  8n JZ n       9n INC Mn     An DEC Mn     Bn ROL n&7
  Cn XOR Mn     Dn OUT Mn     En NOT Mn     Fn HLT
"""
OPS = "NOP LDI ADDI SUBI LD ST ADD JMP JZ INC DEC ROL XOR OUT NOT HLT".split()
CLASSES = ["", "IDLE", "HALT", "ACTIVE", "SELF-MOD", "OUTPUT", "DRAW"]
STEPS = 64


def parse(hexstring):
    b = bytes.fromhex(hexstring.replace(" ", ""))
    if not 1 <= len(b) <= 16:
        raise ValueError("a NANO program has 1..16 bytes")
    return b


def disasm(program):
    return [f"{x:02X} {OPS[x >> 4]} {x & 15:X}" for x in program]


def run(program, a=None, b=None, steps=STEPS, first_output=False, a_cell=5, b_cell=6):
    """Run one program from the zero state. Inputs a, b go into cells a_cell, b_cell (default M5, M6).

    Returns a dict: class, first_out (first OUT value or None), outputs (first 8), steps, halted,
    final memory, self_modified (code changed at any point) and the Dimension42-compatible class.
    first_output=True stops at the first OUT, which is how every catalog and adder result is defined.
    """
    program = bytes(program)
    L = len(program)
    M = list(program) + [0] * (16 - L)
    if a is not None:
        M[a_cell] = a & 255
    if b is not None:
        M[b_cell] = b & 255
    T = M[:]                      # OR-trace of every cell ever seen
    A = pc = n_steps = outc = 0
    tA = 0
    halted = False
    outputs = []
    first = None
    self_modified = False
    while n_steps < steps:
        n_steps += 1
        ins = M[pc]
        pc = (pc + 1) % L
        op, n = ins >> 4, ins & 15
        if op == 1: A = n
        elif op == 2: A = (A + n) & 255
        elif op == 3: A = (A - n) & 255
        elif op == 4: A = M[n]
        elif op == 5: M[n] = A
        elif op == 6: A = (A + M[n]) & 255
        elif op == 7: pc = n % L
        elif op == 8:
            if A == 0: pc = n % L
        elif op == 9: M[n] = (M[n] + 1) & 255
        elif op == 10: M[n] = (M[n] - 1) & 255
        elif op == 11:
            k = n & 7
            A = ((A << k) | (A >> (8 - k))) & 255
        elif op == 12: A ^= M[n]
        elif op == 13:
            if outc < 8: outputs.append(M[n])
            outc += 1
            if first is None:
                first = M[n]
                if first_output:
                    break
        elif op == 14: M[n] ^= 255
        elif op == 15:
            halted = True
            break
        for i in range(16): T[i] |= M[i]
        tA |= A
        if M[:L] != list(program): self_modified = True
    draw = any(T[8:])
    code_changed = M[:L] != list(program)
    active = tA or any(T[L:8])
    cls = 6 if draw else 5 if outc else 4 if code_changed else 3 if active else 2 if halted else 1
    return {"program": program.hex().upper(), "class": CLASSES[cls], "first_out": first,
            "outputs": outputs, "output_count": outc, "steps": n_steps, "halted": halted,
            "A": A, "memory": M, "self_modified": self_modified}


def first_out(program, a, b, **kw):
    return run(program, a, b, first_output=True, **kw)["first_out"]


if __name__ == "__main__":
    import argparse, json
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("program", help="hex bytes, e.g. '17 C5 91 50 71'")
    p.add_argument("-a", type=lambda s: int(s, 0))
    p.add_argument("-b", type=lambda s: int(s, 0))
    p.add_argument("--first-output", action="store_true")
    args = p.parse_args()
    prog = parse(args.program)
    print("\n".join(disasm(prog)))
    print(json.dumps(run(prog, args.a, args.b, first_output=args.first_output), indent=1))
