"""The catalog of every operation a one-symbol (2-bit, operand-free) instruction can be, and a reference interpreter.

With no operand bits an instruction can only (a) move the head by a fixed stride, (b) change the cell under the
head, (c) do both, (d) change control flow on the cell's value, or (e) use an implicit accumulator A. This catalog
lists 36 such operations. A machine (a "table") is any set of four of them, one per symbol 0..3; since relabelling
the symbols permutes programs of equal length, only the set matters: C(36, 4) = 58,905 tables.

Machine: ring of 256 cells of W bits, head at cell 0, accumulator A = 0, PC wraps at the program length L
(NANO's convention), fixed step budget. The C sweeper tools/u2_tables.c implements the same catalog and must agree
with this file on every state (checked by tools/u2_table_search.py).

  python tools/u2_ops.py --list
  python tools/u2_ops.py --table R1,INC_L1,OPEN,CLOSE --W 1 --program 1023 --trace
"""
import argparse
import json

N = 256

# name: (group, description)
OPS = [
    ("R1", "move", "head += 1"),
    ("L1", "move", "head -= 1"),
    ("R2", "move", "head += 2"),
    ("L2", "move", "head -= 2"),
    ("R3", "move", "head += 3"),
    ("L3", "move", "head -= 3"),
    ("INC", "write", "cell += 1 (mod 2^W)"),
    ("DEC", "write", "cell -= 1 (mod 2^W)"),
    ("FLIP", "write", "cell ^= 1 (flip the low bit)"),
    ("NOT", "write", "cell ^= all ones (flip every bit)"),
    ("ZERO", "write", "cell = 0"),
    ("SET", "write", "cell = all ones"),
    ("SHL", "write", "cell = (cell << 1) mod 2^W"),
    ("INC_R1", "pen", "cell += 1, then head += 1"),
    ("INC_L1", "pen", "cell += 1, then head -= 1 (P'' lambda)"),
    ("FLIP_R1", "pen", "cell ^= 1, then head += 1 (Universe-7 flip pen, stride 1)"),
    ("FLIP_L1", "pen", "cell ^= 1, then head -= 1"),
    ("FLIP_R2", "pen", "cell ^= 1, then head += 2"),
    ("FLIP_R3", "pen", "cell ^= 1, then head += 3"),
    ("SET_R1", "pen", "cell = all ones, then head += 1"),
    ("SET_R3", "pen", "cell = all ones, then head += 3"),
    ("ZERO_R1", "pen", "cell = 0, then head += 1"),
    ("ZERO_R3", "pen", "cell = 0, then head += 3"),
    ("SKZ", "control", "skip the next instruction if cell == 0"),
    ("SKNZ", "control", "skip the next instruction if cell != 0"),
    ("JZS", "control", "if cell == 0: PC = 0 (restart)"),
    ("JNZS", "control", "if cell != 0: PC = 0 (restart)"),
    ("JNZB", "control", "if cell != 0: PC -= 2 (repeat the previous instruction and this one)"),
    ("OPEN", "control", "[ : if cell == 0 jump to after the matching ] (unmatched: PC = 0)"),
    ("CLOSE", "control", "] : if cell != 0 jump to after the matching [ (unmatched: PC = 0)"),
    ("HALT", "control", "stop"),
    ("NOP", "control", "nothing"),
    ("LDA", "acc", "A = cell"),
    ("STA", "acc", "cell = A"),
    ("SWAP", "acc", "exchange A and cell"),
    ("ADDA", "acc", "cell += A (mod 2^W)"),
]
NAMES = [o[0] for o in OPS]
INDEX = {n: i for i, n in enumerate(NAMES)}


def matching(program, table):
    """Bracket partners for OPEN/CLOSE symbols (-1 if unmatched)."""
    m = [-1] * len(program)
    stack = []
    for i, s in enumerate(program):
        op = NAMES[table[s]]
        if op == "OPEN":
            stack.append(i)
        elif op == "CLOSE" and stack:
            j = stack.pop()
            m[i], m[j] = j, i
    return m


def run(table, program, W=1, steps=64, trace=False):
    """table: 4 op indices; program: list of symbols 0..3. Returns the final state."""
    mask = (1 << W) - 1
    tape = [0] * N
    head = pc = A = 0
    L = len(program)
    halted = False
    m = matching(program, table)
    hist = []
    done = 0
    for step in range(1, steps + 1):
        s = program[pc]
        op = NAMES[table[s]]
        at = pc
        if trace:
            hist.append((step, pc, s, op, head, tape[head], A))
        pc = (pc + 1) % L
        v = tape[head]
        if op == "R1": head = (head + 1) % N
        elif op == "L1": head = (head - 1) % N
        elif op == "R2": head = (head + 2) % N
        elif op == "L2": head = (head - 2) % N
        elif op == "R3": head = (head + 3) % N
        elif op == "L3": head = (head - 3) % N
        elif op == "INC": tape[head] = (v + 1) & mask
        elif op == "DEC": tape[head] = (v - 1) & mask
        elif op == "FLIP": tape[head] = v ^ 1
        elif op == "NOT": tape[head] = v ^ mask
        elif op == "ZERO": tape[head] = 0
        elif op == "SET": tape[head] = mask
        elif op == "SHL": tape[head] = (v << 1) & mask
        elif op == "INC_R1": tape[head] = (v + 1) & mask; head = (head + 1) % N
        elif op == "INC_L1": tape[head] = (v + 1) & mask; head = (head - 1) % N
        elif op == "FLIP_R1": tape[head] = v ^ 1; head = (head + 1) % N
        elif op == "FLIP_L1": tape[head] = v ^ 1; head = (head - 1) % N
        elif op == "FLIP_R2": tape[head] = v ^ 1; head = (head + 2) % N
        elif op == "FLIP_R3": tape[head] = v ^ 1; head = (head + 3) % N
        elif op == "SET_R1": tape[head] = mask; head = (head + 1) % N
        elif op == "SET_R3": tape[head] = mask; head = (head + 3) % N
        elif op == "ZERO_R1": tape[head] = 0; head = (head + 1) % N
        elif op == "ZERO_R3": tape[head] = 0; head = (head + 3) % N
        elif op == "SKZ":
            if v == 0: pc = (pc + 1) % L
        elif op == "SKNZ":
            if v != 0: pc = (pc + 1) % L
        elif op == "JZS":
            if v == 0: pc = 0
        elif op == "JNZS":
            if v != 0: pc = 0
        elif op == "JNZB":
            if v != 0: pc = (pc - 2) % L
        elif op == "OPEN":
            if v == 0: pc = (m[at] + 1) % L if m[at] >= 0 else 0
        elif op == "CLOSE":
            if v != 0: pc = (m[at] + 1) % L if m[at] >= 0 else 0
        elif op == "HALT":
            halted = True
        elif op == "NOP": pass
        elif op == "LDA": A = v
        elif op == "STA": tape[head] = A
        elif op == "SWAP": tape[head], A = A, v
        elif op == "ADDA": tape[head] = (v + A) & mask
        done = step
        if halted:
            break
    nz = [i for i in range(N) if tape[i]]
    return {"tape_nonzero": {i: tape[i] for i in nz}, "head": head, "A": A, "pc": pc, "halted": halted,
            "steps": done, "trace": hist if trace else None}


def tape_key(state):
    """Canonical key of the final tape: (first nonzero index, values up to the last nonzero); blank = ()."""
    nz = state["tape_nonzero"]
    if not nz:
        return ()
    lo, hi = min(nz), max(nz)
    return (lo,) + tuple(nz.get(i, 0) for i in range(lo, hi + 1))


def parse_table(text):
    names = text.split(",")
    if len(names) != 4:
        raise ValueError("a table has exactly four operations")
    return [INDEX[n.strip()] for n in names]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--table", help="four operation names, comma separated, for symbols 0,1,2,3")
    ap.add_argument("--program", help="symbols 0..3")
    ap.add_argument("--W", type=int, default=1)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--trace", action="store_true")
    args = ap.parse_args()
    if args.list:
        print(f"{len(OPS)} operations, C({len(OPS)},4) = {len(OPS)*(len(OPS)-1)*(len(OPS)-2)*(len(OPS)-3)//24:,} four-instruction tables")
        for i, (n, g, d) in enumerate(OPS):
            print(f"{i:2d} {n:8s} {g:8s} {d}")
        return
    table = parse_table(args.table)
    program = [int(c) for c in args.program.replace(" ", "")]
    st = run(table, program, args.W, args.steps, args.trace)
    if st["trace"]:
        for step, pc, s, op, head, cell, A in st["trace"]:
            print(f"step {step:3d} pc {pc:2d} sym {s} {op:8s} head {head:3d} cell {cell} A {A}")
    st.pop("trace")
    st["tape_key"] = tape_key(st)
    print(json.dumps(st))


if __name__ == "__main__":
    main()
