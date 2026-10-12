"""Is there a language? Signals, listeners and the social task in a DNA-3 ladder checkpoint (docs/17 section 18).

  python tools/dna3_language.py RUN.ckpt --W 3424 [--json out.json]

Reads a version-4 checkpoint (strands, live flags, the Proc array: A, sig, xin, open, ldir) and the genes (codon 0 to
the first STOP, day-6 table). A signal is the 2-bit value a processor last broadcast with SIGNAL (`sig`, visible to
its 8 neighbours); LISTEN loads a neighbour's signal into A and records the direction (`ldir`). Measured:

1. vocabulary: the distribution of `sig` over organisms whose gene contains SIGNAL, and the mutual information (bits)
   between the signal and the sender's own state: low half b = xin & 3, high half a = xin >> 2, the low I/O cell (its
   answer). A signal "means" something about the sender when this is well above 0 (max 2 bits);
2. dialects: for adjacent signallers, P(same signal) against the value expected from the signal frequencies;
3. listening: how many listeners have a recorded direction, how often that neighbour is alive, signals (gene with
   SIGNAL) and has an input;
4. use: among listeners that have answered their current input (open = 0), the fraction whose low I/O cell equals
   b xor b' (the social task's target, b' = low half of the listened neighbour's input), against the same fraction
   with b' taken from the opposite neighbour (a control with the same statistics but no channel), the echo b, and
   1/4. A channel that carries the partner's input shows as target >> control.
This is a snapshot (the I/O cell may have changed since the answer; xin of the neighbour is the current one, not the
one of the answer's tick), so the numbers are estimates, compared only against the built-in control.
"""
import argparse
import collections
import json
import math
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sim"))
sys.path.insert(0, HERE)
import dna_ref as R  # noqa: E402
import dna3_ckpt_genes as K  # noqa: E402

PROC = ["slot", "pc", "H", "P", "Q", "child", "A", "age", "depth", "copied", "E", "xs", "sig", "xin", "open", "streak", "ldir"]


def mi(x, y, nx, ny):
    """mutual information in bits of two small non-negative integer arrays"""
    if len(x) == 0:
        return 0.0
    j = np.bincount(x * ny + y, minlength=nx * ny).reshape(nx, ny).astype(float)
    j /= j.sum()
    px, py = j.sum(1), j.sum(0)
    nz = j > 0
    return float((j[nz] * np.log2(j[nz] / (px[:, None] * py[None, :])[nz])).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt")
    ap.add_argument("--W", type=int, required=True)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    table = np.array(K.table_for([1, 2, 3, 4, 5, 6], False), dtype=np.int16)
    with open(a.ckpt, "rb") as f:
        assert f.read(8) == b"DNA3CKPT"
        ver, N, G, S, seed, flags, t_next = struct.unpack("<7Q", f.read(56))
    assert ver == 4, "needs a version-4 (ladder) checkpoint"
    nb = (N + 3) // 4
    off_fwd = 64
    off_live = 64 + 2 * nb
    off_proc = off_live + S
    fw = np.memmap(a.ckpt, dtype=np.uint8, mode="r", offset=off_fwd, shape=(nb,))
    live = np.memmap(a.ckpt, dtype=np.uint8, mode="r", offset=off_live, shape=(S,))
    proc = np.memmap(a.ckpt, dtype=np.int64, mode="r", offset=off_proc, shape=(S, 17))
    W = a.W
    Rr = S // W
    slots = np.nonzero(live)[0]
    P = {k: np.asarray(proc[slots, i]) for i, k in enumerate(PROC) if k in ("A", "sig", "xin", "open", "ldir")}
    # genes: SIGNAL / LISTEN carriers; low and high I/O cells
    maxc = G // 3
    has_sig = np.zeros(len(slots), bool)
    has_lis = np.zeros(len(slots), bool)
    for lo in range(0, len(slots), 200000):
        sl = slots[lo:lo + 200000]
        idx = sl[:, None] * G + np.arange(3 * maxc)[None, :]
        c = (fw[idx >> 2] >> ((idx & 3) * 2).astype(np.uint8)) & 3
        ops = table[c[:, 0::3] * 16 + c[:, 1::3] * 4 + c[:, 2::3]]
        isstop = ops == R.STOP
        first = np.where(isstop.any(1), isstop.argmax(1), maxc)
        inside = np.arange(maxc)[None, :] < first[:, None]
        inside[:, 0] = False
        has_sig[lo:lo + len(sl)] = ((ops == R.SIGNAL) & inside).any(1)
        has_lis[lo:lo + len(sl)] = ((ops == R.LISTEN) & inside).any(1)

    def cell(s, k):
        i = s * G + k
        return (fw[i >> 2] >> ((i & 3) * 2).astype(np.uint8)) & 3

    lo_cell = cell(slots, G - 1).astype(np.int64)
    out = {"tick": int(t_next), "live": int(len(slots)), "signal_genes": int(has_sig.sum()), "listen_genes": int(has_lis.sum())}
    # 1. vocabulary
    sg = has_sig & (P["xin"] < 16)
    sig = P["sig"][sg] & 3
    out["signal_values"] = np.bincount(sig, minlength=4).tolist()
    out["mi_signal_b"] = mi(sig, P["xin"][sg] & 3, 4, 4)
    out["mi_signal_a"] = mi(sig, P["xin"][sg] >> 2, 4, 4)
    out["mi_signal_lowcell"] = mi(sig, lo_cell[sg], 4, 4)
    out["p_signal_eq_b"] = float((sig == (P["xin"][sg] & 3)).mean()) if len(sig) else None
    # neighbour lookup over the full grid
    full_live = np.asarray(live).astype(bool)
    idx_of = np.full(S, -1, np.int64)
    idx_of[slots] = np.arange(len(slots))
    xs, ys = slots % W, slots // W

    def neigh(d):
        dx, dy = R.NEIGHBOURS[d]
        return ((ys + dy) % Rr) * W + (xs + dx) % W

    # 2. dialects: adjacent signallers (east and south pairs)
    same = tot = 0
    freq = np.bincount(sig, minlength=4) / max(1, len(sig))
    for d in range(8):
        dx, dy = R.NEIGHBOURS[d]
        if (dx, dy) not in ((1, 0), (0, 1)):
            continue
        ns = neigh(d)
        j = idx_of[ns]
        ok = sg & (j >= 0)
        ok[ok] &= sg[j[ok]]
        same += int(((P["sig"][ok] & 3) == (P["sig"][j[ok]] & 3)).sum())
        tot += int(ok.sum())
    out["adjacent_signaller_pairs"] = tot
    out["p_same_signal_adjacent"] = same / tot if tot else None
    out["p_same_signal_expected"] = float((freq ** 2).sum())
    # 3. listening
    L = has_lis & (P["ldir"] < 8)
    out["listeners_with_direction"] = int(L.sum())
    li = np.nonzero(L)[0]
    tgt = np.empty(len(li), np.int64)
    opp = np.empty(len(li), np.int64)
    for d in range(8):
        m = P["ldir"][li] == d
        dx, dy = R.NEIGHBOURS[d]
        od = next(k for k in range(8) if R.NEIGHBOURS[k] == (-dx, -dy))
        tgt[m] = neigh(d)[li[m]]
        opp[m] = neigh(od)[li[m]]
    jt, jo = idx_of[tgt], idx_of[opp]
    alive_t = jt >= 0
    out["listened_neighbour_alive"] = float(alive_t.mean()) if len(li) else None
    out["listened_neighbour_signals"] = float((alive_t & has_sig[np.where(alive_t, jt, 0)]).mean()) if len(li) else None
    # 4. use
    ok = alive_t & (jo >= 0)
    ok &= (P["xin"][li] < 16) & (P["open"][li] == 0)
    ok &= (P["xin"][np.where(ok, jt, 0)] < 16) & (P["xin"][np.where(ok, jo, 0)] < 16)
    v = lo_cell[li][ok]
    b = P["xin"][li][ok] & 3
    bt = P["xin"][jt[ok]] & 3
    bo = P["xin"][jo[ok]] & 3
    out["answered_listeners_tested"] = int(ok.sum())
    out["p_target_listened"] = float((v == (b ^ bt)).mean()) if ok.any() else None
    out["p_target_control_opposite"] = float((v == (b ^ bo)).mean()) if ok.any() else None
    out["p_echo_b"] = float((v == b).mean()) if ok.any() else None
    out["p_listened_b_is_0"] = float((bt == 0).mean()) if ok.any() else None
    sgn = has_sig[jt[ok]]
    out["tested_with_signalling_partner"] = int(sgn.sum())
    if sgn.any():
        out["p_target_listened_signalling_partner"] = float((v[sgn] == (b[sgn] ^ bt[sgn])).mean())
        out["p_target_control_signalling_partner"] = float((v[sgn] == (b[sgn] ^ bo[sgn])).mean())
    for k, val in out.items():
        print("%-40s %s" % (k, ("%.4f" % val) if isinstance(val, float) else val))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
