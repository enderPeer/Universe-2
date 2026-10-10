/* GPU twin of the `primany` mode of tools/u2_tables.c (docs/14): the arithmetic primitives of ONE machine over every
 * program of exactly L symbols in an index range, at EVERY step budget 1..steps at once.
 *
 *   nvcc -O3 -std=c++17 -arch=sm_89 -o u2_mac_any_gpu u2_mac_any.cu      (RTX 4090 / 4080; -arch=sm_86 for the RTX 3060)
 *   nvcc ... -DNCELL=512 -DMAXSTEPS=512 -o u2_mac_any_gpu_n512_s512 ...   (the 512-cell, 512-budget variant, docs/14 section 11)
 *   ./u2_mac_any_gpu PRIMS W L steps OPS start count [--gpu i] [--chunk n] [--tpb n] [--maxin n]
 *
 * Same machine semantics as tools/u2_ops.py and u2_tables.c (ring of NCELL cells of W bits, head 0, A = 0, PC wraps at
 * L, HALT freezes the state), same primitives, inputs and input order as prim_build() there (copied below), same
 * definition as primany: a program is the primitive at budget s when one fixed cell, or the cell under the head
 * after exactly s instructions, holds the target after exactly s instructions for every input. One thread runs one
 * program. The algorithm differs from the CPU mode on purpose (independent implementations must agree): the first
 * four inputs are simulated in lockstep and only the per-step OR of their joint candidate masks is kept, so a
 * program that dies within four inputs (almost all of them) touches no per-step memory at all; a survivor is re-run
 * in lockstep to fill cand[s] for s = 1..steps and then takes the remaining inputs one by one, shrinking cand[s].
 * The four tapes of a thread are packed W bits per cell (64 bytes each for W = 2 on 256 cells) in shared memory,
 * interleaved so that a warp's accesses to the same word index are conflict-free; W must be 1, 2 or 4. The
 * cell-equals-target masks (NCELL / 64 words each) live in registers.
 * Output: the format of the CPU mode (header, per primitive: inputs, order, survivors, found, found_by_step, F lines),
 * except that an F line carries the program and its step ranges only (F <prim> <program> <ranges>); the cells per
 * step are recovered by re-running the found program through the CPU mode (count = 1), as tools/u2_mac_any.py does.
 * Machines with OPEN/CLOSE are refused (no bracket matching on the GPU; none of the machines of docs/13 uses them).
 * stderr: mode=primany ... device=gpu<i>:<name>.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <chrono>
#include <unistd.h>
#include <cuda_runtime.h>

#define NOPS 41
#ifndef NCELL
#define NCELL 256
#endif
#define MAXL 16
#define MAXIN 64
#ifndef MAXSTEPS
#define MAXSTEPS 256
#endif
#define NWM (NCELL / 64)      /* words of a cell mask */
#define SW (MAXSTEPS / 64)    /* words of a step mask */
#define RMASK (NCELL - 1)
#define MAXPRIM 8
#define NSIM 4

static const char *NAMES[NOPS] = {
    "R1", "L1", "R2", "L2", "R3", "L3", "INC", "DEC", "FLIP", "NOT", "ZERO", "SET", "SHL",
    "INC_R1", "INC_L1", "FLIP_R1", "FLIP_L1", "FLIP_R2", "FLIP_R3", "SET_R1", "SET_R3", "ZERO_R1", "ZERO_R3",
    "SKZ", "SKNZ", "JZS", "JNZS", "JNZB", "OPEN", "CLOSE", "HALT", "NOP", "LDA", "STA", "SWAP", "ADDA",
    "EAT", "MATE_A", "MATE_B", "SLEEP", "WASTE"};
enum { R1, L1, R2, L2, R3, L3, INC, DEC, FLIP, NOT, ZERO, SET, SHL, INC_R1, INC_L1, FLIP_R1, FLIP_L1, FLIP_R2,
       FLIP_R3, SET_R1, SET_R3, ZERO_R1, ZERO_R3, SKZ, SKNZ, JZS, JNZS, JNZB, OPEN, CLOSE, HALT, NOP, LDA, STA,
       SWAP, ADDA, EAT, MATE_A, MATE_B, SLEEP, WASTE };

typedef struct { const char *name; int excl0; int nin, ninit; uint8_t init[MAXIN][3]; uint8_t target[MAXIN]; } Prim;

/* verbatim copy of prim_build() in tools/u2_tables.c */
static int prim_build(Prim *p, const char *name, int W)
{
    int mask = (1 << W) - 1, nv = 1 << W, neg = mask;
    uint8_t ti[MAXIN][3]; int tt[MAXIN], n = 0;
    memset(p, 0, sizeof *p);
    p->name = name;
#define PUSH(a, b, c, t) do { ti[n][0] = (uint8_t)(a); ti[n][1] = (uint8_t)(b); ti[n][2] = (uint8_t)(c); tt[n] = (t); n++; } while (0)
    if (strcmp(name, "COPY") == 0) {
        p->ninit = 1; p->excl0 = 1;
        for (int a = 1; a < nv; a++) PUSH(a, 0, 0, a);
        PUSH(0, 0, 0, 0);
    } else if (strcmp(name, "ADD") == 0) {
        p->ninit = 2;
        for (int a = 0; a < nv; a++) for (int b = 0; b < nv; b++) PUSH(a, b, 0, (a + b) & mask);
    } else if (strcmp(name, "MUL_T") == 0) {
        p->ninit = 2;
        const int ws[3] = {1, neg, 0};
        for (int i = 0; i < 3; i++) for (int x = 0; x < nv; x++) PUSH(ws[i], x, 0, (ws[i] * x) & mask);
    } else if (strcmp(name, "MAC_T") == 0) {
        p->ninit = 3;
        const int ws[3] = {1, neg, 0};
        for (int i = 0; i < 3; i++) for (int acc = 0; acc < nv; acc++) for (int x = 0; x < nv; x++) PUSH(acc, ws[i], x, (acc + ws[i] * x) & mask);
    } else if (strcmp(name, "MAC_B") == 0) {
        p->ninit = 3;
        const int cs[3] = {2, 0, 1};
        for (int i = 0; i < 3; i++) for (int acc = 0; acc < nv; acc++) for (int x = 0; x < nv; x++) PUSH(acc, cs[i], x, (acc + (cs[i] - 1) * x) & mask);
    } else if (strcmp(name, "CARRY") == 0) {
        p->ninit = 2;
        for (int a = 0; a < nv; a++) for (int b = 0; b < nv; b++) PUSH(a, b, 0, a + b >= nv);
    } else return 0;
#undef PUSH
    int k = 0;
    for (int pass = 0; pass < 2; pass++)
        for (int i = 0; i < n; i++)
            if ((tt[i] != 0) == (pass == 0)) { memcpy(p->init[k], ti[i], 3); p->target[k] = (uint8_t)tt[i]; k++; }
    p->nin = n;
    return 1;
}

static int find_op(const char *s) { for (int i = 0; i < NOPS; i++) if (strcmp(NAMES[i], s) == 0) return i; return -1; }
static int parse_ops(const char *text, int *table)
{
    char buf[256]; strncpy(buf, text, 255); buf[255] = 0;
    int n = 0;
    for (char *tok = strtok(buf, ","); tok && n < 4; tok = strtok(NULL, ",")) { table[n] = find_op(tok); if (table[n] < 0) return 0; n++; }
    return n == 4;
}

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(3); } } while (0)

__constant__ int c_table[4];
__constant__ uint8_t c_init[MAXPRIM][MAXIN][3];
__constant__ uint8_t c_target[MAXPRIM][MAXIN];
__constant__ int c_nin[MAXPRIM], c_ninit[MAXPRIM], c_excl0[MAXPRIM];

struct Found { unsigned long long q; unsigned long long stepmask[SW]; int prim; int pad; };

/* packed tape: cell c of width W sits in word c / (64/W) at bit offset (c % (64/W)) * W; words are strided by `stride`
 * (the block size) so that the threads of a warp touch consecutive shared-memory words */
template <int W> struct Tape {
    static constexpr int CPW = 64 / W, LOG = (W == 1 ? 6 : W == 2 ? 5 : 4), NW = NCELL / CPW;
    uint64_t *base; int stride;
    __device__ __forceinline__ uint64_t &word(int c) const { return base[(c >> LOG) * stride]; }
    __device__ __forceinline__ uint8_t get(int c) const { return (uint8_t)((word(c) >> ((c & (CPW - 1)) * W)) & ((1u << W) - 1)); }
    __device__ __forceinline__ void set(int c, uint8_t v) const {
        int sh = (c & (CPW - 1)) * W; uint64_t m = (uint64_t)((1u << W) - 1) << sh;
        uint64_t &w = word(c); w = (w & ~m) | ((uint64_t)v << sh);
    }
    __device__ __forceinline__ void clear() const { for (int w = 0; w < NW; w++) base[w * stride] = 0ULL; }
};

template <int W> struct Sim {
    Tape<W> tape; int head, pc, A, halted; uint64_t eq[NWM]; uint8_t t;
    /* eq[] is indexed with a runtime word index only through these two, fully unrolled selects, so it stays in registers */
    __device__ __forceinline__ uint64_t eq_get(int wi) const { uint64_t r = 0;
        #pragma unroll
        for (int k = 0; k < NWM; k++) r |= (k == wi) ? eq[k] : 0ULL;
        return r; }
    __device__ __forceinline__ void eq_put(int wi, uint64_t e) {
        #pragma unroll
        for (int k = 0; k < NWM; k++) if (k == wi) eq[k] = e; }
    __device__ __forceinline__ void eq_bit(int c, int on) {
        uint64_t b = 1ULL << (c & 63); int wi = c >> 6;
        uint64_t e = eq_get(wi);
        eq_put(wi, on ? (e | b) : (e & ~b));
    }
    __device__ __forceinline__ void init(int k, int i) {
        head = pc = A = halted = 0; t = c_target[k][i];
        tape.clear();
        const int ninit = c_ninit[k];
        for (int j = 0; j < ninit; j++) tape.set(j, c_init[k][i][j]);
        const uint64_t all = t == 0 ? ~0ULL : 0ULL;
        #pragma unroll
        for (int k2 = 0; k2 < NWM; k2++) eq[k2] = all;
        for (int c = 0; c < ninit; c++) eq_bit(c, tape.get(c) == t);
    }
    __device__ __forceinline__ int hok(int excl0) const { return tape.get(head) == t && !(excl0 && head == 0); }
    /* one instruction (semantics of run() in u2_tables.c, minus OPEN/CLOSE), then the eq-mask update */
    __device__ __forceinline__ void step(uint32_t pb, int L) {
        if (halted) return;
        const uint8_t mask = (uint8_t)((1u << W) - 1);
        int wh = head;
        int op = c_table[(pb >> (2 * (L - 1 - pc))) & 3];
        pc = pc + 1 == L ? 0 : pc + 1;
        uint8_t v = tape.get(head), nv = v;
        switch (op) {
        case R1: head = (head + 1) & RMASK; break;
        case L1: head = (head - 1) & RMASK; break;
        case R2: head = (head + 2) & RMASK; break;
        case L2: head = (head - 2) & RMASK; break;
        case R3: head = (head + 3) & RMASK; break;
        case L3: head = (head - 3) & RMASK; break;
        case INC: nv = (uint8_t)((v + 1) & mask); break;
        case DEC: nv = (uint8_t)((v - 1) & mask); break;
        case FLIP: nv = v ^ 1; break;
        case NOT: nv = v ^ mask; break;
        case ZERO: nv = 0; break;
        case SET: nv = mask; break;
        case SHL: nv = (uint8_t)((v << 1) & mask); break;
        case INC_R1: nv = (uint8_t)((v + 1) & mask); head = (head + 1) & RMASK; break;
        case INC_L1: nv = (uint8_t)((v + 1) & mask); head = (head - 1) & RMASK; break;
        case FLIP_R1: nv = v ^ 1; head = (head + 1) & RMASK; break;
        case FLIP_L1: nv = v ^ 1; head = (head - 1) & RMASK; break;
        case FLIP_R2: nv = v ^ 1; head = (head + 2) & RMASK; break;
        case FLIP_R3: nv = v ^ 1; head = (head + 3) & RMASK; break;
        case SET_R1: nv = mask; head = (head + 1) & RMASK; break;
        case SET_R3: nv = mask; head = (head + 3) & RMASK; break;
        case ZERO_R1: nv = 0; head = (head + 1) & RMASK; break;
        case ZERO_R3: nv = 0; head = (head + 3) & RMASK; break;
        case SKZ: if (v == 0) pc = pc + 1 == L ? 0 : pc + 1; break;
        case SKNZ: if (v != 0) pc = pc + 1 == L ? 0 : pc + 1; break;
        case JZS: if (v == 0) pc = 0; break;
        case JNZS: if (v != 0) pc = 0; break;
        case JNZB: if (v != 0) { pc -= 2; while (pc < 0) pc += L; } break;
        case HALT: halted = 1; break;
        case NOP: break;
        case LDA: A = v; break;
        case STA: nv = (uint8_t)A; break;
        case SWAP: nv = (uint8_t)A; A = v; break;
        case ADDA: nv = (uint8_t)((v + A) & mask); break;
        case EAT: A = v; nv = 0; break;
        case MATE_A: nv = (uint8_t)((v + A + 1) & mask); head = (head + 1) & RMASK; break;
        case MATE_B: nv = (uint8_t)((A + 1) & mask); head = (head + 1) & RMASK; break;
        case SLEEP: if (A == 0) pc = pc + 1 == L ? 0 : pc + 1; break;
        case WASTE: nv = mask; A = 0; head = (head - 1) & RMASK; break;
        default: break;
        }
        if (nv != v) { tape.set(wh, nv); eq_bit(wh, nv == t); }
    }
};

/* returns the number of inputs survived (nin = found); stepmask: bit s set when step s+1 has a candidate (found only) */
template <int W>
__device__ int eval_any(uint32_t pb, int L, int steps, int k, uint64_t *smem, unsigned long long *stepmask)
{
    const int nin = c_nin[k], excl0 = c_excl0[k];
    const uint64_t cell0 = excl0 ? ~1ULL : ~0ULL;
    const int stride = blockDim.x;
    Sim<W> sim[NSIM];
    for (int j = 0; j < NSIM; j++) { sim[j].tape.base = smem + (size_t)j * Tape<W>::NW * stride + threadIdx.x; sim[j].tape.stride = stride; }
    /* phase 1: inputs 0..3 in lockstep, per-step joint masks, nothing stored */
    for (int j = 0; j < NSIM; j++) sim[j].init(k, j);
    uint64_t any1 = 0, any2 = 0, any3 = 0, any4 = 0;
    for (int s = 0; s < steps; s++) {
        sim[0].step(pb, L); sim[1].step(pb, L); sim[2].step(pb, L); sim[3].step(pb, L);
        uint64_t m[NWM], o;
        #pragma unroll
        for (int w = 0; w < NWM; w++) m[w] = sim[0].eq[w];
        m[0] &= cell0;
        int h = sim[0].hok(excl0);
        o = (uint64_t)h;
        #pragma unroll
        for (int w = 0; w < NWM; w++) o |= m[w];
        any1 |= o;
        #pragma unroll
        for (int w = 0; w < NWM; w++) m[w] &= sim[1].eq[w];
        h &= sim[1].hok(excl0); o = (uint64_t)h;
        #pragma unroll
        for (int w = 0; w < NWM; w++) o |= m[w];
        any2 |= o;
        #pragma unroll
        for (int w = 0; w < NWM; w++) m[w] &= sim[2].eq[w];
        h &= sim[2].hok(excl0); o = (uint64_t)h;
        #pragma unroll
        for (int w = 0; w < NWM; w++) o |= m[w];
        any3 |= o;
        #pragma unroll
        for (int w = 0; w < NWM; w++) m[w] &= sim[3].eq[w];
        h &= sim[3].hok(excl0); o = (uint64_t)h;
        #pragma unroll
        for (int w = 0; w < NWM; w++) o |= m[w];
        any4 |= o;
    }
    if (!any4) return any3 ? 3 : any2 ? 2 : any1 ? 1 : 0;
    /* phase 2: survivor of four inputs; re-run the four in lockstep storing cand[s], then the remaining inputs */
    uint64_t cand[MAXSTEPS][NWM];
    uint64_t hok[SW];
    for (int w = 0; w < SW; w++) hok[w] = 0;
    for (int j = 0; j < NSIM; j++) sim[j].init(k, j);
    for (int s = 0; s < steps; s++) {
        sim[0].step(pb, L); sim[1].step(pb, L); sim[2].step(pb, L); sim[3].step(pb, L);
        #pragma unroll
        for (int w = 0; w < NWM; w++) cand[s][w] = sim[0].eq[w] & sim[1].eq[w] & sim[2].eq[w] & sim[3].eq[w];
        cand[s][0] &= cell0;
        int h = sim[0].hok(excl0) & sim[1].hok(excl0) & sim[2].hok(excl0) & sim[3].hok(excl0);
        if (h) hok[s >> 6] |= 1ULL << (s & 63);
    }
    for (int i = NSIM; i < nin; i++) {
        sim[0].init(k, i);
        uint64_t alive = 0;
        for (int s = 0; s < steps; s++) {
            sim[0].step(pb, L);
            uint64_t o = 0;
            #pragma unroll
            for (int w = 0; w < NWM; w++) { cand[s][w] &= sim[0].eq[w]; o |= cand[s][w]; }
            uint64_t hb = 1ULL << (s & 63);
            if (!sim[0].hok(excl0)) hok[s >> 6] &= ~hb;
            alive |= o | (hok[s >> 6] & hb);
        }
        if (!alive) return i;
    }
    for (int w = 0; w < SW; w++) stepmask[w] = 0;
    for (int s = 0; s < steps; s++) {
        uint64_t o = 0;
        #pragma unroll
        for (int w = 0; w < NWM; w++) o |= cand[s][w];
        if (o || (hok[s >> 6] >> (s & 63) & 1)) stepmask[s >> 6] |= 1ULL << (s & 63);
    }
    return nin;
}

template <int W>
__global__ void k_any(unsigned long long start, unsigned long long count, int L, int steps, int np,
                      unsigned long long *hist, unsigned long long *fbs, Found *found, unsigned *nfound, unsigned maxfound)
{
    extern __shared__ uint64_t smem[];
    __shared__ unsigned long long sh[MAXPRIM * (MAXIN + 1)];
    for (int i = threadIdx.x; i < np * (MAXIN + 1); i += blockDim.x) sh[i] = 0;
    __syncthreads();
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        unsigned long long q = start + idx;
        uint32_t pb = (uint32_t)q;
        for (int k = 0; k < np; k++) {
            unsigned long long sm[SW];
            int ns = eval_any<W>(pb, L, steps, k, smem, sm);
            atomicAdd(&sh[k * (MAXIN + 1) + ns], 1ULL);
            if (ns == c_nin[k]) {
                unsigned slot = atomicAdd(nfound, 1u);
                if (slot < maxfound) { Found f; f.q = q; for (int w = 0; w < SW; w++) f.stepmask[w] = sm[w]; f.prim = k; f.pad = 0; found[slot] = f; }
                for (int s = 0; s < steps; s++) if (sm[s >> 6] >> (s & 63) & 1) atomicAdd(&fbs[k * MAXSTEPS + s], 1ULL);
            }
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < np * (MAXIN + 1); i += blockDim.x) if (sh[i]) atomicAdd(&hist[i], sh[i]);
}

static void ranges(const unsigned long long *sm, int steps, char *out, size_t cap)
{
    size_t n = 0; int rs = -1, first = 1;
    for (int s = 0; s <= steps; s++) {
        int on = s < steps && (sm[s >> 6] >> (s & 63) & 1);
        if (on && rs < 0) rs = s;
        if (!on && rs >= 0) {
            if (s - 1 > rs) n += (size_t)snprintf(out + n, cap - n, first ? "%d-%d" : ",%d-%d", rs + 1, s);
            else n += (size_t)snprintf(out + n, cap - n, first ? "%d" : ",%d", rs + 1);
            first = 0; rs = -1;
        }
    }
    out[n] = 0;
}

template <int W>
static void launch(unsigned blocks, int tpb, size_t smem, unsigned long long start, unsigned long long n, int L, int steps, int np,
                   unsigned long long *d_hist, unsigned long long *d_fbs, Found *d_found, unsigned *d_nfound, unsigned maxfound)
{
    static bool attr = false;
    if (!attr && smem > 48 * 1024) { CK(cudaFuncSetAttribute(k_any<W>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem)); attr = true; }
    k_any<W><<<blocks, tpb, smem>>>(start, n, L, steps, np, d_hist, d_fbs, d_found, d_nfound, maxfound);
}

int main(int argc, char **argv)
{
    if (argc < 8) { fprintf(stderr, "usage: u2_mac_any_gpu PRIMS W L steps OPS start count [--gpu i] [--chunk n] [--tpb n] [--maxin n]\n"); return 2; }
    char names[256]; strncpy(names, argv[1], 255); names[255] = 0;
    int W = atoi(argv[2]), L = atoi(argv[3]), steps = atoi(argv[4]);
    int table[4];
    if (!parse_ops(argv[5], table)) { fprintf(stderr, "unknown operation in %s\n", argv[5]); return 2; }
    for (int i = 0; i < 4; i++) if (table[i] == OPEN || table[i] == CLOSE) { fprintf(stderr, "OPEN/CLOSE machines are not supported on the GPU\n"); return 2; }
    unsigned long long start = strtoull(argv[6], 0, 0), count = strtoull(argv[7], 0, 0);
    int gpu = 0; unsigned long long chunk = 1ULL << 22; int tpb = NCELL > 256 ? 64 : 128; int maxin = 0;
    for (int a = 8; a + 1 < argc; a += 2) {
        if (!strcmp(argv[a], "--gpu")) gpu = atoi(argv[a + 1]);
        else if (!strcmp(argv[a], "--chunk")) chunk = strtoull(argv[a + 1], 0, 0);
        else if (!strcmp(argv[a], "--tpb")) tpb = atoi(argv[a + 1]);
        else if (!strcmp(argv[a], "--maxin")) maxin = atoi(argv[a + 1]);   /* keep only the first N inputs: exhibits the programs that get that far */
        else { fprintf(stderr, "unknown option %s\n", argv[a]); return 2; }
    }
    if (L < 1 || L > MAXL || (W != 1 && W != 2 && W != 4) || steps < 1 || steps > MAXSTEPS) { fprintf(stderr, "1 <= L <= %d, W in {1, 2, 4}, 1 <= steps <= %d\n", MAXL, MAXSTEPS); return 2; }
    Prim prims[MAXPRIM]; int np = 0;
    for (char *tok = strtok(names, ","); tok && np < MAXPRIM; tok = strtok(NULL, ",")) {
        if (!prim_build(&prims[np], tok, W)) { fprintf(stderr, "unknown primitive %s\n", tok); return 2; }
        if (maxin > 0 && maxin < prims[np].nin) prims[np].nin = maxin;
        if (prims[np].nin < NSIM) { fprintf(stderr, "primitive %s has fewer than %d inputs\n", tok, NSIM); return 2; }
        np++;
    }
    CK(cudaSetDevice(gpu));
    CK(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, gpu));
    CK(cudaMemcpyToSymbol(c_table, table, sizeof table));
    {
        static uint8_t hi[MAXPRIM][MAXIN][3], ht[MAXPRIM][MAXIN]; static int hn[MAXPRIM], hni[MAXPRIM], he[MAXPRIM];
        for (int k = 0; k < np; k++) { memcpy(hi[k], prims[k].init, sizeof prims[k].init); memcpy(ht[k], prims[k].target, sizeof prims[k].target); hn[k] = prims[k].nin; hni[k] = prims[k].ninit; he[k] = prims[k].excl0; }
        CK(cudaMemcpyToSymbol(c_init, hi, sizeof hi)); CK(cudaMemcpyToSymbol(c_target, ht, sizeof ht));
        CK(cudaMemcpyToSymbol(c_nin, hn, sizeof hn)); CK(cudaMemcpyToSymbol(c_ninit, hni, sizeof hni)); CK(cudaMemcpyToSymbol(c_excl0, he, sizeof he));
    }
    const unsigned maxfound = 1u << 20;
    unsigned long long *d_hist, *d_fbs; Found *d_found; unsigned *d_nfound;
    CK(cudaMalloc(&d_hist, sizeof(unsigned long long) * MAXPRIM * (MAXIN + 1)));
    CK(cudaMalloc(&d_fbs, sizeof(unsigned long long) * MAXPRIM * MAXSTEPS));
    CK(cudaMalloc(&d_found, sizeof(Found) * maxfound));
    CK(cudaMalloc(&d_nfound, sizeof(unsigned)));
    CK(cudaMemset(d_hist, 0, sizeof(unsigned long long) * MAXPRIM * (MAXIN + 1)));
    CK(cudaMemset(d_fbs, 0, sizeof(unsigned long long) * MAXPRIM * MAXSTEPS));
    const size_t smem = (size_t)NSIM * (NCELL * W / 64) * tpb * sizeof(uint64_t);
    std::vector<Found> found;
    std::vector<Found> buf(maxfound);
    unsigned long long overflow = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (unsigned long long off = 0; off < count; off += chunk) {
        unsigned long long n = std::min(chunk, count - off);
        CK(cudaMemset(d_nfound, 0, sizeof(unsigned)));
        unsigned blocks = (unsigned)((n + tpb - 1) / tpb);
        if (W == 1) launch<1>(blocks, tpb, smem, start + off, n, L, steps, np, d_hist, d_fbs, d_found, d_nfound, maxfound);
        else if (W == 2) launch<2>(blocks, tpb, smem, start + off, n, L, steps, np, d_hist, d_fbs, d_found, d_nfound, maxfound);
        else launch<4>(blocks, tpb, smem, start + off, n, L, steps, np, d_hist, d_fbs, d_found, d_nfound, maxfound);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        unsigned nf; CK(cudaMemcpy(&nf, d_nfound, sizeof nf, cudaMemcpyDeviceToHost));
        if (nf > maxfound) { overflow += nf - maxfound; nf = maxfound; }
        if (nf) { CK(cudaMemcpy(buf.data(), d_found, sizeof(Found) * nf, cudaMemcpyDeviceToHost)); found.insert(found.end(), buf.begin(), buf.begin() + nf); }
    }
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    unsigned long long hist[MAXPRIM * (MAXIN + 1)], fbs[MAXPRIM * MAXSTEPS];
    CK(cudaMemcpy(hist, d_hist, sizeof hist, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(fbs, d_fbs, sizeof fbs, cudaMemcpyDeviceToHost));
    std::sort(found.begin(), found.end(), [](const Found &a, const Found &b) { return a.prim != b.prim ? a.prim < b.prim : a.q < b.q; });
    printf("machine\t%s,%s,%s,%s\nmode\tprimany\nW\t%d\nL\t%d\nsteps\t%d\nstart\t%llu\nprograms\t%llu\n",
           NAMES[table[0]], NAMES[table[1]], NAMES[table[2]], NAMES[table[3]], W, L, steps, start, count);
    char rg[8192];
    for (int k = 0; k < np; k++) {
        const Prim *p = &prims[k];
        printf("prim\t%s\ninputs\t%d\nexcluded_cell0\t%d\norder", p->name, p->nin, p->excl0);
        for (int i = 0; i < p->nin; i++) { printf("\t"); for (int j = 0; j < p->ninit; j++) printf("%s%d", j ? "," : "", p->init[i][j]); printf(":%d", p->target[i]); }
        printf("\nsurvivors");
        unsigned long long surv[MAXIN], alive = 0;
        for (int i = p->nin - 1; i >= 0; i--) { alive += hist[k * (MAXIN + 1) + i + 1]; surv[i] = alive; }
        for (int i = 0; i < p->nin; i++) printf("\t%llu", surv[i]);
        printf("\nfound\t%llu\nfound_by_step", hist[k * (MAXIN + 1) + p->nin]);
        for (int s = 0; s < steps; s++) printf("\t%llu", fbs[k * MAXSTEPS + s]);
        printf("\n");
        for (const Found &f : found) {
            if (f.prim != k) continue;
            char prog[MAXL + 1]; for (int i = 0; i < L; i++) prog[i] = (char)('0' + ((f.q >> (2 * (L - 1 - i))) & 3)); prog[L] = 0;
            ranges(f.stepmask, steps, rg, sizeof rg);
            printf("F\t%s\t%s\t%s\n", p->name, prog, rg);
        }
    }
    if (overflow) printf("found_overflow\t%llu\n", overflow);
    fflush(stdout);
    char host[64] = "?"; gethostname(host, sizeof host);
    fprintf(stderr, "mode=primany prims=%s ops=%s,%s,%s,%s W=%d L=%d steps=%d ncell=%d programs=%llu start=%llu secs=%.1f threads=0 host=%s device=gpu%d:%s tpb=%d smem=%zu%s\n",
            argv[1], NAMES[table[0]], NAMES[table[1]], NAMES[table[2]], NAMES[table[3]], W, L, steps, NCELL, count, start, secs, host, gpu, prop.name, tpb, smem, overflow ? " FOUND_OVERFLOW" : "");
    return overflow ? 4 : 0;
}
