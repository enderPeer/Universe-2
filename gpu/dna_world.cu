/* DNA-1 world engine on one GPU (docs/09_dna1_spec.md, sections 1 to 11).
 *
 * One ring of N cells of 2 bits in two strands (fwd executed, rev = comp(fwd) as backup), fixed slots of G cells,
 * one processor per expressible slot, a deterministic two-phase tick (execute in parallel with snapshot reads,
 * then ALLOC resolution, births, deaths, damage, repair, inflow, census). This file must produce exactly the same
 * world states as the Python reference sim/dna_ref.py on the same seed; the state hash of section 8 is the check.
 * Every choice the spec leaves open is listed in docs/09 section 13 and implemented here as written there.
 *
 *   nvcc -O3 -std=c++17 -arch=sm_89 -o dna_world dna_world.cu        (RTX 4090; compiles without -arch too)
 *   ./dna_world --N 16384 --G 256 --ticks 200 --seed 1 --hash-every 50
 *
 * Options: --N cells --G slot --ticks n --seed s --I instr/tick --T lifetime --mu prob --D auto|n --F auto|n
 *          --no-double --uniform-code --blank --genome "ATG GCT ..." --n-seed k --hash-every k (k>0: after every
 *          k-th tick and after the last, 0: last only; default 1 for N <= 2^24, else never) --no-hash --hash-init
 *          (also hash the initial state, printed as tick -1) --census-every k (default 10, 0: last only)
 *          --births FILE --dump FILE --out FILE --gpu i
 * stdout:  "tick <t> hash <16 hex>" = the state after tick t (t = 0 .. ticks-1, as sim/dna_ref.py prints it),
 *          "census {json}" lines with the reference's key order.
 * stderr:  configuration, then one summary line: ticks, seconds, ticks per second, device memory.
 *
 * Layout (section 1): cell i in byte i >> 2, bits 2*(i & 3)..+1. 64-bit indexing throughout (N = 3e9 is the
 * target). Processors live in a dense array indexed by slot with a live flag (section 9). Memory at N = 3e9,
 * G = 1024: 3 x 750 MB tape (fwd, rev, fwd_old) + 234 MB processors + ~60 MB slot arrays, about 2.6 GB.
 *
 * Honest caveats: the state hash needs the whole tape on the host (1.5 GB per hash at N = 3e9, about 2 s), so
 * big runs hash never by default (the reference defaults to every tick). The --out file holds the final census
 * as one compact JSON line (the reference indents it). The births log is capped at S + F entries per tick
 * (never exceeded: a processor divides at most once per tick). Verified against sim/dna_ref.py tick by tick on
 * the configurations t1 to t6 of docs/09 section 10 (see the task log of 2026-10-09).
 */
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <chrono>

typedef uint64_t u64; typedef int64_t i64; typedef uint32_t u32; typedef int32_t i32; typedef uint8_t u8;

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error at %s:%d: %s (%s)\n", __FILE__, __LINE__, cudaGetErrorString(e_), #x); exit(2); } } while (0)

/* ---------------------------------------------------------------- operations and the codon table (section 3) */
enum Op : u8 { NOP = 0, START, STOP, COPY, R1, L1, R3, L3, JNZB, SETP, SETQ, ALLOC, DIVIDE, SKNZ, SKZ, LDA, STA,
               INC, FLIP, SEEK, COPYR, REWIND, N_OPS };

/* letters ACGT, U read as T, case-insensitive (section 13 "Genome argument") */
static int letter(char c) { switch (c) { case 'A': case 'a': return 0; case 'C': case 'c': return 1; case 'G': case 'g': return 2;
                                         case 'T': case 't': case 'U': case 'u': return 3; } return -1; }

static int codon_value(const char *s)   /* "ATG" -> 16*a + 4*b + c, -1 if malformed */
{
    if (strlen(s) != 3) return -1;
    int a = letter(s[0]), b = letter(s[1]), c = letter(s[2]);
    if (a < 0 || b < 0 || c < 0) return -1;
    return 16 * a + 4 * b + c;
}

static void build_table(u8 *table, bool uniform)
{
    /* table order of section 3: START, STOP, then the 19 others as the table lists them */
    static const u8 order[21] = { START, STOP, COPY, R1, L1, JNZB, SETP, SETQ, ALLOC, DIVIDE, SKNZ, SKZ, LDA, STA,
                                  INC, FLIP, R3, L3, SEEK, COPYR, REWIND };
    if (uniform) { for (int i = 0; i < 64; i++) table[i] = i < 21 ? order[i] : (u8)NOP; return; }
    struct Row { const char *codons; u8 op; };
    static const Row rows[] = {
        { "ATG", START }, { "TAA TAG TGA", STOP }, { "TTA TTG CTT CTC CTA CTG", COPY },
        { "TCT TCC TCA TCG AGT AGC", R1 }, { "CGT CGC CGA CGG AGA AGG", L1 }, { "GTT GTC GTA GTG", JNZB },
        { "CCT CCC CCA CCG", SETP }, { "ACT ACC ACA ACG", SETQ }, { "GCT GCC GCA GCG", ALLOC },
        { "GGT GGC GGA GGG", DIVIDE }, { "ATT ATC ATA", SKNZ }, { "TTT TTC", SKZ }, { "TAT TAC", LDA },
        { "CAT CAC", STA }, { "CAA CAG", INC }, { "AAT AAC", FLIP }, { "AAA AAG", R3 }, { "GAT GAC", L3 },
        { "GAA GAG", SEEK }, { "TGT TGC", COPYR }, { "TGG", REWIND } };
    int seen[64] = { 0 };
    for (const Row &r : rows) {
        char buf[128]; strncpy(buf, r.codons, sizeof buf - 1); buf[sizeof buf - 1] = 0;
        for (char *tok = strtok(buf, " "); tok; tok = strtok(NULL, " ")) {
            int v = codon_value(tok);
            if (v < 0 || seen[v]) { fprintf(stderr, "codon table error at %s\n", tok); exit(1); }
            table[v] = r.op; seen[v] = 1;
        }
    }
    for (int i = 0; i < 64; i++) if (!seen[i]) { fprintf(stderr, "codon table incomplete at %d\n", i); exit(1); }
}

/* ---------------------------------------------------------------- random numbers (section 7) */
#define K1 0x9E3779B97F4A7C15ULL
#define K2 0xBF58476D1CE4E5B9ULL
#define K3 0x94D049BB133111EBULL

__host__ __device__ inline u64 splitmix64(u64 x)
{
    u64 z = x + K1;
    z = (z ^ (z >> 30)) * K2;
    z = (z ^ (z >> 27)) * K3;
    return z ^ (z >> 31);
}

/* rnd(t, k, j) = splitmix64(seed xor mix(t, k, j)); t = -1 allowed (initial state); all unsigned wraparound */
__host__ __device__ inline u64 rnd(u64 seed, i64 t, u64 k, u64 j)
{
    u64 mix = (u64)(t + 1) * K1 + k * K2 + j * K3;
    return splitmix64(seed ^ mix);
}

/* ---------------------------------------------------------------- world state */
struct Params {
    u64 seed, N;
    i64 G, S, I, T, maxk;     /* maxk = last usable codon index = G / 3 - 1 */
    double mu;
    i64 D, F;
    int atomic_writes;        /* 1 when G % 4 != 0: a byte may straddle two slots, so cell writes use word atomics */
};

/* 80 bytes, the ten 64-bit fields of section 4 in the hash order of section 8 */
struct Proc { i64 slot, pc, H, P, Q, child, A, age, depth, copied; };

struct BirthRec { i64 parent, child; u64 phash, chash; };

struct World {
    u8 *fwd, *rev, *fwd_old;  /* packed 2-bit strands; fwd_old = snapshot at the start of the tick */
    Proc *proc;               /* dense, indexed by slot */
    u8 *live;                 /* 1 if slot s has a processor */
    i32 *claimed_by;          /* parent slot that claimed s, or -1 */
    i32 *claim;               /* ALLOC resolution scratch: lowest requesting slot per candidate (atomicMin) */
    u8 *alloc_req;            /* ALLOC requested this tick */
    i32 *div_mark;            /* child slot marked for birth by DIVIDE this tick, or -1 */
    i32 *refill_k;            /* inflow scratch: largest k refilling this slot this tick, or -1 */
    u64 *ghash;               /* gene hash per live processor (set at birth, refreshed for the census) */
    i32 *cens_depth;          /* census scratch: depth of live processor or -1 */
    u64 *counters;            /* births, faithful, spontaneous, deaths (cumulative) */
    BirthRec *log; u32 *log_n; u32 log_cap;   /* births of this tick (only when --births is given) */
    u8 *table;                /* the codon table (64 entries) in device memory */
};

/* ---------------------------------------------------------------- 2-bit cell access (section 1 layout) */
__host__ __device__ inline u32 get2(const u8 *a, u64 i) { return (a[i >> 2] >> ((i & 3) * 2)) & 3u; }

__host__ __device__ inline void set2_plain(u8 *a, u64 i, u32 v)
{
    u32 sh = (u32)(i & 3) * 2;
    a[i >> 2] = (u8)((a[i >> 2] & ~(3u << sh)) | (v << sh));
}

#ifdef __CUDACC__
/* word-atomic variant, safe when several threads write different cells of the same byte (or the same cell with
 * the same value: and-then-or converges to v in every interleaving) */
__device__ inline void set2_atomic(u8 *a, u64 i, u32 v)
{
    u32 *w = (u32 *)(a + ((i >> 4) << 2));
    u32 sh = (u32)(i & 15) * 2;
    atomicAnd(w, ~(3u << sh));
    atomicOr(w, v << sh);
}
__device__ inline void set2(u8 *a, u64 i, u32 v, int atomic) { if (atomic) set2_atomic(a, i, v); else set2_plain(a, i, v); }
#endif

/* ---------------------------------------------------------------- genes (section 2) */
__host__ __device__ inline u32 codon_at(const u8 *f, u64 base, i64 k)
{
    u64 i = base + 3 * (u64)k;
    return 16 * get2(f, i) + 4 * get2(f, i + 1) + get2(f, i + 2);
}

/* codons from 0 up to and excluding the first STOP; maxk + 1 if there is no STOP */
__host__ __device__ inline i64 gene_len(const u8 *f, u64 base, const u8 *table, i64 maxk)
{
    for (i64 k = 0; k <= maxk; k++) if (table[codon_at(f, base, k)] == STOP) return k;
    return maxk + 1;
}

__host__ __device__ inline bool expressible(const u8 *f, u64 base, const u8 *table, i64 maxk)
{
    return table[codon_at(f, base, 0)] == START && gene_len(f, base, table, maxk) >= 2;
}

/* FNV-1a 64 over the codon values of the gene as bytes, START included, STOP excluded */
__host__ __device__ inline u64 gene_hash(const u8 *f, u64 base, const u8 *table, i64 maxk)
{
    u64 h = 14695981039346656037ULL;
    for (i64 k = 0; k <= maxk; k++) {
        u32 c = codon_at(f, base, k);
        if (table[c] == STOP) break;
        h = (h ^ (u64)c) * 1099511628211ULL;
    }
    return h;
}

/* ---------------------------------------------------------------- kernels */
__device__ inline i64 modN(i64 x, i64 N) { x %= N; return x < 0 ? x + N : x; }

/* initial state (section 6): fwd[i] = rnd(-1, i, 0) mod 4 or 0 with --blank; rev = comp; padding bits stay 0 */
__global__ void k_init(World w, Params P, u64 nbytes, int blank)
{
    for (u64 b = (u64)blockIdx.x * blockDim.x + threadIdx.x; b < nbytes; b += (u64)gridDim.x * blockDim.x) {
        u32 f = 0, r = 0;
        for (u32 q = 0; q < 4; q++) {
            u64 i = b * 4 + q;
            if (i >= P.N) break;
            u32 v = blank ? 0u : (u32)(rnd(P.seed, -1, i, 0) & 3u);
            f |= v << (2 * q); r |= (v ^ 3u) << (2 * q);
        }
        w.fwd[b] = (u8)f; w.rev[b] = (u8)r;
    }
}

/* step 1: one thread per slot, I instructions of its processor */
__global__ void k_execute(World w, Params P, i64 t)
{
    i64 s = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= P.S || !w.live[s]) return;
    Proc p = w.proc[s];
    const i64 N = (i64)P.N, G = P.G, maxk = P.maxk;
    const u8 *table = w.table;
    const i64 base = s * G;
    i64 cbase = p.child >= 0 ? p.child * G : -1;
#define IN_OWN(i)   ((i) >= base && (i) < base + G)
#define IN_CHILD(i) (cbase >= 0 && (i) >= cbase && (i) < cbase + G)
    /* snapshot rule: own slot and claimed child live, everything else from the tick-start copy */
#define RD(i)       ((IN_OWN(i) || IN_CHILD(i)) ? get2(w.fwd, (u64)(i)) : get2(w.fwd_old, (u64)(i)))
#define WR(i, v)    do { set2(w.fwd, (u64)(i), (v), P.atomic_writes); set2(w.rev, (u64)(i), (v) ^ 3u, P.atomic_writes); } while (0)
    i64 len = gene_len(w.fwd, base, table, maxk);
    for (i64 n = 0; n < P.I; n++) {
        u32 op = NOP;
        if (p.pc >= 0 && p.pc <= maxk) op = table[codon_at(w.fwd, base, p.pc)];   /* gene fetched live */
        p.pc += 1;
        switch (op) {
        case NOP: case START: case STOP: break;
        case R1: p.H += 1; if (p.H >= N) p.H -= N; break;
        case L1: p.H -= 1; if (p.H < 0) p.H += N; break;
        case R3: p.H += 3; if (p.H >= N) p.H -= N; break;
        case L3: p.H -= 3; if (p.H < 0) p.H += N; break;
        case INC: case FLIP: case STA:
            if (IN_OWN(p.H) || IN_CHILD(p.H)) {
                u32 v = get2(w.fwd, (u64)p.H);
                v = op == INC ? (v + 1) & 3u : op == FLIP ? v ^ 1u : (u32)p.A;
                WR(p.H, v);
                if (IN_OWN(p.H)) len = gene_len(w.fwd, base, table, maxk);   /* the own gene may have changed */
            }
            break;
        case LDA: p.A = RD(p.H); break;
        case SKZ: if (RD(p.H) == 0) p.pc += 1; break;
        case SKNZ: if (RD(p.H) != 0) p.pc += 1; break;
        case JNZB: if (RD(p.H) != 0) { p.pc -= 2; if (p.pc < 1) p.pc = 1; } break;
        case SETP: p.P = p.H; break;
        case SETQ: p.Q = p.H; break;
        case REWIND: p.P = base; if (cbase >= 0) p.Q = cbase; break;
        case SEEK: {
            if (p.pc >= len) break;                       /* template beyond the gene end: NOP */
            u32 tc = codon_at(w.fwd, base, p.pc);         /* the template codon, consumed */
            p.pc += 1;
            u32 t0 = (tc >> 4) ^ 3u, t1 = ((tc >> 2) & 3u) ^ 3u, t2 = (tc & 3u) ^ 3u;
            for (i64 d = 1; d <= 3 * G; d++) {
                i64 h0 = modN(p.H + d, N), h1 = modN(h0 + 1, N), h2 = modN(h0 + 2, N);
                if (RD(h0) == t0 && RD(h1) == t1 && RD(h2) == t2) { p.H = h0; break; }
            }
        } break;
        case ALLOC: if (p.child < 0) w.alloc_req[s] = 1; break;
        case COPY: case COPYR:
            if (p.child >= 0 && IN_CHILD(p.Q)) {
                u32 v = RD(p.P);
                if (op == COPYR) v ^= 3u;
                u64 r = rnd(P.seed, t, (u64)s, 1000 + (u64)p.copied);
                if ((double)(r >> 32) * (1.0 / 4294967296.0) < P.mu) v = (v + 1 + (u32)(r % 3)) & 3u;
                WR(p.Q, v);
                p.copied += 1;
            }
            p.P += op == COPY ? 1 : -1; if (p.P >= N) p.P -= N; if (p.P < 0) p.P += N;
            p.Q += 1; if (p.Q >= N) p.Q -= N;
            break;
        case DIVIDE:
            if (p.child >= 0 && p.copied >= 6) { w.div_mark[s] = (i32)p.child; p.child = -1; cbase = -1; p.copied = 0; p.Q = 0; }
            break;
        }
        if (p.pc >= len) p.pc = 1;                        /* the gene loops, skipping START */
    }
    w.proc[s] = p;
#undef IN_OWN
#undef IN_CHILD
#undef RD
#undef WR
}

/* step 2a: every requesting processor proposes its candidate; the lowest slot index wins (unsigned atomicMin
 * against the sentinel 0xFFFFFFFF, which no slot index reaches since S < 2^31) */
__global__ void k_alloc_propose(World w, Params P, i64 t)
{
    i64 s = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= P.S || !w.live[s] || !w.alloc_req[s]) return;
    i64 cand = (s + 1 + (i64)(rnd(P.seed, t, (u64)s, 0) % (u64)(P.S - 1))) % P.S;
    if (!w.live[cand] && w.claimed_by[cand] < 0) atomicMin((unsigned *)&w.claim[cand], (unsigned)s);
}

/* step 2b: grants */
__global__ void k_alloc_grant(World w, Params P, i64 t)
{
    i64 s = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= P.S || !w.live[s] || !w.alloc_req[s]) return;
    w.alloc_req[s] = 0;
    i64 cand = (s + 1 + (i64)(rnd(P.seed, t, (u64)s, 0) % (u64)(P.S - 1))) % P.S;
    if ((u32)w.claim[cand] == (u32)s) {
        w.proc[s].child = cand; w.proc[s].Q = cand * P.G; w.proc[s].copied = 0;
        w.claimed_by[cand] = (i32)s;
    }
}

/* step 3: births from DIVIDE marks; the claim on the child is released either way */
__global__ void k_births(World w, Params P, i64 t)
{
    i64 s = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= P.S || w.div_mark[s] < 0) return;
    i64 c = w.div_mark[s];
    w.div_mark[s] = -1;
    w.claimed_by[c] = -1;
    if (!expressible(w.fwd, (u64)c * P.G, w.table, P.maxk)) return;
    Proc q; q.slot = c; q.pc = 1; q.H = c * P.G; q.P = c * P.G; q.Q = 0; q.child = -1; q.A = 0; q.age = 0;
    q.depth = w.proc[s].depth + 1; q.copied = 0;
    w.proc[c] = q; w.live[c] = 1;
    u64 ph = gene_hash(w.fwd, (u64)s * P.G, w.table, P.maxk), ch = gene_hash(w.fwd, (u64)c * P.G, w.table, P.maxk);
    w.ghash[c] = ch;
    atomicAdd((unsigned long long *)&w.counters[0], 1ULL);
    if (ph == ch) atomicAdd((unsigned long long *)&w.counters[1], 1ULL);
    if (w.log) { u32 n = atomicAdd(w.log_n, 1u); if (n < w.log_cap) { BirthRec r = { s, c, ph, ch }; w.log[n] = r; } }
}

/* step 4: ageing and deaths */
__global__ void k_deaths(World w, Params P)
{
    i64 s = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= P.S || !w.live[s]) return;
    Proc *p = &w.proc[s];
    p->age += 1;
    if (p->age >= P.T || !expressible(w.fwd, (u64)s * P.G, w.table, P.maxk)) {
        w.live[s] = 0;
        atomicAdd((unsigned long long *)&w.counters[3], 1ULL);
        if (p->child >= 0) w.claimed_by[p->child] = -1;
    }
}

/* step 5: D damage hits on fwd only (xor commutes, so hits are independent) */
__global__ void k_damage(World w, Params P, i64 t)
{
    i64 k = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= P.D) return;
    u64 i = rnd(P.seed, t, (u64)k, 1) % P.N;
    u32 x = 1 + (u32)(rnd(P.seed, t, (u64)k, 2) % 3);
    u32 *wd = (u32 *)(w.fwd + ((i >> 4) << 2));
    atomicXor(wd, x << ((u32)(i & 15) * 2));
}

/* step 6: repair of the cells hit this tick from the reverse strand */
__global__ void k_repair(World w, Params P, i64 t)
{
    i64 k = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= P.D) return;
    u64 i = rnd(P.seed, t, (u64)k, 1) % P.N;
    set2_atomic(w.fwd, i, get2(w.rev, i) ^ 3u);
}

/* step 7a: inflow, which k refills which slot. All refills happen before any birth (section 13); when several k
 * select the same dead unclaimed slot the largest k wins, so an atomicMax per slot picks the winner. */
__global__ void k_inflow_claim(World w, Params P, i64 t)
{
    i64 k = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= P.F) return;
    i64 d = (i64)(rnd(P.seed, t, (u64)k, 3) % (u64)P.S);
    if (!w.live[d] && w.claimed_by[d] < 0) atomicMax(&w.refill_k[d], (i32)k);
}

/* step 7b: one block per k; the winning k writes rnd(t, k, 4 + j) mod 4 into cell j of its slot. When G is a
 * multiple of 4 every byte belongs to one slot, so one thread per byte with plain stores is race-free. Otherwise
 * the first and last byte of a slot can straddle into a neighbouring slot that another block refills in the same
 * tick, so every cell is then written with the word atomics (section 13 "Engine detail"); the and/or sequences of
 * the two blocks converge because they touch disjoint bits. */
__global__ void k_inflow_fill(World w, Params P, i64 t)
{
    i64 k = blockIdx.x;
    i64 d = (i64)(rnd(P.seed, t, (u64)k, 3) % (u64)P.S);
    if (w.refill_k[d] != (i32)k) return;
    u64 base = (u64)d * P.G, b0 = base >> 2, b1 = (base + P.G - 1) >> 2;
    if (P.atomic_writes) {
        for (u64 j = threadIdx.x; j < (u64)P.G; j += blockDim.x) {
            u32 v = (u32)(rnd(P.seed, t, (u64)k, 4 + j) & 3u);
            set2_atomic(w.fwd, base + j, v);
            set2_atomic(w.rev, base + j, v ^ 3u);
        }
        return;
    }
    for (u64 b = b0 + threadIdx.x; b <= b1; b += blockDim.x) {
        u32 f = w.fwd[b], r = w.rev[b];
        for (u32 q = 0; q < 4; q++) {
            u64 i = b * 4 + q;
            if (i < base || i >= base + P.G) continue;
            u32 v = (u32)(rnd(P.seed, t, (u64)k, 4 + (i - base)) & 3u);
            f = (f & ~(3u << (2 * q))) | (v << (2 * q));
            r = (r & ~(3u << (2 * q))) | ((v ^ 3u) << (2 * q));
        }
        w.fwd[b] = (u8)f; w.rev[b] = (u8)r;
    }
}

/* step 7c: spontaneous births in the refilled slots (independent slots, so the order is irrelevant for the
 * state; the host sorts the log by slot). Resets the refill marks. */
__global__ void k_inflow_births(World w, Params P, i64 t)
{
    i64 d = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= P.S || w.refill_k[d] < 0) return;
    w.refill_k[d] = -1;
    u64 base = (u64)d * P.G;
    if (!expressible(w.fwd, base, w.table, P.maxk)) return;
    Proc q; q.slot = d; q.pc = 1; q.H = (i64)base; q.P = (i64)base; q.Q = 0; q.child = -1; q.A = 0;
    q.age = 0; q.depth = 0; q.copied = 0;
    w.proc[d] = q; w.live[d] = 1;
    u64 ch = gene_hash(w.fwd, base, w.table, P.maxk);
    w.ghash[d] = ch;
    atomicAdd((unsigned long long *)&w.counters[0], 1ULL);
    atomicAdd((unsigned long long *)&w.counters[2], 1ULL);
    if (w.log) { u32 n = atomicAdd(w.log_n, 1u); if (n < w.log_cap) { BirthRec r = { -1, d, 0, ch }; w.log[n] = r; } }
}

/* census scratch: depth (-1 if dead) and a fresh gene hash per live processor */
__global__ void k_census(World w, Params P)
{
    i64 s = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= P.S) return;
    if (w.live[s]) { w.cens_depth[s] = (i32)w.proc[s].depth; w.ghash[s] = gene_hash(w.fwd, (u64)s * P.G, w.table, P.maxk); }
    else w.cens_depth[s] = -1;
}

/* ---------------------------------------------------------------- host side */
static inline u64 fnv_bytes(u64 h, const u8 *p, u64 n) { for (u64 i = 0; i < n; i++) h = (h ^ p[i]) * 1099511628211ULL; return h; }
static inline u64 fnv_u64le(u64 h, u64 v) { u8 b[8]; for (int i = 0; i < 8; i++) b[i] = (u8)(v >> (8 * i)); return fnv_bytes(h, b, 8); }
static inline void put_u64le(FILE *f, u64 v) { u8 b[8]; for (int i = 0; i < 8; i++) b[i] = (u8)(v >> (8 * i)); fwrite(b, 1, 8, f); }

struct Host {
    Params P; World w; u64 nbytes; u8 *table;
    std::vector<u8> fwd, rev, live; std::vector<Proc> proc; std::vector<u64> ghash; std::vector<i32> depth;
    u64 counters[4] = { 0, 0, 0, 0 };
};

static void fetch_state(Host &h)   /* the bytes of section 8 */
{
    CUDA_CHECK(cudaMemcpy(h.fwd.data(), h.w.fwd, h.nbytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h.rev.data(), h.w.rev, h.nbytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h.live.data(), h.w.live, h.P.S, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h.proc.data(), h.w.proc, h.P.S * sizeof(Proc), cudaMemcpyDeviceToHost));
}

static u64 state_hash(Host &h)
{
    fetch_state(h);
    u64 x = 14695981039346656037ULL;
    x = fnv_bytes(x, h.fwd.data(), h.nbytes);
    x = fnv_bytes(x, h.rev.data(), h.nbytes);
    for (i64 s = 0; s < h.P.S; s++) if (h.live[s]) {
        const Proc &p = h.proc[s];
        const i64 f[10] = { p.slot, p.pc, p.H, p.P, p.Q, p.child, p.A, p.age, p.depth, p.copied };
        for (int i = 0; i < 10; i++) x = fnv_u64le(x, (u64)f[i]);
    }
    for (int i = 0; i < 4; i++) x = fnv_u64le(x, h.counters[i]);
    return x;
}

static void dump_state(Host &h, const char *path)
{
    fetch_state(h);
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", path); return; }
    fwrite(h.fwd.data(), 1, h.nbytes, f);
    fwrite(h.rev.data(), 1, h.nbytes, f);
    for (i64 s = 0; s < h.P.S; s++) if (h.live[s]) {
        const Proc &p = h.proc[s];
        const i64 v[10] = { p.slot, p.pc, p.H, p.P, p.Q, p.child, p.A, p.age, p.depth, p.copied };
        for (int i = 0; i < 10; i++) put_u64le(f, (u64)v[i]);
    }
    for (int i = 0; i < 4; i++) put_u64le(f, h.counters[i]);
    fclose(f);
}

/* census (step 8) as one JSON object; since-last counters are relative to prev[] */
static std::string census_json(Host &h, i64 tick, const u64 prev[4], bool have_hash, u64 hash)
{
    k_census<<<(unsigned)((h.P.S + 255) / 256), 256>>>(h.w, h.P);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h.depth.data(), h.w.cens_depth, h.P.S * sizeof(i32), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h.ghash.data(), h.w.ghash, h.P.S * sizeof(u64), cudaMemcpyDeviceToHost));
    std::vector<u64> hs; std::map<i32, u64> dh;
    for (i64 s = 0; s < h.P.S; s++) if (h.depth[s] >= 0) { hs.push_back(h.ghash[s]); dh[h.depth[s]]++; }
    std::sort(hs.begin(), hs.end());
    std::vector<std::pair<u64, u64>> runs;   /* (count, hash) */
    for (size_t i = 0; i < hs.size();) { size_t j = i; while (j < hs.size() && hs[j] == hs[i]) j++; runs.push_back({ (u64)(j - i), hs[i] }); i = j; }
    std::sort(runs.begin(), runs.end(), [](const std::pair<u64, u64> &a, const std::pair<u64, u64> &b) {
        return a.first != b.first ? a.first > b.first : a.second < b.second; });
    /* same key order and separators as the reference's json.dumps */
    char buf[512]; std::string js = "{";
    snprintf(buf, sizeof buf, "\"tick\": %lld, \"live\": %zu, \"births\": %llu, \"faithful\": %llu, \"spontaneous\": %llu, \"deaths\": %llu, ",
             (long long)tick, hs.size(), (unsigned long long)(h.counters[0] - prev[0]), (unsigned long long)(h.counters[1] - prev[1]),
             (unsigned long long)(h.counters[2] - prev[2]), (unsigned long long)(h.counters[3] - prev[3]));
    js += buf;
    snprintf(buf, sizeof buf, "\"births_total\": %llu, \"faithful_total\": %llu, \"spontaneous_total\": %llu, \"deaths_total\": %llu, ",
             (unsigned long long)h.counters[0], (unsigned long long)h.counters[1], (unsigned long long)h.counters[2], (unsigned long long)h.counters[3]);
    js += buf;
    snprintf(buf, sizeof buf, "\"distinct_genes\": %zu, \"top_genes\": [", runs.size()); js += buf;
    for (size_t i = 0; i < runs.size() && i < 10; i++) {
        snprintf(buf, sizeof buf, "%s[\"%016llx\", %llu]", i ? ", " : "", (unsigned long long)runs[i].second, (unsigned long long)runs[i].first); js += buf;
    }
    js += "], \"depth_hist\": [";
    bool first = true;
    for (auto &kv : dh) { snprintf(buf, sizeof buf, "%s[%d, %llu]", first ? "" : ", ", kv.first, (unsigned long long)kv.second); js += buf; first = false; }
    js += "], \"hash\": ";
    if (have_hash) { snprintf(buf, sizeof buf, "\"%016llx\"", (unsigned long long)hash); js += buf; } else js += "null";
    js += "}";
    return js;
}

/* write the first ncells cell values of slot `base` into both strands on the device, via the host (seeding only) */
static void write_slot(Host &h, u64 base, const std::vector<u8> &vals, i64 ncells)
{
    u64 b0 = base >> 2, b1 = (base + ncells - 1) >> 2, n = b1 - b0 + 1;
    std::vector<u8> f(n), r(n);
    CUDA_CHECK(cudaMemcpy(f.data(), h.w.fwd + b0, n, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(r.data(), h.w.rev + b0, n, cudaMemcpyDeviceToHost));
    for (i64 j = 0; j < ncells; j++) { set2_plain(f.data(), base - b0 * 4 + j, vals[j]); set2_plain(r.data(), base - b0 * 4 + j, vals[j] ^ 3u); }
    CUDA_CHECK(cudaMemcpy(h.w.fwd + b0, f.data(), n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(h.w.rev + b0, r.data(), n, cudaMemcpyHostToDevice));
}

static void usage()
{
    fprintf(stderr, "usage: dna_world --N cells --G slot --ticks n --seed s [--I 16 --T 64 --mu 0.001 --D auto --F auto\n"
                    "  --no-double --uniform-code --blank --genome \"ATG GCT ...\" --n-seed k --hash-every k --no-hash --hash-init\n"
                    "  --census-every k --births FILE --dump FILE --out FILE --gpu i]\n");
    exit(1);
}

int main(int argc, char **argv)
{
    /* default N = 2,929,687 x 1024 = 2,999,999,488: the S of section 9 (3e9 itself is not a multiple of 1024;
     * section 13 "Default N"). D = F = -1 means "auto" (section 9). */
    u64 N = 2999999488ULL, seed = 1; i64 G = 1024, ticks = 100, I = 16, T = 64, D = -1, F = -1, n_seed = 0;
    double mu = 0.001; bool dbl = true, uniform = false, blank = false, no_hash = false, hash_init = false; i64 hash_every = -2, census_every = 10;
    std::string genome, births_path, dump_path, out_path; int gpu = 0;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char * { if (i + 1 >= argc) usage(); return argv[++i]; };
        if (a == "--N") N = strtoull(next(), NULL, 10);
        else if (a == "--G") G = atoll(next());
        else if (a == "--ticks") ticks = atoll(next());
        else if (a == "--seed") seed = strtoull(next(), NULL, 10);
        else if (a == "--I") I = atoll(next());
        else if (a == "--T") T = atoll(next());
        else if (a == "--mu") mu = atof(next());
        else if (a == "--D") { const char *v = next(); if (!strcmp(v, "auto")) D = -1; else { D = atoll(v);
                               if (D < 0) { fprintf(stderr, "--D must be 'auto' or >= 0\n"); return 1; } } }   /* section 13 Limits */
        else if (a == "--F") { const char *v = next(); if (!strcmp(v, "auto")) F = -1; else { F = atoll(v);
                               if (F < 0) { fprintf(stderr, "--F must be 'auto' or >= 0\n"); return 1; } } }
        else if (a == "--no-double") dbl = false;
        else if (a == "--uniform-code") uniform = true;
        else if (a == "--blank") blank = true;
        else if (a == "--genome") genome = next();
        else if (a == "--n-seed") n_seed = atoll(next());
        else if (a == "--hash-every") hash_every = atoll(next());
        else if (a == "--no-hash") no_hash = true;
        else if (a == "--hash-init") hash_init = true;
        else if (a == "--census-every") census_every = atoll(next());
        else if (a == "--births") births_path = next();
        else if (a == "--dump") dump_path = next();
        else if (a == "--out") out_path = next();
        else if (a == "--gpu") gpu = atoi(next());
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); usage(); }
    }
    if (G < 6 || N % (u64)G != 0) { fprintf(stderr, "N must be a multiple of G and G >= 6\n"); return 1; }
    if (ticks < 1) { fprintf(stderr, "--ticks must be at least 1\n"); return 1; }   /* section 13 Limits */
    i64 S = (i64)(N / (u64)G);
    if (S < 2 || S >= (1LL << 31)) { fprintf(stderr, "need 2 <= S < 2^31 slots (S = %lld)\n", (long long)S); return 1; }
    if (D < 0) D = std::max<i64>(1, (i64)(N >> 20));
    if (F < 0) F = std::max<i64>(1, S >> 12);
    if (no_hash) hash_every = -1; else if (hash_every == -2) hash_every = N <= (1ULL << 24) ? 1 : -1;
    if (n_seed > 0 && genome.empty()) { fprintf(stderr, "--n-seed needs --genome\n"); return 1; }
    if (n_seed > S) { fprintf(stderr, "--n-seed larger than S\n"); return 1; }

    Host h;
    Params &P = h.P;
    P.seed = seed; P.N = N; P.G = G; P.S = S; P.I = I; P.T = T; P.maxk = G / 3 - 1; P.mu = mu; P.D = D; P.F = F;
    P.atomic_writes = (G % 4) != 0;
    u8 table[64]; build_table(table, uniform);

    CUDA_CHECK(cudaSetDevice(gpu));
    size_t mem_free0, mem_total; CUDA_CHECK(cudaMemGetInfo(&mem_free0, &mem_total));
    h.nbytes = (N + 3) / 4;
    u64 nalloc = (h.nbytes + 15) & ~15ULL;      /* word atomics may touch the last partial word */
    size_t allocated = 0;
    auto dalloc = [&](void **p, size_t n) { CUDA_CHECK(cudaMalloc(p, n)); CUDA_CHECK(cudaMemset(*p, 0, n)); allocated += n; };
    World &w = h.w;
    memset(&w, 0, sizeof w);
    dalloc((void **)&w.fwd, nalloc); dalloc((void **)&w.rev, nalloc); dalloc((void **)&w.fwd_old, nalloc);
    dalloc((void **)&w.proc, S * sizeof(Proc)); dalloc((void **)&w.live, S);
    dalloc((void **)&w.claimed_by, S * sizeof(i32)); dalloc((void **)&w.claim, S * sizeof(i32));
    dalloc((void **)&w.alloc_req, S); dalloc((void **)&w.div_mark, S * sizeof(i32)); dalloc((void **)&w.refill_k, S * sizeof(i32));
    dalloc((void **)&w.ghash, S * sizeof(u64)); dalloc((void **)&w.cens_depth, S * sizeof(i32));
    dalloc((void **)&w.counters, 4 * sizeof(u64)); dalloc((void **)&w.table, 64);
    CUDA_CHECK(cudaMemset(w.claimed_by, 0xFF, S * sizeof(i32)));
    CUDA_CHECK(cudaMemset(w.div_mark, 0xFF, S * sizeof(i32)));
    CUDA_CHECK(cudaMemset(w.refill_k, 0xFF, S * sizeof(i32)));
    CUDA_CHECK(cudaMemset(w.claim, 0xFF, S * sizeof(i32)));   /* sentinel 0xFFFFFFFF: above every slot index */
    CUDA_CHECK(cudaMemcpy(w.table, table, 64, cudaMemcpyHostToDevice));
    FILE *births_f = NULL; std::vector<BirthRec> log_host;
    if (!births_path.empty()) {
        w.log_cap = (u32)std::min<i64>(S + F, (1LL << 31) - 1);
        dalloc((void **)&w.log, (size_t)w.log_cap * sizeof(BirthRec)); dalloc((void **)&w.log_n, sizeof(u32));
        births_f = fopen(births_path.c_str(), "w");
        if (!births_f) { fprintf(stderr, "cannot write %s\n", births_path.c_str()); return 1; }
        fprintf(births_f, "#tick\tparent\tchild\tparent_hash\tchild_hash\tfaithful\n");
        log_host.resize(w.log_cap);
    }
    h.fwd.resize(h.nbytes); h.rev.resize(h.nbytes); h.live.resize(S); h.proc.resize(S); h.ghash.resize(S); h.depth.resize(S);

    fprintf(stderr, "dna_world N=%llu G=%lld S=%lld ticks=%lld seed=%llu I=%lld T=%lld mu=%g D=%lld F=%lld double=%d code=%s blank=%d "
                    "n_seed=%lld hash_every=%lld census_every=%lld gpu=%d\n", (unsigned long long)N, (long long)G, (long long)S,
            (long long)ticks, (unsigned long long)seed, (long long)I, (long long)T, mu, (long long)D, (long long)F, dbl ? 1 : 0,
            uniform ? "uniform" : "standard", blank ? 1 : 0, (long long)n_seed, (long long)hash_every, (long long)census_every, gpu);

    auto t_start = std::chrono::steady_clock::now();
    /* initial state (section 6) */
    {
        u64 blocks = std::min<u64>((h.nbytes + 255) / 256, 1u << 20);
        k_init<<<(unsigned)blocks, 256>>>(w, P, h.nbytes, blank ? 1 : 0);
        CUDA_CHECK(cudaGetLastError());
    }
    if (n_seed > 0) {
        /* section 13 "Genome argument": codons separated by whitespace or commas, or one contiguous letter string
         * whose length is a multiple of 3; letters are case-insensitive (letter() above) */
        std::vector<std::string> toks;
        char *buf = strdup(genome.c_str());
        for (char *tok = strtok(buf, " \t\r\n\v\f,"); tok; tok = strtok(NULL, " \t\r\n\v\f,")) toks.push_back(tok);
        free(buf);
        if (toks.size() == 1 && toks[0].size() > 3) {
            std::string s = toks[0];
            if (s.size() % 3) { fprintf(stderr, "--genome length %zu is not a multiple of 3\n", s.size()); return 1; }
            toks.clear();
            for (size_t i = 0; i < s.size(); i += 3) toks.push_back(s.substr(i, 3));
        }
        std::vector<int> codons;
        for (const std::string &tok : toks) {
            int v = codon_value(tok.c_str());
            if (v < 0) { fprintf(stderr, "bad codon %s in --genome\n", tok.c_str()); return 1; }
            codons.push_back(v);
        }
        if ((i64)codons.size() > P.maxk + 1) { fprintf(stderr, "--genome longer than the slot\n"); return 1; }
        int stop = -1; for (int c = 0; c < 64 && stop < 0; c++) if (table[c] == STOP) stop = c;   /* lowest STOP codon pads */
        std::vector<u8> vals(G, 0);
        for (i64 k = 0; k <= P.maxk; k++) {
            int c = k < (i64)codons.size() ? codons[k] : stop;
            vals[3 * k] = (u8)(c >> 4); vals[3 * k + 1] = (u8)((c >> 2) & 3); vals[3 * k + 2] = (u8)(c & 3);
        }
        /* a packed copy of the seeded slot for its gene hash and expressibility (host side) */
        std::vector<u8> packed((G + 3) / 4 + 4, 0);
        for (i64 j = 0; j < G; j++) set2_plain(packed.data(), (u64)j, vals[j]);
        u64 gh = gene_hash(packed.data(), 0, table, P.maxk);
        for (i64 k = 0; k < n_seed; k++) {
            i64 s = k * (S / n_seed); u64 base = (u64)s * G;
            write_slot(h, base, vals, 3 * (P.maxk + 1));   /* the G mod 3 trailing cells keep their soup content */
            Proc q; q.slot = s; q.pc = 1; q.H = (i64)base; q.P = (i64)base; q.Q = 0; q.child = -1; q.A = 0; q.age = 0; q.depth = 0; q.copied = 0;
            u8 one = 1;
            CUDA_CHECK(cudaMemcpy(w.proc + s, &q, sizeof q, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(w.live + s, &one, 1, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(w.ghash + s, &gh, sizeof gh, cudaMemcpyHostToDevice));
        }
        fprintf(stderr, "seeded %lld slots with %zu codons, gene hash %016llx, expressible=%d\n", (long long)n_seed, codons.size(),
                (unsigned long long)gh, expressible(packed.data(), 0, table, P.maxk) ? 1 : 0);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    size_t mem_free1; CUDA_CHECK(cudaMemGetInfo(&mem_free1, &mem_total));

    const unsigned sgrid = (unsigned)((S + 255) / 256), dgrid = (unsigned)((D + 255) / 256), fgrid = (unsigned)((F + 255) / 256);
    u64 prev[4] = { 0, 0, 0, 0 };
    /* convention of section 13 (same as sim/dna_ref.py): "tick t" labels the state after tick t (t = 0 .. ticks-1,
     * the tick that used t in rnd); the initial state is printed as "tick -1" with --hash-init. */
    auto hash_due = [&](i64 t) { bool last = t == ticks - 1; return hash_every > 0 ? (last || (t + 1) % hash_every == 0) : hash_every == 0 ? last : false; };
    auto census_due = [&](i64 t) { return t == ticks - 1 || (census_every > 0 && (t + 1) % census_every == 0); };
    auto report = [&](i64 t) {
        bool hh = hash_due(t); u64 hv = 0;
        if (hh) { hv = state_hash(h); printf("tick %lld hash %016llx\n", (long long)t, (unsigned long long)hv); }
        if (census_due(t)) {
            std::string js = census_json(h, t, prev, hh, hv);
            printf("census %s\n", js.c_str());
            memcpy(prev, h.counters, sizeof prev);
            if (t == ticks - 1 && !out_path.empty()) { FILE *f = fopen(out_path.c_str(), "w"); if (f) { fprintf(f, "%s\n", js.c_str()); fclose(f); } else fprintf(stderr, "cannot write %s\n", out_path.c_str()); }
        }
        fflush(stdout);
    };
    if (hash_init && hash_every >= 0) printf("tick -1 hash %016llx\n", (unsigned long long)state_hash(h));
    fflush(stdout);

    auto t_loop = std::chrono::steady_clock::now();
    for (i64 t = 0; t < ticks; t++) {
        CUDA_CHECK(cudaMemcpyAsync(w.fwd_old, w.fwd, h.nbytes, cudaMemcpyDeviceToDevice));   /* snapshot for reads */
        k_execute<<<sgrid, 256>>>(w, P, t);                                                 /* 1 */
        k_alloc_propose<<<sgrid, 256>>>(w, P, t);                                           /* 2 */
        k_alloc_grant<<<sgrid, 256>>>(w, P, t);
        CUDA_CHECK(cudaMemsetAsync(w.claim, 0xFF, S * sizeof(i32)));
        k_births<<<sgrid, 256>>>(w, P, t);                                                  /* 3 */
        k_deaths<<<sgrid, 256>>>(w, P);                                                     /* 4 */
        if (D > 0) {                                        /* a grid of 0 blocks is an invalid launch */
            k_damage<<<dgrid, 256>>>(w, P, t);                                              /* 5 */
            if (dbl) k_repair<<<dgrid, 256>>>(w, P, t);                                     /* 6 */
        }
        if (F > 0) {
            k_inflow_claim<<<fgrid, 256>>>(w, P, t);                                        /* 7 */
            k_inflow_fill<<<(unsigned)F, 256>>>(w, P, t);
            k_inflow_births<<<sgrid, 256>>>(w, P, t);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(h.counters, w.counters, sizeof h.counters, cudaMemcpyDeviceToHost));   /* also syncs */
        if (births_f) {
            u32 n; CUDA_CHECK(cudaMemcpy(&n, w.log_n, sizeof n, cudaMemcpyDeviceToHost));
            if (n > w.log_cap) { fprintf(stderr, "births log overflow at tick %lld (%u > %u)\n", (long long)t, n, w.log_cap); n = w.log_cap; }
            if (n) {
                CUDA_CHECK(cudaMemcpy(log_host.data(), w.log, (size_t)n * sizeof(BirthRec), cudaMemcpyDeviceToHost));
                /* reference order: DIVIDE births by parent slot, then spontaneous births by slot */
                std::sort(log_host.begin(), log_host.begin() + n, [](const BirthRec &a, const BirthRec &b) {
                    bool sa = a.parent < 0, sb = b.parent < 0;
                    if (sa != sb) return sb;
                    return sa ? a.child < b.child : a.parent < b.parent; });
                for (u32 i = 0; i < n; i++) {
                    const BirthRec &r = log_host[i];
                    char ph[20]; if (r.parent >= 0) snprintf(ph, sizeof ph, "%016llx", (unsigned long long)r.phash); else strcpy(ph, "-");
                    fprintf(births_f, "%lld\t%lld\t%lld\t%s\t%016llx\t%d\n", (long long)t, (long long)r.parent, (long long)r.child, ph,
                            (unsigned long long)r.chash, r.parent >= 0 && r.phash == r.chash ? 1 : 0);
                }
                CUDA_CHECK(cudaMemset(w.log_n, 0, sizeof(u32)));
            }
        }
        report(t);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_loop).count();
    double secs_total = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    if (!dump_path.empty()) dump_state(h, dump_path.c_str());
    if (births_f) fclose(births_f);
    i64 live_n = 0; CUDA_CHECK(cudaMemcpy(h.live.data(), w.live, S, cudaMemcpyDeviceToHost)); for (i64 s = 0; s < S; s++) live_n += h.live[s];
    fprintf(stderr, "summary ticks=%lld seconds=%.3f ticks_per_second=%.2f seconds_total=%.3f device_mem_alloc_bytes=%zu "
                    "device_mem_used_bytes=%zu live=%lld births=%llu faithful=%llu spontaneous=%llu deaths=%llu\n",
            (long long)ticks, secs, ticks > 0 && secs > 0 ? ticks / secs : 0.0, secs_total, allocated, (size_t)(mem_free0 - mem_free1),
            (long long)live_n, (unsigned long long)h.counters[0], (unsigned long long)h.counters[1], (unsigned long long)h.counters[2],
            (unsigned long long)h.counters[3]);
    return 0;
}
