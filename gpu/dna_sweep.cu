/* DNA-1 copier sweep engine: thousands of independent small worlds per kernel launch (docs/09 section 12).
 *
 * Every world is the world of docs/09 with W slots of G cells (default 8 x 256 = 2,048 cells), mu = 0, no damage
 * (D = 0), no inflow (F = 0), a blank tape (or the random soup of section 6 with --soup) and one seeded organism in
 * slot 0: the gene's codons followed by STOP codons up to the last usable codon, exactly as `--genome ... --n-seed 1`
 * seeds a slot in sim/dna_ref.py and gpu/dna_world.cu. One GPU thread runs one world sequentially, processor by
 * processor in slot order, with the snapshot rule, the ALLOC draw, the births, the deaths and the counters of the
 * reference; so a single world of this program must give exactly the state the reference gives on
 *
 *   python sim/dna_ref.py --N W*G --G G --ticks n --seed s --I I --T T --mu 0 --D 0 --F 0 --blank --n-seed 1 --genome "..."
 *
 * which `--genes FILE --hash-ticks OUT` checks tick by tick (the per-tick state hash of section 8).
 *
 *   nvcc -O3 -std=c++17 -arch=sm_89 -o dna_sweep dna_sweep.cu
 *   ./dna_sweep --L 4 --all --out L4.json --log L4_log.tsv                  (all 64^3 codon strings after START)
 *   ./dna_sweep --L 7 --random 1000000000 --rseed 7 --out L7.json --log L7_log.tsv
 *   ./dna_sweep --genes genes.txt --records recs.tsv --hash-ticks hashes.tsv  (verification against the reference)
 *
 * Modes: --L n --all            every gene START + (n-1) codons; gene index g = sum c_k 64^(n-1-k), k = 1..n-1 (the last
 *                               codon is the least significant digit; consecutive threads differ in the last codon)
 *        --L n --random m       m indices g = splitmix64(rseed + i * K1) mod 64^(n-1), i = 0..m-1
 *        --genes FILE           one codon string per line (whitespace, comma or contiguous letters), any first codon
 * World: --W 8 --G 256 --ticks 64 --T 64 --I 16 --seed 1 --uniform-code --soup
 * Run:   --chunk 4194304 (worlds per launch) --chunk-start i --chunk-end j --tpb 64 --gpu 0 --no-prefilter
 *        --slow-seek (SEEK cell by cell instead of 16 cells per word; same result, for checking)
 * Out:   --out summary.json  --log FILE (TSV of the genes selected by --log-mask: bit0 faithful child, bit1 a child
 *        divided in turn, bit2 any child, bit3 divided, bit4 copied a cell, bit5 claimed a slot; default 3)
 *        --records FILE (TSV of every world, small runs only) --hash-ticks FILE (state hash after every tick of every
 *        world, --genes mode only) --chunk-log FILE (one line per launch: range, counts, seconds)
 *
 * Per world the record holds: claims (ALLOC grants to the seed organism), copied (cells it wrote into child slots),
 * divides (its DIVIDEs that marked a child), births0 / faithful0 (its children / those with the same gene hash),
 * births / faithful / deaths (whole world), gchildren (births whose parent is itself a child: a child divided in
 * turn), first_birth (tick + 1, 0 = none), live_end, maxdepth, glen (codons of the gene before its first STOP).
 * "Seed organism" means the processor of depth 0; nothing else has depth 0 because inflow is off.
 *
 * Prefilter (exact, --no-prefilter switches it off for checking): a gene whose codons before the first STOP contain
 * neither ALLOC nor INC, FLIP or STA can never claim a slot nor change its own gene, so its world is static; the
 * record is written without running it (deaths = 1 if the organism dies within the run, i.e. ticks >= T or the gene
 * is not expressible, live_end = 1 - deaths, everything else 0). Verified equal to the full run on lengths 2 to 5.
 *
 * Device code (codon table, PRNG, gene functions) is copied verbatim from gpu/dna_world.cu; the tape here is one
 * packed 2-bit strand per world in shared memory (rev = comp(fwd) always holds because damage is off, so the state
 * hash derives rev from fwd), the tick-start snapshot is kept per slot in local memory and only for slots that can
 * be written in the tick (slots live or claimed at the tick start), which reads the same values as a full copy.
 */
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>

typedef uint64_t u64; typedef int64_t i64; typedef uint32_t u32; typedef int32_t i32; typedef uint8_t u8; typedef uint16_t u16; typedef int16_t i16; typedef int8_t i8;

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error at %s:%d: %s (%s)\n", __FILE__, __LINE__, cudaGetErrorString(e_), #x); exit(2); } } while (0)

/* ---------------------------------------------------------------- operations and the codon table (section 3), as in dna_world.cu */
enum Op : u8 { NOP = 0, START, STOP, COPY, R1, L1, R3, L3, JNZB, SETP, SETQ, ALLOC, DIVIDE, SKNZ, SKZ, LDA, STA,
               INC, FLIP, SEEK, COPYR, REWIND, TERM, N_OPS };   /* TERM only under --dna2 (docs/15) */
static const char *OPNAME[N_OPS] = { "NOP", "START", "STOP", "COPY", "R1", "L1", "R3", "L3", "JNZB", "SETP", "SETQ", "ALLOC", "DIVIDE",
                                     "SKNZ", "SKZ", "LDA", "STA", "INC", "FLIP", "SEEK", "COPYR", "REWIND", "TERM" };

static int letter(char c) { switch (c) { case 'A': case 'a': return 0; case 'C': case 'c': return 1; case 'G': case 'g': return 2;
                                         case 'T': case 't': case 'U': case 'u': return 3; } return -1; }

static int codon_value(const char *s)
{
    if (strlen(s) != 3) return -1;
    int a = letter(s[0]), b = letter(s[1]), c = letter(s[2]);
    if (a < 0 || b < 0 || c < 0) return -1;
    return 16 * a + 4 * b + c;
}

static void codon_text(int c, char *out) { const char *L = "ACGT"; out[0] = L[c >> 4]; out[1] = L[(c >> 2) & 3]; out[2] = L[c & 3]; out[3] = 0; }

static void build_table_dna1(u8 *table, bool uniform);
/* docs/15 fix 1: under --dna2 TERM takes TTA and TTG from Leu (COPY keeps CTT CTC CTA CTG); under the uniform code
 * it takes codon value 21, the first NOP */
static void build_table(u8 *table, bool uniform, bool dna2)
{
    build_table_dna1(table, uniform);
    if (!dna2) return;
    if (uniform) { table[21] = TERM; return; }
    const char *tc[2] = { "TTA", "TTG" };
    for (int i = 0; i < 2; i++) { int v = codon_value(tc[i]); if (table[v] != COPY) { fprintf(stderr, "dna2 table error\n"); exit(1); } table[v] = TERM; }
}
static void build_table_dna1(u8 *table, bool uniform)
{
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

/* ---------------------------------------------------------------- random numbers (section 7), as in dna_world.cu */
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

__host__ __device__ inline u64 rnd(u64 seed, i64 t, u64 k, u64 j)
{
    u64 mix = (u64)(t + 1) * K1 + k * K2 + j * K3;
    return splitmix64(seed ^ mix);
}

/* ---------------------------------------------------------------- limits and records */
#define MAXW 16          /* slots per world */
#define MAXNB 1024       /* packed bytes per world tape: W * G <= 4096 cells */
#define MAXL 32          /* codons of a listed gene */
#define NH 96            /* histogram counters per gene-length bucket */
#define NBUCKET 8        /* buckets: glen 0..7 (7 = 7 or more) */

enum Hist { H_STRINGS = 0, H_EXPR, H_CLAIMED, H_COPIED, H_COPIED6, H_DIVIDED, H_BIRTHS0, H_FAITHFUL0, H_GCHILD, H_BIRTHS, H_FAITHFUL,
            H_PREFILTERED, H_LIVE_END, H_DEPTH2, H_SUM_CLAIMS, H_SUM_COPIED, H_SUM_BIRTHS0, H_SUM_FAITHFUL0, H_SUM_BIRTHS, H_SUM_GCHILD,
            H_FEC = 20 /* 17 bins: faithful0 = 0..15, >= 16 */, H_B0 = 40 /* 17 bins: births0 */, H_DEPTH = 57 /* 7 bins: maxdepth 0..5, >= 6 */,
            H_INTACT0 = 64, H_SUM_INTACT0 = 65, H_IFEC = 66 /* 17 bins: intact0 */ };
static const char *HNAME[20] = { "strings", "expressible", "claimed", "copied_any", "copied_ge6", "divided", "children_any", "faithful_any",
                                 "grandchildren_any", "births_any_world", "faithful_any_world", "prefiltered", "live_at_end", "depth_ge2",
                                 "sum_claims", "sum_copied", "sum_children", "sum_faithful", "sum_births_world", "sum_grandchildren" };

struct Rec {
    u64 gene;
    u16 claims, copied, divides, births0, faithful0, births, faithful, deaths, gchildren, first_birth;
    u16 claims_world, copied_world;      /* all processors of the world, as the reference's done line counts them */
    u8 live_end, maxdepth, L, glen, flags, pad0;
    u16 intact0;                         /* faithful children of the seed organism born while its gene still hashed as seeded */
};  /* 40 bytes; flags bit0 = prefiltered; u16 fields saturate at 65535 */

struct SP {
    u64 seed, n_worlds, gene_start, range, rseed;
    i32 W, G, N, NB, maxk, I, T, ticks, mode, L, prefilter, hash_ticks, soup, slow_seek, dna2;
    u32 log_mask, log_cap;
    u8 start_codon, stop_codon;
};

struct Out {
    u64 *hist; Rec *log; u32 *log_n; Rec *recs; u64 *hashes;
    const u8 *table; const u8 *list_codons; const u8 *list_len; const u8 *soup;
};

/* ---------------------------------------------------------------- the world of one thread */
__device__ __forceinline__ u32 get2s(const u8 *a, i32 i) { return (a[i >> 2] >> ((i & 3) * 2)) & 3u; }
__device__ __forceinline__ void set2s(u8 *a, i32 i, u32 v) { u32 sh = (u32)(i & 3) * 2; a[i >> 2] = (u8)((a[i >> 2] & ~(3u << sh)) | (v << sh)); }
__device__ __forceinline__ u32 codon3(const u8 *f, i32 i) { return 16 * get2s(f, i) + 4 * get2s(f, i + 1) + get2s(f, i + 2); }

__device__ inline i32 glen_of(const u8 *f, i32 base, const u8 *tab, i32 maxk)
{
    for (i32 k = 0; k <= maxk; k++) if (tab[codon3(f, base + 3 * k)] == STOP) return k;
    return maxk + 1;
}

__device__ inline u64 ghash_of(const u8 *f, i32 base, const u8 *tab, i32 maxk)
{
    u64 h = 14695981039346656037ULL;
    for (i32 k = 0; k <= maxk; k++) {
        u32 c = codon3(f, base + 3 * k);
        if (tab[c] == STOP) break;
        h = (h ^ (u64)c) * 1099511628211ULL;
    }
    return h;
}

struct WS {   /* processors dense by slot (section 4 fields), claims, snapshot flags, cached gene lengths */
    i32 pc[MAXW], H[MAXW], P[MAXW], Q[MAXW], child[MAXW], A[MAXW], age[MAXW], depth[MAXW], copied[MAXW];
    i8 claimed_by[MAXW]; u8 live[MAXW], snap[MAXW], svalid[MAXW]; i16 slen[MAXW];
};

__device__ __forceinline__ u64 fnv_u64(u64 h, u64 v) { for (int i = 0; i < 8; i++) h = (h ^ ((v >> (8 * i)) & 0xFF)) * 1099511628211ULL; return h; }

/* the state hash of section 8 for one world (rev = comp(fwd) cell by cell, padding cells 0) */
__device__ u64 world_hash(const u8 *fwd, const SP &P, const WS &w, u64 births, u64 faithful, u64 deaths)
{
    u64 h = 14695981039346656037ULL;
    for (i32 b = 0; b < P.NB; b++) h = (h ^ fwd[b]) * 1099511628211ULL;
    u32 lastmask = (P.N & 3) ? ((1u << (2 * (P.N & 3))) - 1) : 0xFFu;
    for (i32 b = 0; b < P.NB; b++) { u32 r = (~(u32)fwd[b]) & (b == P.NB - 1 ? lastmask : 0xFFu); h = (h ^ r) * 1099511628211ULL; }
    for (i32 s = 0; s < P.W; s++) if (w.live[s]) {
        h = fnv_u64(h, (u64)s); h = fnv_u64(h, (u64)w.pc[s]); h = fnv_u64(h, (u64)w.H[s]); h = fnv_u64(h, (u64)w.P[s]); h = fnv_u64(h, (u64)w.Q[s]);
        h = fnv_u64(h, (u64)(i64)w.child[s]); h = fnv_u64(h, (u64)w.A[s]); h = fnv_u64(h, (u64)w.age[s]); h = fnv_u64(h, (u64)w.depth[s]); h = fnv_u64(h, (u64)w.copied[s]);
    }
    h = fnv_u64(h, births); h = fnv_u64(h, faithful); h = fnv_u64(h, 0); h = fnv_u64(h, deaths);
    return h;
}

__device__ __forceinline__ void born(WS &w, i32 s, i32 depth, i32 G)
{
    w.live[s] = 1; w.pc[s] = 1; w.H[s] = w.P[s] = s * G; w.Q[s] = 0; w.child[s] = -1; w.A[s] = 0; w.age[s] = 0; w.depth[s] = depth; w.copied[s] = 0;
}

/* bit 2k of the result is set iff 2-bit cell k of v equals p (cells k = 0..31 of a 64-bit word) */
__device__ __forceinline__ u64 eq_lanes(u64 v, u32 p)
{
    u64 x = ~(v ^ ((u64)p * 0x5555555555555555ULL));
    return x & (x >> 1) & 0x5555555555555555ULL;
}

/* SEEK, 16 cells per step: the first position h in H+1 .. H+3G (ring) whose cells h, h+1, h+2 read p0, p1, p2 under
 * the snapshot rule, or -1. Needs G % 16 == 0, so that every 16-cell word lies in one slot and has one source array
 * (fwd for the own and the claimed child slot and for slots nobody can write this tick, old for the snapped
 * others), exactly as the per-cell RD() reads them; the two cells after a word come from the next word of the ring
 * with that word's own source. Same result as the per-cell loop in run_world, which stays as the general path. */
__device__ __forceinline__ u32 src_word(const u8 *fwd, const u8 *old, const u8 *snap, i32 base, i32 ch_lo, i32 G, i32 widx)
{
    i32 s = (widx << 4) / G, sb = s * G;
    const u8 *a = (sb == base || sb == ch_lo) ? fwd : (snap[s] ? old : fwd);
    return *(const u32 *)(a + (size_t)widx * 4);
}

__device__ i32 seek_fast(const u8 *fwd, const u8 *old, const u8 *snap, i32 base, i32 ch_lo, i32 G, i32 N, i32 H, u32 p0, u32 p1, u32 p2)
{
    const i32 NW = N >> 4;
    i32 start = H + 1; if (start >= N) start -= N;
    i32 wi = start >> 4, off = start & 15, remaining = 3 * G;
    u32 wcur = src_word(fwd, old, snap, base, ch_lo, G, wi);
    while (remaining > 0) {
        i32 wn = wi + 1; if (wn == NW) wn = 0;
        u32 wnext = src_word(fwd, old, snap, base, ch_lo, G, wn);
        u64 v = (u64)wcur | ((u64)wnext << 32);
        u64 m = eq_lanes(v, p0) & (eq_lanes(v, p1) >> 2) & (eq_lanes(v, p2) >> 4);
        m &= 0xFFFFFFFFULL;                                   /* positions k = 0..15 of this word */
        m &= ~((1ULL << (2 * off)) - 1);                      /* not before the first candidate */
        i32 avail = 16 - off;
        if (remaining < avail) { m &= (1ULL << (2 * (off + remaining))) - 1; avail = remaining; }
        if (m) { i32 k = (__ffsll((long long)m) - 1) >> 1; return (wi << 4) + k; }
        remaining -= avail; off = 0; wi = wn; wcur = wnext;
    }
    return -1;
}

/* runs one world; cod[0..L-1] are the codons written to slot 0 (then STOP padding). r must be zeroed by the caller. */
__device__ void run_world(const SP &P, const u8 *tab, const u8 *xtab, const u8 *cod, i32 L, u8 *fwd, u8 *old, Rec &r, u64 *hash_out, const u8 *soup)
{
    const i32 W = P.W, G = P.G, N = P.N, NB = P.NB, maxk = P.maxk, I = P.I, T = P.T;
    const bool fast_seek = (G & 15) == 0 && !P.slow_seek;
    /* initial state: blank or soup, then slot 0 = the gene padded with the lowest STOP codon up to the last usable codon */
    if (soup) for (i32 b = 0; b < NB; b++) fwd[b] = soup[b]; else for (i32 b = 0; b < NB; b++) fwd[b] = 0;
    for (i32 k = 0; k <= maxk; k++) {
        u32 c = k < L ? cod[k] : P.stop_codon;
        set2s(fwd, 3 * k, c >> 4); set2s(fwd, 3 * k + 1, (c >> 2) & 3); set2s(fwd, 3 * k + 2, c & 3);
    }
    WS w;
    for (i32 s = 0; s < W; s++) { w.live[s] = 0; w.claimed_by[s] = -1; w.svalid[s] = 0; w.child[s] = -1; w.depth[s] = 0; }
    born(w, 0, 0, G);
    i32 glen0 = glen_of(fwd, 0, tab, maxk);
    const u64 hash0 = ghash_of(fwd, 0, tab, maxk);     /* the seeded gene's hash: a parent still hashing so is "intact" */
    r.glen = (u8)(glen0 > 255 ? 255 : glen0);
    r.first_birth = 0;
    bool expr0 = tab[codon3(fwd, 0)] == START && glen0 >= 2;
    if (P.prefilter && !hash_out) {
        bool active = false;
        for (i32 k = 1; k < glen0; k++) { u32 op = tab[codon3(fwd, 3 * k)]; if (op == ALLOC || op == INC || op == FLIP || op == STA) active = true; }
        if (!active) {
            r.flags |= 1;
            r.deaths = (!expr0 || P.ticks >= T) ? 1 : 0;
            r.live_end = 1 - (u8)r.deaths;
            return;
        }
    }
    u32 births = 0, faithful = 0, deaths = 0;
    i32 nlive = 1;
    i32 mark[MAXW];
#define SLOT(i)     ((i) / G)
#define IN_OWN(i)   ((i) >= base && (i) < own_hi)
#define IN_CHILD(i) ((i) >= ch_lo && (i) < ch_hi)
#define RD(i)       ((IN_OWN(i) || IN_CHILD(i)) ? get2s(fwd, (i)) : (w.snap[SLOT(i)] ? get2s(old, (i)) : get2s(fwd, (i))))
    for (i32 t = 0; t < P.ticks; t++) {
        if (nlive == 0 && !hash_out) break;      /* nothing can change any more: no damage, no inflow */
        /* snapshot of the slots that can be written in this tick (live or claimed at the tick start) */
        u32 req_mask = 0;
        for (i32 s = 0; s < W; s++) {
            w.snap[s] = (w.live[s] || w.claimed_by[s] >= 0) ? 1 : 0;
            mark[s] = -1;
            if (w.snap[s]) { i32 b0 = (s * G) >> 2, b1 = (s * G + G - 1) >> 2; for (i32 b = b0; b <= b1; b++) old[b] = fwd[b]; }
        }
        /* 1. execute, in slot order */
        for (i32 s = 0; s < W; s++) {
            if (!w.live[s]) continue;
            const i32 base = s * G, own_hi = base + G;
            i32 pc = w.pc[s], H = w.H[s], Pp = w.P[s], Q = w.Q[s], child = w.child[s], A = w.A[s], copied = w.copied[s];
            i32 ch_lo = child >= 0 ? child * G : -1, ch_hi = child >= 0 ? ch_lo + G : -1;
            if (!w.svalid[s]) { w.slen[s] = (i16)glen_of(fwd, base, tab, maxk); w.svalid[s] = 1; }
            i32 len = w.slen[s];
            bool requested = false;
            const bool seed_org = w.depth[s] == 0;
            for (i32 n = 0; n < I; n++) {
                u32 op = xtab[codon3(fwd, base + 3 * pc)];      /* the gene is fetched live; START and STOP execute as NOP */
                pc += 1;
                switch (op) {
                case NOP: break;
                case COPY: case COPYR: {
                    if (P.dna2 && child < 0) break;              /* docs/15 fix 3: no claimed child, nothing happens */
                    u32 v = (IN_OWN(Pp) || IN_CHILD(Pp)) ? get2s(fwd, Pp) : (w.snap[SLOT(Pp)] ? get2s(old, Pp) : get2s(fwd, Pp));
                    if (op == COPYR) v ^= 3u;
                    if (IN_CHILD(Q)) {                           /* mu = 0: no mutation */
                        set2s(fwd, Q, v); copied += 1; w.svalid[child] = 0;
                        if (seed_org && r.copied < 65535) r.copied += 1;
                        if (r.copied_world < 65535) r.copied_world += 1;
                    }
                    Pp += op == COPY ? 1 : -1; if (Pp >= N) Pp -= N; if (Pp < 0) Pp += N;
                    Q += 1; if (Q >= N) Q -= N;
                } break;
                case R1: H += 1; if (H >= N) H -= N; break;
                case L1: H -= 1; if (H < 0) H += N; break;
                case R3: H += 3; if (H >= N) H -= N; break;
                case L3: H -= 3; if (H < 0) H += N; break;
                case JNZB: if (RD(H) != 0) { pc -= 2; if (pc < 1) pc = 1; } break;
                case SETP: Pp = H; break;
                case SETQ: Q = H; break;
                case TERM: {   /* docs/15 fix 1: skip the next codon unless the template has just passed a STOP codon */
                    i32 o = Pp % G; bool term = false;
                    if (o >= 3 && o % 3 == 0 && o / 3 <= maxk + 1) {
                        u32 c3 = 16u * RD(Pp - 3) + 4u * RD(Pp - 2) + RD(Pp - 1);
                        term = tab[c3] == STOP;
                    }
                    if (!term) pc += 1;
                } break;
                case ALLOC: if (child < 0 && !requested) { requested = true; req_mask |= 1u << s; } break;
                case DIVIDE:
                    if (child >= 0 && copied >= 6) { mark[s] = child; child = -1; copied = 0; Q = 0; ch_lo = ch_hi = -1; if (seed_org) r.divides += 1;
                                                     if (P.dna2) Pp = base; }   /* docs/15 fix 2: re-arm the template */
                    break;
                case SKNZ: if (RD(H) != 0) pc += 1; break;
                case SKZ: if (RD(H) == 0) pc += 1; break;
                case LDA: A = RD(H); break;
                case STA: case INC: case FLIP:
                    if (IN_OWN(H)) {
                        u32 v = op == STA ? (u32)A : op == INC ? (get2s(fwd, H) + 1) & 3u : get2s(fwd, H) ^ 1u;
                        set2s(fwd, H, v);
                        len = glen_of(fwd, base, tab, maxk); w.slen[s] = (i16)len; w.svalid[s] = 1;   /* the own gene may have changed */
                    } else if (IN_CHILD(H)) {
                        u32 v = op == STA ? (u32)A : op == INC ? (get2s(fwd, H) + 1) & 3u : get2s(fwd, H) ^ 1u;
                        set2s(fwd, H, v); w.svalid[child] = 0;
                    }
                    break;
                case SEEK: {
                    if (pc < len) {                              /* the template codon lies inside the gene */
                        i32 tb = base + 3 * pc;
                        u32 p0 = get2s(fwd, tb) ^ 3u, p1 = get2s(fwd, tb + 1) ^ 3u, p2 = get2s(fwd, tb + 2) ^ 3u;
                        pc += 1;                                 /* the template is consumed */
                        if (fast_seek) { i32 f = seek_fast(fwd, old, w.snap, base, ch_lo, G, N, H, p0, p1, p2); if (f >= 0) H = f; break; }
                        i32 h0 = H + 1; if (h0 >= N) h0 -= N;
                        i32 h1 = h0 + 1; if (h1 >= N) h1 -= N;
                        u32 c0 = RD(h0), c1 = RD(h1);
                        i32 h2 = h1;
                        for (i32 d = 1; d <= 3 * G; d++) {       /* candidates H+1 .. H+3G; the window slides one cell per step */
                            h2 += 1; if (h2 >= N) h2 -= N;
                            u32 c2 = RD(h2);
                            if (c0 == p0 && c1 == p1 && c2 == p2) { H = h0; break; }
                            c0 = c1; c1 = c2; h0 += 1; if (h0 >= N) h0 -= N;
                        }
                    }
                } break;
                case REWIND: Pp = base; if (child >= 0) Q = child * G; break;
                default: break;
                }
                if (pc >= len) pc = 1;                           /* the gene loops, skipping START */
            }
            w.pc[s] = pc; w.H[s] = H; w.P[s] = Pp; w.Q[s] = Q; w.child[s] = child; w.A[s] = A; w.copied[s] = copied;
        }
        /* 2. resolve ALLOC in increasing slot order (the lowest requester takes a contested candidate) */
        for (i32 s = 0; s < W; s++) {
            if (!(req_mask & (1u << s))) continue;
            i32 cand = (i32)(((u64)s + 1 + rnd(P.seed, t, (u64)s, 0) % (u64)(W - 1)) % (u64)W);
            if (!w.live[cand] && w.claimed_by[cand] < 0) {
                w.claimed_by[cand] = (i8)s; w.child[s] = cand; w.Q[s] = cand * G; w.copied[s] = 0;
                if (w.depth[s] == 0) r.claims += 1;
                if (r.claims_world < 65535) r.claims_world += 1;
            }
        }
        /* 3. births, in increasing parent order; the claim is released either way */
        for (i32 p = 0; p < W; p++) {
            if (mark[p] < 0) continue;
            i32 ch = mark[p];
            w.claimed_by[ch] = -1;
            if (!w.svalid[ch]) { w.slen[ch] = (i16)glen_of(fwd, ch * G, tab, maxk); w.svalid[ch] = 1; }
            if (tab[codon3(fwd, ch * G)] == START && w.slen[ch] >= 2) {
                born(w, ch, w.depth[p] + 1, G);
                nlive += 1;
                u64 ph = ghash_of(fwd, p * G, tab, maxk), chh = ghash_of(fwd, ch * G, tab, maxk);
                bool f = ph == chh;
                births += 1; faithful += f;
                if (w.depth[p] == 0) { r.births0 += 1; r.faithful0 += f; if (f && ph == hash0) r.intact0 += 1; } else r.gchildren += 1;
                if (w.depth[ch] > r.maxdepth) r.maxdepth = (u8)(w.depth[ch] > 255 ? 255 : w.depth[ch]);
                if (r.first_birth == 0) r.first_birth = (u16)(t + 1);
            }
        }
        /* 4. ageing and deaths (every processor alive now, this tick's newborns included) */
        for (i32 s = 0; s < W; s++) {
            if (!w.live[s]) continue;
            w.age[s] += 1;
            if (!w.svalid[s]) { w.slen[s] = (i16)glen_of(fwd, s * G, tab, maxk); w.svalid[s] = 1; }
            bool expr = tab[codon3(fwd, s * G)] == START && w.slen[s] >= 2;
            if (w.age[s] >= T || !expr) {
                w.live[s] = 0; nlive -= 1; deaths += 1;
                if (w.child[s] >= 0) { w.claimed_by[w.child[s]] = -1; w.child[s] = -1; }
            }
        }
        /* 5-7: D = 0, F = 0: nothing. */
        if (hash_out) hash_out[t] = world_hash(fwd, P, w, births, faithful, deaths);
    }
#undef SLOT
#undef IN_OWN
#undef IN_CHILD
#undef RD
    r.births = (u16)births; r.faithful = (u16)faithful; r.deaths = (u16)deaths; r.live_end = (u8)nlive;
}

/* ---------------------------------------------------------------- the kernel: one thread per world, block histograms */
__global__ void __launch_bounds__(256) k_sweep(SP P, Out O, i32 stride)
{
    extern __shared__ u8 smem[];
    __shared__ u32 shist[NBUCKET * NH];
    __shared__ u8 stab[64], sxtab[64];
    for (int i = threadIdx.x; i < NBUCKET * NH; i += blockDim.x) shist[i] = 0;
    for (int i = threadIdx.x; i < 64; i += blockDim.x) { u8 op = O.table[i]; stab[i] = op; sxtab[i] = (op == START || op == STOP) ? (u8)NOP : op; }
    __syncthreads();
    u8 *fwd = smem + (size_t)threadIdx.x * stride;
    u32 old32[MAXNB / 4];                    /* 4-byte aligned for the word reads of seek_fast */
    u8 *old = (u8 *)old32;
    u64 tid = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < P.n_worlds) {
        u64 widx = P.gene_start + tid;      /* world index within the run */
        u8 cod[MAXL]; i32 L;
        u64 gene;
        if (P.mode == 2) {
            L = O.list_len[widx];
            for (i32 k = 0; k < L; k++) cod[k] = O.list_codons[widx * MAXL + k];
            gene = widx;
        } else {
            gene = P.mode == 0 ? widx : splitmix64(P.rseed + widx * K1) % P.range;
            L = P.L;
            cod[0] = P.start_codon;
            for (i32 k = 1; k < L; k++) cod[k] = (u8)((gene >> (6 * (L - 1 - k))) & 63);
        }
        Rec r; memset(&r, 0, sizeof r);
        r.gene = gene; r.L = (u8)L;
        run_world(P, stab, sxtab, cod, L, fwd, old, r, P.hash_ticks ? O.hashes + widx * (u64)P.ticks : NULL, P.soup ? O.soup : NULL);
        if (O.recs) O.recs[tid] = r;
        /* histogram by effective gene length */
        u32 *hb = shist + (r.glen >= NBUCKET ? NBUCKET - 1 : r.glen) * NH;
        atomicAdd(&hb[H_STRINGS], 1u);
        if (r.glen >= 2 && stab[cod[0]] == START) atomicAdd(&hb[H_EXPR], 1u);
        if (r.claims) atomicAdd(&hb[H_CLAIMED], 1u);
        if (r.copied) atomicAdd(&hb[H_COPIED], 1u);
        if (r.copied >= 6) atomicAdd(&hb[H_COPIED6], 1u);
        if (r.divides) atomicAdd(&hb[H_DIVIDED], 1u);
        if (r.births0) atomicAdd(&hb[H_BIRTHS0], 1u);
        if (r.faithful0) atomicAdd(&hb[H_FAITHFUL0], 1u);
        if (r.gchildren) atomicAdd(&hb[H_GCHILD], 1u);
        if (r.births) atomicAdd(&hb[H_BIRTHS], 1u);
        if (r.faithful) atomicAdd(&hb[H_FAITHFUL], 1u);
        if (r.flags & 1) atomicAdd(&hb[H_PREFILTERED], 1u);
        if (r.live_end) atomicAdd(&hb[H_LIVE_END], 1u);
        if (r.maxdepth >= 2) atomicAdd(&hb[H_DEPTH2], 1u);
        atomicAdd(&hb[H_SUM_CLAIMS], (u32)r.claims); atomicAdd(&hb[H_SUM_COPIED], (u32)r.copied);
        atomicAdd(&hb[H_SUM_BIRTHS0], (u32)r.births0); atomicAdd(&hb[H_SUM_FAITHFUL0], (u32)r.faithful0);
        atomicAdd(&hb[H_SUM_BIRTHS], (u32)r.births); atomicAdd(&hb[H_SUM_GCHILD], (u32)r.gchildren);
        atomicAdd(&hb[H_FEC + (r.faithful0 > 16 ? 16 : r.faithful0)], 1u);
        atomicAdd(&hb[H_B0 + (r.births0 > 16 ? 16 : r.births0)], 1u);
        atomicAdd(&hb[H_DEPTH + (r.maxdepth > 6 ? 6 : r.maxdepth)], 1u);
        if (r.intact0) atomicAdd(&hb[H_INTACT0], 1u);
        atomicAdd(&hb[H_SUM_INTACT0], (u32)r.intact0);
        atomicAdd(&hb[H_IFEC + (r.intact0 > 16 ? 16 : r.intact0)], 1u);
        u32 sel = (r.faithful0 ? 1u : 0u) | (r.gchildren ? 2u : 0u) | (r.births0 ? 4u : 0u) | (r.divides ? 8u : 0u) | (r.copied ? 16u : 0u) | (r.claims ? 32u : 0u)
                | (r.intact0 ? 64u : 0u);
        if (O.log && (sel & P.log_mask)) { u32 n = atomicAdd(O.log_n, 1u); if (n < P.log_cap) O.log[n] = r; }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < NBUCKET * NH; i += blockDim.x) if (shist[i]) atomicAdd((unsigned long long *)&O.hist[i], (unsigned long long)shist[i]);
}

/* ---------------------------------------------------------------- host */
static void usage()
{
    fprintf(stderr, "usage: dna_sweep (--L n --all | --L n --random m [--rseed s] | --genes FILE) [--W 8 --G 256 --ticks 64 --T 64 --I 16 --seed 1\n"
                    "  --uniform-code --soup --no-prefilter --chunk n --chunk-start i --chunk-end j --tpb 64 --gpu 0\n"
                    "  --out summary.json --log FILE --log-mask m --records FILE --hash-ticks FILE --chunk-log FILE]\n");
    exit(1);
}

static std::vector<int> parse_genome(const std::string &genome, bool &ok)
{
    std::vector<std::string> toks; std::vector<int> codons; ok = true;
    char *buf = strdup(genome.c_str());
    for (char *tok = strtok(buf, " \t\r\n\v\f,"); tok; tok = strtok(NULL, " \t\r\n\v\f,")) toks.push_back(tok);
    free(buf);
    if (toks.size() == 1 && toks[0].size() > 3) {
        std::string s = toks[0];
        if (s.size() % 3) { ok = false; return codons; }
        toks.clear();
        for (size_t i = 0; i < s.size(); i += 3) toks.push_back(s.substr(i, 3));
    }
    for (const std::string &tok : toks) { int v = codon_value(tok.c_str()); if (v < 0) { ok = false; return codons; } codons.push_back(v); }
    return codons;
}

static std::string gene_text(const Rec &r, int mode, const std::vector<u8> &list_codons, u8 start_codon)
{
    std::string s; char buf[4];
    if (mode == 2) {
        for (int k = 0; k < r.L; k++) { codon_text(list_codons[r.gene * MAXL + k], buf); if (k) s += ' '; s += buf; }
    } else {
        codon_text(start_codon, buf); s += buf;
        for (int k = 1; k < r.L; k++) { codon_text((int)((r.gene >> (6 * (r.L - 1 - k))) & 63), buf); s += ' '; s += buf; }
    }
    return s;
}

static std::string ops_text(const Rec &r, int mode, const std::vector<u8> &list_codons, u8 start_codon, const u8 *table)
{
    std::string s;
    for (int k = 0; k < r.L; k++) {
        int c = mode == 2 ? list_codons[r.gene * MAXL + k] : (k == 0 ? start_codon : (int)((r.gene >> (6 * (r.L - 1 - k))) & 63));
        if (k) s += ' ';
        s += OPNAME[table[c]];
    }
    return s;
}

static const char *REC_HEADER = "#code\tL\tglen\tgene\tcodons\tops\tclaims\tcopied\tdivides\tchildren\tfaithful\tbirths_world\tfaithful_world\tgrandchildren\tdeaths\tlive_end\tmaxdepth\tfirst_birth\tprefiltered\tclaims_world\tcopied_world\tintact\n";

static void write_rec(FILE *f, const Rec &r, const char *code, int mode, const std::vector<u8> &list_codons, u8 start_codon, const u8 *table)
{
    fprintf(f, "%s\t%d\t%d\t%llu\t%s\t%s\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%d\t%u\t%u\t%u\t%u\n", code, r.L, r.glen, (unsigned long long)r.gene,
            gene_text(r, mode, list_codons, start_codon).c_str(), ops_text(r, mode, list_codons, start_codon, table).c_str(),
            r.claims, r.copied, r.divides, r.births0, r.faithful0, r.births, r.faithful, r.gchildren, r.deaths, r.live_end, r.maxdepth,
            (int)r.first_birth - 1, r.flags & 1, r.claims_world, r.copied_world, r.intact0);
}

int main(int argc, char **argv)
{
    i64 L = 0, W = 8, G = 256, ticks = 64, T = 64, I = 16, chunk = 1 << 22, chunk_start = 0, chunk_end = -1, tpb = 64, random_n = -1;
    bool dna2 = false;
    u64 seed = 1, rseed = 1; bool all = false, uniform = false, soup = false, prefilter = true, slow_seek = false; int gpu = 0; u32 log_mask = 3;
    std::string genes_path, out_path, log_path, records_path, hash_path, chunk_log_path;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char * { if (i + 1 >= argc) usage(); return argv[++i]; };
        if (a == "--L") L = atoll(next());
        else if (a == "--all") all = true;
        else if (a == "--random") random_n = atoll(next());
        else if (a == "--rseed") rseed = strtoull(next(), NULL, 10);
        else if (a == "--genes") genes_path = next();
        else if (a == "--W") W = atoll(next());
        else if (a == "--G") G = atoll(next());
        else if (a == "--ticks") ticks = atoll(next());
        else if (a == "--T") T = atoll(next());
        else if (a == "--I") I = atoll(next());
        else if (a == "--seed") seed = strtoull(next(), NULL, 10);
        else if (a == "--uniform-code") uniform = true;
        else if (a == "--dna2") dna2 = true;
        else if (a == "--soup") soup = true;
        else if (a == "--no-prefilter") prefilter = false;
        else if (a == "--slow-seek") slow_seek = true;
        else if (a == "--chunk") chunk = atoll(next());
        else if (a == "--chunk-start") chunk_start = atoll(next());
        else if (a == "--chunk-end") chunk_end = atoll(next());
        else if (a == "--tpb") tpb = atoll(next());
        else if (a == "--gpu") gpu = atoi(next());
        else if (a == "--out") out_path = next();
        else if (a == "--log") log_path = next();
        else if (a == "--log-mask") log_mask = (u32)strtoul(next(), NULL, 0);
        else if (a == "--records") records_path = next();
        else if (a == "--hash-ticks") hash_path = next();
        else if (a == "--chunk-log") chunk_log_path = next();
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); usage(); }
    }
    int mode = !genes_path.empty() ? 2 : random_n >= 0 ? 1 : all ? 0 : -1;
    if (mode < 0) usage();
    if (mode != 2 && (L < 2 || L > 8)) { fprintf(stderr, "--L must be 2..8 for --all / --random\n"); return 1; }
    if (W < 2 || W > MAXW || G < 6 || W * G > 4 * MAXNB) { fprintf(stderr, "need 2 <= W <= %d, G >= 6, W*G <= %d\n", MAXW, 4 * MAXNB); return 1; }
    if (ticks < 1 || ticks > 65535) { fprintf(stderr, "--ticks must be 1..65535\n"); return 1; }
    if (tpb < 32 || tpb > 256 || tpb % 32) { fprintf(stderr, "--tpb must be a multiple of 32 up to 256\n"); return 1; }
    if (!hash_path.empty() && mode != 2) { fprintf(stderr, "--hash-ticks needs --genes\n"); return 1; }
    if (mode == 2) prefilter = false;

    u8 table[64]; build_table(table, uniform, dna2);
    int start_codon = -1, stop_codon = -1;
    for (int c = 0; c < 64; c++) { if (table[c] == START && start_codon < 0) start_codon = c; if (table[c] == STOP && stop_codon < 0) stop_codon = c; }

    SP P; memset(&P, 0, sizeof P);
    P.seed = seed; P.W = (i32)W; P.G = (i32)G; P.N = (i32)(W * G); P.NB = (P.N + 3) / 4; P.maxk = (i32)(G / 3 - 1); P.I = (i32)I; P.T = (i32)T; P.ticks = (i32)ticks;
    P.mode = mode; P.L = (i32)L; P.prefilter = prefilter ? 1 : 0; P.hash_ticks = hash_path.empty() ? 0 : 1; P.soup = soup ? 1 : 0; P.slow_seek = slow_seek ? 1 : 0; P.dna2 = dna2 ? 1 : 0;
    P.log_mask = log_mask; P.start_codon = (u8)start_codon; P.stop_codon = (u8)stop_codon; P.rseed = rseed;
    const i32 stride = ((P.NB + 3) & ~3) + 4;      /* 4-byte aligned, one bank offset per thread against conflicts */

    /* the worlds of this run */
    u64 n_total;
    std::vector<u8> list_codons, list_len;
    if (mode == 2) {
        FILE *f = fopen(genes_path.c_str(), "r");
        if (!f) { fprintf(stderr, "cannot read %s\n", genes_path.c_str()); return 1; }
        char line[4096];
        while (fgets(line, sizeof line, f)) {
            std::string s = line; size_t h = s.find('#'); if (h != std::string::npos) s = s.substr(0, h);
            bool ok; std::vector<int> cod = parse_genome(s, ok);
            if (cod.empty()) continue;
            if (!ok || (i64)cod.size() > std::min<i64>(MAXL, P.maxk + 1)) { fprintf(stderr, "bad gene line: %s", line); return 1; }
            size_t off = list_codons.size(); list_codons.resize(off + MAXL, 0);
            for (size_t k = 0; k < cod.size(); k++) list_codons[off + k] = (u8)cod[k];
            list_len.push_back((u8)cod.size());
        }
        fclose(f);
        n_total = list_len.size();
        if (n_total == 0) { fprintf(stderr, "no genes in %s\n", genes_path.c_str()); return 1; }
    } else {
        P.range = 1ULL << (6 * (L - 1));
        n_total = mode == 0 ? P.range : (u64)random_n;
    }
    if (chunk_end < 0 || (u64)chunk_end > n_total) chunk_end = (i64)n_total;
    if (chunk_start < 0 || chunk_start > chunk_end) { fprintf(stderr, "bad chunk range\n"); return 1; }
    u64 n_run = (u64)(chunk_end - chunk_start);
    if ((u64)chunk > n_run) chunk = (i64)std::max<u64>(n_run, 1);
    if (!hash_path.empty() && n_run * (u64)ticks > (1ULL << 28)) { fprintf(stderr, "--hash-ticks: too many worlds x ticks\n"); return 1; }

    CUDA_CHECK(cudaSetDevice(gpu));
    Out O; memset(&O, 0, sizeof O);
    u8 *d_table; CUDA_CHECK(cudaMalloc(&d_table, 64)); CUDA_CHECK(cudaMemcpy(d_table, table, 64, cudaMemcpyHostToDevice)); O.table = d_table;
    CUDA_CHECK(cudaMalloc(&O.hist, NBUCKET * NH * sizeof(u64)));
    std::vector<u64> hist_total(NBUCKET * NH, 0), hist_chunk(NBUCKET * NH);
    std::vector<Rec> log_host, recs_host;
    FILE *log_f = NULL, *rec_f = NULL, *chunk_f = NULL;
    if (!log_path.empty()) {
        P.log_cap = (u32)std::min<i64>(chunk, (1LL << 24));
        CUDA_CHECK(cudaMalloc(&O.log, (size_t)P.log_cap * sizeof(Rec))); CUDA_CHECK(cudaMalloc(&O.log_n, sizeof(u32)));
        log_host.resize(P.log_cap);
        log_f = fopen(log_path.c_str(), chunk_start > 0 ? "a" : "w"); if (!log_f) { fprintf(stderr, "cannot write %s\n", log_path.c_str()); return 1; }
        if (chunk_start == 0) fputs(REC_HEADER, log_f);
    }
    if (!records_path.empty()) {
        CUDA_CHECK(cudaMalloc(&O.recs, (size_t)chunk * sizeof(Rec))); recs_host.resize(chunk);
        rec_f = fopen(records_path.c_str(), chunk_start > 0 ? "a" : "w"); if (!rec_f) { fprintf(stderr, "cannot write %s\n", records_path.c_str()); return 1; }
        if (chunk_start == 0) fputs(REC_HEADER, rec_f);
    }
    if (!chunk_log_path.empty()) {
        chunk_f = fopen(chunk_log_path.c_str(), chunk_start > 0 ? "a" : "w"); if (!chunk_f) { fprintf(stderr, "cannot write %s\n", chunk_log_path.c_str()); return 1; }
        if (chunk_start == 0) fprintf(chunk_f, "#chunk_start\tchunk_end\tseconds\tstrings\tclaimed\tcopied_any\tdivided\tchildren_any\tfaithful_any\tgrandchildren_any\tprefiltered\tlogged\n");
    }
    u8 *d_list_codons = NULL, *d_list_len = NULL;
    if (mode == 2) {
        CUDA_CHECK(cudaMalloc(&d_list_codons, list_codons.size())); CUDA_CHECK(cudaMemcpy(d_list_codons, list_codons.data(), list_codons.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_list_len, list_len.size())); CUDA_CHECK(cudaMemcpy(d_list_len, list_len.data(), list_len.size(), cudaMemcpyHostToDevice));
        O.list_codons = d_list_codons; O.list_len = d_list_len;
    }
    std::vector<u64> hashes_host;
    if (!hash_path.empty()) { CUDA_CHECK(cudaMalloc(&O.hashes, (size_t)n_total * ticks * sizeof(u64))); hashes_host.resize((size_t)n_total * ticks); }
    u8 *d_soup = NULL;
    if (soup) {
        std::vector<u8> sp(P.NB, 0);
        for (i32 i = 0; i < P.N; i++) { u32 v = (u32)(rnd(seed, -1, (u64)i, 0) & 3u); sp[i >> 2] |= (u8)(v << ((i & 3) * 2)); }
        CUDA_CHECK(cudaMalloc(&d_soup, P.NB)); CUDA_CHECK(cudaMemcpy(d_soup, sp.data(), P.NB, cudaMemcpyHostToDevice)); O.soup = d_soup;
    }
    size_t smem = (size_t)tpb * stride;
    int smem_max = 0; CUDA_CHECK(cudaDeviceGetAttribute(&smem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, gpu));
    if ((i64)smem + (i64)(NBUCKET * NH * sizeof(u32) + 128) > smem_max) {
        fprintf(stderr, "--tpb %lld needs %zu bytes of shared memory per block, the device allows %d: use a smaller --tpb or world\n", (long long)tpb, smem, smem_max);
        return 1;
    }
    if (smem > 48 * 1024) CUDA_CHECK(cudaFuncSetAttribute(k_sweep, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));

    fprintf(stderr, "dna_sweep mode=%s L=%lld W=%lld G=%lld N=%d ticks=%lld T=%lld I=%lld seed=%llu code=%s soup=%d prefilter=%d worlds=%llu [%lld,%lld) chunk=%lld tpb=%lld smem=%zu gpu=%d\n",
            mode == 0 ? "all" : mode == 1 ? "random" : "genes", (long long)L, (long long)W, (long long)G, P.N, (long long)ticks, (long long)T, (long long)I,
            (unsigned long long)seed, uniform ? (dna2 ? "uniform-dna2" : "uniform") : (dna2 ? "standard-dna2" : "standard"), soup ? 1 : 0, prefilter ? 1 : 0, (unsigned long long)n_total, (long long)chunk_start,
            (long long)chunk_end, (long long)chunk, (long long)tpb, smem, gpu);

    auto t_start = std::chrono::steady_clock::now();
    u64 logged_total = 0, log_overflow = 0; int n_chunks = 0;
    const char *code = uniform ? (dna2 ? "uniform-dna2" : "uniform") : (dna2 ? "standard-dna2" : "standard");
    for (i64 c0 = chunk_start; c0 < chunk_end; c0 += chunk) {
        i64 c1 = std::min<i64>(c0 + chunk, chunk_end);
        auto tc = std::chrono::steady_clock::now();
        P.gene_start = (u64)c0; P.n_worlds = (u64)(c1 - c0);
        CUDA_CHECK(cudaMemset(O.hist, 0, NBUCKET * NH * sizeof(u64)));
        if (O.log_n) CUDA_CHECK(cudaMemset(O.log_n, 0, sizeof(u32)));
        unsigned blocks = (unsigned)((P.n_worlds + tpb - 1) / tpb);
        k_sweep<<<blocks, (unsigned)tpb, smem>>>(P, O, stride);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(hist_chunk.data(), O.hist, NBUCKET * NH * sizeof(u64), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < hist_total.size(); i++) hist_total[i] += hist_chunk[i];
        u32 nlog = 0;
        if (O.log) {
            CUDA_CHECK(cudaMemcpy(&nlog, O.log_n, sizeof nlog, cudaMemcpyDeviceToHost));
            u32 keep = std::min(nlog, P.log_cap);
            if (nlog > P.log_cap) log_overflow += nlog - P.log_cap;
            if (keep) {
                CUDA_CHECK(cudaMemcpy(log_host.data(), O.log, (size_t)keep * sizeof(Rec), cudaMemcpyDeviceToHost));
                std::sort(log_host.begin(), log_host.begin() + keep, [](const Rec &a, const Rec &b) { return a.gene < b.gene; });
                for (u32 i = 0; i < keep; i++) write_rec(log_f, log_host[i], code, mode, list_codons, (u8)start_codon, table);
                fflush(log_f);
            }
            logged_total += keep;
        }
        if (O.recs) {
            CUDA_CHECK(cudaMemcpy(recs_host.data(), O.recs, (size_t)P.n_worlds * sizeof(Rec), cudaMemcpyDeviceToHost));
            for (u64 i = 0; i < P.n_worlds; i++) write_rec(rec_f, recs_host[i], code, mode, list_codons, (u8)start_codon, table);
            fflush(rec_f);
        }
        double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - tc).count();
        u64 cs[NH]; memset(cs, 0, sizeof cs);
        for (int b = 0; b < NBUCKET; b++) for (int i = 0; i < NH; i++) cs[i] += hist_chunk[b * NH + i];
        if (chunk_f) {
            fprintf(chunk_f, "%lld\t%lld\t%.3f\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\t%u\n", (long long)c0, (long long)c1, secs,
                    (unsigned long long)cs[H_STRINGS], (unsigned long long)cs[H_CLAIMED], (unsigned long long)cs[H_COPIED], (unsigned long long)cs[H_DIVIDED],
                    (unsigned long long)cs[H_BIRTHS0], (unsigned long long)cs[H_FAITHFUL0], (unsigned long long)cs[H_GCHILD], (unsigned long long)cs[H_PREFILTERED], nlog);
            fflush(chunk_f);
        }
        n_chunks++;
        double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
        fprintf(stderr, "chunk [%lld,%lld) %.3f s (%.3g worlds/s) claimed=%llu copied=%llu divided=%llu children=%llu faithful=%llu grandchildren=%llu logged=%u elapsed=%.1f s\n",
                (long long)c0, (long long)c1, secs, P.n_worlds / secs, (unsigned long long)cs[H_CLAIMED], (unsigned long long)cs[H_COPIED],
                (unsigned long long)cs[H_DIVIDED], (unsigned long long)cs[H_BIRTHS0], (unsigned long long)cs[H_FAITHFUL0], (unsigned long long)cs[H_GCHILD], nlog, el);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();

    if (!hash_path.empty()) {
        CUDA_CHECK(cudaMemcpy(hashes_host.data(), O.hashes, (size_t)n_total * ticks * sizeof(u64), cudaMemcpyDeviceToHost));
        FILE *f = fopen(hash_path.c_str(), "w");
        if (!f) { fprintf(stderr, "cannot write %s\n", hash_path.c_str()); return 1; }
        fprintf(f, "#gene\ttick\thash\n");
        for (u64 g = (u64)chunk_start; g < (u64)chunk_end; g++) for (i64 t = 0; t < ticks; t++)
            fprintf(f, "%llu\t%lld\t%016llx\n", (unsigned long long)g, (long long)t, (unsigned long long)hashes_host[g * ticks + t]);
        fclose(f);
    }
    if (log_f) fclose(log_f);
    if (rec_f) fclose(rec_f);
    if (chunk_f) fclose(chunk_f);

    /* summary */
    u64 tot[NH]; memset(tot, 0, sizeof tot);
    for (int b = 0; b < NBUCKET; b++) for (int i = 0; i < NH; i++) tot[i] += hist_total[b * NH + i];
    fprintf(stderr, "summary worlds=%llu seconds=%.3f worlds_per_second=%.4g strings=%llu claimed=%llu copied_any=%llu divided=%llu children_any=%llu faithful_any=%llu "
                    "intact_any=%llu grandchildren_any=%llu prefiltered=%llu logged=%llu log_overflow=%llu\n", (unsigned long long)n_run, secs, n_run / secs,
            (unsigned long long)tot[H_STRINGS], (unsigned long long)tot[H_CLAIMED], (unsigned long long)tot[H_COPIED], (unsigned long long)tot[H_DIVIDED],
            (unsigned long long)tot[H_BIRTHS0], (unsigned long long)tot[H_FAITHFUL0], (unsigned long long)tot[H_INTACT0], (unsigned long long)tot[H_GCHILD],
            (unsigned long long)tot[H_PREFILTERED], (unsigned long long)logged_total, (unsigned long long)log_overflow);
    if (!out_path.empty()) {
        FILE *f = fopen(out_path.c_str(), "w");
        if (!f) { fprintf(stderr, "cannot write %s\n", out_path.c_str()); return 1; }
        fprintf(f, "{\n \"params\": {\"mode\": \"%s\", \"L\": %lld, \"W\": %lld, \"G\": %lld, \"N\": %d, \"ticks\": %lld, \"T\": %lld, \"I\": %lld, \"seed\": %llu, "
                   "\"code\": \"%s\", \"soup\": %s, \"prefilter\": %s, \"rseed\": %llu, \"range\": %llu, \"chunk_start\": %lld, \"chunk_end\": %lld, \"chunk\": %lld, \"tpb\": %lld, \"log_mask\": %u},\n",
                mode == 0 ? "all" : mode == 1 ? "random" : "genes", (long long)L, (long long)W, (long long)G, P.N, (long long)ticks, (long long)T, (long long)I,
                (unsigned long long)seed, code, soup ? "true" : "false", prefilter ? "true" : "false", (unsigned long long)rseed, (unsigned long long)P.range,
                (long long)chunk_start, (long long)chunk_end, (long long)chunk, (long long)tpb, log_mask);
        fprintf(f, " \"worlds\": %llu, \"seconds\": %.3f, \"worlds_per_second\": %.6g, \"chunks\": %d, \"logged\": %llu, \"log_overflow\": %llu,\n",
                (unsigned long long)n_run, secs, n_run / secs, n_chunks, (unsigned long long)logged_total, (unsigned long long)log_overflow);
        auto bucket_json = [&](const u64 *h) {
            std::string js = "{";
            char buf[128];
            for (int i = 0; i < 20; i++) { snprintf(buf, sizeof buf, "%s\"%s\": %llu", i ? ", " : "", HNAME[i], (unsigned long long)h[i]); js += buf; }
            js += ", \"fecundity_hist\": [";
            for (int i = 0; i < 17; i++) { snprintf(buf, sizeof buf, "%s%llu", i ? ", " : "", (unsigned long long)h[H_FEC + i]); js += buf; }
            js += "], \"children_hist\": [";
            for (int i = 0; i < 17; i++) { snprintf(buf, sizeof buf, "%s%llu", i ? ", " : "", (unsigned long long)h[H_B0 + i]); js += buf; }
            js += "], \"maxdepth_hist\": [";
            for (int i = 0; i < 7; i++) { snprintf(buf, sizeof buf, "%s%llu", i ? ", " : "", (unsigned long long)h[H_DEPTH + i]); js += buf; }
            snprintf(buf, sizeof buf, "], \"intact_any\": %llu, \"sum_intact\": %llu, \"intact_fecundity_hist\": [", (unsigned long long)h[H_INTACT0], (unsigned long long)h[H_SUM_INTACT0]);
            js += buf;
            for (int i = 0; i < 17; i++) { snprintf(buf, sizeof buf, "%s%llu", i ? ", " : "", (unsigned long long)h[H_IFEC + i]); js += buf; }
            js += "]}";
            return js;
        };
        fprintf(f, " \"total\": %s,\n \"by_gene_length\": {", bucket_json(tot).c_str());
        bool first = true;
        for (int b = 0; b < NBUCKET; b++) {
            if (hist_total[b * NH + H_STRINGS] == 0) continue;
            fprintf(f, "%s\n  \"%d%s\": %s", first ? "" : ",", b, b == NBUCKET - 1 ? "+" : "", bucket_json(&hist_total[b * NH]).c_str());
            first = false;
        }
        fprintf(f, "\n }\n}\n");
        fclose(f);
    }
    return 0;
}
