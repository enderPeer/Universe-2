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
 *          Added for the seeded worlds of docs/12 (2026-10-09; none of them changes the world, only the output):
 *          --log-births FILE   every DIVIDE birth unsampled (no 1-in-1000 FIFO sampling), TSV
 *                              "#tick parent child parent_hash child_hash faithful depth parent_age copied parent_P":
 *                              the child's lineage depth, the parent's age at the division (before the ageing of
 *                              step 4), the cells the parent had copied into the child since the claim, and the
 *                              parent's template pointer P after its instructions of the tick (P - copied is the
 *                              first cell it copied from, modulo N, if nothing moved P in between)
 *          --log-filter divide|faithful|all   rows of --log-births (default divide: step-3 births only; faithful:
 *                              only those with equal hashes; all: spontaneous births too, with depth 0, age -1, copied -1)
 *          --log-every k       only ticks with t mod k = 0 go to --log-births (default 1 = every tick)
 *          --log-codons        two more columns in --log-births: the parent's and the child's gene as codon letters,
 *                              read at the end of the tick (steps 4 to 7 may have changed them: unrepaired damage, or
 *                              the inflow refilling a parent that died in the same tick, both rare); the hashes of
 *                              the row are the ones of step 3
 *          --depth-cap K       census: depth_hist buckets every depth >= K into one entry [K, n] and the census
 *                              gets a "max_depth" field (K = 0, the default, leaves the census as the reference prints it)
 *          --top-genes FILE    at every census, for each of the 10 most common gene hashes among the living, one
 *                              carrier's codons and operations: "tick hash count slot depth codons ops"
 *          --max-seconds s     stop after the first tick (checked every 100 ticks) at which the main loop has run
 *                              for s seconds; that tick is then the last one (final census, --out, --dump as usual)
 *          --slot-map FILE     at every census, one byte per slot appended to FILE (a frame for a video): 0 dead,
 *                              1 .. 254 live with depth + 1 (254 = depth >= 253), 255 dead but claimed as a child;
 *                              the frame order is the census order, S bytes per frame
 *          DNA-3 (docs/17, 2026-10-10; changes the world, verified against sim/dna_ref.py on 13 configurations):
 *          --days 1,2,3 | --dna3   the days of creation (each implies --dna2; day 3 needs 1 and 2): day 1 clock, light
 *                              and energy (--day-len --light-len --lmax --emax --eshift --e-seed --e-spont --g-pass;
 *                              T = 0 = no age limit), day 2 grid and firmament (--W --firm --seed-slots --seed-terrain),
 *                              day 3 terrain, HARVEST, copy cost, maintenance (--gp-sea --gp-land --gh-sea --gh-land
 *                              --c-copy --maint --inflow-land --land-cell --land-thr)
 *          --slot-map-mode m   comma list of depth (FILE itself, the docs/12 map), gene, clade, energy, plant (FILE.<m>);
 *                              with day 2 the terrain goes to FILE.terrain once. The clade (founder slot) is output only.
 *          The warp-reduced ledger in k_execute uses __shfl_down_sync, so every thread of the grid reaches the end.
 *          --op-census FILE    (docs/17 day 7, output only: not hashed, not in the reference, no effect on the world) at
 *                              every census the full histogram of operation sequences of the living (codons 0 .. first
 *                              STOP mapped to operations, synonyms collapsed; op hash = FNV-1a 64 over the operation
 *                              numbers): "C tick live births divide_births op_faithful spontaneous deaths changed
 *                              n_seq" per census (births since the last census; op_faithful = DIVIDE births whose
 *                              child has the parent's operation sequence; changed = living processors whose operation
 *                              sequence differs from the one they were born with, or had at the start or resume of
 *                              this process), "S ophash ops" the first time a sequence is printed by this process,
 *                              and "H ophash count zone0 t0 t1 t2 t3" per sequence (carriers in zone 0; carriers whose
 *                              day-6 streak for task k is 3, i.e. paid on their last answer)
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
#include <ctype.h>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <chrono>

typedef uint64_t u64; typedef int64_t i64; typedef uint32_t u32; typedef int32_t i32; typedef uint8_t u8;

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error at %s:%d: %s (%s)\n", __FILE__, __LINE__, cudaGetErrorString(e_), #x); exit(2); } } while (0)

/* ---------------------------------------------------------------- operations and the codon table (section 3) */
enum Op : u8 { NOP = 0, START, STOP, COPY, R1, L1, R3, L3, JNZB, SETP, SETQ, ALLOC, DIVIDE, SKNZ, SKZ, LDA, STA,
               INC, FLIP, SEEK, COPYR, REWIND, TERM, HARVEST, SENSE, WAIT, EAT, MOVE, EXEC, SIGNAL, LISTEN, IN, OUT, N_OPS };   /* SENSE, WAIT: day 4; EAT, MOVE, EXEC: day 5 */   /* TERM only under --dna2 (docs/15), HARVEST on day 3 (docs/17) */
/* docs/17 section 19: macro i (--macros FILE) is the operation code N_OPS + i, named M0 .. M7 */
#define MACRO_MAX 8
#define MACRO_BODY_MAX 32

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

static void build_table_dna1(u8 *table, bool uniform);
/* docs/15 fix 1: under --dna2 TERM takes TTA and TTG from Leu (COPY keeps CTT CTC CTA CTG); under the uniform code
 * it takes codon value 21, the first NOP */
static void build_table(u8 *table, bool uniform, bool dna2, bool day3);
static void build_table(u8 *table, bool uniform, bool dna2, bool day3, bool day4);
/* docs/17 day 5: EAT on CCA, CCG (from Pro/SETP), MOVE on ACA, ACG (from Thr/SETQ), EXEC on GTA, GTG (from Val/JNZB);
 * uniform code 25, 26, 27 */
static void build_table(u8 *table, bool uniform, bool dna2, bool day3, bool day4, bool day5);
/* docs/17 day 6: SIGNAL on CGA, LISTEN on CGG (from Arg/L1), IN on TCA, OUT on TCG (from Ser/R1); uniform 28..31 */
static void build_table(u8 *table, bool uniform, bool dna2, bool day3, bool day4, bool day5, bool day6)
{
    build_table(table, uniform, dna2, day3, day4, day5);
    if (!day6) return;
    if (uniform) { table[28] = SIGNAL; table[29] = LISTEN; table[30] = IN; table[31] = OUT; return; }
    const char *cod[4] = { "CGA", "CGG", "TCA", "TCG" };
    const u8 old[4] = { L1, L1, R1, R1 }, nw[4] = { SIGNAL, LISTEN, IN, OUT };
    for (int i = 0; i < 4; i++) { int v = codon_value(cod[i]); if (table[v] != old[i]) { fprintf(stderr, "day 6 table error\n"); exit(1); } table[v] = nw[i]; }
}
static void build_table(u8 *table, bool uniform, bool dna2, bool day3, bool day4, bool day5)
{
    build_table(table, uniform, dna2, day3, day4);
    if (!day5) return;
    if (uniform) { table[25] = EAT; table[26] = MOVE; table[27] = EXEC; return; }
    const char *cod[6] = { "CCA", "CCG", "ACA", "ACG", "GTA", "GTG" };
    const u8 old[3] = { SETP, SETQ, JNZB }, nw[3] = { EAT, MOVE, EXEC };
    for (int i = 0; i < 6; i++) { int v = codon_value(cod[i]); if (table[v] != old[i / 2]) { fprintf(stderr, "day 5 table error\n"); exit(1); } table[v] = nw[i / 2]; }
}
static void build_table(u8 *table, bool uniform, bool dna2, bool day3, bool day4)
{
    build_table(table, uniform, dna2, day3);
    if (!day4) return;
    /* docs/17 day 4: SENSE on AGT, WAIT on AGC, taken from Ser (R1 keeps TCT TCC TCA TCG); uniform code 23, 24 */
    if (uniform) { table[23] = SENSE; table[24] = WAIT; return; }
    int a = codon_value("AGT"), b = codon_value("AGC");
    if (table[a] != R1 || table[b] != R1) { fprintf(stderr, "day 4 table error\n"); exit(1); }
    table[a] = SENSE; table[b] = WAIT;
}
static void build_table(u8 *table, bool uniform, bool dna2, bool day3)
{
    build_table_dna1(table, uniform);
    if (!dna2) return;
    if (uniform) table[21] = TERM;
    else {
        const char *tc[2] = { "TTA", "TTG" };
        for (int i = 0; i < 2; i++) { int v = codon_value(tc[i]); if (table[v] != COPY) { fprintf(stderr, "dna2 table error\n"); exit(1); } table[v] = TERM; }
    }
    if (!day3) return;
    /* docs/17 day 3: HARVEST takes AGA and AGG from Arg (L1 keeps CGT CGC CGA CGG); uniform code: codon 22 */
    if (uniform) { table[22] = HARVEST; return; }
    const char *hc[2] = { "AGA", "AGG" };
    for (int i = 0; i < 2; i++) { int v = codon_value(hc[i]); if (table[v] != L1) { fprintf(stderr, "dna3 table error\n"); exit(1); } table[v] = HARVEST; }
}
static void build_table_dna1(u8 *table, bool uniform)
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
    int dna2;                 /* docs/15: TERM, DIVIDE re-arms P, COPY/COPYR NOP without a claimed child */
    /* docs/17, DNA-3 days 1 to 3 */
    int d1, d2, d3;
    i64 day_len, light_len, lmax, emax, eshift, e_seed, e_spont, gp[2], gh[2], c_copy, maint;
    int inflow_land;
    int d4, shade;            /* docs/17 day 4; section 14 shading (a day-3 option) */
    int d5; i64 c_move, c_fly, eat_shift, eat_eff;   /* docs/17 day 5 */
    int d6; i64 c_walk, task_bonus;                   /* docs/17 day 6 */
    int ladder; i64 streak_len, pool_cell, pool_max, pool_in, nreg, rx;   /* docs/17 section 18 */
    int in_fresh;             /* docs/17 section 19: the n-th IN of a processor's tick draws rnd(t, s, 4000 + n) */
    int nmac; int mlen[MACRO_MAX]; u8 mbody[MACRO_MAX * MACRO_BODY_MAX];   /* docs/17 section 19: the macro table */
    i64 year_days, moon_days, season_amp, lat_amp, lmoon;
    i64 W, R, firm;
    int opc;                  /* --op-census (docs/17 day 7): output only, never read by the world's rules */
};

/* the ten 64-bit fields of section 4 in the hash order of section 8, then the docs/17 day-1 energy store E
 * (hashed only on day 1) */
struct Proc { i64 slot, pc, H, P, Q, child, A, age, depth, copied, E, xs, sig, xin, open, streak, ldir, mk, mpc; };   /* mk, mpc: section 19 (the macro being executed, -1 none, and the index into its body) */   /* ldir: section 18 */   /* sig..streak: day 6 */   /* xs: day 5, the slot whose gene runs */

/* one birth of this tick; depth = the child's lineage depth, page = the parent's age at the division (before step
 * 4), copied = cells the parent wrote into the child since the claim; a spontaneous birth has parent -1, page -1,
 * copied -1 */
struct BirthRec { i32 parent, child, depth, page, copied; u64 phash, chash; i64 pP; };   /* pP: the parent's P after step 1 */

struct World {
    u8 *fwd, *rev, *fwd_old;  /* packed 2-bit strands; fwd_old = snapshot at the start of the tick */
    Proc *proc;               /* dense, indexed by slot */
    u8 *live;                 /* 1 if slot s has a processor */
    i32 *claimed_by;          /* parent slot that claimed s, or -1 */
    i32 *claim;               /* ALLOC resolution scratch: lowest requesting slot per candidate (atomicMin) */
    u8 *alloc_req;            /* ALLOC requested this tick */
    i32 *div_mark;            /* child slot marked for birth by DIVIDE this tick, or -1 */
    i32 *div_copied;          /* the parent's copied count at that DIVIDE (for the log only) */
    i32 *refill_k;            /* inflow scratch: largest k refilling this slot this tick, or -1 */
    u64 *ghash;               /* gene hash per live processor (set at birth, refreshed for the census) */
    i32 *cens_depth;          /* census scratch: depth of live processor or -1 */
    u64 *counters;            /* births, faithful, spontaneous, deaths (cumulative) */
    BirthRec *log; u32 *log_n; u32 log_cap;   /* births of this tick (only when --births is given) */
    u8 *table;                /* the codon table (64 entries) in device memory */
    /* docs/17 */
    u8 *terrain;              /* per slot: 0 sea, 1 land, 2 firmament (day 2; all 0 without day 2) */
    i64 *div_energy;          /* endowment of the child marked by DIVIDE this tick */
    u64 *ledger;              /* 9 cumulative energy counters, the reference's order */
    u64 *claim64;             /* day-2 ALLOC: lowest key per candidate (atomicMin) */
    i32 *alloc_cand;          /* day-2 ALLOC: the candidate each requester proposed, or -1 */
    i32 *clade;               /* output only (slot map): slot of the depth-0 ancestor, not hashed, not in the reference */
    u64 *cens_acc;            /* census accumulators: energy, zone 0, zone 1, live on land, harvesters, ... */
    i32 *eat_target, *move_op, *move_to;   /* day 5: prey of this tick's EAT, MOVE operand, granted move target */
    i64 *e_loss, *e_gain, *move_cost;      /* day 5: predation results, cost of a proposed move */
    u8 *sig_old; u64 *task_count;          /* day 6: signals at the start of the tick; rewarded answers per task */
    u64 *macro_exec;                       /* section 19: cumulative macro codons fetched, per macro (census) */
    u8 *xin_old; i64 *pools; u64 *pool_used;   /* section 18: inputs at the tick start (16 none); task pools per region */
    u64 *ophash, *opbirth, *op_cnt; u8 *opflag;   /* --op-census, output only: op hash now / at birth, op-faithful births, flags */
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

/* --op-census (docs/17 day 7): FNV-1a 64 over the operation numbers of the gene (START included, STOP excluded), so
 * synonymous spellings share one hash */
__host__ __device__ inline u64 op_hash(const u8 *f, u64 base, const u8 *table, i64 maxk)
{
    u64 h = 14695981039346656037ULL;
    for (i64 k = 0; k <= maxk; k++) {
        u32 o = table[codon_at(f, base, k)];
        if (o == STOP) break;
        h = (h ^ (u64)o) * 1099511628211ULL;
    }
    return h;
}

/* ---------------------------------------------------------------- kernels */
__device__ inline i64 modN(i64 x, i64 N) { x %= N; return x < 0 ? x + N : x; }

/* docs/17 day 2: neighbour offsets N, NE, E, SE, S, SW, W, NW (y grows downwards) */
__constant__ int NB_DX[8] = { 0, 1, 1, 1, 0, -1, -1, -1 };
__constant__ int NB_DY[8] = { -1, -1, 0, 1, 1, 1, 0, -1 };

/* docs/17 day 6: the signals as they are at the start of the tick (0 for slots without a processor) */
__global__ void k_sig_snapshot(World w, Params P)
{
    i64 s = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (s < P.S) {
        w.sig_old[s] = w.live[s] ? (u8)w.proc[s].sig : 0;
        if (P.ladder) w.xin_old[s] = w.live[s] ? (u8)w.proc[s].xin : 16;
    }
}

/* docs/17 day 6: the task ladder on the 2-bit input x */
__device__ inline u32 task_f(int k, u32 x) { return k == 0 ? x ^ 1u : k == 1 ? (x + 1) & 3u : k == 2 ? x ^ 3u : (x + 3) & 3u; }
__device__ __constant__ int TASK_RANK[4] = { 0, 0, 2, 2 };

/* docs/17 section 18: the long ladder on the 4-bit input x (a = x >> 2, b = x & 3), exactly sim/dna_ref.py LADDER_TASKS;
 * task 18 is the social task (b xor the low half of the listened neighbour's input) */
#define LADDER_NT 19
__device__ __constant__ int LT_MASK[LADDER_NT] = { 15, 15, 15, 15, 15, 15, 15, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15, 3, 3 };
__device__ __constant__ int LT_RANK[LADDER_NT] = { 0, 0, 0, 0, 1, 1, 1, 1, 1, 4, 2, 2, 3, 4, 4, 4, 4, 4, 3 };
__device__ inline u32 ladder_f(int k, u32 x)
{
    switch (k) {
    case 0: return x ^ 1u; case 1: return x ^ 4u; case 2: return (x + 4) & 15u; case 3: return x & 12u;
    case 4: return x ^ 5u; case 5: return (x + 8) & 15u; case 6: return x & 3u; case 7: return (4 * x) & 15u; case 8: return x >> 2;
    case 9: return (x + 1) & 15u; case 10: return x | 3u; case 11: return x >> 2;
    case 12: return 15u - x;
    case 13: return (x + 15) & 15u; case 14: return (2 * x) & 15u; case 15: return x >> 1; case 16: return (u32)__popc(x) & 1u;
    default: return (x >> 2) ^ (x & 3u);
    }
}

/* sum over the 32 lanes of a warp; every lane of the warp must call it */
__device__ inline u64 warp_sum64(u64 v) { for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o); return v; }
__device__ inline void warp_add(u64 *dst, u64 v) { v = warp_sum64(v); if ((threadIdx.x & 31) == 0 && v) atomicAdd((unsigned long long *)dst, (unsigned long long)v); }

/* docs/17 day 4: light of slot s at tick t and the four sky channels (sun, moon, season, stars), exactly as
 * sim/dna_ref.py World.sky */
__host__ __device__ inline i64 tri(i64 u, i64 P) { i64 v = 4 * u - 2 * P; return P - (v < 0 ? -v : v); }
__host__ __device__ inline i64 sky(const Params &P, i64 t, i64 s, u32 ch[4])
{
    i64 Py = P.year_days, Pm = P.moon_days, R = P.R, d = t / P.day_len, phi = t % P.day_len, y = s / P.W;
    i64 sy = tri(d % Py, Py) + Py;
    i64 sz = y < R / 2 ? 2 * Py - sy : sy;
    i64 Lz = P.light_len - P.season_amp + (P.season_amp * sz) / Py;
    bool up = phi < Lz;
    i64 ys = R / 4 + ((R / 2) * sy) / (2 * Py);
    i64 dy = y - ys; if (dy < 0) dy = -dy;
    i64 dist = dy < R - dy ? dy : R - dy;
    i64 fac = 16 - (P.lat_amp * dist) / (R / 2);
    i64 ml = ((tri(d % Pm, Pm) + Pm) * 3 + Pm) / (2 * Pm);
    ch[0] = !up ? 0u : fac >= 14 ? 3u : fac >= 11 ? 2u : 1u;
    ch[1] = (u32)ml;
    ch[2] = (u32)((4 * sz) / (2 * Py + 1));
    ch[3] = up ? 0u : (u32)(1 + (3 * y) / R);
    return up ? (P.lmax * fac) / 16 : P.lmoon * ml;
}

/* docs/17 days 3 and 4: the gene of slot base (codons 1 .. len-1) contains a codon of operation op (opc = N_OPS: any
 * macro codon, section 19) */
__host__ __device__ inline bool has_op(const u8 *f, u64 base, const u8 *table, i64 maxk, u32 opc)
{
    if (table[codon_at(f, base, 0)] == STOP) return false;
    for (i64 k = 1; k <= maxk; k++) { u32 op = table[codon_at(f, base, k)]; if (op == STOP) return false; if (op == opc || (opc == N_OPS && op >= N_OPS)) return true; }
    return false;
}

/* docs/17 day 3: the gene of slot base (codons 1 .. len-1) contains a HARVEST codon */
__host__ __device__ inline bool has_harvest(const u8 *f, u64 base, const u8 *table, i64 maxk)
{
    if (table[codon_at(f, base, 0)] == STOP) return false;   /* gene of length 0 (the reference counts codons from 0) */
    for (i64 k = 1; k <= maxk; k++) { u32 op = table[codon_at(f, base, k)]; if (op == STOP) return false; if (op == HARVEST) return true; }
    return false;
}

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

/* step 1: one thread per slot, I instructions of its processor. docs/17 day 1: first the passive light, then a
 * budget of min(I, E >> eshift) instructions at one energy unit each; day 3: HARVEST and the cost of copied cells.
 * Every thread of the grid runs to the end (no early return) so that the warp reductions of the ledger are legal. */
__global__ void k_execute(World w, Params P, i64 t, i64 lam0)
{
    i64 s = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    u64 l_pass = 0, l_harv = 0, l_instr = 0, l_copy = 0, l_waste = 0, l_task = 0, l_tc[LADDER_NT], l_mx[MACRO_MAX];
    for (int k = 0; k < LADDER_NT; k++) l_tc[k] = 0;
    for (int k = 0; k < MACRO_MAX; k++) l_mx[k] = 0;
    if (s < P.S && w.live[s]) {
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
    /* docs/17 day 5: the code being executed: own gene (live) or a neighbour's (tick-start snapshot) */
    i64 xs = P.d5 ? p.xs : s; const u8 *csrc = w.fwd; i64 cb0 = base; bool ate = false;
    if (xs != s) {
        if (expressible(w.fwd_old, (u64)xs * G, table, maxk)) {
            csrc = w.fwd_old; cb0 = xs * G; len = gene_len(w.fwd_old, (u64)cb0, table, maxk);
            if (p.pc >= len) { xs = s; csrc = w.fwd; cb0 = base; p.pc = 1; len = gene_len(w.fwd, base, table, maxk); }
        } else { xs = s; p.pc = 1; }
    }
#define GO_HOME() do { xs = s; csrc = w.fwd; cb0 = base; len = gene_len(w.fwd, base, table, maxk); } while (0)
    i64 budget = P.I;
    int ter = 0; bool harvested = false; u64 n_in = 0;   /* n_in: INs executed earlier in this tick (section 19 --in-fresh) */
    i64 lam = lam0, shade_k = 9; u32 chan[4] = { 0, 0, 0, 0 };
    if (P.d1) {
        ter = w.terrain[s] == 1 ? 1 : 0;
        if (P.d4) lam = sky(P, t, s, chan);           /* docs/17 day 4: light and sky channels of this slot */
        i64 g = (lam * P.gp[ter]) >> 4;
        if (P.shade) {                                /* docs/17 section 14: (9 - live neighbours) / 9 */
            i64 x = s % P.W, y = s / P.W, nl = 0;
            for (int d = 0; d < 8; d++) nl += w.live[((y + NB_DY[d] + P.R) % P.R) * P.W + (x + NB_DX[d] + P.W) % P.W];
            shade_k = 9 - nl; g = (g * shade_k) / 9;
        }
        i64 add = g < P.emax - p.E ? g : P.emax - p.E;
        p.E += add; l_pass += (u64)add; l_waste += (u64)(g - add);
        budget = p.E >> P.eshift; if (budget > P.I) budget = P.I;
    }
    for (i64 n = 0; n < budget; n++) {
        if (P.d1) { if (p.E < 1) break; p.E -= 1; l_instr += 1; }
        u32 op = NOP; bool in_body = false;
        if (P.nmac && p.mk >= 0) {   /* docs/17 section 19: inside a macro the operation comes from its body, pc is untouched */
            op = P.mbody[p.mk * MACRO_BODY_MAX + p.mpc]; p.mpc += 1; in_body = true;
        } else {
            if (p.pc >= 0 && p.pc <= maxk) op = table[codon_at(csrc, (u64)cb0, p.pc)];   /* own gene live, a foreign one from the snapshot */
            p.pc += 1;
            if (op >= N_OPS) {   /* a macro codon: its unit is paid, the body starts with the next iteration */
                int m = (int)(op - N_OPS); l_mx[m] += 1;
                if (P.mlen[m] > 0) { p.mk = m; p.mpc = 0; }
                op = NOP;
            }
        }
        bool stall = false, wait_end = false;
        switch (op) {
        case SENSE:    /* docs/17 day 4: operand consumed; skip the next codon if the operand's sky channel is nonzero */
            if (p.pc < len) { u32 c = codon_at(csrc, (u64)cb0, p.pc); p.pc += 1; if (chan[c & 3u]) p.pc += 1; }
            break;
        case SIGNAL: p.sig = p.A; break;               /* docs/17 day 6: visible to the neighbours from the next tick */
        case IN: if (P.ladder) {                       /* docs/17 section 18: 4-bit input, a into cell G-2, b into G-1 */
            u32 xv = (u32)(rnd(P.seed, t, (u64)s, 4000 + (P.in_fresh ? n_in : 0)) & 15u); n_in += 1;   /* section 19: --in-fresh */
            WR(base + G - 2, xv >> 2); WR(base + G - 1, xv & 3u);
            p.A = xv & 3u; p.H = base + G - 1;
            if (xs == s) len = gene_len(w.fwd, base, table, maxk);
            p.xin = xv; p.open = 1;
            break;
        } else {                                       /* docs/17 day 6: input into the I/O cell (last cell of the slot), H there, A */
            u32 xv = (u32)(rnd(P.seed, t, (u64)s, 4000 + (P.in_fresh ? n_in : 0)) & 3u); n_in += 1;
            p.A = xv; p.H = base + G - 1; WR(p.H, xv);
            if (xs == s) len = gene_len(w.fwd, base, table, maxk);
            p.xin = xv; p.open = 1;
        } break;
        case OUT:                                      /* docs/17 day 6: one answer per input; streaks of 3 pay */
            if (P.ladder) {                            /* section 18: the answer is the two I/O cells; 3-bit streaks; pools */
                if (!p.open) break;
                p.open = 0;
                u32 v = 4u * get2(w.fwd, (u64)(base + G - 2)) + get2(w.fwd, (u64)(base + G - 1)), xi = (u32)p.xin;
                i64 nlo = -1;
                if (p.ldir < 8) {
                    i64 x = s % P.W, y = s / P.W, nb = ((y + NB_DY[p.ldir] + P.R) % P.R) * P.W + (x + NB_DX[p.ldir] + P.W) % P.W;
                    if (w.xin_old[nb] < 16) nlo = w.xin_old[nb] & 3;
                }
                i64 reg = P.pool_cell ? ((s / P.W) / P.pool_cell) * P.rx + (s % P.W) / P.pool_cell : -1;
                for (int k = 0; k < LADDER_NT; k++) {
                    bool ok = k < LADDER_NT - 1 ? (v & (u32)LT_MASK[k]) == (ladder_f(k, xi) & (u32)LT_MASK[k])
                                                : nlo >= 0 && (v & 3u) == ((xi & 3u) ^ (u32)nlo);
                    i64 c = (p.streak >> (3 * k)) & 7;
                    c = ok ? (c + 1 > 7 ? 7 : c + 1) : 0;
                    p.streak = (p.streak & ~(7LL << (3 * k))) | (c << (3 * k));
                    if (c >= P.streak_len) {
                        i64 g = P.task_bonus << LT_RANK[k];
                        if (reg >= 0) { g = (g * w.pools[reg * LADDER_NT + k]) / P.pool_max; atomicAdd((unsigned long long *)&w.pool_used[reg * LADDER_NT + k], 1ULL); }
                        i64 add = g < P.emax - p.E ? g : P.emax - p.E;
                        p.E += add; l_task += (u64)add; l_waste += (u64)(g - add); l_tc[k] += 1;
                    }
                }
                break;
            }
            if (p.open) {
                p.open = 0;
                for (int k = 0; k < 4; k++) {
                    i64 c = (p.streak >> (2 * k)) & 3;
                    c = task_f(k, (u32)p.xin) == (u32)p.A ? (c + 1 > 3 ? 3 : c + 1) : 0;
                    p.streak = (p.streak & ~(3LL << (2 * k))) | (c << (2 * k));
                    if (c == 3) {
                        i64 g = P.task_bonus << TASK_RANK[k], add = g < P.emax - p.E ? g : P.emax - p.E;
                        p.E += add; l_task += (u64)add; l_waste += (u64)(g - add); l_tc[k] += 1;
                    }
                }
            }
            break;
        case EAT: case MOVE: case EXEC: case LISTEN:   /* docs/17 days 5, 6: operand consumed; NOP if it lies beyond the code's end */
            if (p.pc < len) {
                u32 c = codon_at(csrc, (u64)cb0, p.pc); p.pc += 1;
                i64 x = s % P.W, y = s / P.W, nb = ((y + NB_DY[c & 7u] + P.R) % P.R) * P.W + (x + NB_DX[c & 7u] + P.W) % P.W;
                if (op == LISTEN) { p.A = w.live[nb] ? w.sig_old[nb] : 0; if (P.ladder) p.ldir = (i64)(c & 7u); }
                else if (op == EAT) { if (!ate && w.live[nb]) { ate = true; w.eat_target[s] = (i32)nb; } }
                else if (op == EXEC) {
                    if (expressible(w.fwd_old, (u64)nb * G, table, maxk)) { xs = nb; csrc = w.fwd_old; cb0 = nb * G; len = gene_len(w.fwd_old, (u64)cb0, table, maxk); p.pc = 1; }
                } else if (p.child < 0) { w.move_op[s] = (i32)c; w.alloc_req[s] = 0; wait_end = true; }   /* MOVE ends the tick */
            }
            break;
        case WAIT: wait_end = true; break;             /* docs/17 day 4: the tick ends after this instruction */
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
                if (IN_OWN(p.H) && xs == s) len = gene_len(w.fwd, base, table, maxk);   /* the own gene may have changed */
            }
            break;
        case LDA: p.A = RD(p.H); break;
        case SKZ: if (RD(p.H) == 0) { if (in_body) p.mpc += 1; else p.pc += 1; } break;     /* section 19: skips and jumps act within a body */
        case SKNZ: if (RD(p.H) != 0) { if (in_body) p.mpc += 1; else p.pc += 1; } break;
        case JNZB: if (RD(p.H) != 0) { if (in_body) { p.mpc -= 2; if (p.mpc < 0) p.mpc = 0; } else { p.pc -= 2; if (p.pc < 1) p.pc = 1; } } break;
        case SETP: p.P = p.H; break;
        case SETQ: p.Q = p.H; break;
        case REWIND: p.P = base; if (cbase >= 0) p.Q = cbase; break;
        case SEEK: {
            if (p.pc >= len) break;                       /* template beyond the gene end: NOP */
            u32 tc = codon_at(csrc, (u64)cb0, p.pc);      /* the template codon, consumed */
            p.pc += 1;
            u32 t0 = (tc >> 4) ^ 3u, t1 = ((tc >> 2) & 3u) ^ 3u, t2 = (tc & 3u) ^ 3u;
            for (i64 d = 1; d <= 3 * G; d++) {
                i64 h0 = modN(p.H + d, N), h1 = modN(h0 + 1, N), h2 = modN(h0 + 2, N);
                if (RD(h0) == t0 && RD(h1) == t1 && RD(h2) == t2) { p.H = h0; break; }
            }
        } break;
        case TERM: {   /* docs/15 fix 1: skip the next codon unless the template has just passed a STOP codon */
            i64 o = p.P % G; bool term = false;
            if (o >= 3 && o % 3 == 0 && o / 3 <= maxk + 1) {
                u32 c3 = 16u * RD(p.P - 3) + 4u * RD(p.P - 2) + RD(p.P - 1);
                term = table[c3] == STOP;
            }
            if (!term) p.pc += 1;
        } break;
        case HARVEST:  /* docs/17 day 3: once per tick, light times the terrain's HARVEST gain */
            if (!harvested) {
                harvested = true;
                i64 g = (lam * P.gh[ter]) >> 4; if (P.shade) g = (g * shade_k) / 9;
                i64 add = g < P.emax - p.E ? g : P.emax - p.E;
                p.E += add; l_harv += (u64)add; l_waste += (u64)(g - add);
            }
            break;
        case ALLOC: if (p.child < 0 && !(P.d5 && w.move_op[s] >= 0)) w.alloc_req[s] = 1; break;
        case COPY: case COPYR:
            if (P.dna2 && p.child < 0) break;              /* docs/15 fix 3: no claimed child, nothing happens */
            if (p.child >= 0 && IN_CHILD(p.Q)) {
                if (P.d3) {                                /* docs/17 day 3: no energy for the cell: back to this COPY */
                    if (p.E < P.c_copy) { p.pc -= 1; stall = true; break; }
                    p.E -= P.c_copy; l_copy += (u64)P.c_copy;
                }
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
            if (p.child >= 0 && p.copied >= 6) { w.div_mark[s] = (i32)p.child; w.div_copied[s] = (i32)p.copied; p.child = -1; cbase = -1; p.copied = 0; p.Q = 0;
                                                if (P.dna2) p.P = base;   /* docs/15 fix 2: re-arm the template */
                                                if (P.d1) { i64 x = p.E >> 1; p.E -= x; w.div_energy[s] = x; } }   /* docs/17: half the store to the child */
            break;
        }
        if (stall) break;                                 /* the tick ends before the wrap rule, as in the reference */
        if (in_body && p.mpc >= P.mlen[p.mk]) p.mk = -1;   /* section 19: the body is done */
        if (p.pc >= len) { if (xs != s) GO_HOME(); p.pc = 1; }   /* the gene loops, skipping START; a foreign gene returns home */
        if (wait_end) break;
    }
    if (P.d5) p.xs = xs;
#undef GO_HOME
    w.proc[s] = p;
#undef IN_OWN
#undef IN_CHILD
#undef RD
#undef WR
    }
    if (P.d1) {
        warp_add(&w.ledger[0], l_pass); warp_add(&w.ledger[1], l_harv); warp_add(&w.ledger[3], l_instr);
        warp_add(&w.ledger[4], l_copy); warp_add(&w.ledger[8], l_waste);
    }
    if (P.d6) { warp_add(&w.ledger[12], l_task); for (int k = 0; k < (P.ladder ? LADDER_NT : 4); k++) warp_add(&w.task_count[k], l_tc[k]); }
    if (P.nmac) for (int k = 0; k < P.nmac; k++) warp_add(&w.macro_exec[k], l_mx[k]);   /* section 19 (P.nmac is uniform across the grid) */
}

/* docs/17 section 18, step 1b: the pools pay for this tick's rewards and refill (one thread per region and task) */
__global__ void k_pools(World w, Params P)
{
    i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= P.nreg * LADDER_NT) return;
    i64 v = w.pools[i] - (i64)w.pool_used[i]; if (v < 0) v = 0;
    v += P.pool_in; if (v > P.pool_max) v = P.pool_max;
    w.pools[i] = v; w.pool_used[i] = 0;
}

/* docs/17 day 5, step 2a: predation on the stores after step 1. One thread per prey: its eaters (neighbours whose EAT
 * aimed at it) in increasing key (rnd(t, e, 3000) >> 32) * 2^32 + e take floor(rest / 2^eat_shift) each and keep
 * floor(x * eat_eff / 16); losses and gains are applied afterwards (k_eat_apply), so the order of threads is irrelevant */
__global__ void k_eat(World w, Params P, i64 t)
{
    i64 pr = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    u64 eaten = 0, lost = 0;
    if (pr < P.S && w.live[pr]) {
        i64 x = pr % P.W, y = pr / P.W; u64 key[8]; i64 who[8]; int n = 0;
        for (int d = 0; d < 8; d++) {
            i64 e = ((y + NB_DY[d] + P.R) % P.R) * P.W + (x + NB_DX[d] + P.W) % P.W;
            if (w.eat_target[e] == (i32)pr) {
                bool dup = false; for (int k = 0; k < n; k++) dup |= who[k] == e;   /* a 3-wide torus can repeat a neighbour */
                if (!dup) { key[n] = ((rnd(P.seed, t, (u64)e, 3000) >> 32) << 32) | (u64)e; who[n] = e; n++; }
            }
        }
        if (n) {
            for (int i = 1; i < n; i++) for (int j = i; j > 0 && key[j - 1] > key[j]; j--) { u64 a = key[j]; key[j] = key[j - 1]; key[j - 1] = a; i64 b = who[j]; who[j] = who[j - 1]; who[j - 1] = b; }
            i64 rem = w.proc[pr].E;
            for (int k = 0; k < n; k++) {
                i64 xx = rem >> P.eat_shift; rem -= xx; i64 g = (xx * P.eat_eff) >> 4;
                w.e_gain[who[k]] = g; eaten += (u64)xx; lost += (u64)(xx - g);
            }
            w.e_loss[pr] = w.proc[pr].E - rem;
        }
    }
    warp_add(&w.ledger[9], eaten); warp_add(&w.ledger[10], lost);
}
__global__ void k_eat_apply(World w, Params P)
{
    i64 s = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    u64 over = 0;
    if (s < P.S) {
        w.eat_target[s] = -1;
        if (w.e_loss[s] || w.e_gain[s]) {
            i64 v = w.proc[s].E - w.e_loss[s] + w.e_gain[s];
            if (v > P.emax) { over = (u64)(v - P.emax); v = P.emax; }
            w.proc[s].E = v; w.e_loss[s] = 0; w.e_gain[s] = 0;
        }
    }
    warp_add(&w.ledger[10], over);
}

/* step 2a: every requesting processor proposes its candidate; the lowest slot index wins (unsigned atomicMin
 * against the sentinel 0xFFFFFFFF, which no slot index reaches since S < 2^31) */

/* docs/17 day 2: the requester scans its eight neighbours from direction rnd(t, s, 0) mod 8 on the state after step
 * 1 and proposes the first free one (not live, not claimed, not firmament); the lowest key
 * (rnd(t, s, 2000) >> 32) * 2^32 + s wins a contested candidate */
__device__ inline u64 alloc_key(const Params &P, i64 t, i64 s) { return ((rnd(P.seed, t, (u64)s, 2000) >> 32) << 32) | (u64)s; }

__global__ void k_alloc_propose(World w, Params P, i64 t)
{
    i64 s = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= P.S || !w.live[s]) return;
    if (P.d5 && w.move_op[s] >= 0) {   /* docs/17 day 5: a move of 1 to 4 slots; 1 = swim (sea to sea), more = fly */
        u32 c = (u32)w.move_op[s]; i64 dist = 1 + ((c >> 3) & 3u), x = s % P.W, y = s / P.W;
        i64 tg = ((y + NB_DY[c & 7u] * dist) % P.R + P.R) % P.R * P.W + ((x + NB_DX[c & 7u] * dist) % P.W + P.W) % P.W;
        i64 cost = dist == 1 ? P.c_move : P.c_fly * dist;
        if (dist == 1 && P.d6 && (w.terrain[s] == 1 || w.terrain[tg] == 1)) cost = P.c_walk;   /* docs/17 day 6: walking */
        w.alloc_cand[s] = -1;
        if (w.terrain[tg] == 2 || w.live[tg] || w.claimed_by[tg] >= 0 || w.proc[s].E < cost) return;
        if (dist == 1 && !P.d6 && (w.terrain[s] != 0 || w.terrain[tg] != 0)) return;
        w.alloc_cand[s] = (i32)tg; w.move_cost[s] = cost;
        atomicMin((unsigned long long *)&w.claim64[tg], (unsigned long long)alloc_key(P, t, s));
        return;
    }
    if (!w.alloc_req[s]) return;
    if (P.d2) {
        i64 x = s % P.W, y = s / P.W; u32 r = (u32)(rnd(P.seed, t, (u64)s, 0) & 7u); i32 cand = -1;
        for (int i = 0; i < 8; i++) {
            int d = (int)((r + i) & 7u);
            i64 cx = (x + NB_DX[d] + P.W) % P.W, cy = (y + NB_DY[d] + P.R) % P.R, c = cy * P.W + cx;
            if (w.terrain[c] != 2 && !w.live[c] && w.claimed_by[c] < 0) { cand = (i32)c; break; }
        }
        w.alloc_cand[s] = cand;
        if (cand >= 0) atomicMin((unsigned long long *)&w.claim64[cand], (unsigned long long)alloc_key(P, t, s));
        return;
    }
    i64 cand = (s + 1 + (i64)(rnd(P.seed, t, (u64)s, 0) % (u64)(P.S - 1))) % P.S;
    if (!w.live[cand] && w.claimed_by[cand] < 0) atomicMin((unsigned *)&w.claim[cand], (unsigned)s);
}

/* step 2b: grants */
__global__ void k_alloc_grant(World w, Params P, i64 t)
{
    i64 s = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= P.S || !w.live[s]) return;
    if (P.d5 && w.move_op[s] >= 0) {
        i32 c = w.alloc_cand[s]; w.move_op[s] = -1;
        if (c >= 0 && w.claim64[c] == alloc_key(P, t, s)) w.move_to[s] = c;
        return;
    }
    if (!w.alloc_req[s]) return;
    w.alloc_req[s] = 0;
    if (P.d2) {
        i32 c = w.alloc_cand[s];
        if (c >= 0 && w.claim64[c] == alloc_key(P, t, s)) {
            w.proc[s].child = c; w.proc[s].Q = (i64)c * P.G; w.proc[s].copied = 0;
            w.claimed_by[c] = (i32)s;
        }
        return;
    }
    i64 cand = (s + 1 + (i64)(rnd(P.seed, t, (u64)s, 0) % (u64)(P.S - 1))) % P.S;
    if ((u32)w.claim[cand] == (u32)s) {
        w.proc[s].child = cand; w.proc[s].Q = cand * P.G; w.proc[s].copied = 0;
        w.claimed_by[cand] = (i32)s;
    }
}

/* docs/17 day 5, step 2c: granted moves: the G cells and the processor go to the target, H and P move with it, the
 * source keeps its cells (a shell) and is dead. Sources are live and targets were free, so movers never overlap. */
__global__ void k_move(World w, Params P)
{
    i64 s = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    u64 spent = 0;
    if (s < P.S && w.move_to[s] >= 0) {
        i64 c = w.move_to[s]; w.move_to[s] = -1;
        u64 sb = (u64)s * P.G, tb = (u64)c * P.G;
        for (i64 j = 0; j < P.G; j++) {
            set2(w.fwd, tb + j, get2(w.fwd, sb + j), P.atomic_writes);
            set2(w.rev, tb + j, get2(w.rev, sb + j), P.atomic_writes);
        }
        Proc q = w.proc[s];
        i64 sh = (c - s) * P.G, N = (i64)P.N;
        q.slot = c; q.H = ((q.H + sh) % N + N) % N; q.P = ((q.P + sh) % N + N) % N; q.Q = 0; q.child = -1; q.copied = 0;
        q.E -= w.move_cost[s]; spent = (u64)w.move_cost[s];
        q.xs = q.xs == s ? c : q.xs;
        w.proc[c] = q; w.live[s] = 0; w.live[c] = 1; w.clade[c] = w.clade[s];
        if (P.opc) w.opbirth[c] = w.opbirth[s];   /* output only */
    }
    warp_add(&w.ledger[11], spent);
}

/* step 3: births from DIVIDE marks; the claim on the child is released either way */
__global__ void k_births(World w, Params P, i64 t)
{
    i64 s = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= P.S || w.div_mark[s] < 0) return;
    i64 c = w.div_mark[s];
    w.div_mark[s] = -1;
    w.claimed_by[c] = -1;
    if (!expressible(w.fwd, (u64)c * P.G, w.table, P.maxk)) {
        if (P.d1) atomicAdd((unsigned long long *)&w.ledger[7], (unsigned long long)w.div_energy[s]);   /* lost endowment */
        return;
    }
    Proc q; q.slot = c; q.pc = 1; q.H = c * P.G; q.P = c * P.G; q.Q = 0; q.child = -1; q.A = 0; q.age = 0;
    q.depth = w.proc[s].depth + 1; q.copied = 0; q.E = P.d1 ? w.div_energy[s] : 0; q.xs = c; q.sig = q.xin = q.open = q.streak = 0; q.ldir = 8; if (P.ladder) q.xin = 16;
    q.mk = -1; q.mpc = 0;
    w.proc[c] = q; w.live[c] = 1; w.clade[c] = w.clade[s];
    u64 ph = gene_hash(w.fwd, (u64)s * P.G, w.table, P.maxk), ch = gene_hash(w.fwd, (u64)c * P.G, w.table, P.maxk);
    w.ghash[c] = ch;
    atomicAdd((unsigned long long *)&w.counters[0], 1ULL);
    if (ph == ch) atomicAdd((unsigned long long *)&w.counters[1], 1ULL);
    if (P.opc) {   /* output only */
        u64 po = op_hash(w.fwd, (u64)s * P.G, w.table, P.maxk), co = op_hash(w.fwd, (u64)c * P.G, w.table, P.maxk);
        w.opbirth[c] = co;
        if (po == co) atomicAdd((unsigned long long *)&w.op_cnt[0], 1ULL);
    }
    if (w.log) { u32 n = atomicAdd(w.log_n, 1u); if (n < w.log_cap) { BirthRec r = { (i32)s, (i32)c, (i32)q.depth, (i32)w.proc[s].age, w.div_copied[s], ph, ch, w.proc[s].P }; w.log[n] = r; } }
}

/* step 4: ageing and deaths */
__global__ void k_deaths(World w, Params P)
{
    i64 s = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= P.S || !w.live[s]) return;
    Proc *p = &w.proc[s];
    p->age += 1;
    bool dies = (p->age >= P.T && (P.T > 0 || !P.d1)) || !expressible(w.fwd, (u64)s * P.G, w.table, P.maxk);   /* docs/17: T = 0 on day 1, no limit */
    if (!dies && P.d3) {                     /* docs/17 day 3: maintenance; who cannot pay it starves */
        if (p->E < P.maint) { dies = true; atomicAdd((unsigned long long *)&w.counters[4], 1ULL); }
        else { p->E -= P.maint; if (P.maint) atomicAdd((unsigned long long *)&w.ledger[5], (unsigned long long)P.maint); }
    }
    if (dies) {
        w.live[s] = 0;
        atomicAdd((unsigned long long *)&w.counters[3], 1ULL);
        if (P.d1 && p->E) atomicAdd((unsigned long long *)&w.ledger[6], (unsigned long long)p->E);
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
    if (P.d2) { u8 tr = w.terrain[d]; if (tr == 2 || (tr == 1 && !P.inflow_land)) return; }   /* docs/17: no inflow into firmament (day 2) or land (day 3) */
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
    q.age = 0; q.depth = 0; q.copied = 0; q.E = P.d1 ? P.e_spont : 0; q.xs = d; q.sig = q.xin = q.open = q.streak = 0; q.ldir = 8; if (P.ladder) q.xin = 16;
    q.mk = -1; q.mpc = 0;
    w.proc[d] = q; w.live[d] = 1; w.clade[d] = (i32)d;
    if (P.d1 && P.e_spont) atomicAdd((unsigned long long *)&w.ledger[2], (unsigned long long)P.e_spont);
    u64 ch = gene_hash(w.fwd, base, w.table, P.maxk);
    w.ghash[d] = ch;
    if (P.opc) w.opbirth[d] = op_hash(w.fwd, base, w.table, P.maxk);   /* output only */
    atomicAdd((unsigned long long *)&w.counters[0], 1ULL);
    atomicAdd((unsigned long long *)&w.counters[2], 1ULL);
    if (w.log) { u32 n = atomicAdd(w.log_n, 1u); if (n < w.log_cap) { BirthRec r = { -1, (i32)d, 0, -1, -1, 0, ch, -1 }; w.log[n] = r; } }
}

/* census scratch: depth (-1 if dead, -2 if dead but claimed) and a fresh gene hash per live processor */
__global__ void k_census(World w, Params P)
{
    i64 s = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    u64 e = 0, z0 = 0, z1 = 0, ll = 0, hv = 0, sn = 0, sl = 0, ea = 0, mv = 0, ex = 0, fo = 0, sg = 0, li = 0, tk = 0, mu = 0;
    if (s < P.S) {
        if (w.live[s]) {
            w.cens_depth[s] = (i32)w.proc[s].depth; w.ghash[s] = gene_hash(w.fwd, (u64)s * P.G, w.table, P.maxk);
            e = (u64)w.proc[s].E;
            if (s / P.W < P.R / 2) z0 = 1; else z1 = 1;
            if (P.d3) { ll = w.terrain[s] == 1; hv = has_harvest(w.fwd, (u64)s * P.G, w.table, P.maxk); }
            if (P.d4) { sn = has_op(w.fwd, (u64)s * P.G, w.table, P.maxk, SENSE); sl = has_op(w.fwd, (u64)s * P.G, w.table, P.maxk, WAIT); }
            if (P.d5) { u64 b = (u64)s * P.G; ea = has_op(w.fwd, b, w.table, P.maxk, EAT); mv = has_op(w.fwd, b, w.table, P.maxk, MOVE);
                        ex = has_op(w.fwd, b, w.table, P.maxk, EXEC); fo = w.proc[s].xs != s; }
            if (P.d6) { u64 b = (u64)s * P.G; sg = has_op(w.fwd, b, w.table, P.maxk, SIGNAL); li = has_op(w.fwd, b, w.table, P.maxk, LISTEN);
                        tk = has_op(w.fwd, b, w.table, P.maxk, OUT); if (P.nmac) mu = has_op(w.fwd, b, w.table, P.maxk, N_OPS); }
        }
        else w.cens_depth[s] = w.claimed_by[s] >= 0 ? -2 : -1;
    }
    if (P.d1 || P.d2) {   /* docs/17 census additions (W = 1, R = 2 without day 2 keeps the division harmless) */
        warp_add(&w.cens_acc[0], e); warp_add(&w.cens_acc[1], z0); warp_add(&w.cens_acc[2], z1);
        warp_add(&w.cens_acc[3], ll); warp_add(&w.cens_acc[4], hv); warp_add(&w.cens_acc[5], sn); warp_add(&w.cens_acc[6], sl);
        warp_add(&w.cens_acc[7], ea); warp_add(&w.cens_acc[8], mv); warp_add(&w.cens_acc[9], ex); warp_add(&w.cens_acc[10], fo);
        warp_add(&w.cens_acc[11], sg); warp_add(&w.cens_acc[12], li); warp_add(&w.cens_acc[13], tk); warp_add(&w.cens_acc[14], mu);
    }
}

/* --op-census (docs/17 day 7, output only): the op hash of every live processor and a flag byte: bit k (k = 0..3) =
 * the day-6 streak of task k is 3, bit 6 = zone 0, bit 7 = the operation sequence differs from the one at birth */
__global__ void k_opcensus(World w, Params P)
{
    i64 s = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= P.S || !w.live[s]) return;
    u64 o = op_hash(w.fwd, (u64)s * P.G, w.table, P.maxk);
    u8 fl = 0;
    if (P.d6) { i64 st = w.proc[s].streak; for (int k = 0; k < 4; k++) if (((st >> (2 * k)) & 3) == 3) fl |= (u8)(1 << k); }
    if (s / P.W < P.R / 2) fl |= 64;
    if (o != w.opbirth[s]) fl |= 128;
    w.ophash[s] = o; w.opflag[s] = fl;
}

/* --op-census: at the start (fresh or resumed) every living processor's current sequence counts as its birth sequence */
__global__ void k_opbirth_init(World w, Params P)
{
    i64 s = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (s < P.S && w.live[s]) w.opbirth[s] = op_hash(w.fwd, (u64)s * P.G, w.table, P.maxk);
}

/* output only (docs/17): one byte per slot for the slot-map modes gene, clade, energy, plant (after k_census, which
 * refreshes ghash): 0 dead, 255 dead but claimed, otherwise 1 .. 254 */
__global__ void k_slotmap(World w, Params P, int mode, u8 *out)
{
    i64 s = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= P.S) return;
    u8 b;
    if (!w.live[s]) b = w.claimed_by[s] >= 0 ? 255 : 0;
    else if (mode == 1) b = (u8)(1 + w.ghash[s] % 253);
    else if (mode == 2) b = (u8)(1 + splitmix64((u64)(u32)w.clade[s]) % 253);
    else if (mode == 3) { i64 e = w.proc[s].E * 253 / (P.emax + 1); b = (u8)(1 + (e > 252 ? 252 : e)); }
    else if (mode == 4) b = has_harvest(w.fwd, (u64)s * P.G, w.table, P.maxk) ? 2 : 1;
    else if (mode == 6) {   /* animals (docs/17 day 5): 1 + EAT + 2 MOVE + 4 EXEC + 8 running a neighbour's gene now */
        u64 base = (u64)s * P.G;
        b = (u8)(1 + (has_op(w.fwd, base, w.table, P.maxk, EAT) ? 1 : 0) + (has_op(w.fwd, base, w.table, P.maxk, MOVE) ? 2 : 0)
                 + (has_op(w.fwd, base, w.table, P.maxk, EXEC) ? 4 : 0) + (w.proc[s].xs != s ? 8 : 0));
    }
    else {   /* traits (docs/17 section 14): 1 + HARVEST + 2 SENSE + 4 WAIT */
        u64 base = (u64)s * P.G;
        b = (u8)(1 + (has_op(w.fwd, base, w.table, P.maxk, HARVEST) ? 1 : 0) + (has_op(w.fwd, base, w.table, P.maxk, SENSE) ? 2 : 0)
                 + (has_op(w.fwd, base, w.table, P.maxk, WAIT) ? 4 : 0));
    }
    out[s] = b;
}

/* docs/17 days 2 and 3: terrain per slot (0 sea, 1 land, 2 firmament), exactly as sim/dna_ref.py make_terrain */
static void make_terrain(std::vector<u8> &ter, u64 seed, i64 S, i64 W, i64 firm, bool day3, i64 L, i64 thr16)
{
    i64 R = S / W, half = R / 2;
    ter.assign((size_t)S, 0);
    i64 nx = day3 ? W / L : 0, ny = day3 ? R / L : 0;
    std::vector<u64> lat((size_t)(nx * ny));
    for (i64 i = 0; i < nx * ny; i++) lat[i] = rnd(seed, -2, (u64)i, 9) >> 48;
    u64 thr = (u64)thr16 * (u64)L * (u64)L;
    for (i64 y = 0; y < R; y++) {
        if (firm && y % half < firm) { for (i64 x = 0; x < W; x++) ter[y * W + x] = 2; continue; }
        if (!day3) continue;
        i64 j0 = y / L, fy = y % L, j1 = (j0 + 1) % ny;
        for (i64 x = 0; x < W; x++) {
            i64 i0 = x / L, fx = x % L, i1 = (i0 + 1) % nx;
            u64 v = (lat[j0 * nx + i0] * (L - fx) + lat[j0 * nx + i1] * fx) * (L - fy) + (lat[j1 * nx + i0] * (L - fx) + lat[j1 * nx + i1] * fx) * fy;
            if (v >= thr) ter[y * W + x] = 1;
        }
    }
}

static const char *OP_NAMES[N_OPS + MACRO_MAX] = { "NOP", "START", "STOP", "COPY", "R1", "L1", "R3", "L3", "JNZB", "SETP", "SETQ", "ALLOC",
                                       "DIVIDE", "SKNZ", "SKZ", "LDA", "STA", "INC", "FLIP", "SEEK", "COPYR", "REWIND", "TERM", "HARVEST", "SENSE", "WAIT", "EAT", "MOVE", "EXEC",
                                       "SIGNAL", "LISTEN", "IN", "OUT", "M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7" };   /* M0 .. M7: section 19 */

/* docs/17 section 19.2: the macro table from FILE, one macro per line "NAME CODON OP OP ..." (# comments; an empty body
 * makes the codon a NOP that costs its unit: the control arm C). Fills P.nmac, P.mlen, P.mbody and the codon values
 * (uniform code: 32 + i whatever the column says) and reassigns the codons in the table: a donor codon of the standard
 * code (CTA, CTG from COPY; GCA, GCG from ALLOC; GGA, GGG from DIVIDE; ATA from SKNZ; TGC from COPYR) must still carry
 * its donor's operation, a uniform codon 32 + i must be NOP. */
static const char *MACRO_BODY_OPS[] = { "NOP", "FLIP", "INC", "LDA", "STA", "L1", "R1", "R3", "L3", "SKZ", "SKNZ", "JNZB", NULL };
static const char *MACRO_DONORS[][2] = { { "CTA", "COPY" }, { "CTG", "COPY" }, { "GCA", "ALLOC" }, { "GCG", "ALLOC" }, { "GGA", "DIVIDE" }, { "GGG", "DIVIDE" }, { "ATA", "SKNZ" }, { "TGC", "COPYR" }, { NULL, NULL } };
static void load_macros(Params &P, u8 *table, const std::string &path, bool uniform, std::vector<u64> &codons)
{
    FILE *f = fopen(path.c_str(), "r");
    if (!f) { fprintf(stderr, "cannot read %s\n", path.c_str()); exit(1); }
    char line[4096]; P.nmac = 0; codons.clear();
    for (int k = 0; k < MACRO_MAX; k++) P.mlen[k] = 0;
    memset(P.mbody, 0, sizeof P.mbody);
    while (fgets(line, sizeof line, f)) {
        char *hash = strchr(line, '#'); if (hash) *hash = 0;
        std::vector<std::string> tok;
        for (char *tp = strtok(line, " \t\r\n"); tp; tp = strtok(NULL, " \t\r\n")) tok.push_back(tp);
        if (tok.empty()) continue;
        if (tok.size() < 2) { fprintf(stderr, "macro line needs a name and a codon\n"); exit(1); }
        if (P.nmac >= MACRO_MAX) { fprintf(stderr, "at most %d macros\n", MACRO_MAX); exit(1); }
        int i = P.nmac;
        if ((int)tok.size() - 2 > MACRO_BODY_MAX) { fprintf(stderr, "macro %s: body longer than %d\n", tok[0].c_str(), MACRO_BODY_MAX); exit(1); }
        for (size_t j = 2; j < tok.size(); j++) {
            int code = -1;
            for (int b = 0; MACRO_BODY_OPS[b]; b++) if (tok[j] == MACRO_BODY_OPS[b]) { for (int o = 0; o < N_OPS; o++) if (!strcmp(OP_NAMES[o], MACRO_BODY_OPS[b])) code = o; }
            if (code < 0) { fprintf(stderr, "macro %s: %s is not a body operation\n", tok[0].c_str(), tok[j].c_str()); exit(1); }
            P.mbody[i * MACRO_BODY_MAX + P.mlen[i]] = (u8)code; P.mlen[i] += 1;
        }
        int v;
        if (uniform) { v = 32 + i; if (table[v] != NOP) { fprintf(stderr, "uniform codon %d is not free\n", v); exit(1); } }
        else {
            std::string cod = tok[1]; for (char &c : cod) c = (char)toupper((unsigned char)c);
            const char *donor_op = NULL;
            for (int d = 0; MACRO_DONORS[d][0]; d++) if (cod == MACRO_DONORS[d][0]) donor_op = MACRO_DONORS[d][1];
            if (!donor_op) { fprintf(stderr, "macro %s: codon %s is not a donor (CTA CTG GCA GCG GGA GGG ATA TGC)\n", tok[0].c_str(), cod.c_str()); exit(1); }
            v = codon_value(cod.c_str());
            int want = -1; for (int o = 0; o < N_OPS; o++) if (!strcmp(OP_NAMES[o], donor_op)) want = o;
            if (table[v] != want) { fprintf(stderr, "macro %s: codon %s is already taken\n", tok[0].c_str(), cod.c_str()); exit(1); }
        }
        table[v] = (u8)(N_OPS + i); codons.push_back((u64)v); P.nmac += 1;
    }
    fclose(f);
}

/* ---------------------------------------------------------------- host side */
static inline u64 fnv_bytes(u64 h, const u8 *p, u64 n) { for (u64 i = 0; i < n; i++) h = (h ^ p[i]) * 1099511628211ULL; return h; }
static inline u64 fnv_u64le(u64 h, u64 v) { u8 b[8]; for (int i = 0; i < 8; i++) b[i] = (u8)(v >> (8 * i)); return fnv_bytes(h, b, 8); }
static inline void put_u64le(FILE *f, u64 v) { u8 b[8]; for (int i = 0; i < 8; i++) b[i] = (u8)(v >> (8 * i)); fwrite(b, 1, 8, f); }

struct Host {
    Params P; World w; u64 nbytes; u8 table[64];
    std::vector<u8> fwd, rev, live; std::vector<Proc> proc; std::vector<u64> ghash; std::vector<i32> depth;
    u64 counters[5] = { 0, 0, 0, 0, 0 };     /* births, faithful, spontaneous, deaths, starved (docs/17 day 3) */
    u64 energy_init = 0;
    std::vector<u8> terrain;
    std::vector<int> map_modes; std::vector<FILE *> map_files; u8 *map_dev = NULL;   /* docs/17 slot-map modes */
    i64 depth_cap = 0;                       /* --depth-cap: 0 = the reference's uncapped depth_hist */
    FILE *top_f = NULL, *map_f = NULL;       /* --top-genes, --slot-map */
    std::vector<u8> map_frame;
    FILE *op_f = NULL; std::set<u64> op_named; u64 op_prev = 0;   /* --op-census */
};

/* the codons of slot `base` up to the first STOP, as letters and as operation names (one small device read) */
static void read_gene(Host &h, u64 base, std::string &codons, std::string &ops)
{
    i64 ncells = 3 * (h.P.maxk + 1);
    u64 b0 = base >> 2, b1 = (base + ncells - 1) >> 2, n = b1 - b0 + 1;
    std::vector<u8> f(n);
    CUDA_CHECK(cudaMemcpy(f.data(), h.w.fwd + b0, n, cudaMemcpyDeviceToHost));
    codons.clear(); ops.clear();
    for (i64 k = 0; k <= h.P.maxk; k++) {
        u32 c = codon_at(f.data(), base - b0 * 4, k);
        if (h.table[c] == STOP) break;
        if (k) { codons += ' '; ops += ' '; }
        static const char L[5] = "ACGT";
        codons += L[c >> 4]; codons += L[(c >> 2) & 3]; codons += L[c & 3];
        ops += OP_NAMES[h.table[c]];
    }
}

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
        const i64 f[17] = { p.slot, p.pc, p.H, p.P, p.Q, p.child, p.A, p.age, p.depth, p.copied, p.E, p.xs, p.sig, p.xin, p.open, p.streak, p.ldir };
        for (int i = 0; i < (h.P.ladder ? 17 : h.P.d6 ? 16 : h.P.d5 ? 12 : h.P.d1 ? 11 : 10); i++) x = fnv_u64le(x, (u64)f[i]);   /* docs/17: E day 1, xs day 5, sig..streak day 6 */
        if (h.P.nmac) { x = fnv_u64le(x, (u64)p.mk); x = fnv_u64le(x, (u64)p.mpc); }   /* section 19: after the last day-6 field */
    }
    for (int i = 0; i < (h.P.d3 ? 5 : 4); i++) x = fnv_u64le(x, h.counters[i]);   /* docs/17: starved on day 3 */
    if (h.P.ladder && h.P.nreg) {   /* section 18: the pools, region by region, task by task */
        std::vector<i64> pl((size_t)(h.P.nreg * LADDER_NT));
        CUDA_CHECK(cudaMemcpy(pl.data(), h.w.pools, pl.size() * sizeof(i64), cudaMemcpyDeviceToHost));
        for (i64 v : pl) x = fnv_u64le(x, (u64)v);
    }
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
        const i64 v[17] = { p.slot, p.pc, p.H, p.P, p.Q, p.child, p.A, p.age, p.depth, p.copied, p.E, p.xs, p.sig, p.xin, p.open, p.streak, p.ldir };
        for (int i = 0; i < (h.P.ladder ? 17 : h.P.d6 ? 16 : h.P.d5 ? 12 : h.P.d1 ? 11 : 10); i++) put_u64le(f, (u64)v[i]);
        if (h.P.nmac) { put_u64le(f, (u64)p.mk); put_u64le(f, (u64)p.mpc); }
    }
    for (int i = 0; i < (h.P.d3 ? 5 : 4); i++) put_u64le(f, h.counters[i]);
    fclose(f);
}

/* census (step 8) as one JSON object; since-last counters are relative to prev[] */
static std::string census_json(Host &h, i64 tick, const u64 prev[5], bool have_hash, u64 hash)
{
    if (h.P.d1 || h.P.d2) CUDA_CHECK(cudaMemset(h.w.cens_acc, 0, 15 * sizeof(u64)));
    k_census<<<(unsigned)((h.P.S + 255) / 256), 256>>>(h.w, h.P);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h.depth.data(), h.w.cens_depth, h.P.S * sizeof(i32), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h.ghash.data(), h.w.ghash, h.P.S * sizeof(u64), cudaMemcpyDeviceToHost));
    std::vector<u64> hs; std::map<i32, u64> dh; std::vector<i64> live_slots; i32 max_depth = -1;
    for (i64 s = 0; s < h.P.S; s++) if (h.depth[s] >= 0) {
        hs.push_back(h.ghash[s]); live_slots.push_back(s);
        i32 d = h.depth[s]; if (d > max_depth) max_depth = d;
        dh[h.depth_cap > 0 && d >= h.depth_cap ? (i32)h.depth_cap : d]++;
    }
    std::sort(hs.begin(), hs.end());
    std::vector<std::pair<u64, u64>> runs;   /* (count, hash) */
    for (size_t i = 0; i < hs.size();) { size_t j = i; while (j < hs.size() && hs[j] == hs[i]) j++; runs.push_back({ (u64)(j - i), hs[i] }); i = j; }
    std::sort(runs.begin(), runs.end(), [](const std::pair<u64, u64> &a, const std::pair<u64, u64> &b) {
        return a.first != b.first ? a.first > b.first : a.second < b.second; });
    if (h.top_f) {                           /* --top-genes: one carrier of each of the top 10 hashes */
        size_t ntop = std::min<size_t>(10, runs.size());
        std::vector<i64> carrier(ntop, -1); size_t found = 0;
        for (size_t i = 0; i < live_slots.size() && found < ntop; i++) {
            u64 g = h.ghash[live_slots[i]];
            for (size_t j = 0; j < ntop; j++) if (carrier[j] < 0 && runs[j].second == g) { carrier[j] = live_slots[i]; found++; break; }
        }
        for (size_t j = 0; j < ntop; j++) {
            std::string codons, ops; read_gene(h, (u64)carrier[j] * h.P.G, codons, ops);
            fprintf(h.top_f, "%lld\t%016llx\t%llu\t%lld\t%d\t%s\t%s\n", (long long)tick, (unsigned long long)runs[j].second,
                    (unsigned long long)runs[j].first, (long long)carrier[j], h.depth[carrier[j]], codons.c_str(), ops.c_str());
        }
        fflush(h.top_f);
    }
    if (h.map_f) {                           /* --slot-map: one byte per slot, one frame per census */
        for (i64 s = 0; s < h.P.S; s++) {
            i32 d = h.depth[s];
            h.map_frame[s] = d >= 0 ? (u8)(d >= 253 ? 254 : d + 1) : d == -2 ? 255 : 0;
        }
        fwrite(h.map_frame.data(), 1, h.P.S, h.map_f); fflush(h.map_f);
    }
    for (size_t m = 0; m < h.map_modes.size(); m++) {   /* docs/17 slot-map modes */
        k_slotmap<<<(unsigned)((h.P.S + 255) / 256), 256>>>(h.w, h.P, h.map_modes[m], h.map_dev);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(h.map_frame.data(), h.map_dev, h.P.S, cudaMemcpyDeviceToHost));
        fwrite(h.map_frame.data(), 1, h.P.S, h.map_files[m]); fflush(h.map_files[m]);
    }
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
    js += "]";
    if (h.depth_cap > 0) { snprintf(buf, sizeof buf, ", \"max_depth\": %d", max_depth); js += buf; }
    if (h.P.d1 || h.P.d2) {   /* docs/17 census additions, the reference's key order */
        u64 acc[15], led[13];
        CUDA_CHECK(cudaMemcpy(acc, h.w.cens_acc, sizeof acc, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(led, h.w.ledger, sizeof led, cudaMemcpyDeviceToHost));
        if (h.P.d1) {
            i64 lam = tick % h.P.day_len < h.P.light_len ? h.P.lmax : 0;
            if (h.P.d4) {   /* day 4: the light at column 0 of the middle row of each zone */
                u32 ch[4]; i64 l0 = sky(h.P, tick, (h.P.R / 4) * h.P.W, ch), l1 = sky(h.P, tick, (h.P.R / 2 + h.P.R / 4) * h.P.W, ch);
                snprintf(buf, sizeof buf, ", \"light\": [%lld, %lld]", (long long)l0, (long long)l1); js += buf;
            } else { snprintf(buf, sizeof buf, ", \"light\": %lld", (long long)lam); js += buf; }
            snprintf(buf, sizeof buf, ", \"energy_total\": %llu, \"energy_init\": %llu, \"ledger\": [",
                     (unsigned long long)acc[0], (unsigned long long)h.energy_init); js += buf;
            for (int i = 0; i < (h.P.d6 ? 13 : h.P.d5 ? 12 : 9); i++) { snprintf(buf, sizeof buf, "%s%llu", i ? ", " : "", (unsigned long long)led[i]); js += buf; }
            js += "]";
        }
        if (h.P.d2 && h.P.firm) { snprintf(buf, sizeof buf, ", \"zone_live\": [%llu, %llu]", (unsigned long long)acc[1], (unsigned long long)acc[2]); js += buf; }
        if (h.P.d3) { snprintf(buf, sizeof buf, ", \"starved_total\": %llu, \"live_land\": %llu, \"harvesters\": %llu", (unsigned long long)h.counters[4],
                               (unsigned long long)acc[3], (unsigned long long)acc[4]); js += buf; }
        if (h.P.d4) { snprintf(buf, sizeof buf, ", \"sensers\": %llu, \"sleepers\": %llu", (unsigned long long)acc[5], (unsigned long long)acc[6]); js += buf; }
        if (h.P.d5) { snprintf(buf, sizeof buf, ", \"eaters\": %llu, \"movers\": %llu, \"execers\": %llu, \"foreign\": %llu", (unsigned long long)acc[7],
                               (unsigned long long)acc[8], (unsigned long long)acc[9], (unsigned long long)acc[10]); js += buf; }
        if (h.P.d6) {
            int ntk = h.P.ladder ? LADDER_NT : 4;
            u64 tc[LADDER_NT]; CUDA_CHECK(cudaMemcpy(tc, h.w.task_count, ntk * sizeof(u64), cudaMemcpyDeviceToHost));
            snprintf(buf, sizeof buf, ", \"signalers\": %llu, \"listeners\": %llu, \"taskers\": %llu, \"tasks\": [",
                     (unsigned long long)acc[11], (unsigned long long)acc[12], (unsigned long long)acc[13]); js += buf;
            for (int k = 0; k < ntk; k++) { snprintf(buf, sizeof buf, "%s%llu", k ? ", " : "", (unsigned long long)tc[k]); js += buf; }
            js += "]";
            if (h.P.ladder) {   /* section 18: pool totals per task */
                std::vector<i64> pl((size_t)(h.P.nreg * LADDER_NT));
                if (h.P.nreg) CUDA_CHECK(cudaMemcpy(pl.data(), h.w.pools, pl.size() * sizeof(i64), cudaMemcpyDeviceToHost));
                js += ", \"pools\": [";
                for (int k = 0; k < LADDER_NT; k++) {
                    i64 sum = 0; for (i64 r = 0; r < h.P.nreg; r++) sum += pl[r * LADDER_NT + k];
                    snprintf(buf, sizeof buf, "%s%lld", k ? ", " : "", (long long)sum); js += buf;
                }
                js += "]";
            }
            if (h.P.nmac) {   /* section 19: live genes with a macro codon; macro codons fetched, per macro */
                u64 mx[MACRO_MAX]; CUDA_CHECK(cudaMemcpy(mx, h.w.macro_exec, h.P.nmac * sizeof(u64), cudaMemcpyDeviceToHost));
                snprintf(buf, sizeof buf, ", \"macro_users\": %llu, \"macro_exec\": [", (unsigned long long)acc[14]); js += buf;
                for (int k = 0; k < h.P.nmac; k++) { snprintf(buf, sizeof buf, "%s%llu", k ? ", " : "", (unsigned long long)mx[k]); js += buf; }
                js += "]";
            }
        }
    }
    js += ", \"hash\": ";
    if (have_hash) { snprintf(buf, sizeof buf, "\"%016llx\"", (unsigned long long)hash); js += buf; } else js += "null";
    js += "}";
    return js;
}

/* --op-census (docs/17 day 7): the histogram of operation sequences of the living, after census_json (which leaves
 * the live slots in h.depth); since-last counters are relative to prev[] as in the census */
static void op_census(Host &h, i64 tick, const u64 prev[5])
{
    i64 S = h.P.S;
    k_opcensus<<<(unsigned)((S + 255) / 256), 256>>>(h.w, h.P);
    CUDA_CHECK(cudaGetLastError());
    std::vector<u64> oh((size_t)S); std::vector<u8> fl((size_t)S); u64 opc;
    CUDA_CHECK(cudaMemcpy(oh.data(), h.w.ophash, S * sizeof(u64), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(fl.data(), h.w.opflag, S, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&opc, h.w.op_cnt, sizeof opc, cudaMemcpyDeviceToHost));
    struct Agg { u64 n = 0, z0 = 0, t[4] = { 0, 0, 0, 0 }; i64 carrier = -1; };
    std::map<u64, Agg> by;
    u64 live = 0, changed = 0;
    for (i64 s = 0; s < S; s++) if (h.depth[s] >= 0) {
        Agg &a = by[oh[s]]; a.n++; live++;
        if (a.carrier < 0) a.carrier = s;
        u8 f = fl[s];
        for (int k = 0; k < 4; k++) if (f & (1 << k)) a.t[k]++;
        if (f & 64) a.z0++;
        if (f & 128) changed++;
    }
    u64 births = h.counters[0] - prev[0], spont = h.counters[2] - prev[2];
    fprintf(h.op_f, "C %lld %llu %llu %llu %llu %llu %llu %llu %zu\n", (long long)tick, (unsigned long long)live, (unsigned long long)births,
            (unsigned long long)(births - spont), (unsigned long long)(opc - h.op_prev), (unsigned long long)spont,
            (unsigned long long)(h.counters[3] - prev[3]), (unsigned long long)changed, by.size());
    h.op_prev = opc;
    for (auto &kv : by) {
        if (!h.op_named.count(kv.first)) {
            std::string codons, ops; read_gene(h, (u64)kv.second.carrier * h.P.G, codons, ops);
            fprintf(h.op_f, "S %016llx %s\n", (unsigned long long)kv.first, ops.c_str());
            h.op_named.insert(kv.first);
        }
        const Agg &a = kv.second;
        fprintf(h.op_f, "H %016llx %llu %llu %llu %llu %llu %llu\n", (unsigned long long)kv.first, (unsigned long long)a.n, (unsigned long long)a.z0,
                (unsigned long long)a.t[0], (unsigned long long)a.t[1], (unsigned long long)a.t[2], (unsigned long long)a.t[3]);
    }
    fflush(h.op_f);
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

/* docs/17 section 14: checkpoint and resume (open-ended runs). The file holds everything that survives from one
 * tick to the next: both strands, the live flags, the processors, the claims, the counters, and the output-only
 * ledger, clade map and seeded energy. Header: "DNA3CKPT", version 1, N, G, S, seed, a flag word, the next tick. */
static const char CKPT_MAGIC[8] = { 'D', 'N', 'A', '3', 'C', 'K', 'P', 'T' };
/* flag bits 9 (ladder), 10 (macros) and 11 (in_fresh) may be added by a --fork (docs/17 sections 18 and 19) */
static u64 ckpt_flags(const Params &P, bool uniform) { return (u64)P.dna2 | (u64)P.d1 << 1 | (u64)P.d2 << 2 | (u64)P.d3 << 3 | (u64)P.d4 << 4 | (u64)P.shade << 5 | (u64)uniform << 6 | (u64)P.d5 << 7 | (u64)P.d6 << 8 | (u64)P.ladder << 9 | (u64)(P.nmac > 0) << 10 | (u64)P.in_fresh << 11; }
static const u64 CKPT_FORK_BITS = (1ULL << 9) | (1ULL << 10) | (1ULL << 11);
static void write_checkpoint(Host &h, const std::string &path, i64 t_next, bool uniform, const std::vector<u64> &macro_codons)
{
    const World &w = h.w; const Params &P = h.P; i64 S = P.S;
    std::vector<u8> buf((size_t)h.nbytes);
    std::string tmp = path + ".tmp";
    FILE *f = fopen(tmp.c_str(), "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", tmp.c_str()); return; }
    bool v5 = P.nmac > 0;   /* version 5 (section 19): 19 fields per processor (mk, mpc) and the macro table; version 4 otherwise */
    u64 hd[7] = { v5 ? 5ULL : 4ULL, P.N, (u64)P.G, (u64)S, P.seed, ckpt_flags(P, uniform), (u64)t_next };
    fwrite(CKPT_MAGIC, 1, 8, f); fwrite(hd, 8, 7, f);
    CUDA_CHECK(cudaMemcpy(buf.data(), w.fwd, h.nbytes, cudaMemcpyDeviceToHost)); fwrite(buf.data(), 1, h.nbytes, f);
    CUDA_CHECK(cudaMemcpy(buf.data(), w.rev, h.nbytes, cudaMemcpyDeviceToHost)); fwrite(buf.data(), 1, h.nbytes, f);
    std::vector<u8> live((size_t)S); CUDA_CHECK(cudaMemcpy(live.data(), w.live, S, cudaMemcpyDeviceToHost)); fwrite(live.data(), 1, S, f);
    std::vector<Proc> pr((size_t)S); CUDA_CHECK(cudaMemcpy(pr.data(), w.proc, S * sizeof(Proc), cudaMemcpyDeviceToHost));
    if (v5) fwrite(pr.data(), sizeof(Proc), S, f);
    else for (i64 i = 0; i < S; i++) fwrite(&pr[i], 8, 17, f);   /* the first 17 fields: the version-4 record */
    std::vector<i32> v32((size_t)S);
    CUDA_CHECK(cudaMemcpy(v32.data(), w.claimed_by, S * 4, cudaMemcpyDeviceToHost)); fwrite(v32.data(), 4, S, f);
    CUDA_CHECK(cudaMemcpy(v32.data(), w.clade, S * 4, cudaMemcpyDeviceToHost)); fwrite(v32.data(), 4, S, f);
    u64 led[13], tcs[LADDER_NT]; for (int k = 0; k < LADDER_NT; k++) tcs[k] = 0;
    CUDA_CHECK(cudaMemcpy(led, w.ledger, sizeof led, cudaMemcpyDeviceToHost));
    if (P.d6) CUDA_CHECK(cudaMemcpy(tcs, w.task_count, sizeof tcs, cudaMemcpyDeviceToHost));
    fwrite(h.counters, 8, 5, f); fwrite(led, 8, 13, f); fwrite(tcs, 8, LADDER_NT, f); fwrite(&h.energy_init, 8, 1, f);
    /* version 4 (docs/17 section 18): Proc with ldir, 19 task counts, then the pools */
    u64 npool = (u64)(P.nreg * LADDER_NT); fwrite(&npool, 8, 1, f);
    if (npool) { std::vector<i64> pl(npool); CUDA_CHECK(cudaMemcpy(pl.data(), w.pools, npool * sizeof(i64), cudaMemcpyDeviceToHost)); fwrite(pl.data(), 8, npool, f); }
    if (v5) {   /* the macro table: nmac, then per macro its codon value, body length and body operation codes */
        u64 nm = (u64)P.nmac; fwrite(&nm, 8, 1, f);
        for (int i = 0; i < P.nmac; i++) {
            u64 cv = macro_codons[i], ln = (u64)P.mlen[i]; fwrite(&cv, 8, 1, f); fwrite(&ln, 8, 1, f);
            for (int j = 0; j < P.mlen[i]; j++) { u64 o = P.mbody[i * MACRO_BODY_MAX + j]; fwrite(&o, 8, 1, f); }
        }
    }
    fclose(f);
    if (rename(tmp.c_str(), path.c_str()) != 0) fprintf(stderr, "cannot rename %s\n", tmp.c_str());
}
static i64 read_checkpoint(Host &h, const std::string &path, bool uniform, bool fork, const std::vector<u64> &macro_codons)
{
    World &w = h.w; const Params &P = h.P; i64 S = P.S;
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot read %s\n", path.c_str()); exit(1); }
    char mg[8]; u64 hd[7];
    if (fread(mg, 1, 8, f) != 8 || memcmp(mg, CKPT_MAGIC, 8) || fread(hd, 8, 7, f) != 7 || hd[0] < 3 || hd[0] > 5) {
        fprintf(stderr, "%s is not a version-3, 4 or 5 checkpoint\n", path.c_str()); exit(1); }
    /* --fork (docs/17 sections 18, 19): a checkpoint may be continued with --ladder, --macros and --in-fresh added */
    u64 want = ckpt_flags(P, uniform), added = want & ~hd[5];
    bool forking = fork && hd[5] != want && (hd[5] & ~want) == 0 && (added & ~CKPT_FORK_BITS) == 0;
    if (hd[1] != P.N || hd[2] != (u64)P.G || hd[3] != (u64)S || hd[4] != P.seed || (hd[5] != want && !forking)) {
        fprintf(stderr, "checkpoint %s was written with other N, G, seed or days\n", path.c_str()); exit(1); }
    bool fork_ladder = forking && (added & (1ULL << 9));
    int ver = (int)hd[0];
    auto rd = [&](void *p, size_t sz, size_t n) { if (fread(p, sz, n, f) != n) { fprintf(stderr, "short checkpoint %s\n", path.c_str()); exit(1); } };
    std::vector<u8> buf((size_t)h.nbytes);
    rd(buf.data(), 1, h.nbytes); CUDA_CHECK(cudaMemcpy(w.fwd, buf.data(), h.nbytes, cudaMemcpyHostToDevice));
    rd(buf.data(), 1, h.nbytes); CUDA_CHECK(cudaMemcpy(w.rev, buf.data(), h.nbytes, cudaMemcpyHostToDevice));
    std::vector<u8> live((size_t)S); rd(live.data(), 1, S); CUDA_CHECK(cudaMemcpy(w.live, live.data(), S, cudaMemcpyHostToDevice));
    std::vector<Proc> pr((size_t)S);
    if (ver == 5) rd(pr.data(), sizeof(Proc), S);
    else {   /* version 3: 16 fields per processor (no ldir); version 4: 17 (no mk, mpc) */
        int nf = ver == 4 ? 17 : 16;
        std::vector<i64> rr((size_t)S * nf); rd(rr.data(), 8, (size_t)S * nf);
        for (i64 i = 0; i < S; i++) { memcpy(&pr[i], &rr[i * nf], nf * 8); if (nf == 16) pr[i].ldir = 8; pr[i].mk = -1; pr[i].mpc = 0; }
    }
    if (fork_ladder) for (i64 i = 0; i < S; i++) { pr[i].xin = 16; pr[i].open = 0; pr[i].streak = 0; pr[i].ldir = 8; }
    CUDA_CHECK(cudaMemcpy(w.proc, pr.data(), S * sizeof(Proc), cudaMemcpyHostToDevice));
    std::vector<i32> v32((size_t)S);
    rd(v32.data(), 4, S); CUDA_CHECK(cudaMemcpy(w.claimed_by, v32.data(), S * 4, cudaMemcpyHostToDevice));
    rd(v32.data(), 4, S); CUDA_CHECK(cudaMemcpy(w.clade, v32.data(), S * 4, cudaMemcpyHostToDevice));
    u64 led[13], tcs[LADDER_NT]; for (int k = 0; k < LADDER_NT; k++) tcs[k] = 0;
    rd(h.counters, 8, 5); rd(led, 8, 13); rd(tcs, 8, ver >= 4 ? LADDER_NT : 4);
    if (fork_ladder) for (int k = 0; k < LADDER_NT; k++) tcs[k] = 0;
    if (P.d6) CUDA_CHECK(cudaMemcpy(w.task_count, tcs, sizeof tcs, cudaMemcpyHostToDevice)); rd(&h.energy_init, 8, 1);
    if (ver >= 4) {
        u64 npool; rd(&npool, 8, 1);
        if (npool && npool != (u64)(P.nreg * LADDER_NT)) { fprintf(stderr, "checkpoint %s has %llu pool values, this world needs %lld\n", path.c_str(), (unsigned long long)npool, (long long)(P.nreg * LADDER_NT)); exit(1); }
        if (npool) { std::vector<i64> pl(npool); rd(pl.data(), 8, npool); CUDA_CHECK(cudaMemcpy(w.pools, pl.data(), npool * sizeof(i64), cudaMemcpyHostToDevice)); }
    }
    if (ver >= 5) {   /* the stored macro table must be the one given with --macros */
        u64 nm; rd(&nm, 8, 1);
        bool same = nm == (u64)P.nmac;
        for (u64 i = 0; i < nm; i++) {
            u64 cv, ln; rd(&cv, 8, 1); rd(&ln, 8, 1);
            if (i < (u64)P.nmac && (cv != macro_codons[i] || ln != (u64)P.mlen[i])) same = false;
            for (u64 j = 0; j < ln; j++) { u64 o; rd(&o, 8, 1); if (i < (u64)P.nmac && j < (u64)MACRO_BODY_MAX && o != P.mbody[i * MACRO_BODY_MAX + j]) same = false; }
        }
        if (!same) { fprintf(stderr, "the checkpoint's macro table differs from --macros\n"); exit(1); }
    }
    CUDA_CHECK(cudaMemcpy(w.ledger, led, sizeof led, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(w.counters, h.counters, sizeof h.counters, cudaMemcpyHostToDevice));
    fclose(f);
    return (i64)hd[6];
}

static void usage()
{
    fprintf(stderr, "usage: dna_world --N cells --G slot --ticks n --seed s [--I 16 --T 64 --mu 0.001 --D auto --F auto\n"
                    "  --no-double --uniform-code --blank --genome \"ATG GCT ...\" --n-seed k --hash-every k --no-hash --hash-init\n"
                    "  --census-every k --births FILE --dump FILE --out FILE --gpu i\n"
                    "  --log-births FILE --log-filter divide|faithful|all --log-every k --log-codons --depth-cap K --top-genes FILE --slot-map FILE\n"
                    "  --max-seconds s --op-census FILE]\n");
    exit(1);
}

int main(int argc, char **argv)
{
    /* default N = 2,929,687 x 1024 = 2,999,999,488: the S of section 9 (3e9 itself is not a multiple of 1024;
     * section 13 "Default N"). D = F = -1 means "auto" (section 9). */
    u64 N = 2999999488ULL, seed = 1; i64 G = 1024, ticks = 100, I = 16, T = 64, D = -1, F = -1, n_seed = 0;
    bool dna2 = false;
    double mu = 0.001; bool dbl = true, uniform = false, blank = false, no_hash = false, hash_init = false; i64 hash_every = -2, census_every = 10;
    std::string genome, births_path, dump_path, out_path; int gpu = 0;
    std::string log_path, top_path, map_path; int log_filter = 0; i64 log_every = 1, depth_cap = 0;   /* filter: 0 divide, 1 faithful, 2 all */
    bool log_codons = false; double max_seconds = 0;
    /* docs/17 (DNA-3), the reference's defaults */
    bool day[7] = { false, false, false, false, false, false, false }; std::string seed_slots_arg, seed_terrain = "any", map_mode_arg = "depth";
    i64 day_len = 32, light_len = -1, lmax = 16, emax = 1024, eshift = 4, e_seed = 512, e_spont = 128, g_pass = 16, Wg = 0, firm = 1;
    i64 gp_sea = 8, gp_land = 0, gh_sea = 16, gh_land = 32, c_copy = 1, maint = 1, land_cell = 32, land_thr = 32768; bool inflow_land = false;
    i64 year_days = 16, moon_days = 8, season_amp = -1, lat_amp = 8, lmoon = 1; bool shade = false;
    i64 c_move = 2, c_fly = 8, eat_shift = 1, eat_eff = 8, c_walk = 4, task_bonus = 8;
    bool ladder = false, fork = false; i64 streak_len = 7, pool_cell = 32, pool_max = 4096, pool_in = 64;
    std::string macros_path; bool in_fresh = false;   /* docs/17 section 19 */
    std::string ckpt_path, resume_path; i64 ckpt_every = 0;
    std::string op_path;   /* --op-census (docs/17 day 7) */
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
        else if (a == "--dna2") dna2 = true;
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
        else if (a == "--log-births") log_path = next();
        else if (a == "--log-filter") { const char *v = next(); if (!strcmp(v, "divide")) log_filter = 0; else if (!strcmp(v, "faithful")) log_filter = 1;
                                        else if (!strcmp(v, "all")) log_filter = 2; else { fprintf(stderr, "--log-filter divide|faithful|all\n"); return 1; } }
        else if (a == "--log-every") { log_every = atoll(next()); if (log_every < 1) { fprintf(stderr, "--log-every must be >= 1\n"); return 1; } }
        else if (a == "--depth-cap") { depth_cap = atoll(next()); if (depth_cap < 0) { fprintf(stderr, "--depth-cap must be >= 0\n"); return 1; } }
        else if (a == "--log-codons") log_codons = true;
        else if (a == "--max-seconds") max_seconds = atof(next());
        else if (a == "--top-genes") top_path = next();
        else if (a == "--op-census") op_path = next();
        else if (a == "--slot-map") map_path = next();
        else if (a == "--dna3") day[1] = day[2] = day[3] = true;
        else if (a == "--days") { std::string v = next(); for (char c : v) { if (c >= '1' && c <= '6') day[c - '0'] = true;
                                  else if (c != ',' && c != ' ') { fprintf(stderr, "only days 1 to 6 are implemented\n"); return 1; } } }
        else if (a == "--year-days") year_days = atoll(next());
        else if (a == "--moon-days") moon_days = atoll(next());
        else if (a == "--season-amp") season_amp = atoll(next());
        else if (a == "--lat-amp") lat_amp = atoll(next());
        else if (a == "--lmoon") lmoon = atoll(next());
        else if (a == "--shade") shade = true;
        else if (a == "--c-move") c_move = atoll(next());
        else if (a == "--c-fly") c_fly = atoll(next());
        else if (a == "--eat-shift") eat_shift = atoll(next());
        else if (a == "--eat-eff") eat_eff = atoll(next());
        else if (a == "--c-walk") c_walk = atoll(next());
        else if (a == "--task-bonus") task_bonus = atoll(next());
        else if (a == "--ladder") ladder = true;
        else if (a == "--fork") fork = true;
        else if (a == "--streak-len") streak_len = atoll(next());
        else if (a == "--pool-cell") pool_cell = atoll(next());
        else if (a == "--pool-max") pool_max = atoll(next());
        else if (a == "--pool-in") pool_in = atoll(next());
        else if (a == "--macros") macros_path = next();
        else if (a == "--in-fresh") in_fresh = true;
        else if (a == "--checkpoint") ckpt_path = next();
        else if (a == "--checkpoint-every") ckpt_every = atoll(next());
        else if (a == "--resume") resume_path = next();
        else if (a == "--seed-slots") seed_slots_arg = next();
        else if (a == "--seed-terrain") { seed_terrain = next(); if (seed_terrain != "any" && seed_terrain != "sea" && seed_terrain != "land") usage(); }
        else if (a == "--slot-map-mode") map_mode_arg = next();
        else if (a == "--day-len") day_len = atoll(next());
        else if (a == "--light-len") light_len = atoll(next());
        else if (a == "--lmax") lmax = atoll(next());
        else if (a == "--emax") emax = atoll(next());
        else if (a == "--eshift") eshift = atoll(next());
        else if (a == "--e-seed") e_seed = atoll(next());
        else if (a == "--e-spont") e_spont = atoll(next());
        else if (a == "--g-pass") g_pass = atoll(next());
        else if (a == "--W") Wg = atoll(next());
        else if (a == "--firm") firm = atoll(next());
        else if (a == "--gp-sea") gp_sea = atoll(next());
        else if (a == "--gp-land") gp_land = atoll(next());
        else if (a == "--gh-sea") gh_sea = atoll(next());
        else if (a == "--gh-land") gh_land = atoll(next());
        else if (a == "--c-copy") c_copy = atoll(next());
        else if (a == "--maint") maint = atoll(next());
        else if (a == "--inflow-land") inflow_land = true;
        else if (a == "--land-cell") land_cell = atoll(next());
        else if (a == "--land-thr") land_thr = atoll(next());
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
    /* docs/17: every day implies --dna2; day 3 needs days 1 and 2; the reference's checks */
    bool d1 = day[1], d2 = day[2], d3 = day[3], d4 = day[4];
    if (d3 && !(d1 && d2)) { fprintf(stderr, "day 3 needs days 1 and 2\n"); return 1; }
    if (d4 && !(d1 && d2 && d3)) { fprintf(stderr, "day 4 needs days 1, 2 and 3\n"); return 1; }
    bool d5 = day[5];
    if (d5 && !d4) { fprintf(stderr, "day 5 needs days 1 to 4\n"); return 1; }
    if (std::min({ c_move, c_fly, eat_shift, eat_eff }) < 0 || eat_eff > 16) { fprintf(stderr, "bad day-5 parameters\n"); return 1; }
    bool d6 = day[6];
    if (d6 && !d5) { fprintf(stderr, "day 6 needs days 1 to 5\n"); return 1; }
    if (c_walk < 0 || task_bonus < 0) { fprintf(stderr, "bad day-6 parameters\n"); return 1; }
    if (ladder && !d6) { fprintf(stderr, "--ladder needs day 6\n"); return 1; }
    if ((!macros_path.empty() || in_fresh) && !d6) { fprintf(stderr, "--macros and --in-fresh need day 6\n"); return 1; }
    if (shade && !d3) { fprintf(stderr, "--shade is a day-3 option\n"); return 1; }
    if (d1 || d2 || d3) dna2 = true;
    if (light_len < 0) light_len = day_len / 2;
    if (season_amp < 0) season_amp = day_len / 4;
    if (d4 && (year_days < 1 || moon_days < 1 || season_amp > light_len || light_len + season_amp > day_len || lat_amp < 0 || lat_amp > 16 || lmoon < 0)) {
        fprintf(stderr, "bad day-4 parameters (need 0 <= season_amp <= light_len, light_len + season_amp <= day_len, 0 <= lat_amp <= 16)\n"); return 1; }
    if (d1) {
        if (day_len < 1 || light_len > day_len) { fprintf(stderr, "need day_len >= 1 and 0 <= light_len <= day_len\n"); return 1; }
        if (T < 0 || eshift < 0 || emax < 1 || e_seed < 0 || e_seed > emax || e_spont < 0 || e_spont > emax ||
            std::min({ lmax, g_pass, gp_sea, gp_land, gh_sea, gh_land, c_copy, maint }) < 0) { fprintf(stderr, "bad energy parameters\n"); return 1; }
    } else if (T < 1) { fprintf(stderr, "T must be at least 1 (T = 0, no age limit, exists only on day 1)\n"); return 1; }
    i64 Rg = 2;
    if (d2) {
        if (Wg < 3 || S % Wg || S / Wg < 3) { fprintf(stderr, "day 2 needs --W >= 3 dividing S with S / W >= 3 rows\n"); return 1; }
        Rg = S / Wg;
        if (firm < 0 || (firm && (Rg % 2 || firm > Rg / 2 - 1))) { fprintf(stderr, "the firmament needs an even number of rows R and 0 < firm <= R/2 - 1\n"); return 1; }
        if (d3 && (land_cell < 1 || Wg % land_cell || Rg % land_cell)) { fprintf(stderr, "--land-cell must divide W and R\n"); return 1; }
    } else { Wg = 1; firm = 0; }   /* without day 2: W = 1, R = 2 keep the census's zone arithmetic harmless */

    Host h;
    Params &P = h.P;
    P.seed = seed; P.N = N; P.G = G; P.S = S; P.I = I; P.T = T; P.maxk = G / 3 - 1; P.mu = mu; P.D = D; P.F = F;
    P.atomic_writes = (G % 4) != 0;
    P.dna2 = dna2 ? 1 : 0;
    P.d1 = d1; P.d2 = d2; P.d3 = d3; P.day_len = day_len; P.light_len = light_len; P.lmax = lmax; P.emax = emax; P.eshift = eshift;
    P.e_seed = e_seed; P.e_spont = e_spont; P.gp[0] = d3 ? gp_sea : g_pass; P.gp[1] = d3 ? gp_land : g_pass; P.gh[0] = gh_sea; P.gh[1] = gh_land;
    P.c_copy = c_copy; P.maint = maint; P.inflow_land = inflow_land ? 1 : 0; P.W = Wg; P.R = Rg; P.firm = firm;
    P.d6 = d6; P.c_walk = c_walk; P.task_bonus = task_bonus;
    P.ladder = ladder; P.streak_len = streak_len; P.pool_cell = pool_cell; P.pool_max = pool_max; P.pool_in = pool_in;
    if (ladder && (streak_len < 1 || streak_len > 7 || pool_cell < 0 || pool_max < 1 || pool_in < 0 || (pool_cell && (Wg % pool_cell || Rg % pool_cell)))) {
        fprintf(stderr, "bad ladder parameters (1 <= streak_len <= 7, pool_cell divides W and R)\n"); return 1; }
    P.rx = (ladder && pool_cell) ? Wg / pool_cell : 0; P.nreg = (ladder && pool_cell) ? (Wg / pool_cell) * (Rg / pool_cell) : 0;
    P.d5 = d5; P.c_move = c_move; P.c_fly = c_fly; P.eat_shift = eat_shift; P.eat_eff = eat_eff;
    P.d4 = d4; P.shade = shade; P.year_days = year_days; P.moon_days = moon_days; P.season_amp = season_amp; P.lat_amp = lat_amp; P.lmoon = lmoon;
    if (d2) make_terrain(h.terrain, seed, S, Wg, firm, d3, land_cell, land_thr); else h.terrain.assign((size_t)S, 0);
    u8 *table = h.table; build_table(table, uniform, dna2, d3, d4, d5, d6);
    P.in_fresh = in_fresh ? 1 : 0; P.nmac = 0; for (int k = 0; k < MACRO_MAX; k++) P.mlen[k] = 0; memset(P.mbody, 0, sizeof P.mbody);
    std::vector<u64> macro_codons;   /* docs/17 section 19: the macros' codon values (for the checkpoint) */
    if (!macros_path.empty()) load_macros(P, table, macros_path, uniform, macro_codons);
    h.depth_cap = depth_cap;
    P.opc = op_path.empty() ? 0 : 1;

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
    dalloc((void **)&w.div_copied, S * sizeof(i32));
    dalloc((void **)&w.ghash, S * sizeof(u64)); dalloc((void **)&w.cens_depth, S * sizeof(i32));
    dalloc((void **)&w.counters, 5 * sizeof(u64)); dalloc((void **)&w.table, 64);
    dalloc((void **)&w.terrain, S); dalloc((void **)&w.div_energy, S * sizeof(i64)); dalloc((void **)&w.ledger, 13 * sizeof(u64));
    if (d6) { dalloc((void **)&w.sig_old, S); dalloc((void **)&w.task_count, LADDER_NT * sizeof(u64)); dalloc((void **)&w.macro_exec, MACRO_MAX * sizeof(u64)); }
    if (ladder) {
        dalloc((void **)&w.xin_old, S);
        if (P.nreg) {
            dalloc((void **)&w.pools, P.nreg * LADDER_NT * sizeof(i64)); dalloc((void **)&w.pool_used, P.nreg * LADDER_NT * sizeof(u64));
            std::vector<i64> pl((size_t)(P.nreg * LADDER_NT), pool_max);
            CUDA_CHECK(cudaMemcpy(w.pools, pl.data(), pl.size() * sizeof(i64), cudaMemcpyHostToDevice));
        }
    }
    if (d5) {
        dalloc((void **)&w.eat_target, S * 4); dalloc((void **)&w.move_op, S * 4); dalloc((void **)&w.move_to, S * 4);
        dalloc((void **)&w.e_loss, S * 8); dalloc((void **)&w.e_gain, S * 8); dalloc((void **)&w.move_cost, S * 8);
        CUDA_CHECK(cudaMemset(w.eat_target, 0xFF, S * 4)); CUDA_CHECK(cudaMemset(w.move_op, 0xFF, S * 4)); CUDA_CHECK(cudaMemset(w.move_to, 0xFF, S * 4));
    }
    dalloc((void **)&w.clade, S * sizeof(i32)); dalloc((void **)&w.cens_acc, 14 * sizeof(u64));
    if (P.opc) { dalloc((void **)&w.ophash, S * sizeof(u64)); dalloc((void **)&w.opbirth, S * sizeof(u64)); dalloc((void **)&w.opflag, S);
                 dalloc((void **)&w.op_cnt, sizeof(u64)); }
    if (d2) { dalloc((void **)&w.claim64, S * sizeof(u64)); dalloc((void **)&w.alloc_cand, S * sizeof(i32)); CUDA_CHECK(cudaMemset(w.claim64, 0xFF, S * sizeof(u64))); }
    CUDA_CHECK(cudaMemcpy(w.terrain, h.terrain.data(), S, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(w.claimed_by, 0xFF, S * sizeof(i32)));
    CUDA_CHECK(cudaMemset(w.div_mark, 0xFF, S * sizeof(i32)));
    CUDA_CHECK(cudaMemset(w.refill_k, 0xFF, S * sizeof(i32)));
    CUDA_CHECK(cudaMemset(w.claim, 0xFF, S * sizeof(i32)));   /* sentinel 0xFFFFFFFF: above every slot index */
    CUDA_CHECK(cudaMemcpy(w.table, table, 64, cudaMemcpyHostToDevice));
    FILE *births_f = NULL, *log_f = NULL; std::vector<BirthRec> log_host;
    if (!births_path.empty() || !log_path.empty()) {
        w.log_cap = (u32)std::min<i64>(S + F, (1LL << 31) - 1);
        dalloc((void **)&w.log, (size_t)w.log_cap * sizeof(BirthRec)); dalloc((void **)&w.log_n, sizeof(u32));
        log_host.resize(w.log_cap);
    }
    if (!births_path.empty()) {
        births_f = fopen(births_path.c_str(), "w");
        if (!births_f) { fprintf(stderr, "cannot write %s\n", births_path.c_str()); return 1; }
        fprintf(births_f, "#tick\tparent\tchild\tparent_hash\tchild_hash\tfaithful\n");
    }
    if (!log_path.empty()) {
        log_f = fopen(log_path.c_str(), "w");
        if (!log_f) { fprintf(stderr, "cannot write %s\n", log_path.c_str()); return 1; }
        fprintf(log_f, "#tick\tparent\tchild\tparent_hash\tchild_hash\tfaithful\tdepth\tparent_age\tcopied\tparent_P%s\n",
                log_codons ? "\tparent_codons\tchild_codons" : "");
    }
    if (!top_path.empty()) { h.top_f = fopen(top_path.c_str(), resume_path.empty() ? "w" : "a"); if (!h.top_f) { fprintf(stderr, "cannot write %s\n", top_path.c_str()); return 1; }
                             if (resume_path.empty()) fprintf(h.top_f, "#tick\thash\tcount\tslot\tdepth\tcodons\tops\n"); }
    if (P.opc) { h.op_f = fopen(op_path.c_str(), resume_path.empty() ? "w" : "a"); if (!h.op_f) { fprintf(stderr, "cannot write %s\n", op_path.c_str()); return 1; }
                 if (resume_path.empty()) fprintf(h.op_f, "# op census (docs/17 day 7): C tick live births divide_births op_faithful spontaneous deaths changed n_seq / S ophash ops / H ophash count zone0 t0 t1 t2 t3\n"); }
    if (!map_path.empty()) {
        /* docs/17: --slot-map-mode is a comma list of depth (the docs/12 map, written to FILE itself), gene, clade,
         * energy, plant (written to FILE.<mode>); with day 2 the terrain goes to FILE.terrain once */
        std::string m = map_mode_arg + ",";
        for (size_t i = 0, j; (j = m.find(',', i)) != std::string::npos; i = j + 1) {
            std::string mm = m.substr(i, j - i); if (mm.empty()) continue;
            if (mm == "depth") { h.map_f = fopen(map_path.c_str(), resume_path.empty() ? "wb" : "ab"); if (!h.map_f) { fprintf(stderr, "cannot write %s\n", map_path.c_str()); return 1; } continue; }
            int code = mm == "gene" ? 1 : mm == "clade" ? 2 : mm == "energy" ? 3 : mm == "plant" ? 4 : mm == "traits" ? 5 : mm == "animals" ? 6 : -1;
            if (code < 0) { fprintf(stderr, "--slot-map-mode: depth, gene, clade, energy, plant, traits, animals\n"); return 1; }
            std::string fn = map_path + "." + mm; FILE *f = fopen(fn.c_str(), resume_path.empty() ? "wb" : "ab");   /* resumed runs append */ if (!f) { fprintf(stderr, "cannot write %s\n", fn.c_str()); return 1; }
            h.map_modes.push_back(code); h.map_files.push_back(f);
        }
        h.map_frame.resize(S);
        if (!h.map_modes.empty()) dalloc((void **)&h.map_dev, S);
        if (d2) { std::string fn = map_path + ".terrain"; FILE *f = fopen(fn.c_str(), "wb"); if (f) { fwrite(h.terrain.data(), 1, S, f); fclose(f); } }
    }
    h.fwd.resize(h.nbytes); h.rev.resize(h.nbytes); h.live.resize(S); h.proc.resize(S); h.ghash.resize(S); h.depth.resize(S);

    fprintf(stderr, "dna_world N=%llu G=%lld S=%lld ticks=%lld seed=%llu I=%lld T=%lld mu=%g D=%lld F=%lld double=%d code=%s blank=%d "
                    "n_seed=%lld hash_every=%lld census_every=%lld gpu=%d\n", (unsigned long long)N, (long long)G, (long long)S,
            (long long)ticks, (unsigned long long)seed, (long long)I, (long long)T, mu, (long long)D, (long long)F, dbl ? 1 : 0,
            uniform ? (dna2 ? "uniform-dna2" : "uniform") : (dna2 ? "standard-dna2" : "standard"), blank ? 1 : 0, (long long)n_seed, (long long)hash_every, (long long)census_every, gpu);
    if (d1 || d2) {
        i64 nland = 0, nfirm = 0; for (i64 i = 0; i < S; i++) { nland += h.terrain[i] == 1; nfirm += h.terrain[i] == 2; }
        fprintf(stderr, "dna3 days=%s%s%s day_len=%lld light_len=%lld lmax=%lld emax=%lld eshift=%lld e_seed=%lld e_spont=%lld gp=%lld,%lld gh=%lld,%lld c_copy=%lld maint=%lld "
                        "inflow_land=%d W=%lld R=%lld firm=%lld land=%lld firmament=%lld\n", d1 ? "1" : "", d2 ? "2" : "", d3 ? (d4 ? "34" : "3") : "", (long long)day_len, (long long)light_len,
                (long long)lmax, (long long)emax, (long long)eshift, (long long)e_seed, (long long)e_spont, (long long)P.gp[0], (long long)P.gp[1], (long long)gh_sea, (long long)gh_land,
                (long long)c_copy, (long long)maint, inflow_land ? 1 : 0, (long long)Wg, (long long)Rg, (long long)firm, (long long)nland, (long long)nfirm);
    }
    if (d6) fprintf(stderr, "day6=1 c_walk=%lld task_bonus=%lld in_fresh=%d macros=%d\n", (long long)c_walk, (long long)task_bonus, P.in_fresh, P.nmac);
    for (int i = 0; i < P.nmac; i++) {   /* section 19: the macro table as loaded */
        fprintf(stderr, "macro M%d codon %llu body", i, (unsigned long long)macro_codons[i]);
        for (int j = 0; j < P.mlen[i]; j++) fprintf(stderr, " %s", OP_NAMES[P.mbody[i * MACRO_BODY_MAX + j]]);
        fputc('\n', stderr);
    }
    if (d5) fprintf(stderr, "day5=1 c_move=%lld c_fly=%lld eat_shift=%lld eat_eff=%lld\n", (long long)c_move, (long long)c_fly, (long long)eat_shift, (long long)eat_eff);
    if (d4 || shade)
        fprintf(stderr, "day4=%d year_days=%lld moon_days=%lld season_amp=%lld lat_amp=%lld lmoon=%lld shade=%d\n", d4 ? 1 : 0, (long long)year_days,
                (long long)moon_days, (long long)season_amp, (long long)lat_amp, (long long)lmoon, shade ? 1 : 0);
    if (log_f || h.top_f || h.map_f || depth_cap)
        fprintf(stderr, "log_births=%s log_filter=%s log_every=%lld depth_cap=%lld top_genes=%s slot_map=%s\n", log_path.c_str(),
                log_filter == 0 ? "divide" : log_filter == 1 ? "faithful" : "all", (long long)log_every, (long long)depth_cap, top_path.c_str(), map_path.c_str());

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
        /* docs/17: explicit --seed-slots, or k * (S / n_seed) advanced (with day 2) past firmament, wrong terrain and seeded slots */
        std::vector<i64> chosen; std::vector<u8> used((size_t)S, 0);
        std::vector<i64> explicit_slots;
        if (!seed_slots_arg.empty()) {
            char *b2 = strdup(seed_slots_arg.c_str());
            for (char *tok = strtok(b2, " ,"); tok; tok = strtok(NULL, " ,")) explicit_slots.push_back(atoll(tok));
            free(b2);
            if ((i64)explicit_slots.size() != n_seed) { fprintf(stderr, "--seed-slots needs n_seed distinct slots in [0, S)\n"); return 1; }
        }
        for (i64 k = 0; k < n_seed; k++) {
            i64 s;
            if (!explicit_slots.empty()) {
                s = explicit_slots[k];
                if (s < 0 || s >= S || used[s]) { fprintf(stderr, "--seed-slots needs n_seed distinct slots in [0, S)\n"); return 1; }
                if (d2 && h.terrain[s] == 2) { fprintf(stderr, "seed slot %lld lies in the firmament\n", (long long)s); return 1; }
            } else {
                s = k * (S / n_seed);
                if (d2) {
                    i64 n_try = 0;
                    auto ok = [&](i64 x) { u8 tr = h.terrain[x]; if (tr == 2 || used[x]) return false;
                                           return seed_terrain == "any" || (seed_terrain == "sea" && tr == 0) || (seed_terrain == "land" && tr == 1); };
                    while (!ok(s)) { s = (s + 1) % S; if (++n_try > S) { fprintf(stderr, "no slot for seed %lld\n", (long long)k); return 1; } }
                }
            }
            used[s] = 1; chosen.push_back(s);
        }
        for (i64 k = 0; k < n_seed; k++) {
            i64 s = chosen[k]; u64 base = (u64)s * G;
            write_slot(h, base, vals, 3 * (P.maxk + 1));   /* the G mod 3 trailing cells keep their soup content */
            Proc q; q.slot = s; q.pc = 1; q.H = (i64)base; q.P = (i64)base; q.Q = 0; q.child = -1; q.A = 0; q.age = 0; q.depth = 0; q.copied = 0;
            q.E = d1 ? e_seed : 0; q.xs = s; q.sig = q.xin = q.open = q.streak = 0; q.ldir = 8; q.mk = -1; q.mpc = 0; if (ladder) q.xin = 16; if (d1) h.energy_init += (u64)e_seed;
            i32 cl = (i32)s; CUDA_CHECK(cudaMemcpy(w.clade + s, &cl, sizeof cl, cudaMemcpyHostToDevice));
            u8 one = 1;
            CUDA_CHECK(cudaMemcpy(w.proc + s, &q, sizeof q, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(w.live + s, &one, 1, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(w.ghash + s, &gh, sizeof gh, cudaMemcpyHostToDevice));
        }
        fprintf(stderr, "seeded %lld slots with %zu codons, gene hash %016llx, expressible=%d\n", (long long)n_seed, codons.size(),
                (unsigned long long)gh, expressible(packed.data(), 0, table, P.maxk) ? 1 : 0);
    }
    i64 t_first = 0;
    if (!resume_path.empty()) {   /* docs/17 section 14: continue a run from its checkpoint */
        if (n_seed > 0) { fprintf(stderr, "--resume replaces the initial state; drop --n-seed\n"); return 1; }
        t_first = read_checkpoint(h, resume_path, uniform, fork, macro_codons);
        if (t_first >= ticks) { fprintf(stderr, "checkpoint is at tick %lld, nothing to do for --ticks %lld\n", (long long)t_first, (long long)ticks); return 1; }
        fprintf(stderr, "resumed from %s at tick %lld\n", resume_path.c_str(), (long long)t_first);
    }
    if (P.opc) { k_opbirth_init<<<(unsigned)((S + 255) / 256), 256>>>(w, P); CUDA_CHECK(cudaGetLastError()); }
    if (ckpt_every > 0 && census_every > 0 && ckpt_every % census_every) { fprintf(stderr, "--checkpoint-every must be a multiple of --census-every\n"); return 1; }
    CUDA_CHECK(cudaDeviceSynchronize());
    size_t mem_free1; CUDA_CHECK(cudaMemGetInfo(&mem_free1, &mem_total));

    const unsigned sgrid = (unsigned)((S + 255) / 256), dgrid = (unsigned)((D + 255) / 256), fgrid = (unsigned)((F + 255) / 256);
    u64 prev[5] = { 0, 0, 0, 0, 0 };
    memcpy(prev, h.counters, sizeof prev);   /* a resumed run's census continues from the checkpoint's counters */
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
            if (h.op_f) op_census(h, t, prev);
            memcpy(prev, h.counters, sizeof prev);
            if (t == ticks - 1 && !out_path.empty()) { FILE *f = fopen(out_path.c_str(), "w"); if (f) { fprintf(f, "%s\n", js.c_str()); fclose(f); } else fprintf(stderr, "cannot write %s\n", out_path.c_str()); }
        }
        fflush(stdout);
    };
    if (hash_init && hash_every >= 0) printf("tick -1 hash %016llx\n", (unsigned long long)state_hash(h));
    fflush(stdout);

    auto t_loop = std::chrono::steady_clock::now();
    for (i64 t = t_first; t < ticks; t++) {
        CUDA_CHECK(cudaMemcpyAsync(w.fwd_old, w.fwd, h.nbytes, cudaMemcpyDeviceToDevice));   /* snapshot for reads */
        i64 lam = d1 ? (t % day_len < light_len ? lmax : 0) : 0;                           /* docs/17 day 1 */
        if (d6) k_sig_snapshot<<<sgrid, 256>>>(w, P);                                       /* day 6: signals of the tick start */
        k_execute<<<sgrid, 256>>>(w, P, t, lam);                                            /* 1 */
        if (ladder && P.nreg) k_pools<<<(unsigned)((P.nreg * LADDER_NT + 255) / 256), 256>>>(w, P);   /* 1b, section 18 */
        if (d5) { k_eat<<<sgrid, 256>>>(w, P, t); k_eat_apply<<<sgrid, 256>>>(w, P); }    /* 2a, day 5 */
        k_alloc_propose<<<sgrid, 256>>>(w, P, t);                                           /* 2 */
        k_alloc_grant<<<sgrid, 256>>>(w, P, t);
        CUDA_CHECK(cudaMemsetAsync(w.claim, 0xFF, S * sizeof(i32)));
        if (d2) CUDA_CHECK(cudaMemsetAsync(w.claim64, 0xFF, S * sizeof(u64)));
        if (d5) k_move<<<sgrid, 256>>>(w, P);                                               /* 2c, day 5 */
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
        if (w.log) {
            u32 n; CUDA_CHECK(cudaMemcpy(&n, w.log_n, sizeof n, cudaMemcpyDeviceToHost));
            if (n > w.log_cap) { fprintf(stderr, "births log overflow at tick %lld (%u > %u)\n", (long long)t, n, w.log_cap); n = w.log_cap; }
            bool want_log = log_f && t % log_every == 0;
            if (n && (births_f || want_log)) {
                CUDA_CHECK(cudaMemcpy(log_host.data(), w.log, (size_t)n * sizeof(BirthRec), cudaMemcpyDeviceToHost));
                /* reference order: DIVIDE births by parent slot, then spontaneous births by slot */
                std::sort(log_host.begin(), log_host.begin() + n, [](const BirthRec &a, const BirthRec &b) {
                    bool sa = a.parent < 0, sb = b.parent < 0;
                    if (sa != sb) return sb;
                    return sa ? a.child < b.child : a.parent < b.parent; });
                for (u32 i = 0; i < n; i++) {
                    const BirthRec &r = log_host[i];
                    char ph[20]; if (r.parent >= 0) snprintf(ph, sizeof ph, "%016llx", (unsigned long long)r.phash); else strcpy(ph, "-");
                    int faithful = r.parent >= 0 && r.phash == r.chash ? 1 : 0;
                    if (births_f)
                        fprintf(births_f, "%lld\t%d\t%d\t%s\t%016llx\t%d\n", (long long)t, r.parent, r.child, ph, (unsigned long long)r.chash, faithful);
                    if (want_log && (log_filter == 2 || (r.parent >= 0 && (log_filter == 0 || faithful)))) {
                        fprintf(log_f, "%lld\t%d\t%d\t%s\t%016llx\t%d\t%d\t%d\t%d\t%lld", (long long)t, r.parent, r.child, ph, (unsigned long long)r.chash,
                                faithful, r.depth, r.page, r.copied, (long long)r.pP);
                        if (log_codons) {
                            std::string pc, po, cc, co;
                            if (r.parent >= 0) read_gene(h, (u64)r.parent * G, pc, po); else pc = "-";
                            read_gene(h, (u64)r.child * G, cc, co);
                            fprintf(log_f, "\t%s\t%s", pc.c_str(), cc.c_str());
                        }
                        fputc('\n', log_f);
                    }
                }
            }
            if (n) CUDA_CHECK(cudaMemset(w.log_n, 0, sizeof(u32)));
        }
        if (max_seconds > 0 && t % 100 == 99 && t + 1 < ticks &&
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t_loop).count() >= max_seconds) {
            fprintf(stderr, "max-seconds reached: stopping after tick %lld\n", (long long)t);
            ticks = t + 1;                                    /* this tick becomes the last one */
        }
        report(t);
        if (!ckpt_path.empty() && ((ckpt_every > 0 && (t + 1) % ckpt_every == 0) || t == ticks - 1)) write_checkpoint(h, ckpt_path, t + 1, uniform, macro_codons);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_loop).count();
    double secs_total = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    if (!dump_path.empty()) dump_state(h, dump_path.c_str());
    if (births_f) fclose(births_f);
    if (log_f) fclose(log_f);
    if (h.top_f) fclose(h.top_f);
    if (h.op_f) fclose(h.op_f);
    if (h.map_f) fclose(h.map_f);
    for (FILE *f : h.map_files) fclose(f);
    i64 live_n = 0; CUDA_CHECK(cudaMemcpy(h.live.data(), w.live, S, cudaMemcpyDeviceToHost)); for (i64 s = 0; s < S; s++) live_n += h.live[s];
    fprintf(stderr, "summary ticks=%lld seconds=%.3f ticks_per_second=%.2f seconds_total=%.3f device_mem_alloc_bytes=%zu "
                    "device_mem_used_bytes=%zu live=%lld births=%llu faithful=%llu spontaneous=%llu deaths=%llu\n",
            (long long)ticks, secs, ticks > 0 && secs > 0 ? ticks / secs : 0.0, secs_total, allocated, (size_t)(mem_free0 - mem_free1),
            (long long)live_n, (unsigned long long)h.counters[0], (unsigned long long)h.counters[1], (unsigned long long)h.counters[2],
            (unsigned long long)h.counters[3]);
    return 0;
}
