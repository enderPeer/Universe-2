/* Universe 2 table search: score every four-instruction table of one-symbol (2-bit) operations.
 *
 * Same catalog and semantics as tools/u2_ops.py (36 operations, ring of 256 cells of W bits, head 0, A = 0,
 * PC wraps at the program length, fixed step budget, HALT stops). For each table in [start, start+count) of the
 * C(36,4) = 58,905 four-element subsets (combinatorial order), every program of 1..L symbols is run and the
 * distinct final tapes are counted (canonical: first nonzero cell + values to the last nonzero cell), as
 * Universe-7 scored its drawing laws.
 *
 *   u2_tables W L steps start count threads        -> one TSV line per table on stdout
 *   u2_tables --check                              -> reads "W steps table_id program" lines, prints final states
 *   u2_tables --list                               -> the catalog
 *   u2_tables prim PRIMS W L steps OPS start count threads -> arithmetic primitives (COPY, ADD, MUL_T, MAC_T,
 *       MAC_B, CARRY) of one machine over all programs of exactly L symbols in an index range (docs/13)
 *   u2_tables primany PRIMS W L steps OPS start count threads -> the same primitives at every step budget
 *       1..steps at once (one candidate mask per step; docs/14)
 *
 * TSV columns: table_id, op0..op3 (names), union_after_L1 .. union_after_L<L> (distinct tapes over all programs up
 * to that length), halted, nonblank, checksum (sum of all program tape hashes, order-independent, for cross-node
 * agreement). stderr: "tables=N programs_per_table=P secs=T host=H".
 *
 * gcc -O3 -march=native -fopenmp -std=c11 -Wall -Wextra u2_tables.c -o u2_tables
 */
#define _POSIX_C_SOURCE 200112L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <omp.h>

#define NOPS 41
#define NBASE 36   /* table ids rank 4-subsets of the first 36 operations only; the biological operations are used by name */
#ifndef NCELL
#define NCELL 256   /* ring size; -DNCELL=512 builds the 512-cell variant (a power of two) */
#endif
#define RMASK (NCELL - 1)
#define NW (NCELL / 64)   /* 64-bit words of a cell mask */
static inline uint64_t orw(const uint64_t *m) { uint64_t r = 0; for (int k = 0; k < NW; k++) r |= m[k]; return r; }   /* OR of a cell mask */
#define MAXL 16   /* prim mode runs 16-symbol programs; the table modes stay at L <= 12 */
static const char *NAMES[NOPS] = {
    "R1", "L1", "R2", "L2", "R3", "L3", "INC", "DEC", "FLIP", "NOT", "ZERO", "SET", "SHL",
    "INC_R1", "INC_L1", "FLIP_R1", "FLIP_L1", "FLIP_R2", "FLIP_R3", "SET_R1", "SET_R3", "ZERO_R1", "ZERO_R3",
    "SKZ", "SKNZ", "JZS", "JNZS", "JNZB", "OPEN", "CLOSE", "HALT", "NOP", "LDA", "STA", "SWAP", "ADDA",
    "EAT", "MATE_A", "MATE_B", "SLEEP", "WASTE"};
enum { R1, L1, R2, L2, R3, L3, INC, DEC, FLIP, NOT, ZERO, SET, SHL, INC_R1, INC_L1, FLIP_R1, FLIP_L1, FLIP_R2,
       FLIP_R3, SET_R1, SET_R3, ZERO_R1, ZERO_R3, SKZ, SKNZ, JZS, JNZS, JNZB, OPEN, CLOSE, HALT, NOP, LDA, STA,
       SWAP, ADDA, EAT, MATE_A, MATE_B, SLEEP, WASTE };

typedef struct { uint8_t tape[NCELL]; int head, pc, A, halted, steps; } State;

static void run(const int *table, const uint8_t *prog, int L, int W, int steps, State *st, const uint8_t *init, int ninit)
{
    uint8_t mask = (uint8_t)((1 << W) - 1);
    uint8_t *tape = st->tape;
    memset(tape, 0, NCELL);
    for (int i = 0; i < ninit; i++) tape[i] = init[i];
    int head = 0, pc = 0, A = 0, halted = 0, done = 0;
    int match[MAXL];
    int need_match = 0;
    for (int i = 0; i < 4; i++) need_match |= table[i] == OPEN || table[i] == CLOSE;
    if (need_match) {
        int stack[MAXL], sp = 0;
        for (int i = 0; i < L; i++) match[i] = -1;
        for (int i = 0; i < L; i++) {
            int op = table[prog[i]];
            if (op == OPEN) stack[sp++] = i;
            else if (op == CLOSE && sp) { int j = stack[--sp]; match[i] = j; match[j] = i; }
        }
    }
    for (int step = 1; step <= steps; step++) {
        int at = pc;
        int op = table[prog[pc]];
        pc = (pc + 1) % L;
        uint8_t v = tape[head];
        switch (op) {
        case R1: head = (head + 1) & RMASK; break;
        case L1: head = (head - 1) & RMASK; break;
        case R2: head = (head + 2) & RMASK; break;
        case L2: head = (head - 2) & RMASK; break;
        case R3: head = (head + 3) & RMASK; break;
        case L3: head = (head - 3) & RMASK; break;
        case INC: tape[head] = (uint8_t)((v + 1) & mask); break;
        case DEC: tape[head] = (uint8_t)((v - 1) & mask); break;
        case FLIP: tape[head] = v ^ 1; break;
        case NOT: tape[head] = v ^ mask; break;
        case ZERO: tape[head] = 0; break;
        case SET: tape[head] = mask; break;
        case SHL: tape[head] = (uint8_t)((v << 1) & mask); break;
        case INC_R1: tape[head] = (uint8_t)((v + 1) & mask); head = (head + 1) & RMASK; break;
        case INC_L1: tape[head] = (uint8_t)((v + 1) & mask); head = (head - 1) & RMASK; break;
        case FLIP_R1: tape[head] = v ^ 1; head = (head + 1) & RMASK; break;
        case FLIP_L1: tape[head] = v ^ 1; head = (head - 1) & RMASK; break;
        case FLIP_R2: tape[head] = v ^ 1; head = (head + 2) & RMASK; break;
        case FLIP_R3: tape[head] = v ^ 1; head = (head + 3) & RMASK; break;
        case SET_R1: tape[head] = mask; head = (head + 1) & RMASK; break;
        case SET_R3: tape[head] = mask; head = (head + 3) & RMASK; break;
        case ZERO_R1: tape[head] = 0; head = (head + 1) & RMASK; break;
        case ZERO_R3: tape[head] = 0; head = (head + 3) & RMASK; break;
        case SKZ: if (v == 0) pc = (pc + 1) % L; break;
        case SKNZ: if (v != 0) pc = (pc + 1) % L; break;
        case JZS: if (v == 0) pc = 0; break;
        case JNZS: if (v != 0) pc = 0; break;
        case JNZB: if (v != 0) pc = ((pc - 2) % L + L) % L; break;
        case OPEN: if (v == 0) pc = match[at] >= 0 ? (match[at] + 1) % L : 0; break;
        case CLOSE: if (v != 0) pc = match[at] >= 0 ? (match[at] + 1) % L : 0; break;
        case HALT: halted = 1; break;
        case NOP: break;
        case LDA: A = v; break;
        case STA: tape[head] = (uint8_t)A; break;
        case SWAP: tape[head] = (uint8_t)A; A = v; break;
        case ADDA: tape[head] = (uint8_t)((v + A) & mask); break;
        case EAT: A = v; tape[head] = 0; break;                                            /* eat: cell into the stomach, cell emptied */
        case MATE_A: tape[head] = (uint8_t)((v + A + 1) & mask); head = (head + 1) & RMASK; break;  /* mate: cell += A + 1, step right */
        case MATE_B: tape[head] = (uint8_t)((A + 1) & mask); head = (head + 1) & RMASK; break;      /* mate: cell = A + 1, step right */
        case SLEEP: if (A == 0) pc = (pc + 1) % L; break;                                  /* sleep: skip next if no energy */
        case WASTE: tape[head] = mask; A = 0; head = (head - 1) & RMASK; break;             /* waste: cell = all ones, stomach emptied, step left */
        }
        done = step;
        if (halted) break;
    }
    st->head = head; st->pc = pc; st->A = A; st->halted = halted; st->steps = done;
}

/* canonical tape hash: FNV-1a 64 over (lo, cells lo..hi); blank tape -> fixed constant */
static uint64_t tape_hash(const uint8_t *tape, int *nonblank)
{
    int lo = -1, hi = -1;
    for (int i = 0; i < NCELL; i++) if (tape[i]) { if (lo < 0) lo = i; hi = i; }
    uint64_t h = 1469598103934665603ULL;
    if (lo < 0) { *nonblank = 0; return h; }
    *nonblank = 1;
    h = (h ^ (uint64_t)lo) * 1099511628211ULL;
    for (int i = lo; i <= hi; i++) h = (h ^ tape[i]) * 1099511628211ULL;
    return h;
}

/* table_id -> 4 ascending op indices (combinatorial order) */
static void unrank(long id, int *out)
{
    static long C[NBASE + 1][5];
    static int init = 0;
    if (!init) {
        for (int n = 0; n <= NBASE; n++) { C[n][0] = 1; for (int k = 1; k <= 4; k++) C[n][k] = n ? C[n - 1][k - 1] + C[n - 1][k] : 0; }
        init = 1;
    }
    int start = 0;
    for (int k = 4; k >= 1; k--) {
        for (int v = start; v < NBASE; v++) {
            long block = C[NBASE - 1 - v][k - 1];
            if (id < block) { out[4 - k] = v; start = v + 1; break; }
            id -= block;
        }
    }
}

/* per-thread hash set with generation stamps (no reset between tables) */
#define SLOTS (1u << 22)
typedef struct { uint64_t h; uint32_t gen; } Slot;
static int insert(Slot *set, uint32_t gen, uint64_t h)
{
    uint32_t i = (uint32_t)(h ^ (h >> 29)) & (SLOTS - 1);
    for (;;) {
        if (set[i].gen != gen) { set[i].gen = gen; set[i].h = h; return 1; }
        if (set[i].h == h) return 0;
        i = (i + 1) & (SLOTS - 1);
    }
}


/* ---- function mode: 4-bit input x in the first cells, output = the same cells after the run ---- */
static int ncells_for(int W) { return (4 + W - 1) / W; }
static void encode(int x, int W, uint8_t *cells, int n) { for (int i = 0; i < n; i++) cells[i] = (uint8_t)((x >> (W * i)) & ((1 << W) - 1)); }
static int decode(const uint8_t *cells, int W, int n) { int v = 0; for (int i = 0; i < n; i++) v |= cells[i] << (W * i); return v & 15; }
/* truth table of a program: 16 nibbles packed into 64 bits, input x -> output nibble */
static uint64_t truth(const int *table, const uint8_t *prog, int L, int W, int steps)
{
    int n = ncells_for(W);
    uint8_t init[16];
    State st;
    uint64_t t = 0;
    for (int x = 0; x < 16; x++) {
        encode(x, W, init, n);
        run(table, prog, L, W, steps, &st, init, n);
        t |= (uint64_t)decode(st.tape, W, n) << (4 * x);
    }
    return t;
}
static int nib(uint64_t t, int x) { return (int)((t >> (4 * x)) & 15); }
static int is_bijection(uint64_t t) { int seen = 0; for (int x = 0; x < 16; x++) seen |= 1 << nib(t, x); return seen == 0xFFFF; }
#define NVOCAB 64
static uint64_t VOCAB[NVOCAB];
static const char *VNAME[NVOCAB];
static int nvocab = 0;
static void addv(const char *name, int (*f)(int)) { uint64_t t = 0; for (int x = 0; x < 16; x++) t |= (uint64_t)(f(x) & 15) << (4 * x); VOCAB[nvocab] = t; VNAME[nvocab++] = name; }
static int f_id(int x) { return x; }        static int f_inc(int x) { return x + 1; }      static int f_dec(int x) { return x - 1; }
static int f_not(int x) { return ~x; }      static int f_neg(int x) { return -x; }         static int f_shl(int x) { return x << 1; }
static int f_shr(int x) { return x >> 1; }  static int f_rol(int x) { return (x << 1) | (x >> 3); } static int f_ror(int x) { return (x >> 1) | (x << 3); }
static int f_rev(int x) { return ((x & 1) << 3) | ((x & 2) << 1) | ((x & 4) >> 1) | ((x & 8) >> 3); }
static int f_swap(int x) { return ((x & 3) << 2) | (x >> 2); }  static int f_iszero(int x) { return x == 0; }
static int f_nonzero(int x) { return x != 0; }  static int f_parity(int x) { return __builtin_popcount(x) & 1; }
static int f_popcnt(int x) { return __builtin_popcount(x); }  static int f_sq(int x) { return x * x; }
static int f_mul3(int x) { return 3 * x; }  static int f_mul5(int x) { return 5 * x; }  static int f_half(int x) { return x / 2; }
static int f_ge8(int x) { return x >= 8; }  static int f_lt8(int x) { return x < 8; }   static int f_x2p1(int x) { return 2 * x + 1; }
static int f_add2(int x) { return x + 2; }  static int f_add3(int x) { return x + 3; }  static int f_add4(int x) { return x + 4; }
static int f_add8(int x) { return x + 8; }  static int f_xor1(int x) { return x ^ 1; }  static int f_xor8(int x) { return x ^ 8; }
static int f_and1(int x) { return x & 1; }  static int f_and8(int x) { return x & 8; }  static int f_or1(int x) { return x | 1; }
static int f_c0(int x) { (void)x; return 0; } static int f_c15(int x) { (void)x; return 15; } static int f_c1(int x) { (void)x; return 1; }
static int f_mod3(int x) { return x % 3; }  static int f_mod5(int x) { return x % 5; }  static int f_min7(int x) { return x < 7 ? x : 7; }
static int f_sub_from15(int x) { return 15 - x; } static int f_absdiff8(int x) { return x > 8 ? x - 8 : 8 - x; }
static void init_vocab(void)
{
    addv("id", f_id); addv("x+1", f_inc); addv("x-1", f_dec); addv("~x", f_not); addv("-x", f_neg); addv("x<<1", f_shl);
    addv("x>>1", f_shr); addv("rol1", f_rol); addv("ror1", f_ror); addv("bitrev", f_rev); addv("swapnib", f_swap);
    addv("x==0", f_iszero); addv("x!=0", f_nonzero); addv("parity", f_parity); addv("popcount", f_popcnt); addv("x*x", f_sq);
    addv("3x", f_mul3); addv("5x", f_mul5); addv("x/2", f_half); addv("x>=8", f_ge8); addv("x<8", f_lt8); addv("2x+1", f_x2p1);
    addv("x+2", f_add2); addv("x+3", f_add3); addv("x+4", f_add4); addv("x+8", f_add8); addv("x^1", f_xor1); addv("x^8", f_xor8);
    addv("x&1", f_and1); addv("x&8", f_and8); addv("x|1", f_or1); addv("const0", f_c0); addv("const15", f_c15); addv("const1", f_c1);
    addv("x%3", f_mod3); addv("x%5", f_mod5); addv("min(x,7)", f_min7); addv("15-x", f_sub_from15); addv("|x-8|", f_absdiff8);
}

static int find_op(const char *name) { for (int i = 0; i < NOPS; i++) if (strcmp(NAMES[i], name) == 0) return i; return -1; }
static int parse_ops(const char *text, int *table)
{
    char buf[256]; strncpy(buf, text, 255); buf[255] = 0;
    int n = 0;
    for (char *tok = strtok(buf, ","); tok && n < 4; tok = strtok(NULL, ",")) { table[n] = find_op(tok); if (table[n] < 0) return 0; n++; }
    return n == 4;
}
/* custom MODE W L steps OPS threads : one explicit machine; MODE = tapes | fn ; all programs of 1..L; exact union */
static int custom_mode(int argc, char **argv)
{
    const char *mode = argv[2];
    int W = atoi(argv[3]), L = atoi(argv[4]), steps = atoi(argv[5]);
    int table[4];
    if (!parse_ops(argv[6], table)) { fprintf(stderr, "unknown operation in %s\n", argv[6]); return 2; }
    int threads = atoi(argv[7]);
    int fn = strcmp(mode, "fn") == 0;
    if (fn) init_vocab();
    const unsigned BIG = 1u << 26;
    Slot *global = calloc(BIG, sizeof(Slot));
    long distinct = 0, after[MAXL + 1] = {0}, halted = 0, nonblank = 0;
    uint64_t checksum = 0;
    uint64_t *all = NULL; size_t nall = 0, capall = 0;
    double t0 = omp_get_wtime();
    for (int l = 1; l <= L; l++) {
        long nprog = 1L << (2 * l);
        uint64_t **vecs = calloc((size_t)threads, sizeof(uint64_t *)); size_t *lens = calloc((size_t)threads, sizeof(size_t));
        #pragma omp parallel num_threads(threads) reduction(+:halted,nonblank,checksum)
        {
            int tid = omp_get_thread_num();
            Slot *set = calloc(SLOTS, sizeof(Slot));
            size_t cap = 1 << 12, len = 0; uint64_t *vec = malloc(cap * sizeof(uint64_t));
            uint8_t prog[MAXL]; State st;
            #pragma omp for schedule(static)
            for (long q = 0; q < nprog; q++) {
                for (int i = 0; i < l; i++) prog[i] = (uint8_t)((q >> (2 * (l - 1 - i))) & 3);
                uint64_t val;
                if (fn) val = truth(table, prog, l, W, steps);
                else { run(table, prog, l, W, steps, &st, NULL, 0); int nb; val = tape_hash(st.tape, &nb); nonblank += nb; halted += st.halted; }
                checksum += val * 0x9E3779B97F4A7C15ULL;
                uint64_t h = (val ^ (val >> 31) ^ 0xA5A5A5A5A5A5A5A5ULL) * 0x9E3779B97F4A7C15ULL;
                if (insert(set, 1, h)) { if (len == cap) { cap *= 2; vec = realloc(vec, cap * sizeof(uint64_t)); } vec[len++] = val; }
            }
            vecs[tid] = vec; lens[tid] = len; free(set);
        }
        for (int t = 0; t < threads; t++) {
            for (size_t i = 0; i < lens[t]; i++) {
                uint64_t val = vecs[t][i];
                uint64_t h = (val ^ (val >> 31) ^ 0xA5A5A5A5A5A5A5A5ULL) * 0x9E3779B97F4A7C15ULL;
                uint32_t ix = (uint32_t)(h ^ (h >> 29)) & (BIG - 1);
                int isnew = 0;
                for (;;) { if (global[ix].gen != 1) { global[ix].gen = 1; global[ix].h = val; isnew = 1; break; } if (global[ix].h == val) break; ix = (ix + 1) & (BIG - 1); }
                if (isnew) { distinct++; if (nall == capall) { capall = capall ? capall * 2 : 1 << 16; all = realloc(all, capall * sizeof(uint64_t)); } all[nall++] = val; }
            }
            free(vecs[t]);
        }
        free(vecs); free(lens);
        after[l] = distinct;
        fprintf(stderr, "  L=%d: %ld distinct (%.1f s)\n", l, distinct, omp_get_wtime() - t0);
    }
    double secs = omp_get_wtime() - t0;
    long per_table = 0; for (int l = 1; l <= L; l++) per_table += 1L << (2 * l);
    printf("machine\t%s,%s,%s,%s\nmode\t%s\nW\t%d\nL\t%d\nsteps\t%d\nprograms\t%ld\ndistinct\t%ld\n", NAMES[table[0]], NAMES[table[1]], NAMES[table[2]], NAMES[table[3]], mode, W, L, steps, per_table, distinct);
    printf("after_length"); for (int l = 1; l <= L; l++) printf("\t%ld", after[l]); printf("\n");
    printf("checksum\t%016llx\n", (unsigned long long)checksum);
    if (!fn) printf("halted\t%ld\nnonblank\t%ld\n", halted, nonblank);
    else {
        long bij = 0, consts = 0; uint64_t vmask = 0; long image[17] = {0};
        for (size_t i = 0; i < nall; i++) {
            uint64_t tt = all[i]; int seen = 0; for (int x = 0; x < 16; x++) seen |= 1 << nib(tt, x);
            image[__builtin_popcount(seen)]++;
            if (seen == 0xFFFF) bij++;
            if (__builtin_popcount(seen) == 1) consts++;
            for (int v = 0; v < nvocab; v++) if (VOCAB[v] == tt) vmask |= 1ULL << v;
        }
        printf("bijections\t%ld\nconstants\t%ld\nvocab_hits\t%d\nvocab\t", bij, consts, __builtin_popcountll(vmask));
        for (int v = 0; v < nvocab; v++) if (vmask >> v & 1) printf("%s;", VNAME[v]);
        printf("\nimage_size_histogram"); for (int k = 1; k <= 16; k++) printf("\t%ld", image[k]); printf("\n");
    }
    char host[64] = "?"; gethostname(host, sizeof host);
    printf("secs\t%.1f\nthreads\t%d\nhost\t%s\n", secs, threads, host);
    free(global); free(all);
    return 0;
}

/* ---- prim mode: arithmetic primitives of ONE explicit machine over all programs of EXACTLY L symbols whose index
 * (base-4 number of the program, most significant symbol first) lies in [start, start+count) ----
 *
 *   u2_tables prim PRIMS W L steps OPS start count threads
 *
 * PRIMS is a comma-separated subset of COPY, ADD, MUL_T, MAC_T, MAC_B, CARRY with the definitions and the input
 * order of Ghost/bitnet_u2/prim_search.py (2026-10-09): the inputs are seeded into cells 0, 1, 2 (COPY: a in cell
 * 0; ADD: a, b -> a + b; MUL_T: w in {1, -1, 0}, x -> w * x; MAC_T: acc, w in {1, -1, 0}, x -> acc + w * x; MAC_B:
 * acc, code c in {2, 0, 1} meaning c - 1, x -> acc + (c - 1) * x; CARRY: a, b -> a + b >= 2^W), all mod 2^W.
 * A program is kept when one FIXED cell (for COPY not cell 0), or the cell under the FINAL head, holds the target
 * after the run for EVERY input. Inputs are ordered nonzero target first (stable), and a program stops at its
 * first failing input (candidate cells are a 256-bit mask that only shrinks). Output per primitive: the number of
 * programs still alive after each input, and every program found with its output cells and whether the head cell
 * matched. Several primitives in one call share the program enumeration only.
 */
#define MAXIN 64
typedef struct { const char *name; int excl0; int nin, ninit; uint8_t init[MAXIN][3]; uint8_t target[MAXIN]; } Prim;

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
    /* stable partition: nonzero targets first, as prim_search.py sorts them */
    int k = 0;
    for (int pass = 0; pass < 2; pass++)
        for (int i = 0; i < n; i++)
            if ((tt[i] != 0) == (pass == 0)) { memcpy(p->init[k], ti[i], 3); p->target[k] = (uint8_t)tt[i]; k++; }
    p->nin = n;
    return 1;
}

/* returns the number of inputs the program survived (== p->nin: found); mask/headok describe the output cells */
static int prim_eval(const int *table, const uint8_t *prog, int L, int W, int steps, const Prim *p, State *st, uint64_t *mask, int *headok)
{
    for (int w = 0; w < NW; w++) mask[w] = ~0ULL;
    if (p->excl0) mask[0] &= ~1ULL;
    *headok = 1;
    for (int i = 0; i < p->nin; i++) {
        run(table, prog, L, W, steps, st, p->init[i], p->ninit);
        uint8_t t = p->target[i];
        for (int w = 0; w < NW; w++) {
            uint64_t m = mask[w];
            while (m) { int b = __builtin_ctzll(m); m &= m - 1; if (st->tape[64 * w + b] != t) mask[w] &= ~(1ULL << b); }
        }
        if (*headok && (st->tape[st->head] != t || (p->excl0 && st->head == 0))) *headok = 0;
        if (!orw(mask) && !*headok) return i;
    }
    return p->nin;
}

typedef struct { char *buf; size_t len, cap; } Buf;
static void buf_add(Buf *b, const char *s)
{
    size_t n = strlen(s);
    if (b->len + n + 1 > b->cap) { b->cap = b->cap ? b->cap * 2 : 1 << 16; while (b->len + n + 1 > b->cap) b->cap *= 2; b->buf = realloc(b->buf, b->cap); }
    memcpy(b->buf + b->len, s, n + 1); b->len += n;
}

static int prim_mode(int argc, char **argv)
{
    (void)argc;
    char names[256]; strncpy(names, argv[2], 255); names[255] = 0;
    int W = atoi(argv[3]), L = atoi(argv[4]), steps = atoi(argv[5]);
    int table[4];
    if (!parse_ops(argv[6], table)) { fprintf(stderr, "unknown operation in %s\n", argv[6]); return 2; }
    unsigned long long start = strtoull(argv[7], 0, 0), count = strtoull(argv[8], 0, 0);
    int threads = atoi(argv[9]);
    if (L < 1 || L > MAXL || W < 1 || W > 4) { fprintf(stderr, "1 <= L <= %d, 1 <= W <= 4\n", MAXL); return 2; }
    Prim prims[8]; int np = 0;
    for (char *tok = strtok(names, ","); tok && np < 8; tok = strtok(NULL, ",")) {
        if (!prim_build(&prims[np], tok, W)) { fprintf(stderr, "unknown primitive %s\n", tok); return 2; }
        np++;
    }
    /* per-thread: histogram of "inputs survived" per primitive, found-program text per primitive */
    unsigned long long *hist = calloc((size_t)threads * np * (MAXIN + 1), sizeof(unsigned long long));
    Buf *found = calloc((size_t)threads * np, sizeof(Buf));
    double t0 = omp_get_wtime();
    #pragma omp parallel num_threads(threads)
    {
        int tid = omp_get_thread_num();
        unsigned long long *h = hist + (size_t)tid * np * (MAXIN + 1);
        Buf *fb = found + (size_t)tid * np;
        uint8_t prog[MAXL]; State st; uint64_t mask[NW]; int headok;
        char line[512];
        #pragma omp for schedule(dynamic, 65536)
        for (unsigned long long q = start; q < start + count; q++) {
            for (int i = 0; i < L; i++) prog[i] = (uint8_t)((q >> (2 * (L - 1 - i))) & 3);
            for (int k = 0; k < np; k++) {
                int ns = prim_eval(table, prog, L, W, steps, &prims[k], &st, mask, &headok);
                h[k * (MAXIN + 1) + ns]++;
                if (ns == prims[k].nin) {
                    int n = snprintf(line, sizeof line, "F\t%s\t", prims[k].name);
                    for (int i = 0; i < L; i++) line[n++] = (char)('0' + prog[i]);
                    line[n++] = '\t';
                    int first = 1;
                    for (int c = 0; c < NCELL; c++) if (mask[c >> 6] >> (c & 63) & 1) { n += snprintf(line + n, sizeof line - n, first ? "%d" : ",%d", c); first = 0; }
                    if (first) line[n++] = '-';
                    n += snprintf(line + n, sizeof line - n, "\t%d\n", headok);
                    buf_add(&fb[k], line);
                }
            }
        }
    }
    double secs = omp_get_wtime() - t0;
    printf("machine\t%s,%s,%s,%s\nmode\tprim\nW\t%d\nL\t%d\nsteps\t%d\nstart\t%llu\nprograms\t%llu\n",
           NAMES[table[0]], NAMES[table[1]], NAMES[table[2]], NAMES[table[3]], W, L, steps, start, count);
    for (int k = 0; k < np; k++) {
        const Prim *p = &prims[k];
        unsigned long long tot[MAXIN + 1] = {0}, nfound = 0;
        for (int t = 0; t < threads; t++) for (int i = 0; i <= p->nin; i++) tot[i] += hist[((size_t)t * np + k) * (MAXIN + 1) + i];
        nfound = tot[p->nin];
        printf("prim\t%s\ninputs\t%d\nexcluded_cell0\t%d\norder", p->name, p->nin, p->excl0);
        for (int i = 0; i < p->nin; i++) { printf("\t"); for (int j = 0; j < p->ninit; j++) printf("%s%d", j ? "," : "", p->init[i][j]); printf(":%d", p->target[i]); }
        printf("\nsurvivors");
        unsigned long long surv[MAXIN], alive = 0;
        for (int i = p->nin - 1; i >= 0; i--) { alive += tot[i + 1]; surv[i] = alive; }   /* alive after input i = programs that survived more than i inputs */
        for (int i = 0; i < p->nin; i++) printf("\t%llu", surv[i]);
        printf("\nfound\t%llu\n", nfound);
        for (int t = 0; t < threads; t++) { Buf *b = &found[(size_t)t * np + k]; if (b->len) fputs(b->buf, stdout); free(b->buf); }
    }
    fflush(stdout);
    char host[64] = "?"; gethostname(host, sizeof host);
    fprintf(stderr, "mode=prim prims=%s ops=%s,%s,%s,%s W=%d L=%d steps=%d programs=%llu start=%llu secs=%.1f threads=%d host=%s\n",
            argv[2], NAMES[table[0]], NAMES[table[1]], NAMES[table[2]], NAMES[table[3]], W, L, steps, count, start, secs, threads, host);
    free(hist); free(found);
    return 0;
}

/* ---- primany mode: the primitives of prim mode at EVERY step budget 1..steps at once (docs/14) ----
 *
 *   u2_tables primany PRIMS W L steps OPS start count threads [maxin]
 *
 * Ring size and step limit are compile-time: NCELL (256) and MAXSTEPS (256); gcc -DNCELL=512 -DMAXSTEPS=512 builds the
 * 512-cell, 512-budget variant of docs/14 section 11 (every head move wraps with & (NCELL - 1)).
 *
 * Same machine, programs, primitives, inputs and input order as prim mode (maxin > 0 keeps only the first maxin
 * inputs of every primitive, which exhibits the programs that survive that many). A program is the primitive "at budget s"
 * when one FIXED cell, or the cell under the head after exactly s instructions, holds the target after exactly s
 * instructions for EVERY input. This mode keeps one candidate mask per step, cand[s] (256 cell bits and a head bit),
 * all of which only shrink as the inputs go by; a program stays alive while some step still has a candidate, and it
 * stops at the first input after which no step has one. The cell masks are maintained incrementally: a step writes
 * at most one cell (the cell under the head before the move), so the set of cells holding the target changes by at
 * most one bit per step. Output per primitive: survivors after each input (alive = some step has a candidate),
 * found (programs alive after every input), found_by_step (programs that are the primitive at each step s = 1..steps),
 * and one line per found program:
 *   F <prim> <program> <steps as ranges, e.g. 5-7,64> <s:cells:h;...> (cells = comma list or -, h = head rule holds)
 * The op switch below is a verbatim copy of the one in run() (inserted by tools/u2_mac_any.py's patch of this file).
 */
#ifndef MAXSTEPS
#define MAXSTEPS 256   /* largest step budget of primany; -DMAXSTEPS=512 for the 512-budget variant */
#endif
#define SW (MAXSTEPS / 64)   /* 64-bit words of a step mask */
static int primany_eval(const int *table, const uint8_t *prog, int L, int W, int steps, const Prim *p, uint8_t *tape,
                        uint64_t (*cand)[NW], uint64_t *hok)
{
    uint8_t mask = (uint8_t)((1 << W) - 1);
    int match[MAXL];
    int need_match = 0;
    for (int i = 0; i < 4; i++) need_match |= table[i] == OPEN || table[i] == CLOSE;
    if (need_match) {
        int stack[MAXL], sp = 0;
        for (int i = 0; i < L; i++) match[i] = -1;
        for (int i = 0; i < L; i++) {
            int op = table[prog[i]];
            if (op == OPEN) stack[sp++] = i;
            else if (op == CLOSE && sp) { int j = stack[--sp]; match[i] = j; match[j] = i; }
        }
    }
    const uint64_t cell0 = p->excl0 ? ~1ULL : ~0ULL;
    for (int i = 0; i < p->nin; i++) {
        const uint8_t t = p->target[i];
        memset(tape, 0, NCELL);
        for (int j = 0; j < p->ninit; j++) tape[j] = p->init[i][j];
        uint64_t eq[NW];
        for (int k = 0; k < NW; k++) eq[k] = t == 0 ? ~0ULL : 0;
        for (int c = 0; c < p->ninit; c++) { if (tape[c] == t) eq[0] |= 1ULL << c; else eq[0] &= ~(1ULL << c); }
        int head = 0, pc = 0, A = 0, halted = 0;
        uint64_t alive = 0;
        for (int s = 0; s < steps; s++) {
            if (!halted) {
                int at = pc, wh = head;
                int op = table[prog[pc]];
                pc = (pc + 1) % L;
                uint8_t v = tape[head];
                switch (op) {
                case R1: head = (head + 1) & RMASK; break;
                case L1: head = (head - 1) & RMASK; break;
                case R2: head = (head + 2) & RMASK; break;
                case L2: head = (head - 2) & RMASK; break;
                case R3: head = (head + 3) & RMASK; break;
                case L3: head = (head - 3) & RMASK; break;
                case INC: tape[head] = (uint8_t)((v + 1) & mask); break;
                case DEC: tape[head] = (uint8_t)((v - 1) & mask); break;
                case FLIP: tape[head] = v ^ 1; break;
                case NOT: tape[head] = v ^ mask; break;
                case ZERO: tape[head] = 0; break;
                case SET: tape[head] = mask; break;
                case SHL: tape[head] = (uint8_t)((v << 1) & mask); break;
                case INC_R1: tape[head] = (uint8_t)((v + 1) & mask); head = (head + 1) & RMASK; break;
                case INC_L1: tape[head] = (uint8_t)((v + 1) & mask); head = (head - 1) & RMASK; break;
                case FLIP_R1: tape[head] = v ^ 1; head = (head + 1) & RMASK; break;
                case FLIP_L1: tape[head] = v ^ 1; head = (head - 1) & RMASK; break;
                case FLIP_R2: tape[head] = v ^ 1; head = (head + 2) & RMASK; break;
                case FLIP_R3: tape[head] = v ^ 1; head = (head + 3) & RMASK; break;
                case SET_R1: tape[head] = mask; head = (head + 1) & RMASK; break;
                case SET_R3: tape[head] = mask; head = (head + 3) & RMASK; break;
                case ZERO_R1: tape[head] = 0; head = (head + 1) & RMASK; break;
                case ZERO_R3: tape[head] = 0; head = (head + 3) & RMASK; break;
                case SKZ: if (v == 0) pc = (pc + 1) % L; break;
                case SKNZ: if (v != 0) pc = (pc + 1) % L; break;
                case JZS: if (v == 0) pc = 0; break;
                case JNZS: if (v != 0) pc = 0; break;
                case JNZB: if (v != 0) pc = ((pc - 2) % L + L) % L; break;
                case OPEN: if (v == 0) pc = match[at] >= 0 ? (match[at] + 1) % L : 0; break;
                case CLOSE: if (v != 0) pc = match[at] >= 0 ? (match[at] + 1) % L : 0; break;
                case HALT: halted = 1; break;
                case NOP: break;
                case LDA: A = v; break;
                case STA: tape[head] = (uint8_t)A; break;
                case SWAP: tape[head] = (uint8_t)A; A = v; break;
                case ADDA: tape[head] = (uint8_t)((v + A) & mask); break;
                case EAT: A = v; tape[head] = 0; break;                                            /* eat: cell into the stomach, cell emptied */
                case MATE_A: tape[head] = (uint8_t)((v + A + 1) & mask); head = (head + 1) & RMASK; break;  /* mate: cell += A + 1, step right */
                case MATE_B: tape[head] = (uint8_t)((A + 1) & mask); head = (head + 1) & RMASK; break;      /* mate: cell = A + 1, step right */
                case SLEEP: if (A == 0) pc = (pc + 1) % L; break;                                  /* sleep: skip next if no energy */
                case WASTE: tape[head] = mask; A = 0; head = (head - 1) & RMASK; break;             /* waste: cell = all ones, stomach emptied, step left */
                }
                uint8_t nv = tape[wh];
                if (nv != v) { if (nv == t) eq[wh >> 6] |= 1ULL << (wh & 63); else eq[wh >> 6] &= ~(1ULL << (wh & 63)); }
            }
            int h = tape[head] == t && !(p->excl0 && head == 0);
            uint64_t *c = cand[s];
            uint64_t hb = 1ULL << (s & 63);
            if (i == 0) {
                for (int k = 0; k < NW; k++) c[k] = eq[k];
                c[0] &= cell0;
                if (h) hok[s >> 6] |= hb; else hok[s >> 6] &= ~hb;
            } else {
                for (int k = 0; k < NW; k++) c[k] &= eq[k];
                if (!h) hok[s >> 6] &= ~hb;
            }
            alive |= orw(c) | (hok[s >> 6] & hb);
        }
        if (!alive) return i;
    }
    return p->nin;
}

static int primany_mode(int argc, char **argv)
{
    (void)argc;
    char names[256]; strncpy(names, argv[2], 255); names[255] = 0;
    int W = atoi(argv[3]), L = atoi(argv[4]), steps = atoi(argv[5]);
    int table[4];
    if (!parse_ops(argv[6], table)) { fprintf(stderr, "unknown operation in %s\n", argv[6]); return 2; }
    unsigned long long start = strtoull(argv[7], 0, 0), count = strtoull(argv[8], 0, 0);
    int threads = atoi(argv[9]);
    int maxin = argc > 10 ? atoi(argv[10]) : 0;   /* optional: keep only the first maxin inputs, to exhibit the programs that get that far */
    if (L < 1 || L > MAXL || W < 1 || W > 4 || steps < 1 || steps > MAXSTEPS) { fprintf(stderr, "1 <= L <= %d, 1 <= W <= 4, 1 <= steps <= %d\n", MAXL, MAXSTEPS); return 2; }
    Prim prims[8]; int np = 0;
    for (char *tok = strtok(names, ","); tok && np < 8; tok = strtok(NULL, ",")) {
        if (!prim_build(&prims[np], tok, W)) { fprintf(stderr, "unknown primitive %s\n", tok); return 2; }
        if (maxin > 0 && maxin < prims[np].nin) prims[np].nin = maxin;
        np++;
    }
    unsigned long long *hist = calloc((size_t)threads * np * (MAXIN + 1), sizeof(unsigned long long));
    unsigned long long *fbs = calloc((size_t)threads * np * MAXSTEPS, sizeof(unsigned long long));
    Buf *found = calloc((size_t)threads * np, sizeof(Buf));
    double t0 = omp_get_wtime();
    #pragma omp parallel num_threads(threads)
    {
        int tid = omp_get_thread_num();
        unsigned long long *h = hist + (size_t)tid * np * (MAXIN + 1);
        unsigned long long *fb_s = fbs + (size_t)tid * np * MAXSTEPS;
        Buf *fb = found + (size_t)tid * np;
        uint8_t prog[MAXL], tape[NCELL];
        uint64_t cand[MAXSTEPS][NW], hok[SW];
        const size_t cap = 1 << 20;
        char *line = malloc(cap);
        #pragma omp for schedule(dynamic, 16384)
        for (unsigned long long q = start; q < start + count; q++) {
            for (int i = 0; i < L; i++) prog[i] = (uint8_t)((q >> (2 * (L - 1 - i))) & 3);
            for (int k = 0; k < np; k++) {
                int ns = primany_eval(table, prog, L, W, steps, &prims[k], tape, cand, hok);
                h[k * (MAXIN + 1) + ns]++;
                if (ns == prims[k].nin) {
                    size_t n = (size_t)snprintf(line, cap, "F\t%s\t", prims[k].name);
                    for (int i = 0; i < L; i++) line[n++] = (char)('0' + prog[i]);
                    line[n++] = '\t';
                    /* step ranges */
                    int run_start = -1, first = 1;
                    for (int s = 0; s <= steps; s++) {
                        int on = s < steps && (orw(cand[s]) || (hok[s >> 6] >> (s & 63) & 1));
                        if (on && run_start < 0) run_start = s;
                        if (!on && run_start >= 0) {
                            if (s - 1 > run_start) n += (size_t)snprintf(line + n, cap - n, first ? "%d-%d" : ",%d-%d", run_start + 1, s);
                            else n += (size_t)snprintf(line + n, cap - n, first ? "%d" : ",%d", run_start + 1);
                            first = 0; run_start = -1;
                        }
                    }
                    line[n++] = '\t';
                    first = 1;
                    for (int s = 0; s < steps; s++) {
                        int hb = (int)(hok[s >> 6] >> (s & 63) & 1);
                        if (!orw(cand[s]) && !hb) continue;
                        fb_s[k * MAXSTEPS + s]++;
                        if (n + 2048 > cap) { line[n++] = '!'; break; }   /* never expected: 1 MB per found program */
                        n += (size_t)snprintf(line + n, cap - n, first ? "%d:" : ";%d:", s + 1);
                        first = 0;
                        int fc = 1;
                        for (int c = 0; c < NCELL; c++) if (cand[s][c >> 6] >> (c & 63) & 1) { n += (size_t)snprintf(line + n, cap - n, fc ? "%d" : ",%d", c); fc = 0; }
                        if (fc) line[n++] = '-';
                        n += (size_t)snprintf(line + n, cap - n, ":%d", hb);
                    }
                    line[n++] = '\n'; line[n] = 0;
                    buf_add(&fb[k], line);
                }
            }
        }
        free(line);
    }
    double secs = omp_get_wtime() - t0;
    printf("machine\t%s,%s,%s,%s\nmode\tprimany\nW\t%d\nL\t%d\nsteps\t%d\nstart\t%llu\nprograms\t%llu\n",
           NAMES[table[0]], NAMES[table[1]], NAMES[table[2]], NAMES[table[3]], W, L, steps, start, count);
    for (int k = 0; k < np; k++) {
        const Prim *p = &prims[k];
        unsigned long long tot[MAXIN + 1] = {0};
        for (int t = 0; t < threads; t++) for (int i = 0; i <= p->nin; i++) tot[i] += hist[((size_t)t * np + k) * (MAXIN + 1) + i];
        printf("prim\t%s\ninputs\t%d\nexcluded_cell0\t%d\norder", p->name, p->nin, p->excl0);
        for (int i = 0; i < p->nin; i++) { printf("\t"); for (int j = 0; j < p->ninit; j++) printf("%s%d", j ? "," : "", p->init[i][j]); printf(":%d", p->target[i]); }
        printf("\nsurvivors");
        unsigned long long surv[MAXIN], alive = 0;
        for (int i = p->nin - 1; i >= 0; i--) { alive += tot[i + 1]; surv[i] = alive; }
        for (int i = 0; i < p->nin; i++) printf("\t%llu", surv[i]);
        printf("\nfound\t%llu\nfound_by_step", tot[p->nin]);
        for (int s = 0; s < steps; s++) { unsigned long long v = 0; for (int t = 0; t < threads; t++) v += fbs[((size_t)t * np + k) * MAXSTEPS + s]; printf("\t%llu", v); }
        printf("\n");
        for (int t = 0; t < threads; t++) { Buf *b = &found[(size_t)t * np + k]; if (b->len) fputs(b->buf, stdout); free(b->buf); }
    }
    fflush(stdout);
    char host[64] = "?"; gethostname(host, sizeof host);
    fprintf(stderr, "mode=primany prims=%s ops=%s,%s,%s,%s W=%d L=%d steps=%d ncell=%d programs=%llu start=%llu secs=%.1f threads=%d host=%s device=cpu\n",
            argv[2], NAMES[table[0]], NAMES[table[1]], NAMES[table[2]], NAMES[table[3]], W, L, steps, NCELL, count, start, secs, threads, host);
    free(hist); free(fbs); free(found);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc >= 2 && strcmp(argv[1], "--list") == 0) {
        for (int i = 0; i < NOPS; i++) printf("%d %s\n", i, NAMES[i]);
        return 0;
    }
    if (argc >= 8 && strcmp(argv[1], "custom") == 0) return custom_mode(argc, argv);
    if (argc >= 10 && strcmp(argv[1], "prim") == 0) return prim_mode(argc, argv);
    if (argc >= 10 && strcmp(argv[1], "primany") == 0) return primany_mode(argc, argv);
    if (argc >= 4 && strcmp(argv[1], "--checkops") == 0) {
        /* --checkops OPS W : reads "steps program" lines, prints state and truth table for the explicit machine */
        int table[4]; if (!parse_ops(argv[2], table)) return 2; int W = atoi(argv[3]);
        char line[256];
        while (fgets(line, sizeof line, stdin)) {
            int steps; char ps[64]; if (sscanf(line, "%d %63s", &steps, ps) != 2) continue;
            uint8_t prog[MAXL]; int L = (int)strlen(ps); for (int i = 0; i < L; i++) prog[i] = (uint8_t)(ps[i] - '0');
            State st; run(table, prog, L, W, steps, &st, NULL, 0);
            printf("%s head=%d A=%d pc=%d halted=%d steps=%d tape=", ps, st.head, st.A, st.pc, st.halted, st.steps);
            for (int i = 0; i < NCELL; i++) if (st.tape[i]) printf("%d:%d,", i, st.tape[i]);
            uint64_t t = truth(table, prog, L, W, steps); printf(" truth=");
            for (int x = 0; x < 16; x++) printf("%d,", nib(t, x)); printf("\n");
        }
        return 0;
    }
    if (argc >= 2 && strcmp(argv[1], "--vocab") == 0) {
        init_vocab();
        for (int i = 0; i < nvocab; i++) { printf("%s", VNAME[i]); for (int x = 0; x < 16; x++) printf(" %d", nib(VOCAB[i], x)); printf("\n"); }
        return 0;
    }
    if (argc >= 2 && strcmp(argv[1], "--checkfn") == 0) {
        char line[256];
        while (fgets(line, sizeof line, stdin)) {
            int W, steps; long id; char ps[64];
            if (sscanf(line, "%d %d %ld %63s", &W, &steps, &id, ps) != 4) continue;
            int table[4]; unrank(id, table);
            uint8_t prog[MAXL]; int L = (int)strlen(ps);
            for (int i = 0; i < L; i++) prog[i] = (uint8_t)(ps[i] - '0');
            uint64_t t = truth(table, prog, L, W, steps);
            printf("%ld %s", id, ps);
            for (int x = 0; x < 16; x++) printf(" %d", nib(t, x));
            printf("\n");
        }
        return 0;
    }
    if (argc >= 9 && strcmp(argv[1], "fnall") == 0) {
        /* fnall W L steps table_id start count threads: one machine, all programs of EXACTLY L symbols with index in
           [start, start+count) (index = base-4 number of the program), 16 inputs each; prints every distinct truth
           table found (hex, one per line) so that shards from several nodes can be merged exactly. */
        int W = atoi(argv[2]), L = atoi(argv[3]), steps = atoi(argv[4]);
        long id = atol(argv[5]);
        unsigned long long start = strtoull(argv[6], 0, 0), count = strtoull(argv[7], 0, 0);
        int threads = atoi(argv[8]);
        int table[4]; unrank(id, table);
        const unsigned BIG = 1u << 25;
        uint64_t **vecs = calloc((size_t)threads, sizeof(uint64_t *));
        size_t *lens = calloc((size_t)threads, sizeof(size_t));
        double t0 = omp_get_wtime();
        #pragma omp parallel num_threads(threads)
        {
            int tid = omp_get_thread_num();
            Slot *set = calloc(BIG, sizeof(Slot));
            size_t cap = 1 << 16, len = 0;
            uint64_t *vec = malloc(cap * sizeof(uint64_t));
            uint8_t prog[MAXL];
            #pragma omp for schedule(static)
            for (unsigned long long p = start; p < start + count; p++) {
                for (int i = 0; i < L; i++) prog[i] = (uint8_t)((p >> (2 * (L - 1 - i))) & 3);
                uint64_t tt = truth(table, prog, L, W, steps);
                uint64_t h = (tt ^ (tt >> 31) ^ 0xA5A5A5A5A5A5A5A5ULL) * 0x9E3779B97F4A7C15ULL;
                /* big set: gen 1, open addressing over BIG slots */
                uint32_t ix = (uint32_t)(h ^ (h >> 29)) & (BIG - 1);
                int isnew = 0;
                for (;;) {
                    if (set[ix].gen != 1) { set[ix].gen = 1; set[ix].h = tt; isnew = 1; break; }
                    if (set[ix].h == tt) break;
                    ix = (ix + 1) & (BIG - 1);
                }
                if (isnew) {
                    if (len == cap) { cap *= 2; vec = realloc(vec, cap * sizeof(uint64_t)); }
                    vec[len++] = tt;
                }
            }
            vecs[tid] = vec; lens[tid] = len;
            free(set);
        }
        double secs = omp_get_wtime() - t0;
        size_t total = 0;
        for (int t = 0; t < threads; t++) { for (size_t i = 0; i < lens[t]; i++) printf("%016llx\n", (unsigned long long)vecs[t][i]); total += lens[t]; free(vecs[t]); }
        fflush(stdout);
        char host[64] = "?"; gethostname(host, sizeof host);
        fprintf(stderr, "mode=fnall table=%ld ops=%s,%s,%s,%s W=%d L=%d steps=%d programs=%llu start=%llu lines=%zu secs=%.1f threads=%d host=%s\n",
                id, NAMES[table[0]], NAMES[table[1]], NAMES[table[2]], NAMES[table[3]], W, L, steps, count, start, total, secs, threads, host);
        return 0;
    }
    if (argc >= 8 && strcmp(argv[1], "fn") == 0) {
        /* fn W L steps start count threads: distinct unary 4-bit functions per machine over all programs of 1..L */
        int W = atoi(argv[2]), L = atoi(argv[3]), steps = atoi(argv[4]);
        long start = 0, count = atol(argv[6]);
        long *idlist = NULL;
        if (argv[5][0] == '@') {   /* @file: explicit list of table ids, one per line; count = how many to take */
            FILE *f = fopen(argv[5] + 1, "r");
            if (!f) { fprintf(stderr, "cannot open %s\n", argv[5] + 1); return 2; }
            idlist = malloc((size_t)count * sizeof(long));
            long n = 0;
            while (n < count && fscanf(f, "%ld", &idlist[n]) == 1) n++;
            fclose(f);
            if (n != count) { fprintf(stderr, "id list has %ld entries, expected %ld\n", n, count); return 2; }
        } else start = atol(argv[5]);
        int threads = atoi(argv[7]);
        init_vocab();
        long per_table = 0;
        for (int l = 1; l <= L; l++) per_table += 1L << (2 * l);
        char **lines = calloc((size_t)count, sizeof(char *));
        double t0 = omp_get_wtime();
        #pragma omp parallel num_threads(threads)
        {
            Slot *set = calloc(SLOTS, sizeof(Slot));
            uint32_t gen = 0;
            #pragma omp for schedule(dynamic, 1)
            for (long t = 0; t < count; t++) {
                long id = idlist ? idlist[t] : start + t;
                int table[4]; unrank(id, table);
                gen++;
                long distinct = 0, bij = 0, consts = 0, after[MAXL + 1] = {0};
                uint64_t vmask = 0, checksum = 0;
                uint8_t prog[MAXL];
                for (int l = 1; l <= L; l++) {
                    long nprog = 1L << (2 * l);
                    for (long p = 0; p < nprog; p++) {
                        for (int i = 0; i < l; i++) prog[i] = (uint8_t)((p >> (2 * (l - 1 - i))) & 3);
                        uint64_t tt = truth(table, prog, l, W, steps);
                        checksum += tt * 0x9E3779B97F4A7C15ULL;
                        uint64_t h = (tt ^ (tt >> 31) ^ 0xA5A5A5A5A5A5A5A5ULL) * 0x9E3779B97F4A7C15ULL;
                        if (insert(set, gen, h)) {
                            distinct++;
                            if (is_bijection(tt)) bij++;
                            int c = 1; for (int x = 1; x < 16; x++) if (nib(tt, x) != nib(tt, 0)) { c = 0; break; }
                            consts += c;
                            for (int v = 0; v < nvocab; v++) if (VOCAB[v] == tt) vmask |= 1ULL << v;
                        }
                    }
                    after[l] = distinct;
                }
                char *buf = malloc(1024);
                int n = snprintf(buf, 1024, "%ld\t%s\t%s\t%s\t%s", id, NAMES[table[0]], NAMES[table[1]], NAMES[table[2]], NAMES[table[3]]);
                for (int l = 1; l <= L; l++) n += snprintf(buf + n, 1024 - n, "\t%ld", after[l]);
                n += snprintf(buf + n, 1024 - n, "\t%ld\t%ld\t%d\t%016llx\t", bij, consts, __builtin_popcountll(vmask), (unsigned long long)checksum);
                for (int v = 0; v < nvocab; v++) if (vmask >> v & 1) n += snprintf(buf + n, 1024 - n, "%s,", VNAME[v]);
                lines[t] = buf;
            }
            free(set);
        }
        double secs = omp_get_wtime() - t0;
        for (long t = 0; t < count; t++) { puts(lines[t]); free(lines[t]); }
        fflush(stdout);
        char host[64] = "?"; gethostname(host, sizeof host);
        fprintf(stderr, "mode=fn tables=%ld programs_per_table=%ld inputs=16 W=%d L=%d steps=%d secs=%.1f threads=%d host=%s\n",
                count, per_table, W, L, steps, secs, threads, host);
        return 0;
    }
    if (argc >= 2 && strcmp(argv[1], "--check") == 0) {
        char line[256];
        while (fgets(line, sizeof line, stdin)) {
            int W, steps; long id; char ps[64];
            if (sscanf(line, "%d %d %ld %63s", &W, &steps, &id, ps) != 4) continue;
            int table[4]; unrank(id, table);
            uint8_t prog[MAXL]; int L = (int)strlen(ps);
            for (int i = 0; i < L; i++) prog[i] = (uint8_t)(ps[i] - '0');
            State st; run(table, prog, L, W, steps, &st, NULL, 0);
            printf("%ld %s,%s,%s,%s %s head=%d A=%d pc=%d halted=%d steps=%d tape=", id, NAMES[table[0]], NAMES[table[1]],
                   NAMES[table[2]], NAMES[table[3]], ps, st.head, st.A, st.pc, st.halted, st.steps);
            for (int i = 0; i < NCELL; i++) if (st.tape[i]) printf("%d:%d,", i, st.tape[i]);
            printf("\n");
        }
        return 0;
    }
    if (argc < 7) { fprintf(stderr, "usage: u2_tables W L steps start count threads | prim PRIMS W L steps OPS start count threads | primany ... | --check | --list\n"); return 2; }
    int W = atoi(argv[1]), L = atoi(argv[2]), steps = atoi(argv[3]);
    long start = atol(argv[4]), count = atol(argv[5]);
    int threads = atoi(argv[6]);
    if (L > 12 || W < 1 || W > 8) { fprintf(stderr, "L <= 12, 1 <= W <= 8 (the table modes; prim allows L <= %d)\n", MAXL); return 2; }
    long per_table = 0;
    for (int l = 1; l <= L; l++) per_table += 1L << (2 * l);

    char **lines = calloc((size_t)count, sizeof(char *));
    double t0 = omp_get_wtime();
    #pragma omp parallel num_threads(threads)
    {
        Slot *set = calloc(SLOTS, sizeof(Slot));
        uint32_t gen = 0;
        #pragma omp for schedule(dynamic, 1)
        for (long t = 0; t < count; t++) {
            long id = start + t;
            int table[4]; unrank(id, table);
            gen++;
            long union_after[MAXL + 1] = {0}, halted = 0, nonblank = 0, distinct = 0;
            uint64_t checksum = 0;
            State st;
            uint8_t prog[MAXL];
            for (int l = 1; l <= L; l++) {
                long nprog = 1L << (2 * l);
                for (long p = 0; p < nprog; p++) {
                    for (int i = 0; i < l; i++) prog[i] = (uint8_t)((p >> (2 * (l - 1 - i))) & 3);
                    run(table, prog, l, W, steps, &st, NULL, 0);
                    int nb;
                    uint64_t h = tape_hash(st.tape, &nb);
                    checksum += h;
                    nonblank += nb;
                    halted += st.halted;
                    distinct += insert(set, gen, h);
                }
                union_after[l] = distinct;
            }
            char *buf = malloc(512);
            int n = snprintf(buf, 512, "%ld\t%s\t%s\t%s\t%s", id, NAMES[table[0]], NAMES[table[1]], NAMES[table[2]], NAMES[table[3]]);
            for (int l = 1; l <= L; l++) n += snprintf(buf + n, 512 - n, "\t%ld", union_after[l]);
            snprintf(buf + n, 512 - n, "\t%ld\t%ld\t%016llx", halted, nonblank, (unsigned long long)checksum);
            lines[t] = buf;
        }
        free(set);
    }
    double secs = omp_get_wtime() - t0;
    for (long t = 0; t < count; t++) { puts(lines[t]); free(lines[t]); }
    fflush(stdout);
    char host[64] = "?"; gethostname(host, sizeof host);
    fprintf(stderr, "tables=%ld programs_per_table=%ld W=%d L=%d steps=%d secs=%.1f threads=%d host=%s\n",
            count, per_table, W, L, steps, secs, threads, host);
    return 0;
}
