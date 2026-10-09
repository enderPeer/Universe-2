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
#define NCELL 256
#define MAXL 12
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
        case R1: head = (head + 1) & 255; break;
        case L1: head = (head - 1) & 255; break;
        case R2: head = (head + 2) & 255; break;
        case L2: head = (head - 2) & 255; break;
        case R3: head = (head + 3) & 255; break;
        case L3: head = (head - 3) & 255; break;
        case INC: tape[head] = (uint8_t)((v + 1) & mask); break;
        case DEC: tape[head] = (uint8_t)((v - 1) & mask); break;
        case FLIP: tape[head] = v ^ 1; break;
        case NOT: tape[head] = v ^ mask; break;
        case ZERO: tape[head] = 0; break;
        case SET: tape[head] = mask; break;
        case SHL: tape[head] = (uint8_t)((v << 1) & mask); break;
        case INC_R1: tape[head] = (uint8_t)((v + 1) & mask); head = (head + 1) & 255; break;
        case INC_L1: tape[head] = (uint8_t)((v + 1) & mask); head = (head - 1) & 255; break;
        case FLIP_R1: tape[head] = v ^ 1; head = (head + 1) & 255; break;
        case FLIP_L1: tape[head] = v ^ 1; head = (head - 1) & 255; break;
        case FLIP_R2: tape[head] = v ^ 1; head = (head + 2) & 255; break;
        case FLIP_R3: tape[head] = v ^ 1; head = (head + 3) & 255; break;
        case SET_R1: tape[head] = mask; head = (head + 1) & 255; break;
        case SET_R3: tape[head] = mask; head = (head + 3) & 255; break;
        case ZERO_R1: tape[head] = 0; head = (head + 1) & 255; break;
        case ZERO_R3: tape[head] = 0; head = (head + 3) & 255; break;
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
        case MATE_A: tape[head] = (uint8_t)((v + A + 1) & mask); head = (head + 1) & 255; break;  /* mate: cell += A + 1, step right */
        case MATE_B: tape[head] = (uint8_t)((A + 1) & mask); head = (head + 1) & 255; break;      /* mate: cell = A + 1, step right */
        case SLEEP: if (A == 0) pc = (pc + 1) % L; break;                                  /* sleep: skip next if no energy */
        case WASTE: tape[head] = mask; A = 0; head = (head - 1) & 255; break;             /* waste: cell = all ones, stomach emptied, step left */
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

int main(int argc, char **argv)
{
    if (argc >= 2 && strcmp(argv[1], "--list") == 0) {
        for (int i = 0; i < NOPS; i++) printf("%d %s\n", i, NAMES[i]);
        return 0;
    }
    if (argc >= 8 && strcmp(argv[1], "custom") == 0) return custom_mode(argc, argv);
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
    if (argc < 7) { fprintf(stderr, "usage: u2_tables W L steps start count threads | --check | --list\n"); return 2; }
    int W = atoi(argv[1]), L = atoi(argv[2]), steps = atoi(argv[3]);
    long start = atol(argv[4]), count = atol(argv[5]);
    int threads = atoi(argv[6]);
    if (L > MAXL || W < 1 || W > 8) { fprintf(stderr, "L <= %d, 1 <= W <= 8\n", MAXL); return 2; }
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
