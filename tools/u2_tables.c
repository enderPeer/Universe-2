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

#define NOPS 36
#define NCELL 256
#define MAXL 12
static const char *NAMES[NOPS] = {
    "R1", "L1", "R2", "L2", "R3", "L3", "INC", "DEC", "FLIP", "NOT", "ZERO", "SET", "SHL",
    "INC_R1", "INC_L1", "FLIP_R1", "FLIP_L1", "FLIP_R2", "FLIP_R3", "SET_R1", "SET_R3", "ZERO_R1", "ZERO_R3",
    "SKZ", "SKNZ", "JZS", "JNZS", "JNZB", "OPEN", "CLOSE", "HALT", "NOP", "LDA", "STA", "SWAP", "ADDA"};
enum { R1, L1, R2, L2, R3, L3, INC, DEC, FLIP, NOT, ZERO, SET, SHL, INC_R1, INC_L1, FLIP_R1, FLIP_L1, FLIP_R2,
       FLIP_R3, SET_R1, SET_R3, ZERO_R1, ZERO_R3, SKZ, SKNZ, JZS, JNZS, JNZB, OPEN, CLOSE, HALT, NOP, LDA, STA,
       SWAP, ADDA };

typedef struct { uint8_t tape[NCELL]; int head, pc, A, halted, steps; } State;

static void run(const int *table, const uint8_t *prog, int L, int W, int steps, State *st)
{
    uint8_t mask = (uint8_t)((1 << W) - 1);
    uint8_t *tape = st->tape;
    memset(tape, 0, NCELL);
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
    static long C[NOPS + 1][5];
    static int init = 0;
    if (!init) {
        for (int n = 0; n <= NOPS; n++) { C[n][0] = 1; for (int k = 1; k <= 4; k++) C[n][k] = n ? C[n - 1][k - 1] + C[n - 1][k] : 0; }
        init = 1;
    }
    int start = 0;
    for (int k = 4; k >= 1; k--) {
        for (int v = start; v < NOPS; v++) {
            long block = C[NOPS - 1 - v][k - 1];
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

int main(int argc, char **argv)
{
    if (argc >= 2 && strcmp(argv[1], "--list") == 0) {
        for (int i = 0; i < NOPS; i++) printf("%d %s\n", i, NAMES[i]);
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
            State st; run(table, prog, L, W, steps, &st);
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
                    run(table, prog, l, W, steps, &st);
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
