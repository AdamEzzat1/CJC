/* Bit-compare cjc_repro::dmath outputs (dmath_ext_out.txt) against the musl
 * C sources they were transcribed from, compiled here with no FMA contraction.
 * Paths that call musl's table-driven exp (not transcribed: dmath uses the
 * fdlibm exp) are counted separately as "exp-path". */
#include <stdio.h>
#include <string.h>

typedef double (*f1)(double);
typedef double (*f2)(double, double);
static double hex2d(const char *s) { union {double f; uint64_t i;} u; sscanf(s, "%llx", (unsigned long long *)&u.i); return u.f; }
static uint64_t d2u(double d) { union {double f; uint64_t i;} u; u.f = d; return u.i; }

struct row { const char *name; f1 one; f2 two; long n, same, diff, exppath; };

int main(int argc, char **argv) {
    struct row rows[] = {
        {"exp_m1", m_expm1, 0}, {"ln_1p", m_log1p, 0}, {"tanh", m_tanh, 0},
        {"sinh", m_sinh, 0}, {"cosh", m_cosh, 0}, {"atanh", m_atanh, 0},
        {"atan", m_atan, 0}, {"asin", m_asin, 0}, {"acos", m_acos, 0},
        {"log10", m_log10, 0}, {"atan2", 0, m_atan2}, {"hypot", 0, m_hypot},
    };
    int nrows = sizeof rows / sizeof rows[0];
    FILE *fp = fopen(argv[1], "r");
    char line[256], name[32], a[32], b[32], c[32];
    int shown = 0;
    while (fgets(line, sizeof line, fp)) {
        int k = sscanf(line, "%31s %31s %31s %31s", name, a, b, c);
        for (int r = 0; r < nrows; r++) {
            if (strcmp(rows[r].name, name)) continue;
            double x = hex2d(a), ours, theirs;
            if (rows[r].two) { ours = hex2d(c); theirs = rows[r].two(x, hex2d(b)); }
            else { ours = hex2d(b); theirs = rows[r].one(x); }
            rows[r].n++;
            /* musl cosh uses exp() for |x| >= ln2; sinh uses __expo2 (exp) for |x| >= ln(DBL_MAX). */
            double ax = x < 0 ? -x : x;
            int exppath = (!strcmp(name, "cosh") && ax >= 0.6931471805599453) ||
                          (!strcmp(name, "sinh") && ax >= 709.782712893384);
            if (d2u(ours) == d2u(theirs) || (ours != ours && theirs != theirs)) rows[r].same++;
            else if (exppath) rows[r].exppath++;
            else { rows[r].diff++; if (shown++ < 12) printf("MISMATCH %s(%s): dmath %016llx musl %016llx\n", name, a, (unsigned long long)d2u(ours), (unsigned long long)d2u(theirs)); }
        }
        (void)k;
    }
    long total_diff = 0;
    for (int r = 0; r < nrows; r++) {
        printf("%-7s n=%6ld bit-identical=%6ld differ=%ld exp-path(not comparable)=%ld\n",
               rows[r].name, rows[r].n, rows[r].same, rows[r].diff, rows[r].exppath);
        total_diff += rows[r].diff;
    }
    if (total_diff) printf("RESULT: %ld mismatches outside exp paths\n", total_diff);
    else printf("RESULT: bit-identical to musl on every comparable input\n");
    return total_diff != 0;
}
