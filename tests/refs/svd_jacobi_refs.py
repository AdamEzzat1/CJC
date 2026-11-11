"""Reference singular values for tests/test_svd_jacobi.rs.

Generated with mpmath 1.3.0 at 60 decimal digits (mp.svd_r on the EXACT binary
value of every f64 matrix entry). Values are printed as 17-significant-digit
literals and pasted into the Rust test. Re-run with:

    python -I tests/refs/svd_jacobi_refs.py

Matrices (all entries are exactly the f64 values the Rust test builds):
  * Hilbert n = 2..10, H[i][j] = 1.0 / (i + j + 1) (one correctly rounded f64 division).
  * Vandermonde 7x4, nodes -3..3, V[i][j] = x_i^j (exact integers); also its transpose (4x7).
  * Graded: G = fl(H4 * diag(1, 1e-4, 1e-8, 1e-12) * H4), H4 = I - v v^T / 15, v = [1,2,3,4]
    (an exact rational Householder reflector). The product is formed in mpmath and rounded
    once to f64; the rounded entries are printed so the Rust test uses them verbatim.
  * Rank-deficient 4x3 (col3 = col1 + col2), wide 3x5, wide rank-deficient 3x4 (row3 = row1 - row2).
  * Probe P3: [[1, 1], [1, fl(1 + 1e-10)]].
"""
import mpmath as mp

mp.mp.dps = 60


def sv(rows):
    A = mp.matrix([[mp.mpf(x) for x in r] for r in rows])
    s = mp.svd_r(A, compute_uv=False)
    return [s[i] for i in range(len(s))]


def lit(x):
    # 17 significant digits; tiny values from exactly-singular matrices are
    # mpmath round-off at 60 digits and are emitted as exact zero.
    if abs(x) < mp.mpf(10) ** -40:
        return "0.0"
    s = mp.nstr(x, 17, min_fixed=1, max_fixed=0)
    if "e" not in s and "." not in s:
        s += ".0"
    return s.replace("e-0", "e-").replace("e+", "e")


def emit(name, rows):
    s = sorted(sv(rows), reverse=True)
    print(f"const {name}: &[f64] = &[{', '.join(lit(v) for v in s)}];")


for n in range(2, 11):
    emit(f"HILBERT_{n}", [[1.0 / (i + j + 1) for j in range(n)] for i in range(n)])

vand = [[float(x ** j) for j in range(4)] for x in range(-3, 4)]
emit("VANDERMONDE_7X4", vand)
emit("VANDERMONDE_4X7", [list(c) for c in zip(*vand)])

v = [1, 2, 3, 4]
H = mp.matrix(4, 4)
for i in range(4):
    for j in range(4):
        H[i, j] = (1 if i == j else 0) - mp.mpf(v[i] * v[j]) / 15
D = mp.diag([mp.mpf(1), mp.mpf("1e-4"), mp.mpf("1e-8"), mp.mpf("1e-12")])
G = H * D * H
graded = [[float(G[i, j]) for j in range(4)] for i in range(4)]
print("const GRADED_A: [f64; 16] = [" + ", ".join(repr(x) for r in graded for x in r) + "];")
emit("GRADED", graded)

emit("RANKDEF_4X3", [[1, 2, 3], [4, 5, 9], [7, 8, 15], [2, -1, 1]])
emit("WIDE_3X5", [[2, -1, 0, 3, 1], [1, 4, -2, 0, 5], [0, 3, 1, -1, 2]])
emit("WIDE_RANKDEF_3X4", [[1, 2, 0, -1], [3, -1, 2, 4], [-2, 3, -2, -5]])
emit("PROBE_P3", [[1.0, 1.0], [1.0, 1.0 + 1e-10]])
