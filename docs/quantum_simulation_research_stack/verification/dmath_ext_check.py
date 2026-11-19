"""Accuracy check for the cjc_repro::dmath functions added for the runtime
migration, against mpmath.

    cargo run --release --manifest-path dmath_ext_check/Cargo.toml > dmath_ext_out.txt
    python dmath_ext_check.py dmath_ext_out.txt

Lines are `fn x y` or, for two-argument functions, `fn a b y` (hex bits).
For each the error of y is measured in ulps of the true value (at 160 bits,
1400 for huge tan arguments so the reduction is exact). Reports the max error
per function against its documented bound: < 1 ulp, except the musl
algorithms that are not < 1 ulp themselves. Those bounds are the measured
maxima of *musl's own C code* (dmath is bit-identical to it on every input
here; see musl_bitcompare/): tanh ~2 ulp on [0.1, 0.2554] (documented by
musl), atanh ~1.7 ulp for |x| < 0.5 (documented by musl), sinh 1.67, cosh
1.24, and atan2 1.20 ulp (atan of a rounded y/x).
"""
import struct, sys
import mpmath


def f(h):
    return struct.unpack("<d", struct.pack("<Q", int(h, 16)))[0]


def ulp_of(v):
    d = float(v)
    if d == 0.0 or not mpmath.isfinite(d):
        return mpmath.mpf(2) ** -1074
    m, e = mpmath.frexp(abs(mpmath.mpf(d)))
    return mpmath.mpf(2) ** max(e - 53, -1074)


ONE = {
    "tan": mpmath.tan, "asin": mpmath.asin, "acos": mpmath.acos, "atan": mpmath.atan,
    "sinh": mpmath.sinh, "cosh": mpmath.cosh, "tanh": mpmath.tanh, "atanh": mpmath.atanh,
    "exp_m1": mpmath.expm1, "ln_1p": mpmath.log1p,
    "log2": lambda x: mpmath.log(x, 2), "log10": mpmath.log10,
}
TWO = {
    "atan2": lambda a, b: mpmath.atan2(a, b),
    "pow": lambda a, b: mpmath.power(a, b),
    "hypot": lambda a, b: mpmath.sqrt(a * a + b * b),
}
# Measured maxima of musl's own implementations (bit-identical to dmath).
BOUND = {"tanh": 2.0, "atanh": 1.8, "sinh": 1.7, "cosh": 1.3, "atan2": 1.3}

stats = {}
for line in open(sys.argv[1]):
    parts = line.split()
    name = parts[0]
    if name in TWO:
        a, b, y = f(parts[1]), f(parts[2]), f(parts[3])
        args, key = (a, b), parts[1] + "," + parts[2]
    else:
        a, y = f(parts[1]), f(parts[2])
        args, key = (a,), parts[1]
    if any(v != v for v in args):
        continue
    e = abs(a) and mpmath.frexp(abs(a))[1]
    mpmath.mp.prec = 1400 if (name == "tan" and e > 60) else 160
    fn = TWO.get(name) or ONE[name]
    s = stats.setdefault(name, [0, mpmath.mpf(0), 0, None])
    s[0] += 1
    t = fn(*[mpmath.mpf(v) for v in args])
    if isinstance(t, mpmath.mpc):
        if y == y:  # expected NaN (e.g. pow of negative base, non-integer y)
            s[1], s[3] = mpmath.inf, key
        continue
    if not mpmath.isfinite(t):
        if not (y in (float("inf"), float("-inf")) and (t > 0) == (y > 0)):
            s[1], s[3] = mpmath.inf, key
        continue
    if abs(t) > mpmath.mpf(sys.float_info.max):
        if y not in (float("inf"), float("-inf")):
            s[1], s[3] = mpmath.inf, key
        continue
    if y != y or y in (float("inf"), float("-inf")):
        s[1], s[3] = mpmath.inf, key
        continue
    if t == 0:
        if y != 0:
            s[1], s[3] = mpmath.inf, key
        continue
    err = abs(mpmath.mpf(y) - t) / ulp_of(t)
    if err > s[1]:
        s[1], s[3] = err, key
    if y != float(t):
        s[2] += 1

bad = []
for k, (n, mx, ncr, wx) in sorted(stats.items()):
    bound = BOUND.get(k, 1.0)
    flag = "" if mx < bound else "   <-- EXCEEDS BOUND"
    if mx >= bound:
        bad.append(k)
    print(f"{k:7s} n={n:6d} max_err={float(mx):.4f} ulp (bound {bound})  "
          f"not_correctly_rounded={ncr} ({100 * ncr / max(n, 1):.3f}%)  worst=0x{wx}{flag}")
print("FAIL: " + ", ".join(bad) if bad else "PASS: every function within its documented bound")
