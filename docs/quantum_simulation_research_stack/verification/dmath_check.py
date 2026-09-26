"""Accuracy check for cjc_repro::dmath against mpmath.

    cargo run --release --manifest-path dmath_check/Cargo.toml > dmath_out.txt
    python dmath_check.py dmath_out.txt

For every line `fn x y` computes the error of y in ulps of the true value,
using enough working precision that huge sin/cos arguments reduce exactly.
Reports the max error per function and how often the result is not the
correctly rounded double. fdlibm's contract is max error < 1 ulp.
"""
import struct, sys
import mpmath

def f(h):
    return struct.unpack("<d", struct.pack("<Q", int(h, 16)))[0]

def ulp_of(v):
    # ulp of the double nearest the true value v (v != 0)
    d = float(v)
    if d == 0.0 or not mpmath.isfinite(d):
        return mpmath.mpf(2) ** -1074
    m, e = mpmath.frexp(abs(mpmath.mpf(d)))  # d = m * 2^e, m in [0.5, 1)
    return mpmath.mpf(2) ** max(e - 53, -1074)

FUNCS = {"sin": mpmath.sin, "cos": mpmath.cos, "exp": mpmath.exp, "ln": mpmath.log}
stats = {k: [0, mpmath.mpf(0), 0, None] for k in FUNCS}  # n, max_ulp, not_cr, worst_x
for line in open(sys.argv[1]):
    name, xh, yh = line.split()
    x, y = f(xh), f(yh)
    if x != x:
        continue
    e = abs(x) and mpmath.frexp(abs(x))[1]
    mpmath.mp.prec = 1400 if (name in ("sin", "cos") and e > 60) else 160
    t = FUNCS[name](mpmath.mpf(x))
    s = stats[name]
    s[0] += 1
    if not mpmath.isfinite(t) or t == 0:
        continue
    if abs(t) > mpmath.mpf(sys.float_info.max):
        # Overflows a double: the correct result is inf.
        if y != float("inf"):
            s[1], s[3] = mpmath.inf, xh
        continue
    err = abs(mpmath.mpf(y) - t) / ulp_of(t)
    if err > s[1]:
        s[1], s[3] = err, xh
    if y != float(t):
        s[2] += 1
for k, (n, mx, ncr, wx) in stats.items():
    print(f"{k:4s} n={n:6d} max_err={float(mx):.4f} ulp  not_correctly_rounded={ncr} ({100*ncr/max(n,1):.3f}%)  worst_x=0x{wx}")
bad = [k for k, s in stats.items() if s[1] >= 1]
print("FAIL: >= 1 ulp in " + ", ".join(bad) if bad else "PASS: all < 1 ulp")
