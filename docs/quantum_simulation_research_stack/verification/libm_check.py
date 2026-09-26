"""Compare a libm_bits_<platform>.txt dump against correctly rounded cos/sin.

A platform whose result differs from the correctly rounded value is a point where
two platforms *can* disagree (glibc, musl, macOS libm and MSVC CRT differ in
rounding quality). Matching correct rounding everywhere does not prove another
platform matches too; running the same emitter there and diffing does.
Usage: python libm_check.py libm_bits_windows_msvc.txt
"""
import struct, sys
from mpmath import mp, mpf, cos, sin
from mpmath.libmp import to_float

mp.prec = 300


def f(bits):
    return struct.unpack("<d", struct.pack("<Q", int(bits, 16)))[0]


def cr(fn, x):
    return to_float(fn(mpf(x))._mpf_, rnd="n")


def ulps(a, b):
    ia = struct.unpack("<q", struct.pack("<d", a))[0]
    ib = struct.unpack("<q", struct.pack("<d", b))[0]
    return abs(ia - ib)


path = sys.argv[1]
n = bad_cos = bad_sin = 0
max_ulp = 0
examples = []
by_range = {"small(|t|<=4pi)": [0, 0], "large(|t|>4pi)": [0, 0]}
with open(path) as fh:
    for line in fh:
        tb, cb, sb = line.split()
        t, c, s = f(tb), f(cb), f(sb)
        h = t / 2.0
        rc, rs = cr(cos, h), cr(sin, h)
        key = "small(|t|<=4pi)" if abs(t) <= 4 * 3.141592653589794 else "large(|t|>4pi)"
        by_range[key][0] += 1
        n += 1
        for got, ref, name in ((c, rc, "cos"), (s, rs, "sin")):
            if got != ref:
                u = ulps(got, ref)
                max_ulp = max(max_ulp, u)
                by_range[key][1] += 1
                if name == "cos":
                    bad_cos += 1
                else:
                    bad_sin += 1
                if len(examples) < 5:
                    examples.append((name, t, got, ref, u))
print(f"file: {path}")
print(f"angles: {n}; evaluations: {2 * n}")
print(f"not correctly rounded: cos {bad_cos}, sin {bad_sin}; max error {max_ulp} ulp")
for k, (cnt, bad) in by_range.items():
    print(f"  {k}: {cnt} angles, {bad} non-correctly-rounded results")
for e in examples:
    print("  example: %s(%r/2) got %r ref %r (%d ulp)" % e)
