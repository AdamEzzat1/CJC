"""Reference values for tests/test_erf_fdlibm.rs.

Generated with mpmath 1.3.0 at 60 decimal digits on the EXACT binary value of
each f64 argument, printed as 17-significant-digit literals. Re-run with:

    python -I tests/refs/erf_fdlibm_refs.py

Argument sets (each is exactly representable / built identically in Rust):
  * GRID: x = i / 5.0 for i in -30..=30 (one correctly rounded division) -> erf, erfc
  * TAIL: x = k / 2.0 for k in 1..=52 (0.5 .. 26, exact) -> erfc
  * SPECIAL: fdlibm interval boundaries and tiny arguments -> erf, erfc
  * NCDF: x = i / 2.0 for i in -76..=16 (-38 .. 8, exact) -> normal_cdf = erfc(-x/sqrt2)/2
"""
import mpmath as mp

mp.mp.dps = 60


def lit(x):
    if x == 0:
        return "0.0"
    s = mp.nstr(x, 17, min_fixed=1, max_fixed=0)
    if "e" not in s and "." not in s:
        s += ".0"
    return s.replace("e-0", "e-").replace("e+", "e")


def pairs(name, xs, f):
    print(f"const {name}: &[(f64, f64)] = &[")
    for x in xs:
        print(f"    ({x!r}, {lit(f(mp.mpf(x)))}),")
    print("];")


grid = [i / 5.0 for i in range(-30, 31)]
pairs("ERF_GRID", grid, mp.erf)
pairs("ERFC_GRID", grid, mp.erfc)
tail = [k / 2.0 for k in range(1, 53)]
pairs("ERFC_TAIL", tail, mp.erfc)
special = [5e-324, 1e-300, 1e-20, -1e-20, 1e-10, 3.725290298461914e-09, 0.84375, -0.84375,
           0.84374999999999989, 1.25, 1.2499999999999998, 2.857142857142857, 2.8571428571428577,
           6.0, -6.0, 5.9999999999999991, 0.01, -0.3, 27.0, -27.0]
pairs("ERF_SPECIAL", special, mp.erf)
pairs("ERFC_SPECIAL", special, mp.erfc)
ncdf = [i / 2.0 for i in range(-76, 17)]
pairs("NCDF", ncdf, lambda x: mp.ncdf(x))
