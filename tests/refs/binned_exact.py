#!/usr/bin/env python3
"""Exact-rational oracle for BinnedAccumulatorF64 (bug P5).

Generates the hard-coded expected values in
`tests/test_binned_order_invariance.rs`.  Every expected value is the exact
real sum of the inputs (computed with `fractions.Fraction`, no rounding at
all) rounded ONCE to the nearest f64, ties-to-even.  CPython's
`Fraction.__float__` is `int / int` true division, which is correctly
rounded (including the subnormal range); an exact sum whose rounding
overflows raises OverflowError and is mapped to +-inf, matching IEEE-754
round-to-nearest overflow.

Inputs are emitted as raw u64 bit patterns so that the Rust test does not
depend on decimal parsing or on Python's RNG being reproducible.

Usage:  python -I tests/refs/binned_exact.py > snippet.rs
"""
import random
import struct
from fractions import Fraction

MASK64 = (1 << 64) - 1


def f2b(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def b2f(b: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", b))[0]


def exact_round(xs) -> float:
    s = sum((Fraction(x) for x in xs), Fraction(0))
    if s == 0:
        return 0.0  # documented policy: an exactly-zero sum is +0.0
    try:
        return float(s)
    except OverflowError:
        return float("inf") if s > 0 else float("-inf")


# ---------------------------------------------------------------------------
# SplitMix64 identical to cjc_repro::Rng::next_u64, and the vector generator
# used by `splitmix_vec` in the Rust test.
# ---------------------------------------------------------------------------
class SplitMix:
    def __init__(self, seed):
        self.state = seed & MASK64

    def next_u64(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)


def splitmix_vec(seed, n, wide):
    """Mirror of `splitmix_vec` in tests/test_binned_order_invariance.rs."""
    r = SplitMix(seed)
    out = []
    for _ in range(n):
        u = r.next_u64()
        v = r.next_u64()
        sign = u >> 63
        if wide:
            e = (u >> 52) & 0x7FF
            if e == 0x7FF:
                e = 0x7FE
        else:
            # exponents 1023-40 ..= 1023+40: heavy same-bin collisions
            e = 983 + ((u >> 52) & 0x7FF) % 81
        m = v >> 12
        out.append(b2f((sign << 63) | (e << 52) | m))
    return out


# ---------------------------------------------------------------------------
# Ill-conditioned sums in the spirit of Ogita, Rump & Oishi (2005),
# "Accurate sum and dot product", Algorithm 6.1 (GenDot) adapted to sums:
# the first half has exponents spread over [-b/2, b/2]; each later term is
# chosen as (random * 2^e_i) - fl(exact partial sum), with e_i decreasing
# linearly to 0, so the exact total collapses to a tiny residual and the
# condition number sum|x| / |sum x| is ~2^b.
# ---------------------------------------------------------------------------
def gensum(n, b, seed):
    rnd = random.Random(seed)
    half = n // 2
    xs = []
    for i in range(half):
        e = rnd.randint(-b // 2, b // 2) if i > 0 else b // 2
        if i == half - 1:
            e = 0
        m = 1.0 + rnd.random()
        xs.append((-1.0 if rnd.random() < 0.5 else 1.0) * (m * 2.0**e))
    for i in range(half, n):
        e = round(b / 2 * (1 - (i - half) / max(1, n - half - 1)))
        m = 1.0 + rnd.random()
        partial = sum((Fraction(x) for x in xs), Fraction(0))
        xs.append(m * 2.0**e * (1 if rnd.random() < 0.5 else -1) - float(partial))
    rnd.shuffle(xs)
    return xs


def cond(xs):
    s = sum((Fraction(x) for x in xs), Fraction(0))
    a = sum((abs(Fraction(x)) for x in xs), Fraction(0))
    if s == 0:
        return float("inf")
    try:
        return float(a / abs(s))
    except OverflowError:
        return float("inf")


MAX = 1.7976931348623157e308
B = 1.0000000000000002  # 1 + 2^-52

CASES = [
    ("probe_p5", [1.0, B, B]),
    ("probe_p5_rev", [B, B, 1.0]),
    ("ties_to_even_down", [1.0, 2.0**-53]),
    ("ties_to_even_up", [B, 2.0**-53]),
    ("ties_sticky", [1.0, 2.0**-53, 2.0**-160]),
    ("huge_cancel_small", [1e308, -1e308, 1e-300]),
    ("huge_cancel_mixed", [1e308, 1.0, -1e308, 2.0**-60, 3.0]),
    ("overflow_then_cancel", [1e308, 1e308, -1e308]),
    ("max_max_negmax", [MAX, MAX, -MAX]),
    ("overflow_pos", [MAX, MAX]),
    ("overflow_neg", [-MAX, -MAX]),
    ("max_plus_half_ulp_tie", [MAX, 2.0**970]),  # tie at overflow -> inf
    ("max_plus_below_tie", [MAX, 2.0**969]),
    ("subnormals", [5e-324, 5e-324, 5e-324, -1e-323, 2.2250738585072014e-308, -5e-324]),
    ("subnormal_to_normal", [2.225073858507201e-308, 5e-324]),
    ("cancel_to_zero", [0.1, 0.2, -0.30000000000000004, 1e-17, -1e-17]),
    ("mixed_signs", [3.0, -1e-16, 1e16, -1e16, 7e-17, -2.5, 1e-300, -0.5]),
    ("rump_like", [2.0**53, 1.0, -(2.0**53), 1.0, 2.0**-1074]),
    ("all_same_bin_many", [B] * 17 + [1.0] * 5),
    ("ten_tenths", [0.1] * 10),
]

for i, (n, b, seed) in enumerate([(40, 100, 1), (60, 300, 2), (80, 1000, 3)]):
    CASES.append((f"ogita_rump_oishi_n{n}_b{b}", gensum(n, b, seed)))


def rust_bits(xs):
    return ", ".join(f"0x{f2b(x):016x}" for x in xs)


print("// ---- generated by tests/refs/binned_exact.py (do not edit by hand) ----")
print("const ORACLE_CASES: &[(&str, &[u64], u64)] = &[")
for name, xs in CASES:
    exp = exact_round(xs)
    c = cond(xs)
    print(f"    // {name}: n={len(xs)}, cond={c:.3g}, exact-rounded = {exp!r}")
    print(f'    ("{name}", &[{rust_bits(xs)}], 0x{f2b(exp):016x}),')
print("];")

print()
print("// (seed, n, wide, exact-rounded sum bits) for splitmix_vec")
print("const SPLITMIX_ORACLE: &[(u64, usize, bool, u64)] = &[")
for seed, n, wide in [(1, 1000, False), (2, 1000, False), (3, 5000, False),
                      (4, 257, True), (5, 1000, True), (42, 2000, False)]:
    exp = exact_round(splitmix_vec(seed, n, wide))
    print(f"    ({seed}, {n}, {'true' if wide else 'false'}, 0x{f2b(exp):016x}), // {exp!r}")
print("];")
