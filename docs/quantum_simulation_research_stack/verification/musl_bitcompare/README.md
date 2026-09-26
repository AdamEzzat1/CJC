# musl bit-comparison for `cjc_repro::dmath`

Proves the musl-derived `dmath` functions are exact transcriptions: they
must produce the **same bits** as musl's own C code on every input in
`../dmath_ext_out.txt` (produced by `../dmath_ext_check/`).

## Result (2026-09-26, Windows 11, MinGW gcc)

| function | inputs | bit-identical |
|---|---|---|
| `exp_m1`, `ln_1p`, `tanh`, `atanh`, `atan`, `asin`, `acos`, `log10`, `atan2`, `hypot` | 377,655 | all |
| `sinh` | 48,736 | all but 18 on the `exp` path (below) |
| `cosh` | 48,736 | all but 2,788 on the `exp` path (below) |

Not comparable, by construction: musl's `cosh` (|x| ≥ ln 2) and `sinh`
(|x| ≥ ln `DBL_MAX`) call musl's table-driven `exp`, which `dmath` does not
use (`dmath::exp` is fdlibm, already verified by `../dmath_check/`).
`tan` (musl `__rem_pio2`), `log2`, and `pow` (FreeBSD `msun`) are checked
against mpmath only (`../dmath_ext_check.py`).

Consequence: `sinh` (1.67 ulp), `cosh` (1.24 ulp), and `atan2` (1.20 ulp)
exceed 1 ulp **because musl's algorithms do**, not because of the port.

## Reproduce

1. Download the musl sources (not vendored here) into this directory:
   `expm1 log1p tanh sinh cosh __expo2 atanh atan atan2 asin acos log10 hypot`
   from `https://git.musl-libc.org/cgit/musl/plain/src/math/<name>.c`,
   saved as `musl_<name>.c`.
2. Compile each with the shim force-included and **no FMA contraction**:

   ```
   gcc -std=gnu99 -O2 -ffp-contract=off -fno-fast-math -I . -include shim.h -c musl_<name>.c
   gcc -std=gnu99 -O2 -ffp-contract=off -I . -include shim.h -c compare.c
   gcc -O2 -c exp_stub.c
   gcc *.o -o compare -lm
   ./compare ../dmath_ext_out.txt
   ```

**Do not add `-fno-builtin`.** It routes `sqrt` to the C runtime, whose MinGW
implementation is not correctly rounded; that produced 3 spurious 1-ulp
`hypot` "mismatches" that vanish with the correctly rounded `sqrtsd`
builtin (which is what Rust's `f64::sqrt` compiles to).
