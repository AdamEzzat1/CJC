//! Bug P0-6 regression: `Rng::next_normal_f64` (and therefore `randn`,
//! `Tensor.randn`, every Box–Muller sampler) must use the deterministic
//! `cjc_repro::dmath` elementary functions, not the platform libm, so normal
//! draws are bit-identical on Windows / Linux / macOS.

use cjc_repro::{dmath, Rng};

const N_DRAWS: usize = 10_000;

/// FNV-1a 64 over the little-endian bit patterns.
fn fnv1a(xs: &[f64]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for x in xs {
        for b in x.to_bits().to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

/// Box–Muller reference built from the dmath kernels, consuming the same
/// uniform stream as `Rng::next_normal_f64`.
fn reference_dmath(seed: u64, n: usize) -> Vec<f64> {
    let mut r = Rng::seeded(seed);
    (0..n)
        .map(|_| {
            let u1 = r.next_f64();
            let u2 = r.next_f64();
            (-2.0 * dmath::ln(u1)).sqrt() * dmath::cos(2.0 * std::f64::consts::PI * u2)
        })
        .collect()
}

/// The pre-fix formula (platform libm), kept here only to measure how many
/// draws the fix changes on the machine running the test.
fn reference_libm(seed: u64, n: usize) -> Vec<f64> {
    let mut r = Rng::seeded(seed);
    (0..n)
        .map(|_| {
            let u1 = r.next_f64();
            let u2 = r.next_f64();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        })
        .collect()
}

fn draws(seed: u64, n: usize) -> Vec<f64> {
    let mut r = Rng::seeded(seed);
    (0..n).map(|_| r.next_normal_f64()).collect()
}

/// Golden: FNV-1a of the first 10^4 `next_normal_f64` draws for seed 42,
/// computed from the dmath Box–Muller formula.  dmath is pure IEEE
/// +,-,*,/,sqrt (no FMA), so this value is platform-independent.
const GOLDEN_RANDN_SEED42_FNV: u64 = 0x857ac02faffc60d4;

#[test]
fn next_normal_matches_dmath_box_muller() {
    let got = draws(42, N_DRAWS);
    let want = reference_dmath(42, N_DRAWS);
    let diff = got
        .iter()
        .zip(&want)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(diff, 0, "{diff} of {N_DRAWS} draws differ from the dmath reference");
}

#[test]
fn next_normal_golden_hash_seed42() {
    let h = fnv1a(&draws(42, N_DRAWS));
    let r = fnv1a(&reference_dmath(42, N_DRAWS));
    assert_eq!(h, r, "next_normal_f64 stream hash 0x{h:016x} != dmath reference 0x{r:016x}");
    assert_eq!(h, GOLDEN_RANDN_SEED42_FNV, "golden hash: got 0x{h:016x}");
}

#[test]
fn next_normal_f32_is_narrowed_f64() {
    let mut a = Rng::seeded(9);
    let mut b = Rng::seeded(9);
    for _ in 0..1000 {
        assert_eq!(a.next_normal_f32().to_bits(), (b.next_normal_f64() as f32).to_bits());
    }
}

/// Diagnostic, always passes: how many of the 10^4 draws the libm -> dmath
/// switch changes on THIS machine.  Zero here does not prove cross-platform
/// equality of the old code; it only says this libm agreed with dmath.
#[test]
fn report_libm_vs_dmath_bit_differences() {
    let a = reference_libm(42, N_DRAWS);
    let b = reference_dmath(42, N_DRAWS);
    let diff = a.iter().zip(&b).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
    let maxulp = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x.to_bits() as i64 - y.to_bits() as i64).unsigned_abs())
        .max()
        .unwrap_or(0);
    println!(
        "libm vs dmath (seed 42, {N_DRAWS} draws): {diff} differ, max {maxulp} ulp; \
         libm hash 0x{:016x}, dmath hash 0x{:016x}",
        fnv1a(&a),
        fnv1a(&b)
    );
}

// ---------------------------------------------------------------------------
// Both executors, through .cjcl
// ---------------------------------------------------------------------------

fn run_eval(src: &str, seed: u64) -> Vec<String> {
    let (program, _diag) = cjc_parser::parse_source(src);
    let mut interp = cjc_eval::Interpreter::new(seed);
    let _ = interp.exec(&program);
    interp.output
}

fn run_mir(src: &str, seed: u64) -> Vec<String> {
    let (program, _diag) = cjc_parser::parse_source(src);
    let (_, executor) = cjc_mir_exec::run_program_with_executor(&program, seed)
        .unwrap_or_else(|e| panic!("MIR-exec failed: {e}"));
    executor.output
}

fn assert_parity(src: &str) -> Vec<String> {
    let eval_out = run_eval(src, 42);
    let mir_out = run_mir(src, 42);
    assert_eq!(
        eval_out, mir_out,
        "Parity mismatch!\nEval: {eval_out:?}\nMIR:  {mir_out:?}"
    );
    eval_out
}

/// Every one of 500 `Tensor.randn` draws, printed from both executors,
/// must round-trip to the dmath Box–Muller reference bit pattern.
#[test]
fn cjcl_randn_parity_and_dmath() {
    let out = assert_parity(
        r#"
fn main() {
    let t = Tensor.randn([500]);
    let mut i = 0;
    while i < 500 {
        print(t.get([i]));
        i = i + 1;
    }
}
"#,
    );
    let want = reference_dmath(42, 500);
    assert_eq!(out.len(), 500, "output: {out:?}");
    let diff: Vec<usize> = (0..500)
        .filter(|&k| out[k].parse::<f64>().unwrap().to_bits() != want[k].to_bits())
        .collect();
    assert!(diff.is_empty(), "{} of 500 cjcl randn draws differ from dmath (first idx {:?})", diff.len(), diff.first());
}
