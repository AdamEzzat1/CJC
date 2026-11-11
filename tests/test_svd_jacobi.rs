//! Accuracy, orthogonality, determinism and executor-parity tests for `svd`
//! (one-sided Jacobi / Hestenes, `crates/cjc-runtime/src/linalg.rs`).
//!
//! Bug P3 (docs/julia_comparison): the old `svd` was `eigh(AᵀA)`, which squares
//! the condition number and flushed σ_min of [[1,1],[1,1+1e-10]] to 0.
//!
//! Reference singular values come from mpmath 1.3.0 at 60 digits
//! (`tests/refs/svd_jacobi_refs.py`, `mp.svd_r` on the exact binary value of
//! each f64 entry) and are hard-coded as 17-significant-digit literals.
//!
//! Error bounds (eps = 2^-52, N = max(m, n)):
//!   * σ_i:   |σ̂_i − σ_i| ≤ 10·N·eps·σ_max  (the backward-stable bound; for a
//!            σ_i > 0 this is a relative error ≤ 10·N·eps·κ_i, κ_i = σ_max/σ_i).
//!   * ‖UΣVᵀ − A‖_F / ‖A‖_F ≤ 10·N·eps
//!   * ‖UᵀU − I‖_max ≤ 10·N·eps,  ‖VᵀV − I‖_max ≤ 10·N·eps

use cjc_runtime::Tensor;

const EPS: f64 = f64::EPSILON;

// ---- mpmath references (python -I tests/refs/svd_jacobi_refs.py) ----------
const HILBERT_2: &[f64] = &[1.2675918792439982, 6.5741454089335103e-2];
const HILBERT_3: &[f64] = &[1.4083189271236539, 1.2232706585390586e-1, 2.6873403557735218e-3];
const HILBERT_4: &[f64] = &[1.5002142800592428, 1.6914122022145004e-1, 6.7382736057607223e-3, 9.6702304022600176e-5];
const HILBERT_5: &[f64] = &[1.5670506910982308, 2.0853421861101333e-1, 1.1407491623419802e-2, 3.0589804015118543e-4, 3.2879287721758159e-6];
const HILBERT_6: &[f64] = &[1.6188998589243391, 2.4236087057520955e-1, 1.6321521319875826e-2, 6.1574835418264524e-4, 1.2570757122637029e-5, 1.082799484481101e-7];
const HILBERT_7: &[f64] = &[1.6608853389269311, 2.7192019814934523e-1, 2.1289754908327956e-2, 1.0085876107701271e-3, 2.9386368145932649e-5, 4.8567633617238215e-7, 3.4938985964246711e-9];
const HILBERT_8: &[f64] = &[1.6959389969219494, 2.9812521131693071e-1, 2.6212843578119051e-2, 1.4676881177418471e-3, 5.4369433697508963e-5, 1.2943320918741793e-6, 1.7988737460063012e-8, 1.1115389694888082e-10];
const HILBERT_9: &[f64] = &[1.7258826609018466, 3.2163312229920676e-1, 3.1038925781268329e-2, 1.9789338602159119e-3, 8.758085051460088e-5, 2.6730134106021222e-6, 5.385613349231209e-8, 6.4609052857053329e-10, 3.499685501915387e-12];
const HILBERT_10: &[f64] = &[1.7519196702651775, 3.429295484835091e-1, 3.5741816271639233e-2, 2.5308907686700286e-3, 1.2874961427637339e-4, 4.7296892931900963e-6, 1.2289677387429186e-7, 2.1474388217975422e-9, 2.2667455503810732e-11, 1.0932524334974552e-13];
const VANDERMONDE_7X4: &[f64] = &[4.0152817058687599e1, 1.4144288419086379e1, 1.9368227207377965, 1.7143818471096036];
const VANDERMONDE_4X7: &[f64] = &[4.0152817058687599e1, 1.4144288419086379e1, 1.9368227207377965, 1.7143818471096036];
const GRADED_A: [f64; 16] = [0.87111288928896, -0.12445422142208, -0.18666133413312, -0.24888177617776, -0.12445422142208, 0.01783155715584, 0.02663733173376, 0.03551644764448, -0.18666133413312, 0.02663733173376, 0.04001600160064, 0.05335466346672, -0.24888177617776, 0.03551644764448, 0.05335466346672, 0.07113956195556];
const GRADED: &[f64] = &[1.0, 1.0000000000000012e-4, 9.9999999958466645e-9, 1.0000011312843387e-12];
const RANKDEF_4X3: &[f64] = &[2.179268830147907e1, 2.2536052437322044, 0.0];
const WIDE_3X5: &[f64] = &[7.4715961765809669, 4.1253613017057612, 1.776694882215071];
const WIDE_RANKDEF_3X4: &[f64] = &[8.4112434118134453, 2.6927651711994704, 0.0];
const PROBE_P3: &[f64] = &[2.00000000005, 5.000000413576855e-11];

// ---- matrix builders --------------------------------------------------------

fn hilbert(n: usize) -> Tensor {
    let mut d = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            d.push(1.0 / ((i + j + 1) as f64));
        }
    }
    Tensor::from_vec(d, &[n, n]).unwrap()
}

fn vandermonde_7x4() -> Tensor {
    let mut d = Vec::new();
    for x in -3i64..=3 {
        for j in 0..4u32 {
            d.push(x.pow(j) as f64);
        }
    }
    Tensor::from_vec(d, &[7, 4]).unwrap()
}

fn mat(rows: &[&[f64]]) -> Tensor {
    let m = rows.len();
    let n = rows[0].len();
    let d: Vec<f64> = rows.iter().flat_map(|r| r.iter().copied()).collect();
    Tensor::from_vec(d, &[m, n]).unwrap()
}

fn probe() -> Tensor {
    mat(&[&[1.0, 1.0], &[1.0, 1.0 + 1e-10]])
}

// ---- metrics ----------------------------------------------------------------

/// Kahan-compensated dot product (test-side, independent of the runtime).
fn kdot(xs: impl Iterator<Item = f64>) -> f64 {
    let mut s = 0.0f64;
    let mut c = 0.0f64;
    for x in xs {
        let y = x - c;
        let t = s + y;
        c = (t - s) - y;
        s = t;
    }
    s
}

struct Metrics {
    recon: f64,
    u_orth: f64,
    v_orth: f64,
}

fn metrics(a: &Tensor, u: &Tensor, s: &[f64], vt: &Tensor) -> Metrics {
    let (m, n) = (a.shape()[0], a.shape()[1]);
    let k = s.len();
    assert_eq!(k, m.min(n));
    assert_eq!(u.shape(), &[m, k]);
    assert_eq!(vt.shape(), &[k, n]);
    let ad = a.to_vec();
    let ud = u.to_vec();
    let vd = vt.to_vec();
    let a_norm = kdot(ad.iter().map(|x| x * x)).sqrt();
    let mut res = Vec::with_capacity(m * n);
    for i in 0..m {
        for j in 0..n {
            let r = kdot((0..k).map(|l| ud[i * k + l] * s[l] * vd[l * n + j]));
            res.push(r - ad[i * n + j]);
        }
    }
    let r_norm = kdot(res.iter().map(|x| x * x)).sqrt();
    let recon = if a_norm == 0.0 { r_norm } else { r_norm / a_norm };
    let mut u_orth = 0.0f64;
    let mut v_orth = 0.0f64;
    for p in 0..k {
        for q in 0..k {
            let id = if p == q { 1.0 } else { 0.0 };
            let uu = kdot((0..m).map(|i| ud[i * k + p] * ud[i * k + q]));
            let vv = kdot((0..n).map(|j| vd[p * n + j] * vd[q * n + j]));
            u_orth = u_orth.max((uu - id).abs());
            v_orth = v_orth.max((vv - id).abs());
        }
    }
    Metrics { recon, u_orth, v_orth }
}

/// Full check: σ against mpmath, reconstruction, orthogonality, ordering,
/// sign convention. Returns the max relative σ error (for reporting).
fn check(name: &str, a: &Tensor, reference: &[f64]) -> f64 {
    let (m, n) = (a.shape()[0], a.shape()[1]);
    let big_n = m.max(n) as f64;
    let tol = 10.0 * big_n * EPS;
    let (u, s, vt) = a.svd().unwrap();
    assert_eq!(s.len(), reference.len(), "{name}: wrong number of singular values");
    let smax = reference[0];
    let mut worst_rel = 0.0f64;
    for (i, (&got, &want)) in s.iter().zip(reference).enumerate() {
        let abs_err = (got - want).abs();
        assert!(
            abs_err <= tol * smax,
            "{name}: sigma[{i}] = {got:e}, mpmath {want:e}, |err| = {abs_err:e} > 10*N*eps*smax = {:e}",
            tol * smax
        );
        if want != 0.0 {
            worst_rel = worst_rel.max(abs_err / want);
        }
    }
    for w in s.windows(2) {
        assert!(w[0] >= w[1], "{name}: singular values not descending: {s:?}");
    }
    let mt = metrics(a, &u, &s, &vt);
    assert!(mt.recon <= tol, "{name}: ||U S Vt - A||/||A|| = {:e} > {tol:e}", mt.recon);
    assert!(mt.u_orth <= tol, "{name}: ||UtU - I|| = {:e} > {tol:e}", mt.u_orth);
    assert!(mt.v_orth <= tol, "{name}: ||VtV - I|| = {:e} > {tol:e}", mt.v_orth);
    check_sign_convention(name, &u);
    eprintln!(
        "{name:>18}: max rel sigma err {worst_rel:.3e}  recon {:.3e}  UtU {:.3e}  VtV {:.3e}  (bound {tol:.3e})",
        mt.recon, mt.u_orth, mt.v_orth
    );
    worst_rel
}

/// Sign convention: in every column of U the first entry of largest magnitude
/// is positive.
fn check_sign_convention(name: &str, u: &Tensor) {
    let (m, k) = (u.shape()[0], u.shape()[1]);
    let ud = u.to_vec();
    for c in 0..k {
        let mut best = 0usize;
        for r in 1..m {
            if ud[r * k + c].abs() > ud[best * k + c].abs() {
                best = r;
            }
        }
        assert!(ud[best * k + c] > 0.0, "{name}: U column {c} violates sign convention");
    }
}

// ---- accuracy tests ---------------------------------------------------------

#[test]
fn svd_probe_p3_small_sigma_not_flushed() {
    let s = probe().svd().unwrap().1;
    let rel = (s[1] - PROBE_P3[1]).abs() / PROBE_P3[1];
    // Attainable: 10*N*eps*kappa = 10*2*eps*4e10 ~ 1.8e-4
    assert!(rel < 1.8e-4, "probe sigma_min = {:e}, mpmath {:e}, rel err {rel:e}", s[1], PROBE_P3[1]);
    check("probe_p3", &probe(), PROBE_P3);
}

#[test]
fn svd_hilbert_2_to_10() {
    let refs: [&[f64]; 9] = [
        HILBERT_2, HILBERT_3, HILBERT_4, HILBERT_5, HILBERT_6, HILBERT_7, HILBERT_8, HILBERT_9,
        HILBERT_10,
    ];
    for (i, r) in refs.iter().enumerate() {
        let n = i + 2;
        check(&format!("hilbert_{n}"), &hilbert(n), r);
    }
}

#[test]
fn svd_vandermonde_tall_and_wide() {
    let v = vandermonde_7x4();
    check("vandermonde_7x4", &v, VANDERMONDE_7X4);
    check("vandermonde_4x7", &v.transpose(), VANDERMONDE_4X7);
}

#[test]
fn svd_graded_rotated() {
    let a = Tensor::from_vec(GRADED_A.to_vec(), &[4, 4]).unwrap();
    check("graded", &a, GRADED);
    // The smallest singular value (1e-12) must be resolved, not flushed.
    let s = a.svd().unwrap().1;
    assert!(s[3] > 0.5e-12 && s[3] < 2e-12, "graded sigma_min = {:e}", s[3]);
}

#[test]
fn svd_rank_deficient_tall() {
    let a = mat(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 9.0], &[7.0, 8.0, 15.0], &[2.0, -1.0, 1.0]]);
    check("rankdef_4x3", &a, RANKDEF_4X3);
}

#[test]
fn svd_wide_full_rank_and_rank_deficient() {
    let a = mat(&[&[2.0, -1.0, 0.0, 3.0, 1.0], &[1.0, 4.0, -2.0, 0.0, 5.0], &[0.0, 3.0, 1.0, -1.0, 2.0]]);
    check("wide_3x5", &a, WIDE_3X5);
    let b = mat(&[&[1.0, 2.0, 0.0, -1.0], &[3.0, -1.0, 2.0, 4.0], &[-2.0, 3.0, -2.0, -5.0]]);
    check("wide_rankdef_3x4", &b, WIDE_RANKDEF_3X4);
}

#[test]
fn svd_degenerate_shapes() {
    // 1x1 negative: sigma = 3, U = [1], Vt = [-1].
    let (u, s, vt) = mat(&[&[-3.0]]).svd().unwrap();
    assert_eq!(s, vec![3.0]);
    assert_eq!(u.to_vec(), vec![1.0]);
    assert_eq!(vt.to_vec(), vec![-1.0]);
    // 1x2 row: sigma = 5 (exact 3-4-5).
    check("row_1x2", &mat(&[&[3.0, 4.0]]), &[5.0]);
    // 3x1 column: sigma = 3.
    check("col_3x1", &mat(&[&[1.0], &[-2.0], &[2.0]]), &[3.0]);
    // Zero matrices: sigma = 0, U and V orthonormal anyway.
    check("zero_3x2", &mat(&[&[0.0, 0.0], &[0.0, 0.0], &[0.0, 0.0]]), &[0.0, 0.0]);
    check("zero_2x3", &mat(&[&[0.0, 0.0, 0.0], &[0.0, 0.0, 0.0]]), &[0.0, 0.0]);
    let (u, s, vt) = mat(&[&[0.0, 0.0], &[0.0, 0.0], &[0.0, 0.0]]).svd().unwrap();
    assert_eq!(s, vec![0.0, 0.0]);
    assert_eq!(u.to_vec(), vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    assert_eq!(vt.to_vec(), vec![1.0, 0.0, 0.0, 1.0]);
}

#[test]
fn svd_identity_ties_are_deterministic() {
    // All singular values tie at 1: U = V = I exactly (stable tie-break by index).
    let a = mat(&[&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0]]);
    let (u, s, vt) = a.svd().unwrap();
    assert_eq!(s, vec![1.0, 1.0, 1.0]);
    assert_eq!(u.to_vec(), a.to_vec());
    assert_eq!(vt.to_vec(), a.to_vec());
    // Permuted diagonal: sorted descending, ties broken by column index.
    let d = mat(&[&[2.0, 0.0, 0.0], &[0.0, 5.0, 0.0], &[0.0, 0.0, 2.0]]);
    let (u, s, vt) = d.svd().unwrap();
    assert_eq!(s, vec![5.0, 2.0, 2.0]);
    assert_eq!(u.to_vec(), vec![0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
    assert_eq!(vt.to_vec(), vec![0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
}

#[test]
fn svd_extreme_scaling() {
    // Entries near the overflow / underflow thresholds: the power-of-two
    // pre-scaling keeps the sums of squares finite.
    for scale in [1e300, 1e-300] {
        let a = mat(&[&[3.0 * scale, 0.0], &[0.0, 4.0 * scale]]);
        let s = a.svd().unwrap().1;
        assert_eq!(s, vec![4.0 * scale, 3.0 * scale], "scale {scale:e}");
    }
}

#[test]
fn svd_is_bit_deterministic() {
    let a = hilbert(8);
    let (u1, s1, v1) = a.svd().unwrap();
    for _ in 0..5 {
        let (u2, s2, v2) = a.svd().unwrap();
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&u1.to_vec()), bits(&u2.to_vec()));
        assert_eq!(bits(&s1), bits(&s2));
        assert_eq!(bits(&v1.to_vec()), bits(&v2.to_vec()));
    }
}

// ---- pinv now that small sigma survive -----------------------------------------

#[test]
fn pinv_probe_matches_exact_inverse() {
    // A = [[1,1],[1,c]], det = c - 1 (exact in f64 by Sterbenz), A^-1 = [[c,-1],[-1,1]]/det.
    let c = 1.0 + 1e-10;
    let det = c - 1.0;
    let inv = [c / det, -1.0 / det, -1.0 / det, 1.0 / det];
    let p = probe().pinv().unwrap().to_vec();
    for i in 0..4 {
        let rel = (p[i] - inv[i]).abs() / inv[i].abs();
        // kappa(A) ~ 8e10, so ~kappa*eps ~ 2e-5 is attainable.
        assert!(rel < 1e-4, "pinv[{i}] = {:e}, exact {:e}, rel {rel:e}", p[i], inv[i]);
    }
}

#[test]
fn pinv_rank_deficient_drops_null_direction() {
    // For the rank-2 4x3 matrix, the default tolerance must drop the round-off
    // sigma_3: A A+ A = A and A+ A is the orthogonal projector of rank 2.
    let a = mat(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 9.0], &[7.0, 8.0, 15.0], &[2.0, -1.0, 1.0]]);
    let p = a.pinv().unwrap();
    let apa = a.matmul(&p).unwrap().matmul(&a).unwrap().to_vec();
    for (x, y) in apa.iter().zip(a.to_vec()) {
        assert!((x - y).abs() < 1e-12 * 30.0, "A A+ A != A: {x} vs {y}");
    }
    let pa = p.matmul(&a).unwrap().to_vec();
    let trace = pa[0] + pa[4] + pa[8];
    assert!((trace - 2.0).abs() < 1e-12, "trace(A+ A) = {trace}, want 2 (rank)");
}

// ---- executor parity (cjc-eval vs cjc-mir-exec) ---------------------------------

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
    assert_eq!(eval_out, mir_out, "Parity mismatch!\nEval: {eval_out:?}\nMIR:  {mir_out:?}");
    eval_out
}

const PARITY_SRC: &str = r#"
fn show(a: Any) {
    match svd(a) {
        (u, s, vt) => { print(u); print(s); print(vt); }
    };
    print(pinv(a));
}
fn main() {
    show(Tensor.from_vec([1.0, 1.0, 1.0, 1.0 + 1e-10], [2, 2]));
    show(Tensor.from_vec([2.0, -1.0, 0.0, 3.0, 1.0, 1.0, 4.0, -2.0, 0.0, 5.0, 0.0, 3.0, 1.0, -1.0, 2.0], [3, 5]));
    show(Tensor.from_vec([1.0, 2.0, 3.0, 4.0, 5.0, 9.0, 7.0, 8.0, 15.0, 2.0, -1.0, 1.0], [4, 3]));
    show(Tensor.from_vec([0.0, 0.0, 0.0, 0.0], [2, 2]));
    print(matrix_rank(Tensor.from_vec([1.0, 2.0, 3.0, 4.0, 5.0, 9.0, 7.0, 8.0, 15.0, 2.0, -1.0, 1.0], [4, 3])));
}
"#;

#[test]
fn svd_parity_eval_vs_mir() {
    let out = assert_parity(PARITY_SRC);
    assert!(out.len() >= 17, "unexpected output: {out:?}");
    // First singular-value line is the probe; sigma_min must not be 0.
    assert!(!out[1].contains("0.0]"), "probe sigma_min flushed: {}", out[1]);
}

#[test]
fn svd_parity_output_is_deterministic() {
    let a = assert_parity(PARITY_SRC);
    let b = assert_parity(PARITY_SRC);
    assert_eq!(a, b);
}
