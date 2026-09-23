//! One safe function per kernel: the Bruchion kernel when the switch is on AND the
//! operation is in the routing set ([`routes`]), the Rust body otherwise. The Rust bodies are the *reference* — each is the arithmetic of the CJC
//! function it stands for (or that function itself, for the ones that call here) — and
//! the `parity` tests assert kernel and reference agree bit for bit.
//!
//! Every function is total on its domain and allocates nothing on either path. Lengths
//! are the caller's contract: the kernel reads `n` doubles where `n` is the shorter of
//! the slices involved, so a length mismatch is a silent truncation here and an error
//! in the public function that called (`mse_loss` checks before calling).

use cjc_repro::KahanAccumulatorF64;

/// Whether calls route to the Bruchion kernels: the cargo feature *and* the runtime
/// policy switch, so a build with the feature still runs the Rust bodies until asked.
#[inline]
pub fn enabled() -> bool {
    cfg!(feature = "bruchion-kernels") && crate::runtime_policy::get().bruchion_kernels
}

/// The operations the switch can route, one bit each. The set the policy holds
/// (`RuntimePolicy::bruchion_kernel_ops`) says which of them the switch actually sends to
/// the pack; the default set is [`KernelOps::MEASURED`], and the reason for each
/// membership is the CJC record (`bench_results/bruchion_kernels`), never a guess.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum KernelOp {
    /// `y = a x + y`. Slower 1.3–1.4x on every record: two SSE2 lanes against the Rust
    /// loop's two lanes, with gcc 8's split loads. Off by default.
    Axpy = 1 << 0,
    /// The Kahan dot. Inside the A/A band on every record. On.
    DotKahan = 1 << 1,
    /// `max(0, x)`. Slower about 1.3x on every record (the emitter's mask blend against
    /// `maxpd`). Off.
    Relu = 1 << 2,
    /// The Kahan mean of squares. Inside the A/A band. On.
    Mse = 1 << 3,
    /// The fused loss and gradient. Faster 1.3–1.4x, within A/A. On.
    MseGrad = 1 << 4,
    /// The scaled gradient the fused graph node's backward takes. Not measured on its
    /// own (the graph row that carries it is inside A/A); an elementwise loop like `axpy`,
    /// which loses. Off until measured.
    MseGradScaled = 1 << 5,
    /// The Kahan matmul. Faster about 1.4–1.5x at 64x17x33 on six records, inside A/A at
    /// 128^3. On.
    Matmul = 1 << 6,
    /// `TiledMatmul`'s order. Slower 1.35–1.6x (two lanes against AVX2's four). Off.
    MatmulTiled = 1 << 7,
    /// The Adam step. Slower 1.6–1.7x with the hardware root. Off.
    AdamStep = 1 << 8,
    /// `powi` with a runtime exponent. Not timed, and on MSVC targets routing it CHANGES
    /// bits relative to `f64::powi` (which is the CRT `pow` there) — see
    /// [`f64_powi_is_binary_exponentiation`]. Off, so the default set never moves a bit.
    Powi = 1 << 9,
    /// The heat residual without the gradient. Not timed (nothing routed calls it). Off.
    Heat1dResidual = 1 << 10,
    /// The heat residual and gradient. Its own row is inside A/A or a small win — but the
    /// row compares the kernel with the dispatch twin, a bit reference, not the trainer's
    /// loop: `piml_heat_1d_train` end to end is 2.47x SLOWER routed (record 8), because
    /// the trainer's inlined loop is several times faster than the twin. Off: the
    /// end-to-end row is the one that counts when there is one.
    Heat1dResidualGrad = 1 << 11,
}

impl KernelOp {
    pub const EVERY: [KernelOp; 12] = [
        KernelOp::Axpy, KernelOp::DotKahan, KernelOp::Relu, KernelOp::Mse, KernelOp::MseGrad,
        KernelOp::MseGradScaled, KernelOp::Matmul, KernelOp::MatmulTiled, KernelOp::AdamStep,
        KernelOp::Powi, KernelOp::Heat1dResidual, KernelOp::Heat1dResidualGrad,
    ];

    #[inline]
    pub fn bit(self) -> u16 {
        self as u16
    }

    /// The name the record and the provenance print.
    pub fn name(self) -> &'static str {
        match self {
            KernelOp::Axpy => "axpy",
            KernelOp::DotKahan => "dot_kahan",
            KernelOp::Relu => "relu",
            KernelOp::Mse => "mse",
            KernelOp::MseGrad => "mse_grad",
            KernelOp::MseGradScaled => "mse_grad_scaled",
            KernelOp::Matmul => "matmul",
            KernelOp::MatmulTiled => "matmul_tiled",
            KernelOp::AdamStep => "adam_step",
            KernelOp::Powi => "powi",
            KernelOp::Heat1dResidual => "heat1d_residual",
            KernelOp::Heat1dResidualGrad => "heat1d_residual_grad",
        }
    }
}

/// A set of [`KernelOp`]s, the value `RuntimePolicy::bruchion_kernel_ops` holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KernelOps(pub u16);

impl KernelOps {
    pub const NONE: KernelOps = KernelOps(0);
    pub const ALL: KernelOps = KernelOps((1 << 12) - 1);
    /// The default: the operations whose record shows no loss — a win or the A/A band,
    /// judged on the end-to-end row when one exists — and whose routing moves no bit on
    /// any target.
    pub const MEASURED: KernelOps = KernelOps(
        KernelOp::DotKahan as u16 | KernelOp::Mse as u16 | KernelOp::MseGrad as u16
        | KernelOp::Matmul as u16,
    );

    #[inline]
    pub fn contains(self, op: KernelOp) -> bool {
        self.0 & op.bit() != 0
    }
    pub fn with(self, op: KernelOp) -> KernelOps {
        KernelOps(self.0 | op.bit())
    }
    pub fn without(self, op: KernelOp) -> KernelOps {
        KernelOps(self.0 & !op.bit())
    }
    /// The members' names, in bit order, comma-separated; `(none)` for the empty set.
    pub fn names(self) -> String {
        let v: Vec<&str> = KernelOp::EVERY.iter().filter(|&&o| self.contains(o)).map(|o| o.name()).collect();
        if v.is_empty() { "(none)".into() } else { v.join(", ") }
    }
}

/// Whether `op` goes to the pack right now: the feature, the switch, and the set.
#[inline]
pub fn routes(op: KernelOp) -> bool {
    if !cfg!(feature = "bruchion-kernels") {
        return false;
    }
    let p = crate::runtime_policy::get();
    p.bruchion_kernels && KernelOps(p.bruchion_kernel_ops).contains(op)
}

/// The switch on with every operation in the set, or the switch off with the default
/// set restored — what the parity tests and the benchmark's kernel arm want, so that a
/// kernel the default set leaves out is still held to its bits and still measured.
pub fn route_everything(on: bool) {
    crate::runtime_policy::set_bruchion_kernel_ops(if on { KernelOps::ALL.0 } else { KernelOps::MEASURED.0 });
    crate::runtime_policy::set_bruchion_kernels(on);
}

// ── axpy ───────────────────────────────────────────────────────────────────

/// `y[i] = a * x[i] + y[i]` — separate multiply and add, never fused (the contract
/// `Tensor::fused_axpy` and `tensor_simd::simd_axpy` state).
pub fn axpy(a: f64, x: &[f64], y: &mut [f64]) {
    let n = x.len().min(y.len());
    #[cfg(feature = "bruchion-kernels")]
    if routes(KernelOp::Axpy) {
        axpy_kernel(a, &x[..n], &mut y[..n]);
        return;
    }
    axpy_fallback(a, &x[..n], &mut y[..n]);
}

pub fn axpy_fallback(a: f64, x: &[f64], y: &mut [f64]) {
    for i in 0..x.len().min(y.len()) {
        y[i] = a * x[i] + y[i];
    }
}

#[cfg(feature = "bruchion-kernels")]
pub fn axpy_kernel(a: f64, x: &[f64], y: &mut [f64]) {
    let n = x.len().min(y.len());
    // SAFETY: `x` and `y` are live slices of at least `n` elements; the kernel reads `n`
    // doubles of `x`, reads and writes `n` doubles of `y`, and touches nothing else
    // (`@no_alloc @no_os`, proven by bruchionc and recorded in metadata.json).
    unsafe { super::ffi::cjc_axpy_f64(n as i64, a, x.as_ptr(), y.as_mut_ptr()) }
}

// ── dot, Kahan ─────────────────────────────────────────────────────────────

/// `Σ x[i]·y[i]` through `KahanAccumulatorF64`, products in index order.
pub fn dot_kahan(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    #[cfg(feature = "bruchion-kernels")]
    if routes(KernelOp::DotKahan) {
        return dot_kahan_kernel(&x[..n], &y[..n]);
    }
    dot_kahan_fallback(&x[..n], &y[..n])
}

pub fn dot_kahan_fallback(x: &[f64], y: &[f64]) -> f64 {
    let mut acc = KahanAccumulatorF64::new();
    for i in 0..x.len().min(y.len()) {
        acc.add(x[i] * y[i]);
    }
    acc.finalize()
}

#[cfg(feature = "bruchion-kernels")]
pub fn dot_kahan_kernel(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    // SAFETY: two live slices of at least `n` elements, read only.
    unsafe { super::ffi::cjc_dot_kahan_f64(n as i64, x.as_ptr(), y.as_ptr()) }
}

// ── relu ───────────────────────────────────────────────────────────────────

/// `out[i] = if x[i] > 0.0 { x[i] } else { 0.0 }` — `kernel::relu_raw`'s comparison,
/// so `-0.0` and NaN both map to `+0.0`.
pub fn relu(data: &[f64], out: &mut [f64]) {
    let n = data.len().min(out.len());
    #[cfg(feature = "bruchion-kernels")]
    if routes(KernelOp::Relu) {
        relu_kernel(&data[..n], &mut out[..n]);
        return;
    }
    relu_fallback(&data[..n], &mut out[..n]);
}

pub fn relu_fallback(data: &[f64], out: &mut [f64]) {
    for (o, &x) in out.iter_mut().zip(data.iter()) {
        *o = if x > 0.0 { x } else { 0.0 };
    }
}

#[cfg(feature = "bruchion-kernels")]
pub fn relu_kernel(data: &[f64], out: &mut [f64]) {
    let n = data.len().min(out.len());
    // SAFETY: `data` read and `out` written for `n` elements, both live.
    unsafe { super::ffi::cjc_relu_f64(n as i64, data.as_ptr(), out.as_mut_ptr()) }
}

// ── mse ────────────────────────────────────────────────────────────────────

/// `Σ (pred−target)² / n` with the squares Kahan-summed — `ml::mse_loss` after its
/// argument checks; `0.0` for an empty input (the caller refuses that case first).
pub fn mse(pred: &[f64], target: &[f64]) -> f64 {
    let n = pred.len().min(target.len());
    #[cfg(feature = "bruchion-kernels")]
    if routes(KernelOp::Mse) {
        return mse_kernel(&pred[..n], &target[..n]);
    }
    mse_fallback(&pred[..n], &target[..n])
}

pub fn mse_fallback(pred: &[f64], target: &[f64]) -> f64 {
    let n = pred.len().min(target.len());
    if n == 0 {
        return 0.0;
    }
    let mut acc = KahanAccumulatorF64::new();
    for i in 0..n {
        let d = pred[i] - target[i];
        acc.add(d * d);
    }
    acc.finalize() / n as f64
}

#[cfg(feature = "bruchion-kernels")]
pub fn mse_kernel(pred: &[f64], target: &[f64]) -> f64 {
    let n = pred.len().min(target.len());
    // SAFETY: two live slices of at least `n` elements, read only.
    unsafe { super::ffi::cjc_mse_f64(n as i64, pred.as_ptr(), target.as_ptr()) }
}

/// The fused loss AND gradient of the graph's `mean((pred − target)²)`, in the bits
/// `GradGraph`'s `sub → mul(diff, diff) → mean → backward` produces: the loss is the
/// BINNED sum of the squares over `n` (`Tensor::mean` is `binned_sum_f64 / n`, not
/// `mse`'s Kahan), and `grad[i] = h + h` with `h = (1 / n) * d` — the backward of
/// `mul(diff, diff)` accumulates `grad_val * d` once per operand, `grad_val = 1 / n`.
/// That is not `2 * d / n` (the two differ in the last bit; the tests pin a witness).
/// One pass into the caller's `grad` where the graph builds six buffers —
/// `ml::mse_loss_grad`. `0.0` and no writes for an empty input.
pub fn mse_grad(pred: &[f64], target: &[f64], grad: &mut [f64]) -> f64 {
    let n = pred.len().min(target.len()).min(grad.len());
    #[cfg(feature = "bruchion-kernels")]
    if routes(KernelOp::MseGrad) {
        return mse_grad_kernel(&pred[..n], &target[..n], &mut grad[..n]);
    }
    mse_grad_fallback(&pred[..n], &target[..n], &mut grad[..n])
}

pub fn mse_grad_fallback(pred: &[f64], target: &[f64], grad: &mut [f64]) -> f64 {
    let n = pred.len().min(target.len()).min(grad.len());
    if n == 0 {
        return 0.0;
    }
    let inv = 1.0 / n as f64;
    let mut acc = crate::accumulator::BinnedAccumulatorF64::new();
    for i in 0..n {
        let d = pred[i] - target[i];
        acc.add(d * d);
        let h = inv * d;
        grad[i] = h + h;
    }
    acc.finalize() / n as f64
}

#[cfg(feature = "bruchion-kernels")]
pub fn mse_grad_kernel(pred: &[f64], target: &[f64], grad: &mut [f64]) -> f64 {
    let n = pred.len().min(target.len()).min(grad.len());
    // SAFETY: three live slices of at least `n` elements; `grad` is the only one written.
    unsafe { super::ffi::cjc_mse_grad_f64(n as i64, pred.as_ptr(), target.as_ptr(), grad.as_mut_ptr()) }
}

// ── mse grad, scaled — the fused graph node's backward ─────────────────────

/// The gradient of the graph's `mean((pred − target)²)` with respect to its materialized
/// difference, for an arbitrary upstream gradient `g`: `grad[i] = h + h` with
/// `h = (g / n) * diff[i]` — the bits `GradGraph`'s backward produces through
/// `mean → mul(diff, diff)`: the Mean arm divides the upstream scalar by `n` once, the Mul
/// arm multiplies that by the other operand once per operand, and the two accumulate.
/// With `g = 1` it is [`mse_grad`]'s gradient. `GradOp::MseMean`'s backward calls here;
/// no loss, the graph has it from the forward pass. Writes nothing for an empty input.
pub fn mse_grad_scaled(g: f64, diff: &[f64], grad: &mut [f64]) {
    let n = diff.len().min(grad.len());
    #[cfg(feature = "bruchion-kernels")]
    if routes(KernelOp::MseGradScaled) {
        mse_grad_scaled_kernel(g, &diff[..n], &mut grad[..n]);
        return;
    }
    mse_grad_scaled_fallback(g, &diff[..n], &mut grad[..n]);
}

pub fn mse_grad_scaled_fallback(g: f64, diff: &[f64], grad: &mut [f64]) {
    let n = diff.len().min(grad.len());
    if n == 0 {
        return;
    }
    let s = g / n as f64;
    for i in 0..n {
        let h = s * diff[i];
        grad[i] = h + h;
    }
}

#[cfg(feature = "bruchion-kernels")]
pub fn mse_grad_scaled_kernel(g: f64, diff: &[f64], grad: &mut [f64]) {
    let n = diff.len().min(grad.len());
    // SAFETY: two live slices of at least `n` elements; `grad` is the only one written.
    unsafe { super::ffi::cjc_mse_grad_scaled_f64(n as i64, g, diff.as_ptr(), grad.as_mut_ptr()) }
}

// ── matmul ─────────────────────────────────────────────────────────────────

/// `C[m×n] = A[m×k]·B[k×n]`, row-major, one `KahanAccumulatorF64` per output with the
/// products added in `p` order — `kernel::matmul_raw`.
pub fn matmul(a: &[f64], b: &[f64], c: &mut [f64], m: usize, k: usize, n: usize) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(c.len(), m * n);
    #[cfg(feature = "bruchion-kernels")]
    if routes(KernelOp::Matmul) {
        matmul_kernel(a, b, c, m, k, n);
        return;
    }
    matmul_fallback(a, b, c, m, k, n);
}

pub fn matmul_fallback(a: &[f64], b: &[f64], c: &mut [f64], m: usize, k: usize, n: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut acc = KahanAccumulatorF64::new();
            for p in 0..k {
                acc.add(a[i * k + p] * b[p * n + j]);
            }
            c[i * n + j] = acc.finalize();
        }
    }
}

#[cfg(feature = "bruchion-kernels")]
pub fn matmul_kernel(a: &[f64], b: &[f64], c: &mut [f64], m: usize, k: usize, n: usize) {
    assert!(a.len() >= m * k && b.len() >= k * n && c.len() >= m * n, "matmul: buffer shorter than its shape");
    // SAFETY: the three buffers hold at least m·k, k·n and m·n doubles (asserted).
    unsafe { super::ffi::cjc_matmul_f64(m as i64, k as i64, n as i64, a.as_ptr(), b.as_ptr(), c.as_mut_ptr()) }
}

// ── matmul, TiledMatmul's order ────────────────────────────────────────────

/// `C[m×n] = A[m×k]·B[k×n]` in `TiledMatmul`'s bits — the order `Tensor::matmul` takes
/// when any dimension is 64 or more: each output the plain sequential sum of its products
/// in ascending `p`, no compensation, no fusion. NOT `matmul`'s Kahan value (the parity
/// test asserts the two part at 128^3). The fallback is the engine itself, copied into
/// `c`; the kernel writes `c` directly.
pub fn matmul_tiled(a: &[f64], b: &[f64], c: &mut [f64], m: usize, k: usize, n: usize) {
    debug_assert!(a.len() >= m * k && b.len() >= k * n && c.len() >= m * n);
    #[cfg(feature = "bruchion-kernels")]
    if routes(KernelOp::MatmulTiled) {
        matmul_tiled_kernel(a, b, c, m, k, n);
        return;
    }
    matmul_tiled_fallback(a, b, c, m, k, n);
}

pub fn matmul_tiled_fallback(a: &[f64], b: &[f64], c: &mut [f64], m: usize, k: usize, n: usize) {
    let r = crate::tensor_tiled::TiledMatmul::new().matmul(&a[..m * k], m, k, &b[..k * n], n);
    c[..m * n].copy_from_slice(&r);
}

#[cfg(feature = "bruchion-kernels")]
pub fn matmul_tiled_kernel(a: &[f64], b: &[f64], c: &mut [f64], m: usize, k: usize, n: usize) {
    assert!(a.len() >= m * k && b.len() >= k * n && c.len() >= m * n, "matmul_tiled: buffer shorter than its shape");
    // SAFETY: the three buffers hold at least m·k, k·n and m·n doubles (asserted).
    unsafe { super::ffi::cjc_matmul_tiled_f64(m as i64, k as i64, n as i64, a.as_ptr(), b.as_ptr(), c.as_mut_ptr()) }
}

// ── adam ───────────────────────────────────────────────────────────────────

/// One Adam update over every parameter — `ml::adam_step`'s arithmetic, with the bias
/// corrections `1 − β^t` computed HERE through `powf` (libm stays on this side, so the
/// kernel is libm-free) and passed to the kernel as values.
#[allow(clippy::too_many_arguments)]
pub fn adam_step_raw(params: &mut [f64], grads: &[f64], m: &mut [f64], v: &mut [f64], lr: f64, beta1: f64, beta2: f64, eps: f64, t: f64) {
    let bc1 = 1.0 - beta1.powf(t);
    let bc2 = 1.0 - beta2.powf(t);
    // Routable, but not in the default set: the records at abe6b21..28c104c had the kernel
    // 15-20x slower than this loop because the pack took its square root from the libm-free
    // `std/math`, a software root; with the hardware instruction (`sqrt_f64`: `sqrtsd`,
    // correctly rounded by IEEE 754, so the same bits as `f64::sqrt` here and on every
    // conforming target — the parity test below is the check) it is 1.6-1.7x slower, the
    // elementwise codegen gap, so `KernelOps::MEASURED` leaves it out.
    #[cfg(feature = "bruchion-kernels")]
    if routes(KernelOp::AdamStep) {
        adam_step_kernel(params, grads, m, v, lr, beta1, beta2, eps, bc1, bc2);
        return;
    }
    adam_step_fallback(params, grads, m, v, lr, beta1, beta2, eps, bc1, bc2);
}

#[allow(clippy::too_many_arguments)]
pub fn adam_step_fallback(params: &mut [f64], grads: &[f64], m: &mut [f64], v: &mut [f64], lr: f64, beta1: f64, beta2: f64, eps: f64, bc1: f64, bc2: f64) {
    let n = params.len().min(grads.len()).min(m.len()).min(v.len());
    for i in 0..n {
        let new_m = beta1 * m[i] + (1.0 - beta1) * grads[i];
        let new_v = beta2 * v[i] + (1.0 - beta2) * grads[i] * grads[i];
        m[i] = new_m;
        v[i] = new_v;
        let m_hat = new_m / bc1;
        let v_hat = new_v / bc2;
        params[i] -= lr * m_hat / (v_hat.sqrt() + eps);
    }
}

#[cfg(feature = "bruchion-kernels")]
#[allow(clippy::too_many_arguments)]
pub fn adam_step_kernel(params: &mut [f64], grads: &[f64], m: &mut [f64], v: &mut [f64], lr: f64, beta1: f64, beta2: f64, eps: f64, bc1: f64, bc2: f64) {
    let n = params.len().min(grads.len()).min(m.len()).min(v.len());
    // SAFETY: four live buffers of at least `n` elements; `params`, `m`, `v` are written.
    unsafe {
        super::ffi::cjc_adam_step_f64(n as i64, params.as_mut_ptr(), grads.as_ptr(), m.as_mut_ptr(), v.as_mut_ptr(), lr, beta1, beta2, eps, bc1, bc2)
    }
}

// ── powi ───────────────────────────────────────────────────────────────────

/// `a.powi(b)` — the kernel transcribes compiler-builtins' `__powidf2`, the binary
/// exponentiation `f64::powi` is documented to lower to. **On MSVC targets it does
/// not** (`f64_powi_is_binary_exponentiation`), so there, switching the kernels on
/// changes the last bits of every `powi` with a runtime exponent relative to the Rust
/// path — and makes them agree with Linux instead.
pub fn powi(a: f64, b: i32) -> f64 {
    #[cfg(feature = "bruchion-kernels")]
    if routes(KernelOp::Powi) {
        return powi_kernel(a, b);
    }
    a.powi(b)
}

#[cfg(feature = "bruchion-kernels")]
pub fn powi_kernel(a: f64, b: i32) -> f64 {
    // SAFETY: a pure function of two scalars.
    unsafe { super::ffi::cjc_powi_f64(a, b) }
}

/// compiler-builtins' `__powidf2`, transcribed: `r = r * a` on each set bit of `|b|`,
/// `a = a * a` between bits, `1 / r` for a negative exponent. The one copy of the
/// algorithm lives in `cjc_repro::powi_f64`, which every CJC site with a runtime
/// exponent now calls; this name is kept so the kernel is checked against the
/// *algorithm* on every target, whatever that target's `f64::powi` turns out to compute.
pub fn powi_reference(a: f64, b: i32) -> f64 {
    cjc_repro::powi_f64(a, b)
}

/// Whether this target's `f64::powi` with a runtime exponent is binary exponentiation.
/// 5000 probes: 200 bases in [0.5, 4) at exponents -12..=12, each exponent hidden from
/// the optimizer. True on x86_64-unknown-linux-gnu (rustc 1.98.1); **false on
/// x86_64-pc-windows-msvc (rustc 1.97.1)**, where LLVM has no `__powidf2` libcall and
/// lowers `powi` to the C runtime's `pow` (the binary imports `pow` from
/// `api-ms-win-crt-math-l1-1-0.dll`) — 2728 of the 5000 differ in the last bits. A
/// constant exponent at `-O` is expanded inline (binary exponentiation again), so on
/// that target the same source computes different bits in debug and release builds.
pub fn f64_powi_is_binary_exponentiation() -> bool {
    (0..200).all(|i| {
        let v = 0.5 + i as f64 * 0.0173;
        (-12..=12).all(|e| {
            let e = std::hint::black_box(e);
            v.powi(e).to_bits() == powi_reference(v, e).to_bits()
        })
    })
}

// ── heat1d residual and gradient ───────────────────────────────────────────

/// `u''(x)` of the polynomial `Σ coeffs[i] x^i` — `pinn::poly_eval_dd`, verbatim.
fn poly_eval_dd(coeffs: &[f64], x: f64) -> f64 {
    if coeffs.len() < 3 {
        return 0.0;
    }
    let mut result = 0.0;
    for i in (2..coeffs.len()).rev() {
        result = result * x + coeffs[i] * (i * (i - 1)) as f64;
    }
    result
}

/// The physics-loss loop of `pinn::piml_heat_1d_train` for the polynomial model:
/// `r[j] = u''(x[j]) − f[j]` and the Kahan mean of `r[j]²`. `f` is the source term at
/// each `x[j]`, computed by the caller (libm's `sin` stays on this side). `0.0` and no
/// stores for an empty `x`.
pub fn heat1d_residual(x: &[f64], coeffs: &[f64], f: &[f64], r: &mut [f64]) -> f64 {
    let n = x.len().min(f.len()).min(r.len());
    #[cfg(feature = "bruchion-kernels")]
    if routes(KernelOp::Heat1dResidual) {
        return heat1d_residual_kernel(&x[..n], coeffs, &f[..n], &mut r[..n]);
    }
    heat1d_residual_fallback(&x[..n], coeffs, &f[..n], &mut r[..n])
}

pub fn heat1d_residual_fallback(x: &[f64], coeffs: &[f64], f: &[f64], r: &mut [f64]) -> f64 {
    let n = x.len().min(f.len()).min(r.len());
    if n == 0 {
        return 0.0;
    }
    let mut acc = KahanAccumulatorF64::new();
    for j in 0..n {
        let residual = poly_eval_dd(coeffs, x[j]) - f[j];
        r[j] = residual;
        acc.add(residual * residual);
    }
    acc.finalize() / n as f64
}

#[cfg(feature = "bruchion-kernels")]
pub fn heat1d_residual_kernel(x: &[f64], coeffs: &[f64], f: &[f64], r: &mut [f64]) -> f64 {
    let n = x.len().min(f.len()).min(r.len());
    // SAFETY: `x`, `f` read and `r` written for `n` elements; `coeffs` read for its length.
    unsafe { super::ffi::cjc_heat1d_residual_f64(n as i64, x.as_ptr(), coeffs.len() as i64, coeffs.as_ptr(), f.as_ptr(), r.as_mut_ptr()) }
}

/// The same loop plus its `phys_grads`: `grad[i] += 2·r_j·(i(i−1)·x_j^(i−2)) / n` for
/// `i ≥ 2`, in `j` order, `grad` zeroed first — exactly as `piml_heat_1d_train` does,
/// `x.powi(i − 2)` included. Returns the loss; fills `r` as [`heat1d_residual`] does.
pub fn heat1d_residual_grad(x: &[f64], coeffs: &[f64], f: &[f64], r: &mut [f64], grad: &mut [f64]) -> f64 {
    let n = x.len().min(f.len()).min(r.len());
    #[cfg(feature = "bruchion-kernels")]
    if routes(KernelOp::Heat1dResidualGrad) {
        return heat1d_residual_grad_kernel(&x[..n], coeffs, &f[..n], &mut r[..n], grad);
    }
    heat1d_residual_grad_fallback(&x[..n], coeffs, &f[..n], &mut r[..n], grad)
}

pub fn heat1d_residual_grad_fallback(x: &[f64], coeffs: &[f64], f: &[f64], r: &mut [f64], grad: &mut [f64]) -> f64 {
    let n_params = coeffs.len().min(grad.len());
    for g in grad.iter_mut().take(n_params) {
        *g = 0.0;
    }
    let n = x.len().min(f.len()).min(r.len());
    if n == 0 {
        return 0.0;
    }
    let mut phys_acc = KahanAccumulatorF64::new();
    for j in 0..n {
        let xj = x[j];
        let residual = poly_eval_dd(coeffs, xj) - f[j];
        r[j] = residual;
        phys_acc.add(residual * residual);
        for i in 2..n_params {
            let du_xx_dai = (i * (i - 1)) as f64 * cjc_repro::powi_f64(xj, i as i32 - 2);
            grad[i] += 2.0 * residual * du_xx_dai / n as f64;
        }
    }
    phys_acc.finalize() / n as f64
}

/// The arithmetic the kernel implements, stated in Rust. Since CJC's own loop
/// (`piml_heat_1d_train`, and `heat1d_residual_grad_fallback` above) took
/// `cjc_repro::powi_f64`, the fallback IS this arithmetic on every target; the name is
/// kept so the tests still say which side is the specification.
pub fn heat1d_residual_grad_reference(x: &[f64], coeffs: &[f64], f: &[f64], r: &mut [f64], grad: &mut [f64]) -> f64 {
    heat1d_residual_grad_fallback(x, coeffs, f, r, grad)
}

#[cfg(feature = "bruchion-kernels")]
pub fn heat1d_residual_grad_kernel(x: &[f64], coeffs: &[f64], f: &[f64], r: &mut [f64], grad: &mut [f64]) -> f64 {
    let n = x.len().min(f.len()).min(r.len());
    let n_params = coeffs.len().min(grad.len());
    // SAFETY: `x`, `f`, `coeffs` read; `r` written for `n` and `grad` for `n_params`
    // elements, all live and at least that long.
    unsafe {
        super::ffi::cjc_heat1d_residual_grad_f64(n as i64, x.as_ptr(), n_params as i64, coeffs.as_ptr(), f.as_ptr(), r.as_mut_ptr(), grad.as_mut_ptr())
    }
}

// ── the parity tests ───────────────────────────────────────────────────────

/// Kernel against Rust body, `to_bits` equal, on inputs that span magnitudes and
/// include exact zeros (the Kahan zero skip), `-0.0` and NaN (relu), cancellation
/// (`1e16 − 1e16`), and the case found by search in the Bruchion repository where
/// a Kahan recurrence without the zero skip differs in the last bit.
#[cfg(all(test, feature = "bruchion-kernels"))]
mod parity {
    use super::*;

    fn bits(xs: &[f64]) -> Vec<u64> {
        xs.iter().map(|x| x.to_bits()).collect()
    }

    /// SplitMix64 over a fixed seed: uniform in [-1, 1) scaled by 2^k, k in -8..8, with
    /// every 17th value an exact zero and every 23rd a negative zero.
    fn inputs(n: usize, seed: u64) -> Vec<f64> {
        let mut s = seed;
        (0..n)
            .map(|i| {
                s = s.wrapping_add(0x9E3779B97F4A7C15);
                let mut z = s;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
                z ^= z >> 31;
                let u = (z >> 11) as f64 / 9007199254740992.0 * 2.0 - 1.0;
                let k = (z & 15) as i32 - 8;
                let v = u * 2f64.powi(k);
                if i % 17 == 16 { 0.0 } else if i % 23 == 22 { -0.0 } else { v }
            })
            .collect()
    }

    const SIZES: [usize; 4] = [0, 1, 7, 4093];

    #[test]
    fn axpy_bit_for_bit() {
        for &n in &SIZES {
            let x = inputs(n, 1);
            let y0 = inputs(n, 2);
            let mut y1 = y0.clone();
            let mut y2 = y0.clone();
            axpy_kernel(1.5, &x, &mut y1);
            axpy_fallback(1.5, &x, &mut y2);
            assert_eq!(bits(&y1), bits(&y2), "n = {n}");
        }
    }

    #[test]
    fn dot_kahan_bit_for_bit_including_the_zero_skip_witness() {
        for &n in &SIZES {
            let x = inputs(n, 3);
            let y = inputs(n, 4);
            assert_eq!(dot_kahan_kernel(&x, &y).to_bits(), dot_kahan_fallback(&x, &y).to_bits(), "n = {n}");
        }
        // The witness: a recurrence WITHOUT CJC's zero skip gives 13510798882111490 here.
        let z = [4503599627370497.0, 0.1, 0.0, 9007199254740994.0, 0.0];
        let ones = [1.0; 5];
        assert_eq!(dot_kahan_kernel(&z, &ones).to_bits(), 13510798882111492.0f64.to_bits());
        assert_eq!(dot_kahan_fallback(&z, &ones).to_bits(), 13510798882111492.0f64.to_bits());
        // Cancellation across the scale the accumulator was written for.
        let c = [1.0, 1e16, -1e16, 3.0, 1e100, -1e100, 1.0];
        assert_eq!(dot_kahan_kernel(&c, &[1.0; 7]).to_bits(), dot_kahan_fallback(&c, &[1.0; 7]).to_bits());
    }

    #[test]
    fn relu_bit_for_bit_on_signed_zero_and_nan() {
        let mut x = inputs(4093, 5);
        x[0] = -0.0;
        x[1] = f64::NAN;
        x[2] = f64::INFINITY;
        x[3] = f64::NEG_INFINITY;
        let mut a = vec![7.0; x.len()];
        let mut b = vec![7.0; x.len()];
        relu_kernel(&x, &mut a);
        relu_fallback(&x, &mut b);
        assert_eq!(bits(&a), bits(&b));
        assert_eq!(a[0].to_bits(), 0.0f64.to_bits(), "-0.0 maps to +0.0 on both sides");
        assert_eq!(a[1].to_bits(), 0.0f64.to_bits(), "NaN maps to 0.0 on both sides");
    }

    /// The pack's `cjc_sum_expbinned_f64` is `BinnedAccumulatorF64` — `binned_sum_f64`'s
    /// bits on the spanning inputs (zeros, signed zeros, four sizes), and on the specials
    /// CJC keeps aside: a NaN anywhere (Rust's `f64::NAN` bit for bit), one infinity, both
    /// infinities. The first version of the fused kernel used the pack's OWN binned
    /// accumulator (core's integer bins, correctly rounded) and this parity found the two
    /// part by an ulp at `n = 4093`: CJC's float bins round on every same-exponent add.
    /// The witness is the smallest case: `{1, 1+2^-52, 1+2^-52}` is `3 + 2^-51` exactly,
    /// and CJC's accumulator says `3.0`. The kernel says `3.0` too, and must.
    #[test]
    fn sum_expbinned_is_binned_sum_f64s_bits_and_the_witness() {
        use crate::accumulator::binned_sum_f64;
        for &n in &SIZES {
            let x = inputs(n, 21);
            let k = unsafe { crate::bruchion::ffi::cjc_sum_expbinned_f64(n as i64, x.as_ptr()) };
            assert_eq!(k.to_bits(), binned_sum_f64(&x).to_bits(), "n = {n}");
        }
        let up = f64::from_bits(0x3FF0000000000001);
        let w = [1.0, up, up];
        let k = unsafe { crate::bruchion::ffi::cjc_sum_expbinned_f64(3, w.as_ptr()) };
        assert_eq!(k.to_bits(), 3.0f64.to_bits());
        assert_eq!(binned_sum_f64(&w).to_bits(), 3.0f64.to_bits());
        assert_eq!(f64::from_bits(0x4008000000000001), 3.0 + 2.0f64.powi(-51), "the exact sum is representable");
        assert_ne!(binned_sum_f64(&w).to_bits(), (3.0 + 2.0f64.powi(-51)).to_bits(), "and CJC's accumulator does not give it");
        for s in [
            [1.0, f64::NAN, 2.0],
            [1.0, f64::INFINITY, 2.0],
            [1.0, f64::NEG_INFINITY, 2.0],
            [f64::INFINITY, f64::NEG_INFINITY, 2.0],
            [0.0, -0.0, f64::from_bits(1)],
        ] {
            let k = unsafe { crate::bruchion::ffi::cjc_sum_expbinned_f64(3, s.as_ptr()) };
            assert_eq!(k.to_bits(), binned_sum_f64(&s).to_bits(), "{s:?}");
        }
        let k = unsafe { crate::bruchion::ffi::cjc_sum_expbinned_f64(0, w.as_ptr()) };
        assert_eq!(k.to_bits(), binned_sum_f64(&[]).to_bits());
    }

    /// The tiled kernel against `TiledMatmul` itself, `to_bits`, at shapes on the edges of
    /// the 64-blocking in each dimension, a tiny one, the record's two, and `k = 0`; the
    /// fallback held to the same; the public entry switch-off against switch-on; and the
    /// negative: at 128^3 it is NOT the Kahan `matmul`'s value, so a test that compared
    /// the wrong pair could not pass.
    #[test]
    fn matmul_tiled_is_tiled_matmuls_bits_and_not_kahans() {
        for &(m, k, n) in &[(1usize, 1usize, 1usize), (3, 5, 2), (64, 17, 33), (65, 65, 65), (200, 3, 7), (7, 200, 3), (128, 128, 128), (2, 0, 3)] {
            let a = inputs(m * k, 31);
            let b = inputs(k * n, 32);
            let want = crate::tensor_tiled::TiledMatmul::new().matmul(&a, m, k, &b, n);
            let mut c = vec![7.0; m * n];
            matmul_tiled_kernel(&a, &b, &mut c, m, k, n);
            assert_eq!(bits(&c), bits(&want), "kernel at {m}x{k}x{n}");
            let mut f = vec![7.0; m * n];
            matmul_tiled_fallback(&a, &b, &mut f, m, k, n);
            assert_eq!(bits(&f), bits(&want), "fallback at {m}x{k}x{n}");
        }
        let (m, k, n) = (128usize, 128usize, 128usize);
        let a = inputs(m * k, 31);
        let b = inputs(k * n, 32);
        let mut tiled = vec![0.0; m * n];
        let mut kahan = vec![0.0; m * n];
        matmul_tiled_kernel(&a, &b, &mut tiled, m, k, n);
        matmul_fallback(&a, &b, &mut kahan, m, k, n);
        assert_ne!(bits(&tiled), bits(&kahan), "the plain sum and the Kahan sum must part somewhere at 128^3");
        let mut off = vec![0.0; m * n];
        let mut on = vec![0.0; m * n];
        matmul_tiled(&a, &b, &mut off, m, k, n);
        route_everything(true);
        matmul_tiled(&a, &b, &mut on, m, k, n);
        route_everything(false);
        assert_eq!(bits(&on), bits(&off));
    }

    /// The fused kernel against its Rust twin: the loss (CJC's exponent-binned sum, which
    /// the kernel carries itself — see `sum_expbinned_is_binned_sum_f64s_bits_and_the_witness`
    /// for why not the pack's own accumulator), and every gradient element. The witness
    /// pins the gradient's definition: at `n = 3`, `d = 2.9`, `h + h` and `(2 * d) / n`
    /// differ in the last bit, and the kernel gives the former.
    #[test]
    fn mse_grad_bit_for_bit_and_the_witness() {
        for &n in &SIZES {
            let p = inputs(n, 12);
            let t = inputs(n, 13);
            let mut g1 = vec![7.0; n];
            let mut g2 = vec![7.0; n];
            let l1 = mse_grad_kernel(&p, &t, &mut g1);
            let l2 = mse_grad_fallback(&p, &t, &mut g2);
            assert_eq!(l1.to_bits(), l2.to_bits(), "loss at n = {n}");
            assert_eq!(bits(&g1), bits(&g2), "grad at n = {n}");
        }
        let p = [2.9, 0.2, -0.4];
        let t = [0.0; 3];
        let mut g = [0.0; 3];
        mse_grad_kernel(&p, &t, &mut g);
        let h = (1.0f64 / 3.0) * 2.9;
        assert_eq!(g[0].to_bits(), (h + h).to_bits());
        assert_ne!(g[0].to_bits(), ((2.0f64 * 2.9) / 3.0).to_bits(), "the textbook spelling is a different value here");
        // `ml::mse_loss_grad` through the switch agrees with its own Rust path.
        let p = inputs(1001, 14);
        let t = inputs(1001, 15);
        let mut off = vec![0.0; 1001];
        let mut on = vec![0.0; 1001];
        let l_off = crate::ml::mse_loss_grad(&p, &t, &mut off).unwrap();
        route_everything(true);
        let l_on = crate::ml::mse_loss_grad(&p, &t, &mut on).unwrap();
        route_everything(false);
        assert_eq!(l_on.to_bits(), l_off.to_bits());
        assert_eq!(bits(&on), bits(&off));
    }

    /// The scaled gradient kernel against its Rust twin at four upstream values, and at
    /// `g = 1` against `mse_grad`'s gradient (pred = the input, target = 0, so the
    /// difference is the input itself).
    #[test]
    fn mse_grad_scaled_bit_for_bit_and_at_one_the_fused_gradient() {
        for &n in &SIZES {
            let d = inputs(n, 16);
            for &g in &[1.0f64, -0.37, 3.0, 1e-9] {
                let mut k = vec![7.0; n];
                let mut f = vec![7.0; n];
                mse_grad_scaled_kernel(g, &d, &mut k);
                mse_grad_scaled_fallback(g, &d, &mut f);
                assert_eq!(bits(&k), bits(&f), "n = {n}, g = {g}");
            }
            let z = vec![0.0; n];
            let mut a = vec![7.0; n];
            let mut b = vec![7.0; n];
            mse_grad_scaled_kernel(1.0, &d, &mut a);
            mse_grad_kernel(&d, &z, &mut b);
            assert_eq!(bits(&a), bits(&b), "g = 1 is the fused gradient, n = {n}");
        }
    }

    /// The set: off by default; the switch alone routes the measured set and nothing
    /// else; `route_everything` routes all of them and restores the default on the way
    /// back; a set edited by hand is honoured bit by bit.
    #[test]
    fn the_routing_set_gates_each_operation() {
        route_everything(false);
        assert!(!enabled());
        for op in KernelOp::EVERY {
            assert!(!routes(op), "{}: off by default", op.name());
        }
        crate::runtime_policy::set_bruchion_kernels(true);
        for op in KernelOp::EVERY {
            assert_eq!(routes(op), KernelOps::MEASURED.contains(op), "{}: the default set", op.name());
        }
        assert!(routes(KernelOp::Matmul) && routes(KernelOp::MseGrad) && !routes(KernelOp::Axpy) && !routes(KernelOp::Powi) && !routes(KernelOp::Heat1dResidualGrad));
        crate::runtime_policy::set_bruchion_kernel_ops(KernelOps::MEASURED.with(KernelOp::Relu).without(KernelOp::Mse).0);
        assert!(routes(KernelOp::Relu) && !routes(KernelOp::Mse) && routes(KernelOp::Matmul));
        route_everything(true);
        for op in KernelOp::EVERY {
            assert!(routes(op), "{}: everything", op.name());
        }
        route_everything(false);
        assert_eq!(crate::runtime_policy::get().bruchion_kernel_ops, KernelOps::MEASURED.0);
        assert_eq!(KernelOps::MEASURED.names(), "dot_kahan, mse, mse_grad, matmul");
        assert_eq!(KernelOps::NONE.names(), "(none)");
        assert_eq!(KernelOps::ALL.names().matches(", ").count(), 11);
    }

    #[test]
    fn mse_bit_for_bit() {
        for &n in &SIZES {
            let p = inputs(n, 6);
            let t = inputs(n, 7);
            assert_eq!(mse_kernel(&p, &t).to_bits(), mse_fallback(&p, &t).to_bits(), "n = {n}");
        }
        // `ml::mse_loss` itself, routed through the switch, agrees with its own Rust path.
        let p = inputs(1001, 8);
        let t = inputs(1001, 9);
        let off = crate::ml::mse_loss(&p, &t).unwrap();
        route_everything(true);
        let on = crate::ml::mse_loss(&p, &t).unwrap();
        route_everything(false);
        assert_eq!(on.to_bits(), off.to_bits());
    }

    #[test]
    fn matmul_bit_for_bit_at_several_shapes() {
        for &(m, k, n) in &[(1usize, 1usize, 1usize), (2, 3, 2), (5, 0, 4), (0, 3, 3), (7, 11, 5), (32, 32, 32), (64, 17, 33)] {
            let a = inputs(m * k, 10 + m as u64);
            let b = inputs(k * n, 20 + n as u64);
            let mut c1 = vec![7.0; m * n];
            let mut c2 = vec![7.0; m * n];
            matmul_kernel(&a, &b, &mut c1, m, k, n);
            matmul_fallback(&a, &b, &mut c2, m, k, n);
            assert_eq!(bits(&c1), bits(&c2), "shape {m}x{k}x{n}");
        }
        // And `kernel::matmul_raw` through the switch.
        let a = inputs(12, 30);
        let b = inputs(8, 31);
        let mut off = vec![0.0; 6];
        let mut on = vec![0.0; 6];
        crate::kernel::matmul_raw(&a, &b, &mut off, 3, 4, 2);
        route_everything(true);
        crate::kernel::matmul_raw(&a, &b, &mut on, 3, 4, 2);
        route_everything(false);
        assert_eq!(bits(&on), bits(&off));
    }

    #[test]
    fn adam_step_bit_for_bit_over_several_steps() {
        let n = 257;
        let mut p1 = inputs(n, 40);
        let mut p2 = p1.clone();
        let mut m1 = vec![0.0; n];
        let mut m2 = vec![0.0; n];
        let mut v1 = vec![0.0; n];
        let mut v2 = vec![0.0; n];
        for t in 1..=5u64 {
            let g = inputs(n, 50 + t);
            let tf = t as f64;
            let bc1 = 1.0 - 0.9f64.powf(tf);
            let bc2 = 1.0 - 0.999f64.powf(tf);
            adam_step_kernel(&mut p1, &g, &mut m1, &mut v1, 0.01, 0.9, 0.999, 1e-8, bc1, bc2);
            adam_step_fallback(&mut p2, &g, &mut m2, &mut v2, 0.01, 0.9, 0.999, 1e-8, bc1, bc2);
            assert_eq!(bits(&p1), bits(&p2), "step {t}");
            assert_eq!(bits(&m1), bits(&m2), "step {t}");
            assert_eq!(bits(&v1), bits(&v2), "step {t}");
        }
        // `ml::adam_step` through the switch against itself.
        let mut sa = crate::ml::AdamState::new(n, 0.01);
        let mut sb = crate::ml::AdamState::new(n, 0.01);
        let mut pa = inputs(n, 60);
        let mut pb = pa.clone();
        for t in 1..=3u64 {
            let g = inputs(n, 70 + t);
            crate::ml::adam_step(&mut pa, &g, &mut sa);
            route_everything(true);
            crate::ml::adam_step(&mut pb, &g, &mut sb);
            route_everything(false);
            assert_eq!(bits(&pa), bits(&pb), "step {t}");
        }
    }

    #[test]
    fn powi_kernel_is_compiler_builtins_binary_exponentiation_bit_for_bit() {
        let xs = inputs(200, 80);
        for &x in &xs {
            for b in -12..=12 {
                assert_eq!(powi_kernel(x, b).to_bits(), powi_reference(x, b).to_bits(), "x = {x:e}, b = {b}");
            }
        }
        for &(x, b) in &[(1.1, 5), (0.7, 3), (2.0, 62), (2.0, -3), (f64::NAN, 0), (0.0, 0), (-0.0, 3), (f64::INFINITY, -1)] {
            assert_eq!(powi_kernel(x, b).to_bits(), powi_reference(x, b).to_bits(), "x = {x}, b = {b}");
        }
    }

    /// The first disagreement this integration found (2026-09-22). On
    /// x86_64-pc-windows-msvc (rustc 1.97.1) `f64::powi` with a runtime exponent is not
    /// binary exponentiation: e.g. x = 3.8511975033176533, b = -11 gives
    /// 4510433485740570284 (the C runtime's `pow`), where `__powidf2` and the kernel give
    /// 4510433485740570282. On x86_64-unknown-linux-gnu (rustc 1.98.1) every probe
    /// agrees. So this passes on Linux and is ignored, with the reason, on MSVC. It is a
    /// statement about *Rust's* `f64::powi`, which no CJC arithmetic depends on any more:
    /// every runtime-exponent site calls `cjc_repro::powi_f64` (the test below is the one
    /// that matters for CJC's bits).
    #[test]
    #[cfg_attr(target_env = "msvc", ignore = "f64::powi with a runtime exponent is the C runtime's pow on MSVC targets, not binary exponentiation (docs/bruchion-kernels.md); CJC's own arithmetic uses cjc_repro::powi_f64 and does not depend on it")]
    fn f64_powi_is_binary_exponentiation_on_this_target() {
        assert!(f64_powi_is_binary_exponentiation());
        let xs = inputs(200, 80);
        for &x in &xs {
            for b in -12..=12 {
                assert_eq!(powi_kernel(x, b).to_bits(), x.powi(b).to_bits(), "x = {x:e}, b = {b}");
            }
        }
    }

    const HEAT1D_SHAPES: [(usize, usize); 6] = [(0, 4), (1, 1), (3, 3), (16, 2), (64, 6), (1000, 9)];

    /// Collocation points and the source term as `piml_heat_1d_train` computes them.
    fn heat1d_case(n_colloc: usize, n_params: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let x: Vec<f64> = (0..n_colloc).map(|i| (i as f64 + 0.5) / n_colloc as f64).collect();
        let coeffs = inputs(n_params, 90 + n_params as u64);
        let f: Vec<f64> = x.iter().map(|&x| -std::f64::consts::PI.powi(2) * (std::f64::consts::PI * x).sin()).collect();
        (x, coeffs, f)
    }

    #[test]
    fn heat1d_residual_and_gradient_match_the_reference_bit_for_bit() {
        for &(n_colloc, n_params) in &HEAT1D_SHAPES {
            let (x, coeffs, f) = heat1d_case(n_colloc, n_params);
            let mut r1 = vec![0.0; n_colloc];
            let mut r2 = vec![0.0; n_colloc];
            let l1 = heat1d_residual_kernel(&x, &coeffs, &f, &mut r1);
            let l2 = heat1d_residual_fallback(&x, &coeffs, &f, &mut r2);
            assert_eq!(l1.to_bits(), l2.to_bits(), "loss at {n_colloc}x{n_params}");
            assert_eq!(bits(&r1), bits(&r2), "residuals at {n_colloc}x{n_params}");
            let mut g1 = vec![9.0; n_params];
            let mut g2 = vec![9.0; n_params];
            let l1 = heat1d_residual_grad_kernel(&x, &coeffs, &f, &mut r1, &mut g1);
            let l2 = heat1d_residual_grad_reference(&x, &coeffs, &f, &mut r2, &mut g2);
            assert_eq!(l1.to_bits(), l2.to_bits(), "grad loss at {n_colloc}x{n_params}");
            assert_eq!(bits(&r1), bits(&r2));
            assert_eq!(bits(&g1), bits(&g2), "gradient at {n_colloc}x{n_params}");
        }
    }

    /// CJC's own loop (`f64::powi`) against the kernel. Where `f64::powi` is not binary
    /// exponentiation the two differ — one ulp in `grad[7]` at 1000x9 on
    /// x86_64-pc-windows-msvc, found 2026-09-22 — which means `piml_heat_1d_train` itself
    /// computes different bits on that target than on Linux.
    /// Runs on every target now: CJC's loop uses `cjc_repro::powi_f64`, so the one-ulp
    /// difference in `grad[7]` at 1000x9 that MSVC's `f64::powi` produced is gone, and
    /// this is a cross-platform gate rather than a Linux-only one.
    #[test]
    fn heat1d_gradient_matches_cjcs_own_loop_bit_for_bit() {
        for &(n_colloc, n_params) in &HEAT1D_SHAPES {
            let (x, coeffs, f) = heat1d_case(n_colloc, n_params);
            let mut r1 = vec![0.0; n_colloc];
            let mut r2 = vec![0.0; n_colloc];
            let mut g1 = vec![9.0; n_params];
            let mut g2 = vec![9.0; n_params];
            let l1 = heat1d_residual_grad_kernel(&x, &coeffs, &f, &mut r1, &mut g1);
            let l2 = heat1d_residual_grad_fallback(&x, &coeffs, &f, &mut r2, &mut g2);
            assert_eq!(l1.to_bits(), l2.to_bits(), "grad loss at {n_colloc}x{n_params}");
            assert_eq!(bits(&g1), bits(&g2), "gradient at {n_colloc}x{n_params}");
        }
    }

    #[test]
    fn the_runtime_switch_is_off_by_default_and_routes_when_on() {
        assert!(!enabled(), "the policy default must not route to the kernels");
        route_everything(true);
        assert!(enabled());
        route_everything(false);
        assert!(!enabled());
    }
    /// A probe, not a benchmark record: nanoseconds per element, kernel versus Rust body,
    /// min of 25 rounds after a warm-up, at 2^16 elements. Run with
    /// `cargo test -p cjc-runtime --release --features bruchion-kernels timing_probe -- --ignored --nocapture`.
    /// Everything a registered record needs (interleaving, A/A spread, provenance) is
    /// absent here; the number says whether the call is in the right order of magnitude.
    #[test]
    #[ignore]
    fn timing_probe() {
        use std::time::Instant;
        let n = 1 << 16;
        let x = inputs(n, 41);
        let y0 = inputs(n, 42);
        let mut y = y0.clone();
        let mut out = vec![0.0; n];
        let mut best = |label: &str, mut f: Box<dyn FnMut()>| {
            f();
            let mut min = f64::INFINITY;
            for _ in 0..25 {
                let t0 = Instant::now();
                f();
                let ns = t0.elapsed().as_nanos() as f64;
                if ns < min { min = ns; }
            }
            println!("{label:<24} {:.4} ns/elem", min / n as f64);
        };
        let (xa, ya) = (x.clone(), y.clone());
        let mut y1 = ya.clone();
        best("axpy kernel", Box::new(move || axpy_kernel(1.5, &xa, &mut y1)));
        let (xb, yb) = (x.clone(), y.clone());
        let mut y2 = yb.clone();
        best("axpy fallback", Box::new(move || axpy_fallback(1.5, &xb, &mut y2)));
        let (xc, yc) = (x.clone(), y0.clone());
        best("dot_kahan kernel", Box::new(move || { std::hint::black_box(dot_kahan_kernel(&xc, &yc)); }));
        let (xd, yd) = (x.clone(), y0.clone());
        best("dot_kahan fallback", Box::new(move || { std::hint::black_box(dot_kahan_fallback(&xd, &yd)); }));
        let xe = x.clone();
        let mut oe = out.clone();
        best("relu kernel", Box::new(move || relu_kernel(&xe, &mut oe)));
        let xf = x.clone();
        let mut of = out.clone();
        best("relu fallback", Box::new(move || relu_fallback(&xf, &mut of)));
        let (xg, yg) = (x.clone(), y0.clone());
        best("mse kernel", Box::new(move || { std::hint::black_box(mse_kernel(&xg, &yg)); }));
        let (xh, yh) = (x.clone(), y0.clone());
        best("mse fallback", Box::new(move || { std::hint::black_box(mse_fallback(&xh, &yh)); }));
        y.clear();
        out.clear();
    }
}

/// Without the feature, the switch can be set but nothing is routed: every function is
/// its Rust body, and this crate references no foreign symbol.
#[cfg(all(test, not(feature = "bruchion-kernels")))]
mod without_the_feature {
    use super::*;

    #[test]
    fn enabled_is_always_false_and_the_bodies_run() {
        route_everything(true);
        assert!(!enabled());
        route_everything(false);
        let x = [1.0, 2.0, 3.0];
        let mut y = [0.5, 0.5, 0.5];
        axpy(2.0, &x, &mut y);
        assert_eq!(y, [2.5, 4.5, 6.5]);
        assert_eq!(dot_kahan(&x, &x), 14.0);
        assert_eq!(powi(2.0, 10), 1024.0);
        assert_eq!(powi_reference(2.0, -3).to_bits(), 0.125f64.to_bits());
        assert_eq!(powi_reference(f64::NAN, 0).to_bits(), 1.0f64.to_bits());
    }
}
