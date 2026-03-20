//! Probability distributions — CDF, PDF, PPF for Normal, t, Chi-squared, F,
//! Binomial, Poisson.
//!
//! # Determinism Contract
//! All functions are pure math — no randomness, no iteration order dependency.
//! Approximations use deterministic, fixed-sequence algorithms.

use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Helper: ln_gamma (Lanczos approximation)
// ---------------------------------------------------------------------------

/// Log-gamma function via Lanczos approximation (g=7, n=9 coefficients).
/// Deterministic — fixed coefficient sequence.
pub fn ln_gamma(x: f64) -> f64 {
    if x <= 0.0 && x == x.floor() {
        return f64::INFINITY; // poles at non-positive integers
    }
    let g = 7.0;
    let coeff = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    let xx = if x < 0.5 {
        // Reflection formula
        let reflected = ln_gamma(1.0 - x);
        return (PI / (PI * x).sin()).ln() - reflected;
    } else {
        x - 1.0
    };
    let mut sum = coeff[0];
    for (i, &c) in coeff.iter().enumerate().skip(1) {
        sum += c / (xx + i as f64);
    }
    let t = xx + g + 0.5;
    0.5 * (2.0 * PI).ln() + (t.ln() * (xx + 0.5)) - t + sum.ln()
}

// ---------------------------------------------------------------------------
// Helper: regularized incomplete beta function (Lentz continued fraction)
// ---------------------------------------------------------------------------

/// Regularized incomplete beta function I_x(a, b).
/// Uses continued fraction (Lentz's method) for numerical stability.
fn regularized_incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    if x >= 1.0 { return 1.0; }

    // Use symmetry if x > (a+1)/(a+b+2) for better convergence
    if x > (a + 1.0) / (a + b + 2.0) {
        return 1.0 - regularized_incomplete_beta(b, a, 1.0 - x);
    }

    let lbeta = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);
    let front = (x.ln() * a + (1.0 - x).ln() * b - lbeta).exp() / a;

    // Lentz continued fraction
    let eps = 1e-14;
    let tiny = 1e-30;
    let max_iter = 200;

    let mut f = 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - (a + b) * x / (a + 1.0);
    if d.abs() < tiny { d = tiny; }
    d = 1.0 / d;
    f = d;

    for m in 1..=max_iter {
        let m_f = m as f64;
        // Even step
        let num_even = m_f * (b - m_f) * x / ((a + 2.0 * m_f - 1.0) * (a + 2.0 * m_f));
        d = 1.0 + num_even * d;
        if d.abs() < tiny { d = tiny; }
        c = 1.0 + num_even / c;
        if c.abs() < tiny { c = tiny; }
        d = 1.0 / d;
        f *= c * d;

        // Odd step
        let num_odd = -((a + m_f) * (a + b + m_f) * x)
            / ((a + 2.0 * m_f) * (a + 2.0 * m_f + 1.0));
        d = 1.0 + num_odd * d;
        if d.abs() < tiny { d = tiny; }
        c = 1.0 + num_odd / c;
        if c.abs() < tiny { c = tiny; }
        d = 1.0 / d;
        let delta = c * d;
        f *= delta;

        if (delta - 1.0).abs() < eps {
            break;
        }
    }

    front * f
}

// ---------------------------------------------------------------------------
// Helper: regularized lower incomplete gamma function
// ---------------------------------------------------------------------------

/// Regularized lower incomplete gamma function P(a, x) = gamma(a, x) / Gamma(a).
/// Uses series expansion for x < a+1, continued fraction otherwise.
fn regularized_gamma_p(a: f64, x: f64) -> f64 {
    if x < 0.0 { return 0.0; }
    if x == 0.0 { return 0.0; }

    if x < a + 1.0 {
        // Series expansion
        gamma_series(a, x)
    } else {
        // Continued fraction
        1.0 - gamma_cf(a, x)
    }
}

fn gamma_series(a: f64, x: f64) -> f64 {
    let max_iter = 200;
    let eps = 1e-14;
    let mut sum = 1.0 / a;
    let mut term = 1.0 / a;
    for n in 1..=max_iter {
        term *= x / (a + n as f64);
        sum += term;
        if term.abs() < sum.abs() * eps {
            break;
        }
    }
    sum * (-x + a * x.ln() - ln_gamma(a)).exp()
}

fn gamma_cf(a: f64, x: f64) -> f64 {
    // Lentz continued fraction for upper incomplete gamma Q(a,x)
    // CF: 1/(x+1-a+ K_{n=1}^{inf} n*(n-a)/(x+2n+1-a))
    let max_iter = 200;
    let eps = 1e-14;
    let tiny = 1e-30;

    let mut b = x + 1.0 - a;
    let mut c = 1.0 / tiny;
    let mut d = 1.0 / b;
    let mut f = d;

    for i in 1..=max_iter {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < tiny { d = tiny; }
        c = b + an / c;
        if c.abs() < tiny { c = tiny; }
        d = 1.0 / d;
        let delta = d * c;
        f *= delta;
        if (delta - 1.0).abs() < eps {
            break;
        }
    }

    (-x + a * x.ln() - ln_gamma(a)).exp() * f
}

// ---------------------------------------------------------------------------
// Error functions (Bastion ABI primitives)
// ---------------------------------------------------------------------------

/// Error function erf(x) using Horner-form rational approximation.
///
/// Uses the Abramowitz & Stegun 7.1.28 formula for |x| via the complementary
/// error function. Maximum error: |eps| < 1.5e-7.
///
/// # Determinism Contract
/// Pure math — same input => identical output. No iteration-order dependency.
pub fn erf(x: f64) -> f64 {
    // erf(x) = 1 - erfc(x)
    1.0 - erfc(x)
}

/// Complementary error function erfc(x) = 1 - erf(x).
///
/// Uses Abramowitz & Stegun 7.1.26 polynomial approximation.
/// Maximum error: |eps| < 1.5e-7.
///
/// # Determinism Contract
/// Pure math — deterministic for all finite inputs. NaN in => NaN out.
pub fn erfc(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        return 1.0; // exact
    }
    if x == f64::INFINITY {
        return 0.0;
    }
    if x == f64::NEG_INFINITY {
        return 2.0;
    }

    let a1 =  0.254829592;
    let a2 = -0.284496736;
    let a3 =  1.421413741;
    let a4 = -1.453152027;
    let a5 =  1.061405429;
    let p  =  0.3275911;

    let z = x.abs();
    let t = 1.0 / (1.0 + p * z);
    let y = (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-z * z).exp();

    if x < 0.0 {
        2.0 - y  // erfc(-x) = 2 - erfc(x)
    } else {
        y
    }
}

// ---------------------------------------------------------------------------
// Normal distribution
// ---------------------------------------------------------------------------

/// Normal distribution CDF using Abramowitz & Stegun approximation (7.1.26).
/// Maximum error: |eps| < 1.5e-7. Deterministic.
///
/// Equivalent to 0.5 * erfc(-x / sqrt(2)).
pub fn normal_cdf(x: f64) -> f64 {
    // Constants for the approximation
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x_abs = x.abs() / 2.0_f64.sqrt();
    let t = 1.0 / (1.0 + p * x_abs);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x_abs * x_abs).exp();

    0.5 * (1.0 + sign * y)
}

/// Normal distribution PDF: (1/sqrt(2*pi)) * exp(-x^2/2).
pub fn normal_pdf(x: f64) -> f64 {
    (1.0 / (2.0 * PI).sqrt()) * (-x * x / 2.0).exp()
}

/// Normal distribution PPF (inverse CDF / quantile function).
/// Uses rational approximation (Beasley-Springer-Moro).
/// p must be in (0, 1).
pub fn normal_ppf(p: f64) -> Result<f64, String> {
    if p <= 0.0 || p >= 1.0 {
        return Err(format!("normal_ppf: p must be in (0,1), got {p}"));
    }
    // Rational approximation
    let a = [
        -3.969683028665376e+01,
         2.209460984245205e+02,
        -2.759285104469687e+02,
         1.383577518672690e+02,
        -3.066479806614716e+01,
         2.506628277459239e+00,
    ];
    let b = [
        -5.447609879822406e+01,
         1.615858368580409e+02,
        -1.556989798598866e+02,
         6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    let c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
         4.374664141464968e+00,
         2.938163982698783e+00,
    ];
    let d = [
         7.784695709041462e-03,
         3.224671290700398e-01,
         2.445134137142996e+00,
         3.754408661907416e+00,
    ];

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    let result = if p < p_low {
        // Lower tail
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])
            / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    } else if p <= p_high {
        // Central region
        let q = p - 0.5;
        let r = q * q;
        (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q
            / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0)
    } else {
        // Upper tail
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])
            / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    };
    Ok(result)
}

// ---------------------------------------------------------------------------
// Student's t-distribution
// ---------------------------------------------------------------------------

/// Student's t-distribution CDF.
/// Uses regularized incomplete beta function.
pub fn t_cdf(x: f64, df: f64) -> f64 {
    let t2 = x * x;
    let ix = df / (df + t2);
    let beta = 0.5 * regularized_incomplete_beta(df / 2.0, 0.5, ix);
    if x >= 0.0 {
        1.0 - beta
    } else {
        beta
    }
}

/// Student's t-distribution PPF via bisection.
pub fn t_ppf(p: f64, df: f64) -> Result<f64, String> {
    if p <= 0.0 || p >= 1.0 {
        return Err(format!("t_ppf: p must be in (0,1), got {p}"));
    }
    // Bisection search
    let mut lo = -1000.0;
    let mut hi = 1000.0;
    for _ in 0..100 {
        let mid = (lo + hi) / 2.0;
        if t_cdf(mid, df) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok((lo + hi) / 2.0)
}

// ---------------------------------------------------------------------------
// Chi-squared distribution
// ---------------------------------------------------------------------------

/// Chi-squared distribution CDF.
/// Uses regularized lower incomplete gamma function.
pub fn chi2_cdf(x: f64, df: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    regularized_gamma_p(df / 2.0, x / 2.0)
}

/// Chi-squared distribution PPF via bisection.
pub fn chi2_ppf(p: f64, df: f64) -> Result<f64, String> {
    if p <= 0.0 || p >= 1.0 {
        return Err(format!("chi2_ppf: p must be in (0,1), got {p}"));
    }
    let mut lo = 0.0;
    let mut hi = df + 10.0 * (2.0 * df).sqrt().max(10.0);
    for _ in 0..100 {
        let mid = (lo + hi) / 2.0;
        if chi2_cdf(mid, df) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok((lo + hi) / 2.0)
}

// ---------------------------------------------------------------------------
// F-distribution
// ---------------------------------------------------------------------------

/// F-distribution CDF.
/// Uses regularized incomplete beta function.
pub fn f_cdf(x: f64, df1: f64, df2: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    let ix = df1 * x / (df1 * x + df2);
    regularized_incomplete_beta(df1 / 2.0, df2 / 2.0, ix)
}

/// F-distribution PPF via bisection.
pub fn f_ppf(p: f64, df1: f64, df2: f64) -> Result<f64, String> {
    if p <= 0.0 || p >= 1.0 {
        return Err(format!("f_ppf: p must be in (0,1), got {p}"));
    }
    let mut lo = 0.0;
    let mut hi = 1000.0;
    for _ in 0..100 {
        let mid = (lo + hi) / 2.0;
        if f_cdf(mid, df1, df2) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok((lo + hi) / 2.0)
}

// ---------------------------------------------------------------------------
// Binomial distribution
// ---------------------------------------------------------------------------

/// Binomial PMF: C(n,k) * p^k * (1-p)^(n-k).
pub fn binomial_pmf(k: u64, n: u64, p: f64) -> f64 {
    if k > n { return 0.0; }
    let log_coeff = ln_gamma(n as f64 + 1.0) - ln_gamma(k as f64 + 1.0) - ln_gamma((n - k) as f64 + 1.0);
    let log_prob = k as f64 * p.ln() + (n - k) as f64 * (1.0 - p).ln();
    (log_coeff + log_prob).exp()
}

/// Binomial CDF: sum_{i=0}^{k} binomial_pmf(i, n, p).
pub fn binomial_cdf(k: u64, n: u64, p: f64) -> f64 {
    let mut sum = cjc_repro::KahanAccumulatorF64::new();
    for i in 0..=k {
        sum.add(binomial_pmf(i, n, p));
    }
    sum.finalize()
}

// ---------------------------------------------------------------------------
// Poisson distribution
// ---------------------------------------------------------------------------

/// Poisson PMF: (lambda^k * e^-lambda) / k!
pub fn poisson_pmf(k: u64, lambda: f64) -> f64 {
    let log_prob = k as f64 * lambda.ln() - lambda - ln_gamma(k as f64 + 1.0);
    log_prob.exp()
}

/// Poisson CDF: sum_{i=0}^{k} poisson_pmf(i, lambda).
pub fn poisson_cdf(k: u64, lambda: f64) -> f64 {
    let mut sum = cjc_repro::KahanAccumulatorF64::new();
    for i in 0..=k {
        sum.add(poisson_pmf(i, lambda));
    }
    sum.finalize()
}

// ---------------------------------------------------------------------------
// Phase B6: Beta, Gamma, Exponential, Weibull distributions
// ---------------------------------------------------------------------------

/// Beta distribution PDF: x^(a-1) * (1-x)^(b-1) / B(a,b).
/// x in [0,1], a > 0, b > 0.
pub fn beta_pdf(x: f64, a: f64, b: f64) -> f64 {
    if x < 0.0 || x > 1.0 { return 0.0; }
    if x == 0.0 && a < 1.0 { return f64::INFINITY; }
    if x == 1.0 && b < 1.0 { return f64::INFINITY; }
    let log_beta = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);
    ((a - 1.0) * x.ln() + (b - 1.0) * (1.0 - x).ln() - log_beta).exp()
}

/// Beta distribution CDF via regularized incomplete beta function.
pub fn beta_cdf(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    if x >= 1.0 { return 1.0; }
    regularized_incomplete_beta(a, b, x)
}

/// Gamma distribution PDF: x^(k-1) * exp(-x/theta) / (theta^k * Gamma(k)).
/// x >= 0, k > 0 (shape), theta > 0 (scale).
pub fn gamma_pdf(x: f64, k: f64, theta: f64) -> f64 {
    if x < 0.0 { return 0.0; }
    if x == 0.0 {
        if k < 1.0 { return f64::INFINITY; }
        if k == 1.0 { return 1.0 / theta; }
        return 0.0;
    }
    let log_pdf = (k - 1.0) * x.ln() - x / theta - k * theta.ln() - ln_gamma(k);
    log_pdf.exp()
}

/// Gamma distribution CDF via regularized lower incomplete gamma function.
pub fn gamma_cdf(x: f64, k: f64, theta: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    regularized_gamma_p(k, x / theta)
}

/// Exponential distribution PDF: lambda * exp(-lambda * x).
/// x >= 0, lambda > 0 (rate).
pub fn exp_pdf(x: f64, lambda: f64) -> f64 {
    if x < 0.0 { return 0.0; }
    lambda * (-lambda * x).exp()
}

/// Exponential distribution CDF: 1 - exp(-lambda * x).
pub fn exp_cdf(x: f64, lambda: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    1.0 - (-lambda * x).exp()
}

/// Weibull distribution PDF: (k/lambda) * (x/lambda)^(k-1) * exp(-(x/lambda)^k).
/// x >= 0, k > 0 (shape), lambda > 0 (scale).
pub fn weibull_pdf(x: f64, k: f64, lambda: f64) -> f64 {
    if x < 0.0 { return 0.0; }
    if x == 0.0 {
        if k < 1.0 { return f64::INFINITY; }
        if k == 1.0 { return 1.0 / lambda; }
        return 0.0;
    }
    (k / lambda) * (x / lambda).powf(k - 1.0) * (-(x / lambda).powf(k)).exp()
}

/// Weibull distribution CDF: 1 - exp(-(x/lambda)^k).
pub fn weibull_cdf(x: f64, k: f64, lambda: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    1.0 - (-(x / lambda).powf(k)).exp()
}

// ===========================================================================
// Phase 4: Distribution Sampling Functions
// ===========================================================================
//
// All sampling functions are deterministic given the same RNG state.
// They consume RNG draws in a fixed, predictable order.
// Floating-point reductions use Kahan summation where applicable.

// ---------------------------------------------------------------------------
// Normal sampling
// ---------------------------------------------------------------------------

/// Sample `n` values from Normal(mu, sigma) using Box-Muller via `rng.next_normal_f64()`.
///
/// # Determinism Contract
/// Same RNG state => identical output vector, bit-for-bit.
pub fn normal_sample(mu: f64, sigma: f64, n: usize, rng: &mut cjc_repro::Rng) -> Vec<f64> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(mu + sigma * rng.next_normal_f64());
    }
    out
}

// ---------------------------------------------------------------------------
// Uniform sampling
// ---------------------------------------------------------------------------

/// Sample `n` values from Uniform(a, b).
///
/// Each sample: a + (b - a) * U where U ~ Uniform[0, 1).
pub fn uniform_sample(a: f64, b: f64, n: usize, rng: &mut cjc_repro::Rng) -> Vec<f64> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(a + (b - a) * rng.next_f64());
    }
    out
}

// ---------------------------------------------------------------------------
// Exponential sampling
// ---------------------------------------------------------------------------

/// Sample `n` values from Exponential(lambda) using inverse CDF: -ln(1 - U) / lambda.
///
/// Uses `1.0 - rng.next_f64()` to avoid ln(0).
pub fn exponential_sample(lambda: f64, n: usize, rng: &mut cjc_repro::Rng) -> Vec<f64> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        // 1.0 - next_f64() gives (0, 1] which avoids ln(0)
        let u = 1.0 - rng.next_f64();
        out.push(-u.ln() / lambda);
    }
    out
}

// ---------------------------------------------------------------------------
// Gamma sampling — Marsaglia-Tsang method
// ---------------------------------------------------------------------------

/// Sample a single Gamma(shape, 1) value using Marsaglia-Tsang's method.
///
/// For shape >= 1: direct Marsaglia-Tsang.
/// For shape < 1: sample Gamma(shape + 1, 1) then multiply by U^(1/shape)
/// where U ~ Uniform(0,1).
///
/// Reference: Marsaglia & Tsang, "A Simple Method for Generating Gamma Variables" (2000).
fn gamma_sample_single(shape: f64, rng: &mut cjc_repro::Rng) -> f64 {
    if shape < 1.0 {
        // Shape augmentation trick: Gamma(a, 1) = Gamma(a+1, 1) * U^(1/a)
        let g = gamma_sample_single(shape + 1.0, rng);
        let u = rng.next_f64();
        // Use 1.0 - u to avoid u = 0 (which would give 0^(1/a) = 0 always)
        return g * (1.0 - u).powf(1.0 / shape);
    }

    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();

    loop {
        let x = rng.next_normal_f64();
        let v_base = 1.0 + c * x;
        if v_base <= 0.0 {
            continue;
        }
        let v = v_base * v_base * v_base;
        let u = rng.next_f64();

        // Squeeze test
        if u < 1.0 - 0.0331 * (x * x) * (x * x) {
            return d * v;
        }
        // Full test
        if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}

/// Sample `n` values from Gamma(shape_k, scale_theta).
///
/// Uses Marsaglia-Tsang method (shape >= 1) with shape augmentation for shape < 1.
/// Result = Gamma(shape, 1) * scale.
pub fn gamma_sample(
    shape_k: f64,
    scale_theta: f64,
    n: usize,
    rng: &mut cjc_repro::Rng,
) -> Vec<f64> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(gamma_sample_single(shape_k, rng) * scale_theta);
    }
    out
}

// ---------------------------------------------------------------------------
// Beta sampling — via Gamma
// ---------------------------------------------------------------------------

/// Sample `n` values from Beta(a, b) using the gamma ratio method.
///
/// X ~ Gamma(a, 1), Y ~ Gamma(b, 1), then X / (X + Y) ~ Beta(a, b).
pub fn beta_sample(a: f64, b: f64, n: usize, rng: &mut cjc_repro::Rng) -> Vec<f64> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let x = gamma_sample_single(a, rng);
        let y = gamma_sample_single(b, rng);
        out.push(x / (x + y));
    }
    out
}

// ---------------------------------------------------------------------------
// Chi-squared sampling — via Gamma
// ---------------------------------------------------------------------------

/// Sample `n` values from Chi-squared(df).
///
/// Chi-squared(df) = Gamma(df/2, 2).
pub fn chi2_sample(df: f64, n: usize, rng: &mut cjc_repro::Rng) -> Vec<f64> {
    gamma_sample(df / 2.0, 2.0, n, rng)
}

// ---------------------------------------------------------------------------
// Student's t sampling
// ---------------------------------------------------------------------------

/// Sample `n` values from Student's t(df).
///
/// t = Z / sqrt(V / df) where Z ~ Normal(0,1), V ~ Chi-squared(df).
pub fn t_sample(df: f64, n: usize, rng: &mut cjc_repro::Rng) -> Vec<f64> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let z = rng.next_normal_f64();
        let v = gamma_sample_single(df / 2.0, rng) * 2.0; // Chi-squared(df)
        out.push(z / (v / df).sqrt());
    }
    out
}

// ---------------------------------------------------------------------------
// Poisson sampling
// ---------------------------------------------------------------------------

/// Sample a single Poisson(lambda) value using Knuth's algorithm (lambda < 30)
/// or the transformed rejection method (lambda >= 30).
fn poisson_sample_single(lambda: f64, rng: &mut cjc_repro::Rng) -> i64 {
    if lambda < 30.0 {
        // Knuth's algorithm
        let l = (-lambda).exp();
        let mut k: i64 = 0;
        let mut p = 1.0;
        loop {
            k += 1;
            p *= rng.next_f64();
            if p <= l {
                return k - 1;
            }
        }
    } else {
        // Transformed rejection method (Hoermann, "The Transformed Rejection Method")
        // Approximation: Poisson ~ floor(Normal(lambda, sqrt(lambda)) + 0.5) with
        // acceptance-rejection correction.
        let sqrt_lam = lambda.sqrt();
        let log_lam = lambda.ln();
        let b = 0.931 + 2.53 * sqrt_lam;
        let a = -0.059 + 0.02483 * b;
        let inv_alpha = 1.1239 + 1.1328 / (b - 3.4);
        let v_r = 0.9277 - 3.6224 / (b - 2.0);

        loop {
            let u = rng.next_f64() - 0.5;
            let v = rng.next_f64();
            let us = 0.5 - u.abs();
            let k = ((2.0 * a / us + b) * u + lambda + 0.43).floor() as i64;

            if k < 0 {
                continue;
            }

            // Squeeze acceptance
            if us >= 0.07 && v <= v_r {
                return k;
            }

            // Full acceptance check
            let kf = k as f64;
            let log_fk = ln_gamma(kf + 1.0);
            let log_prob = kf * log_lam - lambda - log_fk;

            if (us >= 0.013 || v <= us)
                && v.ln() + inv_alpha.ln() - (a / (us * us) + b).ln()
                    <= log_prob
            {
                return k;
            }
        }
    }
}

/// Sample `n` values from Poisson(lambda).
///
/// Uses Knuth's algorithm for lambda < 30 and transformed rejection for lambda >= 30.
pub fn poisson_sample(lambda: f64, n: usize, rng: &mut cjc_repro::Rng) -> Vec<i64> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(poisson_sample_single(lambda, rng));
    }
    out
}

// ---------------------------------------------------------------------------
// Binomial sampling
// ---------------------------------------------------------------------------

/// Sample a single Binomial(n_trials, p) value.
///
/// For small n_trials (< 25): direct simulation (sum of Bernoulli trials).
/// For large n_trials: normal approximation with continuity correction,
/// clamped to [0, n_trials].
fn binomial_sample_single(n_trials: usize, p: f64, rng: &mut cjc_repro::Rng) -> i64 {
    if n_trials < 25 {
        // Direct simulation
        let mut count: i64 = 0;
        for _ in 0..n_trials {
            if rng.next_f64() < p {
                count += 1;
            }
        }
        count
    } else {
        // Normal approximation: X ~ round(Normal(np, sqrt(np(1-p))))
        let np = n_trials as f64 * p;
        let sigma = (np * (1.0 - p)).sqrt();
        let z = rng.next_normal_f64();
        let x = (np + sigma * z).round() as i64;
        // Clamp to valid range
        x.max(0).min(n_trials as i64)
    }
}

/// Sample `n` values from Binomial(n_trials, p).
///
/// Uses direct simulation for small n_trials (< 25) and normal approximation
/// for larger values.
pub fn binomial_sample(
    n_trials: usize,
    p: f64,
    n: usize,
    rng: &mut cjc_repro::Rng,
) -> Vec<i64> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(binomial_sample_single(n_trials, p, rng));
    }
    out
}

// ---------------------------------------------------------------------------
// Bernoulli sampling
// ---------------------------------------------------------------------------

/// Sample `n` Bernoulli(p) values.
///
/// Returns `true` with probability `p`, `false` with probability `1-p`.
pub fn bernoulli_sample(p: f64, n: usize, rng: &mut cjc_repro::Rng) -> Vec<bool> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(rng.next_f64() < p);
    }
    out
}

// ---------------------------------------------------------------------------
// Dirichlet sampling
// ---------------------------------------------------------------------------

/// Sample a single Dirichlet(alpha) vector.
///
/// Each component X_i ~ Gamma(alpha_i, 1), then normalize: X_i / sum(X).
/// Normalization uses Kahan summation for deterministic stability.
pub fn dirichlet_sample(alpha: &[f64], rng: &mut cjc_repro::Rng) -> Vec<f64> {
    let k = alpha.len();
    let mut raw = Vec::with_capacity(k);
    let mut sum = cjc_repro::KahanAccumulatorF64::new();

    for &a in alpha {
        let g = gamma_sample_single(a, rng);
        raw.push(g);
        sum.add(g);
    }

    let total = sum.finalize();
    for x in &mut raw {
        *x /= total;
    }
    raw
}

// ---------------------------------------------------------------------------
// Multinomial (categorical) sampling
// ---------------------------------------------------------------------------

/// Sample `n` categorical draws from the given probability vector.
///
/// Returns indices 0..probs.len()-1 sampled according to probabilities.
/// Probabilities are normalized internally using Kahan summation.
/// Uses CDF search for each draw.
pub fn multinomial_sample(probs: &[f64], n: usize, rng: &mut cjc_repro::Rng) -> Vec<usize> {
    if probs.is_empty() {
        return Vec::new();
    }

    // Build normalized CDF using Kahan summation
    let mut total_acc = cjc_repro::KahanAccumulatorF64::new();
    for &p in probs {
        total_acc.add(p);
    }
    let total = total_acc.finalize();

    let k = probs.len();
    let mut cdf = Vec::with_capacity(k);
    let mut cum_acc = cjc_repro::KahanAccumulatorF64::new();
    for &p in probs {
        cum_acc.add(p / total);
        cdf.push(cum_acc.finalize());
    }
    // Ensure last entry is exactly 1.0 to avoid floating-point edge cases
    if let Some(last) = cdf.last_mut() {
        *last = 1.0;
    }

    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let u = rng.next_f64();
        // Linear search through CDF (deterministic ordering)
        let mut idx = 0;
        while idx < k - 1 && u >= cdf[idx] {
            idx += 1;
        }
        out.push(idx);
    }
    out
}

// ---------------------------------------------------------------------------
// Latin Hypercube Sampling
// ---------------------------------------------------------------------------

/// Latin Hypercube Sampling — generates n samples in `dims` dimensions.
///
/// Each dimension is divided into n equal strata. Exactly one sample
/// is drawn from each stratum per dimension, then columns are shuffled
/// independently using the provided seed for deterministic output.
///
/// Returns a Tensor of shape [n, dims] with values in [0, 1).
pub fn latin_hypercube_sample(n: usize, dims: usize, seed: u64) -> crate::tensor::Tensor {
    if n == 0 || dims == 0 {
        return crate::tensor::Tensor::from_vec_unchecked(Vec::new(), &[0, dims.max(1)]);
    }

    // We use separate RNG streams per dimension to ensure independence.
    // Seed for dimension d is derived deterministically from the base seed.
    let mut data = vec![0.0f64; n * dims];

    for d in 0..dims {
        // Derive a deterministic per-dimension seed using a simple mixing function.
        let dim_seed = seed
            .wrapping_add(d as u64)
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let mut rng = cjc_repro::Rng::seeded(dim_seed);

        // Build strata indices [0, 1, ..., n-1].
        let mut strata: Vec<usize> = (0..n).collect();

        // Fisher-Yates shuffle of strata indices.
        for i in (1..n).rev() {
            // Generate a uniform integer in [0, i] using next_f64.
            let j = (rng.next_f64() * (i + 1) as f64) as usize;
            let j = j.min(i); // guard against rounding to i+1
            strata.swap(i, j);
        }

        // Place one random point within each assigned stratum.
        for i in 0..n {
            let stratum = strata[i];
            let offset = rng.next_f64(); // uniform in [0, 1)
            let value = (stratum as f64 + offset) / n as f64;
            data[i * dims + d] = value;
        }
    }

    crate::tensor::Tensor::from_vec_unchecked(data, &[n, dims])
}

// ---------------------------------------------------------------------------
// Sobol-like low-discrepancy sequence (Van der Corput)
// ---------------------------------------------------------------------------

/// Generate a Sobol-like low-discrepancy sequence.
///
/// Uses a simple bit-reversal approach (Van der Corput sequence)
/// for each dimension with different bases. Not a true Sobol sequence
/// but provides good space-filling properties for moderate dimensions.
///
/// Returns a Tensor of shape [n, dims] with values in [0, 1).
pub fn sobol_sequence(n: usize, dims: usize) -> crate::tensor::Tensor {
    if n == 0 || dims == 0 {
        return crate::tensor::Tensor::from_vec_unchecked(Vec::new(), &[0, dims.max(1)]);
    }

    // First 30 primes used as bases for successive dimensions.
    const PRIMES: [u64; 30] = [
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
        31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
        73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
    ];

    let mut data = vec![0.0f64; n * dims];

    for d in 0..dims {
        let base = PRIMES[d % PRIMES.len()];
        for i in 0..n {
            data[i * dims + d] = van_der_corput(i as u64, base);
        }
    }

    crate::tensor::Tensor::from_vec_unchecked(data, &[n, dims])
}

/// Compute the Van der Corput radical-inverse of `index` in the given `base`.
///
/// Returns a value in [0, 1) by reflecting the base-`base` digits of `index`
/// about the decimal point.
fn van_der_corput(mut index: u64, base: u64) -> f64 {
    let mut result = 0.0f64;
    let mut denominator = 1.0f64;
    while index > 0 {
        denominator *= base as f64;
        result += (index % base) as f64 / denominator;
        index /= base;
    }
    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod sampling_tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helper: compute sample mean using Kahan summation
    // -----------------------------------------------------------------------
    fn kahan_mean(data: &[f64]) -> f64 {
        let mut acc = cjc_repro::KahanAccumulatorF64::new();
        for &x in data {
            acc.add(x);
        }
        acc.finalize() / data.len() as f64
    }

    // -----------------------------------------------------------------------
    // Determinism tests: same seed => same output
    // -----------------------------------------------------------------------

    #[test]
    fn test_normal_sample_determinism() {
        let mut r1 = cjc_repro::Rng::seeded(42);
        let mut r2 = cjc_repro::Rng::seeded(42);
        let a = normal_sample(0.0, 1.0, 100, &mut r1);
        let b = normal_sample(0.0, 1.0, 100, &mut r2);
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
    }

    #[test]
    fn test_uniform_sample_determinism() {
        let mut r1 = cjc_repro::Rng::seeded(7);
        let mut r2 = cjc_repro::Rng::seeded(7);
        let a = uniform_sample(0.0, 1.0, 100, &mut r1);
        let b = uniform_sample(0.0, 1.0, 100, &mut r2);
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
    }

    #[test]
    fn test_exponential_sample_determinism() {
        let mut r1 = cjc_repro::Rng::seeded(99);
        let mut r2 = cjc_repro::Rng::seeded(99);
        let a = exponential_sample(2.0, 100, &mut r1);
        let b = exponential_sample(2.0, 100, &mut r2);
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
    }

    #[test]
    fn test_gamma_sample_determinism() {
        let mut r1 = cjc_repro::Rng::seeded(13);
        let mut r2 = cjc_repro::Rng::seeded(13);
        let a = gamma_sample(2.5, 1.0, 100, &mut r1);
        let b = gamma_sample(2.5, 1.0, 100, &mut r2);
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
    }

    #[test]
    fn test_beta_sample_determinism() {
        let mut r1 = cjc_repro::Rng::seeded(55);
        let mut r2 = cjc_repro::Rng::seeded(55);
        let a = beta_sample(2.0, 5.0, 100, &mut r1);
        let b = beta_sample(2.0, 5.0, 100, &mut r2);
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
    }

    #[test]
    fn test_chi2_sample_determinism() {
        let mut r1 = cjc_repro::Rng::seeded(77);
        let mut r2 = cjc_repro::Rng::seeded(77);
        let a = chi2_sample(5.0, 100, &mut r1);
        let b = chi2_sample(5.0, 100, &mut r2);
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
    }

    #[test]
    fn test_t_sample_determinism() {
        let mut r1 = cjc_repro::Rng::seeded(111);
        let mut r2 = cjc_repro::Rng::seeded(111);
        let a = t_sample(10.0, 100, &mut r1);
        let b = t_sample(10.0, 100, &mut r2);
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
    }

    #[test]
    fn test_poisson_sample_determinism() {
        let mut r1 = cjc_repro::Rng::seeded(33);
        let mut r2 = cjc_repro::Rng::seeded(33);
        let a = poisson_sample(5.0, 100, &mut r1);
        let b = poisson_sample(5.0, 100, &mut r2);
        assert_eq!(a, b);
    }

    #[test]
    fn test_poisson_large_lambda_determinism() {
        let mut r1 = cjc_repro::Rng::seeded(44);
        let mut r2 = cjc_repro::Rng::seeded(44);
        let a = poisson_sample(50.0, 100, &mut r1);
        let b = poisson_sample(50.0, 100, &mut r2);
        assert_eq!(a, b);
    }

    #[test]
    fn test_binomial_sample_determinism() {
        let mut r1 = cjc_repro::Rng::seeded(88);
        let mut r2 = cjc_repro::Rng::seeded(88);
        let a = binomial_sample(20, 0.4, 100, &mut r1);
        let b = binomial_sample(20, 0.4, 100, &mut r2);
        assert_eq!(a, b);
    }

    #[test]
    fn test_bernoulli_sample_determinism() {
        let mut r1 = cjc_repro::Rng::seeded(22);
        let mut r2 = cjc_repro::Rng::seeded(22);
        let a = bernoulli_sample(0.7, 100, &mut r1);
        let b = bernoulli_sample(0.7, 100, &mut r2);
        assert_eq!(a, b);
    }

    #[test]
    fn test_dirichlet_sample_determinism() {
        let mut r1 = cjc_repro::Rng::seeded(66);
        let mut r2 = cjc_repro::Rng::seeded(66);
        let a = dirichlet_sample(&[1.0, 2.0, 3.0], &mut r1);
        let b = dirichlet_sample(&[1.0, 2.0, 3.0], &mut r2);
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
    }

    #[test]
    fn test_multinomial_sample_determinism() {
        let mut r1 = cjc_repro::Rng::seeded(101);
        let mut r2 = cjc_repro::Rng::seeded(101);
        let a = multinomial_sample(&[0.2, 0.3, 0.5], 100, &mut r1);
        let b = multinomial_sample(&[0.2, 0.3, 0.5], 100, &mut r2);
        assert_eq!(a, b);
    }

    // -----------------------------------------------------------------------
    // Range correctness tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_uniform_range() {
        let mut rng = cjc_repro::Rng::seeded(1);
        let samples = uniform_sample(2.0, 5.0, 1000, &mut rng);
        for &x in &samples {
            assert!(x >= 2.0 && x < 5.0, "uniform out of range: {x}");
        }
    }

    #[test]
    fn test_exponential_positive() {
        let mut rng = cjc_repro::Rng::seeded(2);
        let samples = exponential_sample(1.5, 1000, &mut rng);
        for &x in &samples {
            assert!(x > 0.0, "exponential not positive: {x}");
        }
    }

    #[test]
    fn test_gamma_positive() {
        let mut rng = cjc_repro::Rng::seeded(3);
        // Test both shape < 1 and shape > 1
        let samples_small = gamma_sample(0.5, 2.0, 500, &mut rng);
        let samples_large = gamma_sample(5.0, 1.0, 500, &mut rng);
        for &x in samples_small.iter().chain(samples_large.iter()) {
            assert!(x > 0.0, "gamma not positive: {x}");
        }
    }

    #[test]
    fn test_beta_unit_interval() {
        let mut rng = cjc_repro::Rng::seeded(4);
        let samples = beta_sample(2.0, 5.0, 1000, &mut rng);
        for &x in &samples {
            assert!(x >= 0.0 && x <= 1.0, "beta out of [0,1]: {x}");
        }
    }

    #[test]
    fn test_chi2_positive() {
        let mut rng = cjc_repro::Rng::seeded(5);
        let samples = chi2_sample(3.0, 1000, &mut rng);
        for &x in &samples {
            assert!(x > 0.0, "chi2 not positive: {x}");
        }
    }

    #[test]
    fn test_poisson_non_negative() {
        let mut rng = cjc_repro::Rng::seeded(6);
        let samples = poisson_sample(4.0, 1000, &mut rng);
        for &x in &samples {
            assert!(x >= 0, "poisson negative: {x}");
        }
    }

    #[test]
    fn test_poisson_large_non_negative() {
        let mut rng = cjc_repro::Rng::seeded(60);
        let samples = poisson_sample(50.0, 1000, &mut rng);
        for &x in &samples {
            assert!(x >= 0, "poisson(50) negative: {x}");
        }
    }

    #[test]
    fn test_binomial_range() {
        let mut rng = cjc_repro::Rng::seeded(7);
        let samples = binomial_sample(10, 0.5, 1000, &mut rng);
        for &x in &samples {
            assert!(x >= 0 && x <= 10, "binomial out of range: {x}");
        }
    }

    #[test]
    fn test_bernoulli_values() {
        let mut rng = cjc_repro::Rng::seeded(8);
        let samples = bernoulli_sample(0.5, 1000, &mut rng);
        // Just check they are booleans (always true) and have both values
        let trues = samples.iter().filter(|&&x| x).count();
        let falses = samples.len() - trues;
        assert!(trues > 0, "no true values");
        assert!(falses > 0, "no false values");
    }

    #[test]
    fn test_dirichlet_simplex() {
        let mut rng = cjc_repro::Rng::seeded(9);
        let sample = dirichlet_sample(&[1.0, 2.0, 3.0, 4.0], &mut rng);
        assert_eq!(sample.len(), 4);
        for &x in &sample {
            assert!(x >= 0.0 && x <= 1.0, "dirichlet component out of [0,1]: {x}");
        }
        let mut sum_acc = cjc_repro::KahanAccumulatorF64::new();
        for &x in &sample {
            sum_acc.add(x);
        }
        let sum = sum_acc.finalize();
        assert!((sum - 1.0).abs() < 1e-12, "dirichlet does not sum to 1: {sum}");
    }

    #[test]
    fn test_multinomial_valid_indices() {
        let mut rng = cjc_repro::Rng::seeded(10);
        let probs = vec![0.1, 0.2, 0.3, 0.4];
        let samples = multinomial_sample(&probs, 1000, &mut rng);
        for &idx in &samples {
            assert!(idx < probs.len(), "multinomial index out of range: {idx}");
        }
    }

    // -----------------------------------------------------------------------
    // Mean convergence tests (large n, loose tolerance)
    // -----------------------------------------------------------------------

    #[test]
    fn test_normal_mean_convergence() {
        let mut rng = cjc_repro::Rng::seeded(1000);
        let mu = 3.0;
        let samples = normal_sample(mu, 1.0, 50_000, &mut rng);
        let mean = kahan_mean(&samples);
        assert!(
            (mean - mu).abs() < 0.05,
            "normal mean {mean} not close to {mu}"
        );
    }

    #[test]
    fn test_uniform_mean_convergence() {
        let mut rng = cjc_repro::Rng::seeded(1001);
        let (a, b) = (2.0, 8.0);
        let expected = (a + b) / 2.0;
        let samples = uniform_sample(a, b, 50_000, &mut rng);
        let mean = kahan_mean(&samples);
        assert!(
            (mean - expected).abs() < 0.05,
            "uniform mean {mean} not close to {expected}"
        );
    }

    #[test]
    fn test_exponential_mean_convergence() {
        let mut rng = cjc_repro::Rng::seeded(1002);
        let lambda = 2.0;
        let expected = 1.0 / lambda;
        let samples = exponential_sample(lambda, 50_000, &mut rng);
        let mean = kahan_mean(&samples);
        assert!(
            (mean - expected).abs() < 0.02,
            "exponential mean {mean} not close to {expected}"
        );
    }

    #[test]
    fn test_gamma_mean_convergence() {
        let mut rng = cjc_repro::Rng::seeded(1003);
        let (shape, scale) = (3.0, 2.0);
        let expected = shape * scale;
        let samples = gamma_sample(shape, scale, 50_000, &mut rng);
        let mean = kahan_mean(&samples);
        assert!(
            (mean - expected).abs() < 0.1,
            "gamma mean {mean} not close to {expected}"
        );
    }

    #[test]
    fn test_gamma_small_shape_mean() {
        let mut rng = cjc_repro::Rng::seeded(1004);
        let (shape, scale) = (0.5, 2.0);
        let expected = shape * scale;
        let samples = gamma_sample(shape, scale, 50_000, &mut rng);
        let mean = kahan_mean(&samples);
        assert!(
            (mean - expected).abs() < 0.1,
            "gamma(0.5) mean {mean} not close to {expected}"
        );
    }

    #[test]
    fn test_beta_mean_convergence() {
        let mut rng = cjc_repro::Rng::seeded(1005);
        let (a, b) = (2.0, 5.0);
        let expected = a / (a + b);
        let samples = beta_sample(a, b, 50_000, &mut rng);
        let mean = kahan_mean(&samples);
        assert!(
            (mean - expected).abs() < 0.02,
            "beta mean {mean} not close to {expected}"
        );
    }

    #[test]
    fn test_chi2_mean_convergence() {
        let mut rng = cjc_repro::Rng::seeded(1006);
        let df = 5.0;
        let samples = chi2_sample(df, 50_000, &mut rng);
        let mean = kahan_mean(&samples);
        assert!(
            (mean - df).abs() < 0.1,
            "chi2 mean {mean} not close to df={df}"
        );
    }

    #[test]
    fn test_t_mean_convergence() {
        let mut rng = cjc_repro::Rng::seeded(1007);
        let df = 10.0; // mean is 0 for df > 1
        let samples = t_sample(df, 50_000, &mut rng);
        let mean = kahan_mean(&samples);
        assert!(
            mean.abs() < 0.05,
            "t mean {mean} not close to 0"
        );
    }

    #[test]
    fn test_poisson_mean_convergence() {
        let mut rng = cjc_repro::Rng::seeded(1008);
        let lambda = 7.0;
        let samples = poisson_sample(lambda, 50_000, &mut rng);
        let fsamples: Vec<f64> = samples.iter().map(|&x| x as f64).collect();
        let mean = kahan_mean(&fsamples);
        assert!(
            (mean - lambda).abs() < 0.1,
            "poisson mean {mean} not close to {lambda}"
        );
    }

    #[test]
    fn test_poisson_large_lambda_mean() {
        let mut rng = cjc_repro::Rng::seeded(1009);
        let lambda = 50.0;
        let samples = poisson_sample(lambda, 50_000, &mut rng);
        let fsamples: Vec<f64> = samples.iter().map(|&x| x as f64).collect();
        let mean = kahan_mean(&fsamples);
        assert!(
            (mean - lambda).abs() < 0.5,
            "poisson(50) mean {mean} not close to {lambda}"
        );
    }

    #[test]
    fn test_binomial_mean_convergence() {
        let mut rng = cjc_repro::Rng::seeded(1010);
        let (n_trials, p) = (20, 0.3);
        let expected = n_trials as f64 * p;
        let samples = binomial_sample(n_trials, p, 50_000, &mut rng);
        let fsamples: Vec<f64> = samples.iter().map(|&x| x as f64).collect();
        let mean = kahan_mean(&fsamples);
        assert!(
            (mean - expected).abs() < 0.1,
            "binomial mean {mean} not close to {expected}"
        );
    }

    #[test]
    fn test_bernoulli_mean_convergence() {
        let mut rng = cjc_repro::Rng::seeded(1011);
        let p = 0.7;
        let samples = bernoulli_sample(p, 50_000, &mut rng);
        let fsamples: Vec<f64> = samples.iter().map(|&x| if x { 1.0 } else { 0.0 }).collect();
        let mean = kahan_mean(&fsamples);
        assert!(
            (mean - p).abs() < 0.02,
            "bernoulli mean {mean} not close to {p}"
        );
    }

    #[test]
    fn test_multinomial_frequency_convergence() {
        let mut rng = cjc_repro::Rng::seeded(1012);
        let probs = vec![0.1, 0.2, 0.3, 0.4];
        let n = 50_000;
        let samples = multinomial_sample(&probs, n, &mut rng);
        let mut counts = vec![0usize; probs.len()];
        for &idx in &samples {
            counts[idx] += 1;
        }
        for (i, (&expected_p, &count)) in probs.iter().zip(counts.iter()).enumerate() {
            let empirical_p = count as f64 / n as f64;
            assert!(
                (empirical_p - expected_p).abs() < 0.02,
                "multinomial category {i}: empirical={empirical_p}, expected={expected_p}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Latin Hypercube Sampling tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_lhs_shape() {
        let samples = latin_hypercube_sample(100, 3, 42);
        assert_eq!(samples.shape(), &[100, 3]);
    }

    #[test]
    fn test_lhs_bounds() {
        let samples = latin_hypercube_sample(50, 2, 123);
        for &v in samples.to_vec().iter() {
            assert!(v >= 0.0 && v < 1.0);
        }
    }

    #[test]
    fn test_lhs_stratification() {
        // Each dimension should have exactly one sample per stratum
        let n = 20;
        let samples = latin_hypercube_sample(n, 2, 42);
        let data = samples.to_vec();
        for dim in 0..2 {
            let mut strata = vec![false; n];
            for i in 0..n {
                let val = data[i * 2 + dim];
                let stratum = (val * n as f64) as usize;
                assert!(!strata[stratum], "Stratum {} used twice in dim {}", stratum, dim);
                strata[stratum] = true;
            }
        }
    }

    #[test]
    fn test_lhs_determinism() {
        let s1 = latin_hypercube_sample(50, 3, 42);
        let s2 = latin_hypercube_sample(50, 3, 42);
        assert_eq!(s1.to_vec(), s2.to_vec(), "LHS must be deterministic");
    }

    // -----------------------------------------------------------------------
    // Sobol sequence tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_sobol_shape() {
        let seq = sobol_sequence(100, 4);
        assert_eq!(seq.shape(), &[100, 4]);
    }

    #[test]
    fn test_sobol_bounds() {
        let seq = sobol_sequence(50, 3);
        for &v in seq.to_vec().iter() {
            assert!(v >= 0.0 && v < 1.0);
        }
    }

    #[test]
    fn test_sobol_determinism() {
        let s1 = sobol_sequence(50, 3);
        let s2 = sobol_sequence(50, 3);
        assert_eq!(s1.to_vec(), s2.to_vec());
    }
}

// ---------------------------------------------------------------------------
// Existing tests (PDF/CDF/PPF)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal_cdf_zero() {
        let r = normal_cdf(0.0);
        assert!((r - 0.5).abs() < 1e-6, "CDF(0) = {r}");
    }

    #[test]
    fn test_normal_cdf_196() {
        let r = normal_cdf(1.96);
        assert!((r - 0.975).abs() < 1e-3, "CDF(1.96) = {r}");
    }

    #[test]
    fn test_normal_pdf_zero() {
        let r = normal_pdf(0.0);
        let expected = 1.0 / (2.0 * PI).sqrt();
        assert!((r - expected).abs() < 1e-12);
    }

    #[test]
    fn test_normal_ppf_half() {
        let r = normal_ppf(0.5).unwrap();
        assert!(r.abs() < 1e-6, "PPF(0.5) = {r}");
    }

    #[test]
    fn test_normal_ppf_975() {
        let r = normal_ppf(0.975).unwrap();
        assert!((r - 1.96).abs() < 0.01, "PPF(0.975) = {r}");
    }

    #[test]
    fn test_t_cdf_symmetry() {
        let cdf_pos = t_cdf(0.0, 10.0);
        assert!((cdf_pos - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_chi2_cdf_basic() {
        // chi2(df=1) at x=3.841 should be ~0.95
        let r = chi2_cdf(3.841, 1.0);
        assert!((r - 0.95).abs() < 0.01, "chi2_cdf = {r}");
    }

    #[test]
    fn test_f_cdf_basic() {
        let r = f_cdf(0.0, 5.0, 10.0);
        assert_eq!(r, 0.0);
    }

    #[test]
    fn test_binomial_pmf() {
        // P(X=0) for n=10, p=0.5 = 1/1024
        let r = binomial_pmf(0, 10, 0.5);
        assert!((r - 1.0 / 1024.0).abs() < 1e-12);
    }

    #[test]
    fn test_poisson_pmf() {
        // P(X=0) for lambda=1 = e^-1
        let r = poisson_pmf(0, 1.0);
        assert!((r - (-1.0_f64).exp()).abs() < 1e-12);
    }

    #[test]
    fn test_ln_gamma_basic() {
        // Gamma(1) = 0! = 1, ln(1) = 0
        assert!(ln_gamma(1.0).abs() < 1e-12);
        // Gamma(5) = 4! = 24, ln(24)
        assert!((ln_gamma(5.0) - 24.0_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_determinism() {
        let r1 = normal_cdf(1.5);
        let r2 = normal_cdf(1.5);
        assert_eq!(r1.to_bits(), r2.to_bits());
    }

    // --- B6: New distribution tests ---

    #[test]
    fn test_beta_pdf_symmetric() {
        // Beta(2, 2) at x=0.5 should be 1.5
        let r = beta_pdf(0.5, 2.0, 2.0);
        assert!((r - 1.5).abs() < 1e-10, "beta_pdf(0.5, 2, 2) = {r}");
    }

    #[test]
    fn test_beta_cdf_uniform() {
        // Beta(1, 1) = Uniform[0,1], CDF(x) = x
        for &x in &[0.1, 0.3, 0.5, 0.7, 0.9] {
            let r = beta_cdf(x, 1.0, 1.0);
            assert!((r - x).abs() < 1e-6, "beta_cdf({x}, 1, 1) = {r}");
        }
    }

    #[test]
    fn test_beta_cdf_endpoints() {
        assert!((beta_cdf(0.0, 2.0, 3.0) - 0.0).abs() < 1e-12);
        assert!((beta_cdf(1.0, 2.0, 3.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_gamma_cdf_exponential() {
        // Gamma(1, 1/lambda) ≈ Exp(lambda)
        let lambda = 2.0;
        for &x in &[0.5, 1.0, 2.0] {
            let gc = gamma_cdf(x, 1.0, 1.0 / lambda);
            let ec = exp_cdf(x, lambda);
            assert!((gc - ec).abs() < 1e-6, "gamma_cdf({x}) = {gc}, exp_cdf = {ec}");
        }
    }

    #[test]
    fn test_exp_cdf_memoryless() {
        // exp_cdf(1/lambda, lambda) ≈ 1 - 1/e
        let lambda = 3.0;
        let r = exp_cdf(1.0 / lambda, lambda);
        let expected = 1.0 - (-1.0_f64).exp();
        assert!((r - expected).abs() < 1e-10, "exp_cdf = {r}, expected {expected}");
    }

    #[test]
    fn test_exp_pdf_integral() {
        // Numerical integration of PDF should be ~1.0
        let lambda = 1.5;
        let dx = 0.001;
        let mut sum = 0.0;
        let mut x = 0.0;
        while x < 20.0 {
            sum += exp_pdf(x, lambda) * dx;
            x += dx;
        }
        assert!((sum - 1.0).abs() < 0.01, "integral = {sum}");
    }

    #[test]
    fn test_weibull_cdf_exponential() {
        // Weibull(k=1, lambda) = Exp(1/lambda)
        let lambda = 2.0;
        for &x in &[0.5, 1.0, 3.0] {
            let wc = weibull_cdf(x, 1.0, lambda);
            let ec = exp_cdf(x, 1.0 / lambda);
            assert!((wc - ec).abs() < 1e-10, "weibull_cdf({x}) = {wc}, exp_cdf = {ec}");
        }
    }

    #[test]
    fn test_weibull_pdf_mode() {
        // For k > 1, mode = lambda * ((k-1)/k)^(1/k)
        let k: f64 = 3.0;
        let lambda: f64 = 2.0;
        let mode = lambda * ((k - 1.0) / k).powf(1.0_f64 / k);
        let pdf_at_mode = weibull_pdf(mode, k, lambda);
        // PDF at mode should be a maximum
        let pdf_left = weibull_pdf(mode - 0.01, k, lambda);
        let pdf_right = weibull_pdf(mode + 0.01, k, lambda);
        assert!(pdf_at_mode >= pdf_left, "mode not a max left");
        assert!(pdf_at_mode >= pdf_right, "mode not a max right");
    }

    #[test]
    fn test_b6_dist_determinism() {
        let r1 = beta_pdf(0.3, 2.0, 5.0);
        let r2 = beta_pdf(0.3, 2.0, 5.0);
        assert_eq!(r1.to_bits(), r2.to_bits());
        let r1 = gamma_cdf(1.5, 3.0, 2.0);
        let r2 = gamma_cdf(1.5, 3.0, 2.0);
        assert_eq!(r1.to_bits(), r2.to_bits());
    }

    // -------------------------------------------------------------------
    // erf / erfc tests (Bastion ABI)
    // -------------------------------------------------------------------

    #[test]
    fn test_erf_known_values() {
        // erf(0) = 0
        assert!((erf(0.0)).abs() < 1e-10);
        // erf(+inf) = 1
        assert!((erf(f64::INFINITY) - 1.0).abs() < 1e-10);
        // erf(-inf) = -1
        assert!((erf(f64::NEG_INFINITY) + 1.0).abs() < 1e-10);
        // erf is odd: erf(-x) = -erf(x)
        assert!((erf(1.0) + erf(-1.0)).abs() < 1e-10);
        assert!((erf(0.5) + erf(-0.5)).abs() < 1e-10);
    }

    #[test]
    fn test_erf_reference_values() {
        // Reference values from mathematical tables
        assert!((erf(0.5) - 0.5204998778).abs() < 2e-7);
        assert!((erf(1.0) - 0.8427007929).abs() < 2e-7);
        assert!((erf(1.5) - 0.9661051465).abs() < 2e-7);
        assert!((erf(2.0) - 0.9953222650).abs() < 2e-7);
        assert!((erf(3.0) - 0.9999779095).abs() < 2e-7);
    }

    #[test]
    fn test_erfc_known_values() {
        // erfc(0) = 1
        assert!((erfc(0.0) - 1.0).abs() < 1e-10);
        // erfc(+inf) = 0
        assert!((erfc(f64::INFINITY)).abs() < 1e-10);
        // erfc(-inf) = 2
        assert!((erfc(f64::NEG_INFINITY) - 2.0).abs() < 1e-10);
        // erfc(x) + erfc(-x) = 2
        assert!((erfc(1.0) + erfc(-1.0) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_erfc_reference_values() {
        assert!((erfc(0.5) - 0.4795001222).abs() < 2e-7);
        assert!((erfc(1.0) - 0.1572992071).abs() < 2e-7);
        assert!((erfc(2.0) - 0.0046777350).abs() < 2e-7);
    }

    #[test]
    fn test_erf_erfc_consistency() {
        // erf(x) + erfc(x) = 1 for all x
        for &x in &[0.0, 0.1, 0.5, 1.0, 1.5, 2.0, 3.0, -0.5, -1.0, -2.0] {
            assert!(
                (erf(x) + erfc(x) - 1.0).abs() < 1e-12,
                "erf({x}) + erfc({x}) != 1: got {}",
                erf(x) + erfc(x)
            );
        }
    }

    #[test]
    fn test_erf_nan() {
        assert!(erf(f64::NAN).is_nan());
        assert!(erfc(f64::NAN).is_nan());
    }

    #[test]
    fn test_erf_normal_cdf_consistency() {
        // normal_cdf(x) should agree with 0.5 * erfc(-x / sqrt(2))
        // Both use the same A&S 7.1.26 approximation but may differ slightly
        // due to independent evaluation paths. Tolerance: 2e-7 (within A&S error bound).
        for &x in &[-3.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0] {
            let via_erfc = 0.5 * erfc(-x / 2.0_f64.sqrt());
            let via_cdf = normal_cdf(x);
            assert!(
                (via_erfc - via_cdf).abs() < 2e-7,
                "normal_cdf({x}) vs erfc route: cdf={via_cdf}, erfc={via_erfc}, diff={}",
                (via_erfc - via_cdf).abs()
            );
        }
    }

    #[test]
    fn test_erf_determinism() {
        let a = erf(1.23456789);
        let b = erf(1.23456789);
        assert_eq!(a.to_bits(), b.to_bits());
        let a = erfc(1.23456789);
        let b = erfc(1.23456789);
        assert_eq!(a.to_bits(), b.to_bits());
    }
}
