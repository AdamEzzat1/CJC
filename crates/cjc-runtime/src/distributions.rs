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
// Normal distribution
// ---------------------------------------------------------------------------

/// Normal distribution CDF using Abramowitz & Stegun approximation (7.1.26).
/// Maximum error: |eps| < 1.5e-7. Deterministic.
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

// ---------------------------------------------------------------------------
// Tests
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
}
