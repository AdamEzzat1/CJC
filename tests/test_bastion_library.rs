//! Integration tests for Bastion pure CJC library functions.
//! Each test inlines the library function and verifies correctness
//! through the MIR executor pipeline (parse -> HIR -> MIR -> exec).

fn parse(src: &str) -> cjc_ast::Program {
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse errors: {:?}", diags.diagnostics);
    program
}

fn mir_output(src: &str) -> Vec<String> {
    let program = parse(src);
    let (_, exec) = cjc_mir_exec::run_program_with_executor(&program, 42).unwrap();
    exec.output.clone()
}

fn assert_close(actual: &str, expected: f64, tol: f64) {
    let v: f64 = actual.parse().unwrap_or_else(|_| panic!("Cannot parse '{actual}' as f64"));
    assert!(
        (v - expected).abs() < tol,
        "Expected ~{expected}, got {v} (tol={tol})"
    );
}

// ═══════════════════════════════════════════════════════════════
// descriptive.cjc tests
// ═══════════════════════════════════════════════════════════════

#[test]
fn bastion_lib_cummean() {
    let out = mir_output(r#"
fn cummean(xs: Any) -> Any {
    let cs = cumsum(xs);
    let n = len(xs);
    let out = [];
    let i = 0;
    while i < n {
        out = array_push(out, cs[i] / float(i + 1));
        i = i + 1;
    }
    out
}
let data = [2.0, 4.0, 6.0, 8.0];
let cm = cummean(data);
print(cm[0]);
print(cm[1]);
print(cm[2]);
print(cm[3]);
"#);
    assert_close(&out[0], 2.0, 1e-10);
    assert_close(&out[1], 3.0, 1e-10);
    assert_close(&out[2], 4.0, 1e-10);
    assert_close(&out[3], 5.0, 1e-10);
}

#[test]
fn bastion_lib_diff() {
    let out = mir_output(r#"
fn diff(xs: Any, lag: i64) -> Any {
    let n = len(xs);
    let out = [];
    let i = lag;
    while i < n {
        out = array_push(out, xs[i] - xs[i - lag]);
        i = i + 1;
    }
    out
}
let data = [1.0, 3.0, 6.0, 10.0, 15.0];
let d = diff(data, 1);
print(d[0]);
print(d[1]);
print(d[2]);
print(d[3]);
"#);
    assert_close(&out[0], 2.0, 1e-10);
    assert_close(&out[1], 3.0, 1e-10);
    assert_close(&out[2], 4.0, 1e-10);
    assert_close(&out[3], 5.0, 1e-10);
}

#[test]
fn bastion_lib_pct_change() {
    let out = mir_output(r#"
fn pct_change(xs: Any, lag: i64) -> Any {
    let n = len(xs);
    let out = [];
    let i = lag;
    while i < n {
        let prev = xs[i - lag];
        if prev != 0.0 {
            out = array_push(out, (xs[i] - prev) / prev);
        } else {
            out = array_push(out, 0.0 / 0.0);
        }
        i = i + 1;
    }
    out
}
let data = [100.0, 110.0, 99.0];
let pc = pct_change(data, 1);
print(pc[0]);
print(pc[1]);
"#);
    assert_close(&out[0], 0.1, 1e-10);   // 110/100 - 1 = 0.1
    assert_close(&out[1], -0.1, 1e-10);  // 99/110 - 1 = -0.1
}

// ═══════════════════════════════════════════════════════════════
// rolling.cjc tests
// ═══════════════════════════════════════════════════════════════

#[test]
fn bastion_lib_ewma() {
    let out = mir_output(r#"
fn ewma(xs: Any, alpha: f64) -> Any {
    let n = len(xs);
    if n == 0 { return []; }
    let out = [];
    let prev = xs[0];
    out = array_push(out, prev);
    let i = 1;
    while i < n {
        prev = alpha * xs[i] + (1.0 - alpha) * prev;
        out = array_push(out, prev);
        i = i + 1;
    }
    out
}
let data = [1.0, 2.0, 3.0, 4.0, 5.0];
let e = ewma(data, 0.5);
print(e[0]);
print(e[1]);
print(e[2]);
"#);
    assert_close(&out[0], 1.0, 1e-10);    // first = 1.0
    assert_close(&out[1], 1.5, 1e-10);    // 0.5*2 + 0.5*1 = 1.5
    assert_close(&out[2], 2.25, 1e-10);   // 0.5*3 + 0.5*1.5 = 2.25
}

#[test]
fn bastion_lib_rolling_var() {
    let out = mir_output(r#"
fn rolling_var(xs: Any, w: i64) -> Any {
    let n = len(xs);
    let xsq = [];
    let i = 0;
    while i < n {
        xsq = array_push(xsq, xs[i] * xs[i]);
        i = i + 1;
    }
    let means = window_mean(xs, w);
    let mean_sq = window_mean(xsq, w);
    let nm = len(means);
    let out = [];
    let j = 0;
    while j < nm {
        let v = mean_sq[j] - means[j] * means[j];
        if v < 0.0 { v = 0.0; }
        out = array_push(out, v);
        j = j + 1;
    }
    out
}
let data = [1.0, 2.0, 3.0, 4.0, 5.0];
let rv = rolling_var(data, 3);
print(rv[0]);
print(rv[1]);
print(rv[2]);
"#);
    // Window [1,2,3]: mean=2, var = (1+4+9)/3 - 4 = 14/3 - 4 = 2/3
    assert_close(&out[0], 2.0/3.0, 1e-10);
    // Window [2,3,4]: mean=3, var = (4+9+16)/3 - 9 = 29/3 - 9 = 2/3
    assert_close(&out[1], 2.0/3.0, 1e-10);
}

// ═══════════════════════════════════════════════════════════════
// robust.cjc tests
// ═══════════════════════════════════════════════════════════════

#[test]
fn bastion_lib_mad_std() {
    let out = mir_output(r#"
fn mad_std(xs: Any) -> f64 {
    mad(xs) * 1.4826
}
let data = [1.0, 2.0, 3.0, 4.0, 5.0];
print(mad_std(data));
"#);
    // median=3, deviations=[2,1,0,1,2], median_dev=1, MAD=1
    // mad_std = 1.4826
    assert_close(&out[0], 1.4826, 1e-3);
}

#[test]
fn bastion_lib_huber_location() {
    let out = mir_output(r#"
fn huber_location(xs: Any, c: f64, tol: f64) -> f64 {
    let n = len(xs);
    let mu = median_fast(xs);
    let max_iter = 50;
    let iter = 0;
    while iter < max_iter {
        let wsum = 0.0;
        let wgt_total = 0.0;
        let i = 0;
        while i < n {
            let r = xs[i] - mu;
            let w = 1.0;
            if abs(r) > c {
                w = c / abs(r);
            }
            wsum = wsum + w * xs[i];
            wgt_total = wgt_total + w;
            i = i + 1;
        }
        let mu_new = wsum / wgt_total;
        if abs(mu_new - mu) < tol {
            return mu_new;
        }
        mu = mu_new;
        iter = iter + 1;
    }
    mu
}
let data = [1.0, 2.0, 3.0, 4.0, 100.0];
let h = huber_location(data, 1.5, 1e-6);
print(h);
"#);
    // With outlier 100, Huber should give robust location near 2.5-3.5
    let v: f64 = out[0].parse().unwrap();
    assert!(v > 1.5 && v < 5.0, "Huber location should be robust: got {v}");
}

#[test]
fn bastion_lib_biweight_midvariance() {
    let out = mir_output(r#"
fn biweight_midvariance(xs: Any) -> f64 {
    let n = len(xs);
    let med = median_fast(xs);
    let s = mad(xs) * 1.4826;
    if s == 0.0 { return 0.0; }
    let c = 9.0;
    let num = 0.0;
    let den = 0.0;
    let i = 0;
    while i < n {
        let u = (xs[i] - med) / (c * s);
        let u2 = u * u;
        if u2 < 1.0 {
            let d = xs[i] - med;
            let w = (1.0 - u2);
            num = num + d * d * w * w * w * w;
            den = den + w * (1.0 - 5.0 * u2);
        }
        i = i + 1;
    }
    let den_sq = den * den;
    if den_sq == 0.0 { return 0.0; }
    float(n) * num / den_sq
}
let data = [1.0, 2.0, 3.0, 4.0, 5.0];
let bv = biweight_midvariance(data);
print(bv);
"#);
    let v: f64 = out[0].parse().unwrap();
    assert!(v > 0.0, "Biweight midvariance should be positive: got {v}");
    assert!(v < 10.0, "Biweight midvariance for small data should be bounded: got {v}");
}

// ═══════════════════════════════════════════════════════════════
// resampling.cjc tests
// ═══════════════════════════════════════════════════════════════

#[test]
fn bastion_lib_bootstrap_mean() {
    let out = mir_output(r#"
fn bootstrap_mean(xs: Any, b: i64, seed: i64) -> f64 {
    let n = len(xs);
    let total = 0.0;
    let i = 0;
    while i < b {
        let idx = sample_indices(n, n, true, seed + i);
        let s = 0.0;
        let j = 0;
        while j < n {
            s = s + xs[idx[j]];
            j = j + 1;
        }
        total = total + s / float(n);
        i = i + 1;
    }
    total / float(b)
}
let data = [1.0, 2.0, 3.0, 4.0, 5.0];
let bm = bootstrap_mean(data, 100, 42);
print(bm);
"#);
    // Bootstrap mean should be near population mean (3.0) with some noise
    assert_close(&out[0], 3.0, 0.5);
}

#[test]
fn bastion_lib_jackknife_mean() {
    let out = mir_output(r#"
fn jackknife_mean(xs: Any) -> Any {
    let n = len(xs);
    let total = 0.0;
    let i = 0;
    while i < n {
        total = total + xs[i];
        i = i + 1;
    }
    let orig_mean = total / float(n);
    let loo_means = [];
    let j = 0;
    while j < n {
        let loo_sum = total - xs[j];
        loo_means = array_push(loo_means, loo_sum / float(n - 1));
        j = j + 1;
    }
    let jk_mean = mean(loo_means);
    let bias = float(n - 1) * (jk_mean - orig_mean);
    let ssq = 0.0;
    let k = 0;
    while k < n {
        let d = loo_means[k] - jk_mean;
        ssq = ssq + d * d;
        k = k + 1;
    }
    let se = sqrt(float(n - 1) / float(n) * ssq);
    [jk_mean, bias, se]
}
let data = [1.0, 2.0, 3.0, 4.0, 5.0];
let jk = jackknife_mean(data);
print(jk[0]);
print(jk[1]);
"#);
    assert_close(&out[0], 3.0, 1e-10);  // Jackknife mean should equal sample mean
    assert_close(&out[1], 0.0, 1e-10);  // Bias should be 0 for linear statistic
}

// ═══════════════════════════════════════════════════════════════
// tsa.cjc tests
// ═══════════════════════════════════════════════════════════════

#[test]
fn bastion_lib_acf() {
    let out = mir_output(r#"
fn acovf(xs: Any, nlags: i64) -> Any {
    let n = len(xs);
    let mu = mean(xs);
    let out = [];
    let k = 0;
    while k <= nlags {
        let s = 0.0;
        let i = 0;
        while i < n - k {
            s = s + (xs[i] - mu) * (xs[i + k] - mu);
            i = i + 1;
        }
        out = array_push(out, s / float(n));
        k = k + 1;
    }
    out
}
fn acf(xs: Any, nlags: i64) -> Any {
    let gamma = acovf(xs, nlags);
    let g0 = gamma[0];
    if g0 == 0.0 {
        let out = [1.0];
        let k = 1;
        while k <= nlags {
            out = array_push(out, 0.0);
            k = k + 1;
        }
        return out;
    }
    let out = [];
    let k = 0;
    while k <= nlags {
        out = array_push(out, gamma[k] / g0);
        k = k + 1;
    }
    out
}
let data = [1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0, 2.0];
let r = acf(data, 3);
print(r[0]);
"#);
    assert_close(&out[0], 1.0, 1e-10); // ACF at lag 0 is always 1.0
}

#[test]
fn bastion_lib_durbin_watson() {
    let out = mir_output(r#"
fn durbin_watson(residuals: Any) -> f64 {
    let n = len(residuals);
    if n < 2 { return 0.0 / 0.0; }
    let num = 0.0;
    let den = 0.0;
    let i = 0;
    while i < n {
        den = den + residuals[i] * residuals[i];
        if i > 0 {
            let d = residuals[i] - residuals[i - 1];
            num = num + d * d;
        }
        i = i + 1;
    }
    if den == 0.0 { return 0.0 / 0.0; }
    num / den
}
let residuals = [0.5, -0.3, 0.2, -0.1, 0.4, -0.5, 0.1];
let dw = durbin_watson(residuals);
print(dw);
"#);
    // DW should be between 0 and 4, near 2 for no autocorrelation
    let v: f64 = out[0].parse().unwrap();
    assert!(v > 0.0 && v < 4.0, "Durbin-Watson should be in [0,4]: got {v}");
}

// ═══════════════════════════════════════════════════════════════
// infer.cjc tests
// ═══════════════════════════════════════════════════════════════

#[test]
fn bastion_lib_cohens_d() {
    let out = mir_output(r#"
fn cohens_d(xs: Any, ys: Any) -> f64 {
    let nx = len(xs);
    let ny = len(ys);
    let mx = mean(xs);
    let my = mean(ys);
    let vx = variance(xs);
    let vy = variance(ys);
    let sp = sqrt((float(nx - 1) * vx + float(ny - 1) * vy) / float(nx + ny - 2));
    if sp == 0.0 { return 0.0; }
    (mx - my) / sp
}
let xs = [1.0, 2.0, 3.0, 4.0, 5.0];
let ys = [3.0, 4.0, 5.0, 6.0, 7.0];
let d = cohens_d(xs, ys);
print(d);
"#);
    // Both have same variance, means differ by 2.0, sd ~ 1.58
    // d = -2.0 / 1.58 ~ -1.26
    assert_close(&out[0], -2.0 / (10.0_f64 / 4.0).sqrt(), 0.01);
}

#[test]
fn bastion_lib_odds_ratio() {
    let out = mir_output(r#"
fn odds_ratio(a: f64, b: f64, c: f64, d: f64) -> Any {
    let or_val = (a * d) / (b * c);
    let log_or = log(or_val);
    let se = sqrt(1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d);
    let lo = exp(log_or - 1.96 * se);
    let hi = exp(log_or + 1.96 * se);
    [or_val, lo, hi]
}
let result = odds_ratio(20.0, 10.0, 5.0, 15.0);
print(result[0]);
"#);
    // OR = (20*15)/(10*5) = 300/50 = 6.0
    assert_close(&out[0], 6.0, 1e-10);
}

// ═══════════════════════════════════════════════════════════════
// dist.cjc tests
// ═══════════════════════════════════════════════════════════════

#[test]
fn bastion_lib_unif_cdf() {
    let out = mir_output(r#"
fn unif_cdf(x: f64, a: f64, b: f64) -> f64 {
    if x < a { return 0.0; }
    if x > b { return 1.0; }
    (x - a) / (b - a)
}
print(unif_cdf(0.5, 0.0, 1.0));
print(unif_cdf(-0.5, 0.0, 1.0));
print(unif_cdf(1.5, 0.0, 1.0));
"#);
    assert_close(&out[0], 0.5, 1e-10);
    assert_close(&out[1], 0.0, 1e-10);
    assert_close(&out[2], 1.0, 1e-10);
}

#[test]
fn bastion_lib_norm_sf() {
    let out = mir_output(r#"
fn norm_sf(x: f64) -> f64 {
    1.0 - normal_cdf(x)
}
print(norm_sf(0.0));
"#);
    assert_close(&out[0], 0.5, 1e-6);
}

#[test]
fn bastion_lib_exp_ppf() {
    let out = mir_output(r#"
fn exp_ppf(q: f64, rate: f64) -> f64 {
    if q < 0.0 { return 0.0 / 0.0; }
    if q >= 1.0 { return 1.0 / 0.0; }
    -log(1.0 - q) / rate
}
print(exp_ppf(0.5, 1.0));
"#);
    // exp_ppf(0.5, 1) = -ln(0.5) = ln(2) ~ 0.6931
    assert_close(&out[0], 2.0_f64.ln(), 1e-6);
}

// ═══════════════════════════════════════════════════════════════
// transform.cjc tests
// ═══════════════════════════════════════════════════════════════

#[test]
fn bastion_lib_minmax_scale() {
    let out = mir_output(r#"
fn minmax_scale(xs: Any) -> Any {
    let n = len(xs);
    let lo = xs[0];
    let hi = xs[0];
    let i = 1;
    while i < n {
        if xs[i] < lo { lo = xs[i]; }
        if xs[i] > hi { hi = xs[i]; }
        i = i + 1;
    }
    let rng = hi - lo;
    let out = [];
    let j = 0;
    while j < n {
        if rng > 0.0 {
            out = array_push(out, (xs[j] - lo) / rng);
        } else {
            out = array_push(out, 0.0);
        }
        j = j + 1;
    }
    out
}
let data = [0.0, 5.0, 10.0];
let s = minmax_scale(data);
print(s[0]);
print(s[1]);
print(s[2]);
"#);
    assert_close(&out[0], 0.0, 1e-10);
    assert_close(&out[1], 0.5, 1e-10);
    assert_close(&out[2], 1.0, 1e-10);
}

#[test]
fn bastion_lib_demean() {
    let out = mir_output(r#"
fn demean(xs: Any) -> Any {
    let n = len(xs);
    let mu = mean(xs);
    let out = [];
    let i = 0;
    while i < n {
        out = array_push(out, xs[i] - mu);
        i = i + 1;
    }
    out
}
let data = [2.0, 4.0, 6.0];
let d = demean(data);
print(d[0]);
print(d[1]);
print(d[2]);
"#);
    assert_close(&out[0], -2.0, 1e-10);
    assert_close(&out[1], 0.0, 1e-10);
    assert_close(&out[2], 2.0, 1e-10);
}

// ═══════════════════════════════════════════════════════════════
// Spectral analysis tests
// ═══════════════════════════════════════════════════════════════

#[test]
fn bastion_lib_spectral_entropy() {
    let out = mir_output(r#"
fn spectral_entropy(psd: Any) -> f64 {
    let n = len(psd);
    let total = 0.0;
    let i = 0;
    while i < n {
        total = total + psd[i];
        i = i + 1;
    }
    if total == 0.0 { return 0.0; }
    let entropy = 0.0;
    let j = 0;
    while j < n {
        let p = psd[j] / total;
        if p > 0.0 {
            entropy = entropy - p * log(p);
        }
        j = j + 1;
    }
    if n > 1 {
        entropy = entropy / log(float(n));
    }
    entropy
}
// Uniform PSD (maximum entropy)
let psd = [1.0, 1.0, 1.0, 1.0];
print(spectral_entropy(psd));
"#);
    // Uniform distribution should have entropy = 1.0 (normalized)
    assert_close(&out[0], 1.0, 1e-10);
}

#[test]
fn bastion_lib_fractional_diff() {
    let out = mir_output(r#"
fn fractional_diff(xs: Any, d: f64, thresh: f64) -> Any {
    let n = len(xs);
    let weights = [1.0];
    let k = 1;
    while k < n {
        let w = weights[k - 1] * (d - float(k) + 1.0) / float(k);
        if abs(w) < thresh {
            k = n;
        } else {
            weights = array_push(weights, w);
            k = k + 1;
        }
    }
    let nw = len(weights);
    let out = [];
    let i = nw - 1;
    while i < n {
        let s = 0.0;
        let j = 0;
        while j < nw {
            s = s + weights[j] * xs[i - j];
            j = j + 1;
        }
        out = array_push(out, s);
        i = i + 1;
    }
    out
}
let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
let fd = fractional_diff(data, 0.5, 1e-4);
print(len(fd));
"#);
    let n: i64 = out[0].parse().unwrap();
    assert!(n > 0, "Fractional diff should produce output");
}
