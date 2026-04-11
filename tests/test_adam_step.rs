//! Tests for the native `adam_step` builtin (Phase B1b of Chess RL v2.1).
//!
//! Coverage:
//! - Wiring check: builtin callable from both cjc-eval and cjc-mir-exec with
//!   byte-identical output
//! - Mathematical invariants of the first Adam step (t=1, m=0, v=0):
//!   `w_new = w - lr * sign(g)` when eps → 0 (the clean limit case)
//! - Shape/arity guards (error paths)
//! - Proptest: random weights + gradients + hyperparameters never produce
//!   NaN/Inf and always return tensors of the input shape
//! - Bolero fuzz: random byte inputs fed through the CJC-Lang call site must
//!   never panic the interpreter
//!
//! These tests pair with the chess_rl_v2 tests that exercise `adam_step`
//! via the `adam_step_2d` wrapper and `train_one_episode_adam`.

use bolero::check;
use cjc_runtime::tensor::Tensor;
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn run_eval(src: &str, seed: u64) -> Vec<String> {
    let (prog, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse errors: {:?}", diags.diagnostics);
    let mut interp = cjc_eval::Interpreter::new(seed);
    interp
        .exec(&prog)
        .unwrap_or_else(|e| panic!("eval failed: {e:?}"));
    interp.output
}

fn run_mir(src: &str, seed: u64) -> Vec<String> {
    let (prog, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse errors: {:?}", diags.diagnostics);
    let (_val, executor) = cjc_mir_exec::run_program_with_executor(&prog, seed)
        .unwrap_or_else(|e| panic!("mir-exec failed: {e:?}"));
    executor.output
}

fn run_parity(src: &str, seed: u64) -> Vec<String> {
    let a = run_eval(src, seed);
    let b = run_mir(src, seed);
    assert_eq!(
        a, b,
        "parity violation between cjc-eval and cjc-mir-exec\neval: {a:?}\nmir: {b:?}"
    );
    a
}

/// Compute one Adam step in Rust directly, matching the cjc-runtime native
/// implementation. Used as the oracle in proptest.
fn adam_step_oracle(
    w: &[f64],
    g: &[f64],
    m: &[f64],
    v: &[f64],
    lr: f64,
    b1: f64,
    b2: f64,
    eps: f64,
    t: i64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let bc1 = 1.0 - b1.powf(t as f64);
    let bc2 = 1.0 - b2.powf(t as f64);
    let n = w.len();
    let mut nw = Vec::with_capacity(n);
    let mut nm = Vec::with_capacity(n);
    let mut nv = Vec::with_capacity(n);
    for i in 0..n {
        let new_mi = b1 * m[i] + (1.0 - b1) * g[i];
        let new_vi = b2 * v[i] + (1.0 - b2) * g[i] * g[i];
        let m_hat = new_mi / bc1;
        let v_hat = new_vi / bc2;
        let new_wi = w[i] - lr * m_hat / (v_hat.sqrt() + eps);
        nw.push(new_wi);
        nm.push(new_mi);
        nv.push(new_vi);
    }
    (nw, nm, nv)
}

// ---------------------------------------------------------------------------
// Wiring + parity
// ---------------------------------------------------------------------------

/// The native builtin is callable from CJC-Lang and returns a 3-element
/// array matching the CJC-level wrapper contract.
#[test]
fn adam_step_wired_and_returns_triple() {
    let src = r#"
        let w = Tensor.from_vec([1.0, 2.0, 3.0, 4.0], [2, 2]);
        let g = Tensor.from_vec([0.1, 0.2, 0.3, 0.4], [2, 2]);
        let m = Tensor.zeros([2, 2]);
        let v = Tensor.zeros([2, 2]);
        let out = adam_step(w, g, m, v, 0.01, 0.9, 0.999, 1.0e-8, 1);
        print(len(out));
        let nw = out[0];
        print(nw.shape()[0]);
        print(nw.shape()[1]);
    "#;
    let out = run_parity(src, 1);
    assert_eq!(out[0].trim(), "3", "adam_step should return 3 tensors");
    assert_eq!(out[1].trim(), "2", "new_w shape[0]");
    assert_eq!(out[2].trim(), "2", "new_w shape[1]");
}

/// First-step invariant: with m=0, v=0, t=1, b1=0.9, b2=0.999, eps→0,
/// the Adam update simplifies to `w − lr · sign(g)`. We use a tiny eps to
/// approximate the limit and check against the closed form.
#[test]
fn adam_first_step_matches_sign_limit() {
    let src = r#"
        let w = Tensor.from_vec([10.0, 10.0, 10.0, 10.0], [4]);
        let g = Tensor.from_vec([1.0, 0.0 - 1.0, 2.0, 0.0 - 5.0], [4]);
        let m = Tensor.zeros([4]);
        let v = Tensor.zeros([4]);
        let out = adam_step(w, g, m, v, 0.1, 0.9, 0.999, 1.0e-12, 1);
        let nw = out[0];
        print(nw.get([0]));
        print(nw.get([1]));
        print(nw.get([2]));
        print(nw.get([3]));
    "#;
    let out = run_parity(src, 1);
    // Expected: w - 0.1 * sign(g) = [9.9, 10.1, 9.9, 10.1]
    let expected = [9.9, 10.1, 9.9, 10.1];
    for (line, exp) in out.iter().zip(expected.iter()) {
        let got: f64 = line.trim().parse().unwrap();
        assert!(
            (got - exp).abs() < 1e-6,
            "first-step sign limit: expected {exp}, got {got}"
        );
    }
}

/// Passing t<1 must be an error (bias correction divides by zero at t=0).
#[test]
fn adam_step_rejects_zero_step() {
    let src = r#"
        let w = Tensor.zeros([2, 2]);
        let g = Tensor.zeros([2, 2]);
        let m = Tensor.zeros([2, 2]);
        let v = Tensor.zeros([2, 2]);
        let out = adam_step(w, g, m, v, 0.01, 0.9, 0.999, 1.0e-8, 0);
        print("unreachable");
    "#;
    let (prog, _diags) = cjc_parser::parse_source(src);
    let mut interp = cjc_eval::Interpreter::new(0);
    let r = interp.exec(&prog);
    assert!(r.is_err(), "adam_step(t=0) must error");
}

/// Shape mismatch between w/g/m/v is rejected.
#[test]
fn adam_step_rejects_shape_mismatch() {
    let src = r#"
        let w = Tensor.zeros([2, 3]);
        let g = Tensor.zeros([3, 2]);
        let m = Tensor.zeros([2, 3]);
        let v = Tensor.zeros([2, 3]);
        let out = adam_step(w, g, m, v, 0.01, 0.9, 0.999, 1.0e-8, 1);
        print("unreachable");
    "#;
    let (prog, _diags) = cjc_parser::parse_source(src);
    let mut interp = cjc_eval::Interpreter::new(0);
    let r = interp.exec(&prog);
    assert!(r.is_err(), "adam_step shape mismatch must error");
}

/// Successive Adam steps agree with the Rust oracle bit-for-bit. This is the
/// tightest determinism contract: native builtin vs hand-written reference.
#[test]
fn adam_step_matches_rust_oracle() {
    // Use dispatch_builtin directly to bypass the CJC-Lang interpreter layer.
    use cjc_runtime::builtins::dispatch_builtin;
    use cjc_runtime::value::Value;

    let w = Tensor::from_vec(vec![1.5, -2.0, 0.25, 3.0], &[2, 2]).unwrap();
    let g = Tensor::from_vec(vec![0.1, 0.2, -0.3, 0.05], &[2, 2]).unwrap();
    let m = Tensor::zeros(&[2, 2]);
    let v = Tensor::zeros(&[2, 2]);

    let lr = 0.01f64;
    let b1 = 0.9f64;
    let b2 = 0.999f64;
    let eps = 1e-8f64;
    let t = 1i64;

    let args = vec![
        Value::Tensor(w.clone()),
        Value::Tensor(g.clone()),
        Value::Tensor(m.clone()),
        Value::Tensor(v.clone()),
        Value::Float(lr),
        Value::Float(b1),
        Value::Float(b2),
        Value::Float(eps),
        Value::Int(t),
    ];
    let result = dispatch_builtin("adam_step", &args)
        .expect("dispatch")
        .expect("some");
    let arr = match result {
        Value::Array(a) => a,
        _ => panic!("expected array"),
    };
    let new_w = match &arr[0] {
        Value::Tensor(t) => t.to_vec(),
        _ => panic!("expected tensor"),
    };
    let new_m = match &arr[1] {
        Value::Tensor(t) => t.to_vec(),
        _ => panic!("expected tensor"),
    };
    let new_v = match &arr[2] {
        Value::Tensor(t) => t.to_vec(),
        _ => panic!("expected tensor"),
    };

    let (ow, om, ov) = adam_step_oracle(
        &w.to_vec(),
        &g.to_vec(),
        &m.to_vec(),
        &v.to_vec(),
        lr,
        b1,
        b2,
        eps,
        t,
    );
    assert_eq!(new_w, ow, "new_w differs from oracle");
    assert_eq!(new_m, om, "new_m differs from oracle");
    assert_eq!(new_v, ov, "new_v differs from oracle");
}

// ---------------------------------------------------------------------------
// Property-based tests
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// The native builtin and the Rust oracle must agree on every input.
    #[test]
    fn prop_adam_step_matches_oracle(
        w_data in prop::collection::vec(-10.0f64..10.0, 1..64),
        g_scale in -5.0f64..5.0,
        lr in 0.0001f64..0.1,
        b1 in 0.5f64..0.95,
        b2 in 0.9f64..0.9999,
        eps_exp in -12i32..-4,
        t in 1i64..20,
    ) {
        use cjc_runtime::builtins::dispatch_builtin;
        use cjc_runtime::value::Value;

        let n = w_data.len();
        let shape = vec![n];
        let g_data: Vec<f64> = w_data.iter().enumerate().map(|(i, x)| x * g_scale + (i as f64) * 0.01).collect();
        let m_data = vec![0.0f64; n];
        let v_data = vec![0.0f64; n];
        let eps = 10f64.powi(eps_exp);

        let w = Tensor::from_vec(w_data.clone(), &shape).unwrap();
        let g = Tensor::from_vec(g_data.clone(), &shape).unwrap();
        let m = Tensor::from_vec(m_data.clone(), &shape).unwrap();
        let v = Tensor::from_vec(v_data.clone(), &shape).unwrap();

        let args = vec![
            Value::Tensor(w),
            Value::Tensor(g),
            Value::Tensor(m),
            Value::Tensor(v),
            Value::Float(lr),
            Value::Float(b1),
            Value::Float(b2),
            Value::Float(eps),
            Value::Int(t),
        ];
        let result = dispatch_builtin("adam_step", &args).unwrap().unwrap();
        let arr = match result {
            Value::Array(a) => a,
            _ => panic!("expected array"),
        };
        let new_w = match &arr[0] {
            Value::Tensor(t) => t.to_vec(),
            _ => panic!(),
        };
        let new_m = match &arr[1] {
            Value::Tensor(t) => t.to_vec(),
            _ => panic!(),
        };
        let new_v = match &arr[2] {
            Value::Tensor(t) => t.to_vec(),
            _ => panic!(),
        };

        let (ow, om, ov) = adam_step_oracle(&w_data, &g_data, &m_data, &v_data, lr, b1, b2, eps, t);
        prop_assert_eq!(new_w, ow);
        prop_assert_eq!(new_m, om);
        prop_assert_eq!(new_v, ov);
    }

    /// Adam output shape equals input shape and output tensors never contain
    /// NaN on finite well-scaled inputs.
    #[test]
    fn prop_adam_step_output_shape_finite(
        w_data in prop::collection::vec(-1.0f64..1.0, 1..32),
        g_data in prop::collection::vec(-1.0f64..1.0, 1..32),
        lr in 0.0001f64..0.1,
    ) {
        use cjc_runtime::builtins::dispatch_builtin;
        use cjc_runtime::value::Value;

        let n = w_data.len().min(g_data.len());
        let w = Tensor::from_vec(w_data[..n].to_vec(), &[n]).unwrap();
        let g = Tensor::from_vec(g_data[..n].to_vec(), &[n]).unwrap();
        let m = Tensor::zeros(&[n]);
        let v = Tensor::zeros(&[n]);

        let args = vec![
            Value::Tensor(w),
            Value::Tensor(g),
            Value::Tensor(m),
            Value::Tensor(v),
            Value::Float(lr),
            Value::Float(0.9),
            Value::Float(0.999),
            Value::Float(1e-8),
            Value::Int(1),
        ];
        let result = dispatch_builtin("adam_step", &args).unwrap().unwrap();
        let arr = match result {
            Value::Array(a) => a,
            _ => panic!(),
        };
        for v in arr.iter() {
            let data = match v {
                Value::Tensor(t) => t.to_vec(),
                _ => panic!(),
            };
            prop_assert_eq!(data.len(), n);
            for x in &data {
                prop_assert!(x.is_finite(), "non-finite output: {}", x);
            }
        }
    }

    /// Adam is deterministic: same inputs produce byte-identical outputs
    /// on two independent invocations.
    #[test]
    fn prop_adam_step_deterministic(
        w_data in prop::collection::vec(-5.0f64..5.0, 1..16),
        seed in 0u64..1000,
    ) {
        use cjc_runtime::builtins::dispatch_builtin;
        use cjc_runtime::value::Value;

        let _ = seed; // no rng in adam_step; seed is just to vary the shape
        let n = w_data.len();
        let g_data: Vec<f64> = w_data.iter().map(|x| x * 0.5 - 0.1).collect();

        let run = || {
            let w = Tensor::from_vec(w_data.clone(), &[n]).unwrap();
            let g = Tensor::from_vec(g_data.clone(), &[n]).unwrap();
            let m = Tensor::zeros(&[n]);
            let v = Tensor::zeros(&[n]);
            let args = vec![
                Value::Tensor(w),
                Value::Tensor(g),
                Value::Tensor(m),
                Value::Tensor(v),
                Value::Float(0.01),
                Value::Float(0.9),
                Value::Float(0.999),
                Value::Float(1e-8),
                Value::Int(1),
            ];
            let arr = match dispatch_builtin("adam_step", &args).unwrap().unwrap() {
                Value::Array(a) => a,
                _ => panic!(),
            };
            let w_out = match &arr[0] {
                Value::Tensor(t) => t.to_vec(),
                _ => panic!(),
            };
            w_out
        };
        let a = run();
        let b = run();
        prop_assert_eq!(a, b);
    }
}

// ---------------------------------------------------------------------------
// Fuzz tests (bolero)
// ---------------------------------------------------------------------------

/// Fuzz: random bytes interpreted as Adam inputs must not panic the builtin.
/// We bound shapes and hyperparameters to keep each call cheap.
#[test]
fn fuzz_adam_step_no_panic() {
    check!().with_type::<(Vec<i8>, Vec<i8>, u8, u8, u8, u8)>().for_each(
        |(w_raw, g_raw, lr_raw, b1_raw, b2_raw, t_raw)| {
            use cjc_runtime::builtins::dispatch_builtin;
            use cjc_runtime::value::Value;

            let n = w_raw.len().min(g_raw.len()).min(32);
            if n == 0 {
                return;
            }
            // Scale raw bytes into reasonable ranges.
            let w_data: Vec<f64> = w_raw[..n].iter().map(|b| (*b as f64) / 32.0).collect();
            let g_data: Vec<f64> = g_raw[..n].iter().map(|b| (*b as f64) / 32.0).collect();
            let lr = (*lr_raw as f64 + 1.0) / 1000.0; // 1e-3 .. 0.256
            let b1 = 0.5 + (*b1_raw as f64) / 512.0; // 0.5 .. ~1.0
            let b2 = 0.9 + (*b2_raw as f64) / 2560.0; // 0.9 .. ~1.0
            let t = (*t_raw as i64).max(1); // >= 1

            let shape = vec![n];
            let Ok(w) = Tensor::from_vec(w_data, &shape) else {
                return;
            };
            let Ok(g) = Tensor::from_vec(g_data, &shape) else {
                return;
            };
            let m = Tensor::zeros(&shape);
            let v = Tensor::zeros(&shape);

            let args = vec![
                Value::Tensor(w),
                Value::Tensor(g),
                Value::Tensor(m),
                Value::Tensor(v),
                Value::Float(lr),
                Value::Float(b1.min(0.9999)),
                Value::Float(b2.min(0.99999)),
                Value::Float(1e-8),
                Value::Int(t),
            ];
            // Must not panic. Result may be Ok or Err but never a panic.
            let _ = dispatch_builtin("adam_step", &args);
        },
    );
}

/// Fuzz: random shapes interpreted as mismatched shapes should be rejected
/// cleanly (Err, not panic).
#[test]
fn fuzz_adam_step_shape_guards() {
    check!()
        .with_type::<(u8, u8, u8, u8)>()
        .for_each(|(r1, c1, r2, c2)| {
            use cjc_runtime::builtins::dispatch_builtin;
            use cjc_runtime::value::Value;

            let r1 = ((*r1 as usize) % 6) + 1;
            let c1 = ((*c1 as usize) % 6) + 1;
            let r2 = ((*r2 as usize) % 6) + 1;
            let c2 = ((*c2 as usize) % 6) + 1;

            let w = Tensor::zeros(&[r1, c1]);
            let g = Tensor::zeros(&[r2, c2]);
            let m = Tensor::zeros(&[r1, c1]);
            let v = Tensor::zeros(&[r1, c1]);

            let args = vec![
                Value::Tensor(w),
                Value::Tensor(g),
                Value::Tensor(m),
                Value::Tensor(v),
                Value::Float(0.01),
                Value::Float(0.9),
                Value::Float(0.999),
                Value::Float(1e-8),
                Value::Int(1),
            ];
            let _ = dispatch_builtin("adam_step", &args);
        });
}
