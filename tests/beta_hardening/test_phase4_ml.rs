//! Phase 4 ML/DL expansion tests — LSTM, GRU, multi-head attention, ARIMA.
//!
//! Rust API tests for tensor-heavy operations, plus CJC-level parity tests
//! for builtin dispatch.

use cjc_runtime::Tensor;

// ---------------------------------------------------------------------------
// Helpers for CJC-level parity tests
// ---------------------------------------------------------------------------

/// Run CJC source through eval, return output lines.
fn run_eval(src: &str, seed: u64) -> Vec<String> {
    let (program, _diag) = cjc_parser::parse_source(src);
    let mut interp = cjc_eval::Interpreter::new(seed);
    let _ = interp.exec(&program);
    interp.output
}

/// Run CJC source through MIR-exec, return output lines.
fn run_mir(src: &str, seed: u64) -> Vec<String> {
    let (program, _diag) = cjc_parser::parse_source(src);
    let (_, executor) = cjc_mir_exec::run_program_with_executor(&program, seed)
        .unwrap_or_else(|e| panic!("MIR-exec failed: {e}"));
    executor.output
}

/// Assert parity between eval and MIR-exec outputs.
fn assert_parity(src: &str) {
    let eval_out = run_eval(src, 42);
    let mir_out = run_mir(src, 42);
    assert_eq!(
        eval_out, mir_out,
        "Parity failure:\n  eval: {:?}\n  mir:  {:?}",
        eval_out, mir_out
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 4.1  LSTM cell tests (Rust API)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn lstm_cell_output_shape() {
    let batch = 2;
    let input_size = 3;
    let hidden_size = 4;

    let x = Tensor::zeros(&[batch, input_size]);
    let h_prev = Tensor::zeros(&[batch, hidden_size]);
    let c_prev = Tensor::zeros(&[batch, hidden_size]);
    let w_ih = Tensor::zeros(&[4 * hidden_size, input_size]);
    let w_hh = Tensor::zeros(&[4 * hidden_size, hidden_size]);
    let b_ih = Tensor::zeros(&[4 * hidden_size]);
    let b_hh = Tensor::zeros(&[4 * hidden_size]);

    let (h_new, c_new) =
        cjc_runtime::ml::lstm_cell(&x, &h_prev, &c_prev, &w_ih, &w_hh, &b_ih, &b_hh)
            .expect("lstm_cell failed");

    assert_eq!(h_new.shape(), &[batch, hidden_size]);
    assert_eq!(c_new.shape(), &[batch, hidden_size]);
}

#[test]
fn lstm_cell_determinism() {
    let batch = 2;
    let input_size = 3;
    let hidden_size = 4;
    let mut rng = cjc_repro::Rng::seeded(99);

    let x = Tensor::randn(&[batch, input_size], &mut rng);
    let h_prev = Tensor::randn(&[batch, hidden_size], &mut rng);
    let c_prev = Tensor::randn(&[batch, hidden_size], &mut rng);
    let w_ih = Tensor::randn(&[4 * hidden_size, input_size], &mut rng);
    let w_hh = Tensor::randn(&[4 * hidden_size, hidden_size], &mut rng);
    let b_ih = Tensor::randn(&[4 * hidden_size], &mut rng);
    let b_hh = Tensor::randn(&[4 * hidden_size], &mut rng);

    let (h1, c1) =
        cjc_runtime::ml::lstm_cell(&x, &h_prev, &c_prev, &w_ih, &w_hh, &b_ih, &b_hh).unwrap();
    let (h2, c2) =
        cjc_runtime::ml::lstm_cell(&x, &h_prev, &c_prev, &w_ih, &w_hh, &b_ih, &b_hh).unwrap();

    // Bit-identical
    assert_eq!(h1.to_vec(), h2.to_vec());
    assert_eq!(c1.to_vec(), c2.to_vec());
}

#[test]
fn lstm_cell_zero_weights_produces_known_values() {
    // With zero weights and biases:
    // gates = 0 => sigmoid(0) = 0.5, tanh(0) = 0.0
    // i=0.5, f=0.5, g=0.0, o=0.5
    // c_new = f*c_prev + i*g = 0.5*c_prev + 0.5*0 = 0.5*c_prev
    // h_new = o * tanh(c_new) = 0.5 * tanh(0.5*c_prev)
    let batch = 1;
    let input_size = 2;
    let hidden_size = 2;

    let x = Tensor::ones(&[batch, input_size]);
    let h_prev = Tensor::zeros(&[batch, hidden_size]);
    let c_prev = Tensor::zeros(&[batch, hidden_size]);
    let w_ih = Tensor::zeros(&[4 * hidden_size, input_size]);
    let w_hh = Tensor::zeros(&[4 * hidden_size, hidden_size]);
    let b_ih = Tensor::zeros(&[4 * hidden_size]);
    let b_hh = Tensor::zeros(&[4 * hidden_size]);

    let (h_new, c_new) =
        cjc_runtime::ml::lstm_cell(&x, &h_prev, &c_prev, &w_ih, &w_hh, &b_ih, &b_hh).unwrap();

    // c_prev = 0 => c_new = 0.5*0 + 0.5*0 = 0
    for &v in c_new.to_vec().iter() {
        assert!(v.abs() < 1e-12, "c_new should be 0, got {v}");
    }
    // h_new = 0.5 * tanh(0) = 0
    for &v in h_new.to_vec().iter() {
        assert!(v.abs() < 1e-12, "h_new should be 0, got {v}");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 4.1  GRU cell tests (Rust API)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn gru_cell_output_shape() {
    let batch = 2;
    let input_size = 3;
    let hidden_size = 4;

    let x = Tensor::zeros(&[batch, input_size]);
    let h_prev = Tensor::zeros(&[batch, hidden_size]);
    let w_ih = Tensor::zeros(&[3 * hidden_size, input_size]);
    let w_hh = Tensor::zeros(&[3 * hidden_size, hidden_size]);
    let b_ih = Tensor::zeros(&[3 * hidden_size]);
    let b_hh = Tensor::zeros(&[3 * hidden_size]);

    let h_new = cjc_runtime::ml::gru_cell(&x, &h_prev, &w_ih, &w_hh, &b_ih, &b_hh)
        .expect("gru_cell failed");

    assert_eq!(h_new.shape(), &[batch, hidden_size]);
}

#[test]
fn gru_cell_determinism() {
    let batch = 2;
    let input_size = 3;
    let hidden_size = 4;
    let mut rng = cjc_repro::Rng::seeded(77);

    let x = Tensor::randn(&[batch, input_size], &mut rng);
    let h_prev = Tensor::randn(&[batch, hidden_size], &mut rng);
    let w_ih = Tensor::randn(&[3 * hidden_size, input_size], &mut rng);
    let w_hh = Tensor::randn(&[3 * hidden_size, hidden_size], &mut rng);
    let b_ih = Tensor::randn(&[3 * hidden_size], &mut rng);
    let b_hh = Tensor::randn(&[3 * hidden_size], &mut rng);

    let h1 = cjc_runtime::ml::gru_cell(&x, &h_prev, &w_ih, &w_hh, &b_ih, &b_hh).unwrap();
    let h2 = cjc_runtime::ml::gru_cell(&x, &h_prev, &w_ih, &w_hh, &b_ih, &b_hh).unwrap();

    assert_eq!(h1.to_vec(), h2.to_vec());
}

#[test]
fn gru_cell_zero_weights_produces_known_values() {
    // With zero weights and biases:
    // r = sigmoid(0) = 0.5, z = sigmoid(0) = 0.5, n = tanh(0) = 0
    // h_new = (1 - 0.5) * 0 + 0.5 * h_prev = 0.5 * h_prev
    let batch = 1;
    let input_size = 2;
    let hidden_size = 2;

    let x = Tensor::ones(&[batch, input_size]);
    let h_prev = Tensor::ones(&[batch, hidden_size]);
    let w_ih = Tensor::zeros(&[3 * hidden_size, input_size]);
    let w_hh = Tensor::zeros(&[3 * hidden_size, hidden_size]);
    let b_ih = Tensor::zeros(&[3 * hidden_size]);
    let b_hh = Tensor::zeros(&[3 * hidden_size]);

    let h_new = cjc_runtime::ml::gru_cell(&x, &h_prev, &w_ih, &w_hh, &b_ih, &b_hh).unwrap();

    for &v in h_new.to_vec().iter() {
        assert!(
            (v - 0.5).abs() < 1e-12,
            "h_new should be 0.5 * h_prev = 0.5, got {v}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 4.2  Multi-Head Attention tests (Rust API)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn mha_output_shape() {
    let batch = 2;
    let seq = 4;
    let model_dim = 8;
    let num_heads = 2;

    let q = Tensor::zeros(&[batch, seq, model_dim]);
    let k = Tensor::zeros(&[batch, seq, model_dim]);
    let v = Tensor::zeros(&[batch, seq, model_dim]);
    let w_q = Tensor::zeros(&[model_dim, model_dim]);
    let w_k = Tensor::zeros(&[model_dim, model_dim]);
    let w_v = Tensor::zeros(&[model_dim, model_dim]);
    let w_o = Tensor::zeros(&[model_dim, model_dim]);
    let b_q = Tensor::zeros(&[model_dim]);
    let b_k = Tensor::zeros(&[model_dim]);
    let b_v = Tensor::zeros(&[model_dim]);
    let b_o = Tensor::zeros(&[model_dim]);

    let out = cjc_runtime::ml::multi_head_attention(
        &q, &k, &v, &w_q, &w_k, &w_v, &w_o, &b_q, &b_k, &b_v, &b_o, num_heads,
    )
    .expect("multi_head_attention failed");

    assert_eq!(out.shape(), &[batch, seq, model_dim]);
}

#[test]
fn mha_determinism() {
    let batch = 1;
    let seq = 3;
    let model_dim = 4;
    let num_heads = 2;
    let mut rng = cjc_repro::Rng::seeded(55);

    let q = Tensor::randn(&[batch, seq, model_dim], &mut rng);
    let k = Tensor::randn(&[batch, seq, model_dim], &mut rng);
    let v = Tensor::randn(&[batch, seq, model_dim], &mut rng);
    let w_q = Tensor::randn(&[model_dim, model_dim], &mut rng);
    let w_k = Tensor::randn(&[model_dim, model_dim], &mut rng);
    let w_v = Tensor::randn(&[model_dim, model_dim], &mut rng);
    let w_o = Tensor::randn(&[model_dim, model_dim], &mut rng);
    let b_q = Tensor::randn(&[model_dim], &mut rng);
    let b_k = Tensor::randn(&[model_dim], &mut rng);
    let b_v = Tensor::randn(&[model_dim], &mut rng);
    let b_o = Tensor::randn(&[model_dim], &mut rng);

    let out1 = cjc_runtime::ml::multi_head_attention(
        &q, &k, &v, &w_q, &w_k, &w_v, &w_o, &b_q, &b_k, &b_v, &b_o, num_heads,
    )
    .unwrap();
    let out2 = cjc_runtime::ml::multi_head_attention(
        &q, &k, &v, &w_q, &w_k, &w_v, &w_o, &b_q, &b_k, &b_v, &b_o, num_heads,
    )
    .unwrap();

    assert_eq!(out1.to_vec(), out2.to_vec());
}

// ═══════════════════════════════════════════════════════════════════════════
// 4.3  ARIMA tests (Rust API)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn arima_diff_d1() {
    let data = vec![1.0, 3.0, 6.0, 10.0];
    let result = cjc_runtime::timeseries::arima_diff(&data, 1);
    assert_eq!(result, vec![2.0, 3.0, 4.0]);
}

#[test]
fn arima_diff_d2() {
    let data = vec![1.0, 3.0, 6.0, 10.0];
    // d=1: [2, 3, 4]
    // d=2: [1, 1]
    let result = cjc_runtime::timeseries::arima_diff(&data, 2);
    assert_eq!(result, vec![1.0, 1.0]);
}

#[test]
fn arima_diff_d0_identity() {
    let data = vec![1.0, 2.0, 3.0];
    let result = cjc_runtime::timeseries::arima_diff(&data, 0);
    assert_eq!(result, data);
}

#[test]
fn ar_fit_recovers_coefficient() {
    // Generate AR(1) data: x[t] = 0.7 * x[t-1] + noise
    let mut rng = cjc_repro::Rng::seeded(42);
    let n = 2000;
    let phi_true = 0.7;
    let mut data = vec![0.0; n];
    for t in 1..n {
        data[t] = phi_true * data[t - 1] + rng.next_normal_f64() * 0.1;
    }

    let coeffs = cjc_runtime::timeseries::ar_fit(&data, 1).expect("ar_fit failed");
    assert_eq!(coeffs.len(), 1);
    assert!(
        (coeffs[0] - phi_true).abs() < 0.05,
        "AR(1) coefficient should be ~{phi_true}, got {}",
        coeffs[0]
    );
}

#[test]
fn ar_forecast_produces_correct_count() {
    let coeffs = vec![0.5];
    let history = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let forecast = cjc_runtime::timeseries::ar_forecast(&coeffs, &history, 3).unwrap();
    assert_eq!(forecast.len(), 3);

    // First prediction: 0.5 * 5.0 = 2.5
    assert!((forecast[0] - 2.5).abs() < 1e-12);
    // Second: 0.5 * 2.5 = 1.25
    assert!((forecast[1] - 1.25).abs() < 1e-12);
}

#[test]
fn ar_fit_determinism() {
    let mut rng = cjc_repro::Rng::seeded(42);
    let n = 500;
    let mut data = vec![0.0; n];
    for t in 1..n {
        data[t] = 0.6 * data[t - 1] + rng.next_normal_f64() * 0.1;
    }

    let c1 = cjc_runtime::timeseries::ar_fit(&data, 2).unwrap();
    let c2 = cjc_runtime::timeseries::ar_fit(&data, 2).unwrap();
    for (a, b) in c1.iter().zip(c2.iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CJC-level parity tests (eval == MIR-exec)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn parity_arima_diff() {
    let src = r#"
let data: Any = [1.0, 3.0, 6.0, 10.0];
let result: Any = arima_diff(data, 1);
print(array_len(result));
"#;
    assert_parity(src);
    let out = run_eval(src, 42);
    assert!(!out.is_empty());
    // Should be 3 elements after d=1 differencing
    assert_eq!(out[0], "3");
}

#[test]
fn parity_ar_fit() {
    // Build a simple dataset deterministically and fit AR(1)
    let src = r#"
let data: Any = [0.0, 0.01, 0.015, 0.0175, 0.01875, 0.019375, 0.0196875, 0.01984375, 0.019921875, 0.0199609375, 0.01998046875, 0.009990234375, 0.014995117188, 0.017497558594, 0.018748779297, 0.019374389648, 0.019687194824, 0.019843597412, 0.019921798706, 0.019960899353];
let coeffs: Any = ar_fit(data, 1);
print(array_len(coeffs));
"#;
    assert_parity(src);
}

#[test]
fn parity_ar_forecast() {
    let src = r#"
let coeffs: Any = [0.5];
let history: Any = [1.0, 2.0, 3.0, 4.0, 5.0];
let fc: Any = ar_forecast(coeffs, history, 2);
print(array_len(fc));
"#;
    assert_parity(src);
}
