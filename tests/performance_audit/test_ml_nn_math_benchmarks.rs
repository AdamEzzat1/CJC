//! Performance Audit: ML, Neural Network, Math, and Pipeline Benchmarks
//!
//! Run with: cargo test --release --test test_performance_audit -- --nocapture

use cjc_runtime::Tensor;
use cjc_repro::Rng;
use std::time::Instant;

// ============================================================================
// Helper: create deterministic random data
// ============================================================================

fn rand_vec(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = Rng::seeded(seed);
    (0..n).map(|_| rng.next_f64()).collect()
}

fn rand_normal_vec(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = Rng::seeded(seed);
    (0..n).map(|_| rng.next_normal_f64()).collect()
}

fn rand_tensor(shape: &[usize], seed: u64) -> Tensor {
    let mut rng = Rng::seeded(seed);
    Tensor::randn(shape, &mut rng)
}

// ============================================================================
// AREA 1: ML Training Performance
// ============================================================================

#[test]
fn bench_logistic_regression_varying_sizes() {
    for &n in &[100, 1000, 5000] {
        let p = 5;
        let x = rand_vec(n * p, 42);
        let mut rng = Rng::seeded(99);
        let y: Vec<f64> = (0..n).map(|_| if rng.next_f64() > 0.5 { 1.0 } else { 0.0 }).collect();

        let start = Instant::now();
        let result = cjc_runtime::hypothesis::logistic_regression(&x, &y, n, p);
        let elapsed = start.elapsed();
        eprintln!("logistic_regression n={} p={}: {:?} (ok={})", n, p, elapsed, result.is_ok());

        // Determinism check
        let result2 = cjc_runtime::hypothesis::logistic_regression(&x, &y, n, p);
        if let (Ok(r1), Ok(r2)) = (&result, &result2) {
            assert_eq!(r1.coefficients, r2.coefficients, "logistic_regression not deterministic at n={}", n);
        }
    }
}

#[test]
fn bench_mse_loss_varying_sizes() {
    for &n in &[1_000, 10_000, 100_000, 1_000_000] {
        let pred = rand_vec(n, 42);
        let target = rand_vec(n, 99);

        let start = Instant::now();
        let result = cjc_runtime::ml::mse_loss(&pred, &target).unwrap();
        let elapsed = start.elapsed();
        eprintln!("mse_loss n={}: {:?} (result={:.6})", n, elapsed, result);

        // Determinism
        let result2 = cjc_runtime::ml::mse_loss(&pred, &target).unwrap();
        assert_eq!(result.to_bits(), result2.to_bits(), "mse_loss not bit-identical at n={}", n);
    }
}

#[test]
fn bench_cross_entropy_loss_varying_sizes() {
    for &n in &[100, 1_000, 10_000] {
        // Create softmax-like predictions (positive, sum to ~1)
        let raw = rand_vec(n, 42);
        let sum: f64 = raw.iter().sum();
        let pred: Vec<f64> = raw.iter().map(|&x| (x / sum).max(1e-12)).collect();
        let target = rand_vec(n, 99);

        let start = Instant::now();
        let result = cjc_runtime::ml::cross_entropy_loss(&pred, &target).unwrap();
        let elapsed = start.elapsed();
        eprintln!("cross_entropy_loss n={}: {:?} (result={:.6})", n, elapsed, result);

        let result2 = cjc_runtime::ml::cross_entropy_loss(&pred, &target).unwrap();
        assert_eq!(result.to_bits(), result2.to_bits(), "cross_entropy_loss not bit-identical");
    }
}

#[test]
fn bench_sgd_step_100_iterations() {
    let n_params = 1000;
    let mut params = rand_vec(n_params, 42);
    let grads = rand_vec(n_params, 99);
    let mut state = cjc_runtime::ml::SgdState::new(n_params, 0.01, 0.9);

    let start = Instant::now();
    for _ in 0..100 {
        cjc_runtime::ml::sgd_step(&mut params, &grads, &mut state);
    }
    let elapsed = start.elapsed();
    eprintln!("sgd_step 100 iterations (n_params={}): {:?}", n_params, elapsed);

    // Determinism: re-run from scratch
    let mut params2 = rand_vec(n_params, 42);
    let mut state2 = cjc_runtime::ml::SgdState::new(n_params, 0.01, 0.9);
    for _ in 0..100 {
        cjc_runtime::ml::sgd_step(&mut params2, &grads, &mut state2);
    }
    assert_eq!(params, params2, "SGD not deterministic");
}

#[test]
fn bench_adam_step_100_iterations() {
    let n_params = 1000;
    let mut params = rand_vec(n_params, 42);
    let grads = rand_vec(n_params, 99);
    let mut state = cjc_runtime::ml::AdamState::new(n_params, 0.001);

    let start = Instant::now();
    for _ in 0..100 {
        cjc_runtime::ml::adam_step(&mut params, &grads, &mut state);
    }
    let elapsed = start.elapsed();
    eprintln!("adam_step 100 iterations (n_params={}): {:?}", n_params, elapsed);

    // Determinism
    let mut params2 = rand_vec(n_params, 42);
    let mut state2 = cjc_runtime::ml::AdamState::new(n_params, 0.001);
    for _ in 0..100 {
        cjc_runtime::ml::adam_step(&mut params2, &grads, &mut state2);
    }
    assert_eq!(params, params2, "Adam not deterministic");
}

// ============================================================================
// AREA 1b: Autodiff Throughput
// ============================================================================

#[test]
fn bench_forward_mode_ad() {
    use cjc_ad::Dual;

    // Evaluate a polynomial f(x) = x^10 + x^9 + ... + x + 1 with forward-mode AD
    for &n_terms in &[10, 100, 1000] {
        let x = Dual::variable(1.5);

        let start = Instant::now();
        let mut result = Dual::constant(0.0);
        for _trial in 0..100 {
            let mut acc = Dual::constant(0.0);
            let mut power = Dual::constant(1.0);
            for _ in 0..n_terms {
                acc = acc + power.clone();
                power = power * x.clone();
            }
            result = acc;
        }
        let elapsed = start.elapsed();
        eprintln!("forward_ad {} terms x100: {:?} (value={:.4}, deriv={:.4})",
            n_terms, elapsed, result.value, result.deriv);

        // Determinism
        let mut result2 = Dual::constant(0.0);
        let mut acc = Dual::constant(0.0);
        let mut power = Dual::constant(1.0);
        let x2 = Dual::variable(1.5);
        for _ in 0..n_terms {
            acc = acc + power.clone();
            power = power * x2.clone();
        }
        result2 = acc;
        assert_eq!(result.value.to_bits(), result2.value.to_bits(), "Forward AD not deterministic");
    }
}

#[test]
fn bench_reverse_mode_ad() {
    use cjc_ad::GradGraph;

    for &n_vars in &[10, 50, 100] {
        let mut graph = GradGraph::new();

        // Create n_vars parameters
        let mut param_ids = Vec::new();
        let mut rng = Rng::seeded(42);
        for _ in 0..n_vars {
            let t = Tensor::from_vec(vec![rng.next_normal_f64()], &[1]).unwrap();
            param_ids.push(graph.parameter(t));
        }

        // Build expression: sum of all params squared (sum(xi^2))
        let start = Instant::now();
        let mut sum_id = graph.mul(param_ids[0], param_ids[0]);
        for i in 1..n_vars {
            let sq = graph.mul(param_ids[i], param_ids[i]);
            sum_id = graph.add(sum_id, sq);
        }
        let loss = graph.sum(sum_id);

        // Backward
        graph.backward(loss);
        let elapsed = start.elapsed();

        eprintln!("reverse_ad n_vars={}: {:?}", n_vars, elapsed);

        // Determinism: rebuild and compare
        let mut graph2 = GradGraph::new();
        let mut param_ids2 = Vec::new();
        let mut rng2 = Rng::seeded(42);
        for _ in 0..n_vars {
            let t = Tensor::from_vec(vec![rng2.next_normal_f64()], &[1]).unwrap();
            param_ids2.push(graph2.parameter(t));
        }
        let mut sum_id2 = graph2.mul(param_ids2[0], param_ids2[0]);
        for i in 1..n_vars {
            let sq = graph2.mul(param_ids2[i], param_ids2[i]);
            sum_id2 = graph2.add(sum_id2, sq);
        }
        let loss2 = graph2.sum(sum_id2);
        graph2.backward(loss2);

        let grad1 = graph.grad(param_ids[0]).unwrap().to_vec();
        let grad2 = graph2.grad(param_ids2[0]).unwrap().to_vec();
        assert_eq!(grad1, grad2, "Reverse AD not deterministic");
    }
}

// ============================================================================
// AREA 2: Neural Network Layer Performance
// ============================================================================

#[test]
fn bench_matmul_as_dense_forward() {
    // dense_forward is essentially matmul + bias via Tensor::linear
    // Benchmark matmul + add which is the core of dense layers
    for &n in &[64, 256, 512] {
        let x = rand_tensor(&[1, n], 42);
        let w = rand_tensor(&[n, n], 99);
        let b = rand_tensor(&[n], 77);

        // Warm up
        let _ = x.linear(&w, &b);

        let start = Instant::now();
        let iters = if n <= 256 { 100 } else { 10 };
        let mut result = None;
        for _ in 0..iters {
            result = Some(x.linear(&w, &b).unwrap());
        }
        let elapsed = start.elapsed();
        let per_iter = elapsed / iters as u32;
        eprintln!("dense_forward (linear) {}->{}:  {:?}/iter ({} iters, total {:?})",
            n, n, per_iter, iters, elapsed);

        // Determinism
        let r1 = x.linear(&w, &b).unwrap();
        let r2 = x.linear(&w, &b).unwrap();
        assert_eq!(r1.to_vec(), r2.to_vec(), "dense_forward not deterministic at n={}", n);
    }
}

#[test]
fn bench_lstm_cell() {
    for &hidden_size in &[64, 256] {
        let batch = 1;
        let input_size = hidden_size;

        let x = rand_tensor(&[batch, input_size], 42);
        let h_prev = rand_tensor(&[batch, hidden_size], 43);
        let c_prev = rand_tensor(&[batch, hidden_size], 44);
        let w_ih = rand_tensor(&[4 * hidden_size, input_size], 45);
        let w_hh = rand_tensor(&[4 * hidden_size, hidden_size], 46);
        let b_ih = rand_tensor(&[4 * hidden_size], 47);
        let b_hh = rand_tensor(&[4 * hidden_size], 48);

        // Warm up
        let _ = cjc_runtime::ml::lstm_cell(&x, &h_prev, &c_prev, &w_ih, &w_hh, &b_ih, &b_hh);

        let iters = if hidden_size <= 64 { 100 } else { 10 };
        let start = Instant::now();
        for _ in 0..iters {
            let _ = cjc_runtime::ml::lstm_cell(&x, &h_prev, &c_prev, &w_ih, &w_hh, &b_ih, &b_hh).unwrap();
        }
        let elapsed = start.elapsed();
        let per_iter = elapsed / iters as u32;
        eprintln!("lstm_cell hidden={}:  {:?}/iter ({} iters, total {:?})",
            hidden_size, per_iter, iters, elapsed);

        // Determinism
        let (h1, c1) = cjc_runtime::ml::lstm_cell(&x, &h_prev, &c_prev, &w_ih, &w_hh, &b_ih, &b_hh).unwrap();
        let (h2, c2) = cjc_runtime::ml::lstm_cell(&x, &h_prev, &c_prev, &w_ih, &w_hh, &b_ih, &b_hh).unwrap();
        assert_eq!(h1.to_vec(), h2.to_vec(), "LSTM h not deterministic at hidden={}", hidden_size);
        assert_eq!(c1.to_vec(), c2.to_vec(), "LSTM c not deterministic at hidden={}", hidden_size);
    }
}

#[test]
fn bench_gru_cell() {
    for &hidden_size in &[64, 256] {
        let batch = 1;
        let input_size = hidden_size;

        let x = rand_tensor(&[batch, input_size], 42);
        let h_prev = rand_tensor(&[batch, hidden_size], 43);
        let w_ih = rand_tensor(&[3 * hidden_size, input_size], 45);
        let w_hh = rand_tensor(&[3 * hidden_size, hidden_size], 46);
        let b_ih = rand_tensor(&[3 * hidden_size], 47);
        let b_hh = rand_tensor(&[3 * hidden_size], 48);

        let iters = if hidden_size <= 64 { 100 } else { 10 };
        let start = Instant::now();
        for _ in 0..iters {
            let _ = cjc_runtime::ml::gru_cell(&x, &h_prev, &w_ih, &w_hh, &b_ih, &b_hh).unwrap();
        }
        let elapsed = start.elapsed();
        let per_iter = elapsed / iters as u32;
        eprintln!("gru_cell hidden={}:  {:?}/iter ({} iters, total {:?})",
            hidden_size, per_iter, iters, elapsed);

        // Determinism
        let r1 = cjc_runtime::ml::gru_cell(&x, &h_prev, &w_ih, &w_hh, &b_ih, &b_hh).unwrap();
        let r2 = cjc_runtime::ml::gru_cell(&x, &h_prev, &w_ih, &w_hh, &b_ih, &b_hh).unwrap();
        assert_eq!(r1.to_vec(), r2.to_vec(), "GRU not deterministic at hidden={}", hidden_size);
    }
}

#[test]
fn bench_multi_head_attention() {
    for &(seq_len, model_dim, num_heads) in &[(32, 64, 4), (64, 128, 8)] {
        let batch = 1;
        let q = rand_tensor(&[batch, seq_len, model_dim], 42);
        let k = rand_tensor(&[batch, seq_len, model_dim], 43);
        let v = rand_tensor(&[batch, seq_len, model_dim], 44);
        let w_q = rand_tensor(&[model_dim, model_dim], 45);
        let w_k = rand_tensor(&[model_dim, model_dim], 46);
        let w_v = rand_tensor(&[model_dim, model_dim], 47);
        let w_o = rand_tensor(&[model_dim, model_dim], 48);
        let b_q = rand_tensor(&[model_dim], 49);
        let b_k = rand_tensor(&[model_dim], 50);
        let b_v = rand_tensor(&[model_dim], 51);
        let b_o = rand_tensor(&[model_dim], 52);

        let iters = 5;
        let start = Instant::now();
        for _ in 0..iters {
            let _ = cjc_runtime::ml::multi_head_attention(
                &q, &k, &v, &w_q, &w_k, &w_v, &w_o, &b_q, &b_k, &b_v, &b_o, num_heads
            ).unwrap();
        }
        let elapsed = start.elapsed();
        let per_iter = elapsed / iters as u32;
        eprintln!("multi_head_attention seq={} dim={} heads={}:  {:?}/iter ({} iters)",
            seq_len, model_dim, num_heads, per_iter, iters);

        // Determinism
        let r1 = cjc_runtime::ml::multi_head_attention(
            &q, &k, &v, &w_q, &w_k, &w_v, &w_o, &b_q, &b_k, &b_v, &b_o, num_heads
        ).unwrap();
        let r2 = cjc_runtime::ml::multi_head_attention(
            &q, &k, &v, &w_q, &w_k, &w_v, &w_o, &b_q, &b_k, &b_v, &b_o, num_heads
        ).unwrap();
        assert_eq!(r1.to_vec(), r2.to_vec(), "MHA not deterministic");
    }
}

#[test]
fn bench_softmax() {
    for &n in &[100, 1_000, 10_000] {
        let t = rand_tensor(&[1, n], 42);

        let iters = 100;
        let start = Instant::now();
        for _ in 0..iters {
            let _ = t.softmax().unwrap();
        }
        let elapsed = start.elapsed();
        let per_iter = elapsed / iters as u32;
        eprintln!("softmax n={}: {:?}/iter ({} iters)", n, per_iter, iters);

        let r1 = t.softmax().unwrap();
        let r2 = t.softmax().unwrap();
        assert_eq!(r1.to_vec(), r2.to_vec(), "softmax not deterministic at n={}", n);
    }
}

#[test]
fn bench_activations() {
    for &n in &[10_000, 100_000] {
        let t = rand_tensor(&[n], 42);

        // ReLU
        let start = Instant::now();
        for _ in 0..100 {
            let _ = t.relu();
        }
        let elapsed = start.elapsed();
        eprintln!("relu n={}: {:?}/iter (100 iters)", n, elapsed / 100);

        // Sigmoid
        let start = Instant::now();
        for _ in 0..100 {
            let _ = t.sigmoid();
        }
        let elapsed = start.elapsed();
        eprintln!("sigmoid n={}: {:?}/iter (100 iters)", n, elapsed / 100);

        // Tanh
        let start = Instant::now();
        for _ in 0..100 {
            let _ = t.tanh_activation();
        }
        let elapsed = start.elapsed();
        eprintln!("tanh n={}: {:?}/iter (100 iters)", n, elapsed / 100);

        // Determinism
        let r1 = t.relu();
        let r2 = t.relu();
        assert_eq!(r1.to_vec(), r2.to_vec(), "relu not deterministic");
        let r1 = t.sigmoid();
        let r2 = t.sigmoid();
        assert_eq!(r1.to_vec(), r2.to_vec(), "sigmoid not deterministic");
    }
}

// ============================================================================
// AREA 3: Math/Numerical Functions
// ============================================================================

#[test]
fn bench_matmul_varying_sizes() {
    for &n in &[64, 128, 256, 512] {
        let a = rand_tensor(&[n, n], 42);
        let b = rand_tensor(&[n, n], 99);

        // Warm up
        let _ = a.matmul(&b);

        let iters = if n <= 256 { 20 } else { 3 };
        let start = Instant::now();
        for _ in 0..iters {
            let _ = a.matmul(&b).unwrap();
        }
        let elapsed = start.elapsed();
        let per_iter = elapsed / iters as u32;

        // Calculate GFLOPS: matmul is 2*n^3 FLOPs
        let flops = 2.0 * (n as f64).powi(3);
        let gflops = flops / per_iter.as_secs_f64() / 1e9;
        eprintln!("matmul {}x{}: {:?}/iter  ({:.2} GFLOPS)", n, n, per_iter, gflops);

        // Determinism
        let r1 = a.matmul(&b).unwrap();
        let r2 = a.matmul(&b).unwrap();
        assert_eq!(r1.to_vec(), r2.to_vec(), "matmul not deterministic at n={}", n);
    }
}

#[test]
fn bench_matmul_1024() {
    let n = 1024;
    let a = rand_tensor(&[n, n], 42);
    let b = rand_tensor(&[n, n], 99);

    let start = Instant::now();
    let r1 = a.matmul(&b).unwrap();
    let elapsed = start.elapsed();

    let flops = 2.0 * (n as f64).powi(3);
    let gflops = flops / elapsed.as_secs_f64() / 1e9;
    eprintln!("matmul {}x{}: {:?}  ({:.2} GFLOPS)", n, n, elapsed, gflops);

    // Determinism
    let r2 = a.matmul(&b).unwrap();
    assert_eq!(r1.to_vec(), r2.to_vec(), "matmul 1024 not deterministic");
}

#[test]
fn bench_solve_linear_system() {
    for &n in &[64, 256] {
        // Create a diagonally dominant matrix for numerical stability
        let mut data = rand_vec(n * n, 42);
        for i in 0..n {
            data[i * n + i] += n as f64 * 2.0; // make diagonally dominant
        }
        let a = Tensor::from_vec(data, &[n, n]).unwrap();
        let b = rand_tensor(&[n], 99);

        let iters = if n <= 64 { 50 } else { 5 };
        let start = Instant::now();
        for _ in 0..iters {
            let _ = a.solve(&b).unwrap();
        }
        let elapsed = start.elapsed();
        let per_iter = elapsed / iters as u32;
        eprintln!("solve {}x{}: {:?}/iter", n, n, per_iter);

        // Determinism
        let r1 = a.solve(&b).unwrap();
        let r2 = a.solve(&b).unwrap();
        assert_eq!(r1.to_vec(), r2.to_vec(), "solve not deterministic at n={}", n);
    }
}

#[test]
fn bench_svd() {
    for &n in &[32, 64] {
        let a = rand_tensor(&[n, n], 42);

        let iters = if n <= 32 { 10 } else { 3 };
        let start = Instant::now();
        for _ in 0..iters {
            let _ = a.svd().unwrap();
        }
        let elapsed = start.elapsed();
        let per_iter = elapsed / iters as u32;
        eprintln!("svd {}x{}: {:?}/iter", n, n, per_iter);

        // Determinism
        let (u1, s1, vt1) = a.svd().unwrap();
        let (u2, s2, vt2) = a.svd().unwrap();
        assert_eq!(s1, s2, "SVD singular values not deterministic at n={}", n);
        assert_eq!(u1.to_vec(), u2.to_vec(), "SVD U not deterministic at n={}", n);
        assert_eq!(vt1.to_vec(), vt2.to_vec(), "SVD Vt not deterministic at n={}", n);
    }
}

#[test]
fn bench_eigenvalue() {
    for &n in &[32, 64] {
        // Create symmetric matrix (needed for eigh)
        let raw = rand_vec(n * n, 42);
        let mut sym = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                sym[i * n + j] = (raw[i * n + j] + raw[j * n + i]) / 2.0;
            }
        }
        let a = Tensor::from_vec(sym, &[n, n]).unwrap();

        let iters = if n <= 32 { 10 } else { 3 };
        let start = Instant::now();
        for _ in 0..iters {
            let _ = a.eigh().unwrap();
        }
        let elapsed = start.elapsed();
        let per_iter = elapsed / iters as u32;
        eprintln!("eigh {}x{}: {:?}/iter", n, n, per_iter);

        // Determinism
        let (ev1, _) = a.eigh().unwrap();
        let (ev2, _) = a.eigh().unwrap();
        assert_eq!(ev1, ev2, "eigh not deterministic at n={}", n);
    }
}

#[test]
fn bench_trapezoid_integration() {
    for &n in &[1_000, 10_000, 100_000] {
        let xs: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| x * x).collect(); // integrate x^2 from 0 to ~1

        let iters = 100;
        let start = Instant::now();
        for _ in 0..iters {
            let _ = cjc_runtime::integrate::trapezoid(&xs, &ys).unwrap();
        }
        let elapsed = start.elapsed();
        let per_iter = elapsed / iters as u32;
        eprintln!("trapezoid n={}: {:?}/iter", n, per_iter);

        let r1 = cjc_runtime::integrate::trapezoid(&xs, &ys).unwrap();
        let r2 = cjc_runtime::integrate::trapezoid(&xs, &ys).unwrap();
        assert_eq!(r1.to_bits(), r2.to_bits(), "trapezoid not bit-identical at n={}", n);
    }
}

#[test]
fn bench_simpson_integration() {
    for &n in &[1_001, 10_001, 100_001] { // odd number for Simpson's rule (even intervals)
        let xs: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| x * x).collect();

        let iters = 100;
        let start = Instant::now();
        for _ in 0..iters {
            let _ = cjc_runtime::integrate::simpson(&xs, &ys).unwrap();
        }
        let elapsed = start.elapsed();
        let per_iter = elapsed / iters as u32;
        eprintln!("simpson n={}: {:?}/iter", n, per_iter);

        let r1 = cjc_runtime::integrate::simpson(&xs, &ys).unwrap();
        let r2 = cjc_runtime::integrate::simpson(&xs, &ys).unwrap();
        assert_eq!(r1.to_bits(), r2.to_bits(), "simpson not bit-identical at n={}", n);
    }
}

#[test]
fn bench_diff_central() {
    for &n in &[1_000, 10_000, 100_000] {
        let xs: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| (x * std::f64::consts::PI).sin()).collect();

        let iters = 100;
        let start = Instant::now();
        for _ in 0..iters {
            let _ = cjc_runtime::differentiate::diff_central(&xs, &ys).unwrap();
        }
        let elapsed = start.elapsed();
        let per_iter = elapsed / iters as u32;
        eprintln!("diff_central n={}: {:?}/iter", n, per_iter);

        let r1 = cjc_runtime::differentiate::diff_central(&xs, &ys).unwrap();
        let r2 = cjc_runtime::differentiate::diff_central(&xs, &ys).unwrap();
        assert_eq!(r1, r2, "diff_central not deterministic at n={}", n);
    }
}

#[test]
fn bench_gradient_1d() {
    for &n in &[1_000, 10_000, 100_000] {
        let dx = 1.0 / n as f64;
        let ys: Vec<f64> = (0..n).map(|i| (i as f64 * dx * std::f64::consts::PI).sin()).collect();

        let iters = 100;
        let start = Instant::now();
        for _ in 0..iters {
            let _ = cjc_runtime::differentiate::gradient_1d(&ys, dx);
        }
        let elapsed = start.elapsed();
        let per_iter = elapsed / iters as u32;
        eprintln!("gradient_1d n={}: {:?}/iter", n, per_iter);

        let r1 = cjc_runtime::differentiate::gradient_1d(&ys, dx);
        let r2 = cjc_runtime::differentiate::gradient_1d(&ys, dx);
        assert_eq!(r1, r2, "gradient_1d not deterministic at n={}", n);
    }
}

#[test]
fn bench_ar_fit() {
    for &(n, p) in &[(1_000, 5), (10_000, 5), (1_000, 20)] {
        let data = rand_normal_vec(n, 42);

        let iters = if n >= 10_000 { 10 } else { 50 };
        let start = Instant::now();
        for _ in 0..iters {
            let _ = cjc_runtime::timeseries::ar_fit(&data, p).unwrap();
        }
        let elapsed = start.elapsed();
        let per_iter = elapsed / iters as u32;
        eprintln!("ar_fit n={} p={}: {:?}/iter", n, p, per_iter);

        let r1 = cjc_runtime::timeseries::ar_fit(&data, p).unwrap();
        let r2 = cjc_runtime::timeseries::ar_fit(&data, p).unwrap();
        assert_eq!(r1, r2, "ar_fit not deterministic at n={} p={}", n, p);
    }
}

#[test]
fn bench_kahan_vs_naive_summation() {
    let n = 1_000_000;
    let data = rand_vec(n, 42);

    // Kahan summation
    let start = Instant::now();
    let kahan_result = cjc_repro::kahan_sum_f64(&data);
    let kahan_elapsed = start.elapsed();

    // Naive summation
    let start = Instant::now();
    let naive_result: f64 = data.iter().sum();
    let naive_elapsed = start.elapsed();

    // BinnedAccumulator
    let start = Instant::now();
    let mut acc = cjc_runtime::accumulator::BinnedAccumulatorF64::new();
    acc.add_slice(&data);
    let binned_result = acc.finalize();
    let binned_elapsed = start.elapsed();

    let kahan_naive_diff = (kahan_result - naive_result).abs();
    let binned_naive_diff = (binned_result - naive_result).abs();

    eprintln!("=== Summation of {} random floats ===", n);
    eprintln!("  naive:  {:?}  result={:.15}", naive_elapsed, naive_result);
    eprintln!("  kahan:  {:?}  result={:.15}  diff_from_naive={:.2e}", kahan_elapsed, kahan_result, kahan_naive_diff);
    eprintln!("  binned: {:?}  result={:.15}  diff_from_naive={:.2e}", binned_elapsed, binned_result, binned_naive_diff);

    // Determinism
    let kahan2 = cjc_repro::kahan_sum_f64(&data);
    assert_eq!(kahan_result.to_bits(), kahan2.to_bits(), "Kahan not bit-identical");

    let mut acc2 = cjc_runtime::accumulator::BinnedAccumulatorF64::new();
    acc2.add_slice(&data);
    let binned2 = acc2.finalize();
    assert_eq!(binned_result.to_bits(), binned2.to_bits(), "Binned not bit-identical");
}

// ============================================================================
// AREA 4: TidyView -> ML Pipeline
// ============================================================================

#[test]
fn bench_dataframe_to_ml_pipeline() {
    use cjc_data::{DataFrame, Column};

    let n = 100_000;
    let mut rng = Rng::seeded(42);

    // Step 1: Create DataFrame with features
    let start_create = Instant::now();
    let feature1: Vec<f64> = (0..n).map(|_| rng.next_normal_f64()).collect();
    let feature2: Vec<f64> = (0..n).map(|_| rng.next_normal_f64()).collect();
    let feature3: Vec<f64> = (0..n).map(|_| rng.next_normal_f64()).collect();
    let label: Vec<f64> = feature1.iter().zip(feature2.iter())
        .map(|(&f1, &f2)| if f1 + 0.5 * f2 > 0.0 { 1.0 } else { 0.0 })
        .collect();

    let df = DataFrame::from_columns(vec![
        ("feature1".to_string(), Column::Float(feature1)),
        ("feature2".to_string(), Column::Float(feature2)),
        ("feature3".to_string(), Column::Float(feature3)),
        ("label".to_string(), Column::Float(label)),
    ]).unwrap();
    let create_elapsed = start_create.elapsed();
    eprintln!("DataFrame creation ({} rows): {:?}", n, create_elapsed);

    // Step 2: Extract columns as tensor data
    let start_extract = Instant::now();
    let (feature_data, shape) = df.to_tensor_data(&["feature1", "feature2", "feature3"]).unwrap();
    let extract_elapsed = start_extract.elapsed();
    eprintln!("DataFrame to_tensor_data: {:?} (shape={:?})", extract_elapsed, shape);

    // Step 3: Get label column
    let start_label = Instant::now();
    let label_col = df.get_column("label").unwrap();
    let labels = match label_col {
        Column::Float(v) => v.clone(),
        _ => panic!("expected float column"),
    };
    let label_elapsed = start_label.elapsed();
    eprintln!("Label extraction: {:?}", label_elapsed);

    // Step 4: Compute MSE loss as a simple "training step"
    let start_ml = Instant::now();
    // Simple prediction: just use feature1 as prediction
    let predictions: Vec<f64> = feature_data.iter().step_by(3).cloned().collect();
    let mse = cjc_runtime::ml::mse_loss(&predictions, &labels).unwrap();
    let ml_elapsed = start_ml.elapsed();
    eprintln!("MSE loss computation: {:?} (mse={:.6})", ml_elapsed, mse);

    let total = create_elapsed + extract_elapsed + label_elapsed + ml_elapsed;
    eprintln!("Total pipeline time: {:?}", total);

    // Determinism: re-run extraction and loss
    let (feature_data2, _) = df.to_tensor_data(&["feature1", "feature2", "feature3"]).unwrap();
    assert_eq!(feature_data, feature_data2, "DataFrame extraction not deterministic");

    let predictions2: Vec<f64> = feature_data2.iter().step_by(3).cloned().collect();
    let mse2 = cjc_runtime::ml::mse_loss(&predictions2, &labels).unwrap();
    assert_eq!(mse.to_bits(), mse2.to_bits(), "Pipeline MSE not bit-identical");
}

// ============================================================================
// AREA 5: Additional Determinism Verification
// ============================================================================

#[test]
fn bench_determinism_triple_run_matmul() {
    // Run matmul 3 times with same seed, verify bit-identical
    let n = 128;
    let results: Vec<Vec<f64>> = (0..3).map(|_| {
        let a = rand_tensor(&[n, n], 42);
        let b = rand_tensor(&[n, n], 99);
        a.matmul(&b).unwrap().to_vec()
    }).collect();

    assert_eq!(results[0], results[1], "matmul run 1 != run 2");
    assert_eq!(results[1], results[2], "matmul run 2 != run 3");
    eprintln!("DETERMINISM OK: matmul {}x{} - 3 runs bit-identical", n, n);
}

#[test]
fn bench_determinism_triple_run_lstm() {
    let hidden = 64;
    let batch = 1;
    let input_size = hidden;

    let results: Vec<(Vec<f64>, Vec<f64>)> = (0..3).map(|_| {
        let x = rand_tensor(&[batch, input_size], 42);
        let h_prev = rand_tensor(&[batch, hidden], 43);
        let c_prev = rand_tensor(&[batch, hidden], 44);
        let w_ih = rand_tensor(&[4 * hidden, input_size], 45);
        let w_hh = rand_tensor(&[4 * hidden, hidden], 46);
        let b_ih = rand_tensor(&[4 * hidden], 47);
        let b_hh = rand_tensor(&[4 * hidden], 48);
        let (h, c) = cjc_runtime::ml::lstm_cell(&x, &h_prev, &c_prev, &w_ih, &w_hh, &b_ih, &b_hh).unwrap();
        (h.to_vec(), c.to_vec())
    }).collect();

    assert_eq!(results[0], results[1], "LSTM run 1 != run 2");
    assert_eq!(results[1], results[2], "LSTM run 2 != run 3");
    eprintln!("DETERMINISM OK: LSTM cell hidden={} - 3 runs bit-identical", hidden);
}

#[test]
fn bench_determinism_triple_run_softmax() {
    let n = 10000;
    let results: Vec<Vec<f64>> = (0..3).map(|_| {
        let t = rand_tensor(&[1, n], 42);
        t.softmax().unwrap().to_vec()
    }).collect();

    assert_eq!(results[0], results[1], "softmax run 1 != run 2");
    assert_eq!(results[1], results[2], "softmax run 2 != run 3");
    eprintln!("DETERMINISM OK: softmax n={} - 3 runs bit-identical", n);
}

#[test]
fn bench_determinism_triple_run_svd() {
    let n = 32;
    let results: Vec<(Vec<f64>, Vec<f64>, Vec<f64>)> = (0..3).map(|_| {
        let a = rand_tensor(&[n, n], 42);
        let (u, s, vt) = a.svd().unwrap();
        (u.to_vec(), s, vt.to_vec())
    }).collect();

    assert_eq!(results[0], results[1], "SVD run 1 != run 2");
    assert_eq!(results[1], results[2], "SVD run 2 != run 3");
    eprintln!("DETERMINISM OK: SVD {}x{} - 3 runs bit-identical", n, n);
}

// ============================================================================
// AREA 3b: LU, QR, Cholesky decomposition benchmarks
// ============================================================================

#[test]
fn bench_lu_decomposition() {
    for &n in &[64, 256] {
        let mut data = rand_vec(n * n, 42);
        for i in 0..n {
            data[i * n + i] += n as f64 * 2.0;
        }
        let a = Tensor::from_vec(data, &[n, n]).unwrap();

        let iters = if n <= 64 { 50 } else { 5 };
        let start = Instant::now();
        for _ in 0..iters {
            let _ = a.lu_decompose().unwrap();
        }
        let elapsed = start.elapsed();
        let per_iter = elapsed / iters as u32;
        eprintln!("lu_decompose {}x{}: {:?}/iter", n, n, per_iter);
    }
}

#[test]
fn bench_qr_decomposition() {
    for &n in &[64, 256] {
        let a = rand_tensor(&[n, n], 42);

        let iters = if n <= 64 { 50 } else { 3 };
        let start = Instant::now();
        for _ in 0..iters {
            let _ = a.qr_decompose().unwrap();
        }
        let elapsed = start.elapsed();
        let per_iter = elapsed / iters as u32;
        eprintln!("qr_decompose {}x{}: {:?}/iter", n, n, per_iter);
    }
}

#[test]
fn bench_cholesky() {
    for &n in &[64, 256] {
        // Create positive definite matrix: A = B^T * B + nI
        let raw = rand_vec(n * n, 42);
        let mut sym = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += raw[k * n + i] * raw[k * n + j];
                }
                sym[i * n + j] = sum;
            }
            sym[i * n + i] += n as f64; // ensure positive definite
        }
        let a = Tensor::from_vec(sym, &[n, n]).unwrap();

        let iters = if n <= 64 { 50 } else { 3 };
        let start = Instant::now();
        for _ in 0..iters {
            let _ = a.cholesky().unwrap();
        }
        let elapsed = start.elapsed();
        let per_iter = elapsed / iters as u32;
        eprintln!("cholesky {}x{}: {:?}/iter", n, n, per_iter);
    }
}

// ============================================================================
// AREA 3c: Statistics benchmarks
// ============================================================================

#[test]
fn bench_variance_and_sd() {
    for &n in &[10_000, 100_000, 1_000_000] {
        let data = rand_vec(n, 42);

        let iters = 20;
        let start = Instant::now();
        for _ in 0..iters {
            let _ = cjc_runtime::stats::variance(&data).unwrap();
        }
        let elapsed = start.elapsed();
        eprintln!("variance n={}: {:?}/iter", n, elapsed / iters as u32);

        let r1 = cjc_runtime::stats::variance(&data).unwrap();
        let r2 = cjc_runtime::stats::variance(&data).unwrap();
        assert_eq!(r1.to_bits(), r2.to_bits(), "variance not bit-identical at n={}", n);
    }
}
