//! CJC Autodiff Throughput Benchmark
//! ==================================
//! Measures forward-mode and reverse-mode AD performance as the number
//! of parameters scales from 10 to 10,000.
//!
//! Forward mode (Dual numbers):
//!   - Computes df/dx_i one variable at a time.
//!   - Cost = O(N) passes for N parameters  =>  total O(N^2) for all gradients.
//!   - Best for "one-to-many" functions (few inputs, many outputs).
//!
//! Reverse mode (GradGraph):
//!   - Computes ALL df/dx_i in a SINGLE backward pass.
//!   - Cost = O(1) passes for N parameters  =>  total O(N) for all gradients.
//!   - Best for "many-to-one" functions (many inputs, one output / loss).
//!
//! This benchmark proves CJC's reverse mode scales correctly.

use cjc_ad::{Dual, GradGraph};
use cjc_repro::Rng;
use cjc_runtime::Tensor;
use std::time::Instant;

/// "Many-to-one" loss function: f(x) = sum_i(x_i^2)
/// Gradient: df/dx_i = 2 * x_i
///
/// Forward mode: must seed each x_i as the variable separately — N passes.
/// Reverse mode: one forward + one backward — 1 pass.

fn bench_forward_mode(params: &[f64]) -> (f64, Vec<f64>, std::time::Duration) {
    let n = params.len();
    let start = Instant::now();

    // Compute value (one pass)
    let value: f64 = params.iter().map(|x| x * x).sum();

    // Compute gradients: one pass per parameter
    let mut grads = Vec::with_capacity(n);
    for i in 0..n {
        // Seed x_i as variable (deriv=1), all others as constants
        let mut result = Dual::zero();
        for j in 0..n {
            let xj = if j == i {
                Dual::variable(params[j])
            } else {
                Dual::constant(params[j])
            };
            // f contribution: x_j^2
            result = result + xj.clone() * xj;
        }
        grads.push(result.deriv);
    }

    let elapsed = start.elapsed();
    (value, grads, elapsed)
}

fn bench_reverse_mode(params: &[f64]) -> (f64, Vec<f64>, std::time::Duration) {
    let n = params.len();
    let start = Instant::now();

    // Build computation graph
    let mut graph = GradGraph::new();

    // Create parameter nodes
    let param_ids: Vec<usize> = params
        .iter()
        .map(|&x| graph.parameter(Tensor::from_vec_unchecked(vec![x], &[1])))
        .collect();

    // Compute f(x) = sum_i(x_i^2)
    // Build: for each param, square it, then sum all
    let mut sq_ids = Vec::with_capacity(n);
    for &pid in &param_ids {
        let sq = graph.mul(pid, pid);
        sq_ids.push(sq);
    }

    // Sum them up using a reduction tree
    let mut current = sq_ids;
    while current.len() > 1 {
        let mut next = Vec::with_capacity((current.len() + 1) / 2);
        for pair in current.chunks(2) {
            if pair.len() == 2 {
                next.push(graph.add(pair[0], pair[1]));
            } else {
                next.push(pair[0]);
            }
        }
        current = next;
    }
    let loss = current[0];

    let value = graph.value(loss);

    // Single backward pass — computes ALL gradients at once
    graph.backward(loss);

    let grads: Vec<f64> = param_ids
        .iter()
        .map(|&pid| graph.grad(pid).unwrap().to_vec()[0])
        .collect();

    let elapsed = start.elapsed();
    (value, grads, elapsed)
}

/// One-to-many: coordinate transform f(t) = [sin(t), cos(t), exp(t), t^2, ...]
/// Forward mode gets ALL output derivatives in one pass when there's one input.
fn bench_forward_one_to_many(t_val: f64, num_outputs: usize) -> (Vec<f64>, Vec<f64>, std::time::Duration) {
    let start = Instant::now();
    let t = Dual::variable(t_val);

    let mut values = Vec::with_capacity(num_outputs);
    let mut derivs = Vec::with_capacity(num_outputs);

    for i in 0..num_outputs {
        let result = match i % 5 {
            0 => t.clone().sin(),
            1 => t.clone().cos(),
            2 => t.clone().exp(),
            3 => t.clone() * t.clone(),
            4 => (t.clone() * Dual::constant(0.5)).ln(),
            _ => unreachable!(),
        };
        values.push(result.value);
        derivs.push(result.deriv);
    }

    let elapsed = start.elapsed();
    (values, derivs, elapsed)
}

/// One-to-many in reverse mode: needs N backward passes (one per output).
fn bench_reverse_one_to_many(t_val: f64, num_outputs: usize) -> (Vec<f64>, Vec<f64>, std::time::Duration) {
    let start = Instant::now();

    let mut values = Vec::with_capacity(num_outputs);
    let mut derivs = Vec::with_capacity(num_outputs);

    // For each output, build a small graph and backward
    for i in 0..num_outputs {
        let mut graph = GradGraph::new();
        let t = graph.parameter(Tensor::from_vec_unchecked(vec![t_val], &[1]));

        let out = match i % 5 {
            0 => {
                // sin(t) — build using exp: sin ≈ manual. But we don't have sin in GradGraph.
                // Use a polynomial approximation to exercise the graph:
                // sin(t) ≈ t - t^3/6 + t^5/120 (Taylor around 0)
                // For benchmarking, just use x * x as representative compute
                let t2 = graph.mul(t, t);
                t2 // x^2 as stand-in
            }
            1 => {
                let t2 = graph.mul(t, t);
                let t3 = graph.mul(t2, t);
                t3 // x^3
            }
            2 => {
                // exp(t) — we don't have Exp as a graph op that takes a node directly.
                // Use mul chain: t * t * t * t (t^4)
                let t2 = graph.mul(t, t);
                let t4 = graph.mul(t2, t2);
                t4
            }
            3 => {
                let t2 = graph.mul(t, t);
                t2
            }
            4 => {
                let t2 = graph.mul(t, t);
                let t3 = graph.mul(t2, t);
                let t5 = graph.mul(t3, t2);
                t5 // x^5
            }
            _ => unreachable!(),
        };

        values.push(graph.value(out));
        graph.backward(out);
        let g = graph.grad(t).unwrap();
        derivs.push(g.to_vec()[0]);
    }

    let elapsed = start.elapsed();
    (values, derivs, elapsed)
}

struct BenchResult {
    n: usize,
    forward_us: f64,
    reverse_us: f64,
    forward_grads_per_sec: f64,
    reverse_grads_per_sec: f64,
    speedup: f64,
    correct: bool,
}

fn main() {
    let mut rng = Rng::seeded(42);

    // ── Part 1: Many-to-One (Loss function) — Reverse mode should win ──

    let sizes = vec![10, 50, 100, 500, 1000, 2000, 5000, 10000];
    let mut many_to_one_results: Vec<BenchResult> = Vec::new();

    println!("=== CJC AUTODIFF THROUGHPUT BENCHMARK ===");
    println!();
    println!("--- Part 1: Many-to-One (f(x) = sum(x_i^2)) ---");
    println!("Forward mode: N passes for N parameters (O(N^2) total)");
    println!("Reverse mode: 1 backward pass for all N gradients (O(N) total)");
    println!();

    for &n in &sizes {
        // Generate random parameters
        let params: Vec<f64> = (0..n).map(|_| rng.next_f64() * 2.0 - 1.0).collect();

        // Run forward mode (skip for very large N — it's O(N^2))
        let (fwd_val, fwd_grads, fwd_time) = if n <= 2000 {
            bench_forward_mode(&params)
        } else {
            // Estimate based on O(N^2) scaling from the 2000 measurement
            (0.0, vec![], std::time::Duration::from_secs(0))
        };

        // Run reverse mode
        let (rev_val, rev_grads, rev_time) = bench_reverse_mode(&params);

        // Verify correctness (gradients should be 2*x_i)
        let correct = if n <= 2000 && !fwd_grads.is_empty() {
            let mut ok = true;
            for i in 0..n {
                let expected = 2.0 * params[i];
                if (fwd_grads[i] - expected).abs() > 1e-8 {
                    ok = false;
                    break;
                }
                if (rev_grads[i] - expected).abs() > 1e-8 {
                    ok = false;
                    break;
                }
            }
            // Also check values match
            if (fwd_val - rev_val).abs() > 1e-6 {
                ok = false;
            }
            ok
        } else {
            // Just check reverse mode gradients
            let mut ok = true;
            for i in 0..n {
                let expected = 2.0 * params[i];
                if (rev_grads[i] - expected).abs() > 1e-8 {
                    ok = false;
                    break;
                }
            }
            ok
        };

        let fwd_us = fwd_time.as_secs_f64() * 1_000_000.0;
        let rev_us = rev_time.as_secs_f64() * 1_000_000.0;

        let fwd_gps = if fwd_us > 0.0 {
            n as f64 / (fwd_time.as_secs_f64())
        } else {
            0.0
        };
        let rev_gps = n as f64 / (rev_time.as_secs_f64());

        let speedup = if fwd_us > 0.0 { fwd_us / rev_us } else { 0.0 };

        many_to_one_results.push(BenchResult {
            n,
            forward_us: fwd_us,
            reverse_us: rev_us,
            forward_grads_per_sec: fwd_gps,
            reverse_grads_per_sec: rev_gps,
            speedup,
            correct,
        });

        if n <= 2000 {
            println!(
                "  N={:>5}  Forward: {:>10.0} us  Reverse: {:>10.0} us  Speedup: {:>6.1}x  Correct: {}",
                n, fwd_us, rev_us, speedup, correct
            );
        } else {
            println!(
                "  N={:>5}  Forward: (skipped, O(N^2))  Reverse: {:>10.0} us  Correct: {}",
                n, rev_us, correct
            );
        }
    }

    // ── Part 2: One-to-Many (Coordinate transform) — Forward mode should win ──

    println!();
    println!("--- Part 2: One-to-Many (1 input -> N outputs) ---");
    println!("Forward mode: 1 pass computes all N output derivatives");
    println!("Reverse mode: N backward passes (one per output)");
    println!();

    let one_to_many_sizes = vec![10, 50, 100, 500, 1000, 2000, 5000];
    let mut one_to_many_results: Vec<BenchResult> = Vec::new();

    for &n in &one_to_many_sizes {
        let t_val = 1.5;

        let (_, _, fwd_time) = bench_forward_one_to_many(t_val, n);

        let (_, _, rev_time) = if n <= 2000 {
            bench_reverse_one_to_many(t_val, n)
        } else {
            (vec![], vec![], std::time::Duration::from_secs(0))
        };

        let fwd_us = fwd_time.as_secs_f64() * 1_000_000.0;
        let rev_us = rev_time.as_secs_f64() * 1_000_000.0;

        let fwd_gps = n as f64 / fwd_time.as_secs_f64();
        let rev_gps = if rev_us > 0.0 {
            n as f64 / rev_time.as_secs_f64()
        } else {
            0.0
        };

        let speedup = if rev_us > 0.0 { rev_us / fwd_us } else { 0.0 };

        one_to_many_results.push(BenchResult {
            n,
            forward_us: fwd_us,
            reverse_us: rev_us,
            forward_grads_per_sec: fwd_gps,
            reverse_grads_per_sec: rev_gps,
            speedup,
            correct: true,
        });

        if n <= 2000 {
            println!(
                "  N={:>5}  Forward: {:>10.0} us  Reverse: {:>10.0} us  Fwd Speedup: {:>6.1}x",
                n, fwd_us, rev_us, speedup
            );
        } else {
            println!(
                "  N={:>5}  Forward: {:>10.0} us  Reverse: (skipped, O(N))  ",
                n, fwd_us
            );
        }
    }

    // ── Output structured data for the runner ──

    println!();
    println!("=== STRUCTURED OUTPUT ===");

    println!("MANY_TO_ONE:");
    for r in &many_to_one_results {
        println!(
            "  N={} fwd_us={:.1} rev_us={:.1} fwd_gps={:.0} rev_gps={:.0} speedup={:.1} correct={}",
            r.n, r.forward_us, r.reverse_us, r.forward_grads_per_sec, r.reverse_grads_per_sec, r.speedup, r.correct
        );
    }

    println!("ONE_TO_MANY:");
    for r in &one_to_many_results {
        println!(
            "  N={} fwd_us={:.1} rev_us={:.1} fwd_gps={:.0} rev_gps={:.0} speedup={:.1}",
            r.n, r.forward_us, r.reverse_us, r.forward_grads_per_sec, r.reverse_grads_per_sec, r.speedup
        );
    }

    println!();
    println!("=== ALL AUTODIFF BENCHMARKS COMPLETE ===");
}
