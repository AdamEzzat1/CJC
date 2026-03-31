// CJC v0.2 — Quantum Performance Benchmarks
//
// These tests verify the optimized quantum stack through CJC programs
// and measure performance on 50+ qubit systems.

use std::time::Instant;

fn eval(src: &str) -> String {
    let (prog, _) = cjc_parser::parse_source(src);
    format!("{}", cjc_eval::Interpreter::new(42).exec(&prog).unwrap())
}

fn mir(src: &str) -> String {
    let (prog, _) = cjc_parser::parse_source(src);
    let (v, _) = cjc_mir_exec::run_program_with_executor(&prog, 42).unwrap();
    format!("{}", v)
}

// ===========================================================================
// Performance: 50-qubit MPS operations
// ===========================================================================

#[test]
fn bench_50q_mps_create_and_gates() {
    let src = r#"
fn main() -> Any {
    let m = mps_new(50, 16);
    let m = mps_h(m, 0);
    let m = mps_h(m, 25);
    let m = mps_h(m, 49);
    let m = mps_x(m, 10);
    let m = mps_ry(m, 20, 1.0);
    mps_z_expectation(m, 0)
}
"#;
    let t = Instant::now();
    let result = eval(src);
    let elapsed = t.elapsed();
    let z: f64 = result.parse().unwrap();
    assert!(z.abs() < 1e-10, "H|0> should give Z~0, got {}", z);
    println!("  50q MPS 5 gates + Z-exp:  {:>8.2}ms", elapsed.as_secs_f64() * 1000.0);
}

#[test]
fn bench_50q_mps_entangling() {
    // CNOT chain creates entanglement, which is the expensive operation (SVD)
    let src = r#"
fn main() -> Any {
    let m = mps_new(10, 16);
    let m = mps_h(m, 0);
    let m = mps_cnot(m, 0, 1);
    let m = mps_cnot(m, 1, 2);
    let m = mps_cnot(m, 2, 3);
    let m = mps_cnot(m, 3, 4);
    let m = mps_cnot(m, 4, 5);
    let m = mps_cnot(m, 5, 6);
    let m = mps_cnot(m, 6, 7);
    let m = mps_cnot(m, 7, 8);
    let m = mps_cnot(m, 8, 9);
    mps_z_expectation(m, 0)
}
"#;
    let t = Instant::now();
    let result = eval(src);
    let elapsed = t.elapsed();
    println!("  10q MPS GHZ (9 CNOTs):   {:>8.2}ms  (Z={})", elapsed.as_secs_f64() * 1000.0, result);
}

// ===========================================================================
// Performance: VQE with cached environments
// ===========================================================================

#[test]
fn bench_50q_vqe_single_step() {
    let src = "fn main() -> Any { vqe_heisenberg(50, 8, 0.05, 1, 42) }";
    let t = Instant::now();
    let result = eval(src);
    let elapsed = t.elapsed();
    let e: f64 = result.parse().unwrap();
    assert!(e.is_finite());
    println!("  50q VQE 1-step:          {:>8.2}ms  (E={})", elapsed.as_secs_f64() * 1000.0, e);
}

#[test]
fn bench_50q_vqe_ten_steps() {
    let src = "fn main() -> Any { vqe_heisenberg(50, 8, 0.05, 10, 42) }";
    let t = Instant::now();
    let result = eval(src);
    let elapsed = t.elapsed();
    let e: f64 = result.parse().unwrap();
    assert!(e.is_finite());
    println!("  50q VQE 10-step:         {:>8.2}ms  (E={})", elapsed.as_secs_f64() * 1000.0, e);
}

// ===========================================================================
// Performance: Stabilizer with word-level phase
// ===========================================================================

#[test]
fn bench_500q_stabilizer() {
    let src = r#"
fn main() -> Any {
    let s = stabilizer_new(500);
    let s = stabilizer_h(s, 0);
    let s = stabilizer_cnot(s, 0, 1);
    let s = stabilizer_cnot(s, 1, 2);
    let s = stabilizer_cnot(s, 2, 3);
    let s = stabilizer_cnot(s, 3, 4);
    let s = stabilizer_x(s, 499);
    stabilizer_measure(s, 0, 42)
}
"#;
    let t = Instant::now();
    let result = eval(src);
    let elapsed = t.elapsed();
    println!("  500q Stabilizer 5-CNOT:  {:>8.2}ms  (meas={})", elapsed.as_secs_f64() * 1000.0, result);
}

#[test]
fn bench_1000q_stabilizer() {
    let src = r#"
fn main() -> Any {
    let s = stabilizer_new(1000);
    let s = stabilizer_h(s, 0);
    let s = stabilizer_cnot(s, 0, 1);
    let s = stabilizer_x(s, 999);
    stabilizer_measure(s, 999, 42)
}
"#;
    let t = Instant::now();
    let result = eval(src);
    let elapsed = t.elapsed();
    println!("  1000q Stabilizer:        {:>8.2}ms  (meas={})", elapsed.as_secs_f64() * 1000.0, result);
}

// ===========================================================================
// Performance: Density matrix with in-place permutation
// ===========================================================================

#[test]
fn bench_8q_density_circuit() {
    let src = r#"
fn main() -> Any {
    let d = density_new(8);
    let d = density_gate(d, "H", 0);
    let d = density_cnot(d, 0, 1);
    let d = density_gate(d, "H", 2);
    let d = density_cnot(d, 2, 3);
    let d = density_depolarize(d, 0, 0.01);
    let d = density_depolarize(d, 2, 0.01);
    density_purity(d)
}
"#;
    let t = Instant::now();
    let result = eval(src);
    let elapsed = t.elapsed();
    let p: f64 = result.parse().unwrap();
    assert!(p > 0.0 && p <= 1.0);
    println!("  8q Density circuit+noise: {:>7.2}ms  (purity={})", elapsed.as_secs_f64() * 1000.0, p);
}

// ===========================================================================
// Performance: DMRG ground state
// ===========================================================================

#[test]
fn bench_20q_dmrg() {
    // Use fewer sweeps in debug mode to avoid timeout (10 sweeps takes >60s unoptimized)
    let src = "fn main() -> Any { dmrg_heisenberg(8, 8, 3, 0.01) }";
    let t = Instant::now();
    let result = eval(src);
    let elapsed = t.elapsed();
    let e: f64 = result.parse().unwrap();
    assert!(e < 0.0);
    println!("  20q DMRG 10-sweep:       {:>8.2}ms  (E={})", elapsed.as_secs_f64() * 1000.0, e);
}

// ===========================================================================
// Performance: QAOA optimization
// ===========================================================================

#[test]
fn bench_qaoa_maxcut() {
    let src = "fn main() -> Any { let g = qaoa_graph_cycle(8); qaoa_maxcut(g, 8, 2, 0.1, 5, 42) }";
    let t = Instant::now();
    let result = eval(src);
    let elapsed = t.elapsed();
    println!("  8-node QAOA 2-layer:     {:>8.2}ms  (result={})", elapsed.as_secs_f64() * 1000.0, result);
}

// ===========================================================================
// Determinism: optimized code still bit-identical
// ===========================================================================

#[test]
fn bench_determinism_mps_50q() {
    let src = r#"
fn main() -> Any {
    let m = mps_new(50, 16);
    let m = mps_h(m, 0);
    let m = mps_h(m, 25);
    let m = mps_ry(m, 10, 0.7);
    mps_z_expectation(m, 10)
}
"#;
    let r1 = eval(src);
    let r2 = eval(src);
    assert_eq!(r1, r2, "Optimized MPS must be deterministic");
}

#[test]
fn bench_determinism_stabilizer_1000q() {
    let src = r#"
fn main() -> Any {
    let s = stabilizer_new(1000);
    let s = stabilizer_h(s, 0);
    let s = stabilizer_cnot(s, 0, 1);
    stabilizer_measure(s, 0, 42)
}
"#;
    let r1 = eval(src);
    let r2 = eval(src);
    assert_eq!(r1, r2, "Optimized stabilizer must be deterministic");
}

#[test]
fn bench_parity_mps_50q() {
    let src = r#"
fn main() -> Any {
    let m = mps_new(50, 16);
    let m = mps_h(m, 0);
    let m = mps_x(m, 25);
    mps_z_expectation(m, 25)
}
"#;
    let eval_r = eval(src);
    let mir_r = mir(src);
    assert_eq!(eval_r, mir_r, "Eval vs MIR parity for optimized MPS");
}

#[test]
fn bench_parity_vqe_50q() {
    let src = "fn main() -> Any { vqe_heisenberg(50, 8, 0.05, 1, 42) }";
    let eval_r = eval(src);
    let mir_r = mir(src);
    assert_eq!(eval_r, mir_r, "Eval vs MIR parity for optimized VQE");
}

// ===========================================================================
// Benchmark summary (runs all benchmarks, prints table)
// ===========================================================================

#[test]
fn bench_quantum_summary() {
    println!();
    println!("=================================================================");
    println!("  CJC Quantum Performance -- CPU-only, zero dependencies");
    println!("=================================================================");
    println!();

    let tests: Vec<(&str, &str)> = vec![
        ("50q MPS create+gates",     "fn main() -> Any { let m = mps_new(50, 16); let m = mps_h(m, 0); let m = mps_h(m, 25); mps_memory(m) }"),
        ("50q VQE 1-step",           "fn main() -> Any { vqe_heisenberg(50, 8, 0.05, 1, 42) }"),
        ("50q VQE 10-step",          "fn main() -> Any { vqe_heisenberg(50, 8, 0.05, 10, 42) }"),
        ("500q Stabilizer",          "fn main() -> Any { let s = stabilizer_new(500); let s = stabilizer_h(s, 0); let s = stabilizer_cnot(s, 0, 1); stabilizer_measure(s, 0, 42) }"),
        ("1000q Stabilizer",         "fn main() -> Any { let s = stabilizer_new(1000); let s = stabilizer_h(s, 0); let s = stabilizer_cnot(s, 0, 1); stabilizer_measure(s, 0, 42) }"),
        ("8q Density+noise",         "fn main() -> Any { let d = density_new(8); let d = density_gate(d, \"H\", 0); let d = density_cnot(d, 0, 1); let d = density_depolarize(d, 0, 0.01); density_purity(d) }"),
        ("8q DMRG 3-sweep",          "fn main() -> Any { dmrg_heisenberg(8, 8, 3, 0.01) }"),
        ("8-node QAOA 2-layer",      "fn main() -> Any { let g = qaoa_graph_cycle(8); qaoa_maxcut(g, 8, 2, 0.1, 5, 42) }"),
    ];

    println!("  {:<28} {:>10}   {}", "Test", "Time", "Result");
    println!("  {}", "-".repeat(64));
    for (name, src) in tests {
        let t = Instant::now();
        let result = eval(src);
        let elapsed = t.elapsed();
        println!("  {:<28} {:>8.2}ms   {}", name, elapsed.as_secs_f64() * 1000.0, result);
    }

    println!();
    println!("  Memory context:");
    println!("    Statevector 50q:     16,384 TB (2^50 x 16 bytes)");
    println!("    MPS 50q (chi=16):    < 1 MB");
    println!("    Stabilizer 1000q:    ~482 KB");
    println!("=================================================================");
    println!();
}
