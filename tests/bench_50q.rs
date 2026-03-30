use std::time::Instant;

fn eval(src: &str) -> String {
    let (prog, _) = cjc_parser::parse_source(src);
    format!("{}", cjc_eval::Interpreter::new(42).exec(&prog).unwrap())
}

#[test]
fn bench_50q_quantum_from_cjc() {
    println!();
    println!("========== CJC Quantum Benchmarks (CPU-only, no GPU) ==========");
    println!();

    let t = Instant::now();
    let r = eval("fn main() -> Any { let m = mps_new(50, 16); mps_memory(m) }");
    println!("  50q MPS create:       {:>8.2}ms   ({} bytes)", t.elapsed().as_secs_f64()*1000.0, r);

    let t = Instant::now();
    let r = eval("fn main() -> Any { let m = mps_new(50, 16); let m = mps_h(m, 0); let m = mps_h(m, 25); let m = mps_h(m, 49); mps_z_expectation(m, 25) }");
    println!("  50q MPS 3xH + Z-exp:  {:>8.2}ms   (Z={})", t.elapsed().as_secs_f64()*1000.0, r);

    let t = Instant::now();
    let r = eval("fn main() -> Any { vqe_heisenberg(50, 8, 0.05, 1, 42) }");
    println!("  50q VQE 1-step:       {:>8.2}ms   (E={})", t.elapsed().as_secs_f64()*1000.0, r);

    let t = Instant::now();
    let r = eval("fn main() -> Any { vqe_heisenberg(50, 8, 0.05, 10, 42) }");
    println!("  50q VQE 10-step:      {:>8.2}ms   (E={})", t.elapsed().as_secs_f64()*1000.0, r);

    let t = Instant::now();
    eval(r#"fn main() -> Any { let s = stabilizer_new(500); let s = stabilizer_h(s, 0); let s = stabilizer_cnot(s, 0, 1); let s = stabilizer_cnot(s, 1, 2); let s = stabilizer_x(s, 499); stabilizer_measure(s, 499, 42) }"#);
    println!("  500q Stabilizer:      {:>8.2}ms", t.elapsed().as_secs_f64()*1000.0);

    let t = Instant::now();
    eval(r#"fn main() -> Any { let s = stabilizer_new(1000); let s = stabilizer_h(s, 0); let s = stabilizer_cnot(s, 0, 1); let s = stabilizer_x(s, 999); stabilizer_measure(s, 999, 42) }"#);
    println!("  1000q Stabilizer:     {:>8.2}ms", t.elapsed().as_secs_f64()*1000.0);

    let t = Instant::now();
    let r = eval("fn main() -> Any { dmrg_heisenberg(20, 16, 10, 0.0001) }");
    println!("  20q DMRG 10-sweep:    {:>8.2}ms   (E={})", t.elapsed().as_secs_f64()*1000.0, r);

    let t = Instant::now();
    let r = eval("fn main() -> Any { let d = density_new(8); let d = density_gate(d, \"H\", 0); let d = density_depolarize(d, 0, 0.1); density_entropy(d) }");
    println!("  8q Density+noise:     {:>8.2}ms   (S={})", t.elapsed().as_secs_f64()*1000.0, r);

    println!();
    println!("  --- Memory context ---");
    println!("  Statevector 50q would need:  2^50 x 16 bytes = 16,384 TB");
    println!("  MPS 50q actually uses:       < 1 MB");
    println!("  Stabilizer 1000q uses:       ~O(n^2) bits = ~125 KB");
    println!();
}
