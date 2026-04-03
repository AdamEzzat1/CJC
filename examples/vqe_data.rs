//! Data extraction for VQE visualization.
use cjc_quantum::vqe::*;

fn main() {
    // 4-qubit VQE convergence
    let result = vqe_heisenberg_1d(4, 16, 0.15, 20, 42);
    println!("CONVERGENCE");
    for (i, e) in result.energy_history.iter().enumerate() {
        println!("{},{:.8}", i, e);
    }

    // Memory scaling: MPS vs Statevector
    println!("MEMORY");
    for n in [4usize, 8, 16, 32, 50, 100] {
        let thetas = vec![0.1; n];
        let mps = build_mps_ansatz(n, &thetas, 8);
        let mps_mem = mps.memory_bytes();
        let sv_mem: u128 = if n <= 30 { (1u128 << n) * 16 } else { (1u128 << n) * 16 };
        println!("{},{},{}", n, mps_mem, sv_mem);
    }

    // Determinism proof
    let r1 = vqe_heisenberg_1d(4, 16, 0.1, 10, 123);
    let r2 = vqe_heisenberg_1d(4, 16, 0.1, 10, 123);
    println!("DETERMINISM");
    for (i, (e1, e2)) in r1.energy_history.iter().zip(&r2.energy_history).enumerate() {
        println!("{},{:.15},{:.15},{}", i, e1, e2, e1.to_bits() == e2.to_bits());
    }

    // 50-qubit VQE
    println!("FIFTY");
    let start = std::time::Instant::now();
    let r50 = vqe_heisenberg_1d(50, 8, 0.05, 3, 42);
    let elapsed = start.elapsed();
    for (i, e) in r50.energy_history.iter().enumerate() {
        println!("{},{:.8}", i, e);
    }
    println!("time_ms={}", elapsed.as_millis());
}
