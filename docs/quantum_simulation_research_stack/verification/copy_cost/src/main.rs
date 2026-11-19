//! Cost of one gate vs. one deep copy for each eager simulator state.
//! Evidence for ADR-0044 (quantum value semantics): under copy-on-write,
//! every `s = gate(s, ...)` pays one copy, because the binding and the
//! argument vector both hold the Rc (measured refcount 2 in both executors).
//!
//!     cargo run --release
use cjc_quantum::circuit::Circuit;
use cjc_quantum::density::DensityMatrix;
use cjc_quantum::gates::Gate;
use cjc_quantum::mps::Mps;
use cjc_quantum::stabilizer::StabilizerState;
use cjc_runtime::complex::ComplexF64;
use std::hint::black_box;
use std::time::Instant;

fn time<F: FnMut()>(reps: usize, mut f: F) -> f64 {
    f(); // warm
    let t = Instant::now();
    for _ in 0..reps {
        f();
    }
    t.elapsed().as_secs_f64() / reps as f64 * 1e6 // µs
}

fn row(name: &str, gate_us: f64, copy_us: f64) {
    println!("{:<34} gate {:>10.3} µs   copy {:>10.3} µs   copy/gate {:>8.1}x", name, gate_us, copy_us, copy_us / gate_us);
}

fn main() {
    for &n in &[100usize, 1000, 10000] {
        let mut s = StabilizerState::new(n);
        for q in 0..n - 1 {
            s.h(q);
            s.cnot(q, q + 1);
        }
        let reps = if n >= 10000 { 20 } else { 200 };
        let g = time(reps, || s.h(n / 2));
        let c = time(reps, || { black_box(s.clone()); });
        row(&format!("stabilizer n={}", n), g, c);
    }
    let h = {
        let r = 1.0 / 2f64.sqrt();
        [[ComplexF64::real(r), ComplexF64::real(r)], [ComplexF64::real(r), ComplexF64::real(-r)]]
    };
    for &(n, chi) in &[(50usize, 16usize), (100, 32)] {
        let mut m = Mps::with_max_bond(n, chi);
        for q in 0..n {
            m.apply_single_qubit(q, h);
        }
        for _ in 0..4 {
            for q in 0..n - 1 {
                m.apply_cnot_adjacent(q, q + 1);
                m.apply_single_qubit(q, h);
            }
        }
        let g = time(200, || m.apply_single_qubit(n / 2, h));
        let c = time(200, || { black_box(m.clone()); });
        row(&format!("mps n={} chi={} (bond-saturated)", n, chi), g, c);
    }
    for &n in &[6usize, 8, 10] {
        let mut d = DensityMatrix::new(n);
        let reps = if n >= 10 { 20 } else { 200 };
        let g = time(reps, || d.apply_gate(&Gate::H(0)));
        let c = time(reps, || { black_box(d.clone()); });
        row(&format!("density n={}", n), g, c);
    }
    for &g_count in &[100usize, 1000, 10000] {
        let mut circ = Circuit::new(8);
        for i in 0..g_count {
            circ.add(Gate::H(i % 8));
        }
        let add = time(2000, || { let mut c2 = Circuit::new(8); c2.add(Gate::X(0)); black_box(c2); });
        let c = time(200, || { black_box(circ.clone()); });
        row(&format!("circuit gates={} (clone vs append)", g_count), add, c);
    }
}
