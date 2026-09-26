//! Cross-platform bit-identity of quantum simulation (ADR-0046).
//!
//! Every code path that evaluates a transcendental function is driven through
//! `dispatch_quantum` (the entry point both executors use), and the raw bits
//! of every float in every result are hashed. CI runs this on Linux, Windows,
//! and macOS: the hash must be the same everywhere. Before ADR-0046 the gate
//! matrices used the platform libm, and Windows and Linux disagreed on 6% of
//! rotation angles, so a hash like this one could not have been shared.
//!
//! If a deliberate change to the simulation alters these bits, update
//! `GOLDEN` and state why in the commit.

use std::rc::Rc;

use cjc_quantum::dispatch_quantum;
use cjc_runtime::value::Value;

fn call(name: &str, args: &[Value]) -> Value {
    dispatch_quantum(name, args)
        .unwrap_or_else(|e| panic!("{name} failed: {e}"))
        .unwrap_or_else(|| panic!("{name} not handled"))
}

fn s(x: &str) -> Value {
    Value::String(Rc::new(x.to_string()))
}

/// FNV-1a over the exact bit patterns of every number in a value tree.
struct Hasher(u64);

impl Hasher {
    fn bytes(&mut self, b: &[u8]) {
        for &x in b {
            self.0 ^= x as u64;
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    fn value(&mut self, v: &Value) {
        match v {
            Value::Float(f) => self.bytes(&f.to_bits().to_le_bytes()),
            Value::Int(i) => self.bytes(&i.to_le_bytes()),
            Value::Complex(c) => {
                self.bytes(&c.re.to_bits().to_le_bytes());
                self.bytes(&c.im.to_bits().to_le_bytes());
            }
            Value::Array(a) => a.iter().for_each(|x| self.value(x)),
            other => panic!("unexpected value in golden hash: {}", other.type_name()),
        }
    }
}

fn angles() -> Vec<f64> {
    let mut v = Vec::new();
    for k in -24i32..=24 {
        v.push(k as f64 * std::f64::consts::FRAC_PI_2); // exact multiples, incl. π/2
        v.push(k as f64 * 0.3711); // generic angles
    }
    v.push(1e7 + 0.5); // Payne–Hanek range
    v
}

fn outputs() -> Vec<Value> {
    let mut out = Vec::new();

    // Dense rotation gates (gates.rs), both backends.
    for backend in [None, Some("pure")] {
        for chunk in angles().chunks(6) {
            let mut c = match backend {
                Some(b) => call("qubits", &[Value::Int(3), s(b)]),
                None => call("qubits", &[Value::Int(3)]),
            };
            for (i, &t) in chunk.iter().enumerate() {
                let gate = ["q_rx", "q_ry", "q_rz"][i % 3];
                c = call(gate, &[c, Value::Int((i % 3) as i64), Value::Float(t)]);
                c = call("q_h", &[c, Value::Int(((i + 1) % 3) as i64)]);
            }
            out.push(call("q_probs", &[c.clone()]));
            if backend.is_none() {
                out.push(call("q_amplitudes", &[c.clone()]));
                out.push(call("q_sample", &[c, Value::Int(16), Value::Int(3)]));
            }
        }
    }

    // MPS rotation (dispatch.rs mps_ry), both backends.
    for backend in [None, Some("pure")] {
        let mut m = match backend {
            Some(b) => call("mps_new", &[Value::Int(4), Value::Int(4), s(b)]),
            None => call("mps_new", &[Value::Int(4), Value::Int(4)]),
        };
        for (i, &t) in angles().iter().take(12).enumerate() {
            m = call("mps_ry", &[m, Value::Int((i % 4) as i64), Value::Float(t)]);
            if i % 4 != 3 {
                m = call("mps_cnot", &[m, Value::Int((i % 4) as i64), Value::Int((i % 4 + 1) as i64)]);
            }
        }
        for q in 0..4 {
            out.push(call("mps_z_expectation", &[m.clone(), Value::Int(q)]));
        }
    }

    // Trotter phases (trotter.rs) and fermion expectations.
    let h = call("q_fermion_lih", &[]);
    let mut c = call("qubits", &[Value::Int(4)]);
    c = call("q_x", &[c, Value::Int(0)]);
    c = call("q_x", &[c, Value::Int(1)]);
    c = call("q_ry", &[c, Value::Int(2), Value::Float(0.2)]);
    for order in [1, 2] {
        let evolved = call(
            "q_trotter_evolve",
            &[h.clone(), c.clone(), Value::Float(0.7), Value::Int(9), Value::Int(order)],
        );
        out.push(call("q_amplitudes", &[evolved.clone()]));
        out.push(call("q_fermion_expectation", &[h.clone(), evolved]));
    }

    // Density rotations, channels, entropy (density.rs: sin/cos/ln), both backends.
    for backend in [None, Some("pure")] {
        let mut d = match backend {
            Some(b) => call("density_new", &[Value::Int(2), s(b)]),
            None => call("density_new", &[Value::Int(2)]),
        };
        d = call("density_gate", &[d, s("H"), Value::Int(0)]);
        d = call("density_cnot", &[d, Value::Int(0), Value::Int(1)]);
        d = call("density_amplitude_damp", &[d, Value::Int(0), Value::Float(0.3)]);
        d = call("density_depolarize", &[d, Value::Int(1), Value::Float(0.11)]);
        out.push(call("density_probs", &[d.clone()]));
        out.push(call("density_entropy", &[d]));
    }

    // Variational algorithms (qaoa.rs, vqe.rs, qml.rs: sin/cos/exp/ln).
    let g = call("qaoa_graph_cycle", &[Value::Int(4)]);
    out.push(call(
        "qaoa_maxcut",
        &[g, Value::Int(8), Value::Int(1), Value::Float(0.1), Value::Int(3), Value::Int(42)],
    ));
    out.push(call(
        "vqe_heisenberg",
        &[Value::Int(4), Value::Int(4), Value::Float(0.05), Value::Int(3), Value::Int(7)],
    ));
    let row = |a: f64, b: f64| Value::Array(Rc::new(vec![Value::Float(a), Value::Float(b)]));
    out.push(call(
        "qml_train",
        &[
            Value::Int(2), Value::Int(1), Value::Int(2), Value::Int(4), Value::Float(0.1),
            Value::Int(2), Value::Int(5),
            Value::Array(Rc::new(vec![row(0.1, 0.9), row(0.8, 0.2), row(0.3, 0.7), row(0.9, 0.1)])),
            Value::Array(Rc::new(vec![Value::Int(0), Value::Int(1), Value::Int(0), Value::Int(1)])),
        ],
    ));

    // Noise scaling (mitigation.rs: pow).
    for noise in ["depolarizing", "dephasing", "amplitude_damping"] {
        out.push(call("q_scale_noise", &[Value::Float(0.013), Value::Float(2.5), s(noise)]));
    }
    out
}

/// Hash recorded on Windows 11 (UCRT) and confirmed identical on Debian 12
/// (glibc 2.36); see docs/quantum_simulation_research_stack/verification/.
///
/// History:
/// - `0x65ee_ad68_55ab_878d`: before 2026-09-25.
/// - `0x5957_1534_8913_7325`: the MPS SVD's Jacobi rotation had the wrong
///   sign (mps.rs `jacobi_rotation_complex`), so its singular values were
///   wrong whenever two columns had unequal norms. The fix changes the MPS
///   and `qml_train` outputs hashed here, and nothing else: reverting only
///   that sign reproduces the old hash with every other 2026-09-25 change
///   (gate kernels, batch sampler, execution cache) in place. Recorded on
///   Windows 11; not yet re-run on Linux (the fix uses only IEEE + - * / and
///   sqrt, which are correctly rounded on every platform, so the Linux run
///   is expected to match).
const GOLDEN: u64 = 0x5957_1534_8913_7325;

#[test]
fn quantum_outputs_are_bit_identical_across_platforms() {
    let mut h = Hasher(0xcbf2_9ce4_8422_2325);
    for v in outputs() {
        h.value(&v);
    }
    assert_eq!(h.0, GOLDEN, "quantum golden hash changed: got {:#018x}", h.0);
}

#[test]
fn quantum_outputs_replay_identically() {
    // Same process, two runs: guards the golden test against hidden state.
    let a = format!("{:?}", outputs().iter().map(|v| v.to_string()).collect::<Vec<_>>());
    let b = format!("{:?}", outputs().iter().map(|v| v.to_string()).collect::<Vec<_>>());
    assert_eq!(a, b);
}
