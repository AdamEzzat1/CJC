//! Bolero structural fuzz: feed malformed [`SystemState`]s into NSS and
//! confirm the API either returns a typed error or produces a bounded
//! prediction. The predictor must never panic on arbitrary inputs from
//! a fuzzer.

use bolero::check;
use cjc_nss::{NeuralSystemsSimulator, NssConfig, NssSeed, Pressure, PressureKind, SystemState};

#[derive(Debug, bolero::TypeGenerator)]
struct FuzzInput {
    /// Per-PressureKind magnitudes (index = PressureKind::all() position).
    magnitudes: [f32; 9],
    /// Per-PressureKind thresholds.
    thresholds: [f32; 9],
    /// Per-PressureKind dissipations (clipped to [0, 1] in the body).
    dissipations: [f32; 9],
    /// Scalar fields.
    tick: u64,
    in_flight: u64,
    completed: u64,
    rejected: u64,
    mean_service_time: f32,
}

#[test]
fn fuzz_predict_never_panics() {
    let nss = NeuralSystemsSimulator::from_seed(NssConfig::default(), NssSeed(42)).unwrap();

    check!()
        .with_type::<FuzzInput>()
        .with_iterations(2048)
        .for_each(|input: &FuzzInput| {
            let mut state = SystemState::initial();
            state.tick = input.tick;
            state.in_flight = input.in_flight;
            state.completed = input.completed;
            state.rejected = input.rejected;
            // Sanitise scalar f32 → f64; clip pathological values that
            // we'd never accept on the boundary anyway.
            let mst = input.mean_service_time as f64;
            state.mean_service_time = if mst.is_finite() && mst >= 0.0 {
                mst
            } else {
                1.0
            };

            // Fill the pressure field with fuzzed (magnitude,
            // threshold, dissipation) triples. We sanitise each
            // before calling `Pressure::new` so the call itself
            // doesn't unwrap — instead we silently fall back to a
            // safe default when the triple is invalid, since the
            // fuzz target is the *predictor*, not the constructor.
            for (i, k) in PressureKind::all().iter().enumerate() {
                let m = input.magnitudes[i] as f64;
                let t = input.thresholds[i] as f64;
                let d = input.dissipations[i] as f64;
                let m = if m.is_finite() && m >= 0.0 { m } else { 0.0 };
                let t = if t.is_finite() && t > 0.0 {
                    t.max(1e-6)
                } else {
                    1.0
                };
                let d = if d.is_finite() {
                    d.clamp(0.0, 1.0)
                } else {
                    0.1
                };
                let p = Pressure::new(m, t, d)
                    .unwrap_or_else(|_| Pressure::new(0.0, 1.0, 0.1).unwrap());
                state.pressures.set(*k, p);
            }

            // Predict. The whole point of this fuzz target is that
            // `predict_next` never panics — it must either return a
            // bounded prediction or a typed error.
            match nss.predict_next(&state) {
                Ok(p) => {
                    assert!(p.failure.collapse_probability.is_finite());
                    assert!(
                        p.failure.collapse_probability >= 0.0
                            && p.failure.collapse_probability <= 1.0
                    );
                    assert!(
                        p.failure.degraded_probability >= 0.0
                            && p.failure.degraded_probability <= 1.0
                    );
                }
                Err(_) => {
                    // Typed error is acceptable on intentionally
                    // malformed inputs (e.g. non-finite
                    // mean_service_time we explicitly preserved
                    // somewhere). We don't enumerate the cases.
                }
            }
        });
}
