//! End-to-end Phase 1 demo: deterministic queue simulator →
//! NSS.fit → predict → replay round-trip → causal attribution print.
//!
//! Run with:
//! ```text
//! cargo run --example queue_pressure_predictor -p cjc-nss
//! ```

use cjc_nss::{
    NeuralSystemsSimulator, NssConfig, NssSeed, PredictionTrace, QueueConfig, QueueSimulator,
    ReplayValidator,
};

fn main() {
    let seed = NssSeed(2026);

    // 1. Generate a deterministic infrastructure trajectory under a
    //    mild-overload regime (queue occasionally collapses).
    let qcfg = QueueConfig {
        workers: 2,
        queue_capacity: 16,
        arrival_rate: 5.0,
        service_min: 0.8,
        service_max: 1.2,
        degraded_knee: 0.65,
        collapse_window: 3,
        retry_amplifier: 0.6,
        propagation: Default::default(),
    };
    let mut sim = QueueSimulator::new(qcfg, seed).expect("valid config");
    let traj = sim.run(128).expect("simulator step");

    let n_collapse = traj
        .iter()
        .filter(|ev| ev.failure.kind == cjc_nss::FailureKind::Collapse)
        .count();
    let n_degraded = traj
        .iter()
        .filter(|ev| ev.failure.kind == cjc_nss::FailureKind::Degraded)
        .count();
    println!(
        "[sim] traj_len={} nominal={} degraded={} collapse={}",
        traj.len(),
        traj.len() - n_collapse - n_degraded,
        n_degraded,
        n_collapse
    );

    // 2. Fit NSS on the trajectory.
    let mut nss =
        NeuralSystemsSimulator::from_seed(NssConfig::default(), seed).expect("valid nss config");
    nss.fit(&traj).expect("fit");

    // 3. Predict the next step from the trajectory's final state.
    let last = traj.last_state().expect("non-empty trajectory");
    let pred = nss.predict_next(last).expect("predict");
    println!(
        "[nss] run_id={} P(collapse)={:.4} P(degraded)={:.4} conf={:.4}",
        pred.run_id,
        pred.failure.collapse_probability,
        pred.failure.degraded_probability,
        pred.failure.confidence,
    );

    // 4. Causal attribution: which pressure kind contributed most to
    //    the collapse logit?
    println!("[nss] causal attribution (top 5):");
    for c in pred.attribution.contributions.iter().take(5) {
        println!("       {:>11}  contribution = {:+.4}", c.kind.label(), c.magnitude);
    }
    println!(
        "[nss] dominant_source = {} (magnitude {:+.4})",
        pred.attribution.dominant_source.kind.label(),
        pred.attribution.dominant_source.magnitude
    );

    // 5. Replay verification — anyone given the trace can reproduce
    //    the prediction without out-of-band knowledge.
    let trace = PredictionTrace {
        transition: pred.transition.clone(),
        input_state: last.clone(),
        input_config: NssConfig::default(),
        input_seed: seed,
        training_trajectory: Some(traj),
    };
    ReplayValidator::new()
        .verify(&trace)
        .expect("replay must verify");
    println!("[replay] verified: {} bytes of canonical trace", trace.canonical_bytes().len());
}
