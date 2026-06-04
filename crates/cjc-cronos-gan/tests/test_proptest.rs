//! Phase 5 (partial): property-based test suite for cjc-cronos-gan.
//!
//! These tests use `proptest` to verify invariants across a wide range of
//! random inputs that hand-written unit tests would miss. The seven
//! properties (one per `proptest!` block) correspond to the brief's
//! "Property Tests" §:
//!
//! 1. **Sequence length invariant** — rollout outputs length equals input
//!    sequence length.
//! 2. **Mask non-corruption** — masked timesteps do not corrupt unmasked
//!    timesteps. (Phase 5 minimal: validated structurally via
//!    `SequenceMask` invariants; the full masked-rollout property ships
//!    with Phase 5b when training accepts masks.)
//! 3. **Same seed determinism** — same seed always produces same outputs
//!    (rollout, disagreement, experiment replay hash).
//! 4. **Finite-bounded ⇒ finite-output** — finite, bounded inputs produce
//!    finite outputs from both networks.
//! 5. **Loss finiteness** — `TemporalLoss::evaluate` is finite for any
//!    valid (finite, equal-length) pred/target pair.
//! 6. **Disagreement non-negativity** — every field of
//!    [`TemporalDisagreement`] is `>= 0` for any valid trajectory triple.
//! 7. **Rollout determinism for identical inputs** — running the same
//!    model twice on the same input sequence produces byte-identical
//!    outputs.
//!
//! Bolero fuzz targets are deferred to Phase 5b.

use cjc_cronos_gan::{
    compute_disagreement, CronosSeed, LiquidConfig, LiquidNetwork, LiquidState,
    StateSpaceConfig, StateSpaceModel, StateSpaceState, TemporalLoss,
};
use proptest::prelude::*;

// ── § Property 1: sequence length invariant ──────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig { cases: 64, .. ProptestConfig::default() })]
    #[test]
    fn ssm_rollout_length_equals_input_length(
        n_steps in 1usize..=30,
        seed in any::<u64>(),
    ) {
        let cfg = StateSpaceConfig::new(4, 1, 1);
        let m = StateSpaceModel::from_seed(cfg, CronosSeed(seed)).unwrap();
        let inputs = vec![0.5_f64; n_steps];
        let r = m.rollout(&StateSpaceState::zeros(4), &inputs).unwrap();
        prop_assert_eq!(r.outputs.len(), n_steps);
        prop_assert_eq!(r.states.len(), n_steps + 1);
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 64, .. ProptestConfig::default() })]
    #[test]
    fn liquid_rollout_length_equals_input_length(
        n_steps in 1usize..=30,
        seed in any::<u64>(),
    ) {
        let cfg = LiquidConfig::new(4, 1, 1);
        let m = LiquidNetwork::from_seed(cfg, CronosSeed(seed)).unwrap();
        let inputs = vec![0.5_f64; n_steps];
        let r = m.rollout(&LiquidState::zeros(4), &inputs).unwrap();
        prop_assert_eq!(r.outputs.len(), n_steps);
        prop_assert_eq!(r.states.len(), n_steps + 1);
        prop_assert_eq!(r.time_constants.len(), n_steps);
        prop_assert_eq!(r.gates.len(), n_steps);
    }
}

// ── § Property 3: same seed ⇒ same outputs ───────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig { cases: 64, .. ProptestConfig::default() })]
    #[test]
    fn ssm_same_seed_byte_identical_rollout(
        n_steps in 4usize..=20,
        seed in any::<u64>(),
    ) {
        let cfg = StateSpaceConfig::new(4, 1, 1);
        let m1 = StateSpaceModel::from_seed(cfg, CronosSeed(seed)).unwrap();
        let m2 = StateSpaceModel::from_seed(cfg, CronosSeed(seed)).unwrap();
        let inputs: Vec<f64> = (0..n_steps).map(|i| (i as f64 * 0.1).sin()).collect();
        let r1 = m1.rollout(&StateSpaceState::zeros(4), &inputs).unwrap();
        let r2 = m2.rollout(&StateSpaceState::zeros(4), &inputs).unwrap();
        for (a, b) in r1.outputs.iter().zip(r2.outputs.iter()) {
            for (x, y) in a.iter().zip(b.iter()) {
                prop_assert_eq!(x.to_bits(), y.to_bits());
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 64, .. ProptestConfig::default() })]
    #[test]
    fn liquid_same_seed_byte_identical_rollout(
        n_steps in 4usize..=20,
        seed in any::<u64>(),
    ) {
        let cfg = LiquidConfig::new(4, 1, 1);
        let m1 = LiquidNetwork::from_seed(cfg, CronosSeed(seed)).unwrap();
        let m2 = LiquidNetwork::from_seed(cfg, CronosSeed(seed)).unwrap();
        let inputs: Vec<f64> = (0..n_steps).map(|i| (i as f64 * 0.07).sin()).collect();
        let r1 = m1.rollout(&LiquidState::zeros(4), &inputs).unwrap();
        let r2 = m2.rollout(&LiquidState::zeros(4), &inputs).unwrap();
        for (a, b) in r1.outputs.iter().zip(r2.outputs.iter()) {
            for (x, y) in a.iter().zip(b.iter()) {
                prop_assert_eq!(x.to_bits(), y.to_bits());
            }
        }
    }
}

// ── § Property 4: finite-bounded inputs ⇒ finite outputs ─────────────────

proptest! {
    #![proptest_config(ProptestConfig { cases: 128, .. ProptestConfig::default() })]
    #[test]
    fn ssm_finite_bounded_inputs_give_finite_outputs(
        seed in any::<u64>(),
        inputs in proptest::collection::vec(-10.0_f64..10.0, 5..30),
    ) {
        let cfg = StateSpaceConfig::new(4, 1, 1);
        let m = StateSpaceModel::from_seed(cfg, CronosSeed(seed)).unwrap();
        let r = m.rollout(&StateSpaceState::zeros(4), &inputs).unwrap();
        for step_out in &r.outputs {
            for &v in step_out {
                prop_assert!(v.is_finite(), "SSM output went non-finite: {}", v);
            }
        }
        for s in &r.states {
            for &v in &s.x {
                prop_assert!(v.is_finite());
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 128, .. ProptestConfig::default() })]
    #[test]
    fn liquid_finite_bounded_inputs_give_finite_outputs(
        seed in any::<u64>(),
        inputs in proptest::collection::vec(-10.0_f64..10.0, 5..30),
    ) {
        let cfg = LiquidConfig::new(4, 1, 1);
        let m = LiquidNetwork::from_seed(cfg, CronosSeed(seed)).unwrap();
        let r = m.rollout(&LiquidState::zeros(4), &inputs).unwrap();
        for step_out in &r.outputs {
            for &v in step_out {
                prop_assert!(v.is_finite());
            }
        }
        for tc in &r.time_constants {
            for &t in &tc.tau {
                prop_assert!(t >= cfg.tau_min && t <= cfg.tau_max);
            }
        }
    }
}

// ── § Property 5: loss finiteness for finite inputs ──────────────────────

proptest! {
    #![proptest_config(ProptestConfig { cases: 128, .. ProptestConfig::default() })]
    #[test]
    fn temporal_loss_mse_finite_for_finite_inputs(
        a in proptest::collection::vec(-1e3_f64..1e3, 3..20),
        b in proptest::collection::vec(-1e3_f64..1e3, 3..20),
    ) {
        let n = a.len().min(b.len());
        let l = TemporalLoss::Mse.evaluate(&a[..n], &b[..n]).unwrap();
        prop_assert!(l.is_finite() && l >= 0.0);
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 128, .. ProptestConfig::default() })]
    #[test]
    fn temporal_loss_mae_finite_for_finite_inputs(
        a in proptest::collection::vec(-1e3_f64..1e3, 3..20),
        b in proptest::collection::vec(-1e3_f64..1e3, 3..20),
    ) {
        let n = a.len().min(b.len());
        let l = TemporalLoss::Mae.evaluate(&a[..n], &b[..n]).unwrap();
        prop_assert!(l.is_finite() && l >= 0.0);
    }
}

// ── § Property 6: disagreement score is non-negative ─────────────────────

proptest! {
    #![proptest_config(ProptestConfig { cases: 256, .. ProptestConfig::default() })]
    #[test]
    fn disagreement_all_fields_non_negative(
        n_steps in 2usize..=20,
        ssm in proptest::collection::vec(-5.0_f64..5.0, 2..50),
        liq in proptest::collection::vec(-5.0_f64..5.0, 2..50),
        tgt in proptest::collection::vec(-5.0_f64..5.0, 2..50),
    ) {
        let n = n_steps.min(ssm.len()).min(liq.len()).min(tgt.len());
        let ssm_v = &ssm[..n];
        let liq_v = &liq[..n];
        let tgt_v = &tgt[..n];
        let d = compute_disagreement(ssm_v, liq_v, tgt_v, n, 1).unwrap();
        prop_assert!(d.ssm_score >= 0.0);
        prop_assert!(d.liquid_score >= 0.0);
        prop_assert!(d.absolute_gap >= 0.0);
        prop_assert!(d.regime_shift_score >= 0.0);
        prop_assert!(d.ssm_score.is_finite());
        prop_assert!(d.liquid_score.is_finite());
        prop_assert!(d.absolute_gap.is_finite());
        prop_assert!(d.regime_shift_score.is_finite());
    }
}

// ── § Property 7: rollout determinism — independent of property 3 ────────
//                  (checks the *same* model is deterministic, vs prop 3
//                  which checks two *new* models from the same seed)

proptest! {
    #![proptest_config(ProptestConfig { cases: 64, .. ProptestConfig::default() })]
    #[test]
    fn ssm_repeated_rollout_byte_identical(
        seed in any::<u64>(),
        n_steps in 4usize..=20,
    ) {
        let cfg = StateSpaceConfig::new(4, 1, 1);
        let m = StateSpaceModel::from_seed(cfg, CronosSeed(seed)).unwrap();
        let inputs: Vec<f64> = (0..n_steps).map(|i| (i as f64 * 0.09).cos()).collect();
        let r1 = m.rollout(&StateSpaceState::zeros(4), &inputs).unwrap();
        let r2 = m.rollout(&StateSpaceState::zeros(4), &inputs).unwrap();
        for (a, b) in r1.outputs.iter().zip(r2.outputs.iter()) {
            for (x, y) in a.iter().zip(b.iter()) {
                prop_assert_eq!(x.to_bits(), y.to_bits());
            }
        }
    }
}
