//! **HEADLINE TEST**: cross-run posterior byte-identity.
//!
//! Per ADR-0045 §determinism: same model + same seed ⇒ byte-identical
//! `PosteriorSamples.content_hash` AND byte-identical sample bit patterns.
//! If this test fails on any platform at any point, the release does NOT ship.
//!
//! Cross-platform parity is verified separately by CI (`ubuntu-latest`,
//! `macos-latest`, `windows-latest`). The local test here proves cross-RUN
//! parity, which is the necessary precondition.

use cjc_tempest::{MetropolisHastings, PosteriorSamples};

fn log_p_unit_gaussian(x: &[f64]) -> f64 {
    -0.5 * x.iter().map(|v| v * v).sum::<f64>()
}

#[test]
fn headline_same_seed_byte_identical_posterior_1d() {
    let mh = MetropolisHastings::new();
    let r1 = mh
        .run(log_p_unit_gaussian, &[0.0], 1, 100, 200, 42)
        .unwrap();
    let r2 = mh
        .run(log_p_unit_gaussian, &[0.0], 1, 100, 200, 42)
        .unwrap();
    assert_byte_identical(&r1, &r2);
}

#[test]
fn headline_same_seed_byte_identical_posterior_5d() {
    let mh = MetropolisHastings::new();
    let init = vec![0.0; 5];
    let r1 = mh
        .run(log_p_unit_gaussian, &init, 4, 200, 500, 17)
        .unwrap();
    let r2 = mh
        .run(log_p_unit_gaussian, &init, 4, 200, 500, 17)
        .unwrap();
    assert_byte_identical(&r1, &r2);
}

#[test]
fn headline_same_seed_byte_identical_under_init_sigma_change_in_grid() {
    // Two runs with the same seed but different init_sigma SHOULD produce
    // different posteriors (init_sigma affects the proposal kernel).
    let r1 = MetropolisHastings::new()
        .with_init_sigma(0.5)
        .run(log_p_unit_gaussian, &[0.0], 1, 100, 200, 42)
        .unwrap();
    let r2 = MetropolisHastings::new()
        .with_init_sigma(1.0)
        .run(log_p_unit_gaussian, &[0.0], 1, 100, 200, 42)
        .unwrap();
    assert_ne!(
        r1.content_hash, r2.content_hash,
        "init_sigma is part of the proposal kernel; different values must give different chains",
    );
}

#[test]
fn headline_doubling_n_chains_does_not_change_per_chain_hashes() {
    // Per the ADR-0045 §test surface proptest property: doubling n_chains
    // should not change the per-chain content hashes. We verify by hashing
    // chain 0's bytes under both `n_chains=2` and `n_chains=4`.
    let r2 = MetropolisHastings::new()
        .run(log_p_unit_gaussian, &[0.0], 2, 100, 200, 42)
        .unwrap();
    let r4 = MetropolisHastings::new()
        .run(log_p_unit_gaussian, &[0.0], 4, 100, 200, 42)
        .unwrap();
    // Chain 0 should be byte-identical across the two runs because the
    // per-chain seed stretch makes chain 0 depend only on base_seed +
    // chain_index = 0.
    for s in 0..200 {
        for d in 0..1 {
            assert_eq!(
                r2.chains[0][s][d].to_bits(),
                r4.chains[0][s][d].to_bits(),
                "chain 0 sample {} differs between n_chains=2 and n_chains=4",
                s,
            );
        }
    }
}

fn assert_byte_identical(r1: &PosteriorSamples, r2: &PosteriorSamples) {
    assert_eq!(r1.content_hash, r2.content_hash);
    assert_eq!(r1.n_chains, r2.n_chains);
    assert_eq!(r1.n_samples_per_chain, r2.n_samples_per_chain);
    assert_eq!(r1.n_dim, r2.n_dim);
    for c in 0..r1.n_chains {
        for s in 0..r1.n_samples_per_chain {
            for d in 0..r1.n_dim {
                assert_eq!(
                    r1.chains[c][s][d].to_bits(),
                    r2.chains[c][s][d].to_bits(),
                    "chain {} sample {} dim {} differs",
                    c, s, d,
                );
            }
        }
    }
}
