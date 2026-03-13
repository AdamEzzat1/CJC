//! Phase 11: Deterministic replay tests.
//!
//! Tests that games can be replayed with identical results given
//! the same seed and weights.

use super::helpers::*;

/// Same seed + same weights = identical game output.
#[test]
fn replay_identical_game() {
    let src = full_program(r#"
        let weights = init_weights();
        let W1 = weights[0];
        let b1 = weights[1];
        let W2 = weights[2];
        let result = play_episode(W1, b1, W2, 10);
        print(result[0]);
        print(result[1]);
    "#);
    let out1 = run_mir(&src, 42);
    let out2 = run_mir(&src, 42);
    assert_eq!(out1, out2, "replay should produce identical output");
}

/// Replay across 5 runs: all identical.
#[test]
fn replay_5_runs_identical() {
    let src = full_program(r#"
        let weights = init_weights();
        let result = play_episode(weights[0], weights[1], weights[2], 8);
        print(result[0]);
        print(result[1]);
    "#);
    let baseline = run_mir(&src, 123);
    for _ in 0..4 {
        let out = run_mir(&src, 123);
        assert_eq!(out, baseline, "replay run not identical to baseline");
    }
}

/// Training replay: same seed = identical training trajectory.
#[test]
fn replay_training_identical() {
    let src = multi_program(r#"
        let result = train_multi_episodes(3, 0.01, 0.99, 0.0, 8);
        print(result[0].get([0, 0]));
    "#);
    let out1 = run_mir(&src, 42);
    let out2 = run_mir(&src, 42);
    assert_eq!(out1, out2, "training replay not identical");
}

/// Self-play replay: same seed = identical match.
#[test]
fn replay_selfplay_identical() {
    let src = selfplay_program(r#"
        let w1 = init_weights();
        let w2 = init_weights();
        let result = selfplay_episode(
            w1[0], w1[1], w1[2],
            w2[0], w2[1], w2[2],
            10
        );
        print(result[0]);
        print(result[1]);
    "#);
    let out1 = run_mir(&src, 42);
    let out2 = run_mir(&src, 42);
    assert_eq!(out1, out2, "selfplay replay not identical");
}

/// Replay with different seed produces different weight initialization.
#[test]
fn replay_different_seed_differs() {
    // With short max_moves, many games draw with the same result.
    // Verify that different seeds at least produce different weight values.
    let src = full_program(r#"
        let weights = init_weights();
        print(weights[0].get([0, 0]));
        print(weights[1].get([0, 0]));
        print(weights[2].get([0, 0]));
    "#);
    let out1 = run_mir(&src, 42);
    let out2 = run_mir(&src, 43);
    assert_ne!(out1, out2, "different seeds should produce different weights");
}

/// Win-rate evaluation replay is deterministic.
#[test]
fn replay_win_rate_deterministic() {
    let src = multi_program(r#"
        let weights = init_weights();
        let wr = eval_win_rate(weights[0], weights[1], weights[2], 4, 8, 1);
        print(wr);
    "#);
    let out1 = run_mir(&src, 42);
    let out2 = run_mir(&src, 42);
    assert_eq!(out1, out2, "win rate replay not deterministic");
}
