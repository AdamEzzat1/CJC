//! Phase 2: Multi-episode training tests.
//!
//! Tests that multi-episode training runs correctly, accumulates metrics,
//! and shows weight propagation between episodes.

use super::helpers::*;

/// Multi-episode training produces output for each episode.
#[test]
fn multi_train_3_episodes_produces_output() {
    let src = multi_program(r#"
        let result = train_multi_episodes(3, 0.01, 0.99, 0.0, 8);
        print("done");
    "#);
    let out = run_mir(&src, 42);
    // 3 episodes * 3 lines (reward, loss, steps) + "done" = 10 lines
    assert!(out.len() >= 10, "expected at least 10 output lines, got {}", out.len());
    assert_eq!(out.last().unwrap(), "done");
}

/// Per-episode rewards are valid floats in [-1, 1].
#[test]
fn multi_train_rewards_valid() {
    let src = multi_program(r#"
        let result = train_multi_episodes(5, 0.01, 0.99, 0.0, 8);
    "#);
    let out = run_mir(&src, 100);
    for ep in 0..5 {
        let reward = parse_float_at(&out, ep * 3);
        assert!(reward >= -1.0 && reward <= 1.0,
            "episode {ep} reward {reward} out of bounds");
    }
}

/// Per-episode step counts are positive.
#[test]
fn multi_train_steps_positive() {
    let src = multi_program(r#"
        let result = train_multi_episodes(3, 0.01, 0.99, 0.0, 10);
    "#);
    let out = run_mir(&src, 77);
    for ep in 0..3 {
        let steps = parse_float_at(&out, ep * 3 + 2);
        assert!(steps >= 1.0, "episode {ep} steps {steps} should be >= 1");
    }
}

/// Per-episode losses are finite.
#[test]
fn multi_train_losses_finite() {
    let src = multi_program(r#"
        let result = train_multi_episodes(3, 0.01, 0.99, 0.0, 8);
    "#);
    let out = run_mir(&src, 55);
    for ep in 0..3 {
        let loss = parse_float_at(&out, ep * 3 + 1);
        assert!(loss.is_finite(), "episode {ep} loss {loss} is not finite");
    }
}

/// Multi-episode training is deterministic: same seed = identical output.
#[test]
fn multi_train_deterministic() {
    let src = multi_program(r#"
        let result = train_multi_episodes(3, 0.01, 0.99, 0.0, 8);
    "#);
    let out1 = run_mir(&src, 42);
    let out2 = run_mir(&src, 42);
    assert_eq!(out1, out2, "multi-episode training not deterministic");
}

/// Different seeds produce different weight initializations.
#[test]
fn multi_train_different_seeds_differ() {
    // With short max_moves, games often draw (reward=0, loss=0).
    // Verify that different seeds produce different initial weights.
    let src = multi_program(r#"
        let weights = init_weights();
        print(weights[0].get([0, 0]));
        print(weights[0].get([1, 0]));
    "#);
    let out1 = run_mir(&src, 42);
    let out2 = run_mir(&src, 99);
    assert_ne!(out1, out2, "different seeds should produce different weights");
}

/// train_episode_returning_weights returns reward, loss, steps, and weights.
#[test]
fn train_episode_returns_weights() {
    let src = multi_program(r#"
        let weights = init_weights();
        let W1 = weights[0];
        let b1 = weights[1];
        let W2 = weights[2];
        let result = train_episode_returning_weights(W1, b1, W2, 0.01, 0.99, 0.0, 8);
        print(result[0]);
        print(result[1]);
        print(result[2]);
        print("has_weights");
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(out.last().unwrap(), "has_weights");
    let reward = parse_float_at(&out, 0);
    assert!(reward >= -1.0 && reward <= 1.0);
}

/// Weights change after training (not stuck at initialization).
#[test]
fn multi_train_weights_change() {
    let src = multi_program(r#"
        let weights = init_weights();
        let W1_init = weights[0];
        let result = train_multi_episodes(3, 0.01, 0.99, 0.0, 8);
        let W1_trained = result[0];
        let init_val = W1_init.get([0, 0]);
        let trained_val = W1_trained.get([0, 0]);
        if init_val == trained_val {
            print("same");
        } else {
            print("changed");
        }
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(out.last().unwrap(), "changed",
        "weights should change after training");
}

/// 5 episodes training produces monotonically increasing episode count.
#[test]
fn multi_train_5_episodes_metric_count() {
    let src = multi_program(r#"
        let result = train_multi_episodes(5, 0.01, 0.99, 0.0, 8);
    "#);
    let out = run_mir(&src, 123);
    // 5 episodes * 3 lines = 15 output lines
    assert_eq!(out.len(), 15, "expected 15 output lines for 5 episodes");
}
