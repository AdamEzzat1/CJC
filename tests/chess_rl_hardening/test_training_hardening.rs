//! Training pipeline hardening tests.
//!
//! Validates training episodes, loss computation, and multi-episode
//! training stability.

use super::helpers::*;

// ============================================================
// Single training episode
// ============================================================

#[test]
fn train_episode_produces_finite_loss() {
    let src = full_program(r#"
        let w = init_weights();
        let W1 = w[0]; let b1 = w[1]; let W2 = w[2];
        let result = train_episode(W1, b1, W2, 0.001, 0.99, 0.0, 30);
        let loss = result[1];
        if isnan(loss) || isinf(loss) {
            print("BAD");
        } else {
            print("OK");
        }
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(out[0], "OK", "loss should be finite");
}

#[test]
fn train_episode_valid_reward() {
    let src = full_program(r#"
        let w = init_weights();
        let W1 = w[0]; let b1 = w[1]; let W2 = w[2];
        let result = train_episode(W1, b1, W2, 0.001, 0.99, 0.0, 30);
        let reward = result[0];
        if reward >= -1.01 && reward <= 1.01 {
            print("OK");
        } else {
            print("BAD");
        }
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(out[0], "OK");
}

#[test]
fn train_episode_positive_steps() {
    let src = full_program(r#"
        let w = init_weights();
        let W1 = w[0]; let b1 = w[1]; let W2 = w[2];
        let result = train_episode(W1, b1, W2, 0.001, 0.99, 0.0, 30);
        let steps = int(result[2]);
        if steps >= 1 { print("OK"); } else { print("BAD"); }
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(out[0], "OK");
}

// ============================================================
// Evaluation vs random
// ============================================================

#[test]
fn eval_vs_random_valid_result() {
    let src = full_program(r#"
        let w = init_weights();
        let W1 = w[0]; let b1 = w[1]; let W2 = w[2];
        let reward = play_episode_random(W1, b1, W2, 30, 1);
        if reward >= -1.01 && reward <= 1.01 {
            print("OK");
        } else {
            print("BAD");
        }
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(out[0], "OK");
}

#[test]
fn eval_vs_random_as_black() {
    let src = full_program(r#"
        let w = init_weights();
        let W1 = w[0]; let b1 = w[1]; let W2 = w[2];
        let reward = play_episode_random(W1, b1, W2, 30, -1);
        if reward >= -1.01 && reward <= 1.01 {
            print("OK");
        } else {
            print("BAD");
        }
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(out[0], "OK");
}

// ============================================================
// Multi-episode stability
// ============================================================

#[test]
fn two_episodes_no_nan() {
    let src = full_program(r#"
        let w = init_weights();
        let W1 = w[0]; let b1 = w[1]; let W2 = w[2];
        let ok = true;
        let ep = 0;
        while ep < 2 {
            let result = train_episode(W1, b1, W2, 0.001, 0.99, 0.0, 20);
            if isnan(result[1]) || isinf(result[1]) { ok = false; }
            ep = ep + 1;
        }
        print(ok);
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(out[0], "true", "training over 2 episodes should produce no NaN/Inf");
}
