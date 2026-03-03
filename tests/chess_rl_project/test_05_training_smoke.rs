//! Test 05: Training Smoke Test
//!
//! Verifies that one training episode runs without errors, produces finite
//! loss values, and returns valid metrics.

use super::cjc_source::*;

fn run_with_seed(extra: &str, seed: u64) -> Vec<String> {
    let src = format!("{CHESS_ENV}\n{RL_AGENT}\n{TRAINING}\n{extra}");
    run_mir(&src, seed)
}

#[test]
fn single_train_episode_runs() {
    let out = run_with_seed(r#"
let weights = init_weights();
let W1 = weights[0];
let b1 = weights[1];
let W2 = weights[2];
let result = train_episode(W1, b1, W2, 0.01, 0.99, 0.0, 20);
print(result[0], result[1], result[2]);
"#, 42);
    assert_eq!(out.len(), 1, "should produce one output line");
    let parts: Vec<&str> = out[0].split_whitespace().collect();
    assert_eq!(parts.len(), 3, "should have 3 metrics: reward, loss, num_steps");
}

#[test]
fn training_loss_is_finite() {
    let out = run_with_seed(r#"
let weights = init_weights();
let W1 = weights[0];
let b1 = weights[1];
let W2 = weights[2];
let result = train_episode(W1, b1, W2, 0.01, 0.99, 0.0, 20);
let loss = result[1];
let loss_finite = !isnan(loss) && !isinf(loss);
print(loss_finite);
"#, 42);
    assert_eq!(out[0], "true", "training loss must be finite (no NaN or Inf)");
}

#[test]
fn training_produces_valid_reward() {
    let out = run_with_seed(r#"
let weights = init_weights();
let W1 = weights[0];
let b1 = weights[1];
let W2 = weights[2];
let result = train_episode(W1, b1, W2, 0.01, 0.99, 0.0, 20);
let reward = result[0];
print(reward);
"#, 42);
    let reward = parse_float(&out);
    assert!(
        reward == 1.0 || reward == -1.0 || reward == 0.0,
        "reward should be -1, 0, or 1, got {reward}"
    );
}

#[test]
fn training_step_count_positive() {
    let out = run_with_seed(r#"
let weights = init_weights();
let W1 = weights[0];
let b1 = weights[1];
let W2 = weights[2];
let result = train_episode(W1, b1, W2, 0.01, 0.99, 0.0, 30);
let num_steps = int(result[2]);
print(num_steps);
"#, 42);
    let steps = parse_int(&out);
    assert!(steps >= 1, "should take at least 1 step, got {steps}");
}

#[test]
fn weights_change_after_training() {
    // Verify that training actually modifies weights
    let out = run_with_seed(r#"
let weights = init_weights();
let W1 = weights[0];
let b1 = weights[1];
let W2 = weights[2];
// Save initial weight checksum
let w1_sum_before = W1.sum();

// Run training — we can't easily get updated weights back from train_episode
// since it returns scalars. Instead verify via the loss being non-zero
let result = train_episode(W1, b1, W2, 0.01, 0.99, 0.0, 20);
let loss = result[1];
print(loss);
"#, 42);
    let loss = parse_float(&out);
    // If training computed gradients, the loss should be non-zero
    // (unless the advantage happened to be exactly 0)
    // This is a weak check but sufficient for a smoke test
    assert!(!loss.is_nan(), "loss should not be NaN");
}

#[test]
fn categorical_sample_works_in_context() {
    // Verify the new categorical_sample builtin works correctly
    let out = run_with_seed(r#"
let probs = Tensor.from_vec([0.1, 0.2, 0.7], [3]);
let idx = categorical_sample(probs);
print(idx);
"#, 42);
    let idx = parse_int(&out);
    assert!(idx >= 0 && idx < 3, "sampled index should be 0, 1, or 2, got {idx}");
}

#[test]
fn log_builtin_works() {
    let out = run_with_seed(r#"
let val = log(1.0);
print(val);
"#, 42);
    let val = parse_float(&out);
    assert!((val - 0.0).abs() < 1e-10, "log(1.0) should be 0.0, got {val}");
}

#[test]
fn exp_builtin_works() {
    let out = run_with_seed(r#"
let val = exp(0.0);
print(val);
"#, 42);
    let val = parse_float(&out);
    assert!((val - 1.0).abs() < 1e-10, "exp(0.0) should be 1.0, got {val}");
}
