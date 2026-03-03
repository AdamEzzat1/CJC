//! Test 06: Training Determinism
//!
//! Verifies that training is fully deterministic: same seed + same config
//! produces bit-identical results.

use super::cjc_source::*;

fn run_with_seed(extra: &str, seed: u64) -> Vec<String> {
    let src = format!("{CHESS_ENV}\n{RL_AGENT}\n{TRAINING}\n{extra}");
    run_mir(&src, seed)
}

#[test]
fn single_episode_training_deterministic() {
    let code = r#"
let weights = init_weights();
let W1 = weights[0];
let b1 = weights[1];
let W2 = weights[2];
let result = train_episode(W1, b1, W2, 0.01, 0.99, 0.0, 20);
print(result[0], result[1], result[2]);
"#;
    let out1 = run_with_seed(code, 42);
    let out2 = run_with_seed(code, 42);
    assert_eq!(out1, out2, "single training episode must be deterministic");
}

#[test]
fn multi_episode_training_deterministic() {
    let code = r#"
let weights = init_weights();
let W1 = weights[0];
let b1 = weights[1];
let W2 = weights[2];
let lr = 0.01;
let baseline = 0.0;
let ep = 0;
let total_reward = 0.0;
while ep < 3 {
    let result = train_episode(W1, b1, W2, lr, 0.99, baseline, 15);
    let reward = result[0];
    total_reward = total_reward + reward;
    // Update baseline (exponential moving average)
    baseline = 0.9 * baseline + 0.1 * reward;
    ep = ep + 1;
}
print(total_reward, baseline);
"#;
    let out1 = run_with_seed(code, 42);
    let out2 = run_with_seed(code, 42);
    assert_eq!(out1, out2, "multi-episode training must be deterministic");
}

#[test]
fn different_seeds_produce_different_training() {
    let code = r#"
let weights = init_weights();
let W1 = weights[0];
let b1 = weights[1];
let W2 = weights[2];
let result = train_episode(W1, b1, W2, 0.01, 0.99, 0.0, 20);
print(result[0], result[1], result[2]);
"#;
    let out1 = run_with_seed(code, 42);
    let out2 = run_with_seed(code, 99);
    // Different seeds will produce different random weights and different
    // categorical samples, leading to different results
    assert_ne!(out1, out2, "different seeds should produce different training");
}

#[test]
fn eval_vs_random_deterministic() {
    let code = r#"
let weights = init_weights();
let W1 = weights[0];
let b1 = weights[1];
let W2 = weights[2];
let result = play_episode_random(W1, b1, W2, 20, 1);
print(result);
"#;
    let out1 = run_with_seed(code, 42);
    let out2 = run_with_seed(code, 42);
    assert_eq!(out1, out2, "eval vs random must be deterministic");
}
