//! Test 04: Rollout Determinism
//!
//! Verifies that full game rollouts (episodes) are deterministic:
//! same seed produces identical trajectories.

use super::cjc_source::*;

fn run_with_seed(extra: &str, seed: u64) -> Vec<String> {
    let src = format!("{CHESS_ENV}\n{RL_AGENT}\n{TRAINING}\n{extra}");
    run_mir(&src, seed)
}

#[test]
fn rollout_same_seed_identical() {
    let code = r#"
let weights = init_weights();
let W1 = weights[0];
let b1 = weights[1];
let W2 = weights[2];
let result = play_episode(W1, b1, W2, 20);
print(result[0], result[1]);
"#;
    let out1 = run_with_seed(code, 42);
    let out2 = run_with_seed(code, 42);
    assert_eq!(out1, out2, "same seed must produce identical rollout");
}

#[test]
fn rollout_different_seed_differs() {
    let code = r#"
let weights = init_weights();
let W1 = weights[0];
let b1 = weights[1];
let W2 = weights[2];
let result = play_episode(W1, b1, W2, 20);
print(result[0], result[1]);
"#;
    let out1 = run_with_seed(code, 42);
    let out2 = run_with_seed(code, 99);
    // Different seeds should produce different rollouts (overwhelmingly likely)
    // It's theoretically possible to get the same output but extremely unlikely
    assert_ne!(out1, out2, "different seeds should produce different rollouts");
}

#[test]
fn rollout_produces_valid_game_length() {
    let out = run_with_seed(r#"
let weights = init_weights();
let W1 = weights[0];
let b1 = weights[1];
let W2 = weights[2];
let result = play_episode(W1, b1, W2, 50);
let moves = int(result[1]);
print(moves);
"#, 42);
    let moves = parse_int(&out);
    assert!(moves >= 1 && moves <= 50, "game should last 1-50 moves, got {moves}");
}

#[test]
fn rollout_reward_in_valid_range() {
    let out = run_with_seed(r#"
let weights = init_weights();
let W1 = weights[0];
let b1 = weights[1];
let W2 = weights[2];
let result = play_episode(W1, b1, W2, 30);
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
fn multiple_rollouts_deterministic() {
    let code = r#"
let weights = init_weights();
let W1 = weights[0];
let b1 = weights[1];
let W2 = weights[2];
let results = [];
let ep = 0;
while ep < 3 {
    let result = play_episode(W1, b1, W2, 20);
    results = array_push(results, result[0]);
    results = array_push(results, result[1]);
    ep = ep + 1;
}
let i = 0;
while i < 6 {
    print(results[i]);
    i = i + 1;
}
"#;
    let out1 = run_with_seed(code, 42);
    let out2 = run_with_seed(code, 42);
    assert_eq!(out1, out2, "multiple rollouts with same seed must be identical");
}
