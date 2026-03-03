//! Test 07: Evaluation vs Random
//!
//! Evaluates the trained agent against a random opponent.
//! Verifies that the evaluation pipeline works correctly and
//! produces valid win-rate metrics.

use super::cjc_source::*;

fn run_with_seed(extra: &str, seed: u64) -> Vec<String> {
    let src = format!("{CHESS_ENV}\n{RL_AGENT}\n{TRAINING}\n{extra}");
    run_mir(&src, seed)
}

#[test]
fn eval_produces_valid_result() {
    let out = run_with_seed(r#"
let weights = init_weights();
let W1 = weights[0];
let b1 = weights[1];
let W2 = weights[2];
let result = play_episode_random(W1, b1, W2, 30, 1);
print(result);
"#, 42);
    let result = parse_float(&out);
    assert!(
        result == 1.0 || result == -1.0 || result == 0.0,
        "eval result should be -1, 0, or 1, got {result}"
    );
}

#[test]
fn eval_multiple_games_produces_metrics() {
    let out = run_with_seed(r#"
let weights = init_weights();
let W1 = weights[0];
let b1 = weights[1];
let W2 = weights[2];
let wins = 0;
let losses = 0;
let draws = 0;
let game = 0;
while game < 5 {
    let result = play_episode_random(W1, b1, W2, 20, 1);
    if result > 0.5 { wins = wins + 1; }
    if result < -0.5 { losses = losses + 1; }
    if result > -0.5 && result < 0.5 { draws = draws + 1; }
    game = game + 1;
}
print(wins, losses, draws);
"#, 42);
    let parts: Vec<i64> = out[0].split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();
    let total = parts[0] + parts[1] + parts[2];
    assert_eq!(total, 5, "should have 5 total games, got {total}");
}

#[test]
fn untrained_win_rate_reported() {
    // An untrained agent with random weights should have some win rate
    // against random — this tests the full evaluation pipeline
    let out = run_with_seed(r#"
let weights = init_weights();
let W1 = weights[0];
let b1 = weights[1];
let W2 = weights[2];
let total_reward = 0.0;
let n_games = 4;
let game = 0;
while game < n_games {
    let r = play_episode_random(W1, b1, W2, 20, 1);
    total_reward = total_reward + r;
    game = game + 1;
}
let avg_reward = total_reward / float(n_games);
print(avg_reward);
"#, 42);
    let avg = parse_float(&out);
    assert!(avg >= -1.0 && avg <= 1.0, "avg reward should be in [-1, 1], got {avg}");
    assert!(!avg.is_nan(), "avg reward should not be NaN");
}

#[test]
fn eval_as_black_works() {
    let out = run_with_seed(r#"
let weights = init_weights();
let W1 = weights[0];
let b1 = weights[1];
let W2 = weights[2];
let result = play_episode_random(W1, b1, W2, 20, -1);
print(result);
"#, 42);
    let result = parse_float(&out);
    assert!(
        result == 1.0 || result == -1.0 || result == 0.0,
        "eval as black should produce valid result, got {result}"
    );
}

#[test]
fn full_benchmark_pipeline() {
    // End-to-end: init -> train 2 episodes -> evaluate 2 games
    // This is the complete pipeline test
    let out = run_with_seed(r#"
let weights = init_weights();
let W1 = weights[0];
let b1 = weights[1];
let W2 = weights[2];

// Train 2 episodes
let ep = 0;
while ep < 2 {
    let result = train_episode(W1, b1, W2, 0.01, 0.99, 0.0, 15);
    ep = ep + 1;
}

// Evaluate 2 games
let wins = 0;
let game = 0;
while game < 2 {
    let r = play_episode_random(W1, b1, W2, 15, 1);
    if r > 0.5 { wins = wins + 1; }
    game = game + 1;
}
print(wins);
"#, 42);
    let wins = parse_int(&out);
    assert!(wins >= 0 && wins <= 2, "wins should be 0-2, got {wins}");
}
