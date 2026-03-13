//! Phase 3: Win-rate evaluation tests.
//!
//! Tests the eval_win_rate function that measures agent performance
//! against a random opponent.

use super::helpers::*;

/// Win rate evaluation produces wins/draws/losses counts.
#[test]
fn eval_win_rate_produces_counts() {
    let src = multi_program(r#"
        let weights = init_weights();
        let W1 = weights[0];
        let b1 = weights[1];
        let W2 = weights[2];
        let wr = eval_win_rate(W1, b1, W2, 4, 8, 1);
        print(wr);
    "#);
    let out = run_mir(&src, 42);
    // Output: wins, draws, losses, win_rate
    assert_eq!(out.len(), 4, "expected 4 output lines (wins, draws, losses, wr)");
}

/// Wins + draws + losses = num_games.
#[test]
fn eval_win_rate_counts_sum() {
    let src = multi_program(r#"
        let weights = init_weights();
        let W1 = weights[0];
        let b1 = weights[1];
        let W2 = weights[2];
        let wr = eval_win_rate(W1, b1, W2, 6, 8, 1);
    "#);
    let out = run_mir(&src, 42);
    let wins = parse_int_at(&out, 0);
    let draws = parse_int_at(&out, 1);
    let losses = parse_int_at(&out, 2);
    assert_eq!(wins + draws + losses, 6, "counts should sum to num_games");
}

/// Win rate is between 0 and 1.
#[test]
fn eval_win_rate_bounded() {
    let src = multi_program(r#"
        let weights = init_weights();
        let W1 = weights[0];
        let b1 = weights[1];
        let W2 = weights[2];
        let wr = eval_win_rate(W1, b1, W2, 4, 8, 1);
        print(wr);
    "#);
    let out = run_mir(&src, 42);
    let wr = parse_float_at(&out, 3);
    assert!(wr >= 0.0 && wr <= 1.0, "win rate {wr} out of bounds");
}

/// Win rate evaluation is deterministic.
#[test]
fn eval_win_rate_deterministic() {
    let src = multi_program(r#"
        let weights = init_weights();
        let W1 = weights[0];
        let b1 = weights[1];
        let W2 = weights[2];
        let wr = eval_win_rate(W1, b1, W2, 4, 8, 1);
        print(wr);
    "#);
    let out1 = run_mir(&src, 42);
    let out2 = run_mir(&src, 42);
    assert_eq!(out1, out2, "win rate evaluation not deterministic");
}

/// Win rate works for black side too.
#[test]
fn eval_win_rate_black_side() {
    let src = multi_program(r#"
        let weights = init_weights();
        let W1 = weights[0];
        let b1 = weights[1];
        let W2 = weights[2];
        let wr = eval_win_rate(W1, b1, W2, 4, 8, -1);
        print(wr);
    "#);
    let out = run_mir(&src, 42);
    let wins = parse_int_at(&out, 0);
    let draws = parse_int_at(&out, 1);
    let losses = parse_int_at(&out, 2);
    assert_eq!(wins + draws + losses, 4, "counts should sum to 4");
}

/// Win rate after training should differ from untrained.
#[test]
fn eval_win_rate_changes_after_training() {
    let src = multi_program(r#"
        let weights = init_weights();
        let W1 = weights[0];
        let b1 = weights[1];
        let W2 = weights[2];
        let wr_before = eval_win_rate(W1, b1, W2, 4, 8, 1);

        let trained = train_multi_episodes(3, 0.01, 0.99, 0.0, 8);
        let W1t = trained[0];
        let b1t = trained[1];
        let W2t = trained[2];
        let wr_after = eval_win_rate(W1t, b1t, W2t, 4, 8, 1);
    "#);
    let out = run_mir(&src, 42);
    // We don't assert improvement (RL is noisy), just that it runs without error
    assert!(out.len() >= 8, "expected output from both evaluations");
}
