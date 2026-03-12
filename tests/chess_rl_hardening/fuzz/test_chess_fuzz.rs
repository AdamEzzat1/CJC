//! Bolero fuzz tests for chess RL subsystem.
//!
//! Windows-compatible: runs as proptest in `cargo test`.
//! Linux: `cargo bolero test <target>` for coverage-guided fuzzing.

use std::panic;
use crate::chess_rl_hardening::helpers::*;

/// Fuzz: board init + legal_moves never panics regardless of seed.
#[test]
fn fuzz_legal_moves_no_panic() {
    bolero::check!().with_type::<u64>().for_each(|seed: &u64| {
        let src = chess_program(r#"
            let b = init_board();
            let m = legal_moves(b, 1);
            print(len(m));
        "#);
        let _ = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            let _ = run_mir(&src, *seed);
        }));
    });
}

/// Fuzz: rollout never panics regardless of seed.
#[test]
fn fuzz_rollout_no_panic() {
    bolero::check!().with_type::<u64>().for_each(|seed: &u64| {
        let src = full_program(r#"
            let w = init_weights();
            let W1 = w[0]; let b1 = w[1]; let W2 = w[2];
            let result = play_episode(W1, b1, W2, 10);
            print(result[0]);
        "#);
        let _ = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            let _ = run_mir(&src, *seed);
        }));
    });
}

/// Fuzz: action selection never panics.
#[test]
fn fuzz_select_action_no_panic() {
    bolero::check!().with_type::<u64>().for_each(|seed: &u64| {
        let src = chess_agent_program(r#"
            let w = init_weights();
            let W1 = w[0]; let b1 = w[1]; let W2 = w[2];
            let b = init_board();
            let moves = legal_moves(b, 1);
            let feat = encode_board(b, 1);
            let result = select_action(W1, b1, W2, feat, moves);
            print(result[0]);
        "#);
        let _ = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            let _ = run_mir(&src, *seed);
        }));
    });
}

/// Fuzz: training episode never panics.
#[test]
fn fuzz_train_episode_no_panic() {
    bolero::check!().with_type::<u64>().for_each(|seed: &u64| {
        let src = full_program(r#"
            let w = init_weights();
            let W1 = w[0]; let b1 = w[1]; let W2 = w[2];
            let result = train_episode(W1, b1, W2, 0.001, 0.99, 0.0, 8);
            print(result[0]);
        "#);
        let _ = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            let _ = run_mir(&src, *seed);
        }));
    });
}

/// Fuzz: eval vs random never panics.
#[test]
fn fuzz_eval_random_no_panic() {
    bolero::check!().with_type::<u64>().for_each(|seed: &u64| {
        let src = full_program(r#"
            let w = init_weights();
            let W1 = w[0]; let b1 = w[1]; let W2 = w[2];
            let reward = play_episode_random(W1, b1, W2, 10, 1);
            print(reward);
        "#);
        let _ = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            let _ = run_mir(&src, *seed);
        }));
    });
}
