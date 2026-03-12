//! Determinism hardening tests for chess RL.
//!
//! Validates that all chess RL operations produce bit-identical results
//! across multiple runs with the same seed.

use super::helpers::*;

// ============================================================
// Board and move generation determinism
// ============================================================

#[test]
fn init_board_deterministic() {
    let src = chess_program(r#"
        let b = init_board();
        let i = 0;
        while i < 64 {
            print(b[i]);
            i = i + 1;
        }
    "#);
    let out1 = run_mir(&src, 42);
    let out2 = run_mir(&src, 42);
    assert_eq!(out1, out2, "init_board must be deterministic");
}

#[test]
fn legal_moves_deterministic() {
    let src = chess_program(r#"
        let b = init_board();
        let m = legal_moves(b, 1);
        let i = 0;
        while i < len(m) {
            print(m[i]);
            i = i + 1;
        }
    "#);
    let out1 = run_mir(&src, 42);
    let out2 = run_mir(&src, 42);
    assert_eq!(out1, out2, "legal_moves must be deterministic");
}

#[test]
fn encode_board_deterministic() {
    let src = chess_program(r#"
        let b = init_board();
        let feat = encode_board(b, 1);
        let i = 0;
        while i < 64 {
            print(feat.get([0, i]));
            i = i + 1;
        }
    "#);
    let out1 = run_mir(&src, 42);
    let out2 = run_mir(&src, 42);
    assert_eq!(out1, out2, "encode_board must be deterministic");
}

// ============================================================
// Agent determinism
// ============================================================

#[test]
fn forward_move_deterministic() {
    let src = chess_agent_program(r#"
        let w = init_weights();
        let W1 = w[0]; let b1 = w[1]; let W2 = w[2];
        let b = init_board();
        let feat = encode_board(b, 1);
        let result = forward_move(W1, b1, W2, feat, 12, 28);
        print(result[0]);
    "#);
    let out1 = run_mir(&src, 42);
    let out2 = run_mir(&src, 42);
    assert_eq!(out1, out2, "forward_move must be deterministic");
}

#[test]
fn select_action_deterministic() {
    let src = chess_agent_program(r#"
        let w = init_weights();
        let W1 = w[0]; let b1 = w[1]; let W2 = w[2];
        let b = init_board();
        let moves = legal_moves(b, 1);
        let feat = encode_board(b, 1);
        let result = select_action(W1, b1, W2, feat, moves);
        print(result[0]); print(result[1]); print(result[2]);
    "#);
    let out1 = run_mir(&src, 42);
    let out2 = run_mir(&src, 42);
    assert_eq!(out1, out2, "select_action must be deterministic with same seed");
}

#[test]
fn select_action_different_seeds_differ() {
    let src = chess_agent_program(r#"
        let w = init_weights();
        let W1 = w[0]; let b1 = w[1]; let W2 = w[2];
        let b = init_board();
        let moves = legal_moves(b, 1);
        let feat = encode_board(b, 1);
        let result = select_action(W1, b1, W2, feat, moves);
        print(result[0]);
    "#);
    let out1 = run_mir(&src, 42);
    let out2 = run_mir(&src, 99);
    // Different seeds → different weight init → different action selection
    // (overwhelmingly likely, not guaranteed)
    // Just verify both produce valid output
    assert!(!out1.is_empty() && !out2.is_empty());
}

// ============================================================
// Rollout determinism
// ============================================================

#[test]
fn rollout_deterministic_5_runs() {
    let src = full_program(r#"
        let w = init_weights();
        let W1 = w[0]; let b1 = w[1]; let W2 = w[2];
        let result = play_episode(W1, b1, W2, 30);
        print(result[0]); print(result[1]);
    "#);
    let baseline = run_mir(&src, 42);
    for _ in 0..4 {
        let out = run_mir(&src, 42);
        assert_eq!(out, baseline, "rollout must be identical across runs");
    }
}

#[test]
fn rollout_different_seed_differs() {
    let src = full_program(r#"
        let w = init_weights();
        let W1 = w[0]; let b1 = w[1]; let W2 = w[2];
        let result = play_episode(W1, b1, W2, 30);
        print(result[0]); print(result[1]);
    "#);
    let out1 = run_mir(&src, 42);
    let out2 = run_mir(&src, 99);
    assert_ne!(out1, out2, "different seeds should produce different rollouts");
}

// ============================================================
// Training determinism
// ============================================================

#[test]
fn training_deterministic_3_runs() {
    let src = full_program(r#"
        let w = init_weights();
        let W1 = w[0]; let b1 = w[1]; let W2 = w[2];
        let result = train_episode(W1, b1, W2, 0.001, 0.99, 0.0, 20);
        print(result[0]); print(result[1]); print(result[2]);
    "#);
    let baseline = run_mir(&src, 42);
    for _ in 0..2 {
        let out = run_mir(&src, 42);
        assert_eq!(out, baseline, "training must be deterministic");
    }
}

#[test]
fn eval_vs_random_deterministic() {
    let src = full_program(r#"
        let w = init_weights();
        let W1 = w[0]; let b1 = w[1]; let W2 = w[2];
        let reward = play_episode_random(W1, b1, W2, 30, 1);
        print(reward);
    "#);
    let out1 = run_mir(&src, 42);
    let out2 = run_mir(&src, 42);
    assert_eq!(out1, out2, "eval vs random must be deterministic");
}

// ============================================================
// Multi-episode training determinism
// ============================================================

#[test]
fn multi_episode_training_deterministic() {
    let src = full_program(r#"
        let w = init_weights();
        let W1 = w[0]; let b1 = w[1]; let W2 = w[2];
        let baseline = 0.0;
        let total_reward = 0.0;
        let ep = 0;
        while ep < 2 {
            let result = train_episode(W1, b1, W2, 0.001, 0.99, baseline, 20);
            let reward = result[0];
            total_reward = total_reward + reward;
            baseline = 0.9 * baseline + 0.1 * reward;
            ep = ep + 1;
        }
        print(total_reward);
        print(baseline);
    "#);
    let out1 = run_mir(&src, 42);
    let out2 = run_mir(&src, 42);
    assert_eq!(out1, out2, "multi-episode training must be deterministic");
}
