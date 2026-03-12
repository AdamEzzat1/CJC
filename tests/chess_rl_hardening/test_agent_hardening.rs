//! RL agent hardening tests.
//!
//! Validates network initialization, forward pass, action selection,
//! and REINFORCE gradient updates.

use super::helpers::*;

// ============================================================
// Weight initialization
// ============================================================

#[test]
fn init_weights_returns_three_tensors() {
    let src = chess_agent_program(r#"
        let w = init_weights();
        print(len(w));
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(out[0], "3", "init_weights should return [W1, b1, W2]");
}

#[test]
fn weight_shapes_correct() {
    let src = chess_agent_program(r#"
        let w = init_weights();
        let W1 = w[0]; let b1 = w[1]; let W2 = w[2];
        let s1 = W1.shape(); let sb = b1.shape(); let s2 = W2.shape();
        print(s1[0]); print(s1[1]);
        print(sb[0]); print(sb[1]);
        print(s2[0]); print(s2[1]);
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(out, vec!["66", "16", "1", "16", "16", "1"]);
}

#[test]
fn weights_are_small() {
    // Initialized with * 0.1, so max should be well under 1.0 typically
    let src = chess_agent_program(r#"
        let w = init_weights();
        let W1 = w[0];
        let max_val = 0.0;
        let i = 0;
        while i < 66 {
            let j = 0;
            while j < 16 {
                let v = W1.get([i, j]);
                if v < 0.0 { v = 0.0 - v; }
                if v > max_val { max_val = v; }
                j = j + 1;
            }
            i = i + 1;
        }
        if max_val < 2.0 { print("OK"); } else { print("BAD"); }
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(out[0], "OK", "weights initialized with *0.1 should be small");
}

// ============================================================
// Forward pass
// ============================================================

#[test]
fn forward_move_returns_single_score() {
    let src = chess_agent_program(r#"
        let w = init_weights();
        let W1 = w[0]; let b1 = w[1]; let W2 = w[2];
        let b = init_board();
        let feat = encode_board(b, 1);
        let result = forward_move(W1, b1, W2, feat, 12, 28);
        print(len(result));
        let score = result[0];
        if isnan(score) { print("NAN"); } else { print("FINITE"); }
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(out[0], "1", "forward_move returns array of length 1");
    assert_eq!(out[1], "FINITE", "score should be finite");
}

#[test]
fn forward_move_different_moves_different_scores() {
    let src = chess_agent_program(r#"
        let w = init_weights();
        let W1 = w[0]; let b1 = w[1]; let W2 = w[2];
        let b = init_board();
        let feat = encode_board(b, 1);
        let s1 = forward_move(W1, b1, W2, feat, 12, 28)[0];
        let s2 = forward_move(W1, b1, W2, feat, 12, 20)[0];
        // Different moves should generally produce different scores
        // (not guaranteed but overwhelmingly likely with random weights)
        if s1 == s2 { print("SAME"); } else { print("DIFFERENT"); }
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(out[0], "DIFFERENT");
}

// ============================================================
// Action selection
// ============================================================

#[test]
fn select_action_returns_valid_index() {
    let src = chess_agent_program(r#"
        let w = init_weights();
        let W1 = w[0]; let b1 = w[1]; let W2 = w[2];
        let b = init_board();
        let moves = legal_moves(b, 1);
        let feat = encode_board(b, 1);
        let result = select_action(W1, b1, W2, feat, moves);
        let action_idx = int(result[0]);
        let num_moves = int(result[2]);
        if action_idx >= 0 && action_idx < num_moves {
            print("VALID");
        } else {
            print("INVALID");
        }
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(out[0], "VALID");
}

#[test]
fn select_action_log_prob_is_negative() {
    // Log probabilities should be <= 0
    let src = chess_agent_program(r#"
        let w = init_weights();
        let W1 = w[0]; let b1 = w[1]; let W2 = w[2];
        let b = init_board();
        let moves = legal_moves(b, 1);
        let feat = encode_board(b, 1);
        let result = select_action(W1, b1, W2, feat, moves);
        let log_prob = result[1];
        if log_prob <= 0.001 { print("OK"); } else { print("BAD"); }
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(out[0], "OK", "log probability should be <= 0");
}

#[test]
fn select_action_num_moves_matches() {
    let src = chess_agent_program(r#"
        let w = init_weights();
        let W1 = w[0]; let b1 = w[1]; let W2 = w[2];
        let b = init_board();
        let moves = legal_moves(b, 1);
        let feat = encode_board(b, 1);
        let result = select_action(W1, b1, W2, feat, moves);
        let reported = int(result[2]);
        let actual = len(moves) / 2;
        print(reported); print(actual);
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(out[0], out[1], "num_moves should match len(moves)/2");
}

// ============================================================
// REINFORCE update
// ============================================================

#[test]
fn reinforce_update_changes_weights() {
    let src = chess_agent_program(r#"
        let w = init_weights();
        let W1 = w[0]; let b1 = w[1]; let W2 = w[2];
        let b = init_board();
        let moves = legal_moves(b, 1);
        let feat = encode_board(b, 1);
        let W1_sum_before = W1.sum();
        let updated = reinforce_update(W1, b1, W2, feat, moves, 0, 1.0, 0.01);
        let new_W1 = updated[0];
        let W1_sum_after = new_W1.sum();
        if W1_sum_before == W1_sum_after {
            print("UNCHANGED");
        } else {
            print("CHANGED");
        }
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(out[0], "CHANGED", "weights should change after gradient update");
}

#[test]
fn reinforce_update_returns_three_tensors() {
    let src = chess_agent_program(r#"
        let w = init_weights();
        let W1 = w[0]; let b1 = w[1]; let W2 = w[2];
        let b = init_board();
        let moves = legal_moves(b, 1);
        let feat = encode_board(b, 1);
        let updated = reinforce_update(W1, b1, W2, feat, moves, 0, 1.0, 0.01);
        print(len(updated));
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(out[0], "3");
}

#[test]
fn reinforce_update_preserves_shapes() {
    let src = chess_agent_program(r#"
        let w = init_weights();
        let W1 = w[0]; let b1 = w[1]; let W2 = w[2];
        let b = init_board();
        let moves = legal_moves(b, 1);
        let feat = encode_board(b, 1);
        let updated = reinforce_update(W1, b1, W2, feat, moves, 0, 1.0, 0.01);
        let nW1 = updated[0]; let nb1 = updated[1]; let nW2 = updated[2];
        let s1 = nW1.shape(); let sb = nb1.shape(); let s2 = nW2.shape();
        print(s1[0]); print(s1[1]);
        print(sb[0]); print(sb[1]);
        print(s2[0]); print(s2[1]);
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(out, vec!["66", "16", "1", "16", "16", "1"]);
}
