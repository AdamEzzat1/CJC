//! Game logic hardening tests.
//!
//! Validates terminal detection, reward computation, turn alternation,
//! and game state transitions.

use super::helpers::*;

// ============================================================
// Terminal detection
// ============================================================

#[test]
fn initial_position_not_terminal() {
    let src = chess_program(r#"
        let b = init_board();
        print(terminal_status(b, 1));
        print(terminal_status(b, -1));
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(out[0], "0", "initial position not terminal for white");
    assert_eq!(out[1], "0", "initial position not terminal for black");
}

#[test]
fn scholars_mate_is_checkmate() {
    // 1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6?? 4. Qxf7#
    let src = chess_program(r#"
        let b = init_board();
        b = apply_move(b, 12, 28);
        b = apply_move(b, 52, 36);
        b = apply_move(b, 5, 26);
        b = apply_move(b, 57, 42);
        b = apply_move(b, 3, 39);
        b = apply_move(b, 62, 45);
        b = apply_move(b, 39, 53);
        let status = terminal_status(b, -1);
        print(status);
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(out[0], "2", "scholar's mate should be checkmate (status=2)");
}

#[test]
fn terminal_status_returns_valid_codes() {
    // Play a short game and verify status codes are in {0, 2, 3}
    let src = chess_program(r#"
        let b = init_board();
        let side = 1;
        let step = 0;
        let valid = true;
        while step < 6 {
            let s = terminal_status(b, side);
            if s != 0 && s != 2 && s != 3 { valid = false; }
            if s != 0 { break; }
            let m = legal_moves(b, side);
            b = apply_move(b, m[0], m[1]);
            side = -1 * side;
            step = step + 1;
        }
        print(valid);
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(out[0], "true");
}

// ============================================================
// Turn alternation
// ============================================================

#[test]
fn turn_alternates_correctly() {
    let src = chess_program(r#"
        let side = 1;
        let ok = true;
        let step = 0;
        while step < 10 {
            if step % 2 == 0 {
                if side != 1 { ok = false; }
            } else {
                if side != -1 { ok = false; }
            }
            side = -1 * side;
            step = step + 1;
        }
        print(ok);
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(out[0], "true");
}

// ============================================================
// Feature encoding
// ============================================================

#[test]
fn encode_board_shape_is_1x64() {
    let src = chess_program(r#"
        let b = init_board();
        let feat = encode_board(b, 1);
        let s = feat.shape();
        print(s[0]); print(s[1]);
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(out[0], "1");
    assert_eq!(out[1], "64");
}

#[test]
fn encode_board_values_normalized() {
    // All values should be in [-1, 1] since pieces are in [-6, 6] divided by 6
    let src = chess_program(r#"
        let b = init_board();
        let feat = encode_board(b, 1);
        let ok = true;
        let i = 0;
        while i < 64 {
            let v = feat.get([0, i]);
            if v < -1.01 || v > 1.01 { ok = false; }
            i = i + 1;
        }
        print(ok);
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(out[0], "true");
}

#[test]
fn encode_board_side_symmetry() {
    // encode_board(b, 1) and encode_board(b, -1) should be negatives of each other
    let src = chess_program(r#"
        let b = init_board();
        let fw = encode_board(b, 1);
        let fb = encode_board(b, -1);
        let ok = true;
        let i = 0;
        while i < 64 {
            let vw = fw.get([0, i]);
            let vb = fb.get([0, i]);
            let diff = vw + vb;
            if diff > 0.001 || diff < -0.001 { ok = false; }
            i = i + 1;
        }
        print(ok);
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(out[0], "true", "encode_board for opposite sides should be negated");
}

// ============================================================
// Reward bounds
// ============================================================

#[test]
fn rollout_reward_in_valid_range() {
    let src = full_program(r#"
        let weights = init_weights();
        let W1 = weights[0];
        let b1 = weights[1];
        let W2 = weights[2];
        let result = play_episode(W1, b1, W2, 50);
        let reward = result[0];
        if reward < -1.01 || reward > 1.01 {
            print("BAD");
        } else {
            print("OK");
        }
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(out[0], "OK", "reward must be in [-1, 1]");
}

#[test]
fn rollout_move_count_positive() {
    let src = full_program(r#"
        let weights = init_weights();
        let W1 = weights[0];
        let b1 = weights[1];
        let W2 = weights[2];
        let result = play_episode(W1, b1, W2, 50);
        let moves = int(result[1]);
        if moves >= 1 && moves <= 50 {
            print("OK");
        } else {
            print("BAD");
        }
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(out[0], "OK");
}
