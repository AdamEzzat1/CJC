//! Phase 6: MIR executor trace tests.
//!
//! Tests that MIR execution can be instrumented to produce trace events.
//! Uses the existing @trace decorator and print-based tracing.

use super::helpers::*;

/// A simple CJC program prints deterministic output (trace baseline).
#[test]
fn trace_baseline_output() {
    let src = chess_program(r#"
        let b = init_board();
        print(b[0]);
        print(b[4]);
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(out.len(), 2);
    assert_eq!(parse_int_at(&out, 0), 4); // white rook at a1
    assert_eq!(parse_int_at(&out, 1), 6); // white king at e1
}

/// Trace: legal moves from initial position are deterministic.
#[test]
fn trace_legal_moves_initial() {
    let src = chess_program(r#"
        let b = init_board();
        let moves = legal_moves(b, 1);
        print(len(moves));
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(parse_int(&out), 40); // 20 moves * 2 (from/to pairs)
}

/// Trace: board state after one move.
#[test]
fn trace_board_after_move() {
    let src = chess_program(r#"
        let b = init_board();
        let moves = legal_moves(b, 1);
        let b2 = apply_move(b, moves[0], moves[1]);
        // Source square should be empty
        print(b2[moves[0]]);
        // Target square should have the piece
        print(b[moves[0]]);
        print(b2[moves[1]]);
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(parse_int_at(&out, 0), 0, "source should be empty after move");
    let moved_piece = parse_int_at(&out, 1);
    let target_piece = parse_int_at(&out, 2);
    assert_eq!(moved_piece, target_piece, "piece should appear at target");
}

/// Trace: action selection produces valid action index.
#[test]
fn trace_action_selection() {
    let src = chess_agent_program(r#"
        let weights = init_weights();
        let b = init_board();
        let moves = legal_moves(b, 1);
        let features = encode_board(b, 1);
        let result = select_action(weights[0], weights[1], weights[2], features, moves);
        let action_idx = int(result[0]);
        let num_moves = int(result[2]);
        print(action_idx);
        print(num_moves);
    "#);
    let out = run_mir(&src, 42);
    let action_idx = parse_int_at(&out, 0);
    let num_moves = parse_int_at(&out, 1);
    assert!(action_idx >= 0 && action_idx < num_moves,
        "action_idx {action_idx} out of range [0, {num_moves})");
}

/// Trace: forward pass score is finite.
#[test]
fn trace_forward_pass_finite() {
    let src = chess_agent_program(r#"
        let weights = init_weights();
        let b = init_board();
        let features = encode_board(b, 1);
        let result = forward_move(weights[0], weights[1], weights[2], features, 8, 16);
        print(result[0]);
    "#);
    let out = run_mir(&src, 42);
    let score = parse_float(&out);
    assert!(score.is_finite(), "forward pass score should be finite: {score}");
}

/// Trace: full game trajectory step by step.
#[test]
fn trace_game_trajectory() {
    let src = chess_agent_program(r#"
        let weights = init_weights();
        let b = init_board();
        let side = 1;
        let move_count = 0;
        while move_count < 4 {
            let status = terminal_status(b, side);
            if status != 0 {
                break;
            }
            let moves = legal_moves(b, side);
            let features = encode_board(b, side);
            let result = select_action(weights[0], weights[1], weights[2], features, moves);
            let action_idx = int(result[0]);
            let from_sq = moves[action_idx * 2];
            let to_sq = moves[action_idx * 2 + 1];
            print(from_sq);
            print(to_sq);
            b = apply_move(b, from_sq, to_sq);
            side = -1 * side;
            move_count = move_count + 1;
        }
        print(move_count);
    "#);
    let out = run_mir(&src, 42);
    let move_count = parse_int_at(&out, out.len() - 1);
    assert!(move_count >= 1 && move_count <= 4);
    // Each move produces 2 output lines (from, to)
    assert_eq!(out.len(), (move_count as usize) * 2 + 1);
}

/// Trace output is deterministic across runs.
#[test]
fn trace_deterministic() {
    let src = chess_agent_program(r#"
        let weights = init_weights();
        let b = init_board();
        let side = 1;
        let move_count = 0;
        while move_count < 3 {
            let moves = legal_moves(b, side);
            let features = encode_board(b, side);
            let result = select_action(weights[0], weights[1], weights[2], features, moves);
            let action_idx = int(result[0]);
            print(action_idx);
            let from_sq = moves[action_idx * 2];
            let to_sq = moves[action_idx * 2 + 1];
            b = apply_move(b, from_sq, to_sq);
            side = -1 * side;
            move_count = move_count + 1;
        }
    "#);
    let out1 = run_mir(&src, 42);
    let out2 = run_mir(&src, 42);
    assert_eq!(out1, out2, "trace output not deterministic");
}
