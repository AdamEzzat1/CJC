//! Player profile generation tests.
//!
//! Validates that statistical player profiles can be derived from
//! game traces: move counts, capture timing, piece preferences.

use super::helpers::*;

/// Agent plays a complete game and produces a valid result.
#[test]
fn game_produces_valid_result() {
    let src = full_program(r#"
        let weights = init_weights();
        let result = play_episode(weights[0], weights[1], weights[2], 20);
        let reward = result[0];
        let moves = result[1];
        print(reward);
        print(moves);
    "#);
    let out = run_mir(&src, 42);
    let reward = parse_float_at(&out, 0);
    let moves = parse_float_at(&out, 1);
    assert!(reward >= -1.0 && reward <= 1.0, "reward must be in [-1, 1]");
    assert!(moves >= 1.0 && moves <= 20.0, "moves must be in [1, 20]");
}

/// Capture detection: we can count pieces remaining after a game.
#[test]
fn piece_count_after_game() {
    let src = chess_agent_program(r#"
        let weights = init_weights();
        let board = init_board();
        let side = 1;
        let move_count = 0;
        while move_count < 10 {
            let status = terminal_status(board, side);
            if status != 0 { break; }
            let moves = legal_moves(board, side);
            let features = encode_board(board, side);
            let result = select_action(weights[0], weights[1], weights[2], features, moves);
            let action_idx = int(result[0]);
            let from_sq = moves[action_idx * 2];
            let to_sq = moves[action_idx * 2 + 1];
            board = apply_move(board, from_sq, to_sq);
            side = -1 * side;
            move_count = move_count + 1;
        }
        // Count remaining pieces
        let pieces = 0;
        let i = 0;
        while i < 64 {
            if board[i] != 0 { pieces = pieces + 1; }
            i = i + 1;
        }
        print(pieces);
    "#);
    let out = run_mir(&src, 42);
    let pieces = parse_int_at(&out, 0);
    assert!(pieces > 0 && pieces <= 32, "piece count must be in (0, 32], got {pieces}");
}

/// Profile metrics are deterministic for the same game.
#[test]
fn profile_metrics_deterministic() {
    let src = chess_agent_program(r#"
        let weights = init_weights();
        let board = init_board();
        let side = 1;
        let move_count = 0;
        let captures = 0;
        while move_count < 10 {
            let status = terminal_status(board, side);
            if status != 0 { break; }
            let moves = legal_moves(board, side);
            let features = encode_board(board, side);
            let result = select_action(weights[0], weights[1], weights[2], features, moves);
            let action_idx = int(result[0]);
            let from_sq = moves[action_idx * 2];
            let to_sq = moves[action_idx * 2 + 1];
            if board[to_sq] != 0 { captures = captures + 1; }
            board = apply_move(board, from_sq, to_sq);
            side = -1 * side;
            move_count = move_count + 1;
        }
        print(move_count);
        print(captures);
    "#);
    let out1 = run_mir(&src, 42);
    let out2 = run_mir(&src, 42);
    assert_eq!(out1, out2, "profile metrics should be deterministic");
}

/// Different seeds produce different game trajectories.
#[test]
fn different_seeds_different_trajectories() {
    let src = chess_agent_program(r#"
        let weights = init_weights();
        let board = init_board();
        let side = 1;
        let move_count = 0;
        while move_count < 6 {
            let status = terminal_status(board, side);
            if status != 0 { break; }
            let moves = legal_moves(board, side);
            let features = encode_board(board, side);
            let result = select_action(weights[0], weights[1], weights[2], features, moves);
            let action_idx = int(result[0]);
            let from_sq = moves[action_idx * 2];
            let to_sq = moves[action_idx * 2 + 1];
            print(from_sq);
            print(to_sq);
            board = apply_move(board, from_sq, to_sq);
            side = -1 * side;
            move_count = move_count + 1;
        }
    "#);
    let out1 = run_mir(&src, 42);
    let out2 = run_mir(&src, 99);
    assert_ne!(out1, out2, "different seeds should produce different game trajectories");
}
