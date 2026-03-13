//! Trace replay determinism tests.
//!
//! Ensures that game traces (move sequences) can be replayed
//! to produce identical board states, and that the trace format
//! is consistent.

use super::helpers::*;

/// Replay: applying the same move sequence reproduces the same board.
#[test]
fn replay_move_sequence_identical() {
    let src = chess_program(r#"
        let board = init_board();
        let board2 = apply_move(board, 12, 28); // e2-e4
        let board3 = apply_move(board2, 52, 36); // e7-e5
        let board4 = apply_move(board3, 6, 21); // Nf3
        // Print final board hash (sum of pieces * position)
        let hash = 0;
        let i = 0;
        while i < 64 {
            hash = hash + board4[i] * (i + 1);
            i = i + 1;
        }
        print(hash);
    "#);
    let out1 = run_mir(&src, 42);
    let out2 = run_mir(&src, 42);
    assert_eq!(out1, out2, "replay should produce identical board state");
}

/// A full agent game with same seed replays identically.
#[test]
fn agent_game_replay_deterministic() {
    let src = full_program(r#"
        let weights = init_weights();
        let result = play_episode(weights[0], weights[1], weights[2], 10);
        print(result[0]);
        print(result[1]);
    "#);
    let out1 = run_mir(&src, 42);
    let out2 = run_mir(&src, 42);
    assert_eq!(out1, out2, "agent game replay should be deterministic");
}

/// Encoding is deterministic for the same board.
#[test]
fn encode_board_deterministic() {
    let src = chess_program(r#"
        let board = init_board();
        let feat = encode_board(board, 1);
        print(feat.get([0, 0]));
        print(feat.get([0, 4]));
        print(feat.get([0, 63]));
    "#);
    let out1 = run_mir(&src, 42);
    let out2 = run_mir(&src, 99);
    assert_eq!(out1, out2, "board encoding should be deterministic (independent of seed)");
}

/// Encoding values are normalized correctly: piece * side / 6.0.
#[test]
fn encode_board_normalization() {
    let src = chess_program(r#"
        let board = init_board();
        let feat = encode_board(board, 1);
        // sq 0 = white rook (4): 4*1/6 = 0.666...
        print(feat.get([0, 0]));
        // sq 4 = white king (6): 6*1/6 = 1.0
        print(feat.get([0, 4]));
        // sq 56 = black rook (-4): -4*1/6 = -0.666...
        print(feat.get([0, 56]));
    "#);
    let out = run_mir(&src, 42);
    let rook = parse_float_at(&out, 0);
    let king = parse_float_at(&out, 1);
    let brk = parse_float_at(&out, 2);
    assert!((rook - 4.0 / 6.0).abs() < 1e-10, "white rook encoding");
    assert!((king - 1.0).abs() < 1e-10, "white king encoding");
    assert!((brk - (-4.0 / 6.0)).abs() < 1e-10, "black rook encoding");
}

/// Move sequence applied in order produces consistent incremental state.
#[test]
fn incremental_apply_move_consistency() {
    let src = chess_program(r#"
        let board = init_board();
        // Apply 3 moves and verify each board is 64 elements
        let b1 = apply_move(board, 12, 28);
        print(len(b1));
        let b2 = apply_move(b1, 52, 36);
        print(len(b2));
        let b3 = apply_move(b2, 6, 21);
        print(len(b3));
        // Verify pieces are where expected
        print(b3[28]); // e4: white pawn
        print(b3[36]); // e5: black pawn
        print(b3[21]); // f3: white knight
        print(b3[6]);  // g1: empty (knight moved)
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(parse_int_at(&out, 0), 64);
    assert_eq!(parse_int_at(&out, 1), 64);
    assert_eq!(parse_int_at(&out, 2), 64);
    assert_eq!(parse_int_at(&out, 3), 1, "e4 should have white pawn");
    assert_eq!(parse_int_at(&out, 4), -1, "e5 should have black pawn");
    assert_eq!(parse_int_at(&out, 5), 2, "f3 should have white knight");
    assert_eq!(parse_int_at(&out, 6), 0, "g1 should be empty");
}

/// Board is 64 elements after initialization.
#[test]
fn board_size_always_64() {
    let src = chess_program(r#"
        let board = init_board();
        print(len(board));
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(parse_int_at(&out, 0), 64);
}
