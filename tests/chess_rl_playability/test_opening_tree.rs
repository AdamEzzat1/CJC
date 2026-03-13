//! Opening tree construction tests.
//!
//! Validates that opening move sequences can be tracked and
//! that the move-prefix tree produces consistent statistics.

use super::helpers::*;

/// First move generation produces consistent set across invocations.
#[test]
fn first_move_set_stable() {
    let src = chess_program(r#"
        let board = init_board();
        let moves = legal_moves(board, 1);
        let i = 0;
        while i < len(moves) {
            print(moves[i]);
            i = i + 1;
        }
    "#);
    let out1 = run_mir(&src, 1);
    let out2 = run_mir(&src, 2);
    assert_eq!(out1, out2, "first move set should be stable across seeds");
}

/// After 1.e4, black's first move set is deterministic.
#[test]
fn black_response_set_stable() {
    let src = chess_program(r#"
        let board = init_board();
        let board2 = apply_move(board, 12, 28);
        let moves = legal_moves(board2, -1);
        print(len(moves) / 2);
        // Print first 4 moves as indicators
        let i = 0;
        while i < 8 && i < len(moves) {
            print(moves[i]);
            i = i + 1;
        }
    "#);
    let out1 = run_mir(&src, 1);
    let out2 = run_mir(&src, 2);
    assert_eq!(out1, out2, "black response set after 1.e4 should be stable");
}

/// Opening sequence can be replayed to consistent board state.
#[test]
fn opening_sequence_replay() {
    // Italian Game: 1.e4 e5 2.Nf3 Nc6 3.Bc4
    let src = chess_program(r#"
        let board = init_board();
        let board = apply_move(board, 12, 28); // e2-e4
        let board = apply_move(board, 52, 36); // e7-e5
        let board = apply_move(board, 6, 21);  // Ng1-f3
        let board = apply_move(board, 57, 42); // Nb8-c6
        let board = apply_move(board, 5, 26);  // Bf1-c4
        // Verify bishop on c4 (sq 26)
        print(board[26]);
        // Verify the position is not terminal
        print(terminal_status(board, -1));
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(parse_int_at(&out, 0), 3, "bishop should be on c4");
    assert_eq!(parse_int_at(&out, 1), 0, "position should not be terminal");
}

/// Move generation order is consistent (sq 0..63 enumeration).
#[test]
fn move_enumeration_order() {
    let src = chess_program(r#"
        let board = init_board();
        let moves = legal_moves(board, 1);
        // First move should come from lowest square with a white piece
        // sq 1 = white knight (b1)
        print(moves[0]);
    "#);
    let out = run_mir(&src, 42);
    let first_from = parse_int_at(&out, 0);
    assert_eq!(first_from, 1, "first legal move should be from sq 1 (b1 knight)");
}

/// After a capture, the target square has the capturing piece.
#[test]
fn capture_replaces_piece() {
    let src = chess_program(r#"
        let board = init_board();
        // Set up a position where white can capture: put black pawn at d3
        let custom = [];
        let i = 0;
        while i < 64 {
            if i == 19 { custom = array_push(custom, -1); }       // black pawn at d3
            else { custom = array_push(custom, board[i]); }
            i = i + 1;
        }
        // White pawn e2 (12) captures black pawn d3 (19)
        let after = apply_move(custom, 12, 19);
        print(after[19]); // should be 1 (white pawn)
        print(after[12]); // should be 0 (empty)
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(parse_int_at(&out, 0), 1, "captured square should have capturing piece");
    assert_eq!(parse_int_at(&out, 1), 0, "source square should be empty after capture");
}
