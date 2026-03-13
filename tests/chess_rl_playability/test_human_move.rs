//! Human move validation tests.
//!
//! Ensures the engine correctly validates human moves:
//! legal moves accepted, illegal moves rejected, check detection,
//! checkmate/stalemate terminal states.

use super::helpers::*;

/// Initial position has 20 legal moves for white.
#[test]
fn initial_position_legal_move_count() {
    let src = chess_program(r#"
        let board = init_board();
        let moves = legal_moves(board, 1);
        print(len(moves) / 2);
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(parse_int_at(&out, 0), 20, "white should have 20 legal moves in initial position");
}

/// Initial position has 20 legal moves for black (mirrored).
#[test]
fn initial_position_black_moves() {
    let src = chess_program(r#"
        let board = init_board();
        let moves = legal_moves(board, -1);
        print(len(moves) / 2);
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(parse_int_at(&out, 0), 20, "black should have 20 legal moves in initial position");
}

/// A move from e2 to e4 is legal in the initial position.
#[test]
fn e2_e4_is_legal() {
    let src = chess_program(r#"
        let board = init_board();
        let moves = legal_moves(board, 1);
        // e2 = sq(1, 4) = 12, e4 = sq(3, 4) = 28
        let found = false;
        let i = 0;
        while i < len(moves) {
            if moves[i] == 12 && moves[i + 1] == 28 {
                found = true;
            }
            i = i + 2;
        }
        if found { print("legal"); } else { print("illegal"); }
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(out[0], "legal");
}

/// An illegal move (e2 to e5) is not in legal moves.
#[test]
fn e2_e5_is_illegal() {
    let src = chess_program(r#"
        let board = init_board();
        let moves = legal_moves(board, 1);
        // e2 = 12, e5 = sq(4, 4) = 36
        let found = false;
        let i = 0;
        while i < len(moves) {
            if moves[i] == 12 && moves[i + 1] == 36 {
                found = true;
            }
            i = i + 2;
        }
        if found { print("legal"); } else { print("illegal"); }
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(out[0], "illegal");
}

/// Apply move and verify board state changes.
#[test]
fn apply_move_updates_board() {
    let src = chess_program(r#"
        let board = init_board();
        // e2(12) -> e4(28)
        let board2 = apply_move(board, 12, 28);
        print(board2[12]); // should be 0 (empty)
        print(board2[28]); // should be 1 (white pawn)
        print(board[12]);  // original unchanged: 1
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(parse_int_at(&out, 0), 0, "from-square should be empty after move");
    assert_eq!(parse_int_at(&out, 1), 1, "to-square should have pawn after move");
    assert_eq!(parse_int_at(&out, 2), 1, "original board unchanged (functional)");
}

/// After 1.e4 e5, white has more than 20 moves (new pawn + bishop/queen access).
#[test]
fn move_count_after_opening() {
    let src = chess_program(r#"
        let board = init_board();
        let board2 = apply_move(board, 12, 28); // e2-e4
        let board3 = apply_move(board2, 52, 36); // e7-e5
        let moves = legal_moves(board3, 1);
        print(len(moves) / 2);
    "#);
    let out = run_mir(&src, 42);
    let count = parse_int_at(&out, 0);
    assert!(count > 20, "after 1.e4 e5, white should have more than 20 moves, got {count}");
}

/// Initial position is not terminal.
#[test]
fn initial_position_not_terminal() {
    let src = chess_program(r#"
        let board = init_board();
        print(terminal_status(board, 1));
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(parse_int_at(&out, 0), 0, "initial position should not be terminal");
}

/// in_check is false for initial position.
#[test]
fn initial_position_not_in_check() {
    let src = chess_program(r#"
        let board = init_board();
        if in_check(board, 1) { print("check"); } else { print("no_check"); }
        if in_check(board, -1) { print("check"); } else { print("no_check"); }
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(out[0], "no_check");
    assert_eq!(out[1], "no_check");
}

/// White king is at square 4 in initial position.
#[test]
fn find_king_initial() {
    let src = chess_program(r#"
        let board = init_board();
        print(find_king(board, 1));
        print(find_king(board, -1));
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(parse_int_at(&out, 0), 4, "white king at e1 (sq 4)");
    assert_eq!(parse_int_at(&out, 1), 60, "black king at e8 (sq 60)");
}

/// Pawn promotion: white pawn on rank 6 moving to rank 7 becomes queen.
#[test]
fn pawn_promotion_to_queen() {
    let src = chess_program(r#"
        // Construct a board with white pawn at a7 (sq 48)
        let board = [];
        let i = 0;
        while i < 64 {
            if i == 4 { board = array_push(board, 6); }    // white king at e1
            else { if i == 48 { board = array_push(board, 1); }  // white pawn at a7
            else { if i == 60 { board = array_push(board, -6); } // black king at e8
            else { board = array_push(board, 0); } } }
            i = i + 1;
        }
        // Move pawn from a7 (48) to a8 (56)
        let board2 = apply_move(board, 48, 56);
        print(board2[56]); // should be 5 (white queen)
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(parse_int_at(&out, 0), 5, "promoted pawn should be queen (5)");
}

/// Move generation is deterministic across runs.
#[test]
fn move_generation_deterministic() {
    let src = chess_program(r#"
        let board = init_board();
        let moves = legal_moves(board, 1);
        let i = 0;
        while i < len(moves) {
            print(moves[i]);
            i = i + 1;
        }
    "#);
    let out1 = run_mir(&src, 42);
    let out2 = run_mir(&src, 99);
    assert_eq!(out1, out2, "legal move generation should be deterministic (independent of seed)");
}
