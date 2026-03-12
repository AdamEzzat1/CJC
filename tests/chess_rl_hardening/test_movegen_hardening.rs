//! Move generation hardening tests.
//!
//! Validates pseudo-legal and legal move generation, move format,
//! and piece-specific movement patterns.

use super::helpers::*;

// ============================================================
// Move format and basic counts
// ============================================================

#[test]
fn moves_are_even_length() {
    // Legal moves array should always have even length (pairs of from,to)
    let src = chess_program(r#"
        let b = init_board();
        let m = legal_moves(b, 1);
        print(len(m) % 2);
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(out[0], "0", "move list must have even length");
}

#[test]
fn initial_white_20_moves() {
    let src = chess_program(r#"
        let b = init_board();
        let m = legal_moves(b, 1);
        print(len(m) / 2);
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(out[0], "20", "white should have 20 legal moves initially");
}

#[test]
fn initial_black_20_moves() {
    let src = chess_program(r#"
        let b = init_board();
        let m = legal_moves(b, -1);
        print(len(m) / 2);
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(out[0], "20", "black should have 20 legal moves initially");
}

// ============================================================
// Move bounds
// ============================================================

#[test]
fn all_moves_in_board_range() {
    let src = chess_program(r#"
        let b = init_board();
        let m = legal_moves(b, 1);
        let ok = true;
        let i = 0;
        while i < len(m) {
            if m[i] < 0 || m[i] > 63 { ok = false; }
            i = i + 1;
        }
        print(ok);
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(out[0], "true", "all move squares must be in [0, 63]");
}

#[test]
fn moves_from_own_pieces_only() {
    // Every "from" square in white's move list should contain a white piece
    let src = chess_program(r#"
        let b = init_board();
        let m = legal_moves(b, 1);
        let ok = true;
        let i = 0;
        while i < len(m) {
            let from = m[i];
            if b[from] <= 0 { ok = false; }
            i = i + 2;
        }
        print(ok);
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(out[0], "true");
}

// ============================================================
// Specific piece movement
// ============================================================

#[test]
fn knight_initial_moves() {
    // White knights at b1(sq1) and g1(sq6) can each reach 2 squares
    let src = chess_program(r#"
        let b = init_board();
        let m = legal_moves(b, 1);
        let knight_moves = 0;
        let i = 0;
        while i < len(m) {
            let from = m[i];
            if b[from] == 2 { knight_moves = knight_moves + 1; }
            i = i + 2;
        }
        print(knight_moves);
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(out[0], "4", "two knights * 2 moves each = 4");
}

#[test]
fn pawn_initial_moves() {
    // 8 pawns: each can move 1 or 2 squares forward = 16 moves
    let src = chess_program(r#"
        let b = init_board();
        let m = legal_moves(b, 1);
        let pawn_moves = 0;
        let i = 0;
        while i < len(m) {
            let from = m[i];
            if b[from] == 1 { pawn_moves = pawn_moves + 1; }
            i = i + 2;
        }
        print(pawn_moves);
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(out[0], "16", "8 pawns * 2 moves each = 16");
}

// ============================================================
// Move after apply
// ============================================================

#[test]
fn move_count_changes_after_move() {
    let src = chess_program(r#"
        let b = init_board();
        let m1 = legal_moves(b, 1);
        let count1 = len(m1) / 2;
        let b2 = apply_move(b, 12, 28);
        let m2 = legal_moves(b2, -1);
        let count2 = len(m2) / 2;
        print(count1);
        print(count2);
    "#);
    let out = run_mir(&src, 42);
    let c1: i64 = out[0].parse().unwrap();
    let c2: i64 = out[1].parse().unwrap();
    assert_eq!(c1, 20);
    // After e2-e4, black still has ~20 moves (may vary slightly)
    assert!(c2 >= 18 && c2 <= 22, "black should have ~20 moves, got {c2}");
}

// ============================================================
// Legal moves exclude self-check
// ============================================================

#[test]
fn legal_moves_never_leave_king_in_check() {
    // Play through first 4 moves, verify legality at each step
    let src = chess_program(r#"
        let b = init_board();
        let side = 1;
        let ok = true;
        let step = 0;
        while step < 4 {
            let m = legal_moves(b, side);
            if len(m) < 2 { break; }
            let from = m[0];
            let to = m[1];
            let b2 = apply_move(b, from, to);
            if in_check(b2, side) { ok = false; }
            b = b2;
            side = -1 * side;
            step = step + 1;
        }
        print(ok);
    "#);
    let out = run_mir(&src, 42);
    assert_eq!(out[0], "true", "legal moves must never leave own king in check");
}
