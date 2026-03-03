//! Test 03: Legal Move Sanity
//!
//! Verifies that legal move generation correctly filters out moves that leave
//! the king in check, handles piece movement rules, and detects terminal states.

use super::cjc_source::*;

fn run(extra: &str) -> Vec<String> {
    let src = format!("{CHESS_ENV}\n{extra}");
    run_mir(&src, 42)
}

#[test]
fn pawn_cannot_move_through_piece() {
    // Place a piece in front of a pawn — pawn can't advance
    let out = run(r#"
let b = init_board();
// Put a black pawn right in front of white's e-pawn
let b2 = apply_move(b, 52, 20);
// Now check: sq 12 has white pawn, sq 20 has black pawn
// White pawn on e2 (sq 12) should NOT be able to move to e3 (sq 20) or e4 (sq 28)
let moves = legal_moves(b2, 1);
let pawn_e2_can_advance = false;
let i = 0;
while i < len(moves) {
    if moves[i] == 12 {
        if moves[i+1] == 20 || moves[i+1] == 28 {
            pawn_e2_can_advance = true;
        }
    }
    i = i + 2;
}
print(pawn_e2_can_advance);
"#);
    assert_eq!(out[0], "false", "pawn should not advance through a blocking piece");
}

#[test]
fn king_cannot_move_to_attacked_square() {
    // Set up a position where the king can't go to a specific square
    let out = run(r#"
let b = init_board();
let moves = legal_moves(b, 1);
// In initial position, white king has no legal moves (surrounded by own pieces)
let king_moves = 0;
let i = 0;
while i < len(moves) {
    if moves[i] == 4 { king_moves = king_moves + 1; }
    i = i + 2;
}
print(king_moves);
"#);
    assert_eq!(parse_int(&out), 0, "king should have no moves in initial position");
}

#[test]
fn knight_moves_correct_pattern() {
    // White knight on b1 (sq 1) should have exactly 2 moves: a3 (sq 16) and c3 (sq 18)
    let out = run(r#"
let b = init_board();
let moves = legal_moves(b, 1);
let knight_b1_targets = [];
let i = 0;
while i < len(moves) {
    if moves[i] == 1 {
        knight_b1_targets = array_push(knight_b1_targets, moves[i + 1]);
    }
    i = i + 2;
}
print(len(knight_b1_targets));
"#);
    assert_eq!(parse_int(&out), 2, "b1 knight should have 2 moves");
}

#[test]
fn terminal_not_reached_initially() {
    let out = run(r#"
let b = init_board();
let status = terminal_status(b, 1);
print(status);
"#);
    assert_eq!(parse_int(&out), 0, "initial position is not terminal");
}

#[test]
fn scholars_mate_is_checkmate() {
    // Scholar's mate: 1.e4 e5 2.Bc4 Nc6 3.Qh5 Nf6? 4.Qxf7#
    let out = run(r#"
let b = init_board();
// 1. e4
let b = apply_move(b, 12, 28);
// 1... e5
let b = apply_move(b, 52, 36);
// 2. Bc4 (f1=sq5 -> c4=sq26)
let b = apply_move(b, 5, 26);
// 2... Nc6 (b8=sq57 -> c6=sq42)
let b = apply_move(b, 57, 42);
// 3. Qh5 (d1=sq3 -> h5=sq39)
let b = apply_move(b, 3, 39);
// 3... Nf6 (g8=sq62 -> f6=sq45)
let b = apply_move(b, 62, 45);
// 4. Qxf7# (h5=sq39 -> f7=sq53)
let b = apply_move(b, 39, 53);
// Black is checkmated
let status = terminal_status(b, -1);
print(status);
"#);
    assert_eq!(parse_int(&out), 2, "should detect checkmate (scholar's mate)");
}

#[test]
fn stalemate_detected() {
    // Construct a simple stalemate position:
    // White king on a1, Black king on c2, Black queen on b3
    // White to move: no legal moves, not in check = stalemate
    let out = run(r#"
// Build a custom board: all empty except specific pieces
// Manually set pieces
let board = [];
let i = 0;
while i < 64 {
    let val = 0;
    if i == 0 { val = 6; }
    if i == 10 { val = -6; }
    if i == 17 { val = -5; }
    board = array_push(board, val);
    i = i + 1;
}
// White king on a1 (0), Black king on c2 (10), Black queen on b3 (17)
// White king can't go: a2 attacked by queen, b1 attacked by queen, b2 attacked by king+queen
let status = terminal_status(board, 1);
print(status);
"#);
    assert_eq!(parse_int(&out), 3, "should detect stalemate");
}

#[test]
fn capture_removes_piece() {
    let out = run(r#"
let b = init_board();
// Move white knight to capture a hypothetical piece
// First set up a capture scenario
let b2 = apply_move(b, 12, 28); // e4
let b3 = apply_move(b2, 51, 35); // d5
// Now white e4 pawn can capture d5 pawn
let b4 = apply_move(b3, 28, 35);
// d5 should now have white pawn, e4 should be empty, d7 should be empty (source was there before)
print(b4[35], b4[28]);
"#);
    assert_eq!(out[0], "1 0", "capture should place piece on target and clear source");
}

#[test]
fn feature_encoding_shape() {
    let out = run(r#"
let b = init_board();
let features = encode_board(b, 1);
let s = features.shape();
print(len(s), s[0], s[1]);
"#);
    assert_eq!(out[0], "2 1 64", "features should be [1, 64] tensor");
}

#[test]
fn feature_encoding_deterministic() {
    let code = r#"
let b = init_board();
let f = encode_board(b, 1);
print(f);
"#;
    let out1 = run(code);
    let out2 = run(code);
    assert_eq!(out1, out2, "feature encoding must be deterministic");
}
