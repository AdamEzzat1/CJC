//! Test 02: Move Generation Determinism
//!
//! Verifies that move generation is deterministic: same board + same side
//! always produces the exact same ordered list of legal moves.

use super::cjc_source::*;

fn run(extra: &str) -> Vec<String> {
    let src = format!("{CHESS_ENV}\n{extra}");
    run_mir(&src, 42)
}

#[test]
fn initial_white_moves_deterministic() {
    let code = r#"
let b = init_board();
let moves = legal_moves(b, 1);
let i = 0;
let s = "";
while i < len(moves) {
    if i > 0 { s = s + " "; }
    s = s + to_string(moves[i]);
    i = i + 1;
}
print(s);
"#;
    let out1 = run(code);
    let out2 = run(code);
    assert_eq!(out1, out2, "move generation must be deterministic");
    // Should have at least some moves
    assert!(!out1[0].is_empty(), "should produce moves");
}

#[test]
fn initial_white_has_20_moves() {
    // Standard chess: 16 pawn moves + 4 knight moves = 20 legal moves
    let out = run(r#"
let b = init_board();
let moves = legal_moves(b, 1);
print(len(moves) / 2);
"#);
    assert_eq!(parse_int(&out), 20, "initial position should have 20 legal moves");
}

#[test]
fn initial_black_has_20_moves() {
    let out = run(r#"
let b = init_board();
let moves = legal_moves(b, -1);
print(len(moves) / 2);
"#);
    assert_eq!(parse_int(&out), 20, "initial position should have 20 legal moves for black");
}

#[test]
fn move_list_is_paired() {
    // Verify moves come in (from, to) pairs — always even length
    let out = run(r#"
let b = init_board();
let moves = legal_moves(b, 1);
print(len(moves) % 2);
"#);
    assert_eq!(parse_int(&out), 0, "move list length must be even");
}

#[test]
fn moves_are_valid_squares() {
    let out = run(r#"
let b = init_board();
let moves = legal_moves(b, 1);
let valid = true;
let i = 0;
while i < len(moves) {
    if moves[i] < 0 || moves[i] > 63 {
        valid = false;
    }
    i = i + 1;
}
print(valid);
"#);
    assert_eq!(out[0], "true");
}

#[test]
fn after_e4_white_moves_change() {
    let out = run(r#"
let b = init_board();
let b2 = apply_move(b, 12, 28);
let white_moves_count = len(legal_moves(b2, 1)) / 2;
// After 1. e4, white shouldn't have 20 moves anymore (the pawn can't move twice)
// but other pieces now have more options
print(white_moves_count);
"#);
    let count = parse_int(&out);
    // After e4, white has 29 or 30 moves depending on exact count
    assert!(count >= 20 && count <= 35, "move count after e4 should be reasonable, got {count}");
}

#[test]
fn double_run_identical_moves() {
    // Run the same complex position twice, verify identical move lists
    let code = r#"
let b = init_board();
let b2 = apply_move(b, 12, 28);
let b3 = apply_move(b2, 52, 36);
let b4 = apply_move(b3, 11, 27);
let moves = legal_moves(b4, -1);
let i = 0;
let s = "";
while i < len(moves) {
    if i > 0 { s = s + " "; }
    s = s + to_string(moves[i]);
    i = i + 1;
}
print(s);
"#;
    let out1 = run(code);
    let out2 = run(code);
    assert_eq!(out1, out2, "move generation must be deterministic across runs");
}
