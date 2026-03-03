//! Test 01: Board Invariants
//!
//! Verifies the chess board representation: correct initial position,
//! piece encoding, and board structure.

use super::cjc_source::*;

fn run(extra: &str) -> Vec<String> {
    let src = format!("{CHESS_ENV}\n{extra}");
    run_mir(&src, 42)
}

#[test]
fn initial_board_has_64_squares() {
    let out = run(r#"
let b = init_board();
print(len(b));
"#);
    assert_eq!(parse_int(&out), 64);
}

#[test]
fn initial_board_white_pieces() {
    // Rank 0: R N B Q K B N R = 4 2 3 5 6 3 2 4
    let out = run(r#"
let b = init_board();
print(b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]);
"#);
    assert_eq!(out[0], "4 2 3 5 6 3 2 4");
}

#[test]
fn initial_board_white_pawns() {
    let out = run(r#"
let b = init_board();
let ok = true;
let i = 8;
while i < 16 {
    if b[i] != 1 { ok = false; }
    i = i + 1;
};
print(ok);
"#);
    assert_eq!(out[0], "true");
}

#[test]
fn initial_board_empty_middle() {
    let out = run(r#"
let b = init_board();
let empty_count = 0;
let i = 16;
while i < 48 {
    if b[i] == 0 { empty_count = empty_count + 1; }
    i = i + 1;
};
print(empty_count);
"#);
    assert_eq!(parse_int(&out), 32);
}

#[test]
fn initial_board_black_pawns() {
    let out = run(r#"
let b = init_board();
let ok = true;
let i = 48;
while i < 56 {
    if b[i] != -1 { ok = false; }
    i = i + 1;
};
print(ok);
"#);
    assert_eq!(out[0], "true");
}

#[test]
fn initial_board_black_pieces() {
    let out = run(r#"
let b = init_board();
print(b[56], b[57], b[58], b[59], b[60], b[61], b[62], b[63]);
"#);
    assert_eq!(out[0], "-4 -2 -3 -5 -6 -3 -2 -4");
}

#[test]
fn initial_board_piece_count() {
    let out = run(r#"
let b = init_board();
let white_count = 0;
let black_count = 0;
let i = 0;
while i < 64 {
    if b[i] > 0 { white_count = white_count + 1; }
    if b[i] < 0 { black_count = black_count + 1; }
    i = i + 1;
};
print(white_count, black_count);
"#);
    assert_eq!(out[0], "16 16");
}

#[test]
fn both_kings_present() {
    let out = run(r#"
let b = init_board();
let wk = find_king(b, 1);
let bk = find_king(b, -1);
print(wk, bk);
"#);
    // White king at e1 = sq 4, Black king at e8 = sq 60
    assert_eq!(out[0], "4 60");
}

#[test]
fn apply_move_preserves_64_squares() {
    let out = run(r#"
let b = init_board();
let b2 = apply_move(b, 12, 28);
print(len(b2));
"#);
    assert_eq!(parse_int(&out), 64);
}

#[test]
fn apply_move_moves_piece() {
    // Move white e2 pawn (sq 12) to e4 (sq 28)
    let out = run(r#"
let b = init_board();
let b2 = apply_move(b, 12, 28);
print(b2[12], b2[28]);
"#);
    // Source should be empty, dest should have pawn
    assert_eq!(out[0], "0 1");
}

#[test]
fn initial_position_not_in_check() {
    let out = run(r#"
let b = init_board();
let wcheck = in_check(b, 1);
let bcheck = in_check(b, -1);
print(wcheck, bcheck);
"#);
    assert_eq!(out[0], "false false");
}
