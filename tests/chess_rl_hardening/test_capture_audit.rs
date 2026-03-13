use super::helpers::*;

// Helper: build a custom board position in CJC
// Takes a list of (square, piece) placements, everything else is 0
fn custom_board_program(placements: &[(i64, i64)], main_body: &str) -> String {
    // Build a 64-element array literal with pieces placed at specified squares
    let mut board_arr = vec![0i64; 64];
    for (sq, piece) in placements {
        board_arr[*sq as usize] = *piece;
    }
    let board_str = board_arr
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(", ");

    chess_program(&format!(
        "let board = [{}];\n    {}",
        board_str, main_body
    ))
}

// Test 1: Pawn captures diagonally (white pawn captures black piece)
#[test]
fn pawn_captures_diagonally() {
    // White pawn on e4 (sq 28), black bishop on d5 (sq 35), white king on e1 (sq 4), black king on e8 (sq 60)
    let src = custom_board_program(
        &[(4, 6), (28, 1), (35, -3), (60, -6)],
        r#"
    let moves = legal_moves(board, 1);
    let num = len(moves) / 2;
    // Check that pawn can capture the bishop
    let found_capture = 0;
    let mi = 0;
    while mi < num {
        let from = moves[mi * 2];
        let to = moves[mi * 2 + 1];
        if from == 28 && to == 35 {
            found_capture = 1;
        }
        mi = mi + 1;
    }
    print(found_capture);
    // Apply capture and verify
    let new_board = apply_move(board, 28, 35);
    print(new_board[35]); // should be 1 (white pawn)
    print(new_board[28]); // should be 0
    "#,
    );
    let out = run_mir(&src, 42);
    assert_eq!(
        out[0].trim(),
        "1",
        "Pawn capture of bishop should be a legal move"
    );
    assert_eq!(
        out[1].trim(),
        "1",
        "Pawn should occupy the capture square"
    );
    assert_eq!(
        out[2].trim(),
        "0",
        "Source square should be empty after capture"
    );
}

// Test 2: Knight captures
#[test]
fn knight_captures_piece() {
    // White knight on c3 (sq 18), black rook on d5 (sq 35), white king on e1 (sq 4), black king on e8 (sq 60)
    let src = custom_board_program(
        &[(4, 6), (18, 2), (35, -4), (60, -6)],
        r#"
    let moves = legal_moves(board, 1);
    let num = len(moves) / 2;
    let found = 0;
    let mi = 0;
    while mi < num {
        if moves[mi * 2] == 18 && moves[mi * 2 + 1] == 35 {
            found = 1;
        }
        mi = mi + 1;
    }
    print(found);
    let new_board = apply_move(board, 18, 35);
    print(new_board[35]);
    print(new_board[18]);
    "#,
    );
    let out = run_mir(&src, 42);
    assert_eq!(out[0].trim(), "1", "Knight should capture rook on d5");
    assert_eq!(out[1].trim(), "2", "Knight should occupy capture square");
    assert_eq!(out[2].trim(), "0", "Source square should be empty");
}

// Test 3: Bishop captures diagonally
#[test]
fn bishop_captures_diagonally() {
    // White bishop on c1 (sq 2), black pawn on f4 (sq 29), white king e1 (sq 4), black king e8 (sq 60)
    let src = custom_board_program(
        &[(4, 6), (2, 3), (29, -1), (60, -6)],
        r#"
    let moves = legal_moves(board, 1);
    let num = len(moves) / 2;
    let found = 0;
    let mi = 0;
    while mi < num {
        if moves[mi * 2] == 2 && moves[mi * 2 + 1] == 29 {
            found = 1;
        }
        mi = mi + 1;
    }
    print(found);
    let new_board = apply_move(board, 2, 29);
    print(new_board[29]);
    print(new_board[2]);
    "#,
    );
    let out = run_mir(&src, 42);
    assert_eq!(
        out[0].trim(),
        "1",
        "Bishop should capture pawn diagonally"
    );
    assert_eq!(out[1].trim(), "3", "Bishop should occupy capture square");
    assert_eq!(out[2].trim(), "0", "Source square should be empty");
}

// Test 4: Rook captures along rank/file
#[test]
fn rook_captures_along_file() {
    // White rook on a1 (sq 0), black queen on a5 (sq 32), white king e1 (sq 4), black king e8 (sq 60)
    let src = custom_board_program(
        &[(4, 6), (0, 4), (32, -5), (60, -6)],
        r#"
    let moves = legal_moves(board, 1);
    let num = len(moves) / 2;
    let found = 0;
    let mi = 0;
    while mi < num {
        if moves[mi * 2] == 0 && moves[mi * 2 + 1] == 32 {
            found = 1;
        }
        mi = mi + 1;
    }
    print(found);
    let new_board = apply_move(board, 0, 32);
    print(new_board[32]);
    print(new_board[0]);
    "#,
    );
    let out = run_mir(&src, 42);
    assert_eq!(
        out[0].trim(),
        "1",
        "Rook should capture queen along a-file"
    );
    assert_eq!(out[1].trim(), "4", "Rook should occupy capture square");
    assert_eq!(out[2].trim(), "0", "Source square empty");
}

// Test 5: Queen captures diagonally
#[test]
fn queen_captures_diagonally_and_straight() {
    // White queen on d4 (sq 27), black knight on g7 (sq 54) - diagonal, white king e1 (sq 4), black king a8 (sq 56)
    let src = custom_board_program(
        &[(4, 6), (27, 5), (54, -2), (56, -6)],
        r#"
    let moves = legal_moves(board, 1);
    let num = len(moves) / 2;
    let found = 0;
    let mi = 0;
    while mi < num {
        if moves[mi * 2] == 27 && moves[mi * 2 + 1] == 54 {
            found = 1;
        }
        mi = mi + 1;
    }
    print(found);
    let new_board = apply_move(board, 27, 54);
    print(new_board[54]);
    "#,
    );
    let out = run_mir(&src, 42);
    assert_eq!(
        out[0].trim(),
        "1",
        "Queen should capture knight diagonally"
    );
    assert_eq!(out[1].trim(), "5", "Queen should occupy capture square");
}

// Test 6: King captures adjacent piece
#[test]
fn king_captures_adjacent() {
    // White king on e4 (sq 28), black pawn on f5 (sq 37) - not defended, black king a8 (sq 56)
    let src = custom_board_program(
        &[(28, 6), (37, -1), (56, -6)],
        r#"
    let moves = legal_moves(board, 1);
    let num = len(moves) / 2;
    let found = 0;
    let mi = 0;
    while mi < num {
        if moves[mi * 2] == 28 && moves[mi * 2 + 1] == 37 {
            found = 1;
        }
        mi = mi + 1;
    }
    print(found);
    let new_board = apply_move(board, 28, 37);
    print(new_board[37]);
    print(new_board[28]);
    "#,
    );
    let out = run_mir(&src, 42);
    assert_eq!(out[0].trim(), "1", "King should capture undefended pawn");
    assert_eq!(out[1].trim(), "6", "King should occupy capture square");
    assert_eq!(out[2].trim(), "0", "Source square empty");
}

// Test 7: Pawn does NOT capture forward (only diagonal)
#[test]
fn pawn_cannot_capture_forward() {
    // White pawn on e4 (sq 28), black pawn on e5 (sq 36), white king e1 (sq 4), black king e8 (sq 60)
    let src = custom_board_program(
        &[(4, 6), (28, 1), (36, -1), (60, -6)],
        r#"
    let moves = legal_moves(board, 1);
    let num = len(moves) / 2;
    let found = 0;
    let mi = 0;
    while mi < num {
        if moves[mi * 2] == 28 && moves[mi * 2 + 1] == 36 {
            found = 1;
        }
        mi = mi + 1;
    }
    print(found);
    "#,
    );
    let out = run_mir(&src, 42);
    assert_eq!(out[0].trim(), "0", "Pawn must not capture forward");
}

// Test 8: Black pawn captures diagonally
#[test]
fn black_pawn_captures_diagonally() {
    // Black pawn on d5 (sq 35), white knight on e4 (sq 28), white king e1 (sq 4), black king e8 (sq 60)
    let src = custom_board_program(
        &[(4, 6), (28, 2), (35, -1), (60, -6)],
        r#"
    let moves = legal_moves(board, -1);
    let num = len(moves) / 2;
    let found = 0;
    let mi = 0;
    while mi < num {
        if moves[mi * 2] == 35 && moves[mi * 2 + 1] == 28 {
            found = 1;
        }
        mi = mi + 1;
    }
    print(found);
    let new_board = apply_move(board, 35, 28);
    print(new_board[28]);
    print(new_board[35]);
    "#,
    );
    let out = run_mir(&src, 42);
    assert_eq!(
        out[0].trim(),
        "1",
        "Black pawn should capture white knight diagonally"
    );
    assert_eq!(
        out[1].trim(),
        "-1",
        "Black pawn should occupy capture square"
    );
    assert_eq!(out[2].trim(), "0", "Source square empty");
}

// Test 9: Sliding piece blocked from capture by intervening piece
#[test]
fn rook_blocked_from_capture_by_intervening() {
    // White rook a1 (sq 0), white pawn a3 (sq 16), black rook a7 (sq 48), white king e1 (sq 4), black king e8 (sq 60)
    let src = custom_board_program(
        &[(4, 6), (0, 4), (16, 1), (48, -4), (60, -6)],
        r#"
    let moves = legal_moves(board, 1);
    let num = len(moves) / 2;
    let found = 0;
    let mi = 0;
    while mi < num {
        if moves[mi * 2] == 0 && moves[mi * 2 + 1] == 48 {
            found = 1;
        }
        mi = mi + 1;
    }
    print(found);
    "#,
    );
    let out = run_mir(&src, 42);
    assert_eq!(
        out[0].trim(),
        "0",
        "Rook should not capture through own pawn"
    );
}

// Test 10: Capture removes captured piece from board (piece count check)
#[test]
fn capture_reduces_piece_count() {
    // White pawn on e4 (sq 28), black pawn on d5 (sq 35), white king e1 (sq 4), black king e8 (sq 60)
    let src = custom_board_program(
        &[(4, 6), (28, 1), (35, -1), (60, -6)],
        r#"
    // Count pieces before
    let count_before = 0;
    let i = 0;
    while i < 64 {
        if board[i] != 0 {
            count_before = count_before + 1;
        }
        i = i + 1;
    }
    let new_board = apply_move(board, 28, 35);
    let count_after = 0;
    let j = 0;
    while j < 64 {
        if new_board[j] != 0 {
            count_after = count_after + 1;
        }
        j = j + 1;
    }
    print(count_before);
    print(count_after);
    "#,
    );
    let out = run_mir(&src, 42);
    let before: i64 = out[0].trim().parse().unwrap();
    let after: i64 = out[1].trim().parse().unwrap();
    assert_eq!(before, 4, "Should have 4 pieces before capture");
    assert_eq!(after, 3, "Should have 3 pieces after capture (one removed)");
}

// Test 11: King cannot capture defended piece
#[test]
fn king_cannot_capture_defended_piece() {
    // White king on e1 (sq 4), black pawn on f2 (sq 13) defended by black bishop on h4 (sq 31)
    // Black king on e8 (sq 60)
    let src = custom_board_program(
        &[(4, 6), (13, -1), (31, -3), (60, -6)],
        r#"
    let moves = legal_moves(board, 1);
    let num = len(moves) / 2;
    let found = 0;
    let mi = 0;
    while mi < num {
        if moves[mi * 2] == 4 && moves[mi * 2 + 1] == 13 {
            found = 1;
        }
        mi = mi + 1;
    }
    print(found);
    "#,
    );
    let out = run_mir(&src, 42);
    assert_eq!(
        out[0].trim(),
        "0",
        "King must not capture defended piece (would be in check)"
    );
}

// Test 12: Full game capture sequence - Scholar's mate captures work
#[test]
fn scholars_mate_captures_valid() {
    // Play Scholar's mate: 1.e4 e5 2.Bc4 Nc6 3.Qh5 Nf6 4.Qxf7#
    let src = chess_program(
        r#"
    let board = init_board();
    // 1.e4
    board = apply_move(board, 12, 28);
    // 1...e5
    board = apply_move(board, 52, 36);
    // 2.Bc4
    board = apply_move(board, 5, 26);
    // 2...Nc6
    board = apply_move(board, 57, 42);
    // 3.Qh5
    board = apply_move(board, 3, 39);
    // 3...Nf6
    board = apply_move(board, 62, 45);
    // 4.Qxf7# - queen captures pawn on f7
    let f7_before = board[53];
    board = apply_move(board, 39, 53);
    let f7_after = board[53];
    print(f7_before);
    print(f7_after);
    print(terminal_status(board, -1));
    "#,
    );
    let out = run_mir(&src, 42);
    assert_eq!(
        out[0].trim(),
        "-1",
        "f7 should have black pawn before capture"
    );
    assert_eq!(
        out[1].trim(),
        "5",
        "f7 should have white queen after capture"
    );
    assert_eq!(out[2].trim(), "2", "Black should be checkmated");
}
