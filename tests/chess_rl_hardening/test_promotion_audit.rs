use super::helpers::*;

// Helper: build a custom board position in CJC
fn custom_board_program(placements: &[(i64, i64)], main_body: &str) -> String {
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

// Test 1: White pawn promotes to queen on rank 7
#[test]
fn white_pawn_promotes_to_queen() {
    // White pawn on e7 (sq 52), white king e1 (sq 4), black king a8 (sq 56)
    let src = custom_board_program(
        &[(4, 6), (52, 1), (56, -6)],
        r#"
    let new_board = apply_move(board, 52, 60);
    print(new_board[60]);
    print(new_board[52]);
    "#,
    );
    let out = run_mir(&src, 42);
    assert_eq!(out[0].trim(), "5", "White pawn should promote to queen (5)");
    assert_eq!(out[1].trim(), "0", "Source square empty after promotion");
}

// Test 2: Black pawn promotes to queen on rank 0
#[test]
fn black_pawn_promotes_to_queen() {
    // Black pawn on d2 (sq 11), white king h1 (sq 7), black king e8 (sq 60)
    let src = custom_board_program(
        &[(7, 6), (11, -1), (60, -6)],
        r#"
    let new_board = apply_move(board, 11, 3);
    print(new_board[3]);
    print(new_board[11]);
    "#,
    );
    let out = run_mir(&src, 42);
    assert_eq!(out[0].trim(), "-5", "Black pawn should promote to black queen (-5)");
    assert_eq!(out[1].trim(), "0", "Source square empty after promotion");
}

// Test 3: Promotion move appears in legal moves
#[test]
fn promotion_move_is_legal() {
    // White pawn on a7 (sq 48), white king e1 (sq 4), black king e8 (sq 60)
    let src = custom_board_program(
        &[(4, 6), (48, 1), (60, -6)],
        r#"
    let moves = legal_moves(board, 1);
    let num = len(moves) / 2;
    let found = 0;
    let mi = 0;
    while mi < num {
        if moves[mi * 2] == 48 && moves[mi * 2 + 1] == 56 {
            found = 1;
        }
        mi = mi + 1;
    }
    print(found);
    "#,
    );
    let out = run_mir(&src, 42);
    assert_eq!(out[0].trim(), "1", "Pawn advance to promotion rank should be legal");
}

// Test 4: Promotion with capture (diagonal capture onto rank 8)
#[test]
fn promotion_with_capture() {
    // White pawn on d7 (sq 51), black rook on e8 (sq 60), white king e1 (sq 4), black king a8 (sq 56)
    let src = custom_board_program(
        &[(4, 6), (51, 1), (60, -4), (56, -6)],
        r#"
    let moves = legal_moves(board, 1);
    let num = len(moves) / 2;
    let found = 0;
    let mi = 0;
    while mi < num {
        if moves[mi * 2] == 51 && moves[mi * 2 + 1] == 60 {
            found = 1;
        }
        mi = mi + 1;
    }
    print(found);
    // Apply promotion-capture
    let new_board = apply_move(board, 51, 60);
    print(new_board[60]);
    print(new_board[51]);
    "#,
    );
    let out = run_mir(&src, 42);
    assert_eq!(out[0].trim(), "1", "Pawn capture-promote should be legal");
    assert_eq!(out[1].trim(), "5", "Piece at target should be white queen after promotion-capture");
    assert_eq!(out[2].trim(), "0", "Source square empty");
}

// Test 5: Black pawn promotion with capture
#[test]
fn black_promotion_with_capture() {
    // Black pawn on g2 (sq 14), white rook on h1 (sq 7), white king a1 (sq 0), black king e8 (sq 60)
    let src = custom_board_program(
        &[(0, 6), (7, 4), (14, -1), (60, -6)],
        r#"
    let moves = legal_moves(board, -1);
    let num = len(moves) / 2;
    let found = 0;
    let mi = 0;
    while mi < num {
        if moves[mi * 2] == 14 && moves[mi * 2 + 1] == 7 {
            found = 1;
        }
        mi = mi + 1;
    }
    print(found);
    let new_board = apply_move(board, 14, 7);
    print(new_board[7]);
    "#,
    );
    let out = run_mir(&src, 42);
    assert_eq!(out[0].trim(), "1", "Black pawn capture-promote should be legal");
    assert_eq!(out[1].trim(), "-5", "Piece should be black queen after capture-promote");
}

// Test 6: Promoted piece generates queen-legal moves
#[test]
fn promoted_queen_generates_moves() {
    // White pawn on a7 (sq 48), white king e1 (sq 4), black king e8 (sq 60)
    // After promotion to a8 (sq 56), the queen should have multiple moves
    let src = custom_board_program(
        &[(4, 6), (48, 1), (60, -6)],
        r#"
    let new_board = apply_move(board, 48, 56);
    // Now queen on a8 (sq 56), should have multiple legal moves
    let moves = legal_moves(new_board, -1);
    let num = len(moves) / 2;
    // Black should still have some king moves (even though white has a queen)
    print(num);
    // Also check white's queen moves after black moves
    let board2 = apply_move(new_board, 60, 52);
    let w_moves = legal_moves(board2, 1);
    let w_num = len(w_moves) / 2;
    // White queen on a8 should have many sliding moves
    let queen_moves = 0;
    let qi = 0;
    while qi < w_num {
        if w_moves[qi * 2] == 56 {
            queen_moves = queen_moves + 1;
        }
        qi = qi + 1;
    }
    print(queen_moves);
    "#,
    );
    let out = run_mir(&src, 42);
    let black_moves: i64 = out[0].trim().parse().unwrap();
    assert!(black_moves > 0, "Black should have moves after promotion");
    let queen_moves: i64 = out[1].trim().parse().unwrap();
    assert!(queen_moves > 5, "Promoted queen should have many moves, got {}", queen_moves);
}

// Test 7: Pawn does not promote on non-promotion rank
#[test]
fn pawn_no_promotion_on_non_last_rank() {
    // White pawn on e6 (sq 44) moves to e7 (sq 52) - not rank 7 target, so no promotion
    // Wait - rank_of(52) = 52/8 = 6, not 7. So no promotion.
    let src = custom_board_program(
        &[(4, 6), (44, 1), (60, -6)],
        r#"
    let new_board = apply_move(board, 44, 52);
    print(new_board[52]);
    "#,
    );
    let out = run_mir(&src, 42);
    assert_eq!(out[0].trim(), "1", "Pawn on rank 6 (not promotion rank) should remain a pawn");
}

// Test 8: Promotion is deterministic across seeds
#[test]
fn promotion_deterministic() {
    let src = custom_board_program(
        &[(4, 6), (52, 1), (56, -6)],
        r#"
    let new_board = apply_move(board, 52, 60);
    print(new_board[60]);
    "#,
    );
    let out1 = run_mir(&src, 1);
    let out2 = run_mir(&src, 999);
    assert_eq!(out1[0].trim(), out2[0].trim(), "Promotion result must be seed-independent");
    assert_eq!(out1[0].trim(), "5", "Must promote to queen");
}

// Test 9: Multiple promotions in a game
#[test]
fn multiple_promotions() {
    // Two white pawns ready to promote: a7 (sq 48) and h7 (sq 55)
    // White king e1 (sq 4), black king e5 (sq 36)
    let src = custom_board_program(
        &[(4, 6), (48, 1), (55, 1), (36, -6)],
        r#"
    let b1 = apply_move(board, 48, 56);
    print(b1[56]);
    // Black makes a king move
    let b2 = apply_move(b1, 36, 44);
    // Second pawn promotes
    let b3 = apply_move(b2, 55, 63);
    print(b3[63]);
    // Count white queens
    let qcount = 0;
    let i = 0;
    while i < 64 {
        if b3[i] == 5 {
            qcount = qcount + 1;
        }
        i = i + 1;
    }
    print(qcount);
    "#,
    );
    let out = run_mir(&src, 42);
    assert_eq!(out[0].trim(), "5", "First promotion should produce queen");
    assert_eq!(out[1].trim(), "5", "Second promotion should produce queen");
    assert_eq!(out[2].trim(), "2", "Should have 2 white queens");
}
