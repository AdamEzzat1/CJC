use super::helpers::*;

// ── Bitwise AND (&) ──

#[test]
fn bitwise_and_basic() {
    let out = run_mir(r#"
        let a: i64 = 0xFF;
        let b: i64 = 0x0F;
        print(bit_and(a, b));
    "#);
    assert_eq!(out[0], "15"); // 0x0F = 15
}

#[test]
fn bitwise_and_zero() {
    let out = run_mir(r#"
        print(bit_and(42, 0));
    "#);
    assert_eq!(out[0], "0");
}

// ── Bitwise OR (|) ──

#[test]
fn bitwise_or_basic() {
    let out = run_mir(r#"
        print(bit_or(0xF0, 0x0F));
    "#);
    assert_eq!(out[0], "255"); // 0xFF = 255
}

// ── Bitwise XOR (^) ──

#[test]
fn bitwise_xor_basic() {
    let out = run_mir(r#"
        print(bit_xor(0xFF, 0x0F));
    "#);
    assert_eq!(out[0], "240"); // 0xF0 = 240
}

#[test]
fn bitwise_xor_self() {
    let out = run_mir(r#"
        print(bit_xor(42, 42));
    "#);
    assert_eq!(out[0], "0");
}

// ── Bitwise NOT (~) ──

#[test]
fn bitwise_not_basic() {
    let out = run_mir(r#"
        print(bit_not(0));
    "#);
    assert_eq!(out[0], "-1"); // !0 = -1 in two's complement
}

// ── Bitwise shift left (<<) ──

#[test]
fn bitwise_shl_basic() {
    let out = run_mir(r#"
        print(bit_shl(1, 8));
    "#);
    assert_eq!(out[0], "256"); // 1 << 8 = 256
}

#[test]
fn bitwise_shl_multiply() {
    let out = run_mir(r#"
        print(bit_shl(5, 3));
    "#);
    assert_eq!(out[0], "40"); // 5 << 3 = 40
}

// ── Bitwise shift right (>>) ──

#[test]
fn bitwise_shr_basic() {
    let out = run_mir(r#"
        print(bit_shr(256, 4));
    "#);
    assert_eq!(out[0], "16"); // 256 >> 4 = 16
}

#[test]
fn bitwise_shr_divide() {
    let out = run_mir(r#"
        print(bit_shr(100, 2));
    "#);
    assert_eq!(out[0], "25"); // 100 >> 2 = 25
}

// ── Bitwise popcount ──

#[test]
fn bitwise_popcount() {
    let out = run_mir(r#"
        print(popcount(0xFF));
    "#);
    assert_eq!(out[0], "8");
}

#[test]
fn bitwise_popcount_zero() {
    let out = run_mir(r#"
        print(popcount(0));
    "#);
    assert_eq!(out[0], "0");
}

// ── Combined bitwise operations ──

#[test]
fn bitwise_mask_extract() {
    // Extract bits 4-7 from a value
    let out = run_mir(r#"
        let val: i64 = 0xABCD;
        let mask: i64 = 0xF0;
        let extracted: i64 = bit_shr(bit_and(val, mask), 4);
        print(extracted);
    "#);
    // 0xABCD & 0xF0 = 0xC0 = 192, 192 >> 4 = 12
    assert_eq!(out[0], "12");
}
