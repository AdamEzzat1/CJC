use super::helpers::*;

// ── Hex literals ──

#[test]
fn hex_literal_basic() {
    let out = run_mir(r#"
        print(0xFF);
    "#);
    assert_eq!(out[0], "255");
}

#[test]
fn hex_literal_upper() {
    let out = run_mir(r#"
        print(0XFF);
    "#);
    assert_eq!(out[0], "255");
}

#[test]
fn hex_literal_mixed_case() {
    let out = run_mir(r#"
        print(0xAbCd);
    "#);
    assert_eq!(out[0], "43981");
}

#[test]
fn hex_literal_with_underscores() {
    let out = run_mir(r#"
        print(0xFF_FF);
    "#);
    assert_eq!(out[0], "65535");
}

#[test]
fn hex_literal_zero() {
    let out = run_mir(r#"
        print(0x0);
    "#);
    assert_eq!(out[0], "0");
}

// ── Binary literals ──

#[test]
fn binary_literal_basic() {
    let out = run_mir(r#"
        print(0b1010);
    "#);
    assert_eq!(out[0], "10");
}

#[test]
fn binary_literal_byte() {
    let out = run_mir(r#"
        print(0b11111111);
    "#);
    assert_eq!(out[0], "255");
}

#[test]
fn binary_literal_with_underscores() {
    let out = run_mir(r#"
        print(0b1111_0000);
    "#);
    assert_eq!(out[0], "240");
}

#[test]
fn binary_literal_one() {
    let out = run_mir(r#"
        print(0b1);
    "#);
    assert_eq!(out[0], "1");
}

// ── Octal literals ──

#[test]
fn octal_literal_basic() {
    let out = run_mir(r#"
        print(0o777);
    "#);
    assert_eq!(out[0], "511");
}

#[test]
fn octal_literal_small() {
    let out = run_mir(r#"
        print(0o10);
    "#);
    assert_eq!(out[0], "8");
}

#[test]
fn octal_literal_with_underscores() {
    let out = run_mir(r#"
        print(0o77_77);
    "#);
    assert_eq!(out[0], "4095");
}

// ── Arithmetic with non-decimal literals ──

#[test]
fn hex_arithmetic() {
    let out = run_mir(r#"
        let a: i64 = 0xFF;
        let b: i64 = 0x01;
        print(a + b);
    "#);
    assert_eq!(out[0], "256");
}

#[test]
fn binary_arithmetic() {
    let out = run_mir(r#"
        let a: i64 = 0b1100;
        let b: i64 = 0b0011;
        print(a + b);
    "#);
    assert_eq!(out[0], "15");
}

#[test]
fn mixed_bases() {
    let out = run_mir(r#"
        let dec: i64 = 10;
        let hex: i64 = 0xA;
        let bin: i64 = 0b1010;
        let oct: i64 = 0o12;
        print(dec == hex);
        print(hex == bin);
        print(bin == oct);
    "#);
    assert_eq!(out[0], "true");
    assert_eq!(out[1], "true");
    assert_eq!(out[2], "true");
}
