use super::helpers::*;

/// Parity tests: verify that eval and MIR-exec produce identical output.

#[test]
fn parity_compound_add() {
    let src = r#"
        let mut x: i64 = 10;
        x += 5;
        print(x);
    "#;
    assert_eq!(run_eval(src), run_mir(src));
}

#[test]
fn parity_compound_mul() {
    let src = r#"
        let mut x: f64 = 2.5;
        x *= 4.0;
        print(x);
    "#;
    assert_eq!(run_eval(src), run_mir(src));
}

#[test]
fn parity_compound_loop() {
    let src = r#"
        let mut sum: i64 = 0;
        for i in 1..11 {
            sum += i;
        }
        print(sum);
    "#;
    assert_eq!(run_eval(src), run_mir(src));
}

#[test]
fn parity_pow_builtin() {
    let src = r#"
        print(pow(2.0, 10.0));
    "#;
    assert_eq!(run_eval(src), run_mir(src));
}

#[test]
fn parity_hex_literal() {
    let src = r#"
        print(0xFF);
        print(0b1010);
        print(0o77);
    "#;
    assert_eq!(run_eval(src), run_mir(src));
}

#[test]
fn parity_stop_gradient() {
    let src = r#"
        let x: f64 = 3.14;
        print(stop_gradient(x));
    "#;
    assert_eq!(run_eval(src), run_mir(src));
}

#[test]
fn parity_clip_grad() {
    let src = r#"
        print(clip_grad(5.0, -1.0, 1.0));
        print(clip_grad(-5.0, -1.0, 1.0));
        print(clip_grad(0.5, -1.0, 1.0));
    "#;
    assert_eq!(run_eval(src), run_mir(src));
}

#[test]
fn parity_grad_scale() {
    let src = r#"
        print(grad_scale(3.0, 2.0));
    "#;
    assert_eq!(run_eval(src), run_mir(src));
}

#[test]
fn parity_bitwise_ops() {
    let src = r#"
        print(bit_and(0xFF, 0x0F));
        print(bit_or(0xF0, 0x0F));
        print(bit_xor(0xFF, 0x0F));
        print(bit_not(0));
        print(bit_shl(1, 8));
        print(bit_shr(256, 4));
    "#;
    assert_eq!(run_eval(src), run_mir(src));
}

#[test]
fn parity_if_expr() {
    let src = r#"
        fn pick(x: i64) -> i64 {
            if x > 0 { 1 } else { 0 }
        }
        print(pick(5));
        print(pick(-3));
    "#;
    assert_eq!(run_eval(src), run_mir(src));
}

#[test]
fn parity_all_mixed() {
    let src = r#"
        let mut x: i64 = 0xFF;
        x += 1;
        let y: i64 = pow(2, 8);
        print(x == y);
        print(bit_and(x, 0xFF));
    "#;
    assert_eq!(run_eval(src), run_mir(src));
}
