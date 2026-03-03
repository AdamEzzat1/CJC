use super::helpers::*;

// ── Compound Assignment: += ──

#[test]
fn compound_add_int() {
    let out = run_mir(r#"
        let mut x: i64 = 10;
        x += 5;
        print(x);
    "#);
    assert_eq!(out[0], "15");
}

#[test]
fn compound_add_float() {
    let out = run_mir(r#"
        let mut x: f64 = 1.5;
        x += 2.5;
        print(x);
    "#);
    assert_close(parse_float(&out[0]), 4.0, 1e-10);
}

// ── Compound Assignment: -= ──

#[test]
fn compound_sub_int() {
    let out = run_mir(r#"
        let mut x: i64 = 20;
        x -= 7;
        print(x);
    "#);
    assert_eq!(out[0], "13");
}

#[test]
fn compound_sub_float() {
    let out = run_mir(r#"
        let mut x: f64 = 10.0;
        x -= 3.5;
        print(x);
    "#);
    assert_close(parse_float(&out[0]), 6.5, 1e-10);
}

// ── Compound Assignment: *= ──

#[test]
fn compound_mul_int() {
    let out = run_mir(r#"
        let mut x: i64 = 6;
        x *= 7;
        print(x);
    "#);
    assert_eq!(out[0], "42");
}

#[test]
fn compound_mul_float() {
    let out = run_mir(r#"
        let mut x: f64 = 2.5;
        x *= 4.0;
        print(x);
    "#);
    assert_close(parse_float(&out[0]), 10.0, 1e-10);
}

// ── Compound Assignment: /= ──

#[test]
fn compound_div_int() {
    let out = run_mir(r#"
        let mut x: i64 = 100;
        x /= 4;
        print(x);
    "#);
    assert_eq!(out[0], "25");
}

#[test]
fn compound_div_float() {
    let out = run_mir(r#"
        let mut x: f64 = 9.0;
        x /= 2.0;
        print(x);
    "#);
    assert_close(parse_float(&out[0]), 4.5, 1e-10);
}

// ── Compound Assignment: %= ──

#[test]
fn compound_mod_int() {
    let out = run_mir(r#"
        let mut x: i64 = 17;
        x %= 5;
        print(x);
    "#);
    assert_eq!(out[0], "2");
}

// ── Chained compound assignments ──

#[test]
fn compound_chained() {
    let out = run_mir(r#"
        let mut x: i64 = 10;
        x += 5;
        x *= 2;
        x -= 3;
        print(x);
    "#);
    assert_eq!(out[0], "27");
}

// ── Compound in loop ──

#[test]
fn compound_in_loop() {
    let out = run_mir(r#"
        let mut sum: i64 = 0;
        for i in 1..6 {
            sum += i;
        }
        print(sum);
    "#);
    assert_eq!(out[0], "15");
}
