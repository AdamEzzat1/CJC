use super::helpers::*;

// ── If as expression: basic ──

#[test]
fn if_expr_basic_true() {
    let out = run_mir(r#"
        fn pick(x: i64) -> i64 {
            if x > 0 { 1 } else { 0 }
        }
        print(pick(5));
    "#);
    assert_eq!(out[0], "1");
}

#[test]
fn if_expr_basic_false() {
    let out = run_mir(r#"
        fn pick(x: i64) -> i64 {
            if x > 0 { 1 } else { 0 }
        }
        print(pick(-3));
    "#);
    assert_eq!(out[0], "0");
}

// ── If expression in let binding ──

#[test]
fn if_expr_in_let() {
    let out = run_eval(r#"
        let x: i64 = 10;
        let result: i64 = if x > 5 { 100 } else { 200 };
        print(result);
    "#);
    assert_eq!(out[0], "100");
}

#[test]
fn if_expr_in_let_false_branch() {
    let out = run_eval(r#"
        let x: i64 = 3;
        let result: i64 = if x > 5 { 100 } else { 200 };
        print(result);
    "#);
    assert_eq!(out[0], "200");
}

// ── Nested if expression ──

#[test]
fn if_expr_nested() {
    let out = run_eval(r#"
        fn classify(x: i64) -> i64 {
            if x > 10 { 3 } else { if x > 5 { 2 } else { 1 } }
        }
        print(classify(15));
        print(classify(7));
        print(classify(2));
    "#);
    assert_eq!(out[0], "3");
    assert_eq!(out[1], "2");
    assert_eq!(out[2], "1");
}

// ── If expression with else-if ──

#[test]
fn if_expr_else_if() {
    let out = run_eval(r#"
        fn grade(score: i64) -> i64 {
            if score >= 90 { 4 } else if score >= 80 { 3 } else if score >= 70 { 2 } else { 1 }
        }
        print(grade(95));
        print(grade(85));
        print(grade(75));
        print(grade(50));
    "#);
    assert_eq!(out[0], "4");
    assert_eq!(out[1], "3");
    assert_eq!(out[2], "2");
    assert_eq!(out[3], "1");
}

// ── If expression with float result ──

#[test]
fn if_expr_float() {
    let out = run_eval(r#"
        let x: f64 = 3.14;
        let y: f64 = if x > 3.0 { x * 2.0 } else { x / 2.0 };
        print(y);
    "#);
    assert_close(parse_float(&out[0]), 6.28, 1e-10);
}
