//! Frozen snapshot of the 8-program `bench/cana_pass_ordering` corpus,
//! used by §3A.3 (cross-corpus validation) of the cost-model handoff.
//!
//! The pass_ordering programs were hand-written by a different author
//! for a different purpose (comparing default vs CANA-recommended pass
//! plans). They make a useful *external* test set for the trained cost
//! model — neither the model's coefficients nor the training corpus
//! have seen these programs.
//!
//! Provenance: copied verbatim from `bench/cana_pass_ordering/main.rs`
//! at the time of §3A.3 implementation. Drift between this snapshot
//! and the source is intentional — cross-corpus evaluation requires a
//! frozen reference. If pass_ordering's programs change in the future,
//! this snapshot stays as-is to keep historical ext_rmse numbers
//! comparable.

use crate::programs::Program;

/// 1. Constant arithmetic — CF + DCE candidate.
const PROG_ARITH: &str = r#"
fn compute(n: i64) -> i64 {
    let a: i64 = 10 * 5 + 2;
    let b: i64 = (a + 100) * 2;
    let c: i64 = b - 50 + n;
    return c + a + b;
}
print(compute(7));
"#;

/// 2. Loop-heavy — LICM matters.
const PROG_LOOP: &str = r#"
fn sum_to(n: i64) -> i64 {
    let mut total: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        total = total + i;
        i = i + 1;
    }
    return total;
}
print(sum_to(1000));
"#;

/// 3. Nested loops.
const PROG_NESTED: &str = r#"
fn nested(n: i64) -> i64 {
    let mut total: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        let mut j: i64 = 0;
        while j < n {
            total = total + i * j;
            j = j + 1;
        }
        i = i + 1;
    }
    return total;
}
print(nested(30));
"#;

/// 4. Many small functions — exercises per-function dispatch.
const PROG_MANY_FN: &str = r#"
fn add1(x: i64) -> i64 { return x + 1; }
fn add2(x: i64) -> i64 { return x + 2; }
fn add3(x: i64) -> i64 { return x + 3; }
fn mul2(x: i64) -> i64 { return x * 2; }
fn mul3(x: i64) -> i64 { return x * 3; }
fn driver() -> i64 {
    let mut r: i64 = 0;
    r = add1(r);
    r = add2(r);
    r = add3(r);
    r = mul2(r);
    r = mul3(r);
    return r;
}
print(driver());
"#;

/// 5. Branchy arithmetic — if/else inside a loop.
const PROG_MIXED: &str = r#"
fn classify(n: i64) -> i64 {
    let mut sum: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        let inc: i64 = if i * 2 > n { i } else { 0 };
        sum = sum + inc;
        i = i + 1;
    }
    return sum;
}
print(classify(40));
"#;

/// 6. Float arithmetic — exercises reduction-axis tracking.
const PROG_FLOAT: &str = r#"
fn polynomial(x: f64) -> f64 {
    let a: f64 = 3.14;
    let b: f64 = 2.71;
    let c: f64 = 1.41;
    return a * x * x + b * x + c;
}
print(polynomial(1.5));
"#;

/// 7. Recursive — call-graph exercise.
const PROG_RECURSIVE: &str = r#"
fn factorial(n: i64) -> i64 {
    let result: i64 = if n <= 1 { 1 } else { n * factorial(n - 1) };
    return result;
}
print(factorial(10));
"#;

/// 8. Larger combined program — multiple loops + functions.
const PROG_LARGE: &str = r#"
fn count_evens(n: i64) -> i64 {
    let mut c: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        if i * 2 / 2 == i {
            c = c + 1;
        }
        i = i + 1;
    }
    return c;
}
fn count_squares(n: i64) -> i64 {
    let mut c: i64 = 0;
    let mut i: i64 = 0;
    while i * i < n {
        c = c + 1;
        i = i + 1;
    }
    return c;
}
fn sum_to(n: i64) -> i64 {
    let mut s: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        s = s + i;
        i = i + 1;
    }
    return s;
}
fn combined(n: i64) -> i64 {
    let a: i64 = count_evens(n);
    let b: i64 = count_squares(n);
    let c: i64 = sum_to(n);
    return a + b + c;
}
print(combined(50));
"#;

pub const EXTERNAL_PROGRAMS: &[Program] = &[
    Program { name: "ext_arith",     source: PROG_ARITH,     expected_dominant_pass: "constant_fold" },
    Program { name: "ext_loop",      source: PROG_LOOP,      expected_dominant_pass: "licm" },
    Program { name: "ext_nested",    source: PROG_NESTED,    expected_dominant_pass: "licm" },
    Program { name: "ext_many_fn",   source: PROG_MANY_FN,   expected_dominant_pass: "dce" },
    Program { name: "ext_mixed",     source: PROG_MIXED,     expected_dominant_pass: "cse" },
    Program { name: "ext_float",     source: PROG_FLOAT,     expected_dominant_pass: "cse" },
    Program { name: "ext_recursive", source: PROG_RECURSIVE, expected_dominant_pass: "constant_fold" },
    Program { name: "ext_large",     source: PROG_LARGE,     expected_dominant_pass: "licm" },
];
