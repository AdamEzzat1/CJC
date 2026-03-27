//! Integration tests for reduction analysis — end-to-end from CJC source code.

use cjc_mir::reduction::{detect_reductions, ReductionKind, ReductionOp};

fn reductions_from_source(src: &str) -> cjc_mir::reduction::ReductionReport {
    let (program, diags) = cjc_parser::parse_source(src);
    if diags.has_errors() {
        panic!("CJC parse failed for:\n{}\n", src);
    }
    let mir = cjc_mir_exec::lower_to_mir(&program);
    detect_reductions(&mir, &[])
}

#[test]
fn test_source_sum_reduction() {
    let src = r#"
fn sum_array(n: i64) -> i64 {
    let mut total: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        total = total + i;
        i = i + 1;
    }
    return total;
}
print(sum_array(5));
"#;
    let report = reductions_from_source(src);
    let sum_reds: Vec<_> = report
        .reductions
        .iter()
        .filter(|r| r.accumulator_var == "total" && r.op == ReductionOp::Add)
        .collect();
    assert!(
        !sum_reds.is_empty(),
        "should detect total = total + i as a reduction"
    );
    assert_eq!(sum_reds[0].kind, ReductionKind::StrictFold);
}

#[test]
fn test_source_product_reduction() {
    let src = r#"
fn factorial(n: i64) -> i64 {
    let mut result: i64 = 1;
    let mut i: i64 = 1;
    while i <= n {
        result = result * i;
        i = i + 1;
    }
    return result;
}
print(factorial(5));
"#;
    let report = reductions_from_source(src);
    let prod_reds: Vec<_> = report
        .reductions
        .iter()
        .filter(|r| r.accumulator_var == "result" && r.op == ReductionOp::Mul)
        .collect();
    assert!(
        !prod_reds.is_empty(),
        "should detect result = result * i as a reduction"
    );
    assert_eq!(prod_reds[0].kind, ReductionKind::StrictFold);
}

#[test]
fn test_source_builtin_sum() {
    let src = r#"
let arr = [1, 2, 3, 4, 5];
let total = sum(arr);
print(total);
"#;
    let report = reductions_from_source(src);
    let builtin_reds: Vec<_> = report
        .reductions
        .iter()
        .filter(|r| r.kind == ReductionKind::BuiltinReduction)
        .collect();
    assert!(
        !builtin_reds.is_empty(),
        "should detect sum() as builtin reduction"
    );
    assert_eq!(builtin_reds[0].builtin_name.as_deref(), Some("sum"));
}

#[test]
fn test_source_builtin_mean() {
    let src = r#"
let data = [1.0, 2.0, 3.0];
let avg = mean(data);
print(avg);
"#;
    let report = reductions_from_source(src);
    let mean_reds: Vec<_> = report
        .reductions
        .iter()
        .filter(|r| r.builtin_name.as_deref() == Some("mean"))
        .collect();
    assert!(!mean_reds.is_empty(), "should detect mean() call");
}

#[test]
fn test_source_no_false_positives() {
    let src = r#"
fn add(a: i64, b: i64) -> i64 { a + b }
let x: i64 = add(3, 4);
print(x);
"#;
    let report = reductions_from_source(src);
    assert!(
        report.is_empty(),
        "simple add function should not have reductions, found {}",
        report.len()
    );
}

#[test]
fn test_source_multiple_reductions() {
    let src = r#"
fn stats(n: i64) -> i64 {
    let mut sum_val: i64 = 0;
    let mut prod_val: i64 = 1;
    let mut i: i64 = 0;
    while i < n {
        sum_val = sum_val + i;
        prod_val = prod_val * i;
        i = i + 1;
    }
    return sum_val + prod_val;
}
print(stats(5));
"#;
    let report = reductions_from_source(src);
    let add_reds: Vec<_> = report
        .reductions
        .iter()
        .filter(|r| r.accumulator_var == "sum_val" && r.op == ReductionOp::Add)
        .collect();
    let mul_reds: Vec<_> = report
        .reductions
        .iter()
        .filter(|r| r.accumulator_var == "prod_val" && r.op == ReductionOp::Mul)
        .collect();
    assert!(!add_reds.is_empty(), "should detect sum_val accumulation");
    assert!(!mul_reds.is_empty(), "should detect prod_val accumulation");
}

#[test]
fn test_source_reduction_determinism() {
    let src = r#"
fn compute(n: i64) -> i64 {
    let mut acc: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        acc = acc + 1;
        i = i + 1;
    }
    let s = sum([1, 2, 3]);
    return acc;
}
print(compute(5));
"#;
    let report1 = reductions_from_source(src);
    let report2 = reductions_from_source(src);
    assert_eq!(report1.len(), report2.len());
    for (a, b) in report1.reductions.iter().zip(report2.reductions.iter()) {
        assert_eq!(a.id, b.id);
        assert_eq!(a.accumulator_var, b.accumulator_var);
        assert_eq!(a.op, b.op);
        assert_eq!(a.kind, b.kind);
    }
}

#[test]
fn test_strict_fold_not_reorderable() {
    let src = r#"
fn accumulate(n: i64) -> i64 {
    let mut total: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        total = total + i;
        i = i + 1;
    }
    return total;
}
print(accumulate(5));
"#;
    let report = reductions_from_source(src);
    for r in &report.reductions {
        if r.kind == ReductionKind::StrictFold {
            assert!(!r.kind.is_reorderable(), "StrictFold must not be reorderable");
            assert!(!r.kind.is_parallelizable(), "StrictFold must not be parallelizable");
            assert!(r.kind.is_strict(), "StrictFold must be strict");
        }
    }
}
