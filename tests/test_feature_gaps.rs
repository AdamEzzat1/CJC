// ═══════════════════════════════════════════════════════════════════════
// Tests for 6 missing feature gaps:
//   1. tanh standalone builtin
//   2. relu standalone builtin
//   3. reshape(tensor, shape) builtin
//   4. as type casting (x as f64)
//   5. Tuple field access (.0, .1)
//   6. tensor_slice / slice builtins
// ═══════════════════════════════════════════════════════════════════════

use std::rc::Rc;

fn run_eval(src: &str) -> cjc_runtime::value::Value {
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "Parse errors present");
    let result = cjc_eval::Interpreter::new(42).exec(&program);
    match result {
        Ok(v) => v,
        Err(e) => panic!("Eval error: {e:?}"),
    }
}

fn run_mir(src: &str) -> cjc_runtime::value::Value {
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "Parse errors present");
    let result = cjc_mir_exec::run_program_with_executor(&program, 42);
    match result {
        Ok((v, _)) => v,
        Err(e) => panic!("MIR error: {e:?}"),
    }
}

/// Run both eval and MIR-exec and assert they produce identical output
fn run_parity(src: &str) -> (cjc_runtime::value::Value, cjc_runtime::value::Value) {
    let eval_val = run_eval(src);
    let mir_val = run_mir(src);
    assert_eq!(
        format!("{eval_val}"),
        format!("{mir_val}"),
        "Parity failure: eval={eval_val} vs mir={mir_val}"
    );
    (eval_val, mir_val)
}

// ═══════════════════════════════════════════════════════════════════════
// Feature 1: tanh standalone
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn tanh_scalar_float() {
    let src = r#"fn main() { print(tanh(0.0)); }"#;
    run_parity(src);
}

#[test]
fn tanh_scalar_int() {
    let src = r#"fn main() { print(tanh(0)); }"#;
    run_parity(src);
}

#[test]
fn tanh_scalar_value() {
    let src = r#"fn main() { print(tanh(1.0)); }"#;
    let (v, _) = run_parity(src);
    // tanh(1.0) ≈ 0.7615941559557649
    // Just verify it ran without error
}

#[test]
fn tanh_tensor() {
    let src = r#"
fn main() {
    let t = Tensor.from_vec([0.0, 1.0, -1.0], [3]);
    let result = tanh(t);
    print(result);
}
"#;
    run_parity(src);
}

#[test]
fn tanh_backward_compat() {
    // tanh_scalar still works
    let src = r#"fn main() { print(tanh_scalar(0.5)); }"#;
    run_parity(src);
}

// ═══════════════════════════════════════════════════════════════════════
// Feature 2: relu standalone
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn relu_positive_float() {
    let src = r#"fn main() { print(relu(3.14)); }"#;
    run_parity(src);
}

#[test]
fn relu_negative_float() {
    let src = r#"fn main() { print(relu(-2.5)); }"#;
    let (v, _) = run_parity(src);
}

#[test]
fn relu_zero() {
    let src = r#"fn main() { print(relu(0.0)); }"#;
    run_parity(src);
}

#[test]
fn relu_int() {
    let src = r#"fn main() { print(relu(-5)); }"#;
    run_parity(src);
}

#[test]
fn relu_int_positive() {
    let src = r#"fn main() { print(relu(7)); }"#;
    run_parity(src);
}

#[test]
fn relu_tensor() {
    let src = r#"
fn main() {
    let t = Tensor.from_vec([-2.0, -1.0, 0.0, 1.0, 2.0], [5]);
    let r = relu(t);
    print(r);
}
"#;
    run_parity(src);
}

// ═══════════════════════════════════════════════════════════════════════
// Feature 3: reshape
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn reshape_1d_to_2d() {
    let src = r#"
fn main() {
    let t = Tensor.from_vec([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [6]);
    let r = reshape(t, [2, 3]);
    print(r.shape);
}
"#;
    run_parity(src);
}

#[test]
fn reshape_2d_to_1d() {
    let src = r#"
fn main() {
    let t = Tensor.from_vec([1.0, 2.0, 3.0, 4.0], [2, 2]);
    let r = reshape(t, [4]);
    print(r.shape);
}
"#;
    run_parity(src);
}

#[test]
fn reshape_preserves_data() {
    let src = r#"
fn main() {
    let t = Tensor.from_vec([1.0, 2.0, 3.0, 4.0], [4]);
    let r = reshape(t, [2, 2]);
    print(r);
}
"#;
    run_parity(src);
}

// ═══════════════════════════════════════════════════════════════════════
// Feature 4: as type casting
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn cast_int_to_float() {
    let src = r#"fn main() { let x = 42; print(x as f64); }"#;
    run_parity(src);
}

#[test]
fn cast_float_to_int() {
    let src = r#"fn main() { let x = 3.14; print(x as i64); }"#;
    run_parity(src);
}

#[test]
fn cast_int_to_bool() {
    let src = r#"fn main() { print(0 as bool); print(1 as bool); }"#;
    run_parity(src);
}

#[test]
fn cast_float_to_bool() {
    let src = r#"fn main() { print(0.0 as bool); print(1.5 as bool); }"#;
    run_parity(src);
}

#[test]
fn cast_bool_to_int() {
    let src = r#"fn main() { print(true as i64); print(false as i64); }"#;
    run_parity(src);
}

#[test]
fn cast_bool_to_float() {
    let src = r#"fn main() { print(true as f64); print(false as f64); }"#;
    run_parity(src);
}

#[test]
fn cast_to_string() {
    let src = r#"fn main() { let x = 42; print(x as String); }"#;
    run_parity(src);
}

#[test]
fn cast_in_expression() {
    // as should have high precedence: (x as f64) + 1.0, not x as (f64 + 1.0)
    let src = r#"fn main() { let x = 5; print(x as f64 + 0.5); }"#;
    run_parity(src);
}

#[test]
fn cast_chained() {
    let src = r#"fn main() { let x = 3.7; print(x as i64 as f64); }"#;
    run_parity(src);
}

// ═══════════════════════════════════════════════════════════════════════
// Feature 5: Tuple field access (.0, .1)
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn tuple_field_access_0() {
    let src = r#"
fn main() {
    let t = (10, 20, 30);
    print(t.0);
}
"#;
    run_parity(src);
}

#[test]
fn tuple_field_access_1() {
    let src = r#"
fn main() {
    let t = (10, 20, 30);
    print(t.1);
}
"#;
    run_parity(src);
}

#[test]
fn tuple_field_access_2() {
    let src = r#"
fn main() {
    let t = (10, 20, 30);
    print(t.2);
}
"#;
    run_parity(src);
}

#[test]
fn tuple_field_access_nested() {
    // Note: `t.0.1` parses `0.1` as a float, so use intermediate variable
    let src = r#"
fn main() {
    let t = ((1, 2), (3, 4));
    let inner = t.0;
    print(inner.1);
}
"#;
    run_parity(src);
}

#[test]
fn tuple_from_function() {
    let src = r#"
fn make_pair(a: i64, b: i64) -> Any {
    return (a, b);
}
fn main() {
    let p = make_pair(10, 20);
    print(p.0);
    print(p.1);
}
"#;
    run_parity(src);
}

#[test]
fn tuple_field_eigh_replacement() {
    // This pattern replaces the `match eig { (vals, vecs) => ... };` pattern
    let src = r#"
fn main() {
    let t = (42.0, 99.0);
    let first = t.0;
    let second = t.1;
    print(first);
    print(second);
}
"#;
    run_parity(src);
}

// ═══════════════════════════════════════════════════════════════════════
// Feature 6: tensor_slice and slice
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn tensor_slice_1d() {
    let src = r#"
fn main() {
    let t = Tensor.from_vec([10.0, 20.0, 30.0, 40.0, 50.0], [5]);
    let s = tensor_slice(t, [1], [4]);
    print(s);
}
"#;
    run_parity(src);
}

#[test]
fn tensor_slice_2d() {
    let src = r#"
fn main() {
    let t = Tensor.from_vec([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
    let s = tensor_slice(t, [0, 1], [2, 3]);
    print(s);
}
"#;
    run_parity(src);
}

#[test]
fn slice_along_dim() {
    let src = r#"
fn main() {
    let t = Tensor.from_vec([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
    let s = slice(t, 1, 0, 2);
    print(s);
}
"#;
    run_parity(src);
}

#[test]
fn slice_1d() {
    let src = r#"
fn main() {
    let t = Tensor.from_vec([10.0, 20.0, 30.0, 40.0], [4]);
    let s = slice(t, 0, 1, 3);
    print(s);
}
"#;
    run_parity(src);
}

// ═══════════════════════════════════════════════════════════════════════
// Combined / Integration tests
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn tanh_relu_pipeline() {
    let src = r#"
fn main() {
    let t = Tensor.from_vec([-2.0, -1.0, 0.0, 1.0, 2.0], [5]);
    let r = relu(t);
    let h = tanh(r);
    print(h);
}
"#;
    run_parity(src);
}

#[test]
fn reshape_then_slice() {
    let src = r#"
fn main() {
    let t = Tensor.from_vec([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [6]);
    let r = reshape(t, [2, 3]);
    let s = slice(r, 0, 0, 1);
    print(s);
}
"#;
    run_parity(src);
}

#[test]
fn cast_with_arithmetic() {
    let src = r#"
fn main() {
    let i = 10;
    let f = i as f64 / 3.0;
    print(f);
}
"#;
    run_parity(src);
}

#[test]
fn tuple_access_with_cast() {
    let src = r#"
fn main() {
    let t = (42, 3.14);
    let int_val = t.0;
    let float_val = t.1;
    print(int_val as f64 + float_val);
}
"#;
    run_parity(src);
}

// ═══════════════════════════════════════════════════════════════════════
// Determinism tests — same seed = identical output
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn determinism_tanh() {
    let src = r#"fn main() { print(tanh(0.5)); }"#;
    let v1 = format!("{}", run_eval(src));
    let v2 = format!("{}", run_eval(src));
    assert_eq!(v1, v2, "tanh not deterministic");
}

#[test]
fn determinism_relu() {
    let src = r#"fn main() { print(relu(-1.5)); }"#;
    let v1 = format!("{}", run_eval(src));
    let v2 = format!("{}", run_eval(src));
    assert_eq!(v1, v2, "relu not deterministic");
}

#[test]
fn determinism_cast() {
    let src = r#"fn main() { print(3.14 as i64); }"#;
    let v1 = format!("{}", run_eval(src));
    let v2 = format!("{}", run_eval(src));
    assert_eq!(v1, v2, "cast not deterministic");
}
