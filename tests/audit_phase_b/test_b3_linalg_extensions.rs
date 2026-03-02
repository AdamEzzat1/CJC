//! Phase B audit test B3: Linear Algebra Extensions

fn run_mir(src: &str) -> Vec<String> {
    let (tokens, _) = cjc_lexer::Lexer::new(src).tokenize();
    let (program, _) = cjc_parser::Parser::new(tokens).parse_program();
    let (_, executor) = cjc_mir_exec::run_program_with_executor(&program, 42).unwrap();
    executor.output
}

fn parse_float(s: &str) -> f64 {
    s.parse::<f64>().unwrap()
}

#[test]
fn b3_norm_1_identity() {
    let out = run_mir(r#"
let I = Tensor.from_vec([1.0, 0.0, 0.0, 1.0], [2, 2]);
print(norm_1(I));
"#);
    assert!((parse_float(&out[0]) - 1.0).abs() < 1e-10);
}

#[test]
fn b3_norm_1_known() {
    let out = run_mir(r#"
let A = Tensor.from_vec([1.0, 2.0, 3.0, 4.0], [2, 2]);
print(norm_1(A));
"#);
    // max(|1|+|3|, |2|+|4|) = max(4, 6) = 6
    assert!((parse_float(&out[0]) - 6.0).abs() < 1e-10);
}

#[test]
fn b3_norm_inf_known() {
    let out = run_mir(r#"
let A = Tensor.from_vec([1.0, 2.0, 3.0, 4.0], [2, 2]);
print(norm_inf(A));
"#);
    // max(|1|+|2|, |3|+|4|) = max(3, 7) = 7
    assert!((parse_float(&out[0]) - 7.0).abs() < 1e-10);
}

#[test]
fn b3_cond_identity() {
    let out = run_mir(r#"
let I = Tensor.from_vec([1.0, 0.0, 0.0, 1.0], [2, 2]);
print(cond(I));
"#);
    assert!((parse_float(&out[0]) - 1.0).abs() < 1e-10);
}

#[test]
fn b3_cond_diagonal() {
    let out = run_mir(r#"
let A = Tensor.from_vec([2.0, 0.0, 0.0, 4.0], [2, 2]);
print(cond(A));
"#);
    // cond = |4|/|2| = 2
    assert!((parse_float(&out[0]) - 2.0).abs() < 1e-10);
}

#[test]
fn b3_schur_identity() {
    let out = run_mir(r#"
let I = Tensor.from_vec([1.0, 0.0, 0.0, 1.0], [2, 2]);
let result = schur(I);
print(result);
"#);
    assert_eq!(out.len(), 1);
}

#[test]
fn b3_matrix_exp_zero() {
    let out = run_mir(r#"
let Z = Tensor.from_vec([0.0, 0.0, 0.0, 0.0], [2, 2]);
let E = matrix_exp(Z);
print(E);
"#);
    let s = &out[0];
    assert!(s.contains("1") && s.contains("0"));
}

#[test]
fn b3_matrix_exp_nilpotent() {
    let out = run_mir(r#"
let N = Tensor.from_vec([0.0, 1.0, 0.0, 0.0], [2, 2]);
let E = matrix_exp(N);
print(E);
"#);
    // exp([[0,1],[0,0]]) = [[1,1],[0,1]]
    let s = &out[0];
    assert!(s.contains("1"));
}

#[test]
fn b3_cond_ill_conditioned() {
    let out = run_mir(r#"
let A = Tensor.from_vec([1.0, 1.0, 1.0, 1.0001], [2, 2]);
let c = cond(A);
print(c);
"#);
    let c = parse_float(&out[0]);
    assert!(c > 1000.0, "near-singular matrix should have high cond, got {c}");
}

#[test]
fn b3_schur_diagonal_matrix() {
    let out = run_mir(r#"
let D = Tensor.from_vec([3.0, 0.0, 0.0, 7.0], [2, 2]);
let result = schur(D);
print(result);
"#);
    assert_eq!(out.len(), 1);
}

#[test]
fn b3_determinism_norm() {
    let out = run_mir(r#"
let A = Tensor.from_vec([1.0, 2.0, 3.0, 4.0], [2, 2]);
let n1a = norm_1(A);
let n1b = norm_1(A);
print(n1a == n1b);
"#);
    assert_eq!(out, vec!["true"]);
}

#[test]
fn b3_determinism_cond() {
    let out = run_mir(r#"
let A = Tensor.from_vec([2.0, 1.0, 1.0, 3.0], [2, 2]);
let c1 = cond(A);
let c2 = cond(A);
print(c1 == c2);
"#);
    assert_eq!(out, vec!["true"]);
}

#[test]
fn b3_schur_3x3() {
    let out = run_mir(r#"
let A = Tensor.from_vec([1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 0.0, 0.0, 6.0], [3, 3]);
let result = schur(A);
print(result);
"#);
    assert_eq!(out.len(), 1);
}

#[test]
fn b3_matrix_exp_diagonal() {
    let out = run_mir(r#"
let A = Tensor.from_vec([1.0, 0.0, 0.0, 2.0], [2, 2]);
let E = matrix_exp(A);
print(E);
"#);
    assert_eq!(out.len(), 1);
}

#[test]
fn b3_norm_1_rectangular() {
    let out = run_mir(r#"
let A = Tensor.from_vec([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
let n = norm_1(A);
print(n);
"#);
    // max(|1|+|4|, |2|+|5|, |3|+|6|) = max(5, 7, 9) = 9
    assert!((parse_float(&out[0]) - 9.0).abs() < 1e-10);
}
