//! Hardening test H16: ML Toolkit (losses, metrics) via MIR executor

fn run_mir(src: &str) -> Vec<String> {
    let (tokens, _) = cjc_lexer::Lexer::new(src).tokenize();
    let (program, _) = cjc_parser::Parser::new(tokens).parse_program();
    let (_, executor) = cjc_mir_exec::run_program_with_executor(&program, 42).unwrap();
    executor.output
}

#[test]
fn h16_mse_loss_nonzero() {
    let out = run_mir(r#"
let pred = [1.0, 2.0, 3.0];
let target = [1.5, 2.5, 3.5];
print(mse_loss(pred, target));
"#);
    let parsed: f64 = out[0].parse().unwrap();
    assert!((parsed - 0.25).abs() < 1e-10, "expected 0.25, got {parsed}");
}

#[test]
fn h16_cross_entropy() {
    let out = run_mir(r#"
let pred = [0.9, 0.05, 0.05];
let target = [1.0, 0.0, 0.0];
let loss = cross_entropy_loss(pred, target);
print(loss);
"#);
    let parsed: f64 = out[0].parse().unwrap();
    assert!(parsed > 0.0 && parsed < 1.0, "cross_entropy should be small for good pred, got {parsed}");
}

#[test]
fn h16_binary_cross_entropy() {
    let out = run_mir(r#"
let pred = [0.9, 0.1];
let target = [1.0, 0.0];
let loss = binary_cross_entropy(pred, target);
print(loss);
"#);
    let parsed: f64 = out[0].parse().unwrap();
    assert!(parsed > 0.0 && parsed < 0.5, "binary CE should be small, got {parsed}");
}

#[test]
fn h16_huber_loss_small() {
    // When errors < delta, huber = mse/2
    let out = run_mir(r#"
let pred = [1.0, 2.0];
let target = [1.1, 2.1];
let loss = huber_loss(pred, target, 1.0);
print(loss);
"#);
    let parsed: f64 = out[0].parse().unwrap();
    assert!(parsed > 0.0 && parsed < 0.01, "expected small huber loss, got {parsed}");
}

#[test]
fn h16_hinge_loss() {
    let out = run_mir(r#"
let pred = [2.0, -1.0];
let target = [1.0, -1.0];
let loss = hinge_loss(pred, target);
print(loss);
"#);
    let parsed: f64 = out[0].parse().unwrap();
    assert!(parsed >= 0.0, "hinge loss should be non-negative, got {parsed}");
}

#[test]
fn h16_confusion_matrix() {
    let out = run_mir(r#"
let pred = [1.0, 1.0, 0.0, 0.0];
let actual = [1.0, 0.0, 0.0, 1.0];
let cm = confusion_matrix(pred, actual);
print(cm.tp);
print(cm.accuracy);
"#);
    assert_eq!(out[0], "1");
    let acc: f64 = out[1].parse().unwrap();
    assert!((acc - 0.5).abs() < 1e-10);
}

#[test]
fn h16_confusion_matrix_perfect() {
    let out = run_mir(r#"
let pred = [1.0, 1.0, 0.0, 0.0];
let actual = [1.0, 1.0, 0.0, 0.0];
let cm = confusion_matrix(pred, actual);
print(cm.precision);
print(cm.recall);
print(cm.f1_score);
"#);
    let prec: f64 = out[0].parse().unwrap();
    let rec: f64 = out[1].parse().unwrap();
    let f1: f64 = out[2].parse().unwrap();
    assert!((prec - 1.0).abs() < 1e-10, "precision should be 1.0");
    assert!((rec - 1.0).abs() < 1e-10, "recall should be 1.0");
    assert!((f1 - 1.0).abs() < 1e-10, "f1 should be 1.0");
}

#[test]
fn h16_auc_roc_perfect() {
    let out = run_mir(r#"
let scores = [0.9, 0.8, 0.1, 0.05];
let labels = [1.0, 1.0, 0.0, 0.0];
let auc = auc_roc(scores, labels);
print(auc);
"#);
    let parsed: f64 = out[0].parse().unwrap();
    assert!((parsed - 1.0).abs() < 1e-10, "perfect AUC should be 1.0, got {parsed}");
}

#[test]
fn h16_auc_roc_separated() {
    // Scores with some separation but not perfect
    let out = run_mir(r#"
let scores = [0.6, 0.4, 0.55, 0.35];
let labels = [1.0, 0.0, 1.0, 0.0];
let auc = auc_roc(scores, labels);
print(auc);
"#);
    let parsed: f64 = out[0].parse().unwrap();
    assert!((parsed - 1.0).abs() < 1e-10, "well-separated scores should give AUC=1.0, got {parsed}");
}

#[test]
fn h16_determinism() {
    let src = r#"
let pred = [0.9, 0.1, 0.3];
let target = [1.0, 0.0, 0.0];
print(mse_loss(pred, target));
print(cross_entropy_loss(pred, target));
let scores = [0.9, 0.1];
let labels = [1.0, 0.0];
print(auc_roc(scores, labels));
"#;
    let out1 = run_mir(src);
    let out2 = run_mir(src);
    assert_eq!(out1, out2);
}
