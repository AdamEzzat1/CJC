use super::helpers::*;

// ── stop_gradient ──

#[test]
fn stop_gradient_passthrough_float() {
    let out = run_mir(r#"
        let x: f64 = 3.14;
        let y: f64 = stop_gradient(x);
        print(y);
    "#);
    assert_close(parse_float(&out[0]), 3.14, 1e-10);
}

#[test]
fn stop_gradient_passthrough_int() {
    let out = run_mir(r#"
        let x: i64 = 42;
        let y: i64 = stop_gradient(x);
        print(y);
    "#);
    assert_eq!(out[0], "42");
}

// ── grad_checkpoint ──

#[test]
fn grad_checkpoint_passthrough() {
    let out = run_mir(r#"
        let x: f64 = 2.718;
        let y: f64 = grad_checkpoint(x);
        print(y);
    "#);
    assert_close(parse_float(&out[0]), 2.718, 1e-10);
}

// ── clip_grad ──

#[test]
fn clip_grad_within_range() {
    let out = run_mir(r#"
        let x: f64 = clip_grad(0.5, -1.0, 1.0);
        print(x);
    "#);
    assert_close(parse_float(&out[0]), 0.5, 1e-10);
}

#[test]
fn clip_grad_above_max() {
    let out = run_mir(r#"
        let x: f64 = clip_grad(5.0, -1.0, 1.0);
        print(x);
    "#);
    assert_close(parse_float(&out[0]), 1.0, 1e-10);
}

#[test]
fn clip_grad_below_min() {
    let out = run_mir(r#"
        let x: f64 = clip_grad(-5.0, -1.0, 1.0);
        print(x);
    "#);
    assert_close(parse_float(&out[0]), -1.0, 1e-10);
}

#[test]
fn clip_grad_at_boundary() {
    let out = run_mir(r#"
        let x: f64 = clip_grad(1.0, -1.0, 1.0);
        print(x);
    "#);
    assert_close(parse_float(&out[0]), 1.0, 1e-10);
}

// ── grad_scale ──

#[test]
fn grad_scale_basic() {
    let out = run_mir(r#"
        let x: f64 = grad_scale(3.0, 2.0);
        print(x);
    "#);
    assert_close(parse_float(&out[0]), 6.0, 1e-10);
}

#[test]
fn grad_scale_fractional() {
    let out = run_mir(r#"
        let x: f64 = grad_scale(10.0, 0.1);
        print(x);
    "#);
    assert_close(parse_float(&out[0]), 1.0, 1e-10);
}

#[test]
fn grad_scale_negative() {
    let out = run_mir(r#"
        let x: f64 = grad_scale(5.0, -1.0);
        print(x);
    "#);
    assert_close(parse_float(&out[0]), -5.0, 1e-10);
}

#[test]
fn grad_scale_zero() {
    let out = run_mir(r#"
        let x: f64 = grad_scale(100.0, 0.0);
        print(x);
    "#);
    assert_close(parse_float(&out[0]), 0.0, 1e-10);
}

// ── Combined ML workflow ──

#[test]
fn ml_workflow_combined() {
    let out = run_mir(r#"
        let lr: f64 = 0.01;
        let gradient: f64 = 5.5;
        let clipped: f64 = clip_grad(gradient, -1.0, 1.0);
        let scaled: f64 = grad_scale(clipped, lr);
        let update: f64 = stop_gradient(scaled);
        print(update);
    "#);
    assert_close(parse_float(&out[0]), 0.01, 1e-10);
}
