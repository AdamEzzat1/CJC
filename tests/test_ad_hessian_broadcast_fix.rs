//! `.cjcl`-level regression tests for two GradGraph bugs, run through both
//! executors (cjc-eval and cjc-mir-exec) with output parity asserted.
//!
//! * P1 — `GradGraph.hessian_diag` returned stale values for composed losses
//!   because intermediate nodes kept their pre-perturbation forward values.
//! * P2 — backward did not reduce broadcast gradients to the input's shape.
//!
//! Expected values come from closed-form derivatives or from finite
//! differences of a plain-f64 re-implementation of each loss, never from the
//! code under test.

fn run_eval(src: &str, seed: u64) -> Vec<String> {
    let (program, diag) = cjc_parser::parse_source(src);
    assert!(!diag.has_errors(), "parse errors: {:?}", diag.diagnostics);
    let mut interp = cjc_eval::Interpreter::new(seed);
    interp
        .exec(&program)
        .unwrap_or_else(|e| panic!("eval failed: {e:?}\nsource:\n{src}"));
    interp.output
}

fn run_mir(src: &str, seed: u64) -> Vec<String> {
    let (program, _diag) = cjc_parser::parse_source(src);
    let (_, executor) = cjc_mir_exec::run_program_with_executor(&program, seed)
        .unwrap_or_else(|e| panic!("MIR-exec failed: {e}\nsource:\n{src}"));
    executor.output
}

/// Run through both executors, assert identical output, return it.
fn parity(src: &str) -> Vec<String> {
    let eval_out = run_eval(src, 42);
    let mir_out = run_mir(src, 42);
    assert_eq!(eval_out, mir_out, "Parity mismatch!\nEval: {eval_out:?}\nMIR:  {mir_out:?}");
    eval_out
}

const DUMP: &str = r#"
fn dump(t: Any) {
    print(t.shape());
    let v = t.to_vec();
    let i = 0;
    while i < len(v) {
        print(v[i]);
        i = i + 1;
    }
}
"#;

/// Parse output produced by `dump` calls: a shape line `[a, b]` followed by
/// `prod(shape)` float lines. Returns (shape, values) per dumped tensor.
fn parse_dumps(lines: &[String]) -> Vec<(Vec<usize>, Vec<f64>)> {
    let mut out = Vec::new();
    let mut k = 0;
    while k < lines.len() {
        let s = lines[k].trim();
        assert!(s.starts_with('[') && s.ends_with(']'), "expected shape line, got {s:?}");
        let shape: Vec<usize> = s[1..s.len() - 1]
            .split(',')
            .map(|d| d.trim().parse().expect("shape dim"))
            .collect();
        let n: usize = shape.iter().product();
        let vals: Vec<f64> = lines[k + 1..k + 1 + n]
            .iter()
            .map(|v| v.trim().parse().expect("float"))
            .collect();
        out.push((shape, vals));
        k += 1 + n;
    }
    out
}

fn lit(v: &[f64]) -> String {
    let parts: Vec<String> = v.iter().map(|x| format!("{x:?}")).collect();
    format!("[{}]", parts.join(", "))
}

fn shape_lit(s: &[usize]) -> String {
    let parts: Vec<String> = s.iter().map(|x| x.to_string()).collect();
    format!("[{}]", parts.join(", "))
}

fn close(got: f64, want: f64, tol: f64) -> bool {
    (got - want).abs() <= tol * (1.0 + want.abs())
}

fn assert_close(label: &str, got: &[f64], want: &[f64], tol: f64) {
    assert_eq!(got.len(), want.len(), "{label}: len {} vs {} (got {got:?})", got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(close(g, w, tol), "{label}[{i}]: got {g}, want {w}; all got {got:?} want {want:?}");
    }
}

// ── P1: hessian_diag ──────────────────────────────────────────────────

fn hessian_prog(p0: &[f64], body: &str) -> String {
    format!(
        "{DUMP}\nfn main() {{\n    let g = GradGraph.new();\n    let p = g.parameter(Tensor.from_vec({}, [{}]));\n{body}\n    g.backward(loss);\n    dump(g.hessian_diag(loss, p));\n    dump(g.hessian_diag(loss, p, 0.00001));\n}}\n",
        lit(p0),
        p0.len()
    )
}

fn check_hessian(label: &str, p0: &[f64], body: &str, want: &[f64]) {
    let out = parity(&hessian_prog(p0, body));
    let dumps = parse_dumps(&out);
    assert_eq!(dumps.len(), 2, "{label}: {out:?}");
    for (shape, vals) in &dumps {
        assert_eq!(shape, &vec![p0.len()], "{label}: shape");
        assert_close(label, vals, want, 1e-6);
    }
}

#[test]
fn hessian_diag_p4_both_executors() {
    // loss = (p*p)*(p*p); d2/dp2 = 12 p^2
    for &p0 in &[1.0_f64, 1.5, -0.7] {
        check_hessian(
            "p^4",
            &[p0],
            "    let y = g.mul(p, p);\n    let loss = g.mul(y, y);",
            &[12.0 * p0 * p0],
        );
    }
}

#[test]
fn hessian_diag_exp_sin_both_executors() {
    // loss = sum(exp(sin(p))); d2/dp2 = e^{sin p}(cos^2 p - sin p)
    let p0 = [0.3_f64, 0.7, -1.2];
    let want: Vec<f64> = p0.iter().map(|&x| x.sin().exp() * (x.cos().powi(2) - x.sin())).collect();
    check_hessian(
        "exp(sin(p))",
        &p0,
        "    let s = g.sin(p);\n    let e = g.exp(s);\n    let loss = g.sum(e);",
        &want,
    );
}

#[test]
fn hessian_diag_chain_intermediate_both_executors() {
    // u = p*w + sin(p); loss = sum(u*u)
    // d2/dp2 = 2(w + cos p)^2 - 2 u sin p
    let p0 = [0.4_f64, -0.9];
    let w0 = [2.0_f64, -3.0];
    let want: Vec<f64> = p0
        .iter()
        .zip(&w0)
        .map(|(&x, &w)| {
            let u = x * w + x.sin();
            2.0 * (w + x.cos()).powi(2) - 2.0 * u * x.sin()
        })
        .collect();
    let body = format!(
        "    let w = g.input(Tensor.from_vec({}, [2]));\n    let pw = g.mul(p, w);\n    let sp = g.sin(p);\n    let u = g.add(pw, sp);\n    let uu = g.mul(u, u);\n    let loss = g.sum(uu);",
        lit(&w0)
    );
    check_hessian("chain", &p0, &body, &want);
}

#[test]
fn hessian_diag_probe_p1_both_executors() {
    // Verbatim shape of docs/julia_comparison/probes/p1_hessian_diag_stale.cjcl
    let src = format!(
        "{DUMP}\nfn main() {{\n    let g = GradGraph.new();\n    let p = g.parameter(Tensor.from_vec([1.0], [1]));\n    let y = g.mul(p, p);\n    let loss = g.mul(y, y);\n    g.backward(loss);\n    dump(g.hessian_diag(loss, p));\n    print(g.value(y));\n    print(g.value(loss));\n    let g2 = GradGraph.new();\n    let q = g2.parameter(Tensor.from_vec([1.0], [1]));\n    let l2 = g2.mul(q, q);\n    g2.backward(l2);\n    dump(g2.hessian_diag(l2, q));\n}}\n"
    );
    let out = parity(&src);
    // dump(hd) = 2 lines, then value(y), value(loss), then dump(hd2) = 2 lines.
    assert_eq!(out.len(), 6, "{out:?}");
    let hd: f64 = out[1].parse().unwrap();
    assert!(close(hd, 12.0, 1e-6), "p^4 hessian_diag = {hd}");
    // Forward values restored after hessian_diag.
    assert_eq!(out[2], "1");
    assert_eq!(out[3], "1");
    let hd2: f64 = out[5].parse().unwrap();
    assert!(close(hd2, 2.0, 1e-6), "p^2 hessian_diag = {hd2}");
}

// ── P2: broadcast gradients ───────────────────────────────────────────

const N: usize = 2;
const M: usize = 3;

fn apply(op: &str, a: f64, b: f64) -> f64 {
    match op {
        "add" => a + b,
        "sub" => a - b,
        "mul" => a * b,
        "div" => a / b,
        _ => unreachable!(),
    }
}

fn bread(data: &[f64], shape: &[usize], r: usize, c: usize) -> f64 {
    match shape {
        [1] => data[0],
        [m] => data[c % m],
        [1, m] => data[c % m],
        [n, 1] => data[r % n],
        [_, m] => data[r * m + c],
        _ => unreachable!(),
    }
}

fn plain_loss(op: &str, l: &[f64], ls: &[usize], r: &[f64], rs: &[usize], w: &[f64]) -> f64 {
    let mut acc = 0.0;
    for i in 0..N {
        for j in 0..M {
            acc += w[i * M + j] * apply(op, bread(l, ls, i, j), bread(r, rs, i, j));
        }
    }
    acc
}

fn fd_grad(f: &dyn Fn(&[f64]) -> f64, v: &[f64]) -> Vec<f64> {
    let h = 1e-6;
    (0..v.len())
        .map(|i| {
            let mut a = v.to_vec();
            a[i] += h;
            let mut b = v.to_vec();
            b[i] -= h;
            (f(&a) - f(&b)) / (2.0 * h)
        })
        .collect()
}

fn data_for(shape: &[usize], base: f64) -> Vec<f64> {
    let n: usize = shape.iter().product();
    (0..n).map(|k| base + 0.37 * k as f64).collect()
}

fn broadcast_prog(op: &str, ld: &[f64], ls: &[usize], rd: &[f64], rs: &[usize], w: &[f64]) -> String {
    format!(
        "{DUMP}\nfn main() {{\n    let g = GradGraph.new();\n    let l = g.parameter(Tensor.from_vec({}, {}));\n    let r = g.parameter(Tensor.from_vec({}, {}));\n    let y = g.{op}(l, r);\n    let w = g.input(Tensor.from_vec({}, [{N}, {M}]));\n    let yw = g.mul(y, w);\n    let loss = g.sum(yw);\n    g.backward(loss);\n    dump(g.grad(l));\n    dump(g.grad(r));\n}}\n",
        lit(ld),
        shape_lit(ls),
        lit(rd),
        shape_lit(rs),
        lit(w)
    )
}

/// Returns failure descriptions so a red run reports every combination.
fn check_broadcast(op: &str) -> Vec<String> {
    let mut failures = Vec::new();
    let big_shape = vec![N, M];
    let big = data_for(&big_shape, 1.1);
    let w: Vec<f64> = (0..N * M).map(|k| 0.5 + 0.25 * k as f64).collect();
    for small_shape in [vec![M], vec![1, M], vec![N, 1], vec![1]] {
        let small = data_for(&small_shape, 1.7);
        for big_left in [true, false] {
            let (ld, ls, rd, rs) = if big_left {
                (big.clone(), big_shape.clone(), small.clone(), small_shape.clone())
            } else {
                (small.clone(), small_shape.clone(), big.clone(), big_shape.clone())
            };
            let label = format!("{op} {ls:?} {rs:?}");
            let src = broadcast_prog(op, &ld, &ls, &rd, &rs, &w);
            let out = parity(&src);
            // Determinism: a second run of each executor is identical.
            assert_eq!(out, run_eval(&src, 42), "{label}: eval not deterministic");
            assert_eq!(out, run_mir(&src, 42), "{label}: mir not deterministic");
            let dumps = parse_dumps(&out);
            let fl = |v: &[f64]| plain_loss(op, v, &ls, &rd, &rs, &w);
            let fr = |v: &[f64]| plain_loss(op, &ld, &ls, v, &rs, &w);
            for (side, (shape, vals), want_shape, want) in [
                ("left", &dumps[0], &ls, fd_grad(&fl, &ld)),
                ("right", &dumps[1], &rs, fd_grad(&fr, &rd)),
            ] {
                if shape != want_shape {
                    failures.push(format!("{label}: grad({side}) shape {shape:?}, want {want_shape:?}"));
                }
                if vals.len() != want.len() || vals.iter().zip(&want).any(|(&a, &b)| !close(a, b, 1e-6)) {
                    failures.push(format!("{label}: grad({side}) = {vals:?}, want {want:?}"));
                }
            }
        }
    }
    failures
}

#[test]
fn broadcast_add_both_executors() {
    let f = check_broadcast("add");
    assert!(f.is_empty(), "{} failures:\n{}", f.len(), f.join("\n"));
}

#[test]
fn broadcast_sub_both_executors() {
    let f = check_broadcast("sub");
    assert!(f.is_empty(), "{} failures:\n{}", f.len(), f.join("\n"));
}

#[test]
fn broadcast_mul_both_executors() {
    let f = check_broadcast("mul");
    assert!(f.is_empty(), "{} failures:\n{}", f.len(), f.join("\n"));
}

#[test]
fn broadcast_div_both_executors() {
    let f = check_broadcast("div");
    assert!(f.is_empty(), "{} failures:\n{}", f.len(), f.join("\n"));
}

#[test]
fn broadcast_probe_p2_both_executors() {
    // Same graphs as docs/julia_comparison/probes/p2_broadcast_backward.cjcl
    let src = format!(
        "{DUMP}\nfn main() {{\n    let g = GradGraph.new();\n    let x = g.input(Tensor.from_vec([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]));\n    let b = g.parameter(Tensor.from_vec([0.0, 0.0, 0.0], [3]));\n    let s = g.add(x, b);\n    let loss = g.sum(s);\n    g.backward(loss);\n    dump(g.grad(b));\n    let g2 = GradGraph.new();\n    let x2 = g2.input(Tensor.from_vec([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]));\n    let b2 = g2.parameter(Tensor.from_vec([0.0, 0.0, 0.0], [1, 3]));\n    let s2 = g2.add(x2, b2);\n    let loss2 = g2.sum(s2);\n    g2.backward(loss2);\n    dump(g2.grad(b2));\n}}\n"
    );
    let d = parse_dumps(&parity(&src));
    assert_eq!(d[0], (vec![3], vec![2.0, 2.0, 2.0]));
    assert_eq!(d[1], (vec![1, 3], vec![2.0, 2.0, 2.0]));
}

#[test]
fn xor_mlp_bias_keeps_shape_both_executors() {
    // The training loop of examples/01_mlp_xor.cjcl (fixed weights instead of
    // randn, fewer epochs) printing b1/b2 shapes every epoch. Before the fix
    // b1 silently grew from [1,4] to [4,4] after the first update.
    let src = r#"
fn main() {
    let X = Tensor.from_vec([0.0, 0.0,  0.0, 1.0,  1.0, 0.0,  1.0, 1.0], [4, 2]);
    let Y = Tensor.from_vec([0.0, 1.0, 1.0, 0.0], [4, 1]);
    let W1 = Tensor.from_vec([0.3, -0.2, 0.5, 0.1, -0.4, 0.25, 0.15, -0.35], [2, 4]);
    let b1 = Tensor.zeros([1, 4]);
    let W2 = Tensor.from_vec([0.2, -0.3, 0.4, 0.1], [4, 1]);
    let b2 = Tensor.zeros([1, 1]);
    let lr = 1.0;
    let epoch = 0;
    while epoch < 4 {
        let g = GradGraph.new();
        let w1_id = g.parameter(W1);
        let b1_id = g.parameter(b1);
        let w2_id = g.parameter(W2);
        let b2_id = g.parameter(b2);
        let x_id = g.input(X);
        let y_id = g.input(Y);
        let z1 = g.matmul(x_id, w1_id);
        let z1b = g.add(z1, b1_id);
        let h = g.sigmoid(z1b);
        let z2 = g.matmul(h, w2_id);
        let z2b = g.add(z2, b2_id);
        let pred = g.sigmoid(z2b);
        let diff = g.sub(pred, y_id);
        let sq = g.mul(diff, diff);
        let loss = g.mean(sq);
        g.backward(loss);
        let gb1 = g.grad(b1_id);
        let gb2 = g.grad(b2_id);
        print(gb1.shape());
        print(gb2.shape());
        W1 = W1 - g.grad(w1_id) * lr;
        b1 = b1 - gb1 * lr;
        W2 = W2 - g.grad(w2_id) * lr;
        b2 = b2 - gb2 * lr;
        print(b1.shape());
        print(b2.shape());
        print(g.value(loss));
        epoch = epoch + 1;
    }
}
"#;
    let out = parity(src);
    assert_eq!(out.len(), 20, "{out:?}");
    for e in 0..4 {
        assert_eq!(out[e * 5], "[1, 4]", "epoch {e} grad(b1) shape: {out:?}");
        assert_eq!(out[e * 5 + 1], "[1, 1]", "epoch {e} grad(b2) shape: {out:?}");
        assert_eq!(out[e * 5 + 2], "[1, 4]", "epoch {e} b1 shape: {out:?}");
        assert_eq!(out[e * 5 + 3], "[1, 1]", "epoch {e} b2 shape: {out:?}");
    }
    // Determinism across repeated runs.
    assert_eq!(out, run_eval(src, 42));
    assert_eq!(out, run_mir(src, 42));
}

#[test]
fn xor_mlp_bias_grad_matches_finite_difference() {
    // One forward/backward of the XOR MLP; grad(b1) [1,4] compared with a
    // central finite difference of a plain-f64 implementation of the loss.
    let x = [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];
    let y = [0.0, 1.0, 1.0, 0.0];
    let w1 = [0.3, -0.2, 0.5, 0.1, -0.4, 0.25, 0.15, -0.35];
    let w2 = [0.2, -0.3, 0.4, 0.1];
    let b1 = [0.05, -0.1, 0.2, 0.0];
    let b2 = 0.1;
    let sig = |z: f64| 1.0 / (1.0 + (-z).exp());
    let loss = |b1: &[f64]| {
        let mut acc = 0.0;
        for r in 0..4 {
            let mut out = b2;
            for j in 0..4 {
                let z = x[r * 2] * w1[j] + x[r * 2 + 1] * w1[4 + j] + b1[j];
                out += sig(z) * w2[j];
            }
            let d = sig(out) - y[r];
            acc += d * d;
        }
        acc / 4.0
    };
    let want = fd_grad(&loss, &b1);
    let src = format!(
        "{DUMP}\nfn main() {{\n    let g = GradGraph.new();\n    let w1 = g.parameter(Tensor.from_vec({}, [2, 4]));\n    let b1 = g.parameter(Tensor.from_vec({}, [1, 4]));\n    let w2 = g.parameter(Tensor.from_vec({}, [4, 1]));\n    let b2 = g.parameter(Tensor.from_vec([{b2:?}], [1, 1]));\n    let x = g.input(Tensor.from_vec({}, [4, 2]));\n    let y = g.input(Tensor.from_vec({}, [4, 1]));\n    let z1 = g.matmul(x, w1);\n    let z1b = g.add(z1, b1);\n    let h = g.sigmoid(z1b);\n    let z2 = g.matmul(h, w2);\n    let z2b = g.add(z2, b2);\n    let pred = g.sigmoid(z2b);\n    let diff = g.sub(pred, y);\n    let sq = g.mul(diff, diff);\n    let loss = g.mean(sq);\n    g.backward(loss);\n    dump(g.grad(b1));\n}}\n",
        lit(&w1),
        lit(&b1),
        lit(&w2),
        lit(&x),
        lit(&y)
    );
    let d = parse_dumps(&parity(&src));
    assert_eq!(d[0].0, vec![1, 4]);
    assert_close("xor grad(b1)", &d[0].1, &want, 1e-6);
}
