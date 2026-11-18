//! Postulate Phase 1 exit gates (see `docs/postulate/PHASE1.md`).
//!
//! Each gate is a checker that returns `Err` with a reason, plus a test that
//! runs it on real executor output and a poison test proving it can fail.

use cjc_ad::export::{GraphExport, SourceSpan};

/// The shipped example, unmodified: a 2-4-1 MLP with broadcast biases.
const XOR: &str = include_str!("../examples/01_mlp_xor.cjcl");
/// `cnn_example.cjcl` in a form that runs (its `Tensor.conv2d(…)` and
/// `GradGraph.input(g, …)` calls exist in neither executor at HEAD).
const CNN: &str = include_str!("postulate/cnn_off_tape.cjcl");

// ── harness ─────────────────────────────────────────────────────────────

fn eval_export(src: &str, seed: u64) -> (Option<GraphExport>, Vec<String>) {
    let (program, _diag) = cjc_parser::parse_source(src);
    let mut interp = cjc_eval::Interpreter::new(seed);
    interp.ad.enabled = true;
    interp.exec(&program).unwrap_or_else(|e| panic!("eval failed: {e:?}"));
    (interp.ad.export(), interp.output.clone())
}

fn mir_export(src: &str, seed: u64) -> (Option<GraphExport>, Vec<String>) {
    let (program, _diag) = cjc_parser::parse_source(src);
    let (_, ex) = cjc_mir_exec::run_program_with_ad(&program, seed)
        .unwrap_or_else(|e| panic!("mir-exec failed: {e}"));
    (ex.ad.export(), ex.output.clone())
}

// ── Gate 1: spans cover the exact call that created each node ───────────

/// Independent oracle: the `g.<op>(…)` calls in the training loop, in source
/// order, as byte ranges. Each one records exactly one tape node in this
/// program (`g.mean` of `g.mul(d, d)` fuses into one `MseMean` node; the
/// `sub` and `mul` it reads are the two calls before it).
fn xor_expected_spans(src: &str) -> Vec<(usize, usize)> {
    const RECORDING: &[&str] = &[
        "parameter", "input", "matmul", "add", "sigmoid", "sub", "mul", "mean",
    ];
    let mut out = Vec::new();
    let mut i = 0;
    while let Some(off) = src[i..].find("g.") {
        let start = i + off;
        i = start + 2;
        let prev_ok = start == 0 || !src.as_bytes()[start - 1].is_ascii_alphanumeric();
        let name_end = start + 2 + src[start + 2..].find('(').unwrap_or(0);
        let name = &src[start + 2..name_end];
        if !prev_ok || !RECORDING.contains(&name) {
            continue;
        }
        // match parentheses to the end of the call
        let mut depth = 0usize;
        let mut end = name_end;
        for (k, c) in src[name_end..].char_indices() {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        end = name_end + k + 1;
                        break;
                    }
                }
                _ => {}
            }
        }
        out.push((start, end));
    }
    out
}

fn check_spans(src: &str, e: &GraphExport) -> Result<(), String> {
    let want = xor_expected_spans(src);
    if e.nodes.len() != want.len() {
        return Err(format!("{} nodes, oracle has {} calls", e.nodes.len(), want.len()));
    }
    for (n, (s, t)) in e.nodes.iter().zip(&want) {
        let got = n.span.ok_or_else(|| format!("node {} ({}) has no span", n.id, n.op))?;
        if (got.start as usize, got.end as usize) != (*s, *t) || got.file != 0 {
            return Err(format!(
                "node {} ({}) span {}..{} = {:?}, want {}..{} = {:?}",
                n.id,
                n.op,
                got.start,
                got.end,
                src.get(got.start as usize..got.end as usize),
                s,
                t,
                &src[*s..*t]
            ));
        }
    }
    Ok(())
}

#[test]
fn gate_spans_cover_the_creating_call_eval() {
    let (e, _) = eval_export(XOR, 42);
    let e = e.expect("XOR trains a graph");
    check_spans(XOR, &e).unwrap();
}

#[test]
fn gate_spans_cover_the_creating_call_mir() {
    let (e, _) = mir_export(XOR, 42);
    let e = e.expect("XOR trains a graph");
    check_spans(XOR, &e).unwrap();
}

#[test]
fn poison_span_shifted_by_one_byte_fails() {
    let (e, _) = eval_export(XOR, 42);
    let mut e = e.unwrap();
    let s = e.nodes[6].span.unwrap();
    e.nodes[6].span = Some(SourceSpan { start: s.start + 1, ..s });
    let err = check_spans(XOR, &e).unwrap_err();
    assert!(err.contains("node 6"), "{err}");
}

// ── Gate 2: both executors export byte-identical GraphExport JSON ───────

fn check_parity(a: &Option<GraphExport>, b: &Option<GraphExport>) -> Result<(), String> {
    let ja = a.as_ref().map(GraphExport::to_json);
    let jb = b.as_ref().map(GraphExport::to_json);
    if ja != jb {
        return Err(format!("eval:\n{ja:?}\nmir:\n{jb:?}"));
    }
    Ok(())
}

#[test]
fn gate_executor_parity_xor() {
    let (a, oa) = eval_export(XOR, 42);
    let (b, ob) = mir_export(XOR, 42);
    assert_eq!(oa, ob, "program output differs");
    check_parity(&a, &b).unwrap();
}

#[test]
fn gate_executor_parity_cnn_off_tape() {
    let (a, _) = eval_export(CNN, 7);
    let (b, _) = mir_export(CNN, 7);
    check_parity(&a, &b).unwrap();
    // Both registered tensors feed only the off-tape conv: both are flagged,
    // with the conv's line.
    let e = a.unwrap();
    let flagged: Vec<_> = e.off_tape.iter().map(|t| (t.op.as_str(), t.node)).collect();
    assert_eq!(flagged, [("Tensor.conv2d", 0), ("Tensor.conv2d", 1)]);
    let sp = e.off_tape[0].span.unwrap();
    assert_eq!(&CNN[sp.start as usize..sp.end as usize], "x_val.conv2d(k_val, conv_bias)");
}

#[test]
fn poison_executor_parity_one_bit_fails() {
    let (a, _) = eval_export(XOR, 42);
    let mut b = a.clone();
    let n = b.as_mut().unwrap().nodes.iter_mut().find(|n| n.grad_norm.is_some()).unwrap();
    n.grad_norm = Some(f64::from_bits(n.grad_norm.unwrap().to_bits() ^ 1));
    assert!(check_parity(&a, &b).is_err());
}

// ── layer grouping and off-tape ─────────────────────────────────────────

#[test]
fn xor_groups_by_enclosing_function() {
    let (e, _) = eval_export(XOR, 42);
    let e = e.unwrap();
    assert_eq!(e.groups.len(), 1);
    assert_eq!(e.groups[0].name, "main");
    assert_eq!(e.groups[0].nodes.len(), e.nodes.len());
    // SGD updates and the eval pass read trained parameters off the tape, but
    // those parameters feed the tape, so none of it is flagged.
    assert!(e.off_tape.is_empty(), "{:?}", e.off_tape);
}

#[test]
fn program_named_layers_override_function_groups() {
    let src = r#"
fn main() {
    let g = GradGraph.new();
    let w = g.parameter(Tensor.from_vec([1.0, 2.0], [2, 1]));
    let x = g.input(Tensor.from_vec([3.0, 4.0], [1, 2]));
    g.layer("hidden");
    let h = g.matmul(x, w);
    let a = g.sigmoid(h);
    g.layer("");
    let l = g.sum(a);
    g.backward(l);
}
"#;
    for e in [eval_export(src, 1).0, mir_export(src, 1).0] {
        let e = e.unwrap();
        let names: Vec<_> = e.groups.iter().map(|g| (g.name.as_str(), g.nodes.clone())).collect();
        assert_eq!(names, [("main", vec![0, 1, 4]), ("hidden", vec![2, 3])]);
    }
}

#[test]
fn layers_named_inside_a_called_function_group_its_ops() {
    for e in [eval_export(TANH_FIT, 9).0, mir_export(TANH_FIT, 9).0] {
        let e = e.unwrap();
        let names: Vec<_> = e.groups.iter().map(|g| (g.name.as_str(), g.nodes.len())).collect();
        // main: 4 registrations + sub, mul, mean; hidden: matmul, tanh; out: matmul.
        assert_eq!(names, [("main", 7), ("hidden", 2), ("out", 1)]);
    }
    let (a, _) = eval_export(TANH_FIT, 9);
    let (b, _) = mir_export(TANH_FIT, 9);
    check_parity(&a, &b).unwrap();
}

#[test]
fn off_tape_ops_on_unconsumed_nodes_are_flagged_with_their_line() {
    let src = r#"
fn main() {
    let g = GradGraph.new();
    let k = Tensor.randn([2, 2]);
    let x = Tensor.randn([2, 2]);
    let kid = g.parameter(k);
    let xid = g.input(x);
    let y = matmul(x, k);
    let s = g.sum(xid);
    g.backward(s);
}
"#;
    for e in [eval_export(src, 3).0, mir_export(src, 3).0] {
        let e = e.unwrap();
        // x feeds `g.sum` on the tape, so only k (node 0) is flagged.
        assert_eq!(e.off_tape.len(), 1, "{:?}", e.off_tape);
        let t = &e.off_tape[0];
        assert_eq!((t.op.as_str(), t.node), ("matmul", 0));
        let sp = t.span.unwrap();
        assert_eq!(&src[sp.start as usize..sp.end as usize], "matmul(x, k)");
    }
}

#[test]
fn span_is_the_method_call_when_its_receiver_is_itself_a_call() {
    // Evaluating `same(g)` dispatches a call of its own, which sets the span;
    // the method call must set it again before it records the node.
    let src = r#"
fn same(g: Any) -> Any {
    return g;
}

fn main() {
    let g = GradGraph.new();
    let w = Tensor.randn([2, 2]);
    let wid = same(g).parameter(w);
    let s = g.sum(wid);
    g.backward(s);
}
"#;
    for e in [eval_export(src, 3).0, mir_export(src, 3).0] {
        let e = e.unwrap();
        let sp = e.nodes[0].span.unwrap();
        assert_eq!(&src[sp.start as usize..sp.end as usize], "same(g).parameter(w)");
    }
}

// ── tracking never changes what the program computes ───────────────────

#[test]
fn tracker_on_and_off_print_the_same_output() {
    for (src, seed) in [(XOR, 42), (CNN, 7)] {
        let (program, _) = cjc_parser::parse_source(src);
        let mut plain = cjc_eval::Interpreter::new(seed);
        plain.exec(&program).unwrap();
        let (_, tracked) = eval_export(src, seed);
        assert_eq!(plain.output, tracked);
        let (_, ex) = cjc_mir_exec::run_program_with_executor(&program, seed).unwrap();
        let (_, tracked) = mir_export(src, seed);
        assert_eq!(ex.output, tracked);
    }
}

// ── Gate 3: telemetry on and off hash the same ──────────────────────────

/// Content hash of a run: cjc-snap SHA-256 over the printed lines and the
/// final value of the program.
fn run_hash(output: &[String]) -> [u8; 32] {
    use cjc_runtime::Value;
    use std::rc::Rc;
    let lines: Vec<Value> = output.iter().map(|l| Value::String(Rc::new(l.clone()))).collect();
    cjc_snap::snap(&Value::Array(Rc::new(lines))).content_hash
}

/// What a run leaves behind: printed lines, telemetry lines, and the exact
/// bits of the last trained graph (values and gradients). Printed output
/// rounds (`to_string`), so a probe that nudges one ulp can hide from it; the
/// state bits cannot hide it.
struct Run {
    output: Vec<String>,
    telemetry: Vec<String>,
    state: Vec<u64>,
}

fn eval_run(src: &str, seed: u64, telemetry: Option<cjc_ad::telemetry::Telemetry>) -> Run {
    let (program, _) = cjc_parser::parse_source(src);
    let mut interp = cjc_eval::Interpreter::new(seed);
    interp.ad.enabled = true;
    interp.ad.telemetry = telemetry;
    interp.exec(&program).unwrap_or_else(|e| panic!("eval failed: {e:?}"));
    let lines = interp.ad.telemetry.as_mut().map(|t| {
        t.flush();
        t.memory.clone().unwrap_or_default()
    });
    Run {
        output: interp.output.clone(),
        telemetry: lines.unwrap_or_default(),
        state: interp.ad.state_bits().unwrap_or_default(),
    }
}

fn mir_run(src: &str, seed: u64, telemetry: Option<cjc_ad::telemetry::Telemetry>) -> Run {
    let (program, _) = cjc_parser::parse_source(src);
    let mut ex = cjc_mir_exec::MirExecutor::new(seed);
    ex.ad.enabled = true;
    ex.ad.telemetry = telemetry;
    let mir = {
        let hir = cjc_hir::AstLowering::new().lower_program(&program);
        let mut m = cjc_mir::HirToMir::new().lower_program(&hir);
        cjc_mir::escape::annotate_program(&mut m);
        m
    };
    ex.exec(&mir).unwrap_or_else(|e| panic!("mir failed: {e}"));
    let lines = ex.ad.telemetry.as_mut().map(|t| {
        t.flush();
        t.memory.clone().unwrap_or_default()
    });
    Run {
        output: ex.output.clone(),
        telemetry: lines.unwrap_or_default(),
        state: ex.ad.state_bits().unwrap_or_default(),
    }
}

const LINREG: &str = include_str!("postulate/linreg.cjcl");
const TANH_FIT: &str = include_str!("postulate/tanh_fit.cjcl");

fn check_telemetry_parity(off: &Run, on: &Run) -> Result<(), String> {
    if run_hash(&off.output) != run_hash(&on.output) {
        return Err(format!("output hash differs with telemetry on
off: {:?}
on:  {:?}", off.output, on.output));
    }
    if off.state != on.state {
        let at = off.state.iter().zip(&on.state).position(|(a, b)| a != b);
        return Err(format!(
            "trained-graph bits differ with telemetry on (first at word {at:?} of {} vs {})",
            off.state.len(),
            on.state.len()
        ));
    }
    Ok(())
}

#[test]
fn gate_telemetry_parity_three_programs() {
    // Three programs that train (so telemetry actually writes), plus the CNN
    // fixture, which never calls backward.
    for (name, src, seed) in [("mlp_xor", XOR, 42), ("linreg", LINREG, 5), ("tanh_fit", TANH_FIT, 9), ("cnn", CNN, 7)] {
        for every in [1, cjc_ad::telemetry::DEFAULT_EVERY] {
            let tel = || Some(cjc_ad::telemetry::Telemetry::in_memory(seed, every));
            let off = eval_run(src, seed, None);
            let on = eval_run(src, seed, tel());
            check_telemetry_parity(&off, &on).unwrap_or_else(|e| panic!("eval {name}: {e}"));
            let moff = mir_run(src, seed, None);
            let mon = mir_run(src, seed, tel());
            check_telemetry_parity(&moff, &mon).unwrap_or_else(|e| panic!("mir {name}: {e}"));
            assert_eq!(off.state, moff.state, "{name}: executors trained different bits");
            assert!(!off.state.is_empty(), "{name}: no graph state captured");
            let (lines, mlines) = (on.telemetry, mon.telemetry);
            assert_eq!(lines, mlines, "{name}: executors wrote different telemetry");
            assert_eq!(lines.is_empty(), name == "cnn", "{name}: {} telemetry lines", lines.len());
        }
    }
}

#[test]
fn telemetry_records_match_the_tape() {
    let lines = eval_run(XOR, 42, Some(cjc_ad::telemetry::Telemetry::in_memory(42, 40))).telemetry;
    // 200 epochs sampled every 40 steps: steps 0, 40, 80, 120, 160.
    assert_eq!(lines.len(), 5, "{lines:?}");
    assert!(lines[0].starts_with("{\"step\":0,\"seed\":42,\"loss\":"), "{}", lines[0]);
    assert!(lines[3].starts_with("{\"step\":120,"));
    for l in &lines {
        for p in ["\"n0\":", "\"n1\":", "\"n2\":", "\"n3\":"] {
            assert!(l.contains(p), "missing {p} in {l}");
        }
        assert!(l.ends_with(",\"non_finite\":[]}"), "{l}");
    }
}

#[test]
fn telemetry_counts_non_finite_values_per_node() {
    let src = r#"
fn main() {
    let g = GradGraph.new();
    let x = g.input(Tensor.from_vec([0.0, 1.0], [2]));
    let y = g.ln(x);
    let s = g.sum(y);
    g.backward(s);
}
"#;
    let lines = eval_run(src, 1, Some(cjc_ad::telemetry::Telemetry::in_memory(1, 1))).telemetry;
    // ln(0) = -inf at node 1; the sum (node 2) is -inf too.
    assert!(lines[0].ends_with("\"non_finite\":[[1,1],[2,1]]}"), "{}", lines[0]);
}

#[test]
fn poison_telemetry_that_perturbs_one_float_fails() {
    // The checker sees what a perturbing probe would leave behind. Two shapes:
    // a printed loss differing in its last digit, and — the one printed output
    // cannot see — one trained bit flipped. (The live version of the second,
    // a probe that flips a gradient's last bit, is
    // docs/postulate/poison/03_telemetry_perturbs_a_gradient.patch.)
    let off = eval_run(XOR, 42, None);
    let mut on = eval_run(XOR, 42, None);
    let i = on.output.iter().position(|l| l.contains("loss=")).unwrap();
    let last = on.output[i].pop().unwrap();
    on.output[i].push(if last == '1' { '2' } else { '1' });
    assert!(check_telemetry_parity(&off, &on).unwrap_err().contains("output hash"));

    let mut on = eval_run(XOR, 42, None);
    let k = on.state.len() - 1;
    on.state[k] ^= 1;
    assert!(check_telemetry_parity(&off, &on).unwrap_err().contains("trained-graph bits"));
}
