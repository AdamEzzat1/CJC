//! Postulate Phase 1 — what `cjc-analyzer` can tell an IDE about a run.
//!
//! Custom requests (params: `{ "textDocument": { "uri" }, "seed"?: u64 }`):
//!
//! | Method | Answers |
//! |---|---|
//! | `postulate/gradGraph` | The last trained `GradGraph`: nodes, spans, groups, off-tape ops, parameter names |
//! | `postulate/snapshotHash` | cjc-snap SHA-256 of the run's output, per executor |
//! | `postulate/executorAgreement` | Do `cjc-eval` and `cjc-mir-exec` agree on output and graph? |
//! | `postulate/runReport` | Wall time and peak heap (measured), telemetry steps, graph size |
//! | `postulate/telemetry` | The run's telemetry records (one per sampled step) |
//! | `postulate/cursorContext` | Symbol, function, training loop and graph nodes at a position (see [`crate::cursor`]) |
//!
//! Plus tensor-shape inlay hints (`textDocument/inlayHint`) from the same run.
//!
//! A document is run on request, once per (version, seed), with stdout echo
//! off and tracking on, and the result is cached. Every field of every
//! response is a deterministic function of (source, seed) except those a
//! response lists under `"measured"` — timing — which the replay gate masks.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::rc::Rc;
use std::time::Instant;

use cjc_ad::export::GraphExport;
use serde_json::{json, Value as Json};

pub const DEFAULT_SEED: u64 = 42;
/// Runs per `postulate/runReport` timing sample.
pub const TIMING_RUNS: usize = 5;

/// One executor's run of a document.
pub struct ExecRun {
    pub output: Vec<String>,
    pub graph: Option<GraphExport>,
    pub error: Option<String>,
    pub telemetry_steps: u64,
    /// Telemetry lines, when the run sampled any.
    pub telemetry: Vec<String>,
    /// Every bit of the last tape (`Tracker::state_bits`).
    pub state: Option<Vec<u64>>,
}

pub struct RunCapture {
    pub seed: u64,
    pub eval: ExecRun,
    pub mir: ExecRun,
    /// The MIR run with telemetry *off* printed the same output and left the
    /// same tape bits as the MIR run with telemetry on.
    pub telemetry_parity: bool,
}

fn panic_text(p: Box<dyn std::any::Any + Send>) -> String {
    p.downcast_ref::<String>()
        .cloned()
        .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_else(|| "panic".into())
}

/// In-memory telemetry sampled every `every` steps, or none.
fn telemetry(seed: u64, every: Option<u64>) -> Option<cjc_ad::telemetry::Telemetry> {
    every.map(|e| cjc_ad::telemetry::Telemetry::in_memory(seed, e))
}

fn telemetry_lines(ad: &mut cjc_ad::tracker::Tracker) -> (u64, Vec<String>) {
    match ad.telemetry.as_mut() {
        Some(t) => {
            t.flush();
            (t.step(), t.memory.clone().unwrap_or_default())
        }
        None => (0, Vec::new()),
    }
}

/// Run on `cjc-eval`, with telemetry sampled every `every` steps (or off).
pub fn run_eval(src: &str, seed: u64, every: Option<u64>) -> ExecRun {
    let (program, _) = cjc_parser::parse_source(src);
    let mut it = cjc_eval::Interpreter::new(seed);
    it.echo = false;
    it.ad.enabled = true;
    it.ad.telemetry = telemetry(seed, every);
    let r = catch_unwind(AssertUnwindSafe(|| it.exec(&program)));
    let error = match r {
        Ok(Ok(_)) => None,
        Ok(Err(e)) => Some(format!("{e:?}")),
        Err(p) => Some(format!("panic: {}", panic_text(p))),
    };
    let (telemetry_steps, telemetry) = telemetry_lines(&mut it.ad);
    ExecRun {
        output: it.output.clone(),
        graph: it.ad.export(),
        error,
        telemetry_steps,
        telemetry,
        state: it.ad.state_bits(),
    }
}

/// Run on `cjc-mir-exec`, with telemetry sampled every `every` steps (or off).
pub fn run_mir(src: &str, seed: u64, every: Option<u64>) -> ExecRun {
    let (program, _) = cjc_parser::parse_source(src);
    let mut ex = cjc_mir_exec::MirExecutor::new(seed);
    ex.echo = false;
    ex.ad.enabled = true;
    ex.ad.telemetry = telemetry(seed, every);
    let r = catch_unwind(AssertUnwindSafe(|| {
        let hir = cjc_hir::AstLowering::new().lower_program(&program);
        let mut mir = cjc_mir::HirToMir::new().lower_program(&hir);
        cjc_mir::escape::annotate_program(&mut mir);
        ex.scan_ast_imports(&program);
        ex.exec(&mir)
    }));
    let error = match r {
        Ok(Ok(_)) => None,
        Ok(Err(e)) => Some(format!("{e}")),
        Err(p) => Some(format!("panic: {}", panic_text(p))),
    };
    let (telemetry_steps, telemetry) = telemetry_lines(&mut ex.ad);
    ExecRun {
        output: ex.output.clone(),
        graph: ex.ad.export(),
        error,
        telemetry_steps,
        telemetry,
        state: ex.ad.state_bits(),
    }
}

const EVERY: Option<u64> = Some(cjc_ad::telemetry::DEFAULT_EVERY);

pub fn capture(src: &str, seed: u64) -> Rc<RunCapture> {
    // The executors print panics through the default hook; keep them off the
    // server's stderr noise but leave the hook as it was afterwards.
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let eval = run_eval(src, seed, EVERY);
    let mir = run_mir(src, seed, EVERY);
    let plain = run_mir(src, seed, None);
    std::panic::set_hook(hook);
    let telemetry_parity = mir.output == plain.output && mir.state == plain.state && mir.error == plain.error;
    Rc::new(RunCapture { seed, eval, mir, telemetry_parity })
}

/// cjc-snap content hash of a run's printed output, as hex.
pub fn output_hash(output: &[String]) -> String {
    use cjc_runtime::Value;
    let lines: Vec<Value> = output.iter().map(|l| Value::String(Rc::new(l.clone()))).collect();
    let h = cjc_snap::snap(&Value::Array(Rc::new(lines))).content_hash;
    h.iter().map(|b| format!("{b:02x}")).collect()
}

/// Parameter names from the source: the identifier passed to `g.parameter(…)`.
pub fn param_names(src: &str, g: &GraphExport) -> Vec<(u32, String)> {
    let mut out = Vec::new();
    for n in &g.nodes {
        if n.op != "Parameter" {
            continue;
        }
        let Some(sp) = n.span else { continue };
        let Some(text) = src.get(sp.start as usize..sp.end as usize) else { continue };
        let (Some(l), Some(r)) = (text.find('('), text.rfind(')')) else { continue };
        let arg = text[l + 1..r].trim();
        if !arg.is_empty() && arg.chars().all(|c| c.is_alphanumeric() || c == '_') {
            out.push((n.id, arg.to_owned()));
        }
    }
    out
}

fn graph_json(g: &GraphExport) -> Json {
    serde_json::from_str(&g.to_json()).expect("GraphExport::to_json is valid JSON")
}

pub fn grad_graph_response(src: &str, run: &RunCapture) -> Json {
    let e = &run.eval;
    json!({
        "seed": run.seed,
        "executor": "cjc-eval",
        "error": e.error,
        "graph": e.graph.as_ref().map(graph_json),
        "param_names": e.graph.as_ref().map(|g| {
            param_names(src, g).into_iter().map(|(id, n)| json!({"node": id, "name": n})).collect::<Vec<_>>()
        }),
    })
}

pub fn snapshot_hash_response(run: &RunCapture) -> Json {
    json!({
        "seed": run.seed,
        "eval": { "hash": output_hash(&run.eval.output), "error": run.eval.error },
        "mir": { "hash": output_hash(&run.mir.output), "error": run.mir.error },
    })
}

pub fn executor_agreement_response(run: &RunCapture) -> Json {
    let output_agree = run.eval.output == run.mir.output && run.eval.error.is_none() == run.mir.error.is_none();
    let ge = run.eval.graph.as_ref().map(GraphExport::to_json);
    let gm = run.mir.graph.as_ref().map(GraphExport::to_json);
    let graph_agree = ge == gm;
    json!({
        "seed": run.seed,
        "executors_agree": output_agree && graph_agree,
        "output_agree": output_agree,
        "graph_agree": graph_agree,
        "eval_hash": output_hash(&run.eval.output),
        "mir_hash": output_hash(&run.mir.output),
        "eval_error": run.eval.error,
        "mir_error": run.mir.error,
        "telemetry_parity": run.telemetry_parity,
    })
}

/// The MIR run's telemetry records, parsed: the same objects
/// `cjcl run --telemetry` writes, one per sampled step.
pub fn telemetry_response(run: &RunCapture) -> Json {
    let records: Vec<Json> = run.mir.telemetry.iter().filter_map(|l| serde_json::from_str(l).ok()).collect();
    json!({
        "seed": run.seed,
        "every": cjc_ad::telemetry::DEFAULT_EVERY,
        "steps": run.mir.telemetry_steps,
        "records": records,
        "error": run.mir.error,
    })
}

/// Median and spread (max − min) of `TIMING_RUNS` MIR runs, in ms, and the
/// peak heap above the starting level during the first run (when the binary
/// installed [`crate::alloc_meter::CountingAlloc`]). Measured, not
/// reproducible: listed under `"measured"`.
pub fn run_report_response(src: &str, run: &RunCapture) -> Json {
    let mut peak_bytes = None;
    let mut ms: Vec<f64> = (0..TIMING_RUNS)
        .map(|i| {
            let t = Instant::now();
            if i == 0 {
                peak_bytes = crate::alloc_meter::measure(|| run_mir(src, run.seed, EVERY)).1;
            } else {
                let _ = run_mir(src, run.seed, EVERY);
            }
            t.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    ms.sort_by(f64::total_cmp);
    let median = ms[ms.len() / 2];
    let spread = ms[ms.len() - 1] - ms[0];
    json!({
        "seed": run.seed,
        "runs": TIMING_RUNS,
        "wall_ms_median": median,
        "wall_ms_spread": spread,
        "measured": ["wall_ms_median", "wall_ms_spread", "peak_bytes"],
        "telemetry_steps": run.mir.telemetry_steps,
        "graph_nodes": run.mir.graph.as_ref().map(|g| g.nodes.len()),
        "peak_bytes": peak_bytes,
        // polytrace wiring (the peak's line, copies, zones) is Phase 5.
        "copies": Json::Null,
        "hottest_zone": Json::Null,
        "error": run.mir.error,
    })
}

/// Byte offset → LSP position (UTF-16 columns).
pub fn position_of(src: &str, byte: usize) -> lsp_types::Position {
    let byte = byte.min(src.len());
    let line_start = src[..byte].rfind('\n').map_or(0, |i| i + 1);
    let line = src[..line_start].matches('\n').count() as u32;
    let character = src[line_start..byte].encode_utf16().count() as u32;
    lsp_types::Position { line, character }
}

/// Tensor-shape hints at the end of each recording call, from the last
/// trained graph. One hint per span (the node a call returned is the last
/// node stamped with that span).
pub fn shape_hints(src: &str, run: &RunCapture, range: lsp_types::Range) -> Vec<lsp_types::InlayHint> {
    let Some(g) = &run.eval.graph else { return Vec::new() };
    let mut last_by_span = std::collections::BTreeMap::new();
    for n in &g.nodes {
        if let Some(sp) = n.span {
            last_by_span.insert((sp.start, sp.end), n);
        }
    }
    last_by_span
        .into_iter()
        .filter_map(|((_, end), n)| {
            let pos = position_of(src, end as usize);
            let inside = (pos.line, pos.character) >= (range.start.line, range.start.character)
                && (pos.line, pos.character) <= (range.end.line, range.end.character);
            inside.then(|| lsp_types::InlayHint {
                position: pos,
                label: lsp_types::InlayHintLabel::String(format!(
                    ": [{}]",
                    n.shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ")
                )),
                kind: Some(lsp_types::InlayHintKind::TYPE),
                text_edits: None,
                tooltip: Some(lsp_types::InlayHintTooltip::String(format!(
                    "GradGraph node {} ({}) — shape from the last trained graph, seed {}",
                    n.id, n.op, run.seed
                ))),
                padding_left: Some(true),
                padding_right: None,
                data: None,
            })
        })
        .collect()
}
