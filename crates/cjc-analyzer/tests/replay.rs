//! Postulate Phase 1 gate — LSP replay.
//!
//! A scripted session (open, hover, completion, inlay hints, document
//! highlights, the `postulate/*` requests including Phase 2's `telemetry` and
//! `cursorContext`, an edit, a second seed) is replayed against a fresh
//! in-process `Analyzer` ten times. Every response must be byte-identical
//! across runs once the fields a response itself lists under `"measured"`
//! (wall time) are masked. The poison test shows the comparison is not
//! vacuous: leave the measured fields in and the same gate fails.

use cjc_analyzer::server::Analyzer;
use lsp_server::{Message, Notification, Request, RequestId};
use serde_json::{json, Value as Json};

const TANH_FIT: &str = include_str!("../../../tests/postulate/tanh_fit.cjcl");
const CNN: &str = include_str!("../../../tests/postulate/cnn_off_tape.cjcl");
const RUNS: usize = 10;

fn req(id: i32, method: &str, params: Json) -> Message {
    Message::Request(Request { id: RequestId::from(id), method: method.into(), params })
}

fn notif(method: &str, params: Json) -> Message {
    Message::Notification(Notification { method: method.into(), params })
}

fn doc(uri: &str) -> Json {
    json!({ "textDocument": { "uri": uri } })
}

fn script() -> Vec<Message> {
    let a = "file:///postulate/tanh_fit.cjcl";
    let b = "file:///postulate/cnn_off_tape.cjcl";
    let all = json!({ "start": {"line": 0, "character": 0}, "end": {"line": 9999, "character": 0} });
    let edited = TANH_FIT.replace("epoch < 60", "epoch < 30");
    vec![
        notif("textDocument/didOpen", json!({ "textDocument": {
            "uri": a, "languageId": "cjc", "version": 1, "text": TANH_FIT } })),
        req(1, "textDocument/hover", json!({ "textDocument": {"uri": a}, "position": {"line": 3, "character": 15} })),
        req(2, "textDocument/completion", json!({ "textDocument": {"uri": a}, "position": {"line": 3, "character": 12} })),
        req(3, "textDocument/inlayHint", json!({ "textDocument": {"uri": a}, "range": all })),
        req(4, "postulate/gradGraph", doc(a)),
        req(5, "postulate/snapshotHash", doc(a)),
        req(6, "postulate/executorAgreement", doc(a)),
        req(7, "postulate/runReport", doc(a)),
        req(8, "postulate/snapshotHash", json!({ "textDocument": {"uri": a}, "seed": 7 })),
        req(14, "postulate/telemetry", doc(a)),
        // Inside the training loop, on the `g` of `g.sub(o, y_id)`.
        req(15, "postulate/cursorContext", json!({ "textDocument": {"uri": a}, "position": {"line": 23, "character": 16} })),
        // Inside `forward`, on `g.matmul(x_id, w1_id)`: not inside the loop's text.
        req(16, "postulate/cursorContext", json!({ "textDocument": {"uri": a}, "position": {"line": 3, "character": 22} })),
        req(17, "textDocument/documentHighlight", json!({ "textDocument": {"uri": a}, "position": {"line": 18, "character": 12} })),
        notif("textDocument/didChange", json!({
            "textDocument": {"uri": a, "version": 2},
            "contentChanges": [{ "text": edited }] })),
        req(9, "postulate/snapshotHash", doc(a)),
        req(10, "postulate/executorAgreement", doc(a)),
        notif("textDocument/didOpen", json!({ "textDocument": {
            "uri": b, "languageId": "cjc", "version": 1, "text": CNN } })),
        req(11, "postulate/gradGraph", doc(b)),
        req(12, "postulate/runReport", doc("file:///postulate/not_open.cjcl")),
        req(13, "postulate/unknown", doc(a)),
        notif("textDocument/didClose", json!({ "textDocument": {"uri": b} })),
    ]
}

/// Replace every field an object names in its own `"measured"` list.
fn mask_measured(v: &mut Json) {
    match v {
        Json::Object(m) => {
            let measured: Vec<String> = m
                .get("measured")
                .and_then(Json::as_array)
                .map(|a| a.iter().filter_map(|s| s.as_str().map(str::to_owned)).collect())
                .unwrap_or_default();
            for k in measured {
                if let Some(f) = m.get_mut(&k) {
                    *f = json!("<measured>");
                }
            }
            m.values_mut().for_each(mask_measured);
        }
        Json::Array(a) => a.iter_mut().for_each(mask_measured),
        _ => {}
    }
}

/// One replay: the transcript of everything the server sent, as JSON text.
fn replay(mask: bool) -> Vec<String> {
    let mut a = Analyzer::new();
    let mut out = Vec::new();
    for m in script() {
        for r in a.handle(m) {
            let mut v = serde_json::to_value(&r).unwrap();
            if mask {
                mask_measured(&mut v);
            }
            out.push(serde_json::to_string(&v).unwrap());
        }
    }
    out
}

fn result_of(transcript: &[String], id: i32) -> Json {
    transcript
        .iter()
        .map(|s| serde_json::from_str::<Json>(s).unwrap())
        .find(|v| v.get("id") == Some(&json!(id)))
        .unwrap_or_else(|| panic!("no response {id}"))
}

#[test]
fn gate_lsp_replay_is_identical_over_ten_runs() {
    let first = replay(true);
    for i in 1..RUNS {
        let again = replay(true);
        assert_eq!(first.len(), again.len(), "run {i}: message count");
        for (k, (x, y)) in first.iter().zip(&again).enumerate() {
            assert_eq!(x, y, "run {i}: message {k} differs");
        }
    }

    // The session says something real, not just the same nothing ten times.
    assert_eq!(first.len(), 21, "17 responses + 4 diagnostics notifications");
    let hints = result_of(&first, 3)["result"].as_array().cloned().unwrap();
    assert!(!hints.is_empty(), "shape hints");
    assert!(hints.iter().any(|h| h["label"] == ": [5, 8]"), "tanh(matmul(x, w1)) is [5, 8]: {hints:?}");
    let g = &result_of(&first, 4)["result"];
    assert_eq!(g["error"], Json::Null);
    assert_eq!(g["param_names"], json!([{"node": 0, "name": "W1"}, {"node": 1, "name": "W2"}]));
    let agree = &result_of(&first, 6)["result"];
    assert_eq!(agree["executors_agree"], json!(true), "{agree}");
    assert_eq!(agree["telemetry_parity"], json!(true), "{agree}");
    let tel = &result_of(&first, 14)["result"];
    let recs = tel["records"].as_array().unwrap();
    assert_eq!((tel["steps"].as_u64(), recs.len()), (Some(60), 6), "60 steps sampled every 10");
    assert_eq!(recs[3]["step"], 30);
    assert!(recs[0]["grad_norms"]["n0"].is_number(), "{}", recs[0]);
    let in_loop = &result_of(&first, 15)["result"];
    assert_eq!(in_loop["symbol"], "g");
    assert_eq!(in_loop["function"], "main");
    assert_eq!(in_loop["training_loop"]["start"]["line"], 16, "{in_loop}");
    assert_eq!(in_loop["nodes"][0]["op"], "Sub", "{in_loop}");
    let in_fn = &result_of(&first, 16)["result"];
    assert_eq!(in_fn["function"], "forward");
    assert_eq!(in_fn["training_loop"], Json::Null, "{in_fn}");
    assert_eq!(in_fn["nodes"][0]["op"], "MatMul", "{in_fn}");
    assert_eq!(result_of(&first, 17)["result"].as_array().unwrap().len(), 5, "w1_id by name: forward(..) param + use, let, call arg, grad");
    let report = &result_of(&first, 7)["result"];
    assert_eq!(report["wall_ms_median"], json!("<measured>"));
    assert!(report["telemetry_steps"].as_u64().unwrap() == 60);
    let h42 = &result_of(&first, 5)["result"]["mir"]["hash"];
    let h7 = &result_of(&first, 8)["result"]["mir"]["hash"];
    let h_edit = &result_of(&first, 9)["result"]["mir"]["hash"];
    assert_ne!(h42, h7, "seed changes the run");
    assert_ne!(h42, h_edit, "an edit invalidates the cached run");
    assert_eq!(result_of(&first, 5)["result"]["eval"]["hash"], *h42);
    let cnn = &result_of(&first, 11)["result"]["graph"]["off_tape"];
    assert_eq!(cnn.as_array().unwrap().len(), 2, "{cnn}");
    assert!(result_of(&first, 12).get("error").is_some(), "unopened document is an error");
    assert!(result_of(&first, 13).get("error").is_some(), "unknown method is an error");
}

/// Poison: compare the raw transcripts (timing left in). The gate must fail,
/// or it could not tell a deterministic server from a non-deterministic one.
#[test]
fn poison_replay_with_measured_fields_left_in_fails() {
    let a = replay(false);
    let b = replay(false);
    let differing: Vec<usize> = (0..a.len()).filter(|&k| a[k] != b[k]).collect();
    assert!(!differing.is_empty(), "timing fields must make raw transcripts differ");
    // ...and only the runReport response differs.
    for k in differing {
        assert!(a[k].contains("wall_ms_median"), "unexpected nondeterminism: {}", a[k]);
    }
}
