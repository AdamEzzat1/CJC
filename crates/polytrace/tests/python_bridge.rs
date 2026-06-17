//! Cross-language contract test: the Rust analyzer reads a trace produced by the
//! pure-Python `seshat` recorder (`python-seshat/`).
//!
//! The fixture `fixtures/python_demo.seshat` was recorded by running a small
//! Python program under `sys.setprofile` (see `python-seshat/examples/gen_fixture.py`)
//! and committed, so this test needs no Python at build time — it proves the
//! `.seshat` format is a stable interface between the two languages.

use polytrace::{analyze_trace, replay};

/// A real `.seshat` recorded by the Python side (json pipeline: a Python `work`
/// function calling `json.loads`, which crosses into the C JSON scanner).
const PY_TRACE: &[u8] = include_bytes!("fixtures/python_demo.seshat");

/// A real `.seshat` recorded from a Python `asyncio` program (coroutines
/// `worker`/`fetch` with real `await` suspend/resume points).
const PY_ASYNC: &[u8] = include_bytes!("fixtures/python_async.seshat");

#[test]
fn rust_analyzer_reads_python_produced_trace() {
    let trace = replay(PY_TRACE).expect("a Python-produced .seshat must replay in Rust");
    let r = analyze_trace(&trace);

    let labels: Vec<&String> = r.flamegraph.frame_total.keys().collect();

    // Real Python frames were captured.
    assert!(
        labels.iter().any(|k| k.starts_with("py:")),
        "expected Python frames, got {labels:?}"
    );
    // The Python→native (= Py↔Rust) boundary was detected.
    assert!(
        labels.iter().any(|k| k.starts_with("ffi:")),
        "expected native/boundary frames, got {labels:?}"
    );
    assert!(
        r.boundary.crossings > 0,
        "expected at least one boundary crossing"
    );

    // A recognizable Python function from the recorded program.
    assert!(
        labels.iter().any(|k| k.contains("work")),
        "expected the `work` Python frame, got {labels:?}"
    );

    // The Python-side `seshat.zone("work")` produced a real pipeline stage.
    assert!(
        r.pipeline.per_stage.contains_key("work"),
        "expected a `work` pipeline zone, got {:?}",
        r.pipeline.per_stage.keys().collect::<Vec<_>>()
    );

    // Analysis of a fixed trace is deterministic regardless of which language
    // produced it.
    assert_eq!(r.content_hash(), analyze_trace(&trace).content_hash());
}

#[test]
fn rust_analyzer_measures_python_async_stalls() {
    let trace = replay(PY_ASYNC).expect("a Python async .seshat must replay in Rust");
    let r = analyze_trace(&trace);

    // Coroutine frames were captured and tagged async.
    assert!(
        r.flamegraph.frame_total.keys().any(|k| k.starts_with("async:")),
        "expected async: frames"
    );

    // Await stalls were measured (not just identified) — AwaitResume edges.
    assert!(
        r.async_stall.total_resumes > 0,
        "expected measured await resumes, got {}",
        r.async_stall.total_resumes
    );

    // The `fetch` coroutine specifically was resumed, with a measured wait.
    let fetch = r
        .async_stall
        .tasks
        .iter()
        .find(|(k, _)| k.contains("fetch"))
        .map(|(_, st)| st)
        .expect("expected the `fetch` coroutine among async tasks");
    assert!(fetch.resumes > 0, "fetch should have been resumed");

    assert_eq!(r.content_hash(), analyze_trace(&trace).content_hash());
}
