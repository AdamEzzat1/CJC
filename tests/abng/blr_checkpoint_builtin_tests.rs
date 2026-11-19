//! Phase 0.9.5 R0-3 — `abng_checkpoint_blr` language builtin.
//!
//! R0-3 requires a graph trained via n=1 `train_step` / `blr_update` to
//! be flushed with `checkpoint_blr` before serialization, or replay
//! fails with `BlrStateHashMismatch`. These tests pin the `.cjcl`
//! surface for that flush: the train → checkpoint → serialize → replay
//! round-trip, the failure without the flush, determinism, AST↔MIR
//! parity, the skip rules, and the builtin's Err paths.

#![allow(clippy::needless_raw_string_hashes)]

use cjc_abng::dispatch::{dispatch_abng, reset_arena};
use cjc_runtime::value::Value;

#[derive(Clone, Copy, Debug)]
enum Backend {
    Eval,
    Mir,
}

fn try_run(backend: Backend, body: &str, seed: u64) -> Result<Vec<String>, String> {
    let src = format!("fn main() {{\n{body}\n}}\n");
    let (program, diags) = cjc_parser::parse_source(&src);
    assert!(
        !diags.has_errors(),
        "parse errors:\n{:#?}\nsource:\n{src}",
        diags.diagnostics,
    );
    reset_arena();
    match backend {
        Backend::Eval => {
            let mut interp = cjc_eval::Interpreter::new(seed);
            interp
                .exec(&program)
                .map(|_| interp.output.clone())
                .map_err(|e| format!("{e:?}"))
        }
        Backend::Mir => cjc_mir_exec::run_program_with_executor(&program, seed)
            .map(|(_v, exec)| exec.output)
            .map_err(|e| format!("{e:?}")),
    }
}

fn run(backend: Backend, body: &str) -> Vec<String> {
    try_run(backend, body, 42)
        .unwrap_or_else(|e| panic!("{backend:?} failed:\n{body}\nerror: {e}"))
}

/// Parity-checked output: eval and MIR must print byte-identical lines.
fn run_parity(body: &str) -> Vec<String> {
    let eval_out = run(Backend::Eval, body);
    let mir_out = run(Backend::Mir, body);
    assert_eq!(eval_out, mir_out, "AST↔MIR parity violation");
    eval_out
}

/// Graph with a d=4 BLR head over four codebook leaves, trained with
/// `rows` single-row `abng_train_step` calls that all route to one leaf.
fn train_body(rows: u32) -> String {
    format!(
        r#"
        let g = abng_new(7);
        let cb = Tensor.from_vec([0.25, 0.5, 0.75], [1, 3]);
        abng_set_codebook(g, cb);
        abng_set_leaf_head(g, 1, Tensor.from_vec([4.0], [1]), 1, "tanh");
        abng_set_blr_prior(g, 2.0, 1.0, 0.5);
        abng_add_node(g, 0, 0);
        abng_add_node(g, 0, 1);
        abng_add_node(g, 0, 2);
        abng_add_node(g, 0, 3);
        let x = Tensor.from_vec([0.45], [1]);
        let phi = Tensor.from_vec([1.0, 0.5, 0.25, 0.125], [4]);
        let mut y: f64 = 0.3;
        let mut i: i64 = 0;
        while i < {rows} {{
            abng_train_step(g, x, phi, y);
            y = y + 0.01;
            i = i + 1;
        }}
        "#
    )
}

#[test]
fn checkpoint_then_replay_roundtrips_with_parity() {
    let body = format!(
        r#"{}
        print(abng_checkpoint_blr(g));
        let head = abng_chain_head(g);
        let g2 = abng_replay(abng_serialize(g));
        print(abng_chain_head(g2) == head);
        print(abng_verify_chain(g2));
        "#,
        train_body(10)
    );
    let out = run_parity(&body);
    assert_eq!(out, vec!["1", "true", "true"], "one mid-interval leaf flushed");
}

#[test]
fn replay_without_checkpoint_fails_in_both_backends() {
    // The contract's other direction, from `.cjcl`: skipping the flush
    // is a loud replay error, never a silently accepted state.
    let body = format!(
        r#"{}
        let g2 = abng_replay(abng_serialize(g));
        print(abng_chain_head(g2));
        "#,
        train_body(10)
    );
    for backend in [Backend::Eval, Backend::Mir] {
        let err = try_run(backend, &body, 42)
            .expect_err("replay of an un-flushed trained graph must fail");
        assert!(
            err.contains("BLR state hash mismatch"),
            "{backend:?}: expected BlrStateHashMismatch, got: {err}"
        );
    }
}

#[test]
fn checkpoint_chain_head_double_run_deterministic() {
    let body = format!(
        r#"{}
        abng_checkpoint_blr(g);
        print(abng_chain_head(g));
        "#,
        train_body(10)
    );
    let a = run_parity(&body);
    let b = run_parity(&body);
    assert_eq!(a, b, "same seed + same rows -> same post-flush chain head");
}

#[test]
fn checkpoint_skips_untrained_and_boundary_graphs() {
    // Never-trained graph: nothing to flush.
    let untrained = format!(
        r#"{}
        print(abng_checkpoint_blr(g));
        "#,
        train_body(0)
    );
    assert_eq!(run_parity(&untrained), vec!["0"]);

    // Exactly BLR_CHECKPOINT_INTERVAL (64) rows: the last TrainStep
    // already carries the full witness, so no flush is needed and the
    // un-flushed graph replays cleanly.
    let boundary = format!(
        r#"{}
        print(abng_checkpoint_blr(g));
        let g2 = abng_replay(abng_serialize(g));
        print(abng_chain_head(g2) == abng_chain_head(g));
        "#,
        train_body(64)
    );
    assert_eq!(run_parity(&boundary), vec!["0", "true"]);
}

#[test]
fn checkpoint_builtin_err_paths() {
    reset_arena();
    let err = dispatch_abng("abng_checkpoint_blr", &[]).unwrap_err();
    assert!(err.contains("expected 1 arguments"), "got: {err}");

    let err = dispatch_abng("abng_checkpoint_blr", &[Value::Int(9_999)]).unwrap_err();
    assert!(err.contains("no graph with id 9999"), "got: {err}");
}
