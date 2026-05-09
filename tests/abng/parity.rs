//! AST eval ↔ MIR exec parity tests for `abng_*` builtins.
//!
//! Each test runs a tiny `.cjcl` snippet through both backends and asserts
//! byte-identical printed output. This is the canonical way of verifying
//! that the satellite-dispatch pattern routes identically from both
//! executors — the same gate Phase 3a/3b/3c held the `grad_graph_*`
//! family to.

#![allow(clippy::needless_raw_string_hashes)]

use cjc_abng::dispatch::reset_arena;

#[derive(Clone, Copy, Debug)]
enum Backend {
    Eval,
    Mir,
}

fn run(backend: Backend, body: &str, seed: u64) -> Vec<String> {
    let src = format!("fn main() {{\n{body}\n}}\n");
    let (program, diags) = cjc_parser::parse_source(&src);
    assert!(
        !diags.has_errors(),
        "parse errors:\n{:#?}\nsource:\n{src}",
        diags.diagnostics,
    );
    // Reset the per-thread arena before every backend invocation so the
    // graph_id space is the same for eval and mir.
    reset_arena();
    match backend {
        Backend::Eval => {
            let mut interp = cjc_eval::Interpreter::new(seed);
            interp
                .exec(&program)
                .unwrap_or_else(|e| panic!("eval failed for snippet:\n{src}\nerror: {e:?}"));
            interp.output
        }
        Backend::Mir => {
            let (_v, exec) = cjc_mir_exec::run_program_with_executor(&program, seed)
                .unwrap_or_else(|e| panic!("MIR-exec failed for snippet:\n{src}\nerror: {e:?}"));
            exec.output
        }
    }
}

fn assert_parity(label: &str, body: &str) {
    let eval_out = run(Backend::Eval, body, 42);
    let mir_out = run(Backend::Mir, body, 42);
    assert_eq!(
        eval_out, mir_out,
        "[{label}] AST↔MIR parity violation\n  eval: {eval_out:?}\n  mir : {mir_out:?}",
    );
}

// ─── Construction / inspection ─────────────────────────────────────────

#[test]
fn parity_new_and_root() {
    assert_parity(
        "abng_new + abng_root",
        r#"
        let g = abng_new(0);
        print(abng_root(g));
        print(abng_node_count(g));
        "#,
    );
}

#[test]
fn parity_audit_len_after_observation() {
    assert_parity(
        "abng_audit_len progression",
        r#"
        let g = abng_new(7);
        print(abng_audit_len(g));
        abng_observe(g, 0, 1.0);
        print(abng_audit_len(g));
        abng_observe(g, 0, 2.0);
        print(abng_audit_len(g));
        "#,
    );
}

#[test]
fn parity_node_stats_after_two_observations() {
    assert_parity(
        "node stats == [2, 2, 2]",
        r#"
        let g = abng_new(0);
        abng_observe(g, 0, 1.0);
        abng_observe(g, 0, 3.0);
        print(abng_node_stats(g, 0));
        "#,
    );
}

#[test]
fn parity_chain_head_changes_on_observe() {
    assert_parity(
        "chain head differs pre/post observe",
        r#"
        let g = abng_new(0);
        let h0 = abng_chain_head(g);
        abng_observe(g, 0, 1.0);
        let h1 = abng_chain_head(g);
        print(h0 == h1);
        "#,
    );
}

#[test]
fn parity_verify_chain_passes() {
    assert_parity(
        "verify_chain returns true",
        r#"
        let g = abng_new(99);
        abng_observe(g, 0, 1.5);
        abng_observe(g, 0, 2.5);
        abng_observe(g, 0, 3.5);
        print(abng_verify_chain(g));
        "#,
    );
}

// ─── Batch observation (Phase 0.6 Item 4 / v13 semantics) ───────────

#[test]
fn parity_observe_batch_chain_differs_from_individual() {
    // Phase 0.6 Item 4 — abng_observe_batch emits ONE
    // BeliefUpdateBatch event vs N BeliefUpdate events. Different
    // audit histories → different chain heads. The parity gate
    // checks AST↔MIR agreement on the (now `false`) boolean print.
    assert_parity(
        "batch chain head differs from per-row chain head",
        r#"
        let g_a = abng_new(0);
        abng_observe(g_a, 0, 1.0);
        abng_observe(g_a, 0, 2.0);
        abng_observe(g_a, 0, 3.0);
        let g_b = abng_new(0);
        let xs = Tensor.from_vec([1.0, 2.0, 3.0], [3]);
        abng_observe_batch(g_b, 0, xs);
        print(abng_chain_head(g_a) == abng_chain_head(g_b));
        "#,
    );
}

#[test]
fn parity_observe_slice_chain_equals_individual() {
    // Phase 0.6 Item 4 — abng_observe_slice preserves the legacy
    // loop-observe semantics: N BeliefUpdate events, same chain
    // head as N per-row observe calls.
    assert_parity(
        "slice chain head equals per-row chain head",
        r#"
        let g_a = abng_new(0);
        abng_observe(g_a, 0, 1.0);
        abng_observe(g_a, 0, 2.0);
        abng_observe(g_a, 0, 3.0);
        let g_b = abng_new(0);
        let xs = Tensor.from_vec([1.0, 2.0, 3.0], [3]);
        abng_observe_slice(g_b, 0, xs);
        print(abng_chain_head(g_a) == abng_chain_head(g_b));
        "#,
    );
}

// ─── Phase 0.6 Item 7: abng_route_to_leaf fused native kernel ────────

#[test]
fn parity_route_to_leaf_matches_three_call_sequence() {
    // Phase 0.6 Item 7 — the fused `abng_route_to_leaf` builtin
    // produces the same leaf id as `encode_prefix → descend →
    // extract index 1` across both eval and mir backends. Headline
    // assertion: AST and MIR agree that the fused and 3-call paths
    // produce identical leaf ids.
    assert_parity(
        "fused route_to_leaf == encode_prefix + descend + extract",
        r#"
        let g = abng_new(0);
        let codebook = Tensor.from_vec([0.25, 0.5, 0.75], [1, 3]);
        abng_set_codebook(g, codebook);
        abng_add_node(g, 0, 0);
        abng_add_node(g, 0, 1);
        abng_add_node(g, 0, 2);
        // Sequence path (3 dispatches + 1 Tensor alloc).
        let x = Tensor.from_vec([0.40], [1]);
        let prefix = abng_encode_prefix(g, x);
        let evidence = abng_descend(g, prefix);
        let leaf_seq = int(evidence.get([1]));
        // Fused path (1 dispatch).
        let x2 = Tensor.from_vec([0.40], [1]);
        let leaf_fused = abng_route_to_leaf(g, x2);
        print(leaf_seq == leaf_fused);
        "#,
    );
}

// ─── Serialize / replay round-trip ────────────────────────────────────

#[test]
fn parity_serialize_replay_chain_head() {
    assert_parity(
        "round-trip preserves chain head",
        r#"
        let g = abng_new(0);
        abng_observe(g, 0, 1.0);
        abng_observe(g, 0, 2.0);
        abng_observe(g, 0, 3.0);
        let head_before = abng_chain_head(g);
        let blob = abng_serialize(g);
        let g2 = abng_replay(blob);
        let head_after = abng_chain_head(g2);
        print(head_before == head_after);
        "#,
    );
}

#[test]
fn parity_double_run_chain_head_byte_identical() {
    // Determinism: same source, same seed → same chain head.
    let body = r#"
        let g = abng_new(42);
        abng_observe(g, 0, 0.1);
        abng_observe(g, 0, 0.2);
        abng_observe(g, 0, 0.3);
        abng_observe(g, 0, 0.4);
        print(abng_chain_head(g));
    "#;
    let a = run(Backend::Eval, body, 0);
    let b = run(Backend::Eval, body, 0);
    assert_eq!(a, b, "eval double-run chain heads differ");
    let c = run(Backend::Mir, body, 0);
    let d = run(Backend::Mir, body, 0);
    assert_eq!(c, d, "MIR double-run chain heads differ");
    assert_eq!(a, c, "eval ↔ MIR chain heads differ");
}

// ─── Phase 0.7 Item 4: abng_train_step (fused) ─────────────────────────

#[test]
fn parity_train_step_chain_head_eval_mir_match() {
    // Phase 0.7 Item 4 — `abng_train_step` must produce byte-identical
    // chain heads under both AST-eval and MIR-exec. Establishes the
    // satellite-dispatch contract for the new builtin.
    assert_parity(
        "abng_train_step + chain_head",
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
        let leaf = abng_train_step(g, x, phi, 0.7);
        print(leaf);
        print(abng_chain_head(g));
        "#,
    );
}
