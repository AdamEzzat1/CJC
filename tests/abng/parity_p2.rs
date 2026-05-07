//! Phase 0.2 — AST eval ↔ MIR exec parity tests for the new `abng_*`
//! builtins. Each test runs a `.cjcl` snippet through both backends and
//! asserts byte-identical printed output.

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
    reset_arena();
    match backend {
        Backend::Eval => {
            let mut interp = cjc_eval::Interpreter::new(seed);
            interp
                .exec(&program)
                .unwrap_or_else(|e| panic!("eval failed:\n{src}\nerror: {e:?}"));
            interp.output
        }
        Backend::Mir => {
            let (_v, exec) = cjc_mir_exec::run_program_with_executor(&program, seed)
                .unwrap_or_else(|e| panic!("MIR-exec failed:\n{src}\nerror: {e:?}"));
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

// ─── Structural mutation ──────────────────────────────────────────

#[test]
fn parity_add_node_basic() {
    assert_parity(
        "add_node + node_count",
        r#"
        let g = abng_new(0);
        let n = abng_add_node(g, 0, 7);
        print(n);
        print(abng_node_count(g));
        print(abng_node_parent(g, n));
        "#,
    );
}

#[test]
fn parity_node_kind_progression() {
    assert_parity(
        "node_kind through promotions",
        r#"
        let g = abng_new(0);
        print(abng_node_kind(g, 0));         // 0 = None
        abng_add_node(g, 0, 0);
        print(abng_node_kind(g, 0));         // 1 = Node4
        abng_add_node(g, 0, 1);
        abng_add_node(g, 0, 2);
        abng_add_node(g, 0, 3);
        abng_add_node(g, 0, 4);
        print(abng_node_kind(g, 0));         // 2 = Node16
        "#,
    );
}

#[test]
fn parity_node_child_lookup() {
    assert_parity(
        "node_child returns id or -1",
        r#"
        let g = abng_new(0);
        let n = abng_add_node(g, 0, 7);
        print(abng_node_child(g, 0, 7));     // n
        print(abng_node_child(g, 0, 99));    // -1
        "#,
    );
}

// ─── Codebook + prefix encoder ────────────────────────────────────

#[test]
fn parity_codebook_dims_zero_when_unset() {
    assert_parity(
        "codebook_dims = 0 by default",
        r#"
        let g = abng_new(0);
        print(abng_codebook_dims(g));
        print(abng_codebook_hash(g));
        "#,
    );
}

#[test]
fn parity_set_codebook_then_encode() {
    assert_parity(
        "encode_prefix returns expected bins",
        r#"
        let g = abng_new(0);
        let bounds = Tensor.from_vec([0.5, 1.5, 2.5, 0.5, 1.5, 2.5], [2, 3]);
        abng_set_codebook(g, bounds);
        print(abng_codebook_dims(g));
        let x = Tensor.from_vec([0.0, 2.0], [2]);
        print(abng_encode_prefix(g, x));
        "#,
    );
}

// ─── Descend / route_path ─────────────────────────────────────────

#[test]
fn parity_descend_root_only() {
    assert_parity(
        "descend with no matching child",
        r#"
        let g = abng_new(0);
        let p = Tensor.from_vec([5.0, 1.0], [2]);
        print(abng_descend(g, p));
        "#,
    );
}

#[test]
fn parity_descend_match_then_bail() {
    assert_parity(
        "descend matches one then bails",
        r#"
        let g = abng_new(0);
        let n1 = abng_add_node(g, 0, 7);
        let p = Tensor.from_vec([7.0, 99.0], [2]);
        let r = abng_descend(g, p);
        print(r);
        "#,
    );
}

#[test]
fn parity_route_path_full_traversal() {
    assert_parity(
        "route_path returns root..leaf",
        r#"
        let g = abng_new(0);
        let n1 = abng_add_node(g, 0, 2);
        let n2 = abng_add_node(g, n1, 4);
        let p = Tensor.from_vec([2.0, 4.0], [2]);
        print(abng_route_path(g, p));
        "#,
    );
}

// ─── Serialize / replay (multinode) ───────────────────────────────

#[test]
fn parity_multinode_serialize_replay() {
    assert_parity(
        "round-trip preserves multi-node chain head",
        r#"
        let g = abng_new(0);
        abng_add_node(g, 0, 1);
        abng_add_node(g, 0, 2);
        abng_observe(g, 0, 1.5);
        abng_observe(g, 1, 2.5);
        let h_before = abng_chain_head(g);
        let blob = abng_serialize(g);
        let g2 = abng_replay(blob);
        let h_after = abng_chain_head(g2);
        print(h_before == h_after);
        "#,
    );
}

#[test]
fn parity_double_run_multinode_chain_head() {
    let body = r#"
        let g = abng_new(7);
        abng_add_node(g, 0, 1);
        abng_add_node(g, 0, 2);
        abng_add_node(g, 1, 3);
        abng_observe(g, 0, 0.1);
        abng_observe(g, 1, 0.2);
        abng_observe(g, 2, 0.3);
        abng_observe(g, 3, 0.4);
        print(abng_chain_head(g));
    "#;
    let a = run(Backend::Eval, body, 0);
    let b = run(Backend::Eval, body, 0);
    assert_eq!(a, b, "eval double-run differs");
    let c = run(Backend::Mir, body, 0);
    let d = run(Backend::Mir, body, 0);
    assert_eq!(c, d, "MIR double-run differs");
    assert_eq!(a, c, "eval ↔ MIR differs");
}

#[test]
fn parity_per_node_chain_isolation() {
    assert_parity(
        "per-node chain only advances on its own observe",
        r#"
        let g = abng_new(0);
        let n1 = abng_add_node(g, 0, 1);
        let h0_before = abng_node_stats_chain_head(g, 0);
        let h1_before = abng_node_stats_chain_head(g, n1);
        abng_observe(g, n1, 5.0);
        let h0_after = abng_node_stats_chain_head(g, 0);
        let h1_after = abng_node_stats_chain_head(g, n1);
        // node 0 chain stayed put; node 1 advanced.
        print(h0_before == h0_after);
        print(h1_before == h1_after);
        "#,
    );
}
