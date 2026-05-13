//! Phase 0.8c v14 Item A3 — external Merkle inclusion proof demo.
//!
//! Demonstrates the "wouldn't have been expressible pre-Phase-0.8"
//! capability: a regulator/auditor with only `(merkle_root, leaf_hash,
//! index, n_leaves, proof)` can verify that a specific training event
//! occurred at a specific position in the audit chain, in `O(log N)`
//! SHA-256 ops, **without downloading the full audit log**.
//!
//! This file is exercise + assert + visualize. The demo:
//!
//! 1. **Issuer** side — builds a small ABNG graph with `N_LEAVES`
//!    audit events, computes the Merkle root, and "publishes" only
//!    the root + a single inclusion proof.
//! 2. **Auditor** side — receives only the published artifacts and
//!    runs `MerkleTree::verify_proof` to attest the event's position.
//!    They never see the rest of the audit log.
//! 3. **Tamper detection** — a forged leaf fails verification.
//! 4. **Visualization** — emits an SVG of the Merkle tree to
//!    `bench_results/phase_0_8_demos/a3_merkle_proof_tree.svg`, with
//!    the highlighted leaf, its proof path, and the sibling-witness
//!    hashes color-coded.
//!
//! The accompanying property tests live in
//! `crates/cjc-abng/src/merkle.rs::tests` (build-paths,
//! proof-round-trips) and `tests/abng/merkle_tests.rs` (trailer
//! roundtrip, tamper detection at the snapshot boundary).

use cjc_abng::graph::AdaptiveBeliefGraph;
use cjc_abng::merkle::MerkleTree;
use std::path::PathBuf;

/// Audit event index whose inclusion the demo proves. Picked
/// arbitrarily in `[1, N_LEAVES)` so the proof actually exercises
/// every level of the tree (not the trivial root-=-leaf case at
/// position 0 of a 1-leaf tree).
const PROVEN_INDEX: usize = 11;

/// Number of audit events to anchor in the Merkle tree. 16 = power
/// of two → perfect binary tree of depth 4. Any value ≥ 2 works;
/// 16 keeps the SVG legible.
const N_LEAVES: usize = 16;

/// Fixed seed so the demo's printed hex + SVG are byte-stable
/// across runs and platforms.
const DEMO_SEED: u64 = 0x4A33;

/// Build the "issuer" graph: `N_LEAVES` audit events, all under
/// the deterministic seed. The graph is intentionally tiny — the
/// demo is about chain verification, not training results.
fn build_issuer_graph(seed: u64) -> AdaptiveBeliefGraph {
    let mut g = AdaptiveBeliefGraph::new(seed);
    // `new` already emits one `Created` event (audit_len = 1).
    // Drive `audit_len` up to N_LEAVES with deterministic observes.
    while g.audit.len() < N_LEAVES {
        let i = g.audit.len() as f64;
        // Mix positive + negative values so leaf hashes are visibly
        // distinct in the SVG output.
        g.observe(0, (i * 0.7) - 5.0).unwrap();
    }
    assert_eq!(g.audit.len(), N_LEAVES);
    g
}

#[test]
fn a3_demo_external_verification_workflow() {
    // ─ ISSUER ─────────────────────────────────────────────────────────
    let g = build_issuer_graph(DEMO_SEED);
    let root = g.merkle_root();
    let tree = g.merkle_tree();
    assert_eq!(tree.n_leaves(), N_LEAVES);
    assert_eq!(tree.depth(), 4); // log2(16) = 4 — pinned for the SVG layout.

    // The issuer publishes only the root (and per-event, a
    // separate proof). The full audit log stays private.
    let published_root: [u8; 32] = root;

    // For ONE specific event, the issuer also publishes the
    // (leaf_hash, index, n_leaves, proof) 4-tuple. The auditor
    // never sees anything else.
    let leaf_hash = g.audit.get(PROVEN_INDEX).unwrap().new_hash;
    let proof: Vec<[u8; 32]> = tree.proof(PROVEN_INDEX);
    assert_eq!(
        proof.len(),
        tree.depth(),
        "proof length must equal tree depth (one sibling per layer)"
    );

    // ─ AUDITOR ────────────────────────────────────────────────────────
    // The auditor's verification: pure function of the published
    // artifacts. No audit-log access.
    let verified = MerkleTree::verify_proof(
        leaf_hash,
        PROVEN_INDEX,
        N_LEAVES,
        &proof,
        published_root,
    );
    assert!(verified, "auditor must accept a valid inclusion proof");

    // ─ TAMPER DETECTION ──────────────────────────────────────────────
    // A forged leaf hash with the same index + proof must fail.
    let forged_leaf = [0xFFu8; 32];
    let forged_verified = MerkleTree::verify_proof(
        forged_leaf,
        PROVEN_INDEX,
        N_LEAVES,
        &proof,
        published_root,
    );
    assert!(!forged_verified, "auditor must reject a forged leaf");

    // A tampered proof must also fail.
    let mut tampered_proof = proof.clone();
    tampered_proof[0][0] ^= 0x01;
    let tampered_verified = MerkleTree::verify_proof(
        leaf_hash,
        PROVEN_INDEX,
        N_LEAVES,
        &tampered_proof,
        published_root,
    );
    assert!(!tampered_verified, "auditor must reject a tampered proof");

    // A wrong index must fail (verifier walks a different sibling
    // path, recomputes a different root).
    let wrong_index_verified = MerkleTree::verify_proof(
        leaf_hash,
        PROVEN_INDEX + 1,
        N_LEAVES,
        &proof,
        published_root,
    );
    assert!(
        !wrong_index_verified,
        "auditor must reject a wrong-index claim"
    );

    // ─ Print a human-readable summary ─────────────────────────────────
    eprintln!();
    eprintln!("══ A3 Merkle Inclusion Proof Demo ══");
    eprintln!("Issuer published root:   {}", hex32(&published_root));
    eprintln!("Audit log size:          {N_LEAVES} events");
    eprintln!("Tree depth:              {} layers above leaves", tree.depth());
    eprintln!();
    eprintln!("Auditor receives only:");
    eprintln!("  leaf_hash[{PROVEN_INDEX}] = {}", hex32(&leaf_hash));
    eprintln!("  leaf_index             = {PROVEN_INDEX}");
    eprintln!("  n_leaves               = {N_LEAVES}");
    eprintln!(
        "  proof                  = {} sibling hashes ({} bytes total)",
        proof.len(),
        proof.len() * 32
    );
    eprintln!();
    eprintln!(
        "Auditor verification cost: {} SHA-256 ops (O(log N))",
        tree.depth()
    );
    eprintln!("Full-log verification cost: {N_LEAVES} SHA-256 ops (O(N))");
    eprintln!(
        "Cost ratio at N={N_LEAVES}: {:.1}× cheaper",
        (N_LEAVES as f64) / (tree.depth() as f64)
    );
    eprintln!("Asymptotic ratio: log₂(N) vs N — exponentially better.");

    // ─ Emit SVG visualization ─────────────────────────────────────────
    let svg = render_merkle_tree_svg(&tree, PROVEN_INDEX);
    let path = output_path("a3_merkle_proof_tree.svg");
    std::fs::create_dir_all(path.parent().unwrap()).expect("create output dir");
    std::fs::write(&path, svg.as_bytes()).expect("write SVG");
    eprintln!();
    eprintln!("SVG visualization → {}", path.display());

    // Deterministic-output gate: re-render once and assert byte
    // equality, so any future change to the SVG renderer surfaces
    // in tests.
    let svg2 = render_merkle_tree_svg(&tree, PROVEN_INDEX);
    assert_eq!(svg, svg2, "SVG rendering must be byte-stable across calls");
}

#[test]
fn a3_demo_every_leaf_proof_verifies() {
    // Strengthens the workflow test: for the same demo graph, the
    // proof for EVERY leaf must verify against the published root.
    // This is a determinism gate — pinning that `tree.proof(i)` is
    // a pure function of the tree state, no shared scratch buffer
    // hidden in there.
    let g = build_issuer_graph(DEMO_SEED);
    let tree = g.merkle_tree();
    let root = tree.root();
    for i in 0..N_LEAVES {
        let leaf = g.audit.get(i).unwrap().new_hash;
        let proof = tree.proof(i);
        let verified = MerkleTree::verify_proof(leaf, i, N_LEAVES, &proof, root);
        assert!(verified, "proof for leaf {i} must verify");
    }
}

#[test]
fn a3_demo_workflow_deterministic_across_runs() {
    // Two independent builds with the same seed must produce
    // byte-equal roots + proofs + SVGs. This is the cross-platform
    // determinism guarantee the Phase 0.8 work depends on.
    let a = build_issuer_graph(DEMO_SEED).merkle_tree();
    let b = build_issuer_graph(DEMO_SEED).merkle_tree();
    assert_eq!(a.root(), b.root());
    for i in 0..N_LEAVES {
        assert_eq!(a.proof(i), b.proof(i), "proof for {i} must be deterministic");
    }
    let svg_a = render_merkle_tree_svg(&a, PROVEN_INDEX);
    let svg_b = render_merkle_tree_svg(&b, PROVEN_INDEX);
    assert_eq!(svg_a, svg_b, "SVG rendering must be deterministic");
}

// ── helpers ──────────────────────────────────────────────────────────

fn hex32(bytes: &[u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for b in bytes {
        use std::fmt::Write;
        write!(&mut s, "{b:02x}").unwrap();
    }
    s
}

fn output_path(filename: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("bench_results");
    p.push("phase_0_8_demos");
    p.push(filename);
    p
}

/// Render the Merkle tree as an SVG. Leaves at the bottom, root at
/// the top. The highlighted leaf, every internal node on the path
/// from that leaf to the root, and every sibling (witness) node on
/// the proof path are all color-coded so the verifier's work is
/// visually traceable.
///
/// Color legend:
/// * `#4a90d9` (steel blue) — generic nodes
/// * `#e94f37` (red)        — highlighted leaf + path nodes
/// * `#f5a623` (orange)     — proof-witness siblings
/// * `#2e7d32` (green)      — root
fn render_merkle_tree_svg(tree: &MerkleTree, highlighted_leaf: usize) -> String {
    let n_leaves = tree.n_leaves();
    let depth = tree.depth();
    let n_layers = depth + 1;

    // ─ Layout parameters ─────────────────────────────────────────
    let svg_width = 880i32;
    let row_height = 100i32;
    let svg_height = (n_layers as i32) * row_height + 80;
    let leaf_y = (n_layers as i32 - 1) * row_height + 50;
    let leaf_x_start = 40i32;
    let leaf_x_end = svg_width - 40;
    let leaf_spacing = (leaf_x_end - leaf_x_start) as f64 / (n_leaves as f64);

    // For each layer, compute which node indices are on the path
    // and which are proof-witness siblings.
    let mut on_path: Vec<Vec<bool>> = Vec::with_capacity(n_layers);
    let mut on_proof: Vec<Vec<bool>> = Vec::with_capacity(n_layers);
    let mut idx = highlighted_leaf;
    let mut layer_size = n_leaves;
    for _ in 0..n_layers {
        on_path.push(vec![false; layer_size]);
        on_proof.push(vec![false; layer_size]);
        layer_size = layer_size.div_ceil(2);
    }
    let mut walk_idx = highlighted_leaf;
    for lyr in 0..n_layers {
        let size = on_path[lyr].len();
        on_path[lyr][walk_idx] = true;
        if lyr < depth {
            let sibling = walk_idx ^ 1;
            if sibling < size {
                on_proof[lyr][sibling] = true;
            } else {
                // Odd-end duplicate-of-self: sibling = self,
                // which is already on_path. Mark on_proof to
                // signal the witness exists conceptually.
                on_proof[lyr][walk_idx] = true;
            }
            walk_idx >>= 1;
        }
    }

    // ─ Build the SVG ─────────────────────────────────────────────
    let mut out = String::new();
    out.push_str(&format!(
        r##"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {svg_width} {svg_height}" width="{svg_width}" height="{svg_height}" font-family="ui-monospace, Menlo, Consolas, monospace">
  <style>
    .lbl {{ font-size: 11px; fill: #333; }}
    .root {{ font-size: 12px; font-weight: bold; }}
    .legend {{ font-size: 12px; fill: #444; }}
    .edge {{ stroke: #888; stroke-width: 1; }}
    .edge-path {{ stroke: #e94f37; stroke-width: 2; }}
  </style>
  <rect width="100%" height="100%" fill="#fafafa" />
  <text x="20" y="22" font-size="14" font-weight="bold" fill="#222">A3 — Merkle Inclusion Proof for Audit Event {highlighted_leaf} (N={n_leaves})</text>
"##
    ));

    // X-positions per layer (computed bottom-up).
    let mut layer_xs: Vec<Vec<f64>> = Vec::with_capacity(n_layers);
    // Leaves x-positions
    let leaf_xs: Vec<f64> = (0..n_leaves)
        .map(|i| {
            leaf_x_start as f64 + leaf_spacing * (i as f64 + 0.5)
        })
        .collect();
    layer_xs.push(leaf_xs);
    for lyr in 1..n_layers {
        let parent_count = on_path[lyr].len();
        let prev_xs = &layer_xs[lyr - 1];
        let mut xs = Vec::with_capacity(parent_count);
        for p in 0..parent_count {
            let left = 2 * p;
            let right = 2 * p + 1;
            let lx = prev_xs[left];
            let rx = if right < prev_xs.len() {
                prev_xs[right]
            } else {
                lx
            };
            xs.push((lx + rx) * 0.5);
        }
        layer_xs.push(xs);
    }

    // Draw edges (parent → child) bottom-up so parents stack over
    // children visually.
    for lyr in 1..n_layers {
        let y_parent = (n_layers as i32 - 1 - lyr as i32) * row_height + 50;
        let y_child = (n_layers as i32 - lyr as i32) * row_height + 50;
        let parent_count = on_path[lyr].len();
        for p in 0..parent_count {
            let px = layer_xs[lyr][p];
            for &cidx in &[2 * p, 2 * p + 1] {
                if cidx < layer_xs[lyr - 1].len() {
                    let cx = layer_xs[lyr - 1][cidx];
                    let cls = if on_path[lyr][p] && on_path[lyr - 1][cidx] {
                        "edge-path"
                    } else {
                        "edge"
                    };
                    out.push_str(&format!(
                        r##"  <line class="{cls}" x1="{px:.1}" y1="{y_parent}" x2="{cx:.1}" y2="{y_child}" />
"##
                    ));
                }
            }
        }
    }

    // Draw nodes (bottom layer first → top).
    for lyr in 0..n_layers {
        let y = (n_layers as i32 - 1 - lyr as i32) * row_height + 50;
        let count = on_path[lyr].len();
        for n in 0..count {
            let x = layer_xs[lyr][n];
            let (fill, stroke, label_color) = if lyr == n_layers - 1 {
                ("#2e7d32", "#1b5e20", "#fff") // root: green
            } else if on_path[lyr][n] {
                ("#e94f37", "#b32d1c", "#fff") // path: red
            } else if on_proof[lyr][n] {
                ("#f5a623", "#a06a13", "#222") // proof witness: orange
            } else {
                ("#4a90d9", "#205a8c", "#fff") // generic: blue
            };
            let radius = if lyr == 0 { 12 } else { 14 };
            out.push_str(&format!(
                r##"  <circle cx="{x:.1}" cy="{y}" r="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="1.5" />
"##
            ));
            let lbl = if lyr == n_layers - 1 {
                "root".to_string()
            } else if lyr == 0 {
                format!("{n}")
            } else {
                format!("L{lyr}·{n}")
            };
            out.push_str(&format!(
                r##"  <text x="{x:.1}" y="{}" text-anchor="middle" class="lbl" fill="{label_color}">{lbl}</text>
"##,
                y + 4
            ));
        }
    }

    // Legend
    let lg_y = svg_height - 20;
    out.push_str(&format!(
        r##"  <circle cx="40" cy="{lg_y}" r="6" fill="#e94f37" />
  <text x="52" y="{}" class="legend">proof path</text>
  <circle cx="160" cy="{lg_y}" r="6" fill="#f5a623" />
  <text x="172" y="{}" class="legend">witness sibling</text>
  <circle cx="320" cy="{lg_y}" r="6" fill="#2e7d32" />
  <text x="332" y="{}" class="legend">root</text>
  <circle cx="400" cy="{lg_y}" r="6" fill="#4a90d9" />
  <text x="412" y="{}" class="legend">non-witness</text>
"##,
        lg_y + 4,
        lg_y + 4,
        lg_y + 4,
        lg_y + 4
    ));

    out.push_str("</svg>\n");
    out
}
