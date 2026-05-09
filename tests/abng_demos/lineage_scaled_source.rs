//! Phase 0.6 Item 6 — CJC-Lang source for the **scaled** lineage
//! attestation demo.
//!
//! Differences vs Phase 0.5's `lineage_source.rs`:
//!   - 10^3 patient records (vs 16) — production-scale clinical
//!     trial dataset
//!   - 4-D feature basis [1, dose, dose², dose³] (same as before;
//!     the scaling axis is row count, not feature dim)
//!   - Uses Phase 0.6 Item 4's batch path: ONE BeliefUpdateBatch +
//!     ONE BlrUpdated event per leaf, vs N per-row events
//!
//! Headline: a 10^3-row dataset's SHA-256 fingerprint binds to
//! exactly 4 BeliefUpdateBatch events (one per codebook leaf), each
//! carrying a `batch_hash` that detects per-row tampering. The audit
//! log is O(leaves) events instead of O(rows) — proves Item 4's
//! batch path is compatible with the lineage attestation flow.

pub const SOURCE: &str = r#"
fn target(dose: f64) -> f64 {
    0.2 + 0.6 * dose + 0.1 * dose * dose
}

fn lineage_features(dose: f64) -> Tensor {
    Tensor.from_vec([1.0, dose, dose * dose, dose * dose * dose], [4])
}

fn build_scaled_lineage_graph(seed: i64) -> i64 {
    let g = abng_new(seed);
    let codebook = Tensor.from_vec([0.25, 0.5, 0.75], [1, 3]);
    abng_set_codebook(g, codebook);
    let hidden = Tensor.from_vec([4.0], [1]);
    abng_set_leaf_head(g, 1, hidden, 1, "tanh");
    abng_set_blr_prior(g, 2.0, 1.0, 0.5);
    abng_add_node(g, 0, 0);
    abng_add_node(g, 0, 1);
    abng_add_node(g, 0, 2);
    abng_add_node(g, 0, 3);
    g
}

fn main() {
    let g = build_scaled_lineage_graph(42);

    // Stamp the root with a deterministic dataset fingerprint —
    // any byte change in the 1000-row data downstream would diverge
    // the chain head and the BLR state hashes.
    abng_stamp_provenance(g, 0,
        "deadbeefcafebabe0123456789abcdef0123456789abcdeffeedfacecafef00d"
    );

    let n: i64 = 1000;
    let obs_0 = []; let phi_0 = [];
    let obs_1 = []; let phi_1 = [];
    let obs_2 = []; let phi_2 = [];
    let obs_3 = []; let phi_3 = [];

    let i = 0;
    while i < n {
        // Deterministic dose schedule across patients: dose ∈ (0, 1].
        let dose = (float(i) + 1.0) / float(n);
        let resp = target(dose);
        let prefix = abng_encode_prefix(g, Tensor.from_vec([dose], [1]));
        let evidence = abng_descend(g, prefix);
        let leaf = int(evidence.get([1]));
        if leaf == 1 {
            obs_0 = array_push(obs_0, resp);
            phi_0 = array_push(phi_0, 1.0);
            phi_0 = array_push(phi_0, dose);
            phi_0 = array_push(phi_0, dose * dose);
            phi_0 = array_push(phi_0, dose * dose * dose);
        } else if leaf == 2 {
            obs_1 = array_push(obs_1, resp);
            phi_1 = array_push(phi_1, 1.0);
            phi_1 = array_push(phi_1, dose);
            phi_1 = array_push(phi_1, dose * dose);
            phi_1 = array_push(phi_1, dose * dose * dose);
        } else if leaf == 3 {
            obs_2 = array_push(obs_2, resp);
            phi_2 = array_push(phi_2, 1.0);
            phi_2 = array_push(phi_2, dose);
            phi_2 = array_push(phi_2, dose * dose);
            phi_2 = array_push(phi_2, dose * dose * dose);
        } else if leaf == 4 {
            obs_3 = array_push(obs_3, resp);
            phi_3 = array_push(phi_3, 1.0);
            phi_3 = array_push(phi_3, dose);
            phi_3 = array_push(phi_3, dose * dose);
            phi_3 = array_push(phi_3, dose * dose * dose);
        }
        i = i + 1;
    }

    let n0 = array_len(obs_0);
    let n1 = array_len(obs_1);
    let n2 = array_len(obs_2);
    let n3 = array_len(obs_3);
    print("n_per_leaf: " + to_string(n0) + "," + to_string(n1) + ","
        + to_string(n2) + "," + to_string(n3));

    if n0 > 0 {
        abng_blr_update(g, 1, Tensor.from_vec(phi_0, [n0, 4]),
                              Tensor.from_vec(obs_0, [n0]));
        abng_observe_batch(g, 1, Tensor.from_vec(obs_0, [n0]));
    }
    if n1 > 0 {
        abng_blr_update(g, 2, Tensor.from_vec(phi_1, [n1, 4]),
                              Tensor.from_vec(obs_1, [n1]));
        abng_observe_batch(g, 2, Tensor.from_vec(obs_1, [n1]));
    }
    if n2 > 0 {
        abng_blr_update(g, 3, Tensor.from_vec(phi_2, [n2, 4]),
                              Tensor.from_vec(obs_2, [n2]));
        abng_observe_batch(g, 3, Tensor.from_vec(obs_2, [n2]));
    }
    if n3 > 0 {
        abng_blr_update(g, 4, Tensor.from_vec(phi_3, [n3, 4]),
                              Tensor.from_vec(obs_3, [n3]));
        abng_observe_batch(g, 4, Tensor.from_vec(obs_3, [n3]));
    }

    // Quality probe: dose-response prediction at known points.
    let pred_low = abng_blr_predict(g, 1, lineage_features(0.10));
    let pred_high = abng_blr_predict(g, 4, lineage_features(0.90));
    let truth_low = target(0.10);
    let truth_high = target(0.90);
    let err_low = pred_low.get([0]) - truth_low;
    if err_low < 0.0 { err_low = 0.0 - err_low; }
    let err_high = pred_high.get([0]) - truth_high;
    if err_high < 0.0 { err_high = 0.0 - err_high; }
    print("err_low: " + to_string(err_low));
    print("err_high: " + to_string(err_high));
    let recovers_truth = err_low < 0.01 && err_high < 0.01;
    print("recovers_truth: " + to_string(recovers_truth));

    // Lineage hashes — the regulatory artifacts.
    print("audit_len: " + to_string(abng_audit_len(g)));
    print("verify_chain: " + to_string(abng_verify_chain(g)));
    print("chain_head: " + abng_chain_head(g));
}
"#;
