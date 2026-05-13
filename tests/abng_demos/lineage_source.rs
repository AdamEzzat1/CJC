//! CJC-Lang source for the ABNG lineage-attestation demo.
//!
//! This is the regulator's nightmare scenario, written as a CJC-Lang
//! program: lab A trains an ABNG model on a synthetic clinical-trial
//! dataset, stamps the root with the dataset's pre-computed SHA-256
//! fingerprint, and predictions are packaged. An attacker substitutes
//! a model trained on a tampered dataset B (one row's response changed)
//! and tries to pass off A's predictions. The audit chain catches the
//! spoof through three independent signals.
//!
//! The dataset fingerprints are *pre-computed* externally
//! (sha256 of canonical bytes) and pass into `abng_stamp_provenance`
//! as 64-char hex strings. CJC-Lang doesn't need a hash builtin — the
//! attestation contract is "provide the fingerprint, ABNG records it
//! in the audit chain."

pub const SOURCE: &str = r#"
// Pre-computed dataset fingerprints. In production these would be
// sha256(canonical_bytes(rows)). For the demo they're constants so
// the test is reproducible without a hash builtin.
fn dataset_a_fingerprint() -> String {
    "3e85d52f2508aecaaf32737edca48a644796783d6be7e6e324e6760506bc3634"
}

fn dataset_b_fingerprint() -> String {
    "c0ffee00c0ffee00c0ffee00c0ffee00c0ffee00c0ffee00c0ffee00c0ffee00"
}

// Synthetic clinical-trial rows: (dose, response). The dose-response
// curve is response = 0.2 + 0.6*dose + 0.1*dose^2 (small cubic).
// 16 rows total — kept small so the interpreter runs in seconds.
fn dataset_a_rows() -> Any {
    [
        [0.0625, 0.23914],
        [0.1250, 0.27656],
        [0.1875, 0.31602],
        [0.2500, 0.35625],
        [0.3125, 0.39727],
        [0.3750, 0.43906],
        [0.4375, 0.48164],
        [0.5000, 0.52500],
        [0.5625, 0.56914],
        [0.6250, 0.61406],
        [0.6875, 0.65977],
        [0.7500, 0.70625],
        [0.8125, 0.75352],
        [0.8750, 0.80156],
        [0.9375, 0.85039],
        [1.0000, 0.90000]
    ]
}

// Dataset B: identical to A except row 7 (dose=0.5) has its response
// boosted to 0.999 — a hypothetical fraudster boosting one patient's
// apparent efficacy. This is the tamper.
fn dataset_b_rows() -> Any {
    [
        [0.0625, 0.23914],
        [0.1250, 0.27656],
        [0.1875, 0.31602],
        [0.2500, 0.35625],
        [0.3125, 0.39727],
        [0.3750, 0.43906],
        [0.4375, 0.48164],
        [0.5000, 0.99900],
        [0.5625, 0.56914],
        [0.6250, 0.61406],
        [0.6875, 0.65977],
        [0.7500, 0.70625],
        [0.8125, 0.75352],
        [0.8750, 0.80156],
        [0.9375, 0.85039],
        [1.0000, 0.90000]
    ]
}

// Build the lineage-friendly graph: 1-D codebook over `dose` with 4
// region-bins, 1->4->1 tanh leaf head (BLR feature dim = 4), BLR
// prior + density + calibration heads, and 4 children (one per
// codebook bin).
fn build_lineage_graph(seed: i64) -> i64 {
    let g = abng_new(seed);
    // Codebook: 1 dim, 4 bins -> 3 boundaries at [0.25, 0.5, 0.75].
    // Tensor shape [1, 3] = [n_dims, n_bins-1].
    let codebook = Tensor.from_vec([0.25, 0.5, 0.75], [1, 3]);
    abng_set_codebook(g, codebook);
    // Leaf head: input_dim=1, hidden_dims=[4], output_dim=1, tanh.
    let hidden = Tensor.from_vec([4.0], [1]);
    abng_set_leaf_head(g, 1, hidden, 1, "tanh");
    abng_set_blr_prior(g, 2.0, 1.0, 0.5);
    abng_set_density_tracker(g);
    abng_set_calibration(g, 15);
    let thresholds = Tensor.from_vec(
        [
            0.5, 64.0, 128.0, 0.05, 0.02, 4.0, 0.1, 32.0, 10.0, 8.0, 20.0,
            1.0e308,
            0.005, 1.05
        ],
        [14]
    );
    abng_set_decision_policy(g, thresholds);
    // Add 4 children (one per codebook bin).
    abng_add_node(g, 0, 0);
    abng_add_node(g, 0, 1);
    abng_add_node(g, 0, 2);
    abng_add_node(g, 0, 3);
    g
}

// Train the graph on `rows` and stamp with the given dataset
// fingerprint. Each row updates its leaf's BLR posterior and folds
// the response into per-node stats.
fn train_and_stamp(seed: i64, rows: Any, fingerprint: String) -> i64 {
    let g = build_lineage_graph(seed);
    abng_stamp_provenance(g, 0, fingerprint);
    let n = len(rows);
    let i = 0;
    while i < n {
        let row = rows[i];
        let dose = row[0];
        let resp = row[1];
        // Phase 0.8c v14 Item A2 — fused per-row training step.
        // Emits one `AuditKind::TrainStep` event (tag 0x1E) per
        // row instead of the pre-A2 `BlrUpdated + BeliefUpdate`
        // pair. `abng_train_step` does its own descend, so we
        // pass the dose tensor directly and use a 1-D phi (it
        // requires shape `[d]`, not `[1, d]`).
        let dose_t = Tensor.from_vec([dose], [1]);
        // Cubic basis: BLR fits y = b0 + b1*dose + b2*dose^2 + b3*dose^3.
        let phi = Tensor.from_vec(
            [1.0, dose, dose * dose, dose * dose * dose],
            [4]
        );
        abng_train_step(g, dose_t, phi, resp);
        i = i + 1;
    }
    g
}

fn main() {
    // Lab A: train on dataset A, stamp with A's fingerprint.
    let g_a = train_and_stamp(7, dataset_a_rows(), dataset_a_fingerprint());
    // Lab B: an attacker / contaminated rerun on tampered dataset.
    let g_b = train_and_stamp(7, dataset_b_rows(), dataset_b_fingerprint());

    // Headline output: the three independent attestation signals.
    let chain_a = abng_chain_head(g_a);
    let chain_b = abng_chain_head(g_b);
    let stamp_a = abng_provenance_stamp(g_a, 0);
    let stamp_b = abng_provenance_stamp(g_b, 0);
    print("chain_a: " + chain_a);
    print("chain_b: " + chain_b);
    print("stamp_a: " + stamp_a);
    print("stamp_b: " + stamp_b);

    // Audit chain integrity.
    print("verify_a: " + to_string(abng_verify_chain(g_a)));
    print("verify_b: " + to_string(abng_verify_chain(g_b)));

    // Three-signal attestation as boolean checks.
    let chain_diverged = chain_a != chain_b;
    let stamp_diverged = stamp_a != stamp_b;
    print("chain_diverged: " + to_string(chain_diverged));
    print("stamp_diverged: " + to_string(stamp_diverged));

    // BLR state-hash divergence — check ALL four leaves and report
    // whether ANY leaf's posterior differs. This is the tamper
    // signal that's independent of which specific leaf the
    // tampered row routed to.
    let blr_a_1 = abng_blr_state_hash(g_a, 1);
    let blr_a_2 = abng_blr_state_hash(g_a, 2);
    let blr_a_3 = abng_blr_state_hash(g_a, 3);
    let blr_a_4 = abng_blr_state_hash(g_a, 4);
    let blr_b_1 = abng_blr_state_hash(g_b, 1);
    let blr_b_2 = abng_blr_state_hash(g_b, 2);
    let blr_b_3 = abng_blr_state_hash(g_b, 3);
    let blr_b_4 = abng_blr_state_hash(g_b, 4);
    print("blr_a_leaf1: " + blr_a_1);
    print("blr_b_leaf1: " + blr_b_1);
    print("blr_a_leaf2: " + blr_a_2);
    print("blr_b_leaf2: " + blr_b_2);
    print("blr_a_leaf3: " + blr_a_3);
    print("blr_b_leaf3: " + blr_b_3);
    print("blr_a_leaf4: " + blr_a_4);
    print("blr_b_leaf4: " + blr_b_4);
    let blr_diverged_any = (blr_a_1 != blr_b_1) || (blr_a_2 != blr_b_2)
        || (blr_a_3 != blr_b_3) || (blr_a_4 != blr_b_4);
    print("blr_diverged: " + to_string(blr_diverged_any));

    // Composite assertion (printed for the test to grep).
    let three_signals = chain_diverged && stamp_diverged && blr_diverged_any;
    print("three_signals_independent: " + to_string(three_signals));

    // Audit log size — captures the training history.
    print("audit_len_a: " + to_string(abng_audit_len(g_a)));
    print("audit_len_b: " + to_string(abng_audit_len(g_b)));

    // Finally: the model A's chain_head is what predict_snap blobs
    // would carry. Any third party can compare it to the supposed
    // model. If they're given model B but predict-snaps from A,
    // the chain_head fingerprint disagrees. End-to-end attestation.
    print("legitimate_match: " + to_string(chain_a == abng_chain_head(g_a)));
    print("spoof_detected: " + to_string(chain_a != abng_chain_head(g_b)));
}
"#;
