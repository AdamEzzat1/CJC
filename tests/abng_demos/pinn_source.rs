//! CJC-Lang source for the ABNG PINN per-region uncertainty demo.
//!
//! Workload: 1-D heat equation `∂u/∂t = α ∂²u/∂x²`, Dirichlet BCs, IC
//! `u(x,0) = sin(πx)`, evaluated at `t = 0.1, α = 1`. Analytical
//! solution `u(x,t) = exp(-π²·0.1) · sin(πx)`. The demo fits a BLR
//! posterior per codebook leaf over the basis `[1, x, sin(πx),
//! cos(πx)]` — exactly representable, so BLR converges to the truth
//! given enough samples.
//!
//! The headline benefit (an MLP can't deliver this for free):
//! after asymmetric training, the *minimum* edge epistemic_leverage
//! exceeds the *maximum* interior epistemic_leverage. The leverage
//! signal cleanly partitions evidence-rich vs evidence-poor x-regions.

pub const SOURCE: &str = r#"
fn pi() -> f64 { 3.141592653589793 }

// Analytical solution at t=0.1, alpha=1:
// u(x, 0.1) = exp(-pi^2 * 0.1) * sin(pi*x).
fn analytical_u(x: f64) -> f64 {
    let decay = exp(-0.9869604401089358);
    decay * sin(pi() * x)
}

// 4-D BLR feature vector [1, x, sin(pi*x), cos(pi*x)] as a 2-D
// [1, 4] tensor — the single-row update shape required by
// abng_blr_update.
fn pinn_features_2d(x: f64) -> Tensor {
    Tensor.from_vec([1.0, x, sin(pi() * x), cos(pi() * x)], [1, 4])
}

// Same features as a 1-D [4] tensor — the shape required by
// abng_blr_predict.
fn pinn_features_1d(x: f64) -> Tensor {
    Tensor.from_vec([1.0, x, sin(pi() * x), cos(pi() * x)], [4])
}

fn build_pinn_graph(seed: i64) -> i64 {
    let g = abng_new(seed);
    let codebook = Tensor.from_vec([0.25, 0.5, 0.75], [1, 3]);
    abng_set_codebook(g, codebook);
    let hidden = Tensor.from_vec([4.0], [1]);
    abng_set_leaf_head(g, 1, hidden, 1, "tanh");
    abng_set_blr_prior(g, 1.0, 1.0, 0.5);
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
    abng_add_node(g, 0, 0);
    abng_add_node(g, 0, 1);
    abng_add_node(g, 0, 2);
    abng_add_node(g, 0, 3);
    g
}

fn route_leaf(g: i64, x: f64) -> i64 {
    let x_t = Tensor.from_vec([x], [1]);
    let prefix = abng_encode_prefix(g, x_t);
    let evidence = abng_descend(g, prefix);
    int(evidence.get([1]))
}

fn train_one(g: i64, x: f64) {
    let leaf = route_leaf(g, x);
    let phi = pinn_features_2d(x);
    let y_val = analytical_u(x);
    let y = Tensor.from_vec([y_val], [1]);
    abng_blr_update(g, leaf, phi, y);
    abng_observe(g, leaf, y_val);
}

// Probe: predict at a query point and return mean+leverage.
// Returns a 2-element f64 array [mean, lev] for printable output.
fn probe(g: i64, x: f64) -> Any {
    let leaf = route_leaf(g, x);
    let phi = pinn_features_1d(x);
    let pred = abng_blr_predict(g, leaf, phi);
    [pred.get([0]), pred.get([1])]
}

fn main() {
    let g = build_pinn_graph(7);
    abng_stamp_provenance(g, 0,
        "a0b1c2d3e4f5061728394a5b6c7d8e9f10213243546576879889a9bacbdcedfe"
    );

    // Asymmetric training: 32 interior samples in [0.25, 0.75],
    // 4 samples each in [0, 0.25) and [0.75, 1].
    let k = 0;
    while k < 32 {
        let x = 0.25 + 0.5 * (float(k) + 0.5) / 32.0;
        train_one(g, x);
        k = k + 1;
    }
    let k2 = 0;
    while k2 < 4 {
        let x_lo = 0.0625 * (float(k2) + 0.5);
        let x_hi = 0.75 + 0.0625 * (float(k2) + 0.5);
        train_one(g, x_lo);
        train_one(g, x_hi);
        k2 = k2 + 1;
    }

    // Lineage hashes.
    print("chain_head: " + abng_chain_head(g));
    print("verify_chain: " + to_string(abng_verify_chain(g)));
    print("audit_len: " + to_string(abng_audit_len(g)));

    // Probe predictions at interior + edge points.
    let i30 = probe(g, 0.30);
    let i50 = probe(g, 0.50);
    let i70 = probe(g, 0.70);
    let e10 = probe(g, 0.10);
    let e90 = probe(g, 0.90);

    // Per-region uncertainty signal (the headline benefit).
    let lev_int_30 = i30[1];
    let lev_int_50 = i50[1];
    let lev_int_70 = i70[1];
    let lev_edge_10 = e10[1];
    let lev_edge_90 = e90[1];

    print("lev_int_30: " + to_string(lev_int_30));
    print("lev_int_50: " + to_string(lev_int_50));
    print("lev_int_70: " + to_string(lev_int_70));
    print("lev_edge_10: " + to_string(lev_edge_10));
    print("lev_edge_90: " + to_string(lev_edge_90));

    // Max interior vs min edge — the strict separation property.
    let max_int = lev_int_30;
    if lev_int_50 > max_int { max_int = lev_int_50; }
    if lev_int_70 > max_int { max_int = lev_int_70; }
    let min_edge = lev_edge_10;
    if lev_edge_90 < min_edge { min_edge = lev_edge_90; }
    print("max_interior_lev: " + to_string(max_int));
    print("min_edge_lev: " + to_string(min_edge));
    let edge_strictly_higher = min_edge > max_int;
    print("edge_strictly_higher: " + to_string(edge_strictly_higher));

    // Fit-quality at probe points.
    let truth_50 = analytical_u(0.50);
    let mean_50 = i50[0];
    let abs_err_50 = mean_50 - truth_50;
    if abs_err_50 < 0.0 { abs_err_50 = 0.0 - abs_err_50; }
    print("truth_50: " + to_string(truth_50));
    print("mean_50: " + to_string(mean_50));
    print("abs_err_50: " + to_string(abs_err_50));
    let interior_fit_ok = abs_err_50 < 0.05;
    print("interior_fit_ok: " + to_string(interior_fit_ok));
}
"#;
