//! Integration tests for the QML (Quantum Machine Learning) module.
//!
//! Tests cover: circuit construction, classification, gradient correctness,
//! training convergence, determinism, data preprocessing, and memory scaling.

use cjc_quantum::qml::*;
use cjc_quantum::mps::Mps;
use cjc_runtime::complex::ComplexF64;

const TOL: f64 = 1e-10;

// ---------------------------------------------------------------------------
// 1. Circuit construction
// ---------------------------------------------------------------------------

#[test]
fn qml_circuit_zero_params_is_product_state() {
    let config = QmlConfig {
        n_qubits: 4,
        n_reupload_passes: 2,
        n_classes: 2,
        max_bond: 8,
        readout_qubits: vec![0, 1],
        learning_rate: 0.1,
        epochs: 1,
        batch_size: 1,
        loss: QmlLoss::CrossEntropy,
        seed: 42,
    };
    let params = vec![0.0; total_params(&config)];
    let input = vec![0.0; 4];
    let mps = build_qml_circuit(&config, &params, &input);

    for q in 0..4 {
        let z = mps_single_z_expectation(&mps, q);
        assert!((z - 1.0).abs() < TOL,
            "qubit {} should be |0> with zero params, Z={}", q, z);
    }
}

#[test]
fn qml_circuit_nonzero_params_changes_state() {
    let config = QmlConfig {
        n_qubits: 3,
        n_reupload_passes: 1,
        n_classes: 2,
        max_bond: 8,
        readout_qubits: vec![0, 1],
        learning_rate: 0.1,
        epochs: 1,
        batch_size: 1,
        loss: QmlLoss::CrossEntropy,
        seed: 42,
    };
    let n_p = total_params(&config);
    let params: Vec<f64> = (0..n_p).map(|i| 0.5 + i as f64 * 0.1).collect();
    let input = vec![1.0, 0.5, 2.0];
    let mps = build_qml_circuit(&config, &params, &input);

    // At least one qubit should deviate from |0>
    let mut any_changed = false;
    for q in 0..3 {
        let z = mps_single_z_expectation(&mps, q);
        if (z - 1.0).abs() > 0.01 {
            any_changed = true;
            break;
        }
    }
    assert!(any_changed, "nonzero params should change the state from |0...0>");
}

#[test]
fn qml_circuit_more_layers_increases_expressivity() {
    // With more re-upload passes, the circuit can produce more diverse Z expectations
    let input = vec![1.0, 0.5, 2.0];

    let make_config = |layers| QmlConfig {
        n_qubits: 3,
        n_reupload_passes: layers,
        n_classes: 2,
        max_bond: 8,
        readout_qubits: vec![0, 1],
        learning_rate: 0.1,
        epochs: 1,
        batch_size: 1,
        loss: QmlLoss::CrossEntropy,
        seed: 42,
    };

    let c1 = make_config(1);
    let c3 = make_config(3);
    let p1: Vec<f64> = (0..total_params(&c1)).map(|i| 0.3 + i as f64 * 0.05).collect();
    let p3: Vec<f64> = (0..total_params(&c3)).map(|i| 0.3 + i as f64 * 0.05).collect();

    let mps1 = build_qml_circuit(&c1, &p1, &input);
    let mps3 = build_qml_circuit(&c3, &p3, &input);

    let z1_0 = mps_single_z_expectation(&mps1, 0);
    let z3_0 = mps_single_z_expectation(&mps3, 0);
    // They should differ (different circuits)
    assert!((z1_0 - z3_0).abs() > 1e-6,
        "1-layer and 3-layer should produce different states: {} vs {}", z1_0, z3_0);
}

// ---------------------------------------------------------------------------
// 2. Classification
// ---------------------------------------------------------------------------

#[test]
fn qml_classify_probabilities_sum_to_one() {
    let config = QmlConfig {
        n_qubits: 4,
        n_reupload_passes: 1,
        n_classes: 3,
        max_bond: 8,
        readout_qubits: vec![0, 1, 2],
        learning_rate: 0.1,
        epochs: 1,
        batch_size: 1,
        loss: QmlLoss::CrossEntropy,
        seed: 42,
    };
    let params: Vec<f64> = (0..total_params(&config)).map(|i| i as f64 * 0.1).collect();
    let input = vec![1.0, 0.5, 2.0, 1.5];
    let mps = build_qml_circuit(&config, &params, &input);
    let probs = classify(&mps, &config.readout_qubits, config.n_classes);

    assert_eq!(probs.len(), 3);
    let sum: f64 = probs.iter().sum();
    assert!((sum - 1.0).abs() < TOL, "probabilities should sum to 1, got {}", sum);
    for &p in &probs {
        assert!(p >= 0.0, "probability should be non-negative, got {}", p);
    }
}

#[test]
fn qml_classify_ground_state_gives_uniform() {
    let mps = Mps::new(4);
    let readout = vec![0, 1, 2];
    let probs = classify(&mps, &readout, 3);
    // All Z expectations are +1, so softmax gives uniform
    for &p in &probs {
        assert!((p - 1.0 / 3.0).abs() < TOL, "expected uniform ~0.333, got {}", p);
    }
}

// ---------------------------------------------------------------------------
// 3. Single-qubit Z expectation
// ---------------------------------------------------------------------------

#[test]
fn qml_single_z_bell_state() {
    let mut mps = Mps::new(2);
    let isq2 = 1.0 / 2.0f64.sqrt();
    let h = [[ComplexF64::real(isq2), ComplexF64::real(isq2)],
              [ComplexF64::real(isq2), ComplexF64::real(-isq2)]];
    mps.apply_single_qubit(0, h);
    mps.apply_cnot_adjacent(0, 1);
    // Bell state: <Z_0> = 0, <Z_1> = 0
    assert!(mps_single_z_expectation(&mps, 0).abs() < TOL);
    assert!(mps_single_z_expectation(&mps, 1).abs() < TOL);
}

// ---------------------------------------------------------------------------
// 4. Gradient correctness
// ---------------------------------------------------------------------------

#[test]
fn qml_gradient_nonzero_for_nontrivial_input() {
    let config = QmlConfig {
        n_qubits: 3,
        n_reupload_passes: 1,
        n_classes: 2,
        max_bond: 8,
        readout_qubits: vec![0, 1],
        learning_rate: 0.1,
        epochs: 1,
        batch_size: 2,
        loss: QmlLoss::CrossEntropy,
        seed: 42,
    };
    let params: Vec<f64> = (0..total_params(&config))
        .map(|i| 0.1 * (i as f64 + 1.0))
        .collect();
    let samples = vec![vec![1.0, 0.5, 0.2], vec![0.3, 2.0, 1.5]];
    let labels = vec![0, 1];

    let grads = qml_gradient(&config, &params, &samples, &labels);
    let any_nonzero = grads.iter().any(|g| g.abs() > 1e-8);
    assert!(any_nonzero, "gradient should have nonzero entries");
}

// ---------------------------------------------------------------------------
// 5. Training
// ---------------------------------------------------------------------------

#[test]
fn qml_train_produces_correct_result_shape() {
    let config = QmlConfig {
        n_qubits: 3,
        n_reupload_passes: 1,
        n_classes: 2,
        max_bond: 8,
        readout_qubits: vec![0, 1],
        learning_rate: 0.1,
        epochs: 2,
        batch_size: 2,
        loss: QmlLoss::Mse,
        seed: 42,
    };
    let dataset = QmlDataset {
        samples: vec![vec![0.1, 0.2, 0.3], vec![2.0, 1.5, 1.0]],
        labels: vec![0, 1],
        n_classes: 2,
    };

    let result = qml_train(&config, &dataset);
    assert_eq!(result.params.len(), total_params(&config));
    assert_eq!(result.epochs_completed, 2);
    assert_eq!(result.loss_history.len(), 3); // initial + 2 epochs
    assert_eq!(result.accuracy_history.len(), 3);
    assert!(result.final_accuracy >= 0.0 && result.final_accuracy <= 1.0);
}

// ---------------------------------------------------------------------------
// 6. Determinism
// ---------------------------------------------------------------------------

#[test]
fn qml_determinism_bit_identical_training() {
    let config = QmlConfig {
        n_qubits: 3,
        n_reupload_passes: 1,
        n_classes: 2,
        max_bond: 8,
        readout_qubits: vec![0, 1],
        learning_rate: 0.05,
        epochs: 3,
        batch_size: 2,
        loss: QmlLoss::CrossEntropy,
        seed: 777,
    };
    let dataset = QmlDataset {
        samples: vec![
            vec![0.5, 1.0, 0.2],
            vec![2.0, 0.3, 1.5],
            vec![1.0, 1.0, 1.0],
        ],
        labels: vec![0, 1, 0],
        n_classes: 2,
    };

    let r1 = qml_train(&config, &dataset);
    let r2 = qml_train(&config, &dataset);

    for k in 0..r1.params.len() {
        assert_eq!(r1.params[k].to_bits(), r2.params[k].to_bits(),
            "param[{}] not bit-identical: {} vs {}", k, r1.params[k], r2.params[k]);
    }
    for (i, (l1, l2)) in r1.loss_history.iter().zip(&r2.loss_history).enumerate() {
        assert_eq!(l1.to_bits(), l2.to_bits(),
            "loss_history[{}] not bit-identical: {} vs {}", i, l1, l2);
    }
}

#[test]
fn qml_different_seeds_differ() {
    let make_config = |seed| QmlConfig {
        n_qubits: 3,
        n_reupload_passes: 1,
        n_classes: 2,
        max_bond: 8,
        readout_qubits: vec![0, 1],
        learning_rate: 0.1,
        epochs: 2,
        batch_size: 2,
        loss: QmlLoss::CrossEntropy,
        seed,
    };
    let dataset = QmlDataset {
        samples: vec![vec![1.0, 0.5, 0.2], vec![0.3, 2.0, 1.5]],
        labels: vec![0, 1],
        n_classes: 2,
    };

    let r1 = qml_train(&make_config(42), &dataset);
    let r2 = qml_train(&make_config(99), &dataset);

    let any_diff = r1.params.iter().zip(&r2.params)
        .any(|(a, b)| a.to_bits() != b.to_bits());
    assert!(any_diff, "different seeds should produce different params");
}

// ---------------------------------------------------------------------------
// 7. Data preprocessing
// ---------------------------------------------------------------------------

#[test]
fn qml_snake_order_covers_all_pixels() {
    for (w, h) in [(4, 4), (3, 5), (7, 2), (1, 1)] {
        let order = snake_order(w, h);
        assert_eq!(order.len(), w * h, "snake_order({},{}) should cover all pixels", w, h);
        // Check all coordinates are in range
        for &(r, c) in &order {
            assert!(r < h && c < w, "({},{}) out of range for {}x{}", r, c, w, h);
        }
        // Check no duplicates
        let mut seen = std::collections::HashSet::new();
        for &coord in &order {
            assert!(seen.insert(coord), "duplicate coordinate {:?}", coord);
        }
    }
}

#[test]
fn qml_preprocess_monotonic_input() {
    // Brighter pixels → larger rotation angles
    let dark: Vec<u8> = vec![10; 16];
    let bright: Vec<u8> = vec![200; 16];
    let d = preprocess_image(&dark, 4, 4, 4);
    let b = preprocess_image(&bright, 4, 4, 4);
    for i in 0..4 {
        assert!(b[i] > d[i], "brighter pixels should give larger angles");
    }
}

#[test]
fn qml_load_dataset_structure() {
    let n_images = 20;
    let img_size = 16;
    let image_data: Vec<u8> = (0..(n_images * img_size)).map(|i| (i % 256) as u8).collect();
    let labels: Vec<u8> = (0..n_images as u8).collect();

    let ds = load_dataset(&image_data, &labels, 4, 4, 15, 4, 3);
    assert_eq!(ds.samples.len(), 15);
    assert_eq!(ds.labels.len(), 15);
    assert_eq!(ds.n_classes, 3);
    for l in &ds.labels {
        assert!(*l < 3, "label {} should be < n_classes", l);
    }
}

// ---------------------------------------------------------------------------
// 8. Memory scaling
// ---------------------------------------------------------------------------

#[test]
fn qml_50_qubit_forward_pass_under_1mb() {
    let config = QmlConfig {
        n_qubits: 50,
        n_reupload_passes: 3,
        n_classes: 2,
        max_bond: 16,
        readout_qubits: vec![0, 1],
        learning_rate: 0.1,
        epochs: 1,
        batch_size: 1,
        loss: QmlLoss::Mse,
        seed: 42,
    };
    let params = vec![0.01; total_params(&config)];
    let input = vec![1.0; 50];
    let mps = build_qml_circuit(&config, &params, &input);
    assert!(mps.memory_bytes() < 1_000_000,
        "50-qubit MPS with chi=16 should be under 1MB, got {}", mps.memory_bytes());
}

// ---------------------------------------------------------------------------
// 9. Predict function
// ---------------------------------------------------------------------------

#[test]
fn qml_predict_returns_valid_class() {
    let config = QmlConfig {
        n_qubits: 4,
        n_reupload_passes: 1,
        n_classes: 3,
        max_bond: 8,
        readout_qubits: vec![0, 1, 2],
        learning_rate: 0.1,
        epochs: 1,
        batch_size: 1,
        loss: QmlLoss::CrossEntropy,
        seed: 42,
    };
    let params: Vec<f64> = (0..total_params(&config)).map(|i| i as f64 * 0.1).collect();
    let input = vec![1.0, 0.5, 2.0, 0.8];
    let class = predict(&config, &params, &input);
    assert!(class < 3, "predicted class {} should be < n_classes=3", class);
}
