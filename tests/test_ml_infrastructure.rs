// ═══════════════════════════════════════════════════════════════════════
// ML Infrastructure Tests
//
// Tests for ML builtins: embedding, avgpool2d (method), batch_indices.
// embedding: standalone function embedding(weights, indices)
// avgpool2d: method on Tensor -- input.avgpool2d(kh, kw, sh, sw)
// batch_indices: standalone function batch_indices(dataset_size, batch_size, seed)
// ═══════════════════════════════════════════════════════════════════════

/// Run CJC source through eval with given seed, return output lines.
fn run_eval(src: &str, seed: u64) -> Result<Vec<String>, String> {
    let (program, diags) = cjc_parser::parse_source(src);
    if diags.has_errors() {
        return Err(format!("Parse errors: {:?}", diags.render_all(src, "<ml-infra>")));
    }
    let mut interp = cjc_eval::Interpreter::new(seed);
    match interp.exec(&program) {
        Ok(_) => Ok(interp.output),
        Err(e) => Err(format!("{e:?}")),
    }
}

/// Run CJC source through MIR-exec with given seed, return output lines.
fn run_mir(src: &str, seed: u64) -> Result<Vec<String>, String> {
    let (program, diags) = cjc_parser::parse_source(src);
    if diags.has_errors() {
        return Err(format!("Parse errors: {:?}", diags.render_all(src, "<ml-infra>")));
    }
    match cjc_mir_exec::run_program_with_executor(&program, seed) {
        Ok((_, executor)) => Ok(executor.output),
        Err(e) => Err(format!("{e}")),
    }
}

/// Assert parity between eval and MIR-exec.
fn assert_parity(src: &str) {
    let eval_out = run_eval(src, 42).expect("Eval failed");
    let mir_out = run_mir(src, 42).expect("MIR failed");
    assert_eq!(
        eval_out, mir_out,
        "Parity violation!\nEval: {eval_out:?}\nMIR:  {mir_out:?}"
    );
}

/// Check that both executors produce errors (for error cases).
fn assert_both_error(src: &str) {
    let eval_res = run_eval(src, 42);
    let mir_res = run_mir(src, 42);
    assert!(
        eval_res.is_err() && mir_res.is_err(),
        "Expected both executors to error.\nEval: {eval_res:?}\nMIR: {mir_res:?}"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// Embedding Tests
//
// embedding(weights, indices) looks up rows from a weight matrix.
// weights: Tensor of shape [vocab_size, embed_dim]
// indices: array of integer indices
// result:  Tensor of shape [len(indices), embed_dim]
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn embedding_basic_lookup() {
    // 3x2 weight matrix, look up rows 0 and 2
    assert_parity(r#"
        fn main() {
            let weights = Tensor.from_vec([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2]);
            let indices: Any = [0, 2];
            let result = embedding(weights, indices);
            print(result);
        }
    "#);
}

#[test]
fn embedding_single_index() {
    assert_parity(r#"
        fn main() {
            let weights = Tensor.from_vec([10.0, 20.0, 30.0, 40.0], [2, 2]);
            let indices: Any = [1];
            let result = embedding(weights, indices);
            print(result);
        }
    "#);
}

#[test]
fn embedding_out_of_bounds() {
    // Index 5 is out of bounds for a 3-row weight matrix
    assert_both_error(r#"
        fn main() {
            let weights = Tensor.from_vec([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2]);
            let indices: Any = [5];
            let result = embedding(weights, indices);
            print(result);
        }
    "#);
}

// ═══════════════════════════════════════════════════════════════════════
// AvgPool2D Tests
//
// avgpool2d is a method on Tensor: input.avgpool2d(kh, kw, sh, sw)
// input: Tensor of shape [C, H, W]
// Returns pooled tensor with reduced spatial dims.
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn avgpool2d_basic_3x3_kernel() {
    // 1x4x4 input, 3x3 kernel, stride 1 => 1x2x2 output
    assert_parity(r#"
        fn main() {
            let input = Tensor.from_vec([
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0
            ], [1, 4, 4]);
            let result = input.avgpool2d(3, 3, 1, 1);
            print(result);
        }
    "#);
}

#[test]
fn avgpool2d_stride2() {
    // 1x4x4 input, 2x2 kernel, stride 2 => 1x2x2 output
    assert_parity(r#"
        fn main() {
            let input = Tensor.from_vec([
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0
            ], [1, 4, 4]);
            let result = input.avgpool2d(2, 2, 2, 2);
            print(result);
        }
    "#);
}

#[test]
fn avgpool2d_shape_validation() {
    // 2x2 kernel on 1x4x4 with stride=1 => 1x3x3
    assert_parity(r#"
        fn main() {
            let input = Tensor.from_vec([
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0
            ], [1, 4, 4]);
            let result = input.avgpool2d(2, 2, 1, 1);
            print(result);
        }
    "#);
}

// ═══════════════════════════════════════════════════════════════════════
// Batch Indices Tests
//
// batch_indices(dataset_size, batch_size, seed) -> array of [start, end] pairs
// Returns index ranges for mini-batch training.
// Must be deterministic: same seed = same result.
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn batch_indices_deterministic() {
    // Same seed must produce same result
    let src = r#"
        fn main() {
            let batches = batch_indices(10, 3, 123);
            print(batches);
        }
    "#;
    let eval1 = run_eval(src, 42).expect("Eval failed");
    let eval2 = run_eval(src, 42).expect("Eval failed");
    assert_eq!(eval1, eval2, "batch_indices not deterministic in eval");

    let mir1 = run_mir(src, 42).expect("MIR failed");
    let mir2 = run_mir(src, 42).expect("MIR failed");
    assert_eq!(mir1, mir2, "batch_indices not deterministic in MIR");
}

#[test]
fn batch_indices_covers_all_samples() {
    // All indices 0..dataset_size should appear in the batch ranges
    assert_parity(r#"
        fn main() {
            let batches = batch_indices(6, 2, 42);
            print(batches);
        }
    "#);
}

#[test]
fn batch_indices_correct_batch_sizes() {
    // 10 samples with batch_size=3 => 4 batch ranges
    assert_parity(r#"
        fn main() {
            let batches = batch_indices(10, 3, 99);
            print(len(batches));
        }
    "#);
}

#[test]
fn batch_indices_parity_across_seeds() {
    let src = r#"
        fn main() {
            let batches = batch_indices(8, 4, 77);
            print(batches);
        }
    "#;
    for seed in 1..=10 {
        let eval_out = run_eval(src, seed).expect("Eval failed");
        let mir_out = run_mir(src, seed).expect("MIR failed");
        assert_eq!(
            eval_out, mir_out,
            "batch_indices parity violation at seed={seed}"
        );
    }
}
