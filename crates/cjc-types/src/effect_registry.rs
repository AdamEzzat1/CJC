//! Effect Registry — Single source of truth for builtin function effects.
//!
//! Maps every known builtin function name to its `EffectSet` flags. This
//! replaces the hardcoded allowlists in `nogc_verify.rs` and `escape.rs`
//! with a single queryable registry.
//!
//! Classification rules:
//! - PURE: no side effects, no allocation, deterministic
//! - IO: prints to stdout, reads/writes files
//! - ALLOC: creates new heap objects (String, Vec, Tensor, etc.)
//! - GC: triggers garbage collection (gc_alloc, gc_collect)
//! - NONDET: nondeterministic (random, clock, hash-order)
//! - MUTATES: modifies its arguments in place
//! - ARENA_OK: result can safely live on a FrameArena (non-escaping)
//! - CAPTURES: may store/capture argument references beyond the call

use std::collections::HashMap;

use super::EffectSet;

/// Build the complete builtin effect registry.
///
/// Returns a map from builtin name (as used in dispatch) to its effect flags.
pub fn builtin_effects() -> HashMap<&'static str, EffectSet> {
    let mut m = HashMap::new();

    // Helper constants for common patterns.
    let pure = EffectSet::PURE;
    let alloc = EffectSet::new(EffectSet::ALLOC);
    let alloc_arena = EffectSet::new(EffectSet::ALLOC | EffectSet::ARENA_OK);
    let io = EffectSet::new(EffectSet::IO);
    let gc = EffectSet::new(EffectSet::GC | EffectSet::ALLOC);
    let _nondet = EffectSet::new(EffectSet::NONDET);
    let nondet_alloc = EffectSet::new(EffectSet::NONDET | EffectSet::ALLOC);
    let mutates = EffectSet::new(EffectSet::MUTATES);
    let mutates_alloc = EffectSet::new(EffectSet::MUTATES | EffectSet::ALLOC);
    let io_alloc = EffectSet::new(EffectSet::IO | EffectSet::ALLOC);

    // -----------------------------------------------------------------
    // GC builtins (trigger GC)
    // -----------------------------------------------------------------
    m.insert("gc_alloc", gc);
    m.insert("gc_collect", EffectSet::new(EffectSet::GC));

    // -----------------------------------------------------------------
    // IO builtins
    // -----------------------------------------------------------------
    m.insert("print", io);
    m.insert("gc_live_count", pure); // query only, no GC

    // -----------------------------------------------------------------
    // Time / nondeterministic
    // -----------------------------------------------------------------
    m.insert("clock", EffectSet::new(EffectSet::IO | EffectSet::NONDET));
    m.insert("Tensor.randn", nondet_alloc);

    // -----------------------------------------------------------------
    // Pure math (scalar) — no allocation, fully deterministic
    // -----------------------------------------------------------------
    for name in &[
        "sqrt", "floor", "int", "float", "isnan", "isinf", "abs",
        "log", "exp",
        // Mathematics Hardening Phase: scalar math
        "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
        "sinh", "cosh", "tanh_scalar",
        "pow", "log2", "log10", "log1p", "expm1",
        "ceil", "round",
        "min", "max", "sign",
        "hypot",
        // Mathematics Hardening Phase: constants
        "PI", "E", "TAU", "INF", "NAN_VAL",
        // Mathematics Hardening Phase: vector operations (pure for dot/norm)
        "dot", "norm",
    ] {
        m.insert(name, pure);
    }
    // Mathematics Hardening Phase: vector ops that allocate
    for name in &["outer", "cross"] {
        m.insert(name, alloc);
    }
    // Mathematics Hardening Phase: tensor constructors (allocate)
    for name in &[
        "Tensor.linspace", "Tensor.arange", "Tensor.eye",
        "Tensor.full", "Tensor.diag",
    ] {
        m.insert(name, alloc);
    }
    // Tensor.uniform is nondeterministic (uses RNG)
    m.insert("Tensor.uniform", nondet_alloc);
    // categorical_sample uses the interpreter RNG — nondeterministic + alloc
    m.insert("categorical_sample", nondet_alloc);
    // sample_indices uses RNG — nondeterministic + alloc
    m.insert("sample_indices", nondet_alloc);

    // -----------------------------------------------------------------
    // Type conversion builtins — no allocation, deterministic
    // -----------------------------------------------------------------
    for name in &[
        "f16_to_f64", "f64_to_f16", "f16_to_f32", "f32_to_f16",
        "bf16_to_f32", "f32_to_bf16",
    ] {
        m.insert(name, pure);
    }

    // -----------------------------------------------------------------
    // Assertions — IO (may panic/print), but deterministic
    // -----------------------------------------------------------------
    m.insert("assert", io);
    m.insert("assert_eq", io);

    // -----------------------------------------------------------------
    // Collection queries — no allocation, deterministic
    // -----------------------------------------------------------------
    m.insert("len", pure);

    // -----------------------------------------------------------------
    // Collection mutations — allocates new collection
    // -----------------------------------------------------------------
    m.insert("push", mutates_alloc);
    m.insert("sort", alloc);
    m.insert("to_string", alloc_arena);

    // -----------------------------------------------------------------
    // Tensor construction — allocates tensor buffer
    // -----------------------------------------------------------------
    for name in &[
        "Tensor.zeros", "Tensor.ones", "Tensor.from_vec",
        "Buffer.alloc", "Tensor.from_bytes",
    ] {
        m.insert(name, alloc);
    }

    // -----------------------------------------------------------------
    // Tensor operations — allocate new tensor, deterministic
    // -----------------------------------------------------------------
    for name in &[
        "matmul", "attention",
        "Tensor.softmax", "Tensor.layer_norm",
        "Tensor.relu", "Tensor.gelu",
        "Tensor.bmm", "Tensor.linear", "Tensor.transpose_last_two",
        "Tensor.conv1d", "Tensor.conv2d", "Tensor.maxpool2d",
        "Tensor.binned_sum",
        "Tensor.matmul", "Tensor.add", "Tensor.sub",
        "Tensor.reshape", "Tensor.transpose", "Tensor.neg",
        "Tensor.scalar_mul", "Tensor.mul",
        "Tensor.sum_axis",
    ] {
        m.insert(name, alloc);
    }

    // -----------------------------------------------------------------
    // Tensor queries — no allocation
    // -----------------------------------------------------------------
    for name in &[
        "Tensor.shape", "Tensor.len", "Tensor.get",
    ] {
        m.insert(name, pure);
    }
    m.insert("Tensor.to_vec", alloc); // creates array
    m.insert("Tensor.set", alloc); // COW creates new tensor

    // -----------------------------------------------------------------
    // Tensor views — stack/bitmask only, no heap alloc in the GC sense
    // -----------------------------------------------------------------
    for name in &[
        "Tensor.slice", "Tensor.broadcast_to",
        "Tensor.split_heads", "Tensor.merge_heads", "Tensor.view_reshape",
        "ByteSlice.as_tensor",
    ] {
        m.insert(name, alloc); // allocates tensor buffer but no GC
    }

    // -----------------------------------------------------------------
    // Linalg builtins — allocate result tensors
    // -----------------------------------------------------------------
    for name in &[
        "linalg.lu", "linalg.qr", "linalg.cholesky", "linalg.inv",
    ] {
        m.insert(name, alloc);
    }

    // -----------------------------------------------------------------
    // Sparse tensor ops — allocate result
    // -----------------------------------------------------------------
    for name in &[
        "SparseCsr.matvec", "SparseCsr.to_dense", "SparseCoo.to_csr",
    ] {
        m.insert(name, alloc);
    }

    // v0.1: Broadcasting builtins — allocate new tensors
    m.insert("broadcast", alloc);
    m.insert("broadcast2", alloc);

    // -----------------------------------------------------------------
    // Complex number ops — stack-only, no allocation
    // -----------------------------------------------------------------
    for name in &[
        "Complex.re", "Complex.im", "Complex.abs", "Complex.conj",
        "Complex.norm_sq", "Complex.add", "Complex.mul",
        "Complex.sub", "Complex.div", "Complex.neg", "Complex.scale",
        "Complex.is_nan", "Complex.is_finite",
    ] {
        m.insert(name, pure);
    }
    m.insert("Complex", alloc); // constructor

    // -----------------------------------------------------------------
    // F16 methods — stack-only, no allocation
    // -----------------------------------------------------------------
    for name in &["F16.to_f64", "F16.to_f32"] {
        m.insert(name, pure);
    }

    // -----------------------------------------------------------------
    // Scratchpad / KV-Cache — pre-allocated, no GC
    // -----------------------------------------------------------------
    for name in &[
        "Scratchpad.new", "Scratchpad.append", "Scratchpad.append_tensor",
        "Scratchpad.as_tensor", "Scratchpad.len", "Scratchpad.capacity",
        "Scratchpad.dim", "Scratchpad.clear", "Scratchpad.is_empty",
    ] {
        m.insert(name, alloc); // initial alloc, but no GC
    }

    for name in &[
        "PagedKvCache.new", "PagedKvCache.append", "PagedKvCache.append_tensor",
        "PagedKvCache.as_tensor", "PagedKvCache.clear", "PagedKvCache.len",
        "PagedKvCache.is_empty", "PagedKvCache.max_tokens", "PagedKvCache.dim",
        "PagedKvCache.num_blocks", "PagedKvCache.blocks_in_use",
        "PagedKvCache.get_token",
    ] {
        m.insert(name, alloc);
    }

    for name in &[
        "AlignedByteSlice.from_bytes", "AlignedByteSlice.as_tensor",
        "AlignedByteSlice.was_realigned", "AlignedByteSlice.len",
        "AlignedByteSlice.is_empty",
    ] {
        m.insert(name, alloc);
    }

    // -----------------------------------------------------------------
    // Deterministic Map — no GC, but allocates
    // -----------------------------------------------------------------
    m.insert("Map.new", alloc);
    m.insert("Map.insert", mutates_alloc);
    m.insert("Map.get", pure);
    m.insert("Map.remove", mutates);
    m.insert("Map.len", pure);
    m.insert("Map.contains_key", pure);
    m.insert("Map.keys", alloc);
    m.insert("Map.values", alloc);

    // -----------------------------------------------------------------
    // Tidy view ops — lightweight bitmask/projection, no GC heap
    // Safe inside @nogc (view-only, O(N/64) bitmask or O(K) index)
    // -----------------------------------------------------------------
    for name in &[
        "tidy_filter", "tidy_select", "tidy_mask_and",
        "tidy_nrows", "tidy_ncols", "tidy_column_names",
        "tidy_group_by", "tidy_ungroup", "tidy_ngroups",
        "tidy_slice", "tidy_slice_head", "tidy_slice_tail", "tidy_slice_sample",
        "tidy_distinct",
        "tidy_semi_join", "tidy_anti_join",
        "tidy_relocate", "tidy_drop_cols",
        "tidy_group_by_fast",
        "fct_collapse",
    ] {
        m.insert(name, alloc); // small alloc (bitmask), but no GC
    }

    // -----------------------------------------------------------------
    // Tidy materializing ops — column buffer allocation (NOT safe in @nogc)
    // These are intentionally ABSENT from the registry so that the nogc
    // verifier treats them as unknown → conservative → rejected.
    // -----------------------------------------------------------------
    // NOT registered: tidy_arrange, tidy_summarise, tidy_mutate,
    //   tidy_inner_join, tidy_left_join, tidy_right_join, tidy_full_join,
    //   tidy_inner_join_typed, tidy_left_join_typed,
    //   tidy_pivot_longer, tidy_pivot_wider,
    //   tidy_bind_rows, tidy_bind_cols,
    //   tidy_mutate_across, tidy_summarise_across,
    //   tidy_rename

    // -----------------------------------------------------------------
    // Category / forcats materializing ops — NOT safe in @nogc
    // -----------------------------------------------------------------
    // NOT registered: fct_encode, fct_lump, fct_reorder

    // -----------------------------------------------------------------
    // Tidy builders — small allocation
    // -----------------------------------------------------------------
    for name in &[
        "col", "desc", "asc", "dexpr_binop",
        "tidy_count", "tidy_sum", "tidy_mean", "tidy_min", "tidy_max",
        "tidy_first", "tidy_last",
    ] {
        m.insert(name, alloc_arena);
    }

    // -----------------------------------------------------------------
    // String operations (allocate new strings)
    // -----------------------------------------------------------------
    for name in &[
        "str_extract", "str_extract_all", "str_replace", "str_replace_all",
        "str_split", "str_trim", "str_to_upper", "str_to_lower", "str_sub",
    ] {
        m.insert(name, alloc_arena);
    }

    // String queries — no allocation
    for name in &[
        "str_detect", "str_count", "str_starts", "str_ends", "str_len",
    ] {
        m.insert(name, pure);
    }

    // -----------------------------------------------------------------
    // Statistics builtins
    // -----------------------------------------------------------------
    m.insert("mean", pure); // Kahan sum / n, no allocation
    m.insert("median", alloc); // sorts internally
    m.insert("sd", alloc);
    m.insert("variance", alloc);
    m.insert("n_distinct", EffectSet::new(EffectSet::ALLOC | EffectSet::NONDET)); // hash-order
    m.insert("se", alloc);
    m.insert("quantile", alloc);
    // Bastion primitives
    m.insert("nth_element", alloc);
    m.insert("median_fast", alloc);
    m.insert("quantile_fast", alloc);
    m.insert("filter_mask", alloc);
    m.insert("erf", pure);
    m.insert("erfc", pure);
    m.insert("iqr", alloc);
    m.insert("skewness", alloc);
    m.insert("kurtosis", alloc);
    m.insert("z_score", alloc);
    m.insert("standardize", alloc);
    m.insert("cor", alloc);
    m.insert("cov", alloc);
    // Cumulative / ranking
    m.insert("cumsum", alloc);
    m.insert("cumprod", alloc);
    m.insert("cummax", alloc);
    m.insert("cummin", alloc);
    m.insert("lag", alloc);
    m.insert("lead", alloc);
    m.insert("rank", alloc);
    m.insert("dense_rank", alloc);
    m.insert("histogram", alloc);
    // Distributions
    m.insert("normal_cdf", pure);
    m.insert("normal_pdf", pure);
    m.insert("normal_ppf", pure);
    m.insert("t_cdf", pure);
    m.insert("chi2_cdf", pure);
    m.insert("f_cdf", pure);
    // Stationarity tests
    m.insert("adf_test", alloc);
    m.insert("kpss_test", alloc);
    m.insert("pp_test", alloc);
    // Hypothesis tests
    m.insert("t_test", alloc);
    m.insert("t_test_two_sample", alloc);
    m.insert("chi_squared_test", alloc);
    // Extended linalg
    m.insert("det", alloc);
    m.insert("solve", alloc);
    m.insert("lstsq", alloc);
    m.insert("trace", pure);
    m.insert("norm_frobenius", pure);
    m.insert("eigh", alloc);
    m.insert("matrix_rank", alloc);
    m.insert("kron", alloc);
    // ML
    m.insert("mse_loss", alloc);
    m.insert("cross_entropy_loss", alloc);
    m.insert("huber_loss", alloc);
    m.insert("binary_cross_entropy", alloc);
    m.insert("hinge_loss", alloc);
    m.insert("confusion_matrix", alloc);
    m.insert("auc_roc", alloc);
    // Additional stats
    m.insert("sample_variance", pure);
    m.insert("sample_sd", pure);
    m.insert("sample_cov", pure);
    m.insert("row_number", alloc);
    // Distribution PPFs
    m.insert("t_ppf", pure);
    m.insert("chi2_ppf", pure);
    m.insert("f_ppf", pure);
    // Discrete distributions
    m.insert("binomial_pmf", pure);
    m.insert("binomial_cdf", pure);
    m.insert("poisson_pmf", pure);
    m.insert("poisson_cdf", pure);
    // Additional hypothesis tests
    m.insert("t_test_paired", alloc);
    m.insert("anova_oneway", alloc);
    m.insert("f_test", alloc);
    m.insert("lm", alloc);
    // FFT
    m.insert("rfft", alloc);
    m.insert("psd", alloc);
    // Tensor activations & utilities
    m.insert("sigmoid", alloc);
    m.insert("tanh_activation", alloc);
    m.insert("leaky_relu", alloc);
    m.insert("silu", alloc);
    m.insert("mish", alloc);
    m.insert("argmax", pure);
    m.insert("argmin", pure);
    m.insert("clamp", alloc);
    m.insert("one_hot", alloc);

    // -----------------------------------------------------------------
    // CSV builtins
    // -----------------------------------------------------------------
    for name in &[
        "Csv.parse", "Csv.parse_tsv", "Csv.stream_sum", "Csv.stream_minmax",
    ] {
        m.insert(name, io_alloc);
    }

    // -----------------------------------------------------------------
    // AD (automatic differentiation) — allocates gradient tensors
    // -----------------------------------------------------------------
    m.insert("grad", alloc);
    m.insert("Dual.new", alloc);

    // -----------------------------------------------------------------
    // JSON builtins — allocate parsed structures
    // -----------------------------------------------------------------
    m.insert("json_parse", alloc);     // parses string → nested Value
    m.insert("json_stringify", alloc); // creates string from Value

    // -----------------------------------------------------------------
    // DateTime builtins — mostly pure arithmetic
    // -----------------------------------------------------------------
    m.insert("datetime_now", EffectSet::new(EffectSet::NONDET | EffectSet::IO)); // nondeterministic
    m.insert("datetime_from_epoch", pure);
    m.insert("datetime_from_parts", pure);
    m.insert("datetime_year", pure);
    m.insert("datetime_month", pure);
    m.insert("datetime_day", pure);
    m.insert("datetime_hour", pure);
    m.insert("datetime_minute", pure);
    m.insert("datetime_second", pure);
    m.insert("datetime_diff", pure);
    m.insert("datetime_add_millis", pure);
    m.insert("datetime_format", alloc); // allocates string

    // -----------------------------------------------------------------
    // File I/O builtins — IO effects
    // -----------------------------------------------------------------
    m.insert("file_read", io_alloc);   // reads file, allocates string
    m.insert("file_write", io);        // writes file
    m.insert("file_exists", io);       // filesystem query
    m.insert("file_lines", io_alloc);  // reads file, allocates array

    // ── CJC Snap builtins ──
    m.insert("snap", alloc);           // allocates SnapBlob struct + bytes
    m.insert("restore", alloc);        // allocates decoded Value
    m.insert("snap_hash", alloc);      // allocates String
    m.insert("snap_save", io);         // writes to filesystem
    m.insert("snap_load", io_alloc);   // reads file + allocates Value
    m.insert("snap_to_json", alloc);   // allocates JSON String
    m.insert("memo_call", alloc);      // allocates cached result

    // ── Phase B1: Weighted & robust statistics ──
    m.insert("weighted_mean", alloc);
    m.insert("weighted_var", alloc);
    m.insert("trimmed_mean", alloc);
    m.insert("winsorize", alloc);
    m.insert("mad", alloc);
    m.insert("mode", alloc);
    m.insert("percentile_rank", alloc);

    // ── Phase B2: Rank correlations ──
    m.insert("spearman_cor", alloc);
    m.insert("kendall_cor", alloc);
    m.insert("partial_cor", alloc);
    m.insert("cor_ci", alloc);

    // ── Phase B3: Linear algebra extensions ──
    m.insert("cond", alloc);
    m.insert("norm_1", pure);
    m.insert("norm_inf", pure);
    m.insert("schur", alloc);
    m.insert("matrix_exp", alloc);

    // ── Phase B4: ML training extensions ──
    m.insert("cat", alloc);
    m.insert("stack", alloc);
    m.insert("topk", alloc);
    m.insert("batch_norm", alloc);
    m.insert("dropout_mask", alloc);
    m.insert("lr_step_decay", pure);
    m.insert("lr_cosine", pure);
    m.insert("lr_linear_warmup", pure);
    m.insert("l1_penalty", alloc);
    m.insert("l2_penalty", alloc);

    // ── Phase B6: Advanced FFT & Distributions ──
    m.insert("hann", alloc);
    m.insert("hamming", alloc);
    m.insert("blackman", alloc);
    m.insert("fft_arbitrary", alloc);
    m.insert("fft_2d", alloc);
    m.insert("ifft_2d", alloc);
    m.insert("beta_pdf", pure);
    m.insert("beta_cdf", pure);
    m.insert("gamma_pdf", pure);
    m.insert("gamma_cdf", pure);
    m.insert("exp_pdf", pure);
    m.insert("exp_cdf", pure);
    m.insert("weibull_pdf", pure);
    m.insert("weibull_cdf", pure);

    // ── Phase B7: Non-parametric tests & multiple comparisons ──
    m.insert("tukey_hsd", alloc);
    m.insert("mann_whitney", alloc);
    m.insert("kruskal_wallis", alloc);
    m.insert("wilcoxon_signed_rank", alloc);
    m.insert("bonferroni", alloc);
    m.insert("fdr_bh", alloc);
    m.insert("logistic_regression", alloc);

    // ── Phase B5: Analyst QoL extensions ──
    m.insert("case_when", alloc);
    m.insert("ntile", alloc);
    m.insert("percent_rank", alloc);
    m.insert("cume_dist", alloc);
    m.insert("wls", alloc);

    // ── Window functions ──
    m.insert("window_sum", alloc);   // allocates result array
    m.insert("window_mean", alloc);  // allocates result array
    m.insert("window_min", alloc);   // allocates result array
    m.insert("window_max", alloc);   // allocates result array

    // ── Phase C1: GradGraph Language API ──
    m.insert("GradGraph.new", alloc);
    m.insert("GradGraph.parameter", mutates_alloc);
    m.insert("GradGraph.input", mutates_alloc);
    m.insert("GradGraph.add", mutates_alloc);
    m.insert("GradGraph.sub", mutates_alloc);
    m.insert("GradGraph.mul", mutates_alloc);
    m.insert("GradGraph.div", mutates_alloc);
    m.insert("GradGraph.neg", mutates_alloc);
    m.insert("GradGraph.matmul", mutates_alloc);
    m.insert("GradGraph.sum", mutates_alloc);
    m.insert("GradGraph.mean", mutates_alloc);
    m.insert("GradGraph.sigmoid", mutates_alloc);
    m.insert("GradGraph.relu", mutates_alloc);
    m.insert("GradGraph.tanh", mutates_alloc);
    m.insert("GradGraph.sin", mutates_alloc);
    m.insert("GradGraph.cos", mutates_alloc);
    m.insert("GradGraph.sqrt", mutates_alloc);
    m.insert("GradGraph.pow", mutates_alloc);
    m.insert("GradGraph.exp", mutates_alloc);
    m.insert("GradGraph.ln", mutates_alloc);
    m.insert("GradGraph.scalar_mul", mutates_alloc);
    m.insert("GradGraph.backward", mutates);
    m.insert("GradGraph.value", pure);
    m.insert("GradGraph.tensor", alloc);
    m.insert("GradGraph.grad", alloc);
    m.insert("GradGraph.set_tensor", mutates);
    m.insert("GradGraph.zero_grad", mutates);

    // ── Phase C2: Optimizer builtins ──
    m.insert("Adam.new", alloc);
    m.insert("Sgd.new", alloc);
    m.insert("OptimizerState.step", mutates_alloc);

    // ── Phase C3: Bitwise operations ──
    for name in &["bit_and", "bit_or", "bit_xor", "bit_not", "bit_shl", "bit_shr", "popcount"] {
        m.insert(*name, pure);
    }

    // ── Phase C4: Sorting & Tensor Indexing ──
    m.insert("argsort", alloc);
    m.insert("gather", alloc);
    m.insert("scatter", alloc);
    m.insert("index_select", alloc);

    // ── Phase C5: Map & Set ──
    // ── Phase C6: I/O & Collection Utilities ──
    m.insert("read_line", EffectSet::new(EffectSet::IO | EffectSet::NONDET | EffectSet::ALLOC));
    m.insert("array_push", alloc);
    m.insert("array_pop", alloc);
    m.insert("array_contains", pure);
    m.insert("array_reverse", alloc);
    m.insert("array_flatten", alloc);
    m.insert("array_len", pure);
    m.insert("array_slice", alloc);

    m.insert("Set.new", alloc);
    m.insert("Map.insert", mutates);
    m.insert("Map.get", pure);
    m.insert("Map.remove", mutates);
    m.insert("Map.contains_key", pure);
    m.insert("Map.keys", alloc);
    m.insert("Map.values", alloc);
    m.insert("Set.add", mutates);
    m.insert("Set.contains", pure);
    m.insert("Set.remove", mutates);
    m.insert("Set.len", pure);
    m.insert("Set.to_array", alloc);

    // -----------------------------------------------------------------
    // ML Autodiff builtins — pure (semantic markers + simple arithmetic)
    // -----------------------------------------------------------------
    for name in &["stop_gradient", "grad_checkpoint", "clip_grad", "grad_scale"] {
        m.insert(name, pure);
    }

    // -----------------------------------------------------------------
    // Vizor builtins — plot construction (alloc) and file output (io)
    // -----------------------------------------------------------------
    m.insert("vizor_plot", alloc);
    m.insert("vizor_plot_xy", alloc);
    m.insert("vizor_plot_cat", alloc);
    m.insert("vizor_plot_matrix", alloc);
    m.insert("vizor_corr_matrix", alloc);
    m.insert("vizor_displot", alloc);
    m.insert("vizor_catplot", alloc);
    m.insert("vizor_relplot", alloc);
    m.insert("vizor_lmplot", alloc);
    m.insert("vizor_jointplot", alloc);
    m.insert("vizor_pairplot", alloc);
    m.insert("vizor_clustermap", alloc);
    // VizorPlot methods — builder methods allocate new PlotSpec
    for name in &[
        "VizorPlot.geom_point", "VizorPlot.geom_line",
        "VizorPlot.geom_bar", "VizorPlot.geom_histogram",
        "VizorPlot.geom_density", "VizorPlot.geom_density_bw",
        "VizorPlot.geom_area", "VizorPlot.geom_rug", "VizorPlot.geom_ecdf",
        "VizorPlot.geom_box", "VizorPlot.geom_violin",
        "VizorPlot.geom_strip", "VizorPlot.geom_swarm", "VizorPlot.geom_boxen",
        "VizorPlot.geom_tile", "VizorPlot.geom_regression", "VizorPlot.geom_residplot",
        "VizorPlot.geom_dendrogram",
        // Phase 3: Polar + 2D density
        "VizorPlot.geom_pie", "VizorPlot.geom_donut",
        "VizorPlot.geom_rose", "VizorPlot.geom_radar",
        "VizorPlot.coord_polar",
        "VizorPlot.geom_density2d", "VizorPlot.geom_contour",
        // Phase 3.2: Error bars, step, legend, scales
        "VizorPlot.geom_errorbar", "VizorPlot.geom_errorbar_col",
        "VizorPlot.geom_step",
        "VizorPlot.no_legend", "VizorPlot.subtitle",
        "VizorPlot.scale_x_log", "VizorPlot.scale_y_log",
        "VizorPlot.scale_color_diverging", "VizorPlot.show_values",
        "VizorPlot.facet_wrap", "VizorPlot.facet_wrap_ncol", "VizorPlot.facet_grid",
        "VizorPlot.title", "VizorPlot.xlab", "VizorPlot.ylab",
        "VizorPlot.xlim", "VizorPlot.ylim",
        "VizorPlot.theme_minimal", "VizorPlot.theme_publication", "VizorPlot.theme_dark",
        "VizorPlot.coord_flip",
        "VizorPlot.size",
        "VizorPlot.annotate_text", "VizorPlot.annotate_regression",
        "VizorPlot.annotate_ci", "VizorPlot.annotate_pvalue",
        "VizorPlot.annotate_event", "VizorPlot.annotate_note",
        "VizorPlot.annotate_data_note", "VizorPlot.annotate_inline_label",
        "VizorPlot.to_svg", "VizorPlot.to_bmp", "VizorPlot.to_png",
    ] {
        m.insert(name, alloc);
    }
    m.insert("VizorPlot.save", io_alloc); // writes file to disk

    m
}

/// Look up the effect set for a builtin. Returns `None` for unknown builtins
/// (which should be treated conservatively).
pub fn lookup(name: &str) -> Option<EffectSet> {
    // Use a thread-local cache to avoid rebuilding the map every time.
    // In a single-threaded interpreter this is fine.
    thread_local! {
        static REGISTRY: HashMap<&'static str, EffectSet> = builtin_effects();
    }
    REGISTRY.with(|r| r.get(name).copied())
}

/// Returns true if the given builtin is known to trigger GC.
pub fn is_gc_builtin(name: &str) -> bool {
    lookup(name).map_or(false, |e| e.has(EffectSet::GC))
}

/// Returns true if the given builtin is known and does NOT trigger GC.
/// Unknown builtins return false (conservative).
pub fn is_safe_builtin(name: &str) -> bool {
    lookup(name).map_or(false, |e| !e.has(EffectSet::GC))
}

/// Returns true if the given builtin may capture/store its arguments.
pub fn may_capture(name: &str) -> bool {
    lookup(name).map_or(true, |e| e.has(EffectSet::CAPTURES))
}

/// Returns true if the given builtin is nondeterministic.
pub fn is_nondeterministic(name: &str) -> bool {
    lookup(name).map_or(false, |e| e.has(EffectSet::NONDET))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gc_builtins_have_gc_flag() {
        let reg = builtin_effects();
        assert!(reg["gc_alloc"].has(EffectSet::GC));
        assert!(reg["gc_collect"].has(EffectSet::GC));
    }

    #[test]
    fn test_pure_math_builtins_are_pure() {
        let reg = builtin_effects();
        for name in &["sqrt", "floor", "abs", "isnan", "isinf", "int", "float"] {
            assert!(
                reg[name].is_pure(),
                "{} should be PURE but is {}",
                name,
                reg[name]
            );
        }
    }

    #[test]
    fn test_nondeterministic_builtins_flagged() {
        let reg = builtin_effects();
        assert!(reg["Tensor.randn"].has(EffectSet::NONDET));
        assert!(reg["clock"].has(EffectSet::NONDET));
        assert!(reg["n_distinct"].has(EffectSet::NONDET));
    }

    #[test]
    fn test_io_builtins_flagged() {
        let reg = builtin_effects();
        assert!(reg["print"].has(EffectSet::IO));
        assert!(reg["clock"].has(EffectSet::IO));
    }

    #[test]
    fn test_allocating_builtins_flagged() {
        let reg = builtin_effects();
        assert!(reg["Tensor.zeros"].has(EffectSet::ALLOC));
        assert!(reg["sort"].has(EffectSet::ALLOC));
        assert!(reg["str_extract"].has(EffectSet::ALLOC));
    }

    #[test]
    fn test_complex_ops_are_pure() {
        let reg = builtin_effects();
        for name in &[
            "Complex.re", "Complex.im", "Complex.abs", "Complex.conj",
            "Complex.norm_sq", "Complex.add", "Complex.mul",
        ] {
            assert!(
                reg[name].is_pure(),
                "{} should be PURE but is {}",
                name,
                reg[name]
            );
        }
    }

    #[test]
    fn test_nogc_safe_classification_matches_legacy() {
        // Every entry in the old is_safe_builtin() should be is_nogc_safe here.
        let legacy_safe = vec![
            "print", "Tensor.zeros", "Tensor.ones", "Tensor.randn",
            "Tensor.from_vec", "matmul", "Buffer.alloc", "len", "push",
            "assert", "assert_eq", "clock", "gc_live_count",
            "linalg.lu", "linalg.qr", "linalg.cholesky", "linalg.inv",
            "Tensor.slice", "Tensor.transpose", "Tensor.broadcast_to",
            "SparseCsr.matvec", "SparseCsr.to_dense", "SparseCoo.to_csr",
            "attention", "Tensor.softmax", "Tensor.layer_norm",
            "Tensor.relu", "Tensor.gelu", "Tensor.bmm", "Tensor.linear",
            "Tensor.transpose_last_two", "Tensor.conv1d", "Tensor.conv2d",
            "Tensor.maxpool2d", "Tensor.binned_sum",
            "Complex.re", "Complex.im", "Complex.abs", "Complex.conj",
            "Complex.norm_sq", "Complex.add", "Complex.mul",
            "F16.to_f64", "F16.to_f32",
            "Tensor.from_bytes", "Tensor.split_heads", "Tensor.merge_heads",
            "Tensor.view_reshape", "ByteSlice.as_tensor",
            "Scratchpad.new", "Scratchpad.append", "Scratchpad.append_tensor",
            "Scratchpad.as_tensor", "Scratchpad.len", "Scratchpad.capacity",
            "Scratchpad.dim", "Scratchpad.clear", "Scratchpad.is_empty",
            "PagedKvCache.new", "PagedKvCache.append", "PagedKvCache.append_tensor",
            "PagedKvCache.as_tensor", "PagedKvCache.clear", "PagedKvCache.len",
            "PagedKvCache.is_empty", "PagedKvCache.max_tokens", "PagedKvCache.dim",
            "PagedKvCache.num_blocks", "PagedKvCache.blocks_in_use",
            "PagedKvCache.get_token",
            "AlignedByteSlice.from_bytes", "AlignedByteSlice.as_tensor",
            "AlignedByteSlice.was_realigned", "AlignedByteSlice.len",
            "AlignedByteSlice.is_empty",
            "Map.new", "Map.insert", "Map.get", "Map.remove",
            "Map.len", "Map.contains_key", "Map.keys", "Map.values",
            "tidy_filter", "tidy_select", "tidy_mask_and",
            "tidy_nrows", "tidy_ncols", "tidy_column_names",
            "tidy_group_by", "tidy_ungroup", "tidy_ngroups",
            "tidy_slice", "tidy_slice_head", "tidy_slice_tail", "tidy_slice_sample",
            "tidy_distinct",
            "tidy_semi_join", "tidy_anti_join",
            "tidy_relocate", "tidy_drop_cols",
            "tidy_group_by_fast",
            "fct_collapse",
        ];
        let reg = builtin_effects();
        for name in &legacy_safe {
            let effects = reg.get(name).unwrap_or_else(|| {
                panic!("Legacy safe builtin '{}' not found in effect registry", name);
            });
            assert!(
                effects.is_nogc_safe(),
                "Legacy safe builtin '{}' should be nogc-safe but has effects: {}",
                name,
                effects
            );
        }
    }

    #[test]
    fn test_gc_builtins_classification_matches_legacy() {
        // gc_alloc and gc_collect are the only GC builtins
        let reg = builtin_effects();
        let gc_builtins: Vec<&str> = reg
            .iter()
            .filter(|(_, e)| e.has(EffectSet::GC))
            .map(|(name, _)| *name)
            .collect();
        assert!(gc_builtins.contains(&"gc_alloc"));
        assert!(gc_builtins.contains(&"gc_collect"));
        assert_eq!(gc_builtins.len(), 2, "Only gc_alloc and gc_collect should have GC flag");
    }
}
