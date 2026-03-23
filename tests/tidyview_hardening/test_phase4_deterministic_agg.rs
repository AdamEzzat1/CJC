//! Phase 4 TidyView hardening tests: specialized aggregate kernels and
//! stable dictionary encoding.

use cjc_data::agg_kernels::*;
use cjc_data::dict_encoding::DictEncoding;
use cjc_repro::kahan_sum_f64;

// ── agg_sum_f64 ──────────────────────────────────────────────────────────────

#[test]
fn test_agg_sum_f64_matches_kahan() {
    let data: Vec<f64> = (1..=1000).map(|i| i as f64 * 0.001).collect();
    let segments = vec![(0, 500), (500, 1000)];
    let sums = agg_sum_f64(&data, &segments);

    let expected_0 = kahan_sum_f64(&data[0..500]);
    let expected_1 = kahan_sum_f64(&data[500..1000]);

    assert_eq!(sums[0].to_bits(), expected_0.to_bits());
    assert_eq!(sums[1].to_bits(), expected_1.to_bits());
}

// ── agg_mean_f64 ─────────────────────────────────────────────────────────────

#[test]
fn test_agg_mean_f64_known_data() {
    let data = vec![2.0, 4.0, 6.0, 10.0, 20.0, 30.0];
    let segments = vec![(0, 3), (3, 6)];
    let means = agg_mean_f64(&data, &segments);

    assert!((means[0] - 4.0).abs() < 1e-12);
    assert!((means[1] - 20.0).abs() < 1e-12);
}

#[test]
fn test_agg_mean_f64_empty_segment() {
    let data = vec![1.0, 2.0];
    let segments = vec![(0, 0)];
    let means = agg_mean_f64(&data, &segments);
    assert!(means[0].is_nan());
}

// ── agg_var_f64 (Welford) ────────────────────────────────────────────────────

#[test]
fn test_agg_var_f64_welford() {
    // Population variance of [2, 4, 4, 4, 5, 5, 7, 9] = 4.0
    let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
    let segments = vec![(0, 8)];
    let vars = agg_var_f64(&data, &segments);
    assert!((vars[0] - 4.0).abs() < 1e-12, "got {}", vars[0]);
}

#[test]
fn test_agg_sd_f64_welford() {
    let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
    let segments = vec![(0, 8)];
    let sds = agg_sd_f64(&data, &segments);
    assert!((sds[0] - 2.0).abs() < 1e-12, "got {}", sds[0]);
}

// ── agg_median_f64 ───────────────────────────────────────────────────────────

#[test]
fn test_agg_median_f64_odd_length() {
    let data = vec![3.0, 1.0, 2.0]; // sorted: 1, 2, 3 -> median 2
    let segments = vec![(0, 3)];
    let medians = agg_median_f64(&data, &segments);
    assert!((medians[0] - 2.0).abs() < 1e-12);
}

#[test]
fn test_agg_median_f64_even_length() {
    let data = vec![4.0, 1.0, 3.0, 2.0]; // sorted: 1, 2, 3, 4 -> median 2.5
    let segments = vec![(0, 4)];
    let medians = agg_median_f64(&data, &segments);
    assert!((medians[0] - 2.5).abs() < 1e-12);
}

// ── agg_n_distinct_str ───────────────────────────────────────────────────────

#[test]
fn test_agg_n_distinct_str_btree() {
    let data: Vec<String> = vec!["a", "b", "a", "c", "b", "c", "d"]
        .into_iter()
        .map(String::from)
        .collect();
    let segments = vec![(0, 4), (4, 7)];
    let counts = agg_n_distinct_str(&data, &segments);
    assert_eq!(counts[0], 3); // a, b, c
    assert_eq!(counts[1], 3); // b, c, d
}

// ── DictEncoding roundtrip ───────────────────────────────────────────────────

#[test]
fn test_dict_encoding_roundtrip() {
    let data: Vec<String> = vec!["banana", "apple", "cherry", "apple", "banana"]
        .into_iter()
        .map(String::from)
        .collect();
    let enc = DictEncoding::encode(&data);
    let decoded = enc.decode();
    assert_eq!(decoded, data);
}

// ── DictEncoding determinism ─────────────────────────────────────────────────

#[test]
fn test_dict_encoding_deterministic() {
    let data: Vec<String> = vec!["zebra", "apple", "mango", "banana", "apple", "zebra"]
        .into_iter()
        .map(String::from)
        .collect();

    let enc1 = DictEncoding::encode(&data);
    let enc2 = DictEncoding::encode(&data);
    let enc3 = DictEncoding::encode(&data);

    assert_eq!(enc1.codes(), enc2.codes());
    assert_eq!(enc2.codes(), enc3.codes());

    // BTreeMap sorts: apple=0, banana=1, mango=2, zebra=3
    assert_eq!(enc1.lookup("apple"), Some(0));
    assert_eq!(enc1.lookup("banana"), Some(1));
    assert_eq!(enc1.lookup("mango"), Some(2));
    assert_eq!(enc1.lookup("zebra"), Some(3));
    assert_eq!(enc1.cardinality(), 4);
}

// ── Stress test: 100K rows, determinism across 3 runs ────────────────────────

#[test]
fn test_stress_100k_determinism() {
    // Build a 100K-element f64 array with a known pattern
    let n = 100_000;
    let data: Vec<f64> = (0..n).map(|i| (i as f64) * 0.0001 + 0.5).collect();

    // 10 segments of 10K each
    let segments: Vec<(usize, usize)> = (0..10)
        .map(|i| (i * 10_000, (i + 1) * 10_000))
        .collect();

    let sums_1 = agg_sum_f64(&data, &segments);
    let sums_2 = agg_sum_f64(&data, &segments);
    let sums_3 = agg_sum_f64(&data, &segments);

    for i in 0..10 {
        assert_eq!(
            sums_1[i].to_bits(),
            sums_2[i].to_bits(),
            "sum mismatch at segment {} between run 1 and 2",
            i
        );
        assert_eq!(
            sums_2[i].to_bits(),
            sums_3[i].to_bits(),
            "sum mismatch at segment {} between run 2 and 3",
            i
        );
    }

    let means_1 = agg_mean_f64(&data, &segments);
    let means_2 = agg_mean_f64(&data, &segments);
    let means_3 = agg_mean_f64(&data, &segments);

    for i in 0..10 {
        assert_eq!(means_1[i].to_bits(), means_2[i].to_bits());
        assert_eq!(means_2[i].to_bits(), means_3[i].to_bits());
    }

    let vars_1 = agg_var_f64(&data, &segments);
    let vars_2 = agg_var_f64(&data, &segments);
    let vars_3 = agg_var_f64(&data, &segments);

    for i in 0..10 {
        assert_eq!(vars_1[i].to_bits(), vars_2[i].to_bits());
        assert_eq!(vars_2[i].to_bits(), vars_3[i].to_bits());
    }
}

// ── Gather-based kernels ─────────────────────────────────────────────────────

#[test]
fn test_gather_agg_sum_f64() {
    let data = vec![10.0, 20.0, 30.0, 40.0, 50.0];
    let groups = vec![vec![0, 2, 4], vec![1, 3]]; // [10+30+50=90, 20+40=60]
    let sums = gather_agg_sum_f64(&data, &groups);
    assert!((sums[0] - 90.0).abs() < 1e-12);
    assert!((sums[1] - 60.0).abs() < 1e-12);
}

#[test]
fn test_gather_agg_n_distinct_str() {
    let data: Vec<String> = vec!["a", "b", "a", "c", "b"]
        .into_iter()
        .map(String::from)
        .collect();
    let groups = vec![vec![0, 1, 2], vec![2, 3, 4]]; // {a,b}, {a,c,b}
    let counts = gather_agg_n_distinct_str(&data, &groups);
    assert_eq!(counts[0], 2);
    assert_eq!(counts[1], 3);
}

// ── agg_quantile_f64 ─────────────────────────────────────────────────────────

#[test]
fn test_agg_quantile_f64_median() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let segments = vec![(0, 5)];
    let q50 = agg_quantile_f64(&data, 0.5, &segments);
    assert!((q50[0] - 3.0).abs() < 1e-12);
}

// ── i64 kernels ──────────────────────────────────────────────────────────────

#[test]
fn test_agg_sum_i64_wrapping() {
    let data = vec![i64::MAX, 1];
    let segments = vec![(0, 2)];
    let sums = agg_sum_i64(&data, &segments);
    assert_eq!(sums[0], i64::MAX.wrapping_add(1));
}

#[test]
fn test_agg_min_max_i64() {
    let data = vec![5, 3, 8, 1, 9, 2];
    let segments = vec![(0, 3), (3, 6)];
    let mins = agg_min_i64(&data, &segments);
    let maxs = agg_max_i64(&data, &segments);
    assert_eq!(mins, vec![3, 1]);
    assert_eq!(maxs, vec![8, 9]);
}
