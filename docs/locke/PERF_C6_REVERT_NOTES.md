# C6 — Combined `distinct_count_and_top_freq`: revert recipe

**Date applied**: 2026-06-01
**Verification state at time of merge**:
- Functional: 408 lib + 277 integration tests passing
- Determinism: every `*_is_deterministic_across_runs` + 2 proptest properties + bolero fuzz target passing
- Performance signal: best A/B (back-to-back, same thermal state) — single-shot 1M dropped from 7.44 µs/row → 4.44 µs/row (**−40%**). At 10K/100K row counts, the difference fell within ~10% noise either way. Bench environment was thermally saturated; final-attempt stable benches showed up to 200% stddev on individual samples and 33% on medians, so the noise floor was higher than any small-N effect. Decision was to keep C6 based on the large-N signal + theoretical analysis (one column scan replaces two).

This note exists so a future revert is mechanical, even if the perf story changes.

---

## What C6 changed

C6 introduced `validation::distinct_count_and_top_freq(col) -> (u64, u64)` — a single-pass alternative to calling `distinct_count(col)` and `top_value_freq(col)` back to back. Two callsites switched:

1. `crates/cjc-locke/src/api.rs` — inside `validate()`, the per-column belief-report loop
2. `crates/cjc-locke/src/validation.rs::detect_constant_and_near_constant`

The standalone `distinct_count` and `top_value_freq` functions are still present and used by `leakage.rs::detect_id_like_columns`.

---

## How to revert (if a cool-system bench shows C6 is a net loss)

### Step 1 — `crates/cjc-locke/src/api.rs`

**Find** (inside `validate()`, around line 120):

```rust
        let missingness_rate = if n_total == 0.0 { 0.0 } else { missing / n_total };
        // Single-pass distinct + top-freq. Replaces two back-to-back full
        // column scans that each built their own per-value count map.
        let (distinct, top_freq) = crate::validation::distinct_count_and_top_freq(col);
        let constant = distinct <= 1;
        let near_constant = if n_total > 0.0 {
            top_freq as f64 / n_total >= opts.config.near_constant_threshold
        } else {
            false
        };
```

**Replace with** (the pre-C6 form — two separate calls, no combined helper):

```rust
        let missingness_rate = if n_total == 0.0 { 0.0 } else { missing / n_total };
        let distinct = crate::validation::distinct_count(col);
        let constant = distinct <= 1;
        let near_constant = if n_total > 0.0 {
            crate::validation::top_value_freq(col) as f64 / n_total
                >= opts.config.near_constant_threshold
        } else {
            false
        };
```

### Step 2 — `crates/cjc-locke/src/validation.rs::detect_constant_and_near_constant`

**Find** (around line 862):

```rust
    for (name, col) in &df.columns {
        let n = col.len() as u64;
        if n == 0 {
            continue;
        }
        // Single-pass: build the count map once, read distinct + top off it.
        // Previously this function scanned the column twice (once in
        // `distinct_count`, once in `top_value_freq`); the combined function
        // halves the per-column work for the common non-constant path.
        let (distinct, top) = distinct_count_and_top_freq(col);
        if distinct <= 1 {
            // ... constant finding ...
            continue;
        }
        let ratio = top as f64 / n as f64;
```

**Replace with** (the pre-C6 form — constant short-circuits before top_value_freq):

```rust
    for (name, col) in &df.columns {
        let n = col.len() as u64;
        if n == 0 {
            continue;
        }
        let distinct = distinct_count(col);
        if distinct <= 1 {
            // ... constant finding ...
            continue;
        }
        let top = top_value_freq(col);
        let ratio = top as f64 / n as f64;
```

(The `// ... constant finding ...` block — the `ValidationFinding::new("E9010", ...)` call — is unchanged in both versions. Keep it as is.)

### Step 3 — `crates/cjc-locke/src/validation.rs` cleanup

The combined helper and its private `count_distinct_and_top` worker become dead code after the revert. Delete this block (it's directly below the existing `fn count_top` helper):

```rust
/// Combined single-pass alternative to back-to-back
/// `(distinct_count(col), top_value_freq(col))`. Builds the per-value
/// count map *once* and reads both numbers off it, instead of scanning
/// the column twice and constructing two separate maps. Bit-identical
/// to the standalone calls for every column type.
///
/// Used by `api::validate()` and `detect_constant_and_near_constant`,
/// each of which previously did two full scans per column.
pub(crate) fn distinct_count_and_top_freq(col: &Column) -> (u64, u64) {
    // ... full match ...
}

fn count_distinct_and_top<T: Ord, I: IntoIterator<Item = T>>(values: I) -> (u64, u64) {
    let mut counts: BTreeMap<T, u64> = BTreeMap::new();
    for v in values {
        *counts.entry(v).or_insert(0) += 1;
    }
    let distinct = counts.len() as u64;
    let top = counts.values().copied().max().unwrap_or(0);
    (distinct, top)
}
```

### Step 4 — Verify

```bash
cargo build --release -p cjc-locke
cargo test --release -p cjc-locke --lib                  # expect 408/408
cargo test --release -p cjc-lang --test locke            # expect 277/277
cargo test --release --test locke -- --ignored --nocapture --test-threads=1 \
    scale_benchmark_single_shot_stable scale_benchmark_drift_compare_stable
```

Test the bench on an **idle, cool system** (close other apps, wait for thermal headroom, run twice to warm caches, compare medians). The bench's stddev is the gating signal — if it's >15%, the result is noise and you need a quieter environment.

---

## Why C6 might lose on a cool system

The theoretical analysis says C6 halves the per-column work. The mechanism that could make C6 *slower* in practice:

- The old `distinct_count` for Float used `BTreeSet<u64>::from_iter` which has an internal bulk-insert optimization. The combined helper uses `BTreeMap::entry().or_insert()` — slower per insertion.
- For columns where the distinct count is the *only* expensive operation (a small, low-cardinality column), the combined function still pays the BTreeMap overhead.
- At small N (≤10K rows), constant per-call overhead can dominate the savings.

If a cool-system bench shows C6 losing across the board, revert. If it shows the same large-N win the back-to-back A/B showed, keep.

---

## Companion change: `binary_search` in `extract_multiclass_target`

The same commit also replaced a `BTreeMap<i64, u32>` lookup table with `Vec::binary_search` on the already-sorted `labels: Vec<i64>` in `crates/cjc-locke/src/leakage.rs::extract_multiclass_target`. This is a strictly mechanical win (less code, less heap, identical complexity) — there is no scenario where it regresses. **Do not revert this** unless you have a specific reason; it is independent of C6.
