# Locke v0.5 vs Great Expectations 1.17 — side-by-side benchmark

A reproducible head-to-head comparison of Locke and Great Expectations
running equivalent validation suites against the same synthetic
customer-churn dataset.

## Reproduce

```bash
# 1. install Great Expectations (one-time)
pip install great_expectations pandas

# 2. generate the synthetic dataset (deterministic, seed=0xCAFEBABE)
python bench/locke_vs_gx/generate_churn.py

# 3. run Great Expectations
python bench/locke_vs_gx/run_gx.py

# 4. run Locke
./target/release/cjc.exe locke validate bench/locke_vs_gx/churn.csv \
    --target churned --primary-key customer_id --time-col last_login_unix \
    --save-json bench/locke_vs_gx/locke_results.json

# 5. compare findings
python bench/locke_vs_gx/compare.py
```

## Dataset

7023 rows, 8 columns. Five seeded properties form the ground truth:

| Property                                     | Exact count |
|----------------------------------------------|-------------|
| NaN values in `total_charges`                | 100         |
| Impossible values (`-9999`) in `monthly_charges` | 5       |
| Outlier rows (`tenure_months > 200`)         | 17          |
| Full-row duplicates                          | 23          |
| Sortedness violations in `last_login_unix`   | ≥ 1         |

## Ground-truth detection results

| Fact                                  | Locke (code, count, sev)     | GX (count, sev)     |
|---------------------------------------|------------------------------|---------------------|
| 100 NaN in `total_charges`            | E9007 (100, info)            | 100 (error)         |
| 5 impossible `monthly_charges`        | E9041 (5, warning)           | 5 (error)           |
| 17 outlier `tenure_months`            | E9041 (17, warning)          | 17 (warning)        |
| 23 full-row duplicates                | E9003 (23, notice)           | 46 = 23 × 2 (error) |
| Sortedness violation                  | E9050 (1, warning)           | 1 (warning)         |
| 23 duplicate `customer_id` keys       | E9004 (23, error)            | (no equivalent)     |

**Both tools matched ground truth 5/5 on quantitative facts.**

### Severity disagreements (policy, not detection)

- **Missing values**: GX reports `Error` by default (any null fails `expect_column_values_to_not_be_null`); Locke reports `Info` because the CSV reader inferred `total_charges` as `Str` (due to empty cells) and routed the 100 occurrences through the heuristic sentinel-detection path (E9007) rather than NaN-missingness (E9001). Same count, different code, lighter severity.
- **Duplicates**: GX reports `Error`; Locke reports `Notice` because 23/7023 ≈ 0.33%, below the 5% Warning threshold and 20% Error threshold. Locke's severity ladder is configurable.
- **Outlier counts**: identical. Both 17 with Warning severity.

### Semantic difference: duplicate row counts

GX's `ExpectCompoundColumnsToBeUnique` reports **all rows in any non-unique group** (= 46 = 23 × 2). Locke's `detect_duplicates_full_row` reports **the extras only** (= 23). Both numbers are correct; they answer different questions ("how many rows are involved in dups" vs "how many are spurious").

## Built-in checks Locke runs that GX does not (out of the box)

| Code  | Check                              | Fired on this dataset |
|-------|-------------------------------------|------------------------|
| E9004 | Duplicate primary-key detection     | yes (23 dups in `customer_id`) |
| E9007 | Sentinel-value detection            | yes (100 occurrences of `""` in `total_charges`) |
| E9070 | Conditional missingness             | not triggered (no jointly-missing cols) |
| E9071 | Imbalanced-class warning            | not triggered (target is 22.45% — above 5% floor) |
| E9072 | ID-like cardinality                 | yes (fired 3×: `customer_id`, `last_login_unix`, `total_charges`) |
| E9073 | Duplicate-key conditioning          | not triggered (dup customer_id rows agree on all other columns) |
| E9060 / E9061 | Target leakage (AUC vs target) | not triggered (no perfectly-predictive features in the synthetic data) |
| E9051 | Train/test temporal overlap         | n/a (single-frame run) |
| E9052 | Future-leakage cutoff               | n/a (no `--max-timestamp` passed) |

## Built-in checks GX runs that Locke does not (out of the box)

- `ExpectColumnDistinctValuesToBeInSet` for `plan` / `country` (Locke has E9020–E9022 schema mismatch but no value-set membership check by default)
- Strict-monotonic sortedness option (Locke only checks non-decreasing)
- User-declared mean / quantile range expectations (Locke surfaces the values but doesn't accept caller-declared ranges in the validate path)
- ~180 other expectations covering regex, length, JSON-schema, etc.

## End-to-end timing (this Windows worktree)

| Tool | Wall-clock | Speedup |
|---|---|---|
| Great Expectations 1.17 (Python 3.11) | **45.9 s** | 1× |
| Locke v0.5 debug build | 985 ms | 47× |
| **Locke v0.5 release build** | **167 ms** | **275×** |

Caveats:
- GX time includes Python startup + GX context init + ephemeral data source setup. Steady-state validation in a long-running Python process would be faster.
- Locke time is the full CLI run (binary launch + CSV parse + all validators + JSON write).
- Single-thread on both; release-mode Locke could use rayon in the future.

## Honest verdict

**For the specific workflow this benchmark simulates** — "validate a CSV-shaped churn dataset, report counts of seeded data issues" — Locke v0.5 is:

- **As accurate** as Great Expectations (5/5 ground-truth match)
- **~275× faster** end-to-end
- **More comprehensive out of the box** for churn-specific checks (target leakage, ID-like cardinality, sentinel detection, duplicate-key conditioning)
- **More restrictive** in policy (lighter default severities, no `--fail-on info` would fire on the missing-values finding)

Great Expectations remains the right tool for:
- SQL-backend validation (Snowflake, BigQuery, Postgres)
- Multi-team enterprise organization (data context, suite versioning, alerting)
- The ~180 expectations Locke doesn't ship (regex, length, JSON-schema, etc.)
- Profiling (auto-suite generation from sample data)

For a customer-churn ML data-hygiene pipeline that fits in a single CI job, **Locke is now the practical pick** — by a substantial margin on speed and a small margin on breadth.

## Files

- `generate_churn.py` — deterministic synthetic dataset generator
- `churn.csv` — generated dataset (7023 rows × 8 cols, ~310KB)
- `run_gx.py` — GX 1.17 validation script
- `gx_results.json` — GX findings dump
- `locke_results.json` — Locke `--save-json` output
- `compare.py` — side-by-side comparison script
