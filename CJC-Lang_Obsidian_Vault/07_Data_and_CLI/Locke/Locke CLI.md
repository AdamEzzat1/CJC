# Locke CLI

The `cjcl locke` subcommand is the user-facing entry point. All output is deterministic and snapshot-friendly.

## Subcommands

```bash
cjcl locke validate <data.csv> [--label NAME] [--json] [--fail-on SEV] [--save-json PATH] [--html PATH]
                                [--time-col COL] [--max-timestamp N] [--gap-threshold N]
                                [--target COL] [--primary-key COL]
cjcl locke drift    <train.csv> <test.csv>    [--json] [--fail-on SEV]
cjcl locke belief   <data.csv>                [--json]
cjcl locke lineage  <data.csv> [--label NAME] [--mermaid]
cjcl locke causal   <data.csv> [--target COL] [--observational-only] [--json]
cjcl locke gate     <reference.json> <current> [--fail-on SEV]
cjcl locke verify   <data.csv> [--runs N]
```

## Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--json`               | off    | emit one JSON-ish record per line |
| `--text`               | on     | emit indented text |
| `--fail-on SEV`        | none   | exit non-zero if worst severity >= SEV (`info`, `notice`, `warning`, `error`) |
| `--label NAME`         | path   | override the dataset label |
| `--target COL`         | none   | target column for confounder hints / target-leakage AUC |
| `--observational-only` | off    | declare observational data; raises causal warning severity |
| `--save-json PATH`     | none   | persist the canonical JSON report (v0.4) |
| `--html PATH`          | none   | emit a self-contained HTML report with inline-SVG heatmap (v0.5) |
| `--time-col COL`       | none   | declare the time column (enables E9050–E9054 temporal checks) |
| `--max-timestamp N`    | none   | declare the maximum-allowed timestamp (E9052 future leakage) |
| `--gap-threshold N`    | none   | E9053 firing threshold for time-gap detection |
| `--primary-key COL`    | none   | declare the primary-key column (E9004 / E9073) |
| `--mermaid` (v0.6)     | off    | for `lineage`: emit a Quarto/Markdown Mermaid block instead of text |
| `--runs N` (v0.6)      | 3      | for `verify`: number of validation runs to compare (must be ≥ 2) |

## v0.6 subcommands

### `cjcl locke verify`

Re-runs `validate` `N` times and asserts every report is byte-identical. Catches accidental nondeterminism in user code that wraps Locke (e.g., a wrapper that mutates a HashMap in the file-reading path). Exits 0 when all runs agree, non-zero with a `status: DIVERGENT` line otherwise.

```bash
$ cjcl locke verify train.csv --runs 5
# Locke Reproducibility Verifier
dataset: train.csv
runs: 5
run_id: 1075bbad9d9b882f
findings: 17
report_bytes: 4823
status: REPRODUCIBLE — all runs byte-identical
```

### `cjcl locke lineage --mermaid`

Replaces the text emit with a Mermaid `flowchart LR` block. Impressions render as cylinder nodes (`[(...)]`), ideas as rounded nodes (`(...)`). Edge labels carry the transformation op id. Output is determined entirely by the lineage graph's content-addressed ids — two runs over the same input emit identical bytes.

```bash
$ cjcl locke lineage train.csv --label "Q3-2026" --mermaid > lineage.qmd
```

The output is suitable for direct pasting into a Quarto document or rendering with the [`mermaid-cli`](https://github.com/mermaid-js/mermaid-cli) tool.

## Examples

### Validate

```bash
$ cjcl locke validate train.csv
# Locke Validation Report
schema_version: 1
dataset: train.csv
n_rows: 5
n_cols: 3
run_id: 1075bbad9d9b882f
severity_counts: info=2 notice=0 warning=0 error=1
assumptions:
  - NaN treated as missing for Float columns; other types report a limitation
  - missingness, drift, and belief use deterministic Kahan summation
  - duplicate detection is byte-canonical
findings:
  - code=E9001 severity=warning column=age id=ab12cd...
    message: 12 of 100 values in `age` are NaN
    ...
```

### Drift

```bash
$ cjcl locke drift train.csv test.csv
# Locke Induction-Risk Report
n_train: 1000
n_test:  200
shared_columns: age, country, income
train_only:
test_only:
findings:
  - code=E9030 severity=warning column=income id=...
    message: mean of `income` shifted: train=50000.000000, test=58000.000000 (relative shift 0.1600)
    ...
```

### Belief

```bash
$ cjcl locke belief data.csv
# Locke Belief Report
dataset: data.csv
n_rows: 1000
sample_score (n=1000): 0.990

overall=0.987
  schema      = 1.000
  missingness = 1.000
  drift       = 1.000
  leakage     = 1.000
  lineage     = 1.000
  sample      = 0.990
  duplication = 1.000
  constraint  = 1.000

assumptions:
  - NaN treated as missing for Float columns; other types report a limitation
  - drift_score = 1.0 by default (no comparison dataframe supplied)
  - leakage_score = 1.0 by default (Locke v0 does not infer leakage automatically)
  - lineage_score = 1.0 by default (no lineage graph supplied)
```

### Lineage

```bash
$ cjcl locke lineage train.csv
# Locke Lineage Graph
run_label: validate-run
root: a1b2c3d4e5f6...
nodes:
  - id=... kind=impression source=train.csv schema=... n_rows=1000
edges:
audit:
  - seq=0 kind=impression subject=... note=added impression
```

### Causal

```bash
$ cjcl locke causal cohort.csv --target survived --observational-only
# Locke Causal-Guardrail Report
disclaimer: Locke does not infer causal effects. Every warning below is an association-based risk indicator, not a causal claim.
mode: ObservationalOnly
n_correlations_inspected: 6
warnings:
  - ObservationalOnly: dataset is declared observational-only ... [a=*, b=*]
  - StrongCorrelationNoIntervention: age and survived show |r|=0.81 ≥ 0.70 ... [a=age, b=survived]
confounder_hints:
  - candidate=class feature=age target=survived r_with_feature=0.62 r_with_target=0.55
```

## CI usage with `--fail-on`

```bash
# Fail the build if any drift finding is Warning or higher.
cjcl locke drift train.csv prod.csv --fail-on warning
echo "exit=$?"
```

Exit codes:

| Code | Meaning |
|------|---------|
| 0    | clean run, no severity exceeded the threshold |
| 1    | worst severity ≥ threshold (only when `--fail-on` is set) |
| 2    | argument parse error / I/O error / CSV parse error |

## Determinism guarantees

Two invocations with the same CSV inputs produce **byte-identical stdout**. Verified by:

- `tests/locke/determinism_tests.rs` (Rust-level invariant)
- the smoke test in this development session (`diff -s a.txt b.txt`)

The `run_id` field in the validation report header is also content-addressed, so it's identical across runs.

## JSON output (alternate)

```bash
$ cjcl locke validate data.csv --json
{"kind":"header","dataset":"data.csv","n_rows":5,"n_cols":3,"run_id":"1075bbad9d9b882f"}
{"kind":"finding","code":"E9001","severity":"warning","column":"age","id":"ab12...","message":"12 of 100 values in `age` are NaN"}
{"kind":"finding","code":"E9003","severity":"error","column":"","id":"cd34...","message":"1 duplicate rows across 1 groups"}
```

One record per line for easy `jq`/`awk` filtering. The format is intentionally newline-delimited rather than a single JSON object — large reports stream cleanly.
