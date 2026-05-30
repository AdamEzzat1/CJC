# cjc-locke (Python)

Thin PyO3 wrapper over the Rust [`cjc-locke`](../crates/cjc-locke) crate —
deterministic dataset validation, drift detection, and lineage audit for
Python users.

## Why

You want Locke's contract (byte-identical reports across runs, machines,
and process boundaries) without leaving Python. This wrapper gives you
exactly that: every call delegates to one Rust function. There is no
business logic on the Python side, no Python-level caching, no extra
allocations beyond a single column copy at the FFI boundary.

## Install

```bash
pip install maturin
cd python/
maturin develop --release          # local dev install
# or
maturin build --release            # produce a wheel in target/wheels/
pip install target/wheels/cjc_locke-*.whl
```

The build needs Rust 1.63+ (Rust 1.91 is what this project uses) and
Python 3.9+. The wheel is `abi3-py39` so a single artifact works for every
CPython 3.9 through latest.

## Performance and determinism guarantees

- **Determinism**: byte-identical to a native Rust call. Python dict
  insertion order is preserved per PEP 468; `cjc_data::DataFrame`
  canonicalises everything downstream via `BTreeMap`. The Rust side does
  not multithread by default and the wrapper introduces no threading.
- **Memory**: one heap allocation per column at the FFI boundary
  (`Vec<f64>` / `Vec<i64>` / `Vec<String>` / `Vec<bool>`), then zero copies
  through to the report. For numpy `f64`/`i64`/`bool` columns the FFI read
  is zero-copy via the buffer protocol; only the `to_vec()` into Rust-owned
  memory counts.
- **Power / thermal**: identical to native — the Rust side does the same
  work it would do from a `cjcl` CLI run. No background threads, no
  asyncio, no extra cores.
- **Float / integer fidelity**: numpy `f64`/`i64` are bit-exact passed
  through. `f32`/`i32` widen via direct cast (lossless). Python `int`
  extracted as `i64` errors on overflow (PyO3 default).

## Quick start

```python
import numpy as np
import cjc_locke

# 1. Dict-of-arrays / lists in.
data = {
    "age": np.array([20.0, 30.0, np.nan, 40.0, 99.0]),
    "city": ["NY", "LA", "NY", "SF", "NY"],
    "is_active": [True, True, False, True, True],
}

report = cjc_locke.validate(data, label="users")
print(report)                       # <LockeReport n_findings=2 ...>
print(report.severity_counts)       # {'info': 0, 'notice': 0, 'warning': 1, 'error': 1}
print(report.finding_codes())       # ['E9001', 'E9041']

# 2. Full per-finding evidence via JSON round-trip (cheap).
detail = report.to_dict()
for f in detail["findings"]:
    print(f["code"], f["severity"], f["message"])

# 3. Canonical bytes for downstream audit chains.
canonical = report.to_json()        # byte-identical to a native Rust call
```

## Drift detection

```python
report = cjc_locke.compare_drift(
    train={"x": np.arange(1000.0)},
    test={"x": np.arange(500.0, 1500.0)},
)
print(report.finding_codes())       # ['E9030', 'E9039', ...]

# Or all in one go:
val, drift, belief = cjc_locke.validate_and_compare(train, test, label="my-data")
print(belief.score_dict())
# {'overall': 0.78, 'schema': 1.0, 'drift': 0.42, ...}
```

## Streaming (out-of-core)

```python
sv = cjc_locke.StreamingValidator(label="stream", config={"sample_cap": 100_000})
for chunk in iter_chunks_somehow():
    sv.ingest_chunk(chunk)          # chunk is a dict like above
final = sv.into_report()
```

## Policy gates

```python
result = cjc_locke.apply_policy(report, policy={
    "suppressions": [
        {"code": "E9001", "column": "phone", "reason": "PII expected"},
    ],
    "owners": [
        {"team": "data-quality", "code": "E9072"},
    ],
    "requirements": [
        {"code": "E9001", "operator": "eq_zero", "threshold": 0},
    ],
})

if result.gate_fails():
    raise SystemExit("policy gate failed")
```

## Lineage audit

```python
b = cjc_locke.LineageBuilder("daily-pipeline")
src = b.add_impression("train.csv", kind="dataset", n_rows=10_000,
                       columns=["x", "y"])
node = b.add_idea(name="filter_active",
                  op_id="filter",
                  parents=[src],
                  params={"expr": "is_active == True"})
graph = b.finish()

assert graph.is_acyclic()
graph.validate_audit_monotonic()    # raises ValueError if violated
print(graph.emit_mermaid())          # graph rendering for docs
```

## Pandas / polars

The Python facade auto-detects pandas or polars DataFrames by duck-typing —
both libraries remain optional dependencies. Internally we call `df[col].values`
(pandas) or `df[col].to_numpy()` (polars) per column, then route through the
same numpy zero-copy path as a raw dict. No extra business logic.

```python
import pandas as pd
df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
report = cjc_locke.validate(df)        # works
```

## What's exposed

Everything from the `cjc-locke` crate's public surface that translates
cleanly to Python:

- `validate`, `compare_drift`, `validate_and_compare`
- `belief_report`, `apply_policy`
- `causal_guardrail`, `detect_temporal_issues`
- `emit_report_json`, `parse_report_json`
- `make_audit_event`
- Classes: `LockeReport`, `InductionRiskReport`, `BeliefReport`,
  `CausalGuardrailReport`, `StreamingValidator`, `LineageBuilder`,
  `LineageGraph`, `AuditEvent`, `PolicyResult`

The Rust-side `TracedDataFrame` (which uses lifetime-borrowed references
to a builder) does not map to Python; use `LineageBuilder.add_idea(...)`
directly to build the same provenance graph.

## License

MIT — same as the rest of the workspace.
