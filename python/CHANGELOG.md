# Changelog

All notable changes to the `cjc-locke` Python wrapper.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
the project follows [SemVer](https://semver.org/).

The Python wrapper sits on top of the [`cjc-locke`](https://crates.io/crates/cjc-locke)
Rust crate. Wrapper versions may bump independently of the underlying
crate (the wrapper's version is in `Cargo.toml` and `pyproject.toml`,
the crate's version is pinned via `Cargo.toml`'s path dependency).

## [0.1.0] — 2026-05-30

Initial public release.

### Added

- Full-surface PyO3 bindings over `cjc-locke` v0.1.10:
  - `validate(data, label, options)` — single-shot DataFrame validation
  - `compare_drift(train, test, drift_config)` — distribution drift
  - `validate_and_compare(train, test, ...)` — combined val + drift + belief
  - `belief_report(report, model)` — derive a `BeliefReport` with the
    8-axis meet-semilattice score
  - `causal_guardrail(data, target, config, ...)` — correlation + confounder audit
  - `detect_temporal_issues(data, time_col, config)` — temporal sanity checks
  - `apply_policy(report, policy)` — suppression/owner/requirement gate
  - `make_audit_event(...)` — manual `AuditEvent` construction
  - `emit_report_json(report)` / `parse_report_json(json)` — canonical
    byte-identical serialization
- Classes: `LockeReport`, `InductionRiskReport`, `BeliefReport`,
  `CausalGuardrailReport`, `StreamingValidator`, `LineageBuilder`,
  `LineageGraph`, `AuditEvent`, `PolicyResult`
- Zero-copy buffer-protocol path for `numpy` `f64`/`i64`/`bool`/`f32`/`i32`
  arrays
- Duck-typed adapters for `pandas.DataFrame` and `polars.DataFrame`
  (both optional — neither is a hard dependency)
- `abi3-py39` wheel: a single wheel covers every CPython 3.9 through
  the latest, with no per-version rebuild
- Release-profile LTO + 1 codegen unit, matching the Rust workspace's
  release settings

### Determinism contract

- `cjc_locke.validate(df).to_json()` produces byte-identical output
  across runs, machines, and CI vendors
- Finding IDs are content-addressed 64-bit fingerprints (no
  timestamps, no UUIDs, no map-iteration nondeterminism)
- Audit chain canaries at the underlying Rust crate's test suite hold
  at 629/629 passing

### Documentation

- `README.md` — install instructions, quick-start examples for
  pandas/polars/numpy/dict inputs, determinism guarantees, full
  Locke capability surface
- Inline docstrings on every public function and class
- Reproduces the side-by-side comparison from the
  [Locke vs Pandera vs Great Expectations](https://adamezzat1.github.io/blog/posts/locke-side-by-side-pandera-ge/)
  blog post

### Known limitations

- `TracedDataFrame` (Rust's lifetime-borrowed DataFrame wrapper) does
  not map to Python; users build provenance graphs via
  `LineageBuilder.add_idea(...)` directly
- HTML emit (`emit_locke_report_html`) is not yet bound through the
  Python wrapper — accessible from the `cjcl locke` CLI
- `_repr_html_` for Jupyter-rich display is not yet implemented
  (planned for 0.2.0 — see the
  [post-launch adoption roadmap](https://github.com/AdamEzzat1/CJC/blob/master/docs/locke/POST_LAUNCH_ADOPTION_HANDOFF.md))
- Apache Arrow `RecordBatch` / Parquet zero-copy path is not yet
  available (planned for 0.2.0)
- No `sklearn.Pipeline` integration yet (planned for 0.2.0)

## Unreleased

Planned for the next minor release:

- `LockeReport._repr_html_` and per-class rich Jupyter rendering
- `pyarrow.Table` / `pyarrow.RecordBatch` zero-copy support
- `LockeCheck` sklearn `BaseEstimator + TransformerMixin` adapter
- `style_with_locke(df, report)` pandas `Styler` helper
- End-to-end notebooks under `python/examples/`
