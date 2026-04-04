# CJC CLI Expansion v0.1.2 — Design Document

## Overview

This document describes the CLI expansion implemented for CJC v0.1.2, adding multi-format
support, second-mode operational flags, and `doctor --fix` capabilities across 15 CLI commands.

**Architectural approach:** Zero new external dependencies. All format parsing (JSONL, binary
metadata extraction) is implemented from scratch using only `std`. Binary formats (Parquet,
Arrow IPC, SQLite) support metadata-only inspection. Model files (.pkl, .onnx, .joblib) are
never deserialized or executed.

**Safety posture:** Model file inspection is limited to safe metadata: file size, magic byte
signatures, SHA-256 hash, and known-safe header fields. Full model inspection requires
specialized tools — this is documented explicitly in all relevant command help text.

**Reproducibility guarantees:** All new features use BTreeMap/BTreeSet (never HashMap/HashSet),
Kahan summation for numeric aggregation, Welford's algorithm for variance, stable output
ordering, and deterministic JSON serialization.

---

## Shared Infrastructure

### formats.rs (new module)

A shared multi-format tabular data loader providing:

| Component | Purpose |
|---|---|
| `DataFormat` enum | 10 variants: CSV, TSV, JSONL, Parquet, Arrow IPC, SQLite, Pickle, ONNX, Joblib, Unknown |
| `detect_format()` | Extension + magic byte detection |
| `load_delimited()` | CSV/TSV → `TabularData` |
| `load_jsonl()` | JSONL → `TabularData` (BTreeSet key discovery, deterministic column ordering) |
| `extract_metadata()` | Safe binary file metadata extraction |
| `TabularData` | Unified headers + rows + format representation |
| `FileMetadata` | Format, size, magic bytes, header info, limitations |

---

## Format Support by Command

| Command | CSV | TSV | JSONL | Parquet | Arrow IPC | SQLite | Model files |
|---------|-----|-----|-------|---------|-----------|--------|-------------|
| flow | full | full | **NEW full** | metadata | metadata | — | rejected |
| schema | full | full | **NEW full** | **NEW metadata** | **NEW metadata** | **NEW metadata** | rejected |
| patch | full | full | **NEW full** | — | — | — | — |
| drift | full | full | **NEW full** | — | — | — | — |
| inspect | full | full | **NEW full** | **NEW metadata** | **NEW metadata** | **NEW metadata** | **NEW safe metadata** |
| doctor | checks | checks | **NEW checks** | **NEW magic check** | — | — | **NEW presence report** |

"full" = complete row/column processing. "metadata" = safe header/magic byte inspection only.
"checks" = health/validity checks. "safe metadata" = size, hash, magic bytes only — never executed.

### Limitations

- **Parquet/Arrow IPC/SQLite:** Only metadata inspection is possible without external crates.
  Full row materialization would require `arrow-rs`, `parquet-rs`, or `rusqlite`. This is
  explicitly documented in command output when these formats are encountered.
- **Model files (.pkl, .onnx, .joblib):** Never deserialized. Only file size, SHA-256 hash,
  magic byte signature, and format detection are reported. Pickle protocol version is extracted
  from header bytes for .pkl files.
- **JSONL parsing:** Uses a custom minimal JSON parser. Supports strings (with escapes),
  numbers, booleans, null, and nested objects/arrays (stored as raw strings). Does not handle
  all JSON edge cases (e.g., very deeply nested structures may truncate).

---

## Second-Mode Flags by Command

### Selected flags and rationale

Flags were selected based on these criteria:
1. **Architectural fit** — naturally extends the command's existing purpose
2. **CI/pipeline value** — enables automated enforcement and gating
3. **Inspect → Preview → Enforce → Report workflow** — completes the operational lifecycle
4. **Low implementation risk** — reuses existing infrastructure

### flow
| Flag | Purpose |
|---|---|
| `--verify` | Run aggregation twice, confirm identical output (determinism check) |
| `--sort-by <metric>` | Sort output columns by aggregation metric |
| `--top <N>` | Show only top N columns |
| `--out <file>` | Write output to file |

### schema
| Flag | Purpose |
|---|---|
| `--save <file>` | Save inferred schema as deterministic JSON |
| `--check <file>` | Validate data against saved schema (exit 1 on mismatch) |
| `--diff <file>` | Show schema differences (+/-/~ notation) |
| `--strict` | Treat type warnings as errors in --check |
| `--full` | Show type distribution percentages and sample values |

### patch
| Flag | Purpose |
|---|---|
| `--plain` | Plain text status output |
| `--json` | JSON summary of patch operation |
| `--dry-run` | Preview transforms without writing output |
| `--plan` | Show structured transform plan |
| `--backup` | Create .bak before overwriting |
| `--in-place` | Modify input file directly |
| `--check` | Validate transforms would apply cleanly |

### drift
| Flag | Purpose |
|---|---|
| `--fail-on-diff` | Exit 1 if any differences (CI gating) |
| `--fail-on-schema-diff` | Exit 1 if column names differ |
| `--threshold <N>` | Alias for --tolerance |
| `--summary-only` | Only summary metrics, no diff details |
| `--stats-only` | Only statistical metrics |
| `--report <file>` | Save diff report as JSON |

### inspect
| Flag | Purpose |
|---|---|
| `--deep` | Compute variance, std, unique count |
| `--header-only` | File metadata only |
| `--schema-only` | Schema/type info only |
| `--hash` | Explicit SHA-256 request |
| `--manifest` | Machine-readable: `<hash> <size> <type> <path>` |
| `--compare <file>` | Side-by-side file comparison |

### doctor
| Flag | Purpose |
|---|---|
| `--fix` | Apply safe auto-fixes |
| `--dry-run` | Preview fixes without applying |
| `--category <type>` | Filter findings by category |
| `--report <file>` | Save findings as JSON |
| `--summary-only` | Only show counts |
| `--fail-on <level>` | Exit 1 at specified severity |

### proof
| Flag | Purpose |
|---|---|
| `--fail-fast` | Stop on first divergence |
| `--hash-output` | SHA-256 of combined stdout |
| `--save-report <file>` | JSON reproducibility report |
| `--executor eval\|mir\|both` | Choose/compare executors |
| `--stdout-only` | Skip GC comparison |

### bench
| Flag | Purpose |
|---|---|
| `--baseline <file>` | Compare against saved baseline |
| `--save-baseline <file>` | Save current results as baseline |
| `--fail-if-slower-than <pct>` | CI gating on performance |
| `--compare eval\|mir` | Side-by-side executor comparison |
| `--csv` | CSV output for pipeline ingestion |
| `--markdown` | Markdown table output |
| `--out <file>` | Write to file |

### seek
| Flag | Purpose |
|---|---|
| `--exclude <glob>` | Exclude matching files |
| `--ignore-build-artifacts` | Skip target/, node_modules/, etc. |
| `--first <N>` | Limit result count |
| `--sort size\|name\|modified` | Sort order |
| `--hash` | SHA-256 per file |
| `--manifest` | Machine-readable manifest output |

### pack
| Flag | Purpose |
|---|---|
| `--verify` | Verify packed manifest after creation |
| `--manifest-only` | Output manifest without creating package |
| `--list` | List existing package contents |
| `--repro-check` | Re-pack and compare against existing manifest |

### audit
| Flag | Purpose |
|---|---|
| `--strict` | Promote INFO to warnings, warnings to errors |
| `--report <file>` | Save findings as JSON |
| `--category <type>` | Filter by finding category |
| `--suggest-only` | Only suggestions, no code context |
| `--baseline <file>` | Compare against baseline |

### mem
| Flag | Purpose |
|---|---|
| `--peak-only` | Only peak metrics |
| `--timeline` | GC timeline per run |
| `--save-report <file>` | JSON memory report |
| `--fail-on-gc` | Exit 1 if any GC occurred |
| `--compare eval\|mir` | Compare executor GC behavior |

### lock
| Flag | Purpose |
|---|---|
| `--update` | Regenerate lockfile |
| `--show` | Display lockfile contents |
| `--diff` | Show lockfile vs current run |
| `--executor eval\|mir` | Executor selection |

### parity
| Flag | Purpose |
|---|---|
| `--explain-mismatch` | Show divergent output with context |
| `--save-report <file>` | JSON parity report |
| `--function <name>` | Test specific function only |

### precision
| Flag | Purpose |
|---|---|
| `--epsilon <N>` | Custom comparison epsilon |
| `--fail-on-instability` | Exit 1 if any value exceeds epsilon |
| `--report <file>` | JSON precision report |
| `--summary-only` | Only overall verdict |

---

## doctor --fix Semantics

### Permitted fixes (safe, deterministic, auditable)

1. **CSV ragged row repair** — pad short rows with empty fields or truncate extra fields
   to match header column count. Skipped if header column count is 0 (ambiguous).

2. **Trailing whitespace removal** — trim trailing spaces/tabs from .cjc source lines.

3. **Trailing newline normalization** — ensure files end with exactly one newline.

### Non-goals
- No speculative semantic rewrites
- No auto-formatting of CJC code beyond whitespace
- No silent changes — every fix is reported in the findings list
- If a fix is ambiguous, it is reported but NOT applied

### Preview mode
`--fix --dry-run` shows what WOULD be fixed without writing any files.

---

## Deferred Work

| Feature | Reason deferred |
|---|---|
| `flow --group-by`, `--where` | Requires expression parser; too large for this phase |
| `schema --promote-rules` | Needs type promotion DSL design |
| `patch --rejects` | Needs rejected-row accumulator |
| `drift --explain` | Needs natural language diff explanation |
| `forge --diff <hash>` | Needs artifact content diff infrastructure |
| `bench --pin-cpu-mode` | Platform-specific, not portable |
| `pack --sign` | Needs cryptographic signing infrastructure |
| `seek --exec` | Security risk without sandboxing |
| `audit --fix` | Only safe with very limited mechanical rewrites |
| `emit/explain/test_cmd flags` | Lower priority; current behavior is sufficient |
| Full Parquet/Arrow/SQLite parsing | Requires external crates; metadata-only is sufficient |

---

## Regression Strategy

- **2,130 total workspace tests, 0 failures**
- 105 cjc-cli unit tests (including 51 new formats module tests)
- 9 cjc-cli integration tests
- 48 new CLI expansion integration tests
- 38 property tests (12 new: CSV round-trip, JSONL consistency, Kahan accuracy, etc.)
- 17 Bolero fuzz tests (7 new: malformed JSONL, random CSV, random magic bytes, etc.)
- All existing fixture tests, hardening tests, and parity tests pass unchanged

---

## Files Changed

| File | Change type | Lines added |
|---|---|---|
| `crates/cjc-cli/src/formats.rs` | NEW | ~600 |
| `crates/cjc-cli/src/commands/flow.rs` | EXPANDED | ~410 |
| `crates/cjc-cli/src/commands/schema.rs` | EXPANDED | ~830 |
| `crates/cjc-cli/src/commands/patch.rs` | EXPANDED | ~1020 |
| `crates/cjc-cli/src/commands/drift.rs` | EXPANDED | ~400 |
| `crates/cjc-cli/src/commands/doctor.rs` | EXPANDED | ~565 |
| `crates/cjc-cli/src/commands/inspect.rs` | EXPANDED | ~720 |
| `crates/cjc-cli/src/commands/proof.rs` | EXPANDED | ~400 |
| `crates/cjc-cli/src/commands/bench.rs` | EXPANDED | ~980 |
| `crates/cjc-cli/src/commands/seek.rs` | EXPANDED | ~140 |
| `crates/cjc-cli/src/commands/pack.rs` | EXPANDED | ~330 |
| `crates/cjc-cli/src/commands/audit.rs` | EXPANDED | ~250 |
| `crates/cjc-cli/src/commands/mem.rs` | EXPANDED | ~260 |
| `crates/cjc-cli/src/commands/lock.rs` | EXPANDED | ~250 |
| `crates/cjc-cli/src/commands/parity.rs` | EXPANDED | ~145 |
| `crates/cjc-cli/src/commands/precision.rs` | EXPANDED | ~165 |
| `crates/cjc-cli/src/main.rs` | MODIFIED | +3 (mod formats) |
| `Cargo.toml` | MODIFIED | +4 (test entry) |
| `tests/test_cli_expansion.rs` | NEW | ~450 |
| `tests/prop_tests/cli_expansion_props.rs` | NEW | ~200 |
| `tests/bolero_fuzz/cli_expansion_fuzz.rs` | NEW | ~150 |
| `tests/fixtures/cli_expansion/*` | NEW | 7 fixture files |

**No unrelated files were touched.** No existing code was deleted except where replaced
by expanded versions of the same functions.

---

## v0.1.2.1 Hardening (post-benchmark)

### Bug Fixes

| ID | Issue | Fix | File |
|---|---|---|---|
| BUG-1 | `pack` silently produced empty packages from directories | Added `discover_dir_files()` directory walker with packable extension list (.cjc, .csv, .tsv, .jsonl, .json, .snap, .lock, .toml). Warning emitted for empty results. | `pack.rs` |
| BUG-2 | `inspect` typed bool columns as "string" while `schema` typed them as "bool" | Added `bool_count` to `ColStats` with case-insensitive bool detection matching schema's logic. | `inspect.rs` |
| BUG-2b | `schema` bool detection was case-sensitive ("true"/"false" only) | Changed to `to_ascii_lowercase()` to match "True"/"FALSE"/etc. | `schema.rs` |
| BUG-3 | `drift` reported inconsistent row counts (100000 vs 100001) depending on flags | Unified all CSV diff paths to always separate header from data rows. Removed legacy mode that included header in data. | `drift.rs` |

### Performance Fixes

| ID | Issue | Before | After | Fix |
|---|---|---|---|---|
| PERF-1 | `patch` JSONL superlinear scaling (44x for 10x data) | Per-row String alloc via `rebuild_jsonl_object()` | Direct `write!()` calls via `write_jsonl_object()` | `patch.rs` |
| PERF-2 | `seek --hash --first 5` took 1.3s for 5 small files | Hashed ALL files before applying --first limit | Hash computed only for displayed files (after truncation) | `seek.rs` |

### Guardrails

| ID | Fix | File |
|---|---|---|
| GUARD-3 | `bench` emits warning when CV > 15% and `--fail-if-slower-than` is active | `bench.rs` |
| GUARD-1 | `pack` warns "no packable files found" for empty directories | `pack.rs` |

### Limitations Updated
- **JSONL column ordering:** JSONL columns are alphabetically ordered (BTreeMap deterministic iteration). CSV columns preserve file order. This is by design for determinism but documented as a known difference.

### Test Additions

- **16 new hardening integration tests** (`tests/test_cli_hardening.rs`)
  - 4 pack discovery tests (dir files, empty dir, unsupported files, roundtrip verify)
  - 4 bool type agreement tests (inspect bool, schema bool, cross-command agreement, numeric types)
  - 4 drift row count tests (default, fail-on-diff, summary-only, cross-flag consistency)
  - 1 seek performance test (hash --first timing)
  - 2 cross-command tests (report JSON validity, bench CSV validity)
  - 1 patch JSONL test (rename preserves rows)

### Regression
- All 105 CLI unit tests pass
- All 9 CLI integration tests pass
- All 48 CLI expansion tests pass
- All 16 CLI hardening tests pass
- **Total: 178 CLI tests, 0 failures**
