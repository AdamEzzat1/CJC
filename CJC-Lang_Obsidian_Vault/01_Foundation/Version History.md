---
title: Version History
tags: [foundation, history]
status: Summarized from CHANGELOG and commit log
---

# Version History

Grounded in the git log, `CHANGELOG.md`, and the progress docs in `docs/`.

## v0.1.7 — Data Science Surface Expansion (2026-04-19)

Six-item expansion of the data-science surface. Additive across data, datetime, string, and regression layers. No pipeline breaks.

**1. DataFrame surface wiring** — 10 new [[TidyView Architecture]] dispatch arms in `crates/cjc-data/src/tidy_dispatch.rs`:

- `df_read_csv(path)` — parse CSV to DataFrame. Gotcha: the CSV reader infers `"0"`/`"1"` as `Bool`; cast explicitly if you need them as `Int`.
- `pivot_wider(df, id_cols, names_from, values_from)` — 4 args, reshape long → wide.
- `pivot_longer(df, cols, names_to, values_to)` — reshape wide → long.
- `df_distinct(df)` — drop duplicate rows.
- `df_rename(df, old_name, new_name)` — rename one column.
- `df_anti_join(df1, df2, on)` — rows in `df1` without a match in `df2`.
- `df_semi_join(df1, df2, on)` — rows in `df1` with a match in `df2`.
- `df_full_join(df1, df2, on)` — full outer join.
- `df_fill_na(df, col_name, fill_val)` — per-column NA replacement (3 args, not global).
- `df_drop_na(df)` — drop rows containing any NA.

See [[DataFrame DSL]] for usage shape.

**2. Default parameters** — `fn solve(x: f64, tol: f64 = 1e-6)` now parses and lowers correctly through AST (`Param::default: Option<Expr>` already existed), HIR (`cjc-hir/src/lib.rs:945`), MIR (`cjc-mir/src/lib.rs:671`). Both executors evaluate defaults in the caller's scope. `tests/test_defaults.rs` — 32 tests, all passing. See [[Default Parameters]].

**3. Date/time surface** — 10+ builtins in `crates/cjc-runtime/src/datetime.rs`: `parse_date`, `date_format`, `year`, `month`, `day`, `hour`, `minute`, `second`, `date_diff`, `date_add`, `now`. Timestamps are **epoch milliseconds (i64)**, not seconds. Unit strings for diff/add: `"ms" | "s" | "min" | "h" | "d"`. Format specifiers: `%Y %m %d %H %M %S`. `parse_date` returns `0` on malformed input (not an error). `tests/test_datetime.rs` — 19 tests. See [[Date Time Surface]].

**4. NA handling** — `NA` is a lexer keyword (`TokenKind::Na`) producing `Value::Na`. Builtins: `is_na`, `fill_na`/`fillna`, `drop_na`, `is_not_null`, `coalesce`. `na_count` is **not** a builtin — users compose it with `is_na` + a loop. `tests/test_na_handling.rs` — 18 tests. See [[NA Handling]].

**5. Ridge / Lasso / ElasticNet regression** — `ridge_regression`, `lasso_regression`, `elastic_net` in `crates/cjc-runtime/src/builtins.rs` (lines ~4735-5289). Deterministic coordinate descent with fixed iteration order (`0..n_features`) and Kahan summation. Returns a struct with `.coefficients`, `.intercept`, `.r_squared`, `.converged`, `.n_iter`, `.alpha` (and `.l1_ratio` for `elastic_net`). `tests/test_regularized_regression.rs` — 32 tests. See [[Regularized Regression]].

**6. String interpolation / f-strings** — `f"hello {name}, x is {x+1}"`. Lexed as `TokenKind::FStringLit` with segments (literal + optional interpolation), parsed to `ExprKind::FStringLit(Vec<(String, Option<Box<Expr>>)>)`. HIR desugars into `BinOp::Add(Str, Str)` chains — which is why `cjc-mir-exec` needs no direct handler. `tests/test_fstring.rs` — 21 tests. See [[Format Strings]].

**Test totals added**: 32 + 32 + 21 + 19 + 18 = **122 new tests, all passing**.

**Builtin count delta**: +10 DataFrame + 10 datetime + 3 regression + 0 NA (arms already existed) + 0 f-string (syntax, not builtin) = **+23 arms**. New totals (verified 2026-04-19 by grepping dispatch-arm pattern in `crates/cjc-runtime/src/builtins.rs`): **478 arms / 473 unique** (395 runtime + 83 quantum arms). See [[Builtins Catalog]].

## v0.1.6 — Regex Capture Groups (2026-04-18)

**Crate**: `cjc-regex`. No pipeline changes — purely additive.

New engine node:
- `NfaNode::Save(usize)` — stores current byte position into a capture slot during Pike VM epsilon closure.

New execution path — **Pike VM** (`pike_match_at`, `pike_add_state`, `pike_search`, `pike_search_from`):
- Each thread carries `slots: Vec<Option<usize>>` (2 slots per group: open + close).
- At `Split` nodes each branch gets its own slot array clone; at `Save(n)` nodes `slots[n] = Some(pos)`.
- First thread to reach a state wins (leftmost-longest priority) — O(n·m) preserved.
- Fast path (`nfa_match_at`) unchanged — `Save` treated as epsilon, zero overhead for non-capture APIs.

New syntax supported:
- `(...)` — capturing group (was grouping-only, now tracked by Pike VM)
- `(?P<name>...)` — named capturing group (Python syntax)
- `(?<name>...)` — named capturing group (PCRE syntax)

New public Rust API:
- `Capture` struct with `.start`, `.end`, `.extract()`, `.extract_str()`
- `CaptureResult` struct: `.full` (group 0), `.groups` (Vec), `.names` (BTreeMap), `.get(idx)`, `.get_named(name)`
- `find_captures()` / `find_all_captures()` — returning `CaptureResult`
- `capture_count()` — number of capturing groups in a pattern

New builtins (wired in all three canonical locations):
- `regex_captures(pattern, flags, text)` → array of strings (full match + groups)
- `regex_named_capture(pattern, flags, text, name)` → captured string or `""`
- `regex_capture_count(pattern, flags)` → integer

Compiler additions:
- `Nfa::num_groups`, `Nfa::group_names: BTreeMap<String, usize>`
- `Compiler::num_groups`, `Compiler::group_names`
- `parse_named_capture_body()` helper — shared by `(?P<name>...)` and `(?<name>...)` paths
- `parse_group_name()` — parses identifier between `<` and `>`

Test count: 320 total (77 internal + 7 doc + 77 integration + 159 organized suite including 41 new capture tests).

## v0.1.5 — Regex Engine Upgrade (2026-04-18)

**Crate**: `cjc-regex`. No pipeline changes — purely additive.

New syntax supported:
- POSIX character classes: `[:alpha:]`, `[:digit:]`, `[:alnum:]`, `[:space:]`, `[:blank:]`, `[:upper:]`, `[:lower:]`, `[:punct:]`, `[:print:]`, `[:graph:]`, `[:cntrl:]`, `[:xdigit:]` (used inside bracket expressions with double-bracket syntax)
- Non-capturing groups `(?:...)`
- Inline flags `(?i)`, `(?m)`, `(?s)`, `(?x)` + scoped `(?i:...)` form
- Absolute anchors `\A`, `\z`, `\Z`
- `\B` non-word-boundary assertion (was missing from executor)
- Counted repetition `{n}`, `{n,}`, `{n,m}` + lazy `{n}?` variants
- Unicode escapes `\uNNNN` and `\u{NNNN}` (emitted as UTF-8 byte sequences)

New public API:
- `MatchResult` struct with `.start`, `.end`, `.len()`, `.is_empty()`, `.extract()`, `.extract_str()`
- `find_match()` / `find_all_matches()` returning `MatchResult`
- `regex_explain()` — human-readable NFA description for debugging

New builtins (wired in all three canonical locations):
- `regex_or(p1, p2, ...)` — alternation composition
- `regex_seq(p1, p2, ...)` — sequence composition
- `regex_explain(pattern, flags?)` — NFA debug string

Safety limits added: `MAX_PATTERN_LEN = 4096`, `MAX_NODES = 65536`.

Test count: 263 total (77 internal + 7 doc + 77 integration + 102 new organized suite).

## v0.1.4 — Rebrand (2026-04-06)
- Project renamed from **CJC** to **CJC-Lang** (Computational Jacobian Core).
- CLI command: `cjc` → `cjcl`.
- File extension: `.cjc` → `.cjcl`.
- Install command: `cargo install cjc-lang`.
- Internal crate names remain `cjc-*` for continuity.

## v0.1.3
- Fixed `cargo install cjc` binary entry point.
- Commit `3061026`: "Enable `cargo install cjc` with binary entry point."

## v0.1.2
- Data science foundation (docs/CJC_DataScience_Readiness_Audit.md).
- ML infrastructure expansion.
- 30+ CLI commands (see [[CLI Surfaces]]).
- Comprehensive test suites added.
- Commit `435940d`: "Hardening, ML infrastructure, and comprehensive test suites."

## v0.1.0 — First public
- Lexer, parser, eval, types, dispatch, runtime, AD, data DSL.
- Tree-walk interpreter only.
- See `docs/RELEASE_NOTES_v0.1.0.md`.

## Stage 2 milestones (between v0.1.0 and v0.1.2)

These are described in `docs/CJC_STAGE2_*.md` and progress files:

- **Stage 2.0** — HIR + MIR + MIR-exec infrastructure bootstrapped.
- **Stage 2.1** — Closures with [[Capture Analysis]].
- **Stage 2.2** — Match expressions + structural destructuring (tuples, structs).
- **Stage 2.3** — For loops (range iteration, desugared to while).
- **Stage 2.4** — [[NoGC Verifier]] + [[MIR Optimizer]] (CF + DCE) + [[Parity Gates]] (G-8, G-10).

## Hardening phases

The `docs/` directory contains a long tail of hardening reports and phase changelogs:

- `BETA_HARDENING_PLAN.md` / `BETA_HARDENING_CHANGELOG.md`
- `PERFORMANCE_OPTIMIZATION_CHANGELOG.md`
- `PERFORMANCE_V2_CHANGELOG.md`
- `TIDYVIEW_HARDENING_CHANGELOG.md`
- `STAGE_2_6_HARDENING.md`
- `phase_b_changelog.md`, `phase_c_changelog.md`

These are **historical** but useful as context. They are not part of the current state model.

## Recent non-version commits

- `cb7c76c` — v0.1.4: Rebrand CJC to CJC-Lang
- `e171b18` — Add .gitignore entries for pack artifacts and proptest regressions
- `9c1270c` — Add comprehensive docstrings across all 20 library crates

## Related

- [[Current State of CJC-Lang]]
- [[Roadmap]]
