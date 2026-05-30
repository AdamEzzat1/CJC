# Locke Deep-Dive Audit — Remaining Fixes Handoff

**Date stamped:** 2026-05-30
**Branch:** `claude/interesting-kalam-9d28ef`
**Last commit shipped:** `4277599` (bench SVG) on top of `a931afa` (perf batch) on top of `3afc3f0` (CRITICAL bugs)
**Audit baseline:** 4 parallel deep-dive subagents covered ~26K LOC of cjc-locke + ~6K LOC of tests/locke and returned **87 prioritised findings**.

This session shipped **11 fixes** (4 CRITICAL bugs + 6 perf wins + 1 XSS escape). **~76 findings remain**, prioritised below by impact × effort × risk.

---

## Section 0 — Read this first

### The non-negotiable constraints

These came from the audit briefs and govern every remaining fix:

1. **Determinism**: every finding ID is content-addressed; reports reproduce byte-for-byte across runs. NO `HashMap`/`HashSet` (must be `BTreeMap`/`BTreeSet`). NO `Instant`/`SystemTime`/`chrono`. NO thread IDs. NO `partial_cmp(...).unwrap()` on potentially-NaN floats.
2. **Byte-identical floats**: Kahan or `KahanAccumulatorF64` for f64 reductions. NO FMA (fused multiply-add). NO parallel reductions that change order. `cjc-repro` ships `kahan_sum_f64`, `pairwise_sum_f64`, `KahanAccumulatorF64x4`, `KahanAccumulatorF64x8`.
3. **Backward compatibility**: existing findings (E9001-E9112) must keep firing with same messages/severity. Changing a finding's evidence vector changes its ID, which breaks audit chains. If a fix MUST change IDs, document loudly and version-bump.
4. **Zero external deps**: only `cjc-data`, `cjc-repro`, `cjc-runtime`. NO `regex`, `serde`, `serde_json`, `unicode-normalization`, `ryu`.

### Test baselines to preserve

| Suite | Baseline | What it covers |
|---|---|---|
| `cargo test -p cjc-locke --lib --release` | **383 passing** | All in-module unit tests |
| `cargo test --test locke --release` | **266 passing**, 5 ignored | Integration tests including 9 deep-dive regression tests |
| `cargo test --test abng --release` | **629/629 canaries unmoved**, 11 ignored | Audit chain determinism canaries — MUST stay at 629 |
| `cargo test -p cjc-cli --lib --release` | **181 passing** | CLI parser tests |

### How to pick up cleanly

```bash
git fetch origin
git checkout claude/interesting-kalam-9d28ef
# Verify baseline:
cargo test -p cjc-locke --lib --release 2>&1 | tail -3   # 383 passing
cargo test --test locke --release 2>&1 | tail -3         # 266 passing
cargo test --test abng --release 2>&1 | tail -3          # 629 passing
```

If any of those numbers don't match, STOP — the substrate has changed since this doc was written.

---

## Section 1 — Already shipped (do NOT redo)

### Commit `3afc3f0` — CRITICAL bug fixes

1. **`CategoricalAdaptive::cell_bytes`** — was writing row index, no duplicate detection possible. Now dereferences dictionary. Shared `b'c'` tag with `Column::Categorical`.
2. **`distinct_count` for CategoricalAdaptive** — was returning row count. Now returns `cc.dictionary().len()`.
3. **`top_value_freq` for CategoricalAdaptive** — was returning 0. Now iterates code stream.
4. **`ks_d_statistic` tied-cluster fix** — was over-counting D on tied inputs (`a=[1,1,2,2], b=[1,2]` returned 0.25 vs textbook 0). Now advances at distinct breakpoints.

9 regression tests in `tests/locke/deep_dive_regression_tests.rs`.

### Commit `a931afa` — perf batch (byte-identical output)

5. `validation.rs::detect_missingness` Float branch — eliminated BTreeSet allocation.
6. `categorical.rs::detect_rare_categories` — removed redundant double-collect.
7. `policy.rs::apply_policy` — O(R×F) → O(F + R log F) via code_counts precompute.
8. `per_value_lineage.rs` — lazy stage groups + hoisted `any_above_threshold`.
9. `html_emit.rs` — Pearson upper-triangle only.
10. `html_emit.rs` — SVG `<title>` tooltip HTML-escape (low-risk XSS).

---

## Section 2 — Recommended Batch 3: JSON parser hardening

**Why first**: The hand-rolled JSON parser is the highest-risk surface (no `serde_json` dep means every bug is yours). Four HIGH-severity findings cluster in one file — fixing them together is cleanest.

### B3.1 — `json_emit.rs::parse_int` silently wraps negatives to u64

**File:** `crates/cjc-locke/src/json_emit.rs` lines 615-628 + every caller using `as u64` (lines 290, 309-318, 369, 454)
**Severity:** HIGH (silent value corruption on malformed input)

`parse_int` returns `i64`. Call sites cast `as u64`. A hostile or corrupted JSON containing `"n_rows": -1` silently wraps to `u64::MAX`. The report parses "successfully" with absurd values. Breaks audit-chain integrity guarantee.

**Fix sketch:**
- Add `parse_u64()` that errors on negative input.
- Replace every `parse_int()? as u64` site with `parse_u64()?`.
- Same treatment for `as usize` row_range parsing (out-of-bounds usize on 32-bit hosts).

**Regression test:**
```
- malformed: {"input": {"n_rows": -1, ...}} should ERROR not wrap
- malformed: {"input": {"n_rows": 99999999999999999999}} should ERROR not wrap
```

**Effort:** SMALL. **Risks determinism:** NO.

### B3.2 — `parse_locke_report_json` leaks memory permanently

**File:** `crates/cjc-locke/src/json_emit.rs` line 396 (`box_leak_str`) called from line 354
**Severity:** HIGH (monotonic heap growth in any long-lived process)

`box_leak_str` calls `Box::leak` on every `code` string per parsed finding. The leak is documented as bounded by distinct codes, BUT codes are NOT deduplicated. A 10K-finding report leaks 10K small strings. `cjcl locke verify --runs 100` and any long-lived service using `dispatch.rs::REPORTS` accumulate.

**Fix sketch:**
- Maintain a `thread_local! HashSet<&'static str>` of leaked codes → check before leaking. (Note: this is the ONE allowed use of HashSet — it's a process-internal interner that never affects output. Document loudly.)
- Or switch `ValidationFinding::code` from `&'static str` to a small interned-string type.
- Document `parse_locke_report_json` as not for hot-path use.

**Effort:** SMALL. **Risks determinism:** NO (the leak doesn't affect output).

### B3.3 — `\uXXXX` surrogate-pair mishandling

**File:** `crates/cjc-locke/src/json_emit.rs` lines 589-599
**Severity:** HIGH (correctness — human-edited JSON with emoji rejected)

The `\u` escape handler reads 4 hex digits and calls `char::from_u32(cp)`. Surrogate pairs (`😀`) fail to parse. Emit side never produces these so emit-then-parse is fine — but a *human-edited* Locke JSON with valid Unicode escapes for codepoints above U+FFFF errors out.

**Fix sketch:**
- On seeing a high surrogate (0xD800-0xDBFF), peek for `\u`, parse the next 4 hex, validate low surrogate range (0xDC00-0xDFFF), combine via `0x10000 + ((hi & 0x3FF) << 10) | (lo & 0x3FF)`.
- Or document loudly that the parser is round-trip-only and reject `\u` escapes that emit doesn't produce.

**Effort:** SMALL. **Risks determinism:** NO.

### B3.4 — `emit_locke_report_json` death by 1000 `format!()` allocations

**File:** `crates/cjc-locke/src/json_emit.rs` throughout (lines 26-46 has no preallocation; lines 130-133, 154, 175, 209, 215, 222 etc. use `format!()` + `push_str`)
**Severity:** MEDIUM-HIGH (10-20MB transient allocations on 10K-finding reports)

**Fix sketch:**
- Preallocate `String::with_capacity(findings.len() * 200)`.
- Replace `format!()` calls with `write!()` into the String via `std::fmt::Write` (infallible for String).
- For integers, use hand-rolled int→bytes if needed (still zero-dep).
- The `{:?}` on f64 is the unavoidable allocation point.

**Effort:** MEDIUM. **Risks determinism:** NO — same bytes, just faster.

### B3.5 — Float round-trip via `{:?}` is fragile

**File:** `crates/cjc-locke/src/json_emit.rs` lines 96-105 (write_float) + 630-654 (parse_float)
**Severity:** CRITICAL (foundation of byte-identical reporting)

Emit uses `format!("{:?}", x)`. Rust's f64 Debug is not formally byte-stable across toolchain versions. A compiler bump could silently break the byte-identical claim.

**Fix sketch:**
- Hand-roll a canonical f64 → shortest-round-trip printer. Pick a single canonical form (e.g. always "x.y" below magnitude N, always "xeY" scientific with fixed mantissa above).
- Document the canonical form at top of `json_emit.rs`.
- Add a regression test with a battery of known-tricky floats (denormals, 1e-30, 1e30, exact powers of 2) asserting the bytes match a frozen golden string.

**Effort:** MEDIUM-LARGE. **Risks determinism:** Tightens it. CRITICAL severity but more invasive — consider deferring to its own commit.

---

## Section 3 — Recommended Batch 4: Algorithmic complexity wins

### B4.1 — BPE training: incremental pair counting

**File:** `crates/cjc-locke/src/tokenizer.rs` lines 117-174 (`Tokenizer::train` main loop)
**Severity:** HIGH (10× perf win on text_drift)

Every merge iteration:
1. Allocates fresh `BTreeMap<(u32,u32), u64>` for counts.
2. Walks every byte of every sequence.
3. Calls `apply_merge` allocating a new `Vec<u32>` per sequence.

For a 10MB corpus and 1024-token vocab: ~10 GB of scanning. Plus the lex-tie-break clones two vocab entries per pair (line 137-146).

**Fix sketch:**
- Maintain an incremental `BTreeMap<(u32,u32), u64>` pair-count structure seeded once.
- On each merge of (L,R): walk sequences. At every position where L,R becomes merged_id, decrement counts for broken pairs (prev,L), (L,R), (R,next) and increment for new pairs (prev, merged), (merged, next).
- Reuse a single `Vec<u32>` per sequence and mutate in place.
- Avoid the vocab clone in lex tie-break by tracking `(first_byte_of_left, first_byte_of_right)` or by sorting after filtering to the top-tied set only.

**Effort:** MEDIUM. **Risks determinism:** NO — incremental is provably equivalent. **Existing tests cover correctness via specific vocab assertions** — they'd all still pass with the incremental implementation.

### B4.2 — Lineage `is_acyclic` / `ancestors` is O(V·E)

**File:** `crates/cjc-locke/src/lineage.rs` lines 217-254 (`is_acyclic`), 257-269 (`ancestors`), 358-377 (`would_introduce_cycle`)
**Severity:** HIGH (quadratic-time on long pipelines)

Inside `while let Some(n) = queue.pop()`, the inner `for e in &self.edges` scans every edge for every node popped — O(V*E) instead of textbook Kahn's O(V+E). Same in `ancestors`. For a 10K-node lineage with 10K edges = 10^8 ops per call.

**Fix sketch:**
- Materialise adjacency once: `BTreeMap<FingerprintId, Vec<(FingerprintId, &str)>>` computed from `self.edges` at start of each traversal.
- Iterate `succ` in O(deg) per pop.
- Cache reverse adjacency for `ancestors`.
- Maintain a `visited: BTreeSet<FingerprintId>` separately from `out` — check before pushing to stack, not just before inserting (fixes Finding I/O #24 simultaneously).

For `add_idea` (line 332-336), maintain a `parents_closure: BTreeMap<FingerprintId, BTreeSet<FingerprintId>>` in the builder — each insert updates it incrementally. Cycle check becomes O(log V).

**Effort:** SMALL for traversal fix, MEDIUM for incremental closure. **Risks determinism:** NO — sorted BTreeMap iteration preserves order.

### B4.3 — Confounder detection O(C³)

**File:** `crates/cjc-locke/src/causal.rs` lines 471-537
**Severity:** HIGH on wide datasets (>100 columns)

Inner loop clones `String` per BTreeMap lookup (lines 503-505). Pairwise map at line 477-486 doubles every correlation (both (a,b) and (b,a)) — 2× memory.

**Fix sketch:**
- Build pairwise as `BTreeMap<(&str, &str), f64>` borrowing strings.
- Don't double-store both directions; look up symmetrically.
- Cache `r_with_target.keys().collect()` once outside the inner loop.

**Effort:** MEDIUM. **Risks determinism:** NO.

### B4.4 — `numeric_ks_finding` double sort

**File:** `crates/cjc-locke/src/drift.rs` lines 348-407 + 295-337 (`wasserstein_1`)
**Severity:** MEDIUM (2× sort cost on numeric drift)

`numeric_ks_finding` calls `ks_d_statistic` (sorts both arrays) then immediately `wasserstein_1` (sorts both arrays again). Pre-sort once, pass sorted slices to both.

**Fix sketch:**
- Introduce `sort_filter_nan(&[f64]) -> Vec<f64>` and a `(ks_d, w1) = ks_and_wasserstein(a_sorted, b_sorted)` accepting pre-sorted slices.
- Both can be public `stats.rs` primitives.

**Effort:** SMALL. **Risks determinism:** NO (modulo the tied-cluster fix already shipped).

---

## Section 4 — Recommended Batch 5: HIGH-severity correctness

### B5.1 — Lineage binary-op symmetry: `join(L,R)` and `join(R,L)` produce same ID

**File:** `crates/cjc-locke/src/lineage.rs` line 117 (`LockeIdea::new` sorts parents); `crates/cjc-locke/src/traced.rs` line 261 (`join`) and 334 (`concat`)
**Severity:** HIGH (determinism leak — semantically distinct ops with same ID)

`LockeIdea::new` sorts parents before fingerprint. The comment says "if the caller cares about parent order (e.g. binary join), include it in transform.params" — but `traced.rs::join` doesn't. So `L LEFT JOIN R` and `R LEFT JOIN L` produce the SAME idea ID.

**Fix sketch:**
- In `traced.rs::join` and `concat`, add to params BTreeMap BEFORE `LockeIdea::new`:
  ```
  "parent_a": format!("{}", parent_a_id),
  "parent_b": format!("{}", parent_b_id),
  ```
- Param-encoded ordering becomes the source of truth; parents-sorted-in-id stays safe for n-ary commutative ops.

**Effort:** SMALL. **Risks determinism:** YES — this fixes a leak. ID changes only for binary-op cases that were previously colliding (the bug case).

**Regression test:**
```
let id1 = join(L, R, "key").id;
let id2 = join(R, L, "key").id;
assert_ne!(id1, id2);
```

### B5.2 — `AuditEvent::new` accepts caller-supplied seq with no monotonic safety check

**File:** `crates/cjc-locke/src/lineage.rs` lines 173-194
**Severity:** HIGH (external callers can build malformed audit chains)

Public API. A consumer outside `cjc-locke` could pass a stale seq. Determinism is broken silently.

**Fix sketch:**
- Make `AuditEvent::new` `pub(crate)` and route external callers through `LineageBuilder`. OR
- Keep pub but add a debug-assert and explicit doc-comment "monotonicity is the caller's responsibility."
- Add `LineageGraph::validate_audit_monotonic` for users to check chains before trusting them.

**Effort:** SMALL. **Risks determinism:** Fixes a leak.

### B5.3 — `relative_shift` false positive on near-zero-mean columns

**File:** `crates/cjc-locke/src/drift.rs` lines 153-156
**Severity:** MEDIUM (spurious E9030 firings)

`let denom = a.abs().max(1e-12);` floors denominator. For a feature with true mean ~1e-15, a "trivial" shift of 1e-12 reads as relative_shift ≈ 1.0 → E9030 fires at default `mean_shift_error = 0.30`.

**Fix sketch:**
- Use symmetric `|t - s| / (|t| + |s|).max(eps)`. OR
- Skip the relative-shift check when `|t| < threshold` and report absolute shift only.
- Expose `eps` via `DriftConfig`.

**Effort:** SMALL. **Risks determinism:** NO (only changes near-zero edge cases).

### B5.4 — `validate_and_compare` cascaded composition silently masks axis collision

**File:** `crates/cjc-locke/src/api.rs` lines 417-437
**Severity:** MEDIUM (documentation gap, but operational surprise)

The doc claims train's drift_score is always 1.0 — true for fresh validate, but if a caller passes a train belief that came from a prior `validate_and_compare`, `min(train.drift, drift_score) != drift_score`. The proptest only covers fresh train.

**Fix sketch:**
- Document the cascade semantics: meet-semilattice property means `drift_axis` floors monotonically.
- Add a proptest that exercises the cascaded case explicitly.
- Or add `validate_and_compare_with_train_belief(train_belief, drift_cfg)` that surfaces the choice.

**Effort:** SMALL. **Risks determinism:** NO.

### B5.5 — `relative_shift` already covered — see B5.3.

### B5.6 — Streaming `into_report` byte-identity claim doesn't hold above `sample_cap`

**File:** `crates/cjc-locke/src/streaming.rs` lines 268-417
**Severity:** HIGH (contract violation, documented as "byte-identical")

Module doc promises "byte-identical to a single-shot `validate(&full_df)`". Holds only when `sample_cap >= n_rows`. Once cap is hit, subsequent values dropped from sample but still counted in Welford/ECDF. `into_report` then validates only the sample.

**Fix sketch:**
- Detect cap breach and emit a new code (`E9XXX: streaming.sample_truncated` Notice) in the report so users see the divergence.
- Or document the doc on line 7 as "byte-identical only when no chunk caused sample truncation" and add tests for the truncated path.

**Effort:** SMALL. **Risks determinism:** NO.

---

## Section 5 — Recommended Batch 6: Memory + scale (long tail)

These are smaller wins but compound. Listed in rough impact order.

### B6.1 — `text_drift::samples_from_freqs` materialises freq-map back to Vec<f64>
**File:** `text_drift.rs` lines 138-145
**Effort:** MEDIUM (introduce `ks_d_statistic_from_counts` over `BTreeMap<u64, u64>` directly)
**Severity:** HIGH (1M-token document → 8MB Vec per side)

### B6.2 — `text_drift::char_3gram_freqs` allocates Vec<char>
**File:** `text_drift.rs` lines 169-179
**Effort:** SMALL (ring buffer of 3 chars over `chars()`)
**Severity:** MEDIUM

### B6.3 — `text_drift` trains tokenizer twice when both E9110 and E9111 fire
**File:** `text_drift.rs` lines 217-219 + 286-287 + `detect_text_drift` lines 442-455
**Effort:** SMALL (refactor to train once per column)
**Severity:** MEDIUM

### B6.4 — `concat_str_values` UTF-8 lossy decode per row on CategoricalAdaptive
**File:** `text_drift.rs` lines 100-122
**Effort:** SMALL (single String with `with_capacity`, push bytes directly via `str::from_utf8`)
**Severity:** MEDIUM

### B6.5 — `categorical.rs::category_counts` recomputed 8× per categorical column
**File:** `categorical.rs` line 119-148 + `detect_all_categorical_quality` 1287-1304
**Effort:** MEDIUM (single shared per-column counts cache threaded to all 8 detectors)
**Severity:** MEDIUM (8× CPU on wide string DataFrames)

### B6.6 — `leakage.rs::format_level` allocates String per (row, col)
**File:** `leakage.rs` lines 593-604
**Effort:** MEDIUM (`enum LevelKey { Int(i64), Str(&'a str), Code(u32), ... }`; typed BTreeMap; format-on-emit)
**Severity:** HIGH (10M+ allocations per `validate` call on diabetes-130 scale)

### B6.7 — Multi-class AUC computed twice (max + per-class evidence)
**File:** `leakage.rs` lines 240-274
**Effort:** SMALL (return `(max_auc, Vec<(class, auc)>)` from the inner pass)
**Severity:** MEDIUM (2× slowdown on multi-class targets)

### B6.8 — Bounded edit-distance has no diagonal band (Ukkonen)
**File:** `categorical.rs` lines 552-585
**Effort:** MEDIUM
**Severity:** MEDIUM (5-10× win when threshold is small)

### B6.9 — Near-duplicate detector O(N²) without length bucketing
**File:** `categorical.rs` lines 661-683
**Effort:** MEDIUM (bucket by char count, pair with `[l-threshold..l+threshold]`)
**Severity:** MEDIUM

### B6.10 — `apply_policy` clones every kept finding into `remaining_findings`
**File:** `policy.rs` lines 470-548
**Effort:** MEDIUM (use `Vec<usize>` indices internally; clone only at `PolicyResult` construction)
**Severity:** MEDIUM at 10K-finding scale

### B6.11 — `compose_many` allocates O(N) intermediate BeliefScores
**File:** `algebra.rs` lines 244-253
**Effort:** MEDIUM (single-pass for Min/Max/GeometricMean; requires byte-identity proptest)
**Severity:** MEDIUM at per-leaf aggregation scale

### B6.12 — PII detectors clone full PII strings into sample BTreeMaps
**File:** `pii.rs` lines 270-291
**Effort:** SMALL (cap each samples map at 32 entries via `if samples.len() < 32`)
**Severity:** LOW (perf) but **MEDIUM (secrets-in-memory)** for SSN

---

## Section 6 — MEDIUM/LOW polish (do last, batch together)

These are all SMALL effort with no determinism risk. Group into one polish commit.

| # | File | Line | Issue |
|---|---|---|---|
| P1 | `temporal.rs` | 37 | Delete dead `TimeColumnConfig.unit_is_millis` field |
| P2 | `categorical.rs` | 311-403 | `to_lowercase()` recomputed per row — memoise outside filter |
| P3 | `categorical.rs` | 799-805, 875-925 | `Vec<char>` collection for two-char windows — use `chars().peekable()` |
| P4 | `algebra.rs` | 226-237 | `compose()` redundant clamps — add `apply_unchecked` for internal use |
| P5 | `belief.rs` | 347-360 | `penalty_from_findings_with_model` > 1.0 silent — clamp to `[0, 1]` |
| P6 | `policy.rs` | 259-265 | `SuppressionRule::pattern("")` yields `Exact("")` — document or reject |
| P7 | `policy.rs` | 470-502 | Document owner first-match-wins (mirror suppression-side test) |
| P8 | `validation.rs` | 905-918 | `NumericRange` on Int uses lossy `as f64` cast — add `IntRange` variant or guard |
| P9 | `validation.rs` | 1391-1417 | `detect_imbalanced_target` Int constant-target silent — emit Info note |
| P10 | `validation.rs` | 597-624 | `detect_duplicates_full_row` sample apportioning across groups |
| P11 | `temporal.rs` | 191 | `saturating_sub` in gap detection masks unsorted runs |
| P12 | `shape.rs` | 89-100 | Numerically-constant column skips both E9010 + E9024 — add `near-zero variance` finding |
| P13 | `pii.rs` | 111-139 | `looks_like_phone` boundary cases not tested — add proptest |
| P14 | `drift.rs` | 119-129 | PSI `eps` clamp doesn't scale with `n_bins` |
| P15 | `drift.rs` | 432-447 | Histogram bin off-by-one risk at upper edge |
| P16 | `drift.rs` | 508-509 | TVD merge-walk instead of sort+dedup |
| P17 | `streaming.rs` | 188-194 | Schema check requires column order — doc disagrees |
| P18 | `streaming.rs` | 246-253 | Bool reconstruction loses ordering + 32-bit truncation |
| P19 | `streaming.rs` | 462-468 | Welford min/max vs ECDF disagree on signed zeros |
| P20 | `tokenizer.rs` | 158-165 | "Structural impossibility" silent break — add `debug_assert!` |
| P21 | `gate.rs` | 192-228 | Box `PolicyResult` to shrink no-policy `ReportDiff` (8 bytes vs 96+) |
| P22 | `causal.rs` | 408-455 | `audit_correlations` dedupes by id but pushes duplicates first |
| P23 | `causal.rs` | 304-306 | `CausalDag.relates` calls `is_reachable` twice |
| P24 | `causal.rs` | 279-299 | `is_reachable("a", "a")` returns true — document or guard |
| P25 | `json_emit.rs` | 335-345 | Trailing comma in array gives confusing error message |
| P26 | `dispatch.rs` | 72-101 | Linear string match — switch to binary-search over sorted `&[(&str, fn)]` |
| P27 | `report.rs` | 158-183 | `derive_id` excludes `assumptions` — document or include |

---

## Section 7 — Out of scope / future work

These are LARGE-effort items the audit surfaced but explicitly deferred. Document them as "known limitations" rather than fix.

1. **128-bit `FingerprintId`** — birthday-collision risk at ~2^32 distinct entities (~4 billion). For multi-year audit aggregation. (`id.rs:16`, audit finding I/O #16)
2. **JSON streaming emit** — full streaming `Write` impl for `LockeReport` to remove the in-memory String for huge reports. (`json_emit.rs` overall)
3. **Streaming dup-tracking 128-bit fingerprint mode** — `BTreeMap<u128, u64>` keyed by a deterministic 128-bit fingerprint. (`streaming.rs` lines 530-562, audit finding D-P #14)
4. **TracedDataFrame opt-out** — `.untraced()` method for hot-path users who don't want full lineage. (`traced.rs` lines 73-96)
5. **Welford → Kahan replacement** — eliminate Welford drift at N > 1e8. (`streaming.rs` line 469, audit finding D-P #19)
6. **Lossy-UTF-8 detection** — `category_counts` for CategoricalAdaptive silently collapses non-UTF-8 entries via `from_utf8_lossy`. Surface a new code (e.g. E9009). (`categorical.rs` line 137-145, audit finding V #3)
7. **CategoricalAdaptive duplicate sample tracking** — sample row indices in E9003 for adaptive-categorical columns now that the duplicate-detection bug is fixed. Verify sample evidence is correct.

---

## Section 8 — Suggested next-session order

If you have **one session** (~4-6 hours of focused work):
1. **Batch 3 first** (B3.1-B3.5) — JSON parser hardening. Five related items, one file. Highest correctness impact.
2. Then Batch 6 polish items P1-P10 — they're all SMALL and group naturally.

If you have **two sessions**:
1. Session A: Batch 3 + Batch 6 polish.
2. Session B: Batch 4 (algorithmic complexity) + Batch 5 (HIGH correctness).

If you have **three sessions**:
1. Session A: Batch 3.
2. Session B: Batch 4 (BPE incremental — the biggest perf lever).
3. Session C: Batch 5 + remaining polish.

**Always after each batch:**
```bash
cargo test -p cjc-locke --lib --release    # expect 383+
cargo test --test locke --release          # expect 266+ (each batch adds tests)
cargo test --test abng --release           # expect 629/629 unmoved
cargo test -p cjc-cli --lib --release      # expect 181+
```

If any number drops, ROLL BACK that fix before continuing. The deep-dive principle is: each fix is byte-identical or comes with explicit ID-change documentation.

---

## Section 9 — Where each finding lives in the audit transcripts

The full 87 findings (with severity, description, fix sketch, effort, determinism risk, test coverage) live in four agent transcripts. If you need more detail than this handoff covers, the originals are in the session memory under:

- **Cluster A (validators, 22 findings)** — agent `a29e90387e6451e0c`
- **Cluster B (data-processing, 22 findings)** — agent `a75f856586b4fa7ed`
- **Cluster C (algebra+policy, 18 findings)** — agent `a3be1c8baedee4024`
- **Cluster D (I/O+lineage+causal, 25 findings)** — agent `a54761e6390973454`

This handoff distills them, but each agent's full output has more nuance per finding.

---

## Closing note

The deep dive confirmed Locke's **determinism foundation is solid**: no `HashMap`/`HashSet`/`Instant`/`partial_cmp().unwrap()` anywhere; `id.rs` uses SplitMix64 with fixed domain salts and full test coverage. All remaining work is correctness-tightening + perf + scale, not "fix the determinism contract."

The shipped fixes already retroactively made every previous v0.7+ work *correct* on adaptive-categorical data (which is what diabetes-130 and ABNG use by default) and *textbook-conformant* on KS-D over tied data (which is what every integer column is). The remaining items continue that pattern.

Test deltas locked in: cjc-locke --lib 383 / tests/locke 266 / tests/abng 629/629 / cjc-cli --lib 181. Don't let them slip.

Good luck.
