---
title: "ADR-0017: Adaptive TidyView Selection"
tags: [adr, data, tidyview, performance, determinism]
status: Accepted
date: 2026-04-26
---

# ADR-0017: Adaptive TidyView Selection

## Status

Accepted

## Context

Until v0.1.6, every `TidyView` carried its visible-row set as a single
`BitMask` — a `Vec<u64>` of bit-packed flags, one bit per row of the base
`DataFrame`. The bitmask is excellent at the dense end (a 30%-pass filter
costs `nrows / 64` words) but degrades at the sparse end: the chess-RL
profiling work and several internal benchmarks showed that filters with
≪1% match rate paid the same `O(nrows / 64)` traversal cost as filters
with 50% match rate, because every consumer (`iter_set`, joins,
`group_by`, materialize) walked the full bitmap.

Two specific sites motivated the change:

- **Joins after a sparse filter.** `inner_join` calls
  `selection.iter_indices()` on both sides; on a 1M-row left table with 9
  surviving rows we were word-walking 15,625 u64s to emit 9 indices.
- **Empty / All-pass filters.** Both terminals walked the full bitmap and
  allocated index buffers identical in shape to the dense case.

The fix had to honor the existing determinism contract (ascending order,
`BTreeMap`-only, no FMA, no rayon) and could not change `TidyView`'s
public API or `DataFrame::tidy()` entry shape. Anything that materializes
indices for a downstream consumer had to remain bit-identical to the old
bitmask path.

## Decision

### 1. Five-arm `AdaptiveSelection` enum, density-classified at construction

```rust
pub enum AdaptiveSelection {
    Empty { nrows: usize },
    All { nrows: usize },
    SelectionVector { rows: Vec<u32>, nrows: usize },
    VerbatimMask { mask: BitMask },
    Hybrid { mask: BitMask }, // reserved; behaves as VerbatimMask in v2.0
}
```

The classifier `from_predicate_result(words, nrows)` runs after every
predicate evaluation and picks the arm by integer count:

| Density `count` vs `nrows` | Arm |
|---|---|
| `count == 0` | `Empty` |
| `count == nrows` | `All` |
| `count < nrows / 1024` (sparse) | `SelectionVector` |
| `count * 10 > nrows * 3` (dense, > 30 %) | `VerbatimMask` |
| otherwise (mid) | `VerbatimMask` (v2.0); `Hybrid` reserved for v2.1 |

The dense check uses 128-bit intermediates so it cannot overflow on the
extreme end. Every threshold is integer arithmetic — there is no
floating-point density in the classifier and therefore no ordering
ambiguity.

### 2. Mode-invariant trait surface

Every consumer reaches the selection through one of:

```
len()  count()  contains(i)  iter_indices()
intersect(&other)  union(&other)
materialize_mask()  materialize_indices()
explain_selection_mode() -> &'static str
```

`iter_indices()` is the choke point — there are 20+ call sites in
`cjc-data` (joins, group_by, distinct, materialize, pivot, summarise,
mutate-with-mask) and **all** of them migrate by replacing
`self.mask.iter_set()` with `self.mask.iter_indices()`. The new iterator
is enum-dispatched; the `SelectionVector` arm drops straight into the
backing `Vec<u32>` with zero per-row branching, while the
`VerbatimMask` / `All` arms reuse the existing `BitMask::iter_set` path
verbatim.

### 3. Hybrid is reserved, not implemented

The brief calls for a chunked `Hybrid` representation that defers
materialization. v2.0 ships the *variant* in the enum so callers do not
need re-patching when the chunked path lands; concretely it currently
matches `VerbatimMask` semantics. That decision keeps the v2.0 diff
minimal and lets us prove the four-arm correctness story before adding a
fifth runtime path.

### 4. No public-API break

`TidyView::mask()` previously returned `&BitMask`. To keep the chain-call
surface (`view.mask().iter_set()`, `view.mask().count_ones()`) working
unchanged in user code and existing tests, `mask()` now returns *owned*
`BitMask` materialized via `materialize_mask()`. New code can reach for
`view.selection() -> &AdaptiveSelection` to avoid materializing.
`explain_selection_mode()` is the public introspection hook used by tests
and the benchmark scaffold.

### 5. Determinism is preserved by construction

- Every arm yields strictly ascending row indices (proved by
  `iteration_order_is_strictly_ascending_for_every_arm`).
- `intersect` / `union` go through `materialize_mask` and re-classify, so
  set-op outputs satisfy the same ascending invariant and the
  cardinality identity `|A| + |B| = |A∪B| + |A∩B|` (fuzzed across
  arbitrary mode-mixed inputs).
- Repeating the same filter ten times produces the same arm, the same
  row vector, and the same materialized bitmask words (10× determinism
  test).
- No `HashMap`, no rayon, no FMA introduced anywhere on the path.

## Consequences

### Positive

- Sparse-filter consumers (joins, group_by) no longer pay
  `O(nrows / 64)` to enumerate `O(count)` rows.
- `Empty` and `All` short-circuit at the trait level — `count()` returns
  in O(1), `iter_indices()` yields zero rows or `0..nrows` respectively.
- The brief's "Hybrid / chunked / deferred" path has a reserved seat in
  the enum; v2.1 can land it without re-touching consumers.
- 4 modes are demonstrably reachable in CI via a smoke benchmark
  (`bench_density_crossover_smoke`); a 12-step ignored bench
  (`bench_density_crossover`) sweeps 0% → 100% density at N = 1M for
  manual perf validation.

### Negative

- One additional indirection (`enum AdaptiveSelection` vs raw
  `BitMask`) on every selection access. The cost is a single branch on
  a 5-variant enum and is dwarfed by the savings on sparse paths;
  `tidyview_benchmarks` shows zero regression on the dense suite.
- `mask()` materializes a `BitMask` on every call rather than returning
  a reference. Callers that need repeated bitmap access should bind it
  once or migrate to `selection()`.

## Alternatives considered

- **Keep `BitMask`, add a `RoaringBitmap` library.** Rejected: external
  dependency (HARD RULE), and roaring's container choice is opaque to
  the determinism contract — we want the classifier rule visible in
  source.
- **Selection-vector-only.** Rejected: regresses dense filters, where
  bitmask AND across chained filters is a pure word-level operation.
- **Defer to a chunked structure on day one.** Rejected as v2.0 scope:
  the four-arm version subsumes the practical sparse / dense crossover
  and ships behind a stable enum, leaving `Hybrid` available without
  re-patching consumers.

## Test surface

- 16 unit tests in `cjc-data::adaptive_selection` (classifier, count /
  contains / iter agreement, set-op identities, density edges).
- 6 integration tests in `tests/tidy_tests/test_adaptive_selection_integration.rs`
  (real `TidyView` filters land in the expected arm).
- 5 determinism tests in
  `tests/tidy_tests/test_adaptive_selection_determinism.rs` (10×
  repeats, cross-mode equivalence, ascending iteration).
- 3 adversarial join + group_by tests in
  `tests/tidy_tests/test_adaptive_selection_join_adversarial.rs`
  (sparse-filter joins, sparse-vs-chained equivalence, group_by after
  sparse filter).
- 3 bolero fuzz targets in
  `tests/bolero_fuzz/adaptive_selection_fuzz.rs` (round-trip,
  mode-invariance, set-op stability + cardinality identity).
- 2 bench scaffolds (`bench_density_crossover` ignored,
  `bench_density_crossover_smoke` always-runs).

## v2.1 Amendment (2026-04-26 — same release window)

The "Hybrid reserved" arm and the "predicate AST walk" path called out in
the original v2.0 work both became live in the same release. Updated
inventory:

### Hybrid arm activated

`Hybrid { nrows, chunks: Vec<HybridChunk> }` is no longer a placeholder.
It splits the row space into `HYBRID_CHUNK_SIZE = 4096`-row blocks; each
block is independently classified as `Empty` / `All` / `Sparse(Vec<u16>)`
(when the block has ≤ 128 hits, sparse-storage threshold = chunk_size/32)
or `Dense(Box<[u64]>)` (verbatim 64-word bitmap of the block).

Activation rule: a predicate result enters the Hybrid arm only when
`nrows >= 2 * HYBRID_CHUNK_SIZE` and the global density falls in the
mid band (between the sparse cutoff `nrows/1024` and the dense cutoff
`>30%`). Below 8192 rows the per-chunk overhead beats the win, so we
fall back to `VerbatimMask`.

### Merge-walk fast paths in `intersect`/`union`

- `Sparse ∩ Sparse` → O(|A|+|B|) merge walk over the two `Vec<u32>`
  buffers (no materialization to bitmask).
- `Sparse ∩ VerbatimMask` → filter-walk: iterate the sparse side and
  test each index against the dense bitmap.
- `Sparse ∪ Sparse` → O(|A|+|B|) sorted-merge with re-classification at
  the end (result may exit the sparse band, in which case it materializes
  to `VerbatimMask` or upgrades to `Hybrid`).

### Predicate bytecode (new module)

`crates/cjc-data/src/predicate_bytecode.rs` (≈340 LOC) replaces the
recursive AST-walk in `try_eval_predicate_columnar` as the primary
columnar fast path. Three opcodes:

```rust
pub enum PredicateOp {
    Cmp { kind: LeafKind, op: CmpKind },
    And,
    Or,
}
```

Each `Cmp` carries a fully-resolved `(col_idx, literal, op)` triple — the
column lookup and i64↔f64 promotion are decided at lowering time, so
interpretation is a flat loop with no recursion or symbol-table cost.
The semantics are bit-identical to the old AST-walk because every leaf
delegates to the unchanged `columnar_cmp_f64` / `columnar_cmp_i64`
kernels and the per-leaf ANDs in the recursive form fold into a single
final AND-with-`existing_mask` (algebraically `((a | b) & c) & m == (a |
b) & (c & m)`).

The legacy AST-walk path is retained as a no-cost-on-the-hot-path
oracle: bytecode lowers first; only an unsupported shape falls through
to the AST walk, then to row-wise eval.

### v2.1 test inventory delta

- **+10 unit tests** in `adaptive_selection.rs` (Hybrid construction,
  per-chunk classifier, merge-walk intersect/union, sparse-band
  reclassification after union, chunk-by-chunk iter ordering).
- **+11 unit tests** in `predicate_bytecode.rs` (postorder lowering,
  reversed-literal flip, AND/OR stack semantics, NaN/promotion
  invariants).
- **+23 integration parity tests** in
  `tests/tidy_tests/test_v2_1_bytecode_parity.rs` — bytecode-driven
  `filter()` vs scalar Rust loop across six relations × two column
  types × reversed × mixed × compound × density mix.
- **+3 bolero fuzz targets** in
  `tests/bolero_fuzz/v2_1_bytecode_fuzz.rs` (bytecode-vs-scalar oracle,
  determinism across runs, `|A|+|B|=|A∩B|+|A∪B|` cardinality identity).
- Smoke bench (`bench_density_crossover_smoke`) updated to exercise the
  new mid-band `Hybrid` path explicitly.

### v2.1 regression gate

```
cjc-data        155/155
tidy            358/358 (was 335 in v2.0; +23 from new parity file)
bolero fuzz      23/23  (was 20; +3 from new bytecode targets)
tidyview-bench   38/38
parity_stress    11/11
builtin_parity   10/10
physics_ml       71/71  (2 ignored long-converge)
cjc-runtime     707/707
```

### v2.1 ignored bench (1M-row crossover)

```
threshold        hits      mode             filter_us  iter_us
0                0         Empty            ~2.5k      4
1                1         SelectionVector  ~2.3k      0
100              100       SelectionVector  ~2.1k      2
1_000            1_000     Hybrid           ~2.1k      9
100_000          100_000   Hybrid           ~2.2k      519
300_000          300_000   Hybrid           ~2.6k      2_467
500_000          500_000   VerbatimMask     ~2.6k      2_761
1_000_000        1_000_000 All              ~3.6k      7_304
```

Filter cost is flat (bytecode adds no measurable overhead vs the AST
walk; both are dominated by the column scan). The Hybrid arm preserves
the O(count) iteration cost of `SelectionVector` into the mid band, so
mid-density consumers (typical for joins after a non-trivial WHERE) no
longer pay full `O(nrows/64)` for what is effectively an
O(count)-bounded result.

## v2.2 amendment (2026-04-27): sparse-gather chained filters + group_by allocation cut

v2.1 unified two filter paths (bytecode for column predicates, AST walk
fallback) but treated every call the same way: build a leaf mask by
sequential column scan, AND it with the existing selection. That cost
was *constant in the column scan*, even when the parent selection had
already narrowed to a handful of rows.

v2.2 closes that gap on the eager (non-lazy) path.

### Sparse-gather predicate path

`PredicateBytecode::interpret_sparse(base, existing_indices, nrows)` is
a second interpretation strategy that, instead of scanning the whole
column buffer, gathers only the values at `existing_indices` and runs
the same VM scalar-wise. The threshold is integer-only, matching the
v2 determinism contract:

```rust
pub fn should_use_sparse_path(count: usize, nrows: usize) -> bool {
    count.saturating_mul(4) < nrows  // i.e. count/nrows < 25%
}
```

Below 25% retention, `TidyView::filter` calls `interpret_sparse` over
the existing index iterator; above it, the v2.1 dense column-scan path.
AND/OR are *monotone* (output ⊆ input), so the sparse path doesn't need
a final AND with the existing mask — the result is bounded to
`existing_indices` by construction. This makes the proof trivial:
`sparse(existing) = dense() ∩ existing` for column predicates.

A small but load-bearing refactor moved
`let current_mask = self.mask.materialize_mask()` *inside* the dense
branch (and the AST-walk fallback). v2.1 paid this cost unconditionally
— ~125 KB allocation for a 1M-row bitmap — even when the sparse path
took over. Bench delta on a 1M-row two-step chain (`x<1` then
`x<0`):

| p1 hits | step2 v2.1 (μs) | step2 v2.2 (μs) | speedup |
|---|---|---|---|
| 1 | ~1057 | 23 | ~46× |
| 10 | ~993 | 25 | ~40× |
| 100 | ~1790 | 76 | ~24× |
| 1 000 | ~1064 | 99 | ~11× |
| 10 000 | ~1301 | 205 | ~6× |
| 100 000 | ~2818 | 966 | ~3× |

The dense branch (≥25% retention) is unchanged — same column scan.

### group_by allocation cut

`GroupIndex::build_fast` previously took `&[usize]`, forcing
`TidyView::group_by` to materialise the entire selection iterator into
a `Vec<usize>` (8 MB for a 1M-row visible set) before grouping. v2.2
generalises the signature to `impl IntoIterator<Item = usize>` and
passes `self.mask.iter_indices()` directly. Group order semantics are
unchanged — first-occurrence on ascending visible-row scan is exactly
what the iterator already yields.

### v2.2 deferred

- **Lazy-plan filter reordering** (was tentatively scoped) is now
  documented as a no-op today: `lazy::optimize` already runs
  `merge_filters` first, which collapses any chain of consecutive
  `Filter` nodes into a single AND-predicate. After merging there is
  exactly one filter — nothing left to reorder. For the eager path,
  the sparse-gather *is* the runtime selectivity adaptation; user
  filter order ceases to matter once parent retention crosses the 25%
  threshold. A future reorder pass would only have signal if some
  future plan node blocks merging in a way that current Select / Arrange
  pushdown does not — none currently do.

### v2.2 testing & regression

- **+5 chain-parity tests** in
  `tests/tidy_tests/test_v2_2_sparse_chain_parity.rs` pinning the
  triangle: chained `df.filter(p1).filter(p2)` ≡ single
  `df.filter(p1 AND p2)` ≡ row-by-row scalar reference. Cases include
  pure-sparse chains, OR predicates on sparse parents, three-step
  chains, empty intermediates, and dense→sparse transitions.
- **+4 unit tests** in `predicate_bytecode.rs`:
  `sparse_path_density_threshold` (boundary check at exactly 25%),
  `interpret_sparse_simple_lt`, `interpret_sparse_matches_dense_on_same_inputs`,
  `interpret_sparse_or_monotone`. Total predicate_bytecode unit tests:
  15/15.
- **+1 design-validation bench**
  (`bench_sparse_chain`, ignored) sweeping p1 hits over
  `[1, 10, 100, 1000, 10_000, 100_000]` on 1M rows. Numbers above are
  from this bench.

```
cjc-data        159/159
tidy            363/363 (+5 v2.2 sparse-chain parity)
bolero fuzz      23/23
tidyview-bench   38/38
physics_ml       71/71  (2 ignored long-converge)
```

Weight hashes / determinism: AND/OR sparse evaluation is stable under
fixed `existing_indices` order; `iter_indices()` already yields
ascending row numbers in all five `AdaptiveSelection` arms (Empty/All/
SelectionVector/VerbatimMask/Hybrid), so the gather is bit-deterministic.

## v3 Phase 3 — Hybrid streaming set-op fast paths (2026-04-28)

The v3 brief had three Phase-3 candidates: cat-aware joins, full
`Column` wiring, and chunked streaming set ops. The user authorized
**Phase 3 = chunked streaming first, Phase 4 = bundled (cat-aware joins
+ Column wiring) next**. This subsection records what shipped under
Phase 3.

### v3 Phase 3 problem

`AdaptiveSelection::intersect`/`union` previously routed any operand
combination involving `Hybrid` through `to_verbatim_mask()` →
`materialize-and-AND` → re-classify. For an 8M-row Hybrid that meant
allocating a `nrows/64` u64 buffer (~125 KB for 1M rows; ~1 MB for 8M)
twice — once per operand — even when only a few chunks intersected. The
chunk layout was discarded immediately after construction.

### v3 Phase 3 design

Per-chunk dispatch on the 4×4 `HybridChunk` shape table (Empty / All /
Sparse(Vec<u16>) / Dense(Box<[u64]>)), collapsing to **5 effective
shapes** after Empty/All folding:

| left × right | path |
|---|---|
| `Sparse ∩ Sparse` | merge-walk on two ascending `Vec<u16>` |
| `Sparse ∩ Dense`  | filter-walk: bit-test each sparse row against partner words |
| `Dense ∩ Dense`   | per-word AND over `chunk_len.div_ceil(64)` u64s |
| `Empty ∩ *`       | fold to `Empty` |
| `All ∩ x`         | fold to `x` |

Symmetric union table, plus density classification after each chunk so
the result chunk lands in the cheapest shape:

```rust
fn classify_sparse_chunk(rows: Vec<u16>, chunk_len: usize) -> HybridChunk {
    if rows.is_empty()              { Empty }
    else if rows.len() == chunk_len { All }
    else if rows.len() <= HYBRID_CHUNK_SPARSE_THRESHOLD { Sparse(rows) }
    else                            { Dense(rows_to_words(&rows, chunk_len)) }
}
```

`HYBRID_CHUNK_SPARSE_THRESHOLD = 128` (= `HYBRID_CHUNK_SIZE / 32`)
keeps the per-chunk decision aligned with the global v0.1.6 5–30%
density band.

### v3 Phase 3 — `simplify_hybrid` (no re-globalize)

`Self::simplify_hybrid(nrows, chunks)` collapses to `All`/`Empty` only
when *every* chunk agrees. A mid-band Hybrid never re-globalizes into a
single `VerbatimMask` — that would defeat the locality the chunk layout
was designed to preserve. Three chained intersects on a Hybrid stay
chunked; partner sparseness only shrinks the chunks.

### v3 Phase 3 — Hybrid × non-Hybrid

- **`Hybrid ∩ SelectionVector`**: walk the partner indices once,
  bucket by chunk, dispatch each chunk against the bucket. No
  `nrows/64`-word allocation.
- **`Hybrid ∩ VerbatimMask`**: per-chunk word-AND against the
  matching slice of the partner mask. Sparse chunks stay sparse;
  Dense chunks word-AND in place.
- **`Hybrid ∪ SelectionVector`**: scatter-by-chunk with a
  while-loop bucket cursor, then per-chunk union dispatch.
- **`Hybrid ∪ VerbatimMask`**: per-chunk word-OR.

### v3 Phase 3 — surface contract (production reach)

A finding worth pinning explicitly: `TidyView::filter` AND-collapses
inside `predicate_bytecode::interpret` /
`interpret_sparse` (which apply the final mask AND themselves and
return the merged result). `AdaptiveSelection::intersect` and
`union` are **not** on `filter()`'s hot path today — they are public
algebraic operations on the selection lattice, called from tests and
reserved for **Phase 4** consumers.

The first natural production consumer is **cat-aware joins**
(Phase 4): a left-mask × right-mask probe-and-build over join keys
will land directly on `Hybrid ∩ Hybrid`. Phase 3 is the building
block; Phase 4 is the wire-up.

### v3 Phase 3 — testing & regression

- **+15 unit tests** under `phase3_*` names in
  `crates/cjc-data/src/adaptive_selection.rs`: every chunk-shape
  combination, partial-final-chunk safety, simplify_hybrid
  collapse-vs-stay-chunked, density-classification helpers, three-way
  chained intersect.
- **+12 integration parity tests** in
  `tests/tidy_tests/test_v3_phase3_hybrid_streaming.rs`. The 100k-row
  fixture uses bit-pattern columns (stride 50 → 82 hits/chunk →
  Sparse-shaped chunks; stride 12 → 341 hits/chunk → Dense-shaped
  chunks) so all six shape combinations exercise through the public
  `.filter(...).selection()` API. `DBinOp::Mod` does not exist in the
  TidyView lazy-plan vocabulary, so periodic-pattern columns are
  precomputed at frame build time using only `==`/`<`/etc.
- **+3 bolero fuzz targets** in
  `tests/bolero_fuzz/hybrid_streaming_fuzz.rs` forcing
  `FUZZ_NROWS = 16_384` (above the 8192 Hybrid activation threshold);
  scalar oracle is BitMask AND/OR + bit-iteration, so any divergence
  between the chunked path and the materialize-and-AND fallback
  triggers immediately.
- **+1 design-validation bench** (`bench_phase3_hybrid_set_op`,
  ignored) — same-process comparison of the chunked path vs.
  pre-Phase-3 materialize-and-AND oracle on the favorable case
  (sparse-chunked × dense-chunked, 100k rows):

  ```
  Phase 3 chunked path:           4.69 μs avg (n=100)
  Pre-Phase-3 materialize path:  391.41 μs avg (n=100)
  Speedup:                        83.46×
  ```

  The 83× headline is the favorable case where allocation dominates
  the old path. Contiguous-Hybrid case (Dense × Dense everywhere)
  shrinks the win to ~5–10× — still net positive but no longer
  allocation-bound.

```
cjc-data         208/208 (+15 v3 Phase 3 chunk-dispatch unit tests)
test_phase10_tidy 402/402 (+12 v3 Phase 3 integration parity)
bolero fuzz       29/29  (+3 v3 Phase 3 hybrid-streaming fuzz)
tidyview-bench    38/38
physics_ml        71/71  (2 ignored long-converge)
```

Determinism: per-chunk paths only enumerate ascending row indices and
ascending word indices; `iter_indices()` over the result preserves the
v0.1.6 ascending invariant; bolero determinism target asserts byte-equal
output across two calls on the same selection pair.

## v3 Phase 4 — Cat-aware joins + Column ↔ CategoricalColumn wiring (2026-04-28)

User-approved sequencing finished here: Phase 3 shipped chunked
streaming; Phase 4 = bundled (cat-aware joins + Column wiring of
Phase 1's `CategoricalColumn`). Both deliverables shipped in the same
release window.

### v3 Phase 4(a) — Cat-aware joins

The string-key path in `inner_join` / `left_join` / `semi_join` /
`anti_join` materialized `Vec<String>` keys via `get_display(row)` per
left and per right row. For 100k × 100k joins on a 100-level
categorical that was 200 000 `String::clone()` calls plus 200 000
allocator hits — pure overhead, since the row identity was already a
`u32` code on both sides.

When every join-key column is `Column::Categorical` on **both** frames,
Phase 4 routes through a `BTreeMap<Vec<u32>, Vec<usize>>` probe with a
deterministic per-column remap:

```rust
right_to_left[ki][right_code] = Some(left_code) | None
```

`None` means the level doesn't exist on the left dictionary — that
right row can never join, so it's skipped before BTreeMap insertion.

The cross-frame dictionary mismatch is the whole reason this was
deferred from Phase 2: each DataFrame owns its own dictionary, so
left-side code 3 ≠ right-side code 3 in general. The remap is built
deterministically by walking the left levels into a `BTreeMap<&str,
u32>` (BTree, not Hash, to keep build order stable), then mapping each
right level through it. Codes ↔ levels are 1:1 within one frame, so
BTreeMap slot assignment, output row order, and join multiplicity are
byte-equal to the string path.

Mixed-type keys (e.g., one categorical + one int) cause
`collect_categorical_join_keys` to return `None`, falling back to the
string-key path — pinned by
`phase4_inner_join_mixed_keys_falls_back_to_string_path`.

### v3 Phase 4(b) — Column ↔ CategoricalColumn wiring (limited scope)

Phase 1 shipped `CategoricalColumn` (in `byte_dict.rs`) with adaptive
code widths (`AdaptiveCodes::U8/U16/U32/U64` auto-promoting at 256 /
65 536 / 2³² thresholds), shared/frozen dictionaries with
`UnknownCategoryPolicy`, and `CategoryOrdering::FirstSeen|Lexical|
Explicit`. Phase 4 wires this into the DataFrame surface as a pair of
**lossless conversions** rather than a wholesale `Column::Categorical`
replacement (the full replacement would touch every column reader —
hundreds of sites — and is deferred to a future phase):

| Direction | Method | Notes |
|---|---|---|
| `Column → CategoricalColumn` | `Column::to_categorical_column(&self)` | Uses `CategoryOrdering::Explicit` to pin the level→code map exactly. |
| `CategoricalColumn → Column` | `Column::from_categorical_column(cc)` | Returns `None` if any level is non-UTF-8 (byte-dict is byte-keyed; `Column::Categorical` is `String`-keyed) or if the column has nulls (`Column::Categorical` has no null bitmap). |

Round-trip is byte-equal:
`from_categorical_column(to_categorical_column(c))` produces the same
`levels`/`codes` for any `Column::Categorical c`. Pinned by
`phase4_column_to_categorical_column_roundtrip`.

### v3 Phase 4 — surface contract (production reach)

Cat-aware joins are now **on the production hot path**: every
`TidyView::inner_join` / `left_join` / `semi_join` / `anti_join` call
checks for the all-categorical-keys shape and routes through the fast
path automatically. No opt-in. No language-level surface change.

Cat-aware joins are also Phase 3's first natural production consumer
of `AdaptiveSelection` set ops — `iter_indices()` over the visible
mask is what feeds the BTreeMap build, so a Hybrid left view with
cat keys flows: `Hybrid mask → ascending row indices → cat-aware
probe`.

### v3 Phase 4 — testing & regression

- **+7 unit tests** in `crates/cjc-data/src/lib.rs`:
  `phase4_collect_cat_keys_returns_some_when_all_categorical`,
  `phase4_collect_cat_keys_returns_none_on_mixed_types`,
  `phase4_collect_cat_keys_unknown_right_level_yields_none_in_remap`,
  `phase4_column_to_categorical_column_roundtrip`,
  `phase4_column_to_categorical_column_none_for_non_categorical`,
  `phase4_column_from_categorical_column_rejects_nulls`,
  `phase4_column_from_categorical_column_rejects_non_utf8`.
- **+13 integration parity tests** in
  `tests/tidy_tests/test_v3_phase4_categorical_joins.rs` — every
  join shape (inner/left/semi/anti) under the shadow-frame technique
  (parallel `Column::Str` and `Column::Categorical` DataFrames must
  produce byte-equal output), single + composite keys, disjoint
  dictionaries, mixed-type fallback, pre-filtered visible masks,
  empty-side edge cases, two-call determinism.
- **+3 bolero fuzz targets** in
  `tests/bolero_fuzz/categorical_join_fuzz.rs` over inner / left /
  semi+anti — random `(level, value)` pair streams; assertion is
  byte-equal output rows vs the string-key oracle.
- **+1 same-process bench** (`bench_phase4_categorical_inner_join`,
  ignored): 100k × 100k rows, 100 unique levels, single-key inner
  join. String-key path **16.82 s** vs cat-aware path **2.77 s** = **6.08×
  speedup** (5-run average).

```
cjc-data         215/215 (+7 v3 Phase 4 unit)
test_phase10_tidy 415/415 (+13 v3 Phase 4 integration parity)
bolero fuzz       32/32  (+3 v3 Phase 4 cat-join fuzz)
tidyview-bench    38/38
physics_ml        71/71  (2 ignored long-converge)
```

Determinism: BTreeMap not HashMap on both the remap build and the
right-lookup probe; `iter_indices()` already yields ascending row
numbers in all five `AdaptiveSelection` arms; the cat-aware path is a
strict re-encoding of the string path's BTreeMap structure with
integer keys.

### v3 Phase 4 — what's deferred

The full replacement of `Column::Categorical { levels: Vec<String>,
codes: Vec<u32> }` with `byte_dict::CategoricalColumn` is **not** done.
Hundreds of column-reader call sites would migrate from
`&Vec<String>`/`&Vec<u32>` to `&ByteDictionary`/`&AdaptiveCodes` — a
much bigger refactor than Phase 4's scope. The Phase 4 conversion
methods are the gateway: callers that explicitly need adaptive code
widths or shared/frozen dictionaries can `to_categorical_column()`,
operate on the new type, and `from_categorical_column()` back.

## v3 Phase 5 — Full Column wiring + filter Hybrid path + cat-aware arrange (2026-04-28)

After Phase 4 the user authorized closing the deferred items: full
Column migration, filter-chain visibility for `AdaptiveSelection::
intersect`/`union`, and a new Phase 5 deliverable. Phase 5 ships
three things in one bundle.

### v3 Phase 5(a) — Full Column wiring (additive variant)

The literal replacement of `Column::Categorical { levels: Vec<String>,
codes: Vec<u32> }` with `byte_dict::CategoricalColumn` would touch
hundreds of pattern-match sites — genuinely a multi-session migration.
Phase 5 takes the **safe form**: an additive variant
`Column::CategoricalAdaptive(Box<CategoricalColumn>)` makes Phase 1's
adaptive engine a first-class column type without breaking the
existing surface.

**Bridging shim:** `Column::to_legacy_categorical(&self)` materializes
any `CategoricalAdaptive` into a `Column::Categorical` (lossless when
levels are UTF-8 and no nulls; falls back to `Column::Str` otherwise).
21 consumer sites — `filter_column`, `slice_column`, `gather_column`,
`gather_column_nullable`, `gather_column_nullable_null`, `column_value_str`,
`compare_column_rows`, `try_eval_expr_column_vectorized`, `eval_expr_row`,
`pivot_longer`, `df_drop_na`, `df_fill_na`, `format_describe`,
`column_to_value` (×2: tidy_dispatch + cjc-eval + cjc-mir-exec),
`dataframe_to_value` (cjc-eval + cjc-mir-exec), `ColumnStats::compute`,
`NullCol::from_column`, `push_row` — all converted via the shim or
inlined the per-row UTF-8-lossy fallback.

The fast paths (Phase 2 cat-aware group_by, Phase 4 cat-aware joins,
Phase 5(c) cat-aware arrange) bypass the shim entirely — they read
the legacy `Column::Categorical` directly. The shim is the back-compat
floor for code paths that haven't been adaptively-wired yet.

### v3 Phase 5(b) — Filter-chain Hybrid intersect path

`predicate_bytecode::evaluate_to_selection(base, nrows)` is new — it
returns an `AdaptiveSelection` from a fresh predicate evaluation
without ANDing into an existing mask. `interpret(base, existing_mask)`
now factors through this via a shared `evaluate_words` helper.

`TidyView::filter` routes through Phase 3's per-chunk dispatch when
the existing mask is `Hybrid` and the count is above the sparse
threshold:

```rust
if matches!(self.mask, AdaptiveSelection::Hybrid { .. })
    && !predicate_bytecode::should_use_sparse_path(count, nrows_base)
{
    let fresh = bc.evaluate_to_selection(&self.base, nrows_base);
    let intersected = self.mask.intersect(&fresh);
    return Ok(TidyView { mask: intersected, .. });
}
```

This is **Phase 3's first production wiring** beyond joins. Hybrid
left view + chunk-sparse predicate result intersect through the
per-chunk dispatch (5 effective shapes) instead of allocating a
`nrows/64`-word BitMask twice and AND-ing.

### v3 Phase 5(c) — Cat-aware arrange

`TidyView::arrange` is now O(N log N) **integer** comparisons for
`Column::Categorical` keys whose `levels` are lex-sorted (the Phase 17
`forcats` invariant). When `levels[code_a].cmp(&levels[code_b]) ==
code_a.cmp(&code_b)`, we sort by `u32` codes directly — no string
lookup, no bytewise compare per pair.

Pre-resolution per key: `levels_are_sorted(levels)` is checked once
before the sort. If true, the comparator becomes a pure `u32` compare;
if false, falls back to the legacy `compare_column_rows` (string
compare via level lookup). Mixed-type composite keys mix and match
inside the comparator on a per-key basis.

### v3 Phase 5 — testing & regression

- **+12 integration tests** in
  `tests/tidy_tests/test_v3_phase5_full_wiring.rs` covering all
  three deliverables: CategoricalAdaptive accessors / DataFrame
  rendering / round-trip / null fallback / join via legacy shim;
  filter chain on Hybrid existing visible-rows match a single-pred
  AND reference; arrange on sorted-levels-Categorical byte-equal to
  Str arrange; arrange on unsorted-levels falls back; descending and
  composite keys.
- **+1 same-process bench** `bench_phase5_arrange_cat_vs_string`
  (ignored, 5-run avg, 100k rows × 100 levels):

  ```
  Str:    92.73 ms
  Cat:     9.58 ms
  Speedup: 9.68×
  ```

  Marginally faster than Phase 4's join cat-aware win (6.08×) because
  arrange's hot path is more comparison-bound — the sort calls the
  comparator O(N log N) ≈ 1.6M times for 100k rows, every
  string-cmp eliminated counts.

```
cjc-data         215/215  (no new unit; the 7 Phase 4 unit tests cover)
test_phase10_tidy 427/427  (+12 v3 Phase 5 integration; 11 ignored benches now)
bolero fuzz       32/32   (no new fuzz; Phase 4 cat-join fuzz exercises adaptive path through legacy shim)
tidyview-bench    38/38
physics_ml        71/71   (2 ignored long-converge)
```

Workspace build: clean, including cjc-eval and cjc-mir-exec arms for
`Column::CategoricalAdaptive` in `dataframe_to_value` and the
`to_dataframe` builtin.

### v3 Phase 5 — surface contract (production reach)

| Verb | Production cat-aware reach | Mechanism |
|---|---|---|
| `group_by` / `distinct` | Phase 2 ✓ | `collect_categorical_keys` → `BTreeMap<Vec<u32>, _>` |
| `inner_join` / `left_join` / `semi_join` / `anti_join` | Phase 4 ✓ | `collect_categorical_join_keys` + cross-frame remap |
| `filter` (Hybrid existing mask) | Phase 5(b) ✓ | `evaluate_to_selection` + Phase 3 `intersect` |
| `arrange` | Phase 5(c) ✓ | `u32` code compare when levels lex-sorted |
| `mutate` / `select` / `slice` / `pivot_*` | Through legacy shim | `to_legacy_categorical` for `CategoricalAdaptive` rows |

The Phase 5 deliverables collectively close every "deferred from
earlier phase" item the v3 brief surfaced. `Column::Categorical`
remains the default categorical storage; `CategoricalAdaptive`
opt-in for callers that need adaptive widths or shared/frozen
dictionaries.

## v3 Phase 6 — Streaming summarise + cat-aware mutate + lazy optimizer (2026-04-28)

After Phase 5 closed every deferred item, the user authorized **three
new deliverables in one bundle**: streaming aggregations for huge
data, cat-aware mutate, and a lazy-plan optimizer pass for the
cat-aware verbs.

### v3 Phase 6(a) — Streaming summarise

`TidyView::summarise_streaming(keys, aggs)` is a sibling of
`group_by + summarise` that **skips the GroupIndex materialisation
step entirely**. The legacy path builds `Vec<usize>` row indices per
group (8 bytes × N rows ≈ 800 MB for 100M rows), then walks each
per-group vector once per aggregation.

The streaming path walks visible rows ONCE, maintaining a
`BTreeMap<key, Vec<AccState>>` where each accumulator holds running
state of constant size:

| `StreamingAgg` | `AccState` | Algorithm |
|---|---|---|
| `Count` | `{ n: u64 }` | counter |
| `Sum(col)` | `{ sum, c }` | Kahan running sum |
| `Mean(col)` | `{ sum, c, n }` | Kahan + count |
| `Min(col)` / `Max(col)` | `{ cur, any }` | NaN-aware running extremum |
| `Var(col)` / `Sd(col)` | `{ n, mean, m2 }` | Welford's online variance |

Median / Quantile / NDistinct / IQR / First / Last require the full
row index list and are **not streaming**; callers fall back to the
legacy `summarise`.

**Memory:** O(K · acc_size) instead of O(N · usize). For 100M rows
/ 1000 groups / 32-byte accumulator: ~32 KB vs ~800 MB — **roughly
25 000× less memory.**

**Cat-aware:** when every key column is `Column::Categorical`, the
key tuple is `Vec<u32>` codes (bit-equal to Phase 2's group_by fast
path). Falls back to `Vec<String>` displays on mixed-type keys.

**Determinism:** BTreeMap (not HashMap), Kahan / Welford (not naive
fp sum). Output row order is the BTreeMap iteration order; byte-equal
to legacy on the streamable subset for deterministic-keyed inputs.

### v3 Phase 6(b) — Cat-aware mutate

Pre-Phase-6, `mutate("k_copy", DExpr::Col("k"))` over a `Categorical`
column would degrade to `Column::Str` because `eval_dexpr_row` for
`Col(name)` over a Categorical column produces `ExprValue::Str`, then
the row-wise reconstruction builds a `Column::Str`. Downstream verbs
(group_by, joins, arrange) lose the cat-aware fast path on `k_copy`.

Phase 6 adds an early-return at the top of `eval_expr_column`:

```rust
if let DExpr::Col(name) = expr {
    if let Some(src) = df.get_column(name) {
        match src {
            Column::Categorical { .. } | Column::CategoricalAdaptive(_) => {
                return Ok(src.clone());
            }
            _ => {}
        }
    }
}
```

The clone is structural (Vec clone), not full re-encoding. Categorical
type and level table both survive `mutate`, keeping downstream verbs
on the cat-aware fast path.

### v3 Phase 6(c) — Lazy-plan optimizer for cat-aware verbs

New variant `ViewNode::StreamingGroupSummarise { input, group_keys,
aggregations: Vec<(String, StreamingAgg)> }`. New optimizer pass
`annotate_streamable_summarise` rewrites `GroupSummarise` →
`StreamingGroupSummarise` when **every** aggregation is one of
{Count, Sum, Mean, Min, Max, Var, Sd}.

Rule is all-or-nothing: any non-streamable agg keeps the whole node
on the legacy path. Mixed-mode dispatch would require executing the
node twice, defeating the streaming win.

The pass is wired into both `execute` and `execute_batched`
executors, plus `node_output_columns`, `count_filters`, `innermost`,
`kind`, `node_kinds`, and `is_pipeline_breaker` — all the lazy plan
inspection helpers.

### v3 Phase 6 — testing & regression

- **+13 integration tests** in
  `tests/tidy_tests/test_v3_phase6_streaming_and_lazy.rs`: streaming
  Count / Sum / Mean / Min / Max / Var byte-equal to legacy (Welford
  vs two-pass within 1e-9 rel err), cat-aware key path, filter then
  streaming summarise on visible-rows-only, error on unknown column;
  cat-aware mutate type preservation; lazy rewrite kicks in for
  all-streamable, declines for non-streamable, declines for mixed,
  end-to-end execute via collect.
- **+1 same-process bench** `bench_phase6_streaming_vs_legacy_summarise`
  (ignored, 1M rows × 1000 groups, Count + Sum, 3-run avg):

  ```
  Legacy summarise:    588.7 ms
  Streaming summarise: 536.9 ms
  Speedup:             1.10×
  ```

  The CPU speedup is modest at 1M × 1000 — both paths are dominated
  by per-row Kahan sums and BTreeMap operations; the GroupIndex
  alloc overhead is small relative to the workload. **The real
  Phase 6 streaming win is memory** (25 000× at 100M × 1000) which
  enables datasets that previously OOM'd. CPU savings scale with
  cardinality / row count beyond the L2/L3 fit boundary.

```
cjc-data         215/215  (no new unit; lazy::tests pass after
                          assertion update for the streaming rewrite)
test_phase10_tidy 440/440 (+13 v3 Phase 6 integration; 12 ignored
                          benches)
bolero fuzz       32/32
tidyview-bench    38/38
physics_ml        71/71   (2 ignored long-converge)
```

### v3 Phase 6 — production reach summary

Beyond the per-verb cat-aware tables in Phase 5:

| New surface | Reach |
|---|---|
| `summarise_streaming` | Direct API call, opt-in. Lazy-plan pass auto-rewrites streamable `GroupSummarise` to streaming form. |
| Cat-aware `mutate` (Categorical pass-through) | Implicit on every `mutate` over a categorical key column — preserves the fast path for downstream group_by/joins/arrange. |
| Lazy `StreamingGroupSummarise` rewrite | Every `LazyView::group_summarise` with all-streamable aggs auto-routes through streaming at `.collect()`. |

## Related

- [[TidyView Architecture]] — updated in v0.1.6 with the
  `AdaptiveSelection` section, in v2.2 with the sparse-gather
  predicate path subsection, in v3 Phase 3 with the per-chunk
  dispatch table, in v3 Phase 4 with the cat-aware join key
  path, in v3 Phase 5 with the full Column wiring + filter
  Hybrid path + cat-aware arrange, and in v3 Phase 6 with
  streaming summarise + cat-aware mutate + lazy optimizer
- [[Determinism Contract]]
- [[ADR-0001 Tree-form MIR]] — same "deterministic everything" lineage
