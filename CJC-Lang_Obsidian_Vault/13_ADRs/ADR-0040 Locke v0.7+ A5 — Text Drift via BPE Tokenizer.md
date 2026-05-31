# ADR-0040 Locke v0.7+ A5 — Text Drift via BPE Tokenizer

- **Status:** Accepted (2026-05-29)
- **Crate:** `cjc-locke` (extended from ADRs 0028–0039)
- **Companion docs:** [[Locke Roadmap]] §v0.7+ heavy A5, [[ADR-0029 Locke v0.2 Capabilities]] (numeric KS path that text drift mirrors)

## Context

Numeric column drift is well-covered (E9039 KS-D, E9033/E9034 TVD, E9018/E9019 cardinality/entropy on categorical columns). The remaining blind spot in v0.7's drift surface was **free-text** columns: patient notes, transaction descriptions, support tickets, product reviews. A handoff-recommended split:

- **A5.1** — zero-dep, deterministic byte-pair-encoding tokenizer (the gating dependency).
- **A5.2** — vocabulary KS drift (E9110) + token entropy drift (E9111), reusing the existing `stats::ks_d_statistic` numeric path.
- **A5.3** — language distribution shift via character 3-gram fingerprint (E9112), no tokenizer dependency.

This ADR ships all three in a single session.

## Decisions

### 1. New module `crates/cjc-locke/src/tokenizer.rs` (~470 LOC including tests)

```rust
pub struct Tokenizer { /* vocab + merges + reverse map */ }

pub struct TokenizerTrainConfig {
    pub target_vocab_size: u32,      // default 1024
    pub min_pair_frequency: u64,     // default 2
}

impl Tokenizer {
    pub fn train(corpus: &[&str], cfg: &TokenizerTrainConfig) -> Self;
    pub fn encode(&self, text: &str) -> Vec<u32>;
    pub fn decode(&self, ids: &[u32]) -> String;
    pub fn vocab_size(&self) -> usize;
    pub fn merge_count(&self) -> usize;
    pub fn fingerprint(&self) -> FingerprintId;
}
```

#### Algorithm

1. Initial vocab is the 256 raw byte tokens. IDs `0..256` are bytes; IDs `>=256` are learned merges.
2. Training: count adjacent token-pair frequencies across the corpus (BTreeMap keyed by `(u32, u32)`). Pick the most-frequent pair (with explicit lex tie-break — see #2). Add a new token for the merged byte sequence. Replay merge across all training sequences. Repeat until target vocab size hit or no pair satisfies `min_pair_frequency`.
3. Encoding: replay each merge rule **in training order** against the input's byte sequence. Greedy left-to-right.
4. Decoding: concatenate token byte sequences; UTF-8 decode lossily.

#### 2. Deterministic tie-break

When multiple pairs tie on frequency (very common at small vocab sizes), the choice would otherwise depend on iteration order or hash table layout. BPE implementations that don't pin a tie-break produce non-deterministic vocabularies.

The fix: explicit lex comparison of `(left_token_bytes, right_token_bytes)` when frequencies are equal. Two runs over the same corpus produce a byte-identical `Tokenizer`, with byte-identical `fingerprint()`.

The brief specifically called this out as the *single biggest determinism trap* in BPE.

### 3. New module `crates/cjc-locke/src/text_drift.rs` (~600 LOC)

Three finding codes — each maps to a specific drift signal.

#### E9110 — Vocabulary KS drift (Warning / Error)

```rust
pub fn detect_vocabulary_ks_drift_on_column(
    column: &str,
    train_text: &str,
    test_text: &str,
    cfg: &TextDriftConfig,
) -> Option<ValidationFinding>;
```

- Trains a tokenizer on `train + test` combined (shared vocab is essential for the comparison to be meaningful).
- Encodes each side; builds token-ID frequency maps.
- Expands frequency maps into f64 sample vectors (token ID repeated `count` times) and runs the existing `stats::ks_d_statistic`.
- Fires on KS-D ≥ `vocab_ks_warn` (default 0.20) or `vocab_ks_error` (default 0.40).

This catches "same words but different proportions" — the most common production drift scenario.

#### E9111 — Token entropy drift (Warning / Error)

```rust
pub fn detect_token_entropy_drift_on_column(...) -> Option<ValidationFinding>;
```

- Shannon entropy (nats) over the token frequencies per side, using Kahan summation.
- Fires when `|H_train − H_test|` ≥ `entropy_warn` (default 0.30 nats).

Catches "vocabulary collapsed" or "vocabulary expanded" — orthogonal to KS-D. E9110 + E9111 fire together when the full distribution shifted shape; only E9110 when individual tokens redistributed; only E9111 when entropy dropped without specific token swap.

#### E9112 — Language distribution shift via character 3-gram (Warning / Error)

```rust
pub fn detect_language_distribution_shift_on_column(...) -> Option<ValidationFinding>;
```

- Character-level 3-gram frequency map per side (Unicode scalars, NOT bytes).
- Sorted union of 3-grams → integer indices.
- KS-D over the f64-expanded sample vectors.
- No tokenizer dependency — purely Unicode character tuples.

Catches "train is English, test is French" (Latin script bigrams differ) or "test has substantial emoji content the train set lacked" (BMP plane bigrams differ from astral plane). Robust to tokenizer choice because it works below the tokenizer layer.

### 4. DataFrame dispatcher

```rust
pub fn detect_text_drift(
    train_df: &DataFrame,
    test_df: &DataFrame,
    cfg: &TextDriftConfig,
) -> Vec<ValidationFinding>;
```

Iterates `train_df.columns` in insertion order. For every `Str` / `Categorical` / `CategoricalAdaptive` column that exists in both sides, runs all three detectors. Numeric / bool / datetime columns are silently skipped (the existing `crate::drift::compare` already covers those).

The dispatcher is intentionally **standalone** — *not* wired into `drift::compare()` yet. Adding a config flag (`DriftConfig::enable_text_drift`) is A5.4. The current design lets users opt in by calling `detect_text_drift` directly.

### 5. Determinism contract

- Tokenizer: BTreeMap pair counts, lex tie-break, fingerprint over (vocab + merges in order).
- Token frequencies: BTreeMap.
- Char 3-gram frequencies: BTreeMap.
- All sorts use `f64::total_cmp` (matches the existing KS-D path).
- Entropy uses Kahan summation.

Proptest verifies tokenizer training is deterministic over arbitrary `[a-z ]{20,200}` corpora and tokenizer round-trips on arbitrary ASCII inputs. Bolero fuzz verifies tokenizer and text drift never panic on arbitrary `String` inputs.

### 6. UTF-8 invariants

- Tokenizer operates on **bytes**, not chars. `encode` walks `text.bytes()`; `decode` reconstructs via `String::from_utf8_lossy`.
- This means the tokenizer can produce a merge rule whose merged byte sequence is *invalid UTF-8* (e.g. middle of a multi-byte char). That's fine — encoding/decoding still round-trip because the merge is applied symmetrically.
- For drift detection it's irrelevant: token IDs are compared, not their byte content.
- `Column::Str` is guaranteed UTF-8 by `cjc-data`, so user inputs are always valid.

## Use-case map

| Need | Detector | Example |
|---|---|---|
| "Did our vocabulary shift between training and prod?" | E9110 vocab KS | Q4 reviews vs Q1 reviews |
| "Did vocabulary collapse or expand dramatically?" | E9111 entropy | Customer-service ticket categorization went from 1000 templates to 12 |
| "Did the language change?" | E9112 char 3-gram | Multi-locale ingestion bug routed French notes to the English pipeline |

## Consequences

1. **Free-text columns are no longer a Locke blind spot.** Patient-note, ticket-description, and review-content columns get the same level of drift visibility as numeric features.
2. **The tokenizer is reusable.** Future Locke features (PII recall on tokens, token-level anomaly scoring) get a deterministic primitive for free.
3. **The combined-corpus tokenizer is intentional.** Training on `train + test` shares the vocab so the comparison is meaningful. Per-side tokenizers would produce incomparable token IDs.
4. **Production thresholds are calibrated for documented English text.** For synthetic / heavy-emoji / highly-collapsed corpora the defaults may need relaxation; the config struct exposes all knobs.

## Out of scope (v0.7+ A5.1-A5.3)

Deferred to A5.4+:

- **`drift::compare()` integration.** Currently `detect_text_drift` is standalone; wiring into the compare pipeline is a config-flag addition.
- **Tokenizer serialization.** `Tokenizer` cannot yet round-trip through `cjc-snap`. Train each run; serialize is A5.4.
- **Multi-language tokenizer presets.** No prebuilt vocabularies for medical / legal / code corpora. A5.5.
- **Token-level n-gram drift.** Currently only single-token frequency drift. Bigram / trigram token drift is a richer signal for some workloads.

## Test infrastructure

| Layer | Location | New count |
|---|---|---|
| Unit (tokenizer) | `tokenizer.rs::tests` | 21 |
| Unit (text drift) | `text_drift.rs::tests` | 15 |
| Integration | `tests/locke/text_drift_tests.rs` | 13 |
| Property (proptest) | `tests/locke/locke_proptest.rs` | 3 (round-trip, train determinism, drift determinism) |
| Bolero structural fuzz | `tests/locke/locke_fuzz.rs` | 2 (tokenizer, text drift) |

## Net delta after v0.7+ A5

- cjc-locke `--lib`: 335 → **371** (+36 = 21 tokenizer + 15 text drift)
- tests/locke: 225 → **242** (+17 — 13 integration + 3 proptest + 1 net new bolero test marker, the second bolero target shares an existing harness)
- cjc-cli `--lib`: 175 unchanged (no CLI surface added yet — coming in A5.4 alongside `drift::compare()` integration)
- tests/abng: 629 unchanged
- Workspace clean.

Related: [[ADR-0028 Locke Data Skepticism Layer]], [[ADR-0029 Locke v0.2 Capabilities]] (numeric KS path), [[ADR-0036 Locke v0.7 — Per-axis BeliefScore Composition Algebra]], [[ADR-0038 Locke v0.7+ A2 — Per-Value Category Lineage]], [[ADR-0039 Locke v0.7+ A3 — Governance Workflows]], [[Locke Roadmap]].
