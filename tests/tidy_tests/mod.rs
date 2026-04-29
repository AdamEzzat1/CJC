// Phase 10–17: Tidy Primitives — Test Suite
//
// Run standalone: cargo test --test test_phase10_tidy
// Run specific:   cargo test --test test_phase10_tidy filter
//
// Sub-suites:
//   filter:         cargo test --test test_phase10_tidy tidy_filter
//   select:         cargo test --test test_phase10_tidy tidy_select
//   mutate:         cargo test --test test_phase10_tidy tidy_mutate
//   alias:          cargo test --test test_phase10_tidy tidy_alias
//   group_by:       cargo test --test test_phase10_tidy phase11_group_by
//   summarise:      cargo test --test test_phase10_tidy phase11_summarise
//   arrange:        cargo test --test test_phase10_tidy phase11_arrange
//   slice:          cargo test --test test_phase10_tidy phase11_slice
//   distinct:       cargo test --test test_phase10_tidy phase12_distinct
//   joins:          cargo test --test test_phase10_tidy phase12_joins
//   pivot_longer:   cargo test --test test_phase10_tidy phase13_pivot_longer
//   pivot_wider:    cargo test --test test_phase10_tidy phase13_pivot_wider
//   bind:           cargo test --test test_phase10_tidy phase13_bind
//   rename/reloc:   cargo test --test test_phase10_tidy phase13_rename
//   across:         cargo test --test test_phase10_tidy phase14_across
//   nullable:       cargo test --test test_phase10_tidy phase14_nullable
//   join_maturity:  cargo test --test test_phase10_tidy phase15_join
//   group_perf:     cargo test --test test_phase10_tidy phase16_group
//   forcats:        cargo test --test test_phase10_tidy phase17
//   nogc:           cargo test --test test_phase10_tidy tidy_nogc
//   perf:           cargo test --test test_phase10_tidy -- --ignored (perf gate only)

// Phase 10
pub mod test_tidy_filter_empty_result;
pub mod test_tidy_filter_empty_df;
pub mod test_tidy_filter_chain_mask_merge;
pub mod test_tidy_select_reorder;
pub mod test_tidy_select_zero_cols;
pub mod test_tidy_mutate_type_promotion;
pub mod test_tidy_mutate_ordering;
pub mod test_tidy_mutate_masked_view;
pub mod test_tidy_alias_safety;
pub mod test_tidy_nogc_rejection;
pub mod test_tidy_speed_gate;

// Phase 11
pub mod test_phase11_group_by;
pub mod test_phase11_summarise;
pub mod test_phase11_arrange;
pub mod test_phase11_slice;

// Phase 12
pub mod test_phase12_distinct;
pub mod test_phase12_joins;

// Phase 11-12 NoGC negative tests
pub mod test_phase11_12_nogc;

// Phase 13: Reshape + Column Management
pub mod test_phase13_pivot_longer;
pub mod test_phase13_pivot_wider;
pub mod test_phase13_bind;
pub mod test_phase13_rename_relocate_drop;

// Phase 14: Across + Nullable Semantics
pub mod test_phase14_across;
pub mod test_phase14_nullable_semantics;

// Phase 15: Join Maturity
pub mod test_phase15_join_maturity;

// Phase 16: Group Perf + NoGC Negative
pub mod test_phase16_group_perf_semantics;
pub mod test_phase16_nogc_negative;

// Phase 17: Categorical Foundations (forcats)
pub mod test_phase17_forcats;

// Adaptive TidyView Engine v2 (sparse/dense selection modes)
pub mod test_adaptive_selection_integration;
pub mod test_adaptive_selection_determinism;
pub mod test_adaptive_selection_join_adversarial;
pub mod test_adaptive_selection_bench;

// Adaptive TidyView Engine v2.1 — predicate bytecode parity
pub mod test_v2_1_bytecode_parity;

// Adaptive TidyView Engine v2.2 — sparse-chain filter parity
pub mod test_v2_2_sparse_chain_parity;

// v3 Phase 2 — Categorical-aware key path for group_by + distinct
//   When all key columns are Column::Categorical, build the lookup
//   BTreeMap over Vec<u32> codes instead of Vec<String> displays.
//   Bit-identical to the string-key path; falls back automatically on
//   mixed-type keys.
pub mod test_v3_phase2_categorical_keys;

// v3 Phase 3 — Hybrid streaming set-op fast paths
//   AdaptiveSelection::intersect/union now have per-chunk dispatch when one
//   or both operands are Hybrid. Pins parity vs scalar oracle for every
//   shape combination and validates the chunked layout survives chained
//   set ops without spurious bitmap materialization.
pub mod test_v3_phase3_hybrid_streaming;

// v3 Phase 10 — DHarht Memory profile port (u64 keys, sparse paged
//   16-bit front directory, packed u64 entries with 3-bit tags,
//   MicroBucket4/8/16 with parallel match_mask). Benchmarks against
//   BTreeMap, std::HashMap, and our DHarht v.01 (byte-key) on the
//   same workload shape used in the user-supplied
//   `D-HARHT-Blueprint-and-Code.md` reference bench.
pub mod dharht_3way_u64_bench;

// v3 Phase 11 — Memory + Security comparison: BTreeMap vs HashMap vs
//   DHarhtMemory. Pins the non-speed value: bytes/entry footprint,
//   determinism contract (BTreeMap and DHarhtMemory byte-equal across
//   builds; HashMap NOT — seeded SipHash randomizes iter), and
//   adversarial-collision safety (full key equality, MicroBucket16
//   bound, no silent loss). Also pins the new wiring of
//   `SealedU64Map` into `ByteDictionary::seal_with_u64_hash_index`.
pub mod dharht_memory_security;

// v3 Phase 7 — Deterministic collection family + DHarht Memory profile.
//   New module `cjc_data::detcoll` ships IndexVec, TinyDetMap,
//   SortedVecMap, DetOpenMap, and DHarht (Memory profile). DHarht is
//   wired into `ByteDictionary::seal_for_lookup()` as a post-seal
//   lookup accelerator. ADR-0019 covers the architecture.
pub mod dharht_memory_backend;
pub mod deterministic_collections;
pub mod tidyview_categorical_dharht;
pub mod prop_dharht_memory;
pub mod bolero_dharht_memory;

// v3 Phase 6 — Streaming summarise + cat-aware mutate + lazy optimizer
//   (1) `TidyView::summarise_streaming` does single-pass aggregation
//       over visible rows (Kahan sum, Welford variance), avoiding the
//       O(N · usize) GroupIndex row-index buffers.
//   (2) `mutate("col", DExpr::Col("cat_col"))` preserves Categorical /
//       CategoricalAdaptive type instead of degrading to Str.
//   (3) Lazy-plan optimizer rewrites GroupSummarise →
//       StreamingGroupSummarise when every aggregation is in
//       {Count, Sum, Mean, Min, Max, Var, Sd}.
pub mod test_v3_phase6_streaming_and_lazy;

// v3 Phase 5 — Full Column wiring + filter-chain Hybrid path + cat-aware arrange
//   Three deliverables in one phase: (1) Column::CategoricalAdaptive
//   variant wraps Phase 1's CategoricalColumn as a first-class column
//   type with `to_legacy_categorical` shim covering 19 consumer sites;
//   (2) TidyView::filter routes through `existing.intersect(fresh)`
//   when existing mask is Hybrid, exercising Phase 3 fast paths in
//   production; (3) arrange uses u32 code comparison when Categorical
//   levels are lex-sorted, byte-equal to string compare.
pub mod test_v3_phase5_full_wiring;

// v3 Phase 4 — Categorical-aware join key path
//   When all join-key columns are Column::Categorical on BOTH sides,
//   inner/left/semi/anti joins probe on Vec<u32> code keys (with a
//   deterministic right_code → Option<left_code> remap) instead of
//   Vec<String> displays. Bit-identical output to the string path;
//   automatic fallback on mixed-type keys.
pub mod test_v3_phase4_categorical_joins;

// v3 Phase 1 — Deterministic Adaptive Dictionary Engine
//   Standalone byte-first categorical column engine. Phase 1 does NOT wire
//   into TidyView verbs (Phase 2). Tests cover round-trip, lexical-seal
//   determinism, frozen rejection, unknown-category policies, profile.
pub mod test_adaptive_dictionary_engine;

// v3 Phase 0 — Tidyverse integration parity
//   Pins that the language-level dispatch path (dispatch_tidy_method)
//   produces bit-identical output to the direct Rust TidyView API across
//   filter / select / arrange / distinct / group_by+summarise pipelines.
pub mod test_tidyverse_integration_parity;
