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
