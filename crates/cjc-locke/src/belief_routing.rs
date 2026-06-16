//! Single source of truth: built-in finding code → belief axis routing.
//!
//! Before v0.9 the mapping from finding codes to belief axes lived as a
//! hand-maintained `||` chain inside `api::belief_axis_scores_from_report`.
//! Two axes the detector layer can populate — **drift** and **leakage** —
//! were never wired in, so a report could contain an E9060 (`|AUC| ≥ 0.95`,
//! **Error**) leakage finding while `leakage_score` stayed a pristine
//! `1.0`. That is the same "perfect score on a broken column" class of bug
//! the silent-failures audit (`docs/locke/SILENT_FAILURES_AUDIT.md`) was
//! created to kill.
//!
//! This module makes the routing **data, not control flow**, with two
//! guarantees enforced by tests in `tests/locke/belief_tests.rs`:
//!
//! 1. **Reachability** — every built-in code Locke can emit
//!    ([`ALL_BUILTIN_CODES`]) is either routed to ≥1 belief axis by
//!    [`builtin_axis_for`] or listed in [`ADVISORY_CODES`]. A new detector
//!    whose code is added to `ALL_BUILTIN_CODES` but forgotten in both
//!    fails the guard — it can no longer silently gate CI without ever
//!    moving a belief score.
//! 2. **Partition** — no code is both routed and advisory.
//!
//! ## Determinism
//!
//! Pure `match` over `&str`, no allocation, no collection iteration. The
//! belief derivation that consumes this is unchanged in shape; only the
//! *set* of codes the drift/leakage axes recognise grows. `LockeReport`
//! JSON does not contain the belief score (it is derived on demand), so
//! this change does **not** alter any serialized report bytes — the
//! determinism contract (byte-identical `to_json()`) is untouched.

use crate::custom_detector::BeliefAxisSet;

/// Which belief axes a built-in finding `code` weakens.
///
/// Returns [`BeliefAxisSet::NONE`] for advisory codes (those that inform a
/// baseline rate, are diagnostic-only, or are meta-warnings about a
/// caller-supplied rule rather than the data). The per-axis penalty in
/// [`crate::api`] is `builtin_axis_for(code).contains(AXIS) ||
/// custom_contains(code, AXIS)`.
pub fn builtin_axis_for(code: &str) -> BeliefAxisSet {
    use BeliefAxisSet as A;
    match code {
        // -- duplication ----------------------------------------------------
        "E9003" | "E9004" => A::DUPLICATION,

        // -- schema (shape + effective-alphabet ambiguity) ------------------
        // True schema-shape codes + label-encoding risk + the categorical
        // fragmentation family (case-fold / whitespace / near-dup /
        // confusable-script / mojibake / transitive-cluster / NFC-NFD).
        "E9020" | "E9021" | "E9022" | "E9023" | "E9017" | "E9080" | "E9081"
        | "E9082" | "E9083" | "E9084" | "E9085" | "E9086" => A::SCHEMA,

        // -- constraint (violations + governance/PII presence) --------------
        "E9014" | "E9016" | "E9090" | "E9091" | "E9092" | "E9093" => A::CONSTRAINT,

        // -- drift (v0.9: NEWLY WIRED) --------------------------------------
        // Numeric drift (mean/std/range/PSI/KS), categorical drift (TVD /
        // cardinality / entropy), missingness-rate shift, small-test power,
        // column add/drop between splits, and the free-text drift family.
        // These only appear in a report when a train/test compare populated
        // them; routing here means a compared report's drift_score finally
        // reflects them instead of staying 1.0.
        "E9018" | "E9019" | "E9030" | "E9031" | "E9032" | "E9033" | "E9034"
        | "E9035" | "E9036" | "E9037" | "E9038" | "E9039" | "E9110" | "E9111"
        | "E9112" => A::DRIFT,

        // -- leakage (v0.9: NEWLY WIRED — the headline fix) -----------------
        // Per-feature near-deterministic AUC (E9060/E9061), multi-class
        // one-vs-rest AUC (E9063), and per-level deterministic-outcome
        // leakage (E9064), and v0.9 categorical-feature association leakage
        // (E9065, Cramér's V). E9062 ("target not binary — skipped") is a
        // diagnostic Notice, not leakage evidence, so it stays advisory.
        "E9060" | "E9061" | "E9063" | "E9064" | "E9065" => A::LEAKAGE,

        // -- advisory (no axis) ---------------------------------------------
        // Reaches no belief axis on purpose. Either the signal is captured
        // elsewhere (E9001 missingness → the missingness baseline RATE, not
        // a penalty), it is diagnostic/meta (E9002 no-mask, E9005 key-not-
        // found, E9006 mask-OOB, E9012/E9013 malformed impossible-rule,
        // E9062 target-not-binary, E9075 imbalance-col-not-found), or it is
        // a hint whose false-positive rate on legitimate data is too high to
        // gate a score on (E9007/E9008 sentinels, E9015/E9072 cardinality,
        // E9009 promotion, E9070 conditional-missingness, E9071 imbalance,
        // E9073 join-disagreement, E9010/E9011 constant, E9024 shape,
        // E9040/E9041 outliers, E9050–E9055 temporal). See `ADVISORY_CODES`
        // for the canonical list; several are documented upgrade candidates
        // in `docs/locke/BELIEF_AXIS_ROUTING.md`.
        _ => A::NONE,
    }
}

/// Every built-in finding code Locke can emit, sorted. The exhaustiveness
/// guard (`tests/locke/belief_tests.rs::every_builtin_code_is_routed_or_advisory`)
/// asserts each is routed by [`builtin_axis_for`] or in [`ADVISORY_CODES`].
///
/// **When you add a detector, add its code here.** That is the one manual
/// step the guard cannot perform for you — and the guard exists so the
/// *second* step (deciding its belief axis) can never be silently skipped.
pub const ALL_BUILTIN_CODES: &[&str] = &[
    "E9001", "E9002", "E9003", "E9004", "E9005", "E9006", "E9007", "E9008",
    "E9009", "E9010", "E9011", "E9012", "E9013", "E9014", "E9015", "E9016",
    "E9017", "E9018", "E9019", "E9020", "E9021", "E9022", "E9023", "E9024",
    "E9030", "E9031", "E9032", "E9033", "E9034", "E9035", "E9036", "E9037",
    "E9038", "E9039", "E9040", "E9041", "E9050", "E9051", "E9052", "E9053",
    "E9054", "E9055", "E9060", "E9061", "E9062", "E9063", "E9064", "E9065",
    "E9070", "E9071", "E9072", "E9073", "E9075", "E9080", "E9081", "E9082", "E9083",
    "E9084", "E9085", "E9086", "E9090", "E9091", "E9092", "E9093", "E9110",
    "E9111", "E9112",
];

/// Codes intentionally routed to NO belief axis. Kept explicit (rather than
/// "anything `builtin_axis_for` returns NONE for") so that adding a code to
/// [`ALL_BUILTIN_CODES`] forces a deliberate routed-or-advisory decision:
/// the guard test fails if a code is in neither place.
pub const ADVISORY_CODES: &[&str] = &[
    "E9001", "E9002", "E9005", "E9006", "E9007", "E9008", "E9009", "E9010",
    "E9011", "E9012", "E9013", "E9015", "E9024", "E9040", "E9041", "E9050",
    "E9051", "E9052", "E9053", "E9054", "E9055", "E9062", "E9070", "E9071",
    "E9072", "E9073", "E9075",
];
