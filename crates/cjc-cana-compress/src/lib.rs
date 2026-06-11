//! # `cjc-cana-compress` â€” Quantum-inspired CANA compression + energy ranking
//!
//! Phase 6 of the CANA roadmap. Adds an **advisory** compression layer that
//! sits between CANA's feature/pass-history substrate and NSS's pressure
//! model, plus an Ising/QAOA-style **energy ranker** that scores already-legal
//! compiler plans against a multi-objective scalar.
//!
//! ## Design pillars
//!
//! 1. **Advisory, not authoritative.** Nothing in this crate can mutate the
//!    MIR, change pass legality, or override the verifier. Compression
//!    decisions and energy rankings *recommend*; the existing
//!    `cjc_cana::LegalityGate` and the MIR verifier remain final authority.
//!    The architecture mirrors how branch predictors are advisory while the
//!    ISA semantics are authoritative.
//!
//! 2. **Lossless on semantic-critical facts.** Every
//!    [`CompressionCandidate`] carries an explicit
//!    [`Criticality`](candidate::Criticality) tag. Semantic-critical inputs
//!    (pass histories, alias/effect facts, no-GC facts, audit chains, exact
//!    shape facts) may only use lossless [`CompressionKind`] variants;
//!    constructing a `SemanticCritical` candidate with a lossy kind returns
//!    [`CompressionError::LossyOnCritical`].
//!
//! 3. **Quantum-inspired â‰  quantum-dependent.** We reuse the deterministic
//!    discipline of [`cjc_quantum`] â€” sign-stabilized SVD, Kahan
//!    accumulation, fixed iteration order, no FMA â€” but no quantum hardware
//!    is required. The Ising/QAOA terminology is honest only because (a)
//!    the energy decomposition is exposed for audit (no hidden weights),
//!    and (b) the optimization is over a finite, deterministic candidate
//!    set with stable tie-breaking, not a stochastic minimum search.
//!
//! 4. **Determinism contract.** Identical input â†’ byte-identical
//!    [`CompressionReport`] hash, identical
//!    [`EnergyRanking`](energy::EnergyRanking) order, identical
//!    [`PressureDensityState`](cjc_nss::PressureDensityState) bytes. The
//!    determinism story is enforced through:
//!    - [`BTreeMap`](std::collections::BTreeMap) everywhere (no
//!      `HashMap` iteration in decision paths).
//!    - All reductions via [`cjc_repro::KahanAccumulatorF64`].
//!    - Stable candidate IDs ([`candidate::CandidateId`]) for tie-breaking
//!      in [`EnergyRanker::rank`](energy::EnergyRanker::rank).
//!    - [`cjc_cana::CanaHasher`] (FNV-1a) for every byte stream.
//!    - No wall-clock time, no thread-local randomness in decision outputs.
//!
//! ## Architecture
//!
//! ```text
//!   CANA features / pass history
//!      â”‚
//!      â–¼
//!   CompressionCandidate (advisory or semantic-critical)
//!      â”‚
//!      â–¼
//!   CompressionPlan â”€â–º chosen [`CompressionKind`]:
//!      â”‚                 LosslessTrace, MotifDictionary,
//!      â”‚                 LowRankAdvisory, TensorTrainAdvisory
//!      â–¼
//!   CompressionReport (input/compressed/reconstructed hashes,
//!                      observed reconstruction error, stable bytes)
//!      â”‚
//!      â”œâ”€â–º [`bridge::pressure_delta`] â”€â–º NSS PressureDensityState Î”
//!      â”‚       â””â”€â–º [`cjc_nss::PressureCorrelationSummary`] update
//!      â”‚
//!      â””â”€â–º [`energy::EnergyRanker`] â”€â–º deterministic ordering
//!              of candidate plans by Ising/QAOA-style scalar
//!
//!   â®¡ legality / verifier remain final authority over MIR shape
//! ```
//!
//! ## Cross-references
//!
//! - [`cjc_cana`] â€” the passive observer this crate extends.
//! - [`cjc_nss`] â€” pressure-modelling substrate this crate emits deltas
//!   into; the new `density` module ships
//!   [`cjc_nss::PressureDensityState`].
//! - [`cjc_quantum`] â€” source of MPS truncation + sign-stabilized SVD
//!   used by [`tensor_train`].
//! - `docs/cana_compress/DESIGN.md` â€” design rationale.
//! - `docs/cana_compress/ARCHITECTURE.md` â€” diagram + flow.
//! - `docs/cana_compress/BLOG_NOTES.md` â€” blog seed for the eventual
//!   write-up.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod bridge;
pub mod candidate;
pub mod energy;
pub mod energy_pass_ranker;
pub mod lossless_trace;
pub mod lowrank;
pub mod motif_dictionary;
pub mod energy_bundle;
pub mod pinn_bundle;
pub mod plan;
pub mod profile_db;
pub mod report;
pub mod sidecar;
pub mod tensor_train;

pub use bridge::{
    compression_pressure_delta, BridgeCoefficients, CompressionAwarePressurePredictor,
    CompressionPressureDelta,
};
pub use candidate::{
    CandidateId, CompressionCandidate, CompressionError, CompressionKind, Criticality,
    CriticalityTag,
};
pub use energy::{
    EnergyComponents, EnergyRanker, EnergyRanking, EnergyScore, RankedCandidate, RankingMetadata,
};
pub use energy_pass_ranker::{
    derive_energy_components, pass_benefit_channel, pass_code_size_factor, BenefitChannel,
    EnergyAuditEntry, EnergyAwarePassRanker, EnergyComponentsConfig,
};
pub use lossless_trace::{compress_pass_history, decompress_pass_history, LosslessTracePayload};
pub use lowrank::{compress_low_rank, LowRankPayload};
pub use motif_dictionary::{
    compress_motif_dictionary, decompress_motif_dictionary, MotifDictionaryPayload,
};
pub use pinn_bundle::{
    read_bundle, write_bundle, PinnBundle, PinnBundleError, PINN_BUNDLE_SCHEMA_VERSION,
};
pub use plan::{encode_low_rank_payload, encode_tensor_train_payload, CompressionPlan, PlanEntry};
pub use profile_db::{
    append_row, read_all, CompilationProfile, ProfileDbError, PROFILE_SCHEMA_VERSION,
};
pub use report::{CompressionReport, EntryStatus, ReportEntry, ReportHash};
pub use sidecar::{CompressedCanaSidecar, SidecarIoError};
pub use tensor_train::{compress_tensor_train, TensorTrainPayload};

/// Crate-level version string stamped into [`CompressionReport::canonical_bytes`]
/// so consumers can detect when a report was produced by an older version.
pub const COMPRESS_VERSION: &str = "cana-compress-0.1.0";
