//! Identification assumptions a caller must declare for any causal estimate.
//!
//! Every estimator in this crate takes a `&[IdentificationAssumption]` slice
//! at `estimate()` time. There are **no default assumptions**: the analyst's
//! declared assumptions are part of the report's content-addressed identifier,
//! so two estimates from the same data with different assumption sets carry
//! different IDs.
//!
//! The variants enumerate the assumptions cjc-causal's v0.1 estimators
//! recognise. They are deliberately a closed enum (not a string) so a typo
//! is a compile error, not a silently-misclassified analysis.

/// A causal-identification assumption the caller commits to when requesting
/// an estimate. See Hernán & Robins (2020), Chapters 3 and 16, for the
/// canonical definitions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum IdentificationAssumption {
    /// Conditional exchangeability: given the declared covariates, treatment
    /// assignment is independent of potential outcomes. Required by
    /// [`super::PropensityScoreMatcher`] (planned) and the outcome arm of
    /// [`super::DoubleMLEstimator`] (planned).
    Unconfoundedness,

    /// Positivity / common support: every covariate stratum has a non-zero
    /// probability of receiving each treatment level. Violations surface as
    /// Locke finding `E9101`.
    Positivity,

    /// Instrument validity — exclusion restriction: the instrument affects
    /// the outcome only through the treatment. Required by [`super::IVRegression`].
    ExcludabilityOfInstrument,

    /// Instrument validity — monotonicity: the instrument never moves any
    /// unit *against* the direction it moves the average unit (no defiers).
    /// Required for LATE interpretation of IV estimates.
    MonotonicityOfInstrument,

    /// Parallel trends: in the absence of treatment, treated and control
    /// groups would have followed the same trend. Required by
    /// difference-in-differences (deferred to v0.2).
    ParallelTrends,

    /// Local randomisation: assignment is as-if-random in a neighborhood of
    /// the cutoff. Required by regression discontinuity (deferred to v0.2).
    LocalRandomization,

    /// Stable unit treatment value assumption (SUTVA): no interference
    /// between units; one consistent version of treatment. Required by all
    /// estimators.
    NoInterference,
}
