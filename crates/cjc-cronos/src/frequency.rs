//! Time-series frequency enum.
//!
//! Carried by [`super::TimeSeries`] alongside the time index + value array.
//! Used by ETS / ARIMA / Kalman model selection to decide seasonal period
//! defaults. Deliberately closed (not stringly-typed) so a typo is a
//! compile error, not a silently-wrong analysis.

/// Discrete sampling frequency of a time series.
///
/// Seasonal periods (number of observations per cycle) for downstream
/// methods that need them:
///
/// | Frequency  | Typical seasonal period |
/// |------------|-------------------------|
/// | `Hourly`   | 24 (daily cycle)         |
/// | `Daily`    | 7 (weekly cycle)         |
/// | `Weekly`   | 52 (yearly cycle)        |
/// | `Monthly`  | 12 (yearly cycle)        |
/// | `Quarterly`| 4 (yearly cycle)         |
/// | `Annual`   | 1 (no shorter cycle)     |
/// | `Irregular`| undefined                |
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Frequency {
    Hourly,
    Daily,
    Weekly,
    Monthly,
    Quarterly,
    Annual,
    /// Used when the time index has no fixed cadence (e.g., transaction
    /// timestamps). Downstream methods that need a seasonal period must
    /// either fail with [`super::CronosError::Unsupported`] or document
    /// their handling explicitly.
    Irregular,
}

impl Frequency {
    /// Default seasonal period for this frequency. Returns `1` for `Annual`
    /// (no shorter cycle) and `0` for `Irregular` (sentinel — caller must
    /// supply an explicit period).
    pub const fn default_seasonal_period(self) -> usize {
        match self {
            Frequency::Hourly => 24,
            Frequency::Daily => 7,
            Frequency::Weekly => 52,
            Frequency::Monthly => 12,
            Frequency::Quarterly => 4,
            Frequency::Annual => 1,
            Frequency::Irregular => 0,
        }
    }
}
