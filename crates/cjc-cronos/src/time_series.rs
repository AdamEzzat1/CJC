//! [`TimeSeries`] wrapper — the foundational type passed to every
//! cjc-cronos method.
//!
//! Per the handoff §3.1 (recommendation locked in ADR-0044), v0.1 adopts a
//! `TimeSeries<f64>` wrapper struct rather than a column-name-and-DataFrame
//! convention. The wrapper carries:
//!
//! - a `Vec<i64>` time index (epoch milliseconds for `Frequency::Hourly`+,
//!   monotonically increasing observation count for `Frequency::Irregular`)
//! - a `Vec<f64>` value array
//! - a [`super::Frequency`] tag
//!
//! Construction is fallible — [`TimeSeries::from_dataframe`] checks that
//! the time + value columns exist, have matching lengths, and that the
//! time index is monotonically increasing.

use crate::error::CronosError;
use crate::frequency::Frequency;
use cjc_data::{Column, DataFrame};

/// Time-indexed univariate sequence of `f64` values.
///
/// Constructors enforce: time index is monotonically increasing, time and
/// values have equal length, both columns are numeric (`i64` for time,
/// `f64` for value).
#[derive(Clone, Debug, PartialEq)]
pub struct TimeSeries {
    time: Vec<i64>,
    values: Vec<f64>,
    frequency: Frequency,
}

impl TimeSeries {
    /// Construct directly from vectors. The time index must be monotonically
    /// increasing; otherwise returns [`CronosError::UnsortedTimeIndex`].
    pub fn new(time: Vec<i64>, values: Vec<f64>, frequency: Frequency) -> Result<Self, CronosError> {
        if time.len() != values.len() {
            return Err(CronosError::Numerical {
                detail: format!(
                    "TimeSeries: time has {} elements but values has {}",
                    time.len(),
                    values.len()
                ),
            });
        }
        for i in 1..time.len() {
            if time[i] <= time[i - 1] {
                return Err(CronosError::UnsortedTimeIndex { first_offending_row: i });
            }
        }
        Ok(Self { time, values, frequency })
    }

    /// Construct from a [`DataFrame`] by reading the named time + value
    /// columns. The time column must be `Column::Int` (epoch ms or an
    /// observation counter); the value column must be `Column::Float` or
    /// `Column::Int` (the latter is cast to `f64`).
    pub fn from_dataframe(
        df: &DataFrame,
        time_col: &str,
        value_col: &str,
        frequency: Frequency,
    ) -> Result<Self, CronosError> {
        let time_column = df
            .get_column(time_col)
            .ok_or_else(|| CronosError::UnknownColumn { name: time_col.to_string() })?;
        let value_column = df
            .get_column(value_col)
            .ok_or_else(|| CronosError::UnknownColumn { name: value_col.to_string() })?;

        let time = match time_column {
            Column::Int(v) => v.clone(),
            Column::DateTime(v) => v.clone(),
            other => {
                return Err(CronosError::WrongColumnType {
                    name: time_col.to_string(),
                    expected: "Int or DateTime".to_string(),
                    found: other.type_name().to_string(),
                })
            }
        };
        let values = match value_column {
            Column::Float(v) => v.clone(),
            Column::Int(v) => v.iter().map(|&x| x as f64).collect(),
            other => {
                return Err(CronosError::WrongColumnType {
                    name: value_col.to_string(),
                    expected: "Float or Int".to_string(),
                    found: other.type_name().to_string(),
                })
            }
        };
        Self::new(time, values, frequency)
    }

    /// Number of observations.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// `true` if the series has zero observations.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Time index, as a read-only slice.
    pub fn time(&self) -> &[i64] {
        &self.time
    }

    /// Value array, as a read-only slice.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Declared [`Frequency`].
    pub fn frequency(&self) -> Frequency {
        self.frequency
    }
}
