//! Integration tests for the v0.6 batch 2 seasonality detector (E9055).

use cjc_data::{Column, DataFrame};
use cjc_locke::{detect_seasonality, FindingSeverity, SeasonalityConfig};

fn df_with_times(times: Vec<i64>) -> DataFrame {
    DataFrame::from_columns(vec![("ts".into(), Column::Int(times))]).unwrap()
}

const HOUR_MS: i64 = 3600_000;
const DAY_MS: i64 = 86_400_000;

#[test]
fn uniform_timestamps_are_quiet() {
    // 504 events spaced exactly 1 hour apart = 21 days × 24 hours/day.
    // Each hour-of-day bucket gets exactly 21 hits (one per day), each
    // day-of-week bucket gets exactly 72 hits (24 × 3 weeks). Both
    // axes' index of dispersion = 0 → no E9055 fires.
    let times: Vec<i64> = (0..504).map(|i| (i as i64) * HOUR_MS).collect();
    let df = df_with_times(times);
    let f = detect_seasonality(&df, "ts", &SeasonalityConfig::default());
    assert!(f.is_empty(), "expected quiet, got {:?}", f);
}

#[test]
fn business_hour_concentration_fires() {
    // 200 timestamps all in the 9am hour across many days → very high
    // dispersion in hour-of-day buckets.
    let times: Vec<i64> = (0..200)
        .map(|i| (i as i64) * DAY_MS + 9 * HOUR_MS)
        .collect();
    let df = df_with_times(times);
    let f = detect_seasonality(&df, "ts", &SeasonalityConfig::default());
    let hod = f
        .iter()
        .find(|x| {
            x.code == "E9055"
                && x.evidence.iter().any(|e| matches!(
                    e,
                    cjc_locke::FindingEvidence::Sample { label, value } if label == "axis" && value == "hour_of_day"
                ))
        })
        .expect("hour_of_day periodicity expected, got {:?}");
    assert_eq!(hod.severity, FindingSeverity::Notice);
}

#[test]
fn weekday_only_pattern_fires_on_dow() {
    // 200 timestamps, all on Mondays (DOW 0).
    // 1970-01-05 was a Monday → epoch ms 4*DAY_MS.
    let monday_anchor_ms = 4 * DAY_MS;
    let times: Vec<i64> = (0..200)
        .map(|i| monday_anchor_ms + (i as i64) * 7 * DAY_MS + 12 * HOUR_MS)
        .collect();
    let df = df_with_times(times);
    let f = detect_seasonality(&df, "ts", &SeasonalityConfig::default());
    let dow_fired = f.iter().any(|x| {
        x.code == "E9055"
            && x.evidence.iter().any(|e| matches!(
                e,
                cjc_locke::FindingEvidence::Sample { label, value } if label == "axis" && value == "day_of_week"
            ))
    });
    assert!(dow_fired, "day_of_week periodicity expected, got {:?}", f);
}

#[test]
fn small_sample_returns_no_findings() {
    let times: Vec<i64> = (0..10).map(|i| i * HOUR_MS).collect();
    let df = df_with_times(times);
    let cfg = SeasonalityConfig::default();
    assert!(detect_seasonality(&df, "ts", &cfg).is_empty());
}

#[test]
fn seasonality_is_deterministic() {
    let times: Vec<i64> = (0..200).map(|i| (i as i64) * DAY_MS + 9 * HOUR_MS).collect();
    let df = df_with_times(times);
    let cfg = SeasonalityConfig::default();
    let a = detect_seasonality(&df, "ts", &cfg);
    let b = detect_seasonality(&df, "ts", &cfg);
    assert_eq!(a, b);
}

#[test]
fn missing_time_column_silent() {
    let df = DataFrame::from_columns(vec![("other".into(), Column::Float(vec![1.0; 150]))]).unwrap();
    assert!(detect_seasonality(&df, "ts", &SeasonalityConfig::default()).is_empty());
}
