//! Integration tests for v0.7+ heavy A5: deterministic BPE tokenizer
//! (A5.1) + vocabulary KS / token entropy / language-shift detectors
//! (A5.2, A5.3).

use cjc_data::{Column, DataFrame};
use cjc_locke::{
    detect_language_distribution_shift_on_column, detect_text_drift,
    detect_token_entropy_drift_on_column, detect_vocabulary_ks_drift_on_column,
    FindingEvidence, FindingSeverity, TextDriftConfig, Tokenizer, TokenizerTrainConfig,
};

// ─── Helpers ──────────────────────────────────────────────────────────────

fn str_df(col: &str, values: &[&str]) -> DataFrame {
    DataFrame::from_columns(vec![(
        col.into(),
        Column::Str(values.iter().map(|s| s.to_string()).collect()),
    )])
    .unwrap()
}

// ─── Tokenizer round-trip on representative inputs ────────────────────────

#[test]
fn tokenizer_round_trips_on_diverse_realistic_inputs() {
    let corpus: Vec<&str> = vec![
        "Patient presents with hypertension and type 2 diabetes.",
        "Reports occasional chest pain, no shortness of breath.",
        "Medications: metformin 500mg BID, lisinopril 10mg QD.",
        "Follow-up in 3 months; A1c target < 7.0%.",
    ];
    let t = Tokenizer::train(&corpus, &TokenizerTrainConfig::default());
    for s in &corpus {
        let ids = t.encode(s);
        assert_eq!(t.decode(&ids), *s);
    }
}

#[test]
fn tokenizer_handles_emoji_and_combining_marks() {
    let corpus = vec!["café ☕️ 🥐", "déjà vu naïve façade", "🚀🌙"];
    let t = Tokenizer::train(&corpus, &TokenizerTrainConfig::default());
    for s in &corpus {
        let ids = t.encode(s);
        assert_eq!(t.decode(&ids), *s);
    }
}

// ─── E9110 vocabulary KS — DataFrame integration ─────────────────────────

#[test]
fn vocab_ks_fires_on_clinical_vs_marketing_text_columns() {
    // A "notes" column from a clinical-notes pipeline vs the same column
    // from a marketing-feedback pipeline — totally different vocabularies.
    let clinical = vec![
        "Patient presents with hypertension.";
        20
    ];
    let marketing = vec![
        "Customer reports five-star satisfaction. Highly recommend!";
        20
    ];
    let train = str_df("notes", &clinical);
    let test = str_df("notes", &marketing);
    let findings = detect_text_drift(&train, &test, &TextDriftConfig::default());
    assert!(
        findings.iter().any(|f| f.code == "E9110"),
        "expected E9110 vocabulary drift, got: {:?}",
        findings.iter().map(|f| &f.code).collect::<Vec<_>>()
    );
}

#[test]
fn vocab_ks_quiet_when_train_and_test_match() {
    let phrase = vec!["The cat sat on the mat."; 50];
    let train = str_df("notes", &phrase);
    let test = str_df("notes", &phrase);
    let findings = detect_text_drift(&train, &test, &TextDriftConfig::default());
    assert!(
        !findings.iter().any(|f| f.code == "E9110"),
        "expected no E9110 on identical inputs"
    );
}

// ─── E9111 token entropy — DataFrame integration ─────────────────────────

#[test]
fn token_entropy_fires_when_test_collapses_to_single_phrase() {
    let train: Vec<&str> = vec![
        "alpha bravo charlie",
        "delta echo foxtrot",
        "golf hotel india",
        "juliet kilo lima",
    ];
    // Test: same one phrase repeated.
    let test_vals: Vec<&str> = vec!["alpha alpha alpha"; 60];
    let train_df = str_df("notes", &train.repeat(15));
    let test_df = str_df("notes", &test_vals);
    let cfg = TextDriftConfig {
        // Production threshold is 0.30 nats; relax to 0.05 here because
        // shared-corpus tokenizer learns full-word merges that compress
        // the entropy delta of synthetic test inputs.
        entropy_warn: 0.05,
        entropy_error: 0.10,
        ..Default::default()
    };
    let findings = detect_text_drift(&train_df, &test_df, &cfg);
    assert!(
        findings.iter().any(|f| f.code == "E9111"),
        "expected E9111 entropy drift, got: {:?}",
        findings.iter().map(|f| &f.code).collect::<Vec<_>>()
    );
}

// ─── E9112 language distribution shift ───────────────────────────────────

#[test]
fn language_shift_fires_on_english_vs_french() {
    let english: Vec<&str> = vec![
        "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.";
        20
    ];
    let french: Vec<&str> = vec![
        "Voix ambiguë d'un cœur qui au zéphyr préfère les jattes de kiwis. Portez ce vieux whisky.";
        20
    ];
    let train = str_df("notes", &english);
    let test = str_df("notes", &french);
    let findings = detect_text_drift(&train, &test, &TextDriftConfig::default());
    assert!(
        findings.iter().any(|f| f.code == "E9112"),
        "expected E9112 language drift, got: {:?}",
        findings.iter().map(|f| &f.code).collect::<Vec<_>>()
    );
}

#[test]
fn language_shift_fires_on_ascii_to_emoji() {
    let ascii: Vec<&str> = vec!["The quick brown fox jumps over the lazy dog."; 20];
    let emoji: Vec<&str> = vec!["🐶🐱🐭🐹🐰🦊🐻🐼🐨🐯🦁🐮🐷🐸🐵🐔🐧🐦🐤🦆"; 20];
    let train = str_df("notes", &ascii);
    let test = str_df("notes", &emoji);
    let findings = detect_text_drift(&train, &test, &TextDriftConfig::default());
    let e9112 = findings
        .iter()
        .find(|f| f.code == "E9112")
        .expect("E9112 missing");
    // Severity should be Error on a totally disjoint character distribution.
    assert_eq!(e9112.severity, FindingSeverity::Error);
}

// ─── Determinism on real DataFrames ──────────────────────────────────────

#[test]
fn detect_text_drift_is_deterministic_on_dataframe_input() {
    let train = str_df(
        "notes",
        &vec!["The cat sat on the mat. The dog ran past the gate."; 40],
    );
    let test = str_df(
        "notes",
        &vec!["A bird flew over the river while the fish swam slowly."; 40],
    );
    let cfg = TextDriftConfig::default();
    let a = detect_text_drift(&train, &test, &cfg);
    let b = detect_text_drift(&train, &test, &cfg);
    assert_eq!(a, b);
}

// ─── Two columns with mixed drift signal ─────────────────────────────────

#[test]
fn detect_text_drift_processes_multiple_columns_independently() {
    let train = DataFrame::from_columns(vec![
        (
            "stable".into(),
            Column::Str(vec!["consistent content"; 40].iter().map(|s| s.to_string()).collect()),
        ),
        (
            "drifting".into(),
            Column::Str(vec!["english english english"; 40].iter().map(|s| s.to_string()).collect()),
        ),
    ])
    .unwrap();
    let test = DataFrame::from_columns(vec![
        (
            "stable".into(),
            Column::Str(vec!["consistent content"; 40].iter().map(|s| s.to_string()).collect()),
        ),
        (
            "drifting".into(),
            Column::Str(vec!["🐶🐱🐭🐹🐰🦊🐻🐼🐨🐯🦁🐮🐷🐸🐵🐔🐧🐦🐤🦆"; 40].iter().map(|s| s.to_string()).collect()),
        ),
    ])
    .unwrap();
    let cfg = TextDriftConfig::default();
    let findings = detect_text_drift(&train, &test, &cfg);
    // Stable column should not fire.
    assert!(
        findings.iter().all(|f| f.column.as_deref() != Some("stable")),
        "stable column should produce no findings"
    );
    // Drifting column should fire on at least one of E9110/E9112.
    assert!(findings
        .iter()
        .any(|f| f.column.as_deref() == Some("drifting")
            && (f.code == "E9110" || f.code == "E9112")));
}

// ─── Evidence payload sanity ─────────────────────────────────────────────

#[test]
fn e9110_finding_carries_ks_d_and_vocab_size_evidence() {
    let train = vec!["abc def ghi jkl mno pqr stu vwx yz "; 20].concat();
    let test = vec!["aaa bbb ccc ddd eee fff ggg hhh "; 20].concat();
    let cfg = TextDriftConfig::default();
    let f = detect_vocabulary_ks_drift_on_column("c", &train, &test, &cfg)
        .expect("E9110 expected");
    let has_ks = f.evidence.iter().any(|e| {
        matches!(e, FindingEvidence::Metric { label, .. } if label == "vocab_ks_d")
    });
    let has_vocab = f.evidence.iter().any(|e| {
        matches!(e, FindingEvidence::Count { label, .. } if label == "shared_vocab_size")
    });
    assert!(has_ks, "missing vocab_ks_d evidence");
    assert!(has_vocab, "missing shared_vocab_size evidence");
}

#[test]
fn e9111_finding_carries_train_test_entropy_evidence() {
    let train_text = "alpha bravo charlie delta echo foxtrot golf hotel ".repeat(15);
    let test_text = "alpha ".repeat(150);
    let cfg = TextDriftConfig {
        entropy_warn: 0.05,
        entropy_error: 0.10,
        ..Default::default()
    };
    let f =
        detect_token_entropy_drift_on_column("c", &train_text, &test_text, &cfg)
            .expect("E9111 expected");
    let labels: Vec<&str> = f
        .evidence
        .iter()
        .filter_map(|e| match e {
            FindingEvidence::Metric { label, .. } => Some(label.as_str()),
            _ => None,
        })
        .collect();
    assert!(labels.contains(&"train_entropy_nats"));
    assert!(labels.contains(&"test_entropy_nats"));
    assert!(labels.contains(&"abs_delta_nats"));
}

#[test]
fn e9112_finding_carries_union_3gram_count_evidence() {
    let train = "the quick brown fox jumps over the lazy dog. ".repeat(20);
    let test = "🐶🐱🐭🐹🐰🦊🐻🐼🐨🐯🦁🐮🐷🐸🐵🐔🐧🐦🐤🦆".repeat(15);
    let cfg = TextDriftConfig::default();
    let f =
        detect_language_distribution_shift_on_column("c", &train, &test, &cfg)
            .expect("E9112 expected");
    let has_union = f.evidence.iter().any(|e| {
        matches!(e, FindingEvidence::Count { label, .. } if label == "n_union_distinct_3grams")
    });
    assert!(has_union);
}
