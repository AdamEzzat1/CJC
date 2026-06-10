//! Density-matrix-inspired pressure correlation summary.
//!
//! ## What this is
//!
//! A deterministic, classical analogue of a quantum density matrix
//! `ρ ∈ ℂ^{N×N}` for the `N = 9` pressure dimensions in
//! [`PressureKind`]. The structure is:
//!
//! - **Diagonal** `D[k] = current_magnitude(k)`: per-kind instantaneous
//!   pressure.
//! - **Off-diagonal** `C[i, j] = cov(m_i, m_j) / (σ_i σ_j)` ∈ `[-1, 1]`:
//!   the *correlation* between pressures `i` and `j` measured over a
//!   sliding window of [`PressureField`] snapshots. (In the quantum
//!   analogy these are the "coherences"; here they're plain statistical
//!   correlations.)
//!
//! From this we derive [`PressureCorrelationSummary`]:
//!
//! - `saturation_score`: average (magnitude / threshold) across all
//!   kinds — answers "how loaded is the system overall?"
//! - `collapse_risk`: max (magnitude / threshold) across all kinds —
//!   answers "which dimension is closest to its instability point?"
//! - `dominant_coupling`: the off-diagonal cell with the largest
//!   absolute value — answers "which two pressures are moving
//!   together?"
//!
//! ## What this is NOT
//!
//! - **Not quantum**: no complex numbers, no Hermiticity proofs, no
//!   tracial constraints. We borrow the *shape* (diagonal + symmetric
//!   off-diagonal) but the contents are real statistical quantities.
//! - **Not authoritative**: like every CANA-side advisory layer, the
//!   summary informs decisions but never overrides legality / verifier
//!   gates.
//!
//! ## Determinism contract
//!
//! - Iteration over [`PressureKind::all`] is variant-declaration-order
//!   (same convention as elsewhere in the crate).
//! - All reductions use [`cjc_repro::KahanAccumulatorF64`].
//! - The off-diagonal correlation matrix is keyed by ordered
//!   `(PressureKind, PressureKind)` tuples in a [`BTreeMap`] so
//!   serialization order is total and stable.
//! - [`PressureDensityState::stable_hash`] is FNV-1a over the canonical
//!   bytes — bit-identical for byte-identical input.
//! - [`PressureCorrelationSummary::to_json`] writes a fixed key order;
//!   no `HashMap` iteration anywhere.
//!
//! ## Use in the CANA bridge
//!
//! The new `cjc-cana-compress` crate computes a *delta*
//! [`PressureDensityState`] from a compression report (memory pressure
//! ↓ by the compression reward, reconstruction pressure ↑ by the
//! observed advisory error) and re-runs the summary to get the
//! post-compression risk profile. The bridge is in `cjc-cana-compress`,
//! not here, so this module's only obligation is to provide a clean,
//! deterministic primitive.

use std::collections::BTreeMap;

use cjc_repro::KahanAccumulatorF64;

use crate::pressure::{PressureField, PressureKind};

// ---------------------------------------------------------------------------
// Local FNV-1a (avoid pulling cjc-cana into cjc-nss's dep graph)
// ---------------------------------------------------------------------------

const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut state = FNV_OFFSET_BASIS;
    for &b in bytes {
        state ^= b as u64;
        state = state.wrapping_mul(FNV_PRIME);
    }
    state
}

// ---------------------------------------------------------------------------
// PressureDensityState
// ---------------------------------------------------------------------------

/// A density-matrix-inspired summary of a pressure trajectory.
///
/// Internally stores:
/// - `diagonal: BTreeMap<PressureKind, f64>` — per-kind magnitude
///   (typically the *last* value in the trajectory, but
///   [`Self::from_trajectory_mean_diagonal`] uses the mean).
/// - `thresholds: BTreeMap<PressureKind, f64>` — instability thresholds
///   used to normalize the diagonal.
/// - `correlations: BTreeMap<(PressureKind, PressureKind), f64>` —
///   sample Pearson correlation for every unordered pair of kinds
///   present in the trajectory (stored once per ordered key, with
///   `src < dst` by [`Ord`]). Self-pairs `(k, k)` are *not* stored —
///   their correlation is always `1` by definition.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct PressureDensityState {
    diagonal: BTreeMap<PressureKind, f64>,
    thresholds: BTreeMap<PressureKind, f64>,
    correlations: BTreeMap<(PressureKind, PressureKind), f64>,
}

impl PressureDensityState {
    /// Build an empty state (no kinds, no correlations). Useful for
    /// tests and for the `Default` impl.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Construct from a trajectory of [`PressureField`]s.
    ///
    /// Diagonal entries are populated from the **last** field's
    /// magnitudes (the most recent observation). Correlations are the
    /// sample Pearson coefficient over the trajectory.
    ///
    /// For a trajectory of length `< 2`, correlations are all zero (no
    /// variance to compute).
    pub fn from_trajectory(trajectory: &[PressureField]) -> Self {
        let mut diagonal: BTreeMap<PressureKind, f64> = BTreeMap::new();
        let mut thresholds: BTreeMap<PressureKind, f64> = BTreeMap::new();

        let Some(last) = trajectory.last() else {
            return Self::empty();
        };
        for (kind, p) in last.iter() {
            diagonal.insert(*kind, p.magnitude);
            thresholds.insert(*kind, p.instability_threshold);
        }

        let correlations = if trajectory.len() < 2 {
            BTreeMap::new()
        } else {
            compute_correlations(trajectory, &diagonal)
        };

        Self {
            diagonal,
            thresholds,
            correlations,
        }
    }

    /// Like [`Self::from_trajectory`] but uses Kahan-summed *mean*
    /// magnitudes for the diagonal instead of the last-tick value.
    /// Useful when callers want to summarize a steady-state regime
    /// rather than a transient.
    pub fn from_trajectory_mean_diagonal(trajectory: &[PressureField]) -> Self {
        let mut state = Self::from_trajectory(trajectory);
        if trajectory.is_empty() {
            return state;
        }
        for kind in PressureKind::all() {
            let mut acc = KahanAccumulatorF64::new();
            let mut count = 0usize;
            for f in trajectory {
                if let Some(p) = f.get(kind) {
                    acc.add(p.magnitude);
                    count += 1;
                }
            }
            if count > 0 {
                state.diagonal.insert(kind, acc.finalize() / count as f64);
            }
        }
        state
    }

    /// Apply a per-kind magnitude delta to the diagonal (in-place).
    /// Adding `+0.1` to `Memory` raises memory pressure; adding `-0.05`
    /// to `Memory` lowers it. The values are clamped to `>= 0` because
    /// pressure can't go negative.
    pub fn apply_delta(&mut self, kind: PressureKind, delta: f64) {
        if !delta.is_finite() {
            return;
        }
        let cur = self.diagonal.get(&kind).copied().unwrap_or(0.0);
        let new_mag = (cur + delta).max(0.0);
        self.diagonal.insert(kind, new_mag);
        // Threshold defaults to 1.0 if not set, so we can normalize
        // even after a fresh `apply_delta` on a kind that wasn't in
        // the original trajectory.
        self.thresholds.entry(kind).or_insert(1.0);
    }

    /// Diagonal lookup. `0.0` if `kind` was never set.
    pub fn magnitude(&self, kind: PressureKind) -> f64 {
        self.diagonal.get(&kind).copied().unwrap_or(0.0)
    }

    /// Threshold lookup. `1.0` (the default) if `kind` was never set.
    pub fn threshold(&self, kind: PressureKind) -> f64 {
        self.thresholds.get(&kind).copied().unwrap_or(1.0)
    }

    /// Correlation lookup. Self-pairs return `1.0`; pairs the state
    /// never observed return `0.0`. The lookup is symmetric:
    /// `(a, b) == (b, a)`.
    pub fn correlation(&self, a: PressureKind, b: PressureKind) -> f64 {
        if a == b {
            return 1.0;
        }
        let key = if a < b { (a, b) } else { (b, a) };
        self.correlations.get(&key).copied().unwrap_or(0.0)
    }

    /// Iterate the diagonal in canonical order.
    pub fn iter_diagonal(&self) -> impl Iterator<Item = (PressureKind, f64)> + '_ {
        self.diagonal.iter().map(|(k, v)| (*k, *v))
    }

    /// Iterate stored correlations in canonical `(src, dst)` order.
    pub fn iter_correlations(
        &self,
    ) -> impl Iterator<Item = (PressureKind, PressureKind, f64)> + '_ {
        self.correlations.iter().map(|((a, b), v)| (*a, *b, *v))
    }

    /// Number of distinct pressure kinds with diagonal entries.
    pub fn kinds_observed(&self) -> usize {
        self.diagonal.len()
    }

    /// Derive a [`PressureCorrelationSummary`].
    pub fn summary(&self) -> PressureCorrelationSummary {
        let mut saturation_acc = KahanAccumulatorF64::new();
        let mut max_sat = 0.0f64;
        let mut max_sat_kind: Option<PressureKind> = None;
        for kind in PressureKind::all() {
            let mag = self.magnitude(kind);
            let thr = self.threshold(kind);
            if thr <= 0.0 {
                continue;
            }
            let s = (mag / thr).max(0.0);
            saturation_acc.add(s);
            if s > max_sat {
                max_sat = s;
                max_sat_kind = Some(kind);
            }
        }
        let kinds_count = PressureKind::all().len() as f64;
        let saturation_score = (saturation_acc.finalize() / kinds_count).min(1.0);
        let collapse_risk = max_sat.min(1.0);

        // Dominant coupling: max |correlation| over off-diagonal entries.
        let mut dominant: Option<(PressureKind, PressureKind, f64)> = None;
        let mut dominant_abs = 0.0f64;
        for ((a, b), v) in &self.correlations {
            let av = v.abs();
            if av > dominant_abs {
                dominant_abs = av;
                dominant = Some((*a, *b, *v));
            }
        }
        // Hash for fingerprint.
        let stable_hash = self.stable_hash();

        PressureCorrelationSummary {
            saturation_score,
            collapse_risk,
            dominant_coupling: dominant,
            dominant_kind_for_risk: max_sat_kind,
            kinds_observed: self.kinds_observed(),
            stable_hash,
        }
    }

    /// Canonical bytes. Diagonal entries first (in kind order), then
    /// thresholds, then correlations. Deterministic for byte-identical
    /// state.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out =
            Vec::with_capacity(64 + self.diagonal.len() * 24 + self.correlations.len() * 24);
        out.extend_from_slice(b"NDS0"); // NSS Density State v0
                                        // Diagonal.
        out.extend_from_slice(&(self.diagonal.len() as u32).to_le_bytes());
        for (k, v) in self.diagonal.iter() {
            out.extend_from_slice(k.label().as_bytes());
            out.push(b'|');
            out.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        // Thresholds.
        out.extend_from_slice(&(self.thresholds.len() as u32).to_le_bytes());
        for (k, v) in self.thresholds.iter() {
            out.extend_from_slice(k.label().as_bytes());
            out.push(b'|');
            out.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        // Correlations.
        out.extend_from_slice(&(self.correlations.len() as u32).to_le_bytes());
        for ((a, b), v) in self.correlations.iter() {
            out.extend_from_slice(a.label().as_bytes());
            out.push(b'>');
            out.extend_from_slice(b.label().as_bytes());
            out.push(b'|');
            out.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        out
    }

    /// FNV-1a stable hash of [`Self::canonical_bytes`]. Two density
    /// states with byte-identical canonical bytes produce the same hash.
    pub fn stable_hash(&self) -> u64 {
        fnv1a(&self.canonical_bytes())
    }
}

fn compute_correlations(
    trajectory: &[PressureField],
    diagonal: &BTreeMap<PressureKind, f64>,
) -> BTreeMap<(PressureKind, PressureKind), f64> {
    // Per-kind series with means and variances (Kahan).
    let kinds: Vec<PressureKind> = diagonal.keys().copied().collect();
    let mut series: BTreeMap<PressureKind, Vec<f64>> = BTreeMap::new();
    for &k in &kinds {
        let mut v = Vec::with_capacity(trajectory.len());
        for f in trajectory {
            v.push(f.get(k).map(|p| p.magnitude).unwrap_or(0.0));
        }
        series.insert(k, v);
    }
    let mut means: BTreeMap<PressureKind, f64> = BTreeMap::new();
    let mut variances: BTreeMap<PressureKind, f64> = BTreeMap::new();
    for &k in &kinds {
        let v = series.get(&k).expect("series populated above");
        let mut macc = KahanAccumulatorF64::new();
        for x in v {
            macc.add(*x);
        }
        let m = macc.finalize() / v.len() as f64;
        means.insert(k, m);
        let mut vacc = KahanAccumulatorF64::new();
        for x in v {
            let d = *x - m;
            vacc.add(d * d);
        }
        let var = vacc.finalize() / v.len() as f64;
        variances.insert(k, var);
    }

    let mut out: BTreeMap<(PressureKind, PressureKind), f64> = BTreeMap::new();
    for i in 0..kinds.len() {
        for j in (i + 1)..kinds.len() {
            let a = kinds[i];
            let b = kinds[j];
            let va = series.get(&a).unwrap();
            let vb = series.get(&b).unwrap();
            let ma = means[&a];
            let mb = means[&b];
            let var_a = variances[&a];
            let var_b = variances[&b];
            let denom = (var_a * var_b).sqrt();
            if denom < 1e-15 {
                // Degenerate (zero variance) — no signal; correlation
                // is undefined. Default to 0 for the determinism
                // contract.
                out.insert(if a < b { (a, b) } else { (b, a) }, 0.0);
                continue;
            }
            let mut cov_acc = KahanAccumulatorF64::new();
            for t in 0..va.len() {
                cov_acc.add((va[t] - ma) * (vb[t] - mb));
            }
            let cov = cov_acc.finalize() / va.len() as f64;
            let rho = (cov / denom).clamp(-1.0, 1.0);
            out.insert(if a < b { (a, b) } else { (b, a) }, rho);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// PressureCorrelationSummary
// ---------------------------------------------------------------------------

/// Reportable summary derived from a [`PressureDensityState`].
#[derive(Clone, Debug, PartialEq)]
pub struct PressureCorrelationSummary {
    /// Average per-kind saturation, normalized to `[0, 1]`.
    pub saturation_score: f64,
    /// Maximum per-kind saturation, normalized to `[0, 1]`. Higher means
    /// at least one dimension is near its instability threshold.
    pub collapse_risk: f64,
    /// `(src, dst, ρ)` of the pair with the largest absolute
    /// correlation. `None` when no off-diagonal correlations were
    /// computed.
    pub dominant_coupling: Option<(PressureKind, PressureKind, f64)>,
    /// The kind responsible for `collapse_risk`. `None` when the state
    /// was empty.
    pub dominant_kind_for_risk: Option<PressureKind>,
    /// Number of distinct pressure kinds observed.
    pub kinds_observed: usize,
    /// FNV-1a hash of the source density state's canonical bytes.
    pub stable_hash: u64,
}

impl PressureCorrelationSummary {
    /// Render a deterministic JSON document.
    pub fn to_json(&self) -> String {
        use std::fmt::Write as _;
        let mut s = String::with_capacity(256);
        s.push_str("{\n");
        writeln!(s, "  \"schema_version\": 1,").unwrap();
        writeln!(s, "  \"stable_hash\": \"{:016x}\",", self.stable_hash).unwrap();
        writeln!(
            s,
            "  \"saturation_score\": {},",
            f64_to_json(self.saturation_score)
        )
        .unwrap();
        writeln!(
            s,
            "  \"collapse_risk\": {},",
            f64_to_json(self.collapse_risk)
        )
        .unwrap();
        writeln!(s, "  \"kinds_observed\": {},", self.kinds_observed).unwrap();
        match self.dominant_kind_for_risk {
            Some(k) => writeln!(s, "  \"dominant_kind_for_risk\": \"{}\",", k.label()).unwrap(),
            None => s.push_str("  \"dominant_kind_for_risk\": null,\n"),
        }
        match self.dominant_coupling {
            Some((a, b, v)) => writeln!(
                s,
                "  \"dominant_coupling\": {{ \"src\": \"{}\", \"dst\": \"{}\", \"rho\": {} }}",
                a.label(),
                b.label(),
                f64_to_json(v)
            )
            .unwrap(),
            None => s.push_str("  \"dominant_coupling\": null\n"),
        }
        s.push_str("}\n");
        s
    }
}

fn f64_to_json(x: f64) -> String {
    if x.is_nan() {
        "\"NaN\"".to_string()
    } else if x.is_infinite() {
        if x.is_sign_positive() {
            "\"Inf\"".to_string()
        } else {
            "\"-Inf\"".to_string()
        }
    } else {
        format!("{x:.10e}")
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pressure::{Pressure, PressureField};

    fn field_with(magnitudes: &[(PressureKind, f64)]) -> PressureField {
        let mut f = PressureField::empty();
        for (k, m) in magnitudes {
            let p = Pressure::new(*m, 1.0, 0.1).unwrap();
            f.set(*k, p);
        }
        f
    }

    #[test]
    fn empty_trajectory_produces_empty_state() {
        let state = PressureDensityState::from_trajectory(&[]);
        assert_eq!(state.kinds_observed(), 0);
        let summary = state.summary();
        assert_eq!(summary.kinds_observed, 0);
        assert_eq!(summary.saturation_score, 0.0);
        assert_eq!(summary.collapse_risk, 0.0);
    }

    #[test]
    fn single_field_has_no_correlations() {
        let f = field_with(&[(PressureKind::Memory, 0.5), (PressureKind::Cpu, 0.2)]);
        let state = PressureDensityState::from_trajectory(&[f]);
        assert_eq!(state.iter_correlations().count(), 0);
        assert_eq!(state.magnitude(PressureKind::Memory), 0.5);
    }

    #[test]
    fn correlation_self_pair_is_one() {
        let state = PressureDensityState::empty();
        assert_eq!(
            state.correlation(PressureKind::Memory, PressureKind::Memory),
            1.0
        );
    }

    #[test]
    fn correlation_unknown_pair_is_zero() {
        let state = PressureDensityState::empty();
        assert_eq!(
            state.correlation(PressureKind::Memory, PressureKind::Cpu),
            0.0
        );
    }

    #[test]
    fn perfectly_correlated_series_gives_rho_one() {
        // Two pressures that move identically should produce ρ = 1.
        let mut traj = Vec::new();
        for t in 0..10 {
            let v = t as f64 * 0.1;
            traj.push(field_with(&[
                (PressureKind::Memory, v),
                (PressureKind::Cpu, v),
            ]));
        }
        let state = PressureDensityState::from_trajectory(&traj);
        let rho = state.correlation(PressureKind::Memory, PressureKind::Cpu);
        assert!((rho - 1.0).abs() < 1e-6, "expected ρ ≈ 1, got {}", rho);
    }

    #[test]
    fn perfectly_anticorrelated_series_gives_rho_neg_one() {
        let mut traj = Vec::new();
        for t in 0..10 {
            let v = t as f64 * 0.1;
            traj.push(field_with(&[
                (PressureKind::Memory, v),
                (PressureKind::Cpu, 1.0 - v),
            ]));
        }
        let state = PressureDensityState::from_trajectory(&traj);
        let rho = state.correlation(PressureKind::Memory, PressureKind::Cpu);
        assert!((rho + 1.0).abs() < 1e-6, "expected ρ ≈ -1, got {}", rho);
    }

    #[test]
    fn correlation_lookup_is_symmetric() {
        let mut traj = Vec::new();
        for t in 0..5 {
            let v = t as f64 * 0.1;
            traj.push(field_with(&[
                (PressureKind::Memory, v),
                (PressureKind::Cpu, v * 0.5),
            ]));
        }
        let state = PressureDensityState::from_trajectory(&traj);
        let a = state.correlation(PressureKind::Memory, PressureKind::Cpu);
        let b = state.correlation(PressureKind::Cpu, PressureKind::Memory);
        assert_eq!(a, b);
    }

    #[test]
    fn apply_delta_clamps_to_zero() {
        let mut state = PressureDensityState::empty();
        state.apply_delta(PressureKind::Memory, 0.5);
        assert_eq!(state.magnitude(PressureKind::Memory), 0.5);
        state.apply_delta(PressureKind::Memory, -1.0);
        assert_eq!(state.magnitude(PressureKind::Memory), 0.0);
    }

    #[test]
    fn apply_delta_ignores_non_finite() {
        let mut state = PressureDensityState::empty();
        state.apply_delta(PressureKind::Memory, 0.3);
        state.apply_delta(PressureKind::Memory, f64::NAN);
        state.apply_delta(PressureKind::Memory, f64::INFINITY);
        assert_eq!(state.magnitude(PressureKind::Memory), 0.3);
    }

    #[test]
    fn summary_collapse_risk_tracks_max_saturation() {
        let mut state = PressureDensityState::empty();
        state.apply_delta(PressureKind::Memory, 0.95);
        state.apply_delta(PressureKind::Cpu, 0.10);
        let summary = state.summary();
        assert!(summary.collapse_risk >= 0.95);
        assert!(summary.collapse_risk <= 1.0);
        assert_eq!(summary.dominant_kind_for_risk, Some(PressureKind::Memory));
    }

    #[test]
    fn summary_saturation_score_is_average() {
        let mut state = PressureDensityState::empty();
        state.apply_delta(PressureKind::Memory, 0.9);
        state.apply_delta(PressureKind::Cpu, 0.1);
        let summary = state.summary();
        // Average over 9 kinds: (0.9 + 0.1 + 7*0) / 9 = ~0.111
        assert!((summary.saturation_score - 1.0 / 9.0).abs() < 1e-9);
    }

    #[test]
    fn canonical_bytes_stable_under_insertion_order() {
        let mut a = PressureDensityState::empty();
        a.apply_delta(PressureKind::Memory, 0.5);
        a.apply_delta(PressureKind::Cpu, 0.3);
        let mut b = PressureDensityState::empty();
        b.apply_delta(PressureKind::Cpu, 0.3);
        b.apply_delta(PressureKind::Memory, 0.5);
        assert_eq!(a.canonical_bytes(), b.canonical_bytes());
        assert_eq!(a.stable_hash(), b.stable_hash());
    }

    #[test]
    fn stable_hash_distinguishes_states() {
        let mut a = PressureDensityState::empty();
        a.apply_delta(PressureKind::Memory, 0.5);
        let mut b = PressureDensityState::empty();
        b.apply_delta(PressureKind::Memory, 0.6);
        assert_ne!(a.stable_hash(), b.stable_hash());
    }

    #[test]
    fn summary_finds_dominant_coupling() {
        let mut traj = Vec::new();
        for t in 0..8 {
            let v = t as f64 * 0.1;
            traj.push(field_with(&[
                (PressureKind::Memory, v),
                (PressureKind::Cpu, v),      // ρ = 1.0
                (PressureKind::Io, 1.0 - v), // ρ = -1.0 with Memory/Cpu
            ]));
        }
        let state = PressureDensityState::from_trajectory(&traj);
        let summary = state.summary();
        assert!(summary.dominant_coupling.is_some());
        let (_a, _b, rho) = summary.dominant_coupling.unwrap();
        assert!(
            rho.abs() > 0.99,
            "expected near-unit dominant coupling, got {rho}"
        );
    }

    #[test]
    fn to_json_includes_required_fields() {
        let mut state = PressureDensityState::empty();
        state.apply_delta(PressureKind::Memory, 0.5);
        let summary = state.summary();
        let j = summary.to_json();
        assert!(j.contains("\"schema_version\": 1"));
        assert!(j.contains("\"stable_hash\""));
        assert!(j.contains("\"saturation_score\""));
        assert!(j.contains("\"collapse_risk\""));
        assert!(j.contains("\"kinds_observed\""));
        assert!(j.contains("\"memory\""));
    }

    #[test]
    fn to_json_handles_no_dominant_coupling() {
        let state = PressureDensityState::empty();
        let j = state.summary().to_json();
        assert!(j.contains("\"dominant_coupling\": null"));
        assert!(j.contains("\"dominant_kind_for_risk\": null"));
    }

    #[test]
    fn from_trajectory_mean_diagonal_uses_average() {
        let f1 = field_with(&[(PressureKind::Memory, 0.0)]);
        let f2 = field_with(&[(PressureKind::Memory, 1.0)]);
        let state = PressureDensityState::from_trajectory_mean_diagonal(&[f1, f2]);
        assert!((state.magnitude(PressureKind::Memory) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn deterministic_across_repeated_summarization() {
        let mut traj = Vec::new();
        for t in 0..5 {
            let v = t as f64 * 0.1;
            traj.push(field_with(&[
                (PressureKind::Memory, v),
                (PressureKind::Cpu, v * 0.5),
            ]));
        }
        let state = PressureDensityState::from_trajectory(&traj);
        let s1 = state.summary();
        for _ in 0..50 {
            let s2 = state.summary();
            assert_eq!(s1, s2);
        }
    }
}
