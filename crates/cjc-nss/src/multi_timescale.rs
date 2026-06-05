//! Phase 3b — multi-timescale pressure memory.
//!
//! The Phase 1/2 [`crate::TemporalStateEngine`] is structurally a single
//! SSM with one spectral-norm bound `α`. That gives the model **one**
//! decay rate — about 95% retention per tick at α=0.95 — which is fine
//! for short-horizon prediction but loses two important regimes:
//!
//! - **Short-term spikes** (queue bursts, jitter peaks): an α=0.95
//!   buffer dampens these too aggressively. Want α≈0.5 so a spike at
//!   tick T is still half-visible at T+1.
//! - **Structural drift** (memory fragmentation, scheduler imbalance):
//!   an α=0.95 buffer forgets these by tick T+15. Want α≈0.99 so the
//!   drift is still ~85% visible at T+20.
//!
//! [`MultiTimescaleEngine`] runs **N parallel SSM buffers** at distinct
//! α's, all seeing the same input. Their outputs concatenate into a
//! wider latent the head can attend to. The architecture deliberately
//! mirrors how biological memory works (different brain regions cache
//! at different timescales) and how the Liquid Time-Constant networks
//! in `cjc-cronos-gan` handle multi-scale temporal data.
//!
//! ## Canonical timescales
//!
//! [`Timescale`] is a closed enum with four named α's:
//!
//! | Variant      | Default α | Half-life (ticks) | Domain examples            |
//! |--------------|-----------|-------------------|----------------------------|
//! | Short        | 0.50      | ≈1                | queue spikes, jitter bursts |
//! | Medium       | 0.85      | ≈4                | thermal buildup             |
//! | Long         | 0.95      | ≈13               | sustained pressure trends   |
//! | Structural   | 0.99      | ≈69               | fragmentation, imbalance    |
//!
//! Half-life `≈ ln(0.5) / ln(α)` — the number of ticks before a unit
//! impulse decays to 50% magnitude.
//!
//! ## Determinism contract
//!
//! Each timescale has its **own seeded substream** for its A and B
//! matrices: `temporal.short.A`, `temporal.medium.A`, etc. So the
//! short and structural buffers can never accidentally couple — a
//! property that matters because the entire point of multi-timescale
//! memory is to *decorrelate* the signals.

use crate::error::NssError;
use crate::seed::NssSeed;
use crate::temporal::{TemporalStateConfig, TemporalStateEngine};
use cjc_repro::KahanAccumulatorF64;

/// One named decay timescale. Closed enum so the propagation pattern
/// is exhaustive.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Timescale {
    /// Short-term memory, default α=0.50. Catches queue spikes,
    /// jitter bursts, single-tick pressure surges.
    Short,
    /// Medium-term memory, default α=0.85. Catches thermal buildup,
    /// gradual queue growth, multi-tick sync stalls.
    Medium,
    /// Long-term memory, default α=0.95. Catches sustained pressure
    /// trends across a training-iteration window.
    Long,
    /// Structural memory, default α=0.99. Catches memory
    /// fragmentation, scheduler imbalance — drift that accumulates
    /// over many ticks and doesn't dissipate without intervention.
    Structural,
}

impl Timescale {
    /// Default spectral-norm bound for this timescale.
    pub fn default_alpha(self) -> f64 {
        match self {
            Timescale::Short => 0.50,
            Timescale::Medium => 0.85,
            Timescale::Long => 0.95,
            Timescale::Structural => 0.99,
        }
    }

    /// Approximate exponential half-life in ticks: `ln(0.5) / ln(α)`.
    /// Useful for documentation + tests asserting the decay separation.
    pub fn half_life_ticks(self) -> f64 {
        (0.5_f64.ln()) / (self.default_alpha().ln())
    }

    /// Canonical short label used by RNG salt + canonical bytes.
    pub fn label(self) -> &'static str {
        match self {
            Timescale::Short => "short",
            Timescale::Medium => "medium",
            Timescale::Long => "long",
            Timescale::Structural => "structural",
        }
    }

    /// All four timescales in canonical (`Ord`) order. This is the
    /// concatenation order — `MultiTimescaleEngine` emits states in
    /// `[short | medium | long | structural]` layout.
    pub const ALL: [Timescale; 4] = [
        Timescale::Short,
        Timescale::Medium,
        Timescale::Long,
        Timescale::Structural,
    ];
}

/// Knobs for [`MultiTimescaleEngine`].
#[derive(Clone, Debug, PartialEq)]
pub struct MultiTimescaleConfig {
    /// Per-scale hidden dimensionality. The concatenated output has
    /// `timescales.len() * per_scale_dim` length.
    pub per_scale_dim: usize,
    /// Encoder output dim. Must equal each per-scale engine's
    /// `input_dim`.
    pub input_dim: usize,
    /// Which timescales to instantiate. Must be non-empty + sorted.
    pub timescales: Vec<Timescale>,
    /// Standard-deviation scale for the random-normal init of `B`.
    pub init_scale: f64,
}

impl Default for MultiTimescaleConfig {
    fn default() -> Self {
        Self {
            per_scale_dim: 16,
            input_dim: 16,
            timescales: Timescale::ALL.to_vec(),
            init_scale: 0.1,
        }
    }
}

impl MultiTimescaleConfig {
    /// Validate.
    pub fn validate(&self) -> Result<(), NssError> {
        if self.per_scale_dim == 0 {
            return Err(NssError::InvalidConfig {
                detail: "MultiTimescaleConfig.per_scale_dim must be >= 1".into(),
            });
        }
        if self.input_dim == 0 {
            return Err(NssError::InvalidConfig {
                detail: "MultiTimescaleConfig.input_dim must be >= 1".into(),
            });
        }
        if self.timescales.is_empty() {
            return Err(NssError::InvalidConfig {
                detail: "MultiTimescaleConfig.timescales must be non-empty".into(),
            });
        }
        let mut sorted = self.timescales.clone();
        sorted.sort();
        if sorted != self.timescales {
            return Err(NssError::InvalidConfig {
                detail: "MultiTimescaleConfig.timescales must be in canonical Ord order".into(),
            });
        }
        // Reject duplicates (would collide on RNG salt).
        for w in self.timescales.windows(2) {
            if w[0] == w[1] {
                return Err(NssError::InvalidConfig {
                    detail: format!("duplicate timescale {:?}", w[0]),
                });
            }
        }
        if !self.init_scale.is_finite() || self.init_scale <= 0.0 {
            return Err(NssError::InvalidConfig {
                detail: format!("init_scale must be > 0 and finite, got {}", self.init_scale),
            });
        }
        Ok(())
    }

    /// Total concatenated state dimensionality.
    pub fn total_state_dim(&self) -> usize {
        self.per_scale_dim * self.timescales.len()
    }

    /// Canonical bytes (used by `ClusterNssConfig::canonical_bytes`).
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(64);
        bytes.extend_from_slice(&(self.per_scale_dim as u64).to_le_bytes());
        bytes.extend_from_slice(&(self.input_dim as u64).to_le_bytes());
        bytes.extend_from_slice(&(self.timescales.len() as u64).to_le_bytes());
        for t in &self.timescales {
            bytes.extend_from_slice(t.label().as_bytes());
            bytes.push(b'|');
            bytes.extend_from_slice(&t.default_alpha().to_bits().to_le_bytes());
        }
        bytes.extend_from_slice(&self.init_scale.to_bits().to_le_bytes());
        bytes
    }
}

/// Parallel multi-buffer SSM engine. One [`TemporalStateEngine`] per
/// configured timescale, all sharing the same input dimensionality.
#[derive(Clone, Debug, PartialEq)]
pub struct MultiTimescaleEngine {
    cfg: MultiTimescaleConfig,
    engines: Vec<TemporalStateEngine>,
}

impl MultiTimescaleEngine {
    /// Build from a seed. Each timescale gets its own RNG substream
    /// (`temporal.short.A`, `temporal.medium.A`, etc.) so adding,
    /// removing, or reordering timescales doesn't perturb the others.
    pub fn from_seed(cfg: MultiTimescaleConfig, seed: NssSeed) -> Result<Self, NssError> {
        cfg.validate()?;
        let mut engines = Vec::with_capacity(cfg.timescales.len());
        for ts in &cfg.timescales {
            // Build a per-scale TemporalStateConfig with the timescale's
            // default α. The seed is the same master seed; the
            // *substream salt* differs per timescale because
            // `TemporalStateEngine::from_seed` uses `temporal.A` /
            // `temporal.B`. To get distinct streams we derive a
            // per-timescale child seed by mixing the label hash.
            //
            // Phase 3b approach: instead of relying on
            // `TemporalStateEngine`'s fixed salts, we construct a
            // *new* `NssSeed` per timescale using the substream
            // pattern. This keeps the cross-timescale independence
            // structural rather than implicit.
            let child_seed = derive_child_seed(seed, ts.label());
            let cfg_per = TemporalStateConfig {
                state_dim: cfg.per_scale_dim,
                input_dim: cfg.input_dim,
                alpha: ts.default_alpha(),
                init_scale: cfg.init_scale,
            };
            let eng = TemporalStateEngine::from_seed(cfg_per, child_seed)?;
            engines.push(eng);
        }
        Ok(Self { cfg, engines })
    }

    /// Borrow config.
    pub fn config(&self) -> &MultiTimescaleConfig {
        &self.cfg
    }

    /// Total concatenated state dimension.
    pub fn total_state_dim(&self) -> usize {
        self.cfg.total_state_dim()
    }

    /// Borrow individual engines (for tests + audit).
    pub fn engines(&self) -> &[TemporalStateEngine] {
        &self.engines
    }

    /// Zero state for the *concatenated* layout.
    pub fn zero_state_concatenated(&self) -> Vec<f64> {
        vec![0.0; self.cfg.total_state_dim()]
    }

    /// Zero state for the *per-timescale* layout: `Vec<Vec<f64>>`
    /// of length `timescales.len()`, each of `per_scale_dim`.
    pub fn zero_state_per_scale(&self) -> Vec<Vec<f64>> {
        self.cfg
            .timescales
            .iter()
            .map(|_| vec![0.0; self.cfg.per_scale_dim])
            .collect()
    }

    /// Step every timescale once and return the concatenated next
    /// state `[h_short | h_medium | h_long | h_structural]`.
    ///
    /// `h_prev` must have length `total_state_dim()`; the function
    /// slices it into per-scale chunks in `Timescale` order.
    pub fn step_concatenated(
        &self,
        h_prev: &[f64],
        z: &[f64],
    ) -> Result<Vec<f64>, NssError> {
        let total = self.cfg.total_state_dim();
        if h_prev.len() != total {
            return Err(NssError::InvalidState {
                detail: format!(
                    "h_prev length {} != total_state_dim {}",
                    h_prev.len(),
                    total
                ),
            });
        }
        if z.len() != self.cfg.input_dim {
            return Err(NssError::InvalidState {
                detail: format!("z length {} != input_dim {}", z.len(), self.cfg.input_dim),
            });
        }
        let per = self.cfg.per_scale_dim;
        let mut out = Vec::with_capacity(total);
        for (i, eng) in self.engines.iter().enumerate() {
            let slice = &h_prev[i * per..(i + 1) * per];
            let next = eng.step(slice, z)?;
            out.extend_from_slice(&next);
        }
        Ok(out)
    }

    /// Step every timescale and return per-scale states explicitly.
    pub fn step_per_scale(
        &self,
        h_prev: &[Vec<f64>],
        z: &[f64],
    ) -> Result<Vec<Vec<f64>>, NssError> {
        if h_prev.len() != self.cfg.timescales.len() {
            return Err(NssError::InvalidState {
                detail: format!(
                    "h_prev has {} buffers, expected {}",
                    h_prev.len(),
                    self.cfg.timescales.len()
                ),
            });
        }
        let mut out = Vec::with_capacity(self.cfg.timescales.len());
        for (eng, h) in self.engines.iter().zip(h_prev.iter()) {
            out.push(eng.step(h, z)?);
        }
        Ok(out)
    }

    /// Magnitude of each timescale's state vector — useful for
    /// inspecting which timescales are "lit up" in a given prediction.
    /// Returns `Vec<(Timescale, norm)>` in canonical order.
    pub fn state_magnitudes(&self, h: &[f64]) -> Vec<(Timescale, f64)> {
        let per = self.cfg.per_scale_dim;
        let mut out = Vec::with_capacity(self.cfg.timescales.len());
        for (i, ts) in self.cfg.timescales.iter().enumerate() {
            let slice = &h[i * per..(i + 1) * per];
            let mut acc = KahanAccumulatorF64::new();
            for v in slice {
                acc.add(v * v);
            }
            out.push((*ts, acc.finalize().sqrt()));
        }
        out
    }
}

/// Derive a per-timescale `NssSeed` so each TemporalStateEngine sees a
/// genuinely independent RNG stream (not just a different salt on the
/// same master).
fn derive_child_seed(parent: NssSeed, label: &str) -> NssSeed {
    let mut rng = parent.substream(&format!("multi_timescale.{}", label));
    NssSeed(rng.next_u64())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timescales_are_in_increasing_alpha_order() {
        // Short < Medium < Long < Structural in declaration order;
        // alphas must also be monotonic.
        let alphas: Vec<f64> = Timescale::ALL.iter().map(|t| t.default_alpha()).collect();
        for w in alphas.windows(2) {
            assert!(w[0] < w[1], "timescales must have monotonic alpha");
        }
    }

    #[test]
    fn half_life_grows_with_alpha() {
        let hls: Vec<f64> = Timescale::ALL.iter().map(|t| t.half_life_ticks()).collect();
        for w in hls.windows(2) {
            assert!(w[0] < w[1], "half-life must grow with alpha");
        }
        // Short half-life ~1, structural ~69.
        assert!(hls[0] < 2.0);
        assert!(hls[3] > 50.0);
    }

    #[test]
    fn config_default_validates() {
        assert!(MultiTimescaleConfig::default().validate().is_ok());
    }

    #[test]
    fn config_rejects_unsorted_timescales() {
        let cfg = MultiTimescaleConfig {
            timescales: vec![Timescale::Long, Timescale::Short],
            ..MultiTimescaleConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_rejects_duplicate_timescales() {
        let cfg = MultiTimescaleConfig {
            timescales: vec![Timescale::Short, Timescale::Short],
            ..MultiTimescaleConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn total_state_dim_matches_concatenation() {
        let cfg = MultiTimescaleConfig {
            per_scale_dim: 8,
            input_dim: 16,
            timescales: Timescale::ALL.to_vec(),
            init_scale: 0.1,
        };
        assert_eq!(cfg.total_state_dim(), 32); // 4 scales * 8 per-scale dim
    }

    #[test]
    fn determinism_same_seed_same_engine() {
        let cfg = MultiTimescaleConfig::default();
        let a = MultiTimescaleEngine::from_seed(cfg.clone(), NssSeed(42)).unwrap();
        let b = MultiTimescaleEngine::from_seed(cfg, NssSeed(42)).unwrap();
        // Engine vectors must be byte-equal (Eq for Vec<TemporalStateEngine>
        // is implemented via Vec's PartialEq + the engine's PartialEq).
        assert_eq!(a.engines().len(), b.engines().len());
        for (ea, eb) in a.engines().iter().zip(b.engines().iter()) {
            assert_eq!(ea.transition_matrix(), eb.transition_matrix());
        }
    }

    #[test]
    fn timescales_have_independent_rng_streams() {
        // The short and structural engines must have *different*
        // transition matrices even though they're seeded from the
        // same master seed — the per-timescale child seed derivation
        // ensures independence.
        let cfg = MultiTimescaleConfig::default();
        let eng = MultiTimescaleEngine::from_seed(cfg, NssSeed(42)).unwrap();
        let a_short = eng.engines()[0].transition_matrix();
        let a_struct = eng.engines()[3].transition_matrix();
        assert_ne!(a_short, a_struct);
    }

    #[test]
    fn concatenated_step_preserves_finiteness() {
        let cfg = MultiTimescaleConfig::default();
        let eng = MultiTimescaleEngine::from_seed(cfg, NssSeed(42)).unwrap();
        let mut h = eng.zero_state_concatenated();
        let z = vec![0.5; 16];
        for _ in 0..256 {
            h = eng.step_concatenated(&h, &z).unwrap();
            for v in &h {
                assert!(v.is_finite() && v.abs() <= 1.0);
            }
        }
    }

    #[test]
    fn impulse_decay_separates_timescales() {
        // Feed a single impulse, then zero input for many ticks.
        // After many ticks, the short-term state should be ~zero and
        // the structural state should still be nonzero. This is the
        // *defining property* of multi-timescale memory.
        let cfg = MultiTimescaleConfig {
            per_scale_dim: 4,
            input_dim: 4,
            timescales: Timescale::ALL.to_vec(),
            init_scale: 0.5,
        };
        let eng = MultiTimescaleEngine::from_seed(cfg.clone(), NssSeed(42)).unwrap();
        let impulse = vec![1.0; 4];
        let zero = vec![0.0; 4];
        let mut h = eng.zero_state_concatenated();
        // Apply impulse once.
        h = eng.step_concatenated(&h, &impulse).unwrap();
        // Then run forward for N ticks with zero input. The short
        // buffer's α=0.5 decays roughly 50% per tick; structural
        // α=0.99 decays ~1% per tick.
        for _ in 0..32 {
            h = eng.step_concatenated(&h, &zero).unwrap();
        }
        let mags = eng.state_magnitudes(&h);
        // mags is in [(Short, m0), (Medium, m1), (Long, m2), (Structural, m3)].
        let short_mag = mags[0].1;
        let structural_mag = mags[3].1;
        assert!(
            structural_mag > short_mag * 2.0,
            "structural magnitude must dominate after impulse decay (short={}, structural={})",
            short_mag,
            structural_mag,
        );
        // Short should be ~zero (well below 0.1 after 32 ticks of α=0.5 decay
        // starting from a tanh-bounded state).
        assert!(short_mag < 0.01, "short-term memory must decay fast, got {}", short_mag);
    }

    #[test]
    fn step_rejects_wrong_input_size() {
        let cfg = MultiTimescaleConfig::default();
        let eng = MultiTimescaleEngine::from_seed(cfg, NssSeed(42)).unwrap();
        let h = eng.zero_state_concatenated();
        let bad_z = vec![0.0; 99];
        assert!(eng.step_concatenated(&h, &bad_z).is_err());
        let bad_h = vec![0.0; 99];
        let good_z = vec![0.0; 16];
        assert!(eng.step_concatenated(&bad_h, &good_z).is_err());
    }

    #[test]
    fn per_scale_step_matches_concatenated_step() {
        let cfg = MultiTimescaleConfig::default();
        let eng = MultiTimescaleEngine::from_seed(cfg.clone(), NssSeed(42)).unwrap();
        let z = vec![0.1; 16];
        let h_concat = eng.zero_state_concatenated();
        let h_per_scale = eng.zero_state_per_scale();
        let next_concat = eng.step_concatenated(&h_concat, &z).unwrap();
        let next_per_scale = eng.step_per_scale(&h_per_scale, &z).unwrap();
        // Flatten per-scale and compare.
        let flat: Vec<f64> = next_per_scale.into_iter().flatten().collect();
        assert_eq!(flat, next_concat);
    }
}
