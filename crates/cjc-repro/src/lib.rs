pub mod kahan;
pub use kahan::{KahanAccumulatorF32, KahanAccumulatorF64};

/// Deterministic RNG using SplitMix64.
/// Guarantees identical sequences for the same seed across all platforms.
#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn seeded(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Generate next u64 using SplitMix64.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    /// Generate f64 in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Generate f32 in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Standard normal via Box-Muller transform.
    pub fn next_normal_f64(&mut self) -> f64 {
        let u1 = self.next_f64();
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    pub fn next_normal_f32(&mut self) -> f32 {
        self.next_normal_f64() as f32
    }

    /// Fork the RNG for independent sub-streams (deterministic).
    pub fn fork(&mut self) -> Rng {
        Rng {
            state: self.next_u64(),
        }
    }
}

/// Kahan summation for stable floating-point reduction.
pub fn kahan_sum_f64(values: &[f64]) -> f64 {
    let mut sum = 0.0f64;
    let mut compensation = 0.0f64;
    for &val in values {
        let y = val - compensation;
        let t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
    }
    sum
}

pub fn kahan_sum_f32(values: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    let mut compensation = 0.0f32;
    for &val in values {
        let y = val - compensation;
        let t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
    }
    sum
}

/// Pairwise summation — another stable reduction.
pub fn pairwise_sum_f64(values: &[f64]) -> f64 {
    if values.len() <= 32 {
        return kahan_sum_f64(values);
    }
    let mid = values.len() / 2;
    pairwise_sum_f64(&values[..mid]) + pairwise_sum_f64(&values[mid..])
}

/// Reproducibility configuration.
#[derive(Debug, Clone)]
pub struct ReproConfig {
    pub enabled: bool,
    pub seed: u64,
}

impl ReproConfig {
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            seed: 0,
        }
    }

    pub fn enabled(seed: u64) -> Self {
        Self {
            enabled: true,
            seed,
        }
    }
}

impl Default for ReproConfig {
    fn default() -> Self {
        Self::disabled()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rng_deterministic() {
        let mut rng1 = Rng::seeded(42);
        let mut rng2 = Rng::seeded(42);

        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_rng_f64_range() {
        let mut rng = Rng::seeded(123);
        for _ in 0..1000 {
            let v = rng.next_f64();
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn test_rng_fork_deterministic() {
        let mut rng1 = Rng::seeded(42);
        let mut rng2 = Rng::seeded(42);

        let mut fork1 = rng1.fork();
        let mut fork2 = rng2.fork();

        for _ in 0..50 {
            assert_eq!(fork1.next_u64(), fork2.next_u64());
        }
    }

    #[test]
    fn test_kahan_sum() {
        // Sum of many small values where naive sum would lose precision
        let values: Vec<f64> = (0..10000).map(|_| 0.0001).collect();
        let result = kahan_sum_f64(&values);
        assert!((result - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_kahan_sum_f32() {
        let values: Vec<f32> = (0..10000).map(|_| 0.0001f32).collect();
        let result = kahan_sum_f32(&values);
        assert!((result - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_pairwise_sum() {
        let values: Vec<f64> = (0..10000).map(|_| 0.0001).collect();
        let result = pairwise_sum_f64(&values);
        assert!((result - 1.0).abs() < 1e-10);
    }
}
