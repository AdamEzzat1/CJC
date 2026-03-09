//! Statistical computations for Vizor plot types.
//!
//! All functions are fully deterministic: same input → same output.
//! No external dependencies. Pure Rust, no RNG.

use std::f64::consts::PI;

// ─── KDE ─────────────────────────────────────────────────────────────

/// Silverman's rule-of-thumb bandwidth for Gaussian KDE.
pub fn silverman_bandwidth(values: &[f64]) -> f64 {
    let n = values.len() as f64;
    if n < 2.0 {
        return 1.0;
    }
    let mean = values.iter().sum::<f64>() / n;
    let var = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let std = var.sqrt();
    if std < 1e-15 {
        return 1.0;
    }
    1.06 * std * n.powf(-0.2)
}

/// Gaussian kernel density estimation.
///
/// Returns `(x_grid, density_values)` of length `n_points`.
/// Grid spans from `min - 3*bw` to `max + 3*bw`.
pub fn kde(values: &[f64], n_points: usize) -> (Vec<f64>, Vec<f64>) {
    kde_bw(values, n_points, None)
}

/// KDE with optional bandwidth override.
pub fn kde_bw(values: &[f64], n_points: usize, bw_override: Option<f64>) -> (Vec<f64>, Vec<f64>) {
    if values.is_empty() {
        return (vec![], vec![]);
    }

    let bw = bw_override.unwrap_or_else(|| silverman_bandwidth(values));
    let n = values.len() as f64;

    // Compute data range.
    let mut vmin = f64::INFINITY;
    let mut vmax = f64::NEG_INFINITY;
    for &v in values {
        if v < vmin { vmin = v; }
        if v > vmax { vmax = v; }
    }

    let pad = 3.0 * bw;
    let lo = vmin - pad;
    let hi = vmax + pad;
    let step = if n_points > 1 { (hi - lo) / (n_points as f64 - 1.0) } else { 1.0 };

    let norm = 1.0 / (n * bw * (2.0 * PI).sqrt());
    let mut x_grid = Vec::with_capacity(n_points);
    let mut density = Vec::with_capacity(n_points);

    for i in 0..n_points {
        let x = lo + i as f64 * step;
        x_grid.push(x);
        let mut sum = 0.0;
        for &v in values {
            let u = (x - v) / bw;
            sum += (-0.5 * u * u).exp();
        }
        density.push(sum * norm);
    }

    (x_grid, density)
}

/// KDE with explicit bandwidth value.
pub fn kde_with_bandwidth(values: &[f64], n_points: usize, bw: f64) -> (Vec<f64>, Vec<f64>) {
    kde_bw(values, n_points, Some(bw))
}

// ─── ECDF ────────────────────────────────────────────────────────────

/// Empirical cumulative distribution function.
///
/// Returns `(sorted_values, cumulative_probabilities)`.
/// Each `cumulative_probabilities[i] = (i + 1) / n`.
pub fn ecdf(values: &[f64]) -> (Vec<f64>, Vec<f64>) {
    if values.is_empty() {
        return (vec![], vec![]);
    }
    let mut sorted: Vec<f64> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len() as f64;
    let probs: Vec<f64> = (1..=sorted.len()).map(|i| i as f64 / n).collect();
    (sorted, probs)
}

// ─── Quantiles ───────────────────────────────────────────────────────

/// Quantile via linear interpolation on sorted data.
/// `q` must be in `[0.0, 1.0]`.
pub fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let q = q.clamp(0.0, 1.0);
    let idx = q * (sorted.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = lo + 1;
    let frac = idx - lo as f64;
    if hi >= sorted.len() {
        sorted[lo]
    } else {
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

/// Five-number summary: (min, q1, median, q3, max).
pub fn five_number_summary(values: &[f64]) -> (f64, f64, f64, f64, f64) {
    let mut sorted: Vec<f64> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let min = sorted.first().copied().unwrap_or(0.0);
    let max = sorted.last().copied().unwrap_or(0.0);
    let q1 = quantile(&sorted, 0.25);
    let median = quantile(&sorted, 0.5);
    let q3 = quantile(&sorted, 0.75);
    (min, q1, median, q3, max)
}

// ─── Box Stats ───────────────────────────────────────────────────────

/// Box plot statistics with Tukey fences (1.5 × IQR).
#[derive(Debug, Clone)]
pub struct BoxStats {
    pub lower_whisker: f64,
    pub q1: f64,
    pub median: f64,
    pub q3: f64,
    pub upper_whisker: f64,
    pub outliers: Vec<f64>,
}

/// Compute box plot statistics from raw (unsorted) values.
pub fn box_stats(values: &[f64]) -> BoxStats {
    let mut sorted: Vec<f64> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let q1 = quantile(&sorted, 0.25);
    let median = quantile(&sorted, 0.5);
    let q3 = quantile(&sorted, 0.75);
    let iqr = q3 - q1;

    let lower_fence = q1 - 1.5 * iqr;
    let upper_fence = q3 + 1.5 * iqr;

    // Whiskers extend to the most extreme data point within the fences.
    let lower_whisker = sorted.iter().copied()
        .find(|&v| v >= lower_fence)
        .unwrap_or(q1);
    let upper_whisker = sorted.iter().rev().copied()
        .find(|&v| v <= upper_fence)
        .unwrap_or(q3);

    let outliers: Vec<f64> = sorted.iter().copied()
        .filter(|&v| v < lower_fence || v > upper_fence)
        .collect();

    BoxStats { lower_whisker, q1, median, q3, upper_whisker, outliers }
}

// ─── Letter-Value (Boxen) ────────────────────────────────────────────

/// Letter-value plot statistics.
///
/// Returns a series of `(lower, upper)` boundaries, from the outermost
/// (near the whiskers) inward toward the median. Each successive box
/// represents a halving of the data (letter values: M, F, E, D, ...).
pub fn letter_value_stats(values: &[f64]) -> Vec<(f64, f64)> {
    if values.is_empty() {
        return vec![];
    }
    let mut sorted: Vec<f64> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted.len();
    // Number of letter-value levels: floor(log2(n)), minimum 1.
    let k = ((n as f64).log2().floor() as usize).max(1);

    let mut levels = Vec::with_capacity(k);
    for i in (1..=k).rev() {
        let p = 0.5_f64.powi(i as i32);
        let lo = quantile(&sorted, p);
        let hi = quantile(&sorted, 1.0 - p);
        // Skip degenerate levels where lo ≈ hi (e.g., the median-only level).
        // The median is drawn as a separate line in the renderer.
        if (hi - lo).abs() < 1e-12 { continue; }
        levels.push((lo, hi));
    }
    levels
}

// ─── Jitter + Swarm ──────────────────────────────────────────────────

/// Deterministic jitter: spreads `n` points across `[-width, +width]`.
///
/// Uses an index-based deterministic sequence (no RNG). Points are
/// evenly spaced within the jitter band.
pub fn deterministic_jitter(n: usize, width: f64) -> Vec<f64> {
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![0.0];
    }
    // van der Corput sequence in base 2 for deterministic quasi-random spread.
    (0..n).map(|i| {
        let t = van_der_corput(i);
        (t * 2.0 - 1.0) * width
    }).collect()
}

/// Van der Corput sequence in base 2 (deterministic quasi-random in [0,1]).
fn van_der_corput(mut n: usize) -> f64 {
    let mut result = 0.0;
    let mut denom = 1.0;
    while n > 0 {
        denom *= 2.0;
        result += (n % 2) as f64 / denom;
        n /= 2;
    }
    result
}

/// Swarm layout: position `n` points without overlap.
///
/// Returns x-offsets for each point (in value-order). Uses a greedy
/// algorithm: sort by value, place each point at the closest available
/// position to center.
pub fn swarm_offsets(values: &[f64], point_radius: f64, max_width: f64) -> Vec<f64> {
    if values.is_empty() {
        return vec![];
    }
    let diameter = point_radius * 2.0;
    let n = values.len();

    // Sort indices by value.
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| values[a].partial_cmp(&values[b]).unwrap_or(std::cmp::Ordering::Equal));

    let mut offsets = vec![0.0; n];
    // Track placed positions grouped by "row" (y-bin).
    let mut placed: Vec<(f64, f64)> = Vec::new(); // (value, x_offset)

    for &idx in &indices {
        let val = values[idx];
        // Find the closest available x-offset that doesn't overlap with
        // any already-placed point at a similar y-value.
        let mut best_x = 0.0;
        let mut found = false;

        // Try offsets: 0, -1*d, +1*d, -2*d, +2*d, ...
        for step in 0..100 {
            let candidates = if step == 0 {
                vec![0.0]
            } else {
                vec![-(step as f64) * diameter, (step as f64) * diameter]
            };
            for &cx in &candidates {
                if cx.abs() > max_width {
                    continue;
                }
                let overlaps = placed.iter().any(|&(pv, px)| {
                    let dy = (val - pv).abs();
                    let dx = (cx - px).abs();
                    // Circle overlap test (approximate).
                    dy < diameter && dx < diameter
                });
                if !overlaps {
                    best_x = cx;
                    found = true;
                    break;
                }
            }
            if found {
                break;
            }
        }
        offsets[idx] = best_x;
        placed.push((val, best_x));
    }

    offsets
}

// ─── Linear Regression ───────────────────────────────────────────────

/// Simple linear regression: `y = slope * x + intercept`.
///
/// Returns `(slope, intercept, r_squared)`.
pub fn linear_regression(x: &[f64], y: &[f64]) -> (f64, f64, f64) {
    let n = x.len().min(y.len()) as f64;
    if n < 2.0 {
        return (0.0, 0.0, 0.0);
    }

    let sum_x: f64 = x.iter().take(n as usize).sum();
    let sum_y: f64 = y.iter().take(n as usize).sum();
    let mean_x = sum_x / n;
    let mean_y = sum_y / n;

    let mut ss_xx = 0.0;
    let mut ss_xy = 0.0;
    let mut ss_yy = 0.0;
    for i in 0..n as usize {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        ss_xx += dx * dx;
        ss_xy += dx * dy;
        ss_yy += dy * dy;
    }

    if ss_xx.abs() < 1e-15 {
        return (0.0, mean_y, 0.0);
    }

    let slope = ss_xy / ss_xx;
    let intercept = mean_y - slope * mean_x;
    let r_squared = if ss_yy.abs() < 1e-15 { 1.0 } else { (ss_xy * ss_xy) / (ss_xx * ss_yy) };

    (slope, intercept, r_squared)
}

/// Compute residuals from a linear fit.
pub fn residuals(x: &[f64], y: &[f64], slope: f64, intercept: f64) -> Vec<f64> {
    x.iter().zip(y.iter())
        .map(|(&xi, &yi)| yi - (slope * xi + intercept))
        .collect()
}

// ─── Correlation Matrix ──────────────────────────────────────────────

/// Pearson correlation matrix for a set of numeric columns.
///
/// Returns an `n x n` matrix where `n = columns.len()`.
pub fn correlation_matrix(columns: &[&[f64]]) -> Vec<Vec<f64>> {
    let n = columns.len();
    let mut result = vec![vec![0.0; n]; n];

    // Precompute means and stds.
    let means: Vec<f64> = columns.iter()
        .map(|c| c.iter().sum::<f64>() / c.len().max(1) as f64)
        .collect();
    let stds: Vec<f64> = columns.iter().enumerate()
        .map(|(i, c)| {
            let var = c.iter().map(|&v| (v - means[i]).powi(2)).sum::<f64>()
                / (c.len().max(1) - 1).max(1) as f64;
            var.sqrt()
        })
        .collect();

    for i in 0..n {
        for j in 0..n {
            if i == j {
                result[i][j] = 1.0;
            } else if j > i {
                let len = columns[i].len().min(columns[j].len());
                if len < 2 || stds[i] < 1e-15 || stds[j] < 1e-15 {
                    result[i][j] = 0.0;
                } else {
                    let cov: f64 = (0..len)
                        .map(|k| (columns[i][k] - means[i]) * (columns[j][k] - means[j]))
                        .sum::<f64>() / (len - 1) as f64;
                    result[i][j] = cov / (stds[i] * stds[j]);
                }
                result[j][i] = result[i][j];
            }
        }
    }

    result
}

// ─── Hierarchical Clustering ─────────────────────────────────────────

/// Linkage method for hierarchical clustering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Linkage {
    Single,
    Complete,
    Average,
}

/// A merge step in hierarchical clustering.
#[derive(Debug, Clone)]
pub struct MergeStep {
    pub left: usize,
    pub right: usize,
    pub distance: f64,
    pub size: usize,
}

/// Compute pairwise Euclidean distance matrix (rows = observations).
pub fn distance_matrix(data: &[&[f64]]) -> Vec<Vec<f64>> {
    let n = data.len();
    let mut dist = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d: f64 = data[i].iter().zip(data[j].iter())
                .map(|(&a, &b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            dist[i][j] = d;
            dist[j][i] = d;
        }
    }
    dist
}

/// Hierarchical agglomerative clustering.
///
/// Returns merge steps in order of increasing distance.
pub fn hierarchical_cluster(dist: &[Vec<f64>], linkage: Linkage) -> Vec<MergeStep> {
    let n = dist.len();
    if n < 2 {
        return vec![];
    }

    // Mutable copy of distance matrix.
    let mut d: Vec<Vec<f64>> = dist.to_vec();
    let mut active: Vec<bool> = vec![true; n];
    let mut sizes: Vec<usize> = vec![1; n];
    let mut merges = Vec::with_capacity(n - 1);

    for _ in 0..(n - 1) {
        // Find the closest pair of active clusters.
        let mut best_d = f64::INFINITY;
        let mut best_i = 0;
        let mut best_j = 0;
        for i in 0..n {
            if !active[i] { continue; }
            for j in (i + 1)..n {
                if !active[j] { continue; }
                if d[i][j] < best_d {
                    best_d = d[i][j];
                    best_i = i;
                    best_j = j;
                }
            }
        }

        let new_size = sizes[best_i] + sizes[best_j];
        merges.push(MergeStep {
            left: best_i,
            right: best_j,
            distance: best_d,
            size: new_size,
        });

        // Update distances: merge best_j into best_i.
        for k in 0..n {
            if !active[k] || k == best_i || k == best_j { continue; }
            let new_dist = match linkage {
                Linkage::Single => d[best_i][k].min(d[best_j][k]),
                Linkage::Complete => d[best_i][k].max(d[best_j][k]),
                Linkage::Average => {
                    (d[best_i][k] * sizes[best_i] as f64
                        + d[best_j][k] * sizes[best_j] as f64)
                        / new_size as f64
                }
            };
            d[best_i][k] = new_dist;
            d[k][best_i] = new_dist;
        }
        active[best_j] = false;
        sizes[best_i] = new_size;
    }

    merges
}

/// Compute leaf ordering from hierarchical clustering merge steps.
///
/// Returns indices in the order they appear when the dendrogram is
/// drawn (in-order traversal of the binary tree).
pub fn dendrogram_leaf_order(merges: &[MergeStep], n: usize) -> Vec<usize> {
    if n == 0 { return vec![]; }
    if merges.is_empty() { return (0..n).collect(); }

    // Build a tree: each merge creates a new "cluster node" indexed n+i.
    // children[node] = (left, right) for internal nodes.
    let mut children: Vec<Option<(usize, usize)>> = vec![None; n + merges.len()];
    let mut active_id: Vec<usize> = (0..n).collect();

    for (i, step) in merges.iter().enumerate() {
        let new_id = n + i;
        let left_id = active_id[step.left];
        let right_id = active_id[step.right];
        children[new_id] = Some((left_id, right_id));
        active_id[step.left] = new_id;
    }

    // In-order traversal of the tree rooted at the last merge.
    let root = n + merges.len() - 1;
    let mut order = Vec::with_capacity(n);
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        if let Some((left, right)) = children[node] {
            // Push right first so left is processed first (in-order).
            stack.push(right);
            stack.push(left);
        } else {
            // Leaf node.
            order.push(node);
        }
    }
    order
}

// ─── 2D Kernel Density Estimation ────────────────────────────────────

/// 2D Gaussian KDE on a regular grid.
///
/// Returns a `grid_size × grid_size` density matrix, along with the
/// x and y grid coordinates: `(x_grid, y_grid, density_matrix)`.
/// Uses independent Silverman bandwidths for x and y.
pub fn kde_2d(
    x: &[f64],
    y: &[f64],
    grid_size: usize,
) -> (Vec<f64>, Vec<f64>, Vec<Vec<f64>>) {
    let n = x.len().min(y.len());
    if n < 2 || grid_size == 0 {
        return (vec![], vec![], vec![]);
    }

    let bw_x = silverman_bandwidth(&x[..n]);
    let bw_y = silverman_bandwidth(&y[..n]);

    // Compute data ranges.
    let (x_min, x_max) = data_range_f64(&x[..n]);
    let (y_min, y_max) = data_range_f64(&y[..n]);

    let x_pad = 3.0 * bw_x;
    let y_pad = 3.0 * bw_y;
    let x_lo = x_min - x_pad;
    let x_hi = x_max + x_pad;
    let y_lo = y_min - y_pad;
    let y_hi = y_max + y_pad;

    let x_step = if grid_size > 1 { (x_hi - x_lo) / (grid_size - 1) as f64 } else { 1.0 };
    let y_step = if grid_size > 1 { (y_hi - y_lo) / (grid_size - 1) as f64 } else { 1.0 };

    let x_grid: Vec<f64> = (0..grid_size).map(|i| x_lo + i as f64 * x_step).collect();
    let y_grid: Vec<f64> = (0..grid_size).map(|j| y_lo + j as f64 * y_step).collect();

    let norm = 1.0 / (n as f64 * 2.0 * PI * bw_x * bw_y);
    let mut density = vec![vec![0.0; grid_size]; grid_size];

    for i in 0..grid_size {
        for j in 0..grid_size {
            let gx = x_grid[i];
            let gy = y_grid[j];
            let mut sum = 0.0;
            for k in 0..n {
                let ux = (gx - x[k]) / bw_x;
                let uy = (gy - y[k]) / bw_y;
                sum += (-0.5 * (ux * ux + uy * uy)).exp();
            }
            density[i][j] = sum * norm;
        }
    }

    (x_grid, y_grid, density)
}

/// Compute contour levels from a density matrix at evenly spaced percentiles.
///
/// Returns `n_levels` threshold values dividing the density range.
pub fn contour_levels(density: &[Vec<f64>], n_levels: usize) -> Vec<f64> {
    if n_levels == 0 || density.is_empty() {
        return vec![];
    }

    // Collect all non-zero density values.
    let mut vals: Vec<f64> = density.iter()
        .flat_map(|row| row.iter().copied())
        .filter(|&v| v > 0.0 && v.is_finite())
        .collect();

    if vals.is_empty() {
        return vec![0.0; n_levels];
    }

    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Evenly spaced percentiles from ~10% to ~90%.
    (0..n_levels)
        .map(|i| {
            let t = (i as f64 + 1.0) / (n_levels as f64 + 1.0);
            quantile(&vals, t)
        })
        .collect()
}

/// Compute (min, max) for a slice, skipping non-finite values.
fn data_range_f64(values: &[f64]) -> (f64, f64) {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for &v in values {
        if v.is_finite() {
            if v < min { min = v; }
            if v > max { max = v; }
        }
    }
    (min, max)
}

// ─── Categorical Grouping ────────────────────────────────────────────

/// Group y-values by categorical x-values.
///
/// Returns `(unique_categories, grouped_values)` where categories are
/// in first-seen order (deterministic given stable input).
pub fn group_by_category(categories: &[String], values: &[f64]) -> (Vec<String>, Vec<Vec<f64>>) {
    let mut seen: Vec<String> = Vec::new();
    let mut groups: Vec<Vec<f64>> = Vec::new();

    for (cat, &val) in categories.iter().zip(values.iter()) {
        if let Some(idx) = seen.iter().position(|s| s == cat) {
            groups[idx].push(val);
        } else {
            seen.push(cat.clone());
            groups.push(vec![val]);
        }
    }

    (seen, groups)
}

// ─── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_silverman_bandwidth_basic() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let bw = silverman_bandwidth(&values);
        assert!(bw > 0.0);
        assert!(bw < 5.0);
    }

    #[test]
    fn test_silverman_bandwidth_constant() {
        let values = vec![5.0, 5.0, 5.0, 5.0];
        let bw = silverman_bandwidth(&values);
        assert_eq!(bw, 1.0); // fallback for zero std
    }

    #[test]
    fn test_kde_basic() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (x, d) = kde(&values, 50);
        assert_eq!(x.len(), 50);
        assert_eq!(d.len(), 50);
        // Density should be positive.
        assert!(d.iter().all(|&v| v >= 0.0));
        // Peak should be near the data center.
        let peak_idx = d.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap().0;
        assert!((x[peak_idx] - 3.0).abs() < 2.0);
    }

    #[test]
    fn test_kde_deterministic() {
        let values = vec![1.0, 3.0, 5.0, 7.0, 9.0];
        let (x1, d1) = kde(&values, 100);
        let (x2, d2) = kde(&values, 100);
        assert_eq!(x1, x2);
        assert_eq!(d1, d2);
    }

    #[test]
    fn test_ecdf_basic() {
        let values = vec![3.0, 1.0, 2.0];
        let (sorted, probs) = ecdf(&values);
        assert_eq!(sorted, vec![1.0, 2.0, 3.0]);
        assert_eq!(probs.len(), 3);
        assert!((probs[0] - 1.0 / 3.0).abs() < 1e-10);
        assert!((probs[1] - 2.0 / 3.0).abs() < 1e-10);
        assert!((probs[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ecdf_deterministic() {
        let values = vec![5.0, 3.0, 1.0, 4.0, 2.0];
        let (s1, p1) = ecdf(&values);
        let (s2, p2) = ecdf(&values);
        assert_eq!(s1, s2);
        assert_eq!(p1, p2);
    }

    #[test]
    fn test_quantile_median() {
        let sorted = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((quantile(&sorted, 0.5) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_quantile_extremes() {
        let sorted = vec![10.0, 20.0, 30.0];
        assert!((quantile(&sorted, 0.0) - 10.0).abs() < 1e-10);
        assert!((quantile(&sorted, 1.0) - 30.0).abs() < 1e-10);
    }

    #[test]
    fn test_five_number_summary() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (min, q1, med, q3, max) = five_number_summary(&values);
        assert!((min - 1.0).abs() < 1e-10);
        assert!((max - 5.0).abs() < 1e-10);
        assert!((med - 3.0).abs() < 1e-10);
        assert!(q1 >= 1.0 && q1 <= 3.0);
        assert!(q3 >= 3.0 && q3 <= 5.0);
    }

    #[test]
    fn test_box_stats_no_outliers() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let bs = box_stats(&values);
        assert!(bs.outliers.is_empty());
        assert!((bs.median - 3.0).abs() < 1e-10);
        assert!(bs.lower_whisker >= 1.0);
        assert!(bs.upper_whisker <= 5.0);
    }

    #[test]
    fn test_box_stats_with_outliers() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0];
        let bs = box_stats(&values);
        assert!(bs.outliers.contains(&100.0));
    }

    #[test]
    fn test_letter_value_stats() {
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let lvs = letter_value_stats(&values);
        assert!(!lvs.is_empty());
        // Outermost level should span most of the data.
        assert!(lvs[0].0 < 10.0);
        assert!(lvs[0].1 > 90.0);
        // Innermost should be near the median.
        let last = lvs.last().unwrap();
        assert!((last.0 + last.1) / 2.0 > 30.0);
        assert!((last.0 + last.1) / 2.0 < 70.0);
    }

    #[test]
    fn test_deterministic_jitter() {
        let j1 = deterministic_jitter(10, 0.3);
        let j2 = deterministic_jitter(10, 0.3);
        assert_eq!(j1, j2);
        assert_eq!(j1.len(), 10);
        // All within bounds.
        assert!(j1.iter().all(|&v| v >= -0.3 && v <= 0.3));
    }

    #[test]
    fn test_swarm_offsets_no_overlap() {
        let values = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let offsets = swarm_offsets(&values, 3.0, 30.0);
        assert_eq!(offsets.len(), 5);
        // No two should have the same offset.
        for i in 0..offsets.len() {
            for j in (i + 1)..offsets.len() {
                assert!((offsets[i] - offsets[j]).abs() > 1.0);
            }
        }
    }

    #[test]
    fn test_linear_regression_perfect() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // y = 2x
        let (slope, intercept, r2) = linear_regression(&x, &y);
        assert!((slope - 2.0).abs() < 1e-10);
        assert!((intercept).abs() < 1e-10);
        assert!((r2 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_linear_regression_with_intercept() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![1.0, 3.0, 5.0, 7.0]; // y = 2x + 1
        let (slope, intercept, r2) = linear_regression(&x, &y);
        assert!((slope - 2.0).abs() < 1e-10);
        assert!((intercept - 1.0).abs() < 1e-10);
        assert!((r2 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_residuals() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![2.5, 4.0, 6.5]; // y ≈ 2x + 0.33...
        let (slope, intercept, _) = linear_regression(&x, &y);
        let r = residuals(&x, &y, slope, intercept);
        assert_eq!(r.len(), 3);
        // Residuals should sum to approximately 0.
        assert!(r.iter().sum::<f64>().abs() < 1e-10);
    }

    #[test]
    fn test_correlation_matrix_identity() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let cols: Vec<&[f64]> = vec![&a, &a];
        let corr = correlation_matrix(&cols);
        assert_eq!(corr.len(), 2);
        assert!((corr[0][0] - 1.0).abs() < 1e-10);
        assert!((corr[0][1] - 1.0).abs() < 1e-10);
        assert!((corr[1][0] - 1.0).abs() < 1e-10);
        assert!((corr[1][1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_correlation_matrix_negative() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let cols: Vec<&[f64]> = vec![&a, &b];
        let corr = correlation_matrix(&cols);
        assert!((corr[0][1] - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_group_by_category() {
        let cats = vec!["A".to_string(), "B".to_string(), "A".to_string(), "C".to_string(), "B".to_string()];
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (unique, groups) = group_by_category(&cats, &vals);
        assert_eq!(unique, vec!["A", "B", "C"]);
        assert_eq!(groups[0], vec![1.0, 3.0]); // A
        assert_eq!(groups[1], vec![2.0, 5.0]); // B
        assert_eq!(groups[2], vec![4.0]);       // C
    }

    #[test]
    fn test_hierarchical_cluster_basic() {
        // 3 points: (0), (1), (10) — should merge 0 and 1 first.
        let dist = vec![
            vec![0.0, 1.0, 10.0],
            vec![1.0, 0.0, 9.0],
            vec![10.0, 9.0, 0.0],
        ];
        let merges = hierarchical_cluster(&dist, Linkage::Single);
        assert_eq!(merges.len(), 2);
        assert_eq!(merges[0].left, 0);
        assert_eq!(merges[0].right, 1);
        assert!((merges[0].distance - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_dendrogram_leaf_order() {
        let dist = vec![
            vec![0.0, 1.0, 10.0],
            vec![1.0, 0.0, 9.0],
            vec![10.0, 9.0, 0.0],
        ];
        let merges = hierarchical_cluster(&dist, Linkage::Single);
        let order = dendrogram_leaf_order(&merges, 3);
        assert_eq!(order.len(), 3);
        // All original indices should appear.
        assert!(order.contains(&0));
        assert!(order.contains(&1));
        assert!(order.contains(&2));
    }

    #[test]
    fn test_van_der_corput() {
        assert!((van_der_corput(0) - 0.0).abs() < 1e-10);
        assert!((van_der_corput(1) - 0.5).abs() < 1e-10);
        assert!((van_der_corput(2) - 0.25).abs() < 1e-10);
        assert!((van_der_corput(3) - 0.75).abs() < 1e-10);
    }

    // ── Phase 5 (Audit): letter_value_stats degenerate level fix ──

    #[test]
    fn test_letter_value_stats_small_group() {
        // With only 5 identical values, the median level lo==hi should be skipped.
        let values = vec![3.0, 3.0, 3.0, 3.0, 3.0];
        let lvs = letter_value_stats(&values);
        // All levels should have lo < hi (no degenerate boxes).
        for &(lo, hi) in &lvs {
            assert!(hi - lo > 1e-12,
                "Degenerate level found: lo={}, hi={}", lo, hi);
        }
    }

    #[test]
    fn test_letter_value_stats_five_distinct() {
        // 5 distinct values: median is the middle one.
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let lvs = letter_value_stats(&values);
        // Levels with lo == hi == median should be filtered out.
        for &(lo, hi) in &lvs {
            assert!(hi - lo > 1e-12,
                "Degenerate level found: lo={}, hi={}", lo, hi);
        }
        // There should be at least one valid level (the full range).
        assert!(!lvs.is_empty());
        // Outermost level should span near the full range.
        assert!(lvs[0].0 <= 2.0);
        assert!(lvs[0].1 >= 4.0);
    }

    #[test]
    fn test_letter_value_stats_empty() {
        let lvs = letter_value_stats(&[]);
        assert!(lvs.is_empty());
    }

    #[test]
    fn test_letter_value_stats_single_value() {
        let lvs = letter_value_stats(&[42.0]);
        // A single value means all levels lo==hi, so all are filtered.
        for &(lo, hi) in &lvs {
            assert!(hi - lo > 1e-12,
                "Degenerate level found: lo={}, hi={}", lo, hi);
        }
    }

    #[test]
    fn test_letter_value_stats_deterministic() {
        let values: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let a = letter_value_stats(&values);
        let b = letter_value_stats(&values);
        assert_eq!(a, b, "letter_value_stats should be deterministic");
    }

    // ── Phase 5 (Audit): group_by_category preserves first-seen order ──

    #[test]
    fn test_group_by_category_first_seen_order() {
        let cats = vec!["Z".to_string(), "A".to_string(), "M".to_string(), "A".to_string(), "Z".to_string()];
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (unique, groups) = group_by_category(&cats, &vals);
        // First-seen order: Z, A, M (not alphabetically sorted).
        assert_eq!(unique, vec!["Z", "A", "M"]);
        assert_eq!(groups[0], vec![1.0, 5.0]); // Z
        assert_eq!(groups[1], vec![2.0, 4.0]); // A
        assert_eq!(groups[2], vec![3.0]);       // M
    }
}
