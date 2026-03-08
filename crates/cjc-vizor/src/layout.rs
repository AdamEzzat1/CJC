//! Layout engine — coordinate mapping, tick generation, axis computation.
//!
//! All operations are deterministic: same input → same output.

use crate::spec::{PlotSpec, Scale, CoordSystem};
use crate::text::format_tick;

/// Result of layout computation.
#[derive(Debug, Clone)]
pub struct LayoutResult {
    /// Plot area bounding box (in pixel coordinates).
    pub plot_x: f64,
    pub plot_y: f64,
    pub plot_w: f64,
    pub plot_h: f64,
    /// X-axis data range.
    pub x_min: f64,
    pub x_max: f64,
    /// Y-axis data range.
    pub y_min: f64,
    pub y_max: f64,
    /// Tick positions and labels.
    pub x_ticks: Vec<(f64, String)>,
    pub y_ticks: Vec<(f64, String)>,
    /// Whether x-axis is discrete.
    pub x_discrete: bool,
    /// Whether x-axis uses log scale.
    pub x_log: Option<f64>,
    /// Whether y-axis uses log scale.
    pub y_log: Option<f64>,
}

impl LayoutResult {
    /// Map a data x-value to pixel x-coordinate.
    /// Supports both linear and log scale mapping.
    pub fn map_x(&self, x: f64) -> f64 {
        if (self.x_max - self.x_min).abs() < 1e-15 {
            return self.plot_x + self.plot_w / 2.0;
        }
        if let Some(base) = self.x_log {
            // Log-scale mapping.
            let log_min = self.x_min.max(1e-15).log(base);
            let log_max = self.x_max.max(1e-15).log(base);
            let log_x = x.max(1e-15).log(base);
            if (log_max - log_min).abs() < 1e-15 {
                return self.plot_x + self.plot_w / 2.0;
            }
            self.plot_x + (log_x - log_min) / (log_max - log_min) * self.plot_w
        } else {
            self.plot_x + (x - self.x_min) / (self.x_max - self.x_min) * self.plot_w
        }
    }

    /// Map a data y-value to pixel y-coordinate (y increases downward).
    /// Supports both linear and log scale mapping.
    pub fn map_y(&self, y: f64) -> f64 {
        if (self.y_max - self.y_min).abs() < 1e-15 {
            return self.plot_y + self.plot_h / 2.0;
        }
        if let Some(base) = self.y_log {
            // Log-scale mapping.
            let log_min = self.y_min.max(1e-15).log(base);
            let log_max = self.y_max.max(1e-15).log(base);
            let log_y = y.max(1e-15).log(base);
            if (log_max - log_min).abs() < 1e-15 {
                return self.plot_y + self.plot_h / 2.0;
            }
            self.plot_y + self.plot_h - (log_y - log_min) / (log_max - log_min) * self.plot_h
        } else {
            self.plot_y + self.plot_h - (y - self.y_min) / (self.y_max - self.y_min) * self.plot_h
        }
    }
}

/// Compute layout from a plot spec.
pub fn compute_layout(spec: &PlotSpec) -> LayoutResult {
    let theme = &spec.theme;

    // Plot area.
    let plot_x = theme.margin_left;
    let plot_y = theme.margin_top;
    let plot_w = spec.width as f64 - theme.margin_left - theme.margin_right;
    let plot_h = spec.height as f64 - theme.margin_top - theme.margin_bottom;

    // Determine data ranges from all layers.
    let (mut x_min, mut x_max, mut y_min, mut y_max) = (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
    let mut x_discrete = false;

    for layer in &spec.layers {
        if let Some(ref x_col) = layer.aes.x {
            if let Some(col) = spec.data.get(x_col) {
                x_discrete = col.is_discrete();
                let vals = col.to_f64();
                for &v in &vals {
                    if v.is_finite() {
                        if v < x_min { x_min = v; }
                        if v > x_max { x_max = v; }
                    }
                }
            }
        }
        if let Some(ref y_col) = layer.aes.y {
            if let Some(col) = spec.data.get(y_col) {
                let vals = col.to_f64();
                for &v in &vals {
                    if v.is_finite() {
                        if v < y_min { y_min = v; }
                        if v > y_max { y_max = v; }
                    }
                }
            }
        }
    }

    // Handle histogram: compute bins and y range.
    for layer in &spec.layers {
        if layer.geom == crate::spec::Geom::Histogram {
            if let Some(ref x_col) = layer.aes.x {
                if let Some(col) = spec.data.get(x_col) {
                    let vals = col.to_f64();
                    let bins = layer.params.bins.unwrap_or(10);
                    let counts = histogram_counts(&vals, x_min, x_max, bins);
                    let max_count = counts.iter().copied().max().unwrap_or(0) as f64;
                    y_min = 0.0;
                    if max_count > y_max { y_max = max_count; }
                }
            }
        }
    }

    // Handle density: compute KDE and set y range.
    for layer in &spec.layers {
        if layer.geom == crate::spec::Geom::Density {
            if let Some(ref x_col_name) = layer.aes.x {
                if let Some(col) = spec.data.get(x_col_name) {
                    let vals = col.to_f64();
                    let n_pts = layer.params.n_grid_points.unwrap_or(200);
                    let (grid, density) = if let Some(bw) = layer.params.bandwidth {
                        crate::stats::kde_with_bandwidth(&vals, n_pts, bw)
                    } else {
                        crate::stats::kde(&vals, n_pts)
                    };
                    // X range: from KDE grid
                    if let Some(&gmin) = grid.first() {
                        if gmin < x_min { x_min = gmin; }
                    }
                    if let Some(&gmax) = grid.last() {
                        if gmax > x_max { x_max = gmax; }
                    }
                    // Y range: 0 to max density
                    y_min = 0.0;
                    for &d in &density {
                        if d > y_max { y_max = d; }
                    }
                }
            }
        }
    }

    // Handle ECDF: y range is always [0, 1].
    for layer in &spec.layers {
        if layer.geom == crate::spec::Geom::Ecdf {
            y_min = 0.0;
            if 1.0 > y_max { y_max = 1.0; }
        }
    }

    // Handle categorical geoms: compute y range from grouped data.
    let cat_geoms = [
        crate::spec::Geom::Box,
        crate::spec::Geom::Violin,
        crate::spec::Geom::Strip,
        crate::spec::Geom::Swarm,
        crate::spec::Geom::Boxen,
    ];
    for layer in &spec.layers {
        if cat_geoms.contains(&layer.geom) {
            // For categorical geoms, y-range comes from all y-values.
            if let Some(col) = spec.data.get("y") {
                let vals = col.to_f64();
                for &v in &vals {
                    if v.is_finite() {
                        if v < y_min { y_min = v; }
                        if v > y_max { y_max = v; }
                    }
                }
            }
        }
    }

    // Handle residual plots: y-range from residuals.
    for layer in &spec.layers {
        if layer.geom == crate::spec::Geom::Residual {
            if let (Some(x_data), Some(y_data)) = (spec.data.get("x"), spec.data.get("y")) {
                let xv = x_data.to_f64();
                let yv = y_data.to_f64();
                let n = xv.len().min(yv.len());
                if n >= 2 {
                    let (slope, intercept, _) = crate::stats::linear_regression(&xv[..n], &yv[..n]);
                    let resid = crate::stats::residuals(&xv[..n], &yv[..n], slope, intercept);
                    for &r in &resid {
                        if r.is_finite() {
                            if r < y_min { y_min = r; }
                            if r > y_max { y_max = r; }
                        }
                    }
                    // Ensure zero line is visible.
                    if 0.0 < y_min { y_min = 0.0; }
                    if 0.0 > y_max { y_max = 0.0; }
                }
            }
        }
    }

    // Handle dendrogram: y-range is [0, 1] (normalized distance).
    for layer in &spec.layers {
        if layer.geom == crate::spec::Geom::Dendrogram {
            y_min = 0.0;
            if 1.0 > y_max { y_max = 1.0; }
        }
    }

    // Determine if log scale is active.
    let x_log = if let Scale::Log { base } = spec.scales.x { Some(base) } else { None };
    let y_log = if let Scale::Log { base } = spec.scales.y { Some(base) } else { None };

    // Apply scale overrides.
    if let Scale::Linear { min: Some(m), .. } = spec.scales.x { x_min = m; }
    if let Scale::Linear { max: Some(m), .. } = spec.scales.x { x_max = m; }
    if let Scale::Linear { min: Some(m), .. } = spec.scales.y { y_min = m; }
    if let Scale::Linear { max: Some(m), .. } = spec.scales.y { y_max = m; }

    // For log scales, clamp min to positive values.
    if x_log.is_some() && x_min <= 0.0 { x_min = 1e-1; }
    if y_log.is_some() && y_min <= 0.0 { y_min = 1e-1; }

    // Handle degenerate ranges.
    if !x_min.is_finite() || !x_max.is_finite() { x_min = 0.0; x_max = 1.0; }
    if !y_min.is_finite() || !y_max.is_finite() { y_min = 0.0; y_max = 1.0; }
    if (x_max - x_min).abs() < 1e-15 { x_min -= 0.5; x_max += 0.5; }
    if (y_max - y_min).abs() < 1e-15 { y_min -= 0.5; y_max += 0.5; }

    // Add padding for bars.
    if x_discrete {
        x_min -= 0.5;
        x_max += 0.5;
    }

    // Generate ticks: log scale uses log_ticks(), linear uses nice_ticks().
    let x_ticks_raw: Vec<f64>;
    let y_ticks_raw: Vec<f64>;

    if let Some(base) = x_log {
        x_ticks_raw = log_ticks(x_min, x_max, base);
    } else {
        x_ticks_raw = nice_ticks(x_min, x_max, 7);
    }

    if let Some(base) = y_log {
        y_ticks_raw = log_ticks(y_min, y_max, base);
    } else {
        y_ticks_raw = nice_ticks(y_min, y_max, 7);
    }

    // Expand range to include tick endpoints.
    if let Some(&first) = x_ticks_raw.first() { if first < x_min { x_min = first; } }
    if let Some(&last) = x_ticks_raw.last() { if last > x_max { x_max = last; } }
    if let Some(&first) = y_ticks_raw.first() { if first < y_min { y_min = first; } }
    if let Some(&last) = y_ticks_raw.last() { if last > y_max { y_max = last; } }

    // Format tick labels.
    let x_ticks: Vec<(f64, String)> = if x_discrete {
        if let Some(col) = spec.data.get("x") {
            let labels = col.labels();
            labels.into_iter().enumerate()
                .map(|(i, l)| (i as f64, l))
                .collect()
        } else {
            x_ticks_raw.iter().map(|&v| (v, format_tick(v))).collect()
        }
    } else {
        x_ticks_raw.iter().map(|&v| (v, format_tick(v))).collect()
    };

    let y_ticks: Vec<(f64, String)> = y_ticks_raw.iter().map(|&v| (v, format_tick(v))).collect();

    // Apply coord flip if needed.
    if matches!(spec.coord, CoordSystem::FlipXY) {
        return LayoutResult {
            plot_x, plot_y, plot_w, plot_h,
            x_min: y_min, x_max: y_max,
            y_min: x_min, y_max: x_max,
            x_ticks: y_ticks,
            y_ticks: x_ticks,
            x_discrete: false,
            x_log: y_log,
            y_log: x_log,
        };
    }

    LayoutResult {
        plot_x, plot_y, plot_w, plot_h,
        x_min, x_max, y_min, y_max,
        x_ticks, y_ticks, x_discrete,
        x_log,
        y_log,
    }
}

/// Generate "nice" tick positions for a numeric axis.
///
/// Uses the Wilkinson / Heckbert "nice numbers" algorithm.
/// Fully deterministic: no floating-point-order-dependent operations.
pub fn nice_ticks(data_min: f64, data_max: f64, target_count: usize) -> Vec<f64> {
    if !data_min.is_finite() || !data_max.is_finite() || target_count == 0 {
        return vec![];
    }
    if (data_max - data_min).abs() < 1e-15 {
        return vec![data_min];
    }

    let range = data_max - data_min;
    let rough_step = range / target_count as f64;

    // Find the "nice" step size.
    let step = nice_step(rough_step);

    let start = (data_min / step).floor() * step;
    let end = (data_max / step).ceil() * step;

    let mut ticks = Vec::new();
    let mut v = start;
    // Safety bound: never generate more than 100 ticks.
    while v <= end + step * 0.001 && ticks.len() < 100 {
        ticks.push(v);
        v += step;
    }

    ticks
}

/// Round a step size to a "nice" number (1, 2, 5, 10, 20, 50, ...).
fn nice_step(rough: f64) -> f64 {
    let exp = rough.log10().floor();
    let frac = rough / 10.0_f64.powf(exp);

    let nice_frac = if frac <= 1.5 {
        1.0
    } else if frac <= 3.5 {
        2.0
    } else if frac <= 7.5 {
        5.0
    } else {
        10.0
    };

    nice_frac * 10.0_f64.powf(exp)
}

/// Generate tick positions for a logarithmic scale.
///
/// Places ticks at powers of the base (e.g., 1, 10, 100, 1000 for base 10).
/// Deterministic: same input → same output.
pub fn log_ticks(data_min: f64, data_max: f64, base: f64) -> Vec<f64> {
    if !data_min.is_finite() || !data_max.is_finite() || data_min <= 0.0 || data_max <= 0.0 || base <= 1.0 {
        return vec![];
    }

    let log_min = data_min.log(base).floor() as i32;
    let log_max = data_max.log(base).ceil() as i32;

    let mut ticks = Vec::new();
    for exp in log_min..=log_max {
        let val = base.powi(exp);
        if val >= data_min * 0.999 && val <= data_max * 1.001 && ticks.len() < 20 {
            ticks.push(val);
        }
    }

    // If too few ticks, add sub-decade ticks (2× and 5× for base 10).
    if ticks.len() < 3 && (base - 10.0).abs() < 1e-10 {
        let mut sub_ticks = Vec::new();
        for exp in (log_min - 1)..=log_max {
            for &mult in &[2.0, 5.0] {
                let val = mult * base.powi(exp);
                if val >= data_min * 0.999 && val <= data_max * 1.001 {
                    sub_ticks.push(val);
                }
            }
        }
        ticks.extend(sub_ticks);
        ticks.sort_by(|a, b| a.partial_cmp(b).unwrap());
        ticks.dedup_by(|a, b| (*a - *b).abs() < 1e-15);
    }

    ticks
}

/// Compute histogram bin counts. Deterministic.
pub fn histogram_counts(values: &[f64], min: f64, max: f64, bins: usize) -> Vec<usize> {
    if bins == 0 || values.is_empty() {
        return vec![];
    }
    let mut counts = vec![0usize; bins];
    let range = max - min;
    if range <= 0.0 {
        counts[0] = values.len();
        return counts;
    }
    for &v in values {
        if !v.is_finite() { continue; }
        let idx = ((v - min) / range * bins as f64).floor() as usize;
        let idx = idx.min(bins - 1);
        counts[idx] += 1;
    }
    counts
}

/// Compute data range (min, max) for f64 values, skipping NaN/Inf.
pub fn data_range(values: &[f64]) -> (f64, f64) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nice_ticks_basic() {
        let ticks = nice_ticks(0.0, 10.0, 5);
        assert!(!ticks.is_empty());
        assert!(ticks[0] <= 0.0);
        assert!(*ticks.last().unwrap() >= 10.0);
    }

    #[test]
    fn test_nice_ticks_deterministic() {
        let a = nice_ticks(0.0, 100.0, 7);
        let b = nice_ticks(0.0, 100.0, 7);
        assert_eq!(a, b);
    }

    #[test]
    fn test_nice_ticks_negative() {
        let ticks = nice_ticks(-5.0, 5.0, 5);
        assert!(!ticks.is_empty());
        assert!(ticks.iter().any(|&t| t < 0.0));
        assert!(ticks.iter().any(|&t| t > 0.0));
    }

    #[test]
    fn test_nice_ticks_degenerate() {
        let ticks = nice_ticks(5.0, 5.0, 5);
        assert_eq!(ticks, vec![5.0]);
    }

    #[test]
    fn test_histogram_counts() {
        let vals = vec![0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5];
        let counts = histogram_counts(&vals, 0.0, 10.0, 5);
        assert_eq!(counts.len(), 5);
        assert_eq!(counts.iter().sum::<usize>(), 10);
    }

    #[test]
    fn test_data_range() {
        let (min, max) = data_range(&[3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]);
        assert_eq!(min, 1.0);
        assert_eq!(max, 9.0);
    }

    #[test]
    fn test_data_range_with_nan() {
        let (min, max) = data_range(&[1.0, f64::NAN, 3.0, f64::INFINITY, 2.0]);
        assert_eq!(min, 1.0);
        assert_eq!(max, 3.0);
    }
}
