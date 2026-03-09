# Vizor Rendering Trust Audit

**Date:** 2026-03-09
**Scope:** All 24 Vizor geom types + annotations, themes, color palette
**Result:** ALL 11 rendering issues identified and FIXED. Zero regressions.

---

## Phase Summary

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Repository Discovery | Done |
| 1 | Chart Quality Classification | Done |
| 2 | Rendering / Geometry Corrections | Done |
| 3 | Annotation Legibility Hardening | Done |
| 4 | Plot Polish Pass | Done |
| 5 | Testing Expansion | Done |
| 6 | Full Regression Run | Done |
| 7 | Documentation | Done |

---

## Issues Found & Fixed (Phase 2)

### 1. Categorical Axis Labels Missing (Box, Violin, Strip, Swarm, Boxen)

**Root cause:** `compute_layout()` used raw float data for x-tick generation instead of unique category names from discrete x-columns.

**Fix:** Added `unique_categories()` extraction from the x column (first-seen order), set x-range to `0..n_cats-1` with +/-0.5 padding, and generated categorical tick labels instead of numeric ones. Key constants: `CAT_GEOMS` (Box/Violin/Strip/Swarm/Boxen), `has_categorical_geom()`.

**Files:** `layout.rs`

### 2. Density2d Overflow Outside Plot Bounds

**Root cause:** Grid cell corners were not clipped to the plot area, producing `<rect>` elements extending beyond the viewport.

**Fix:** Rewrote density2d rendering to compute cell pixel corners from grid coordinates and clip each cell to `[plot_x, plot_y, plot_x+plot_w, plot_y+plot_h]`.

**Files:** `render.rs`

### 3. Violin Overflow (y extends below zero)

**Root cause:** KDE tails extended beyond the data range, mapping to pixel y-values above/below the plot bounds.

**Fix:** Added `.clamp(py_top, py_bottom)` to all violin polyline y-value computations.

**Files:** `render.rs`

### 4. Boxen Inner Box Height = 0

**Root cause:** `letter_value_stats()` produced degenerate levels where lo == hi (the median-only level with 5 data points).

**Fix:** Added `if (hi - lo).abs() < 1e-12 { continue; }` to skip zero-height levels.

**Files:** `stats.rs`

### 5. Error Bars Extending Outside Viewport

**Root cause:** `compute_layout()` y-range did not account for y +/- error values.

**Fix:** Added error bar y-range expansion loop that scans the error column and expands `y_min`/`y_max` to include `y - error` and `y + error`.

**Files:** `layout.rs`

### 6. Error Bars Missing Data Column

**Root cause:** No way to add an "error" data column to a plot spec from CJC code.

**Fix:** Added `PlotSpec::add_column()` builder method + dispatch wiring + docs + effect registry.

**Files:** `spec.rs`, `dispatch.rs`, `docs.rs`, `effect_registry.rs`

### 7. Polar Geoms Show Cartesian Axes

**Root cause:** Pie, Rose, and Radar charts rendered with standard Cartesian gridlines and axes.

**Fix:** Added `is_all_polar()` / `is_all_tile()` / `is_all_dendrogram()` checks in `build_single_panel()` to suppress gridlines and axes for these geom types.

**Files:** `render.rs`, `layout.rs`

### 8. Tile/Heatmap Shows Cartesian Axes

**Fix:** Same as #7 — `is_all_tile()` suppresses standard axes.

### 9. Dendrogram Shows Cartesian Axes + Overflow

**Root cause:** Dendrogram x-range was set from matrix column data instead of leaf positions. Axes were not suppressed.

**Fix:** Set dendrogram x-range to `[0, n_leaves-1]`, y-range to `[0, 1]`. Added `is_all_dendrogram()` axis suppression.

**Files:** `layout.rs`, `render.rs`

### 10. Residual Plot Y-Axis Contaminated by Raw Y Data

**Root cause:** `compute_layout()` scanned raw y-values for all geom layers, including residual plots which should only show residual magnitudes.

**Fix:** Added `has_residual_only` detection to skip raw y-data scan. Residual layers compute their own y-range from actual residuals, with 15% padding and zero-line visibility guaranteed.

**Files:** `layout.rs`

### 11. Annotation Overlaps

**Root cause:** Multiple annotations at the same named position (e.g., TopRight) stacked on top of each other.

**Fix:** Implemented annotation stacking with per-position y-offset tracking. Added `render_annotation_bg()` for semi-transparent background boxes behind all annotation text.

**Files:** `render.rs`

---

## Phase 3: Annotation Legibility Hardening

- Added `render_annotation_bg()` helper: draws semi-transparent rect behind annotation text
- All annotation types (Text, Note, PValue, Regression, CI, Event, Coordinate) receive background boxes
- Stacking offset tracked per named position (TopRight, TopLeft, BottomLeft, BottomRight)
- Background color: plot_background with 0.85 alpha

---

## Phase 4: Plot Polish Pass

### Theme Updates (`theme.rs`)
- `margin_top`: 40 -> 45, `margin_right`: 20 -> 25
- `plot_background`: White -> rgb(252, 252, 252) (subtle off-white)
- `axis_color`: DarkGray -> rgb(80, 80, 80)
- `grid_color`: LightGray -> rgb(230, 230, 230) (lighter for cleaner look)
- `text_color`: DarkGray -> rgb(50, 50, 50)
- `grid_line_width`: 0.5 -> 0.4

### Color Palette Updates (`color.rs`)
Replaced saturated ggplot2-style palette with muted ColorBrewer/Tableau-inspired colors:

| Index | Old (ggplot2) | New (Muted) |
|-------|--------------|-------------|
| 0 | rgb(228,26,28) Red | rgb(31,119,180) Blue |
| 1 | rgb(55,126,184) Blue | rgb(255,127,14) Orange |
| 2 | rgb(77,175,74) Green | rgb(44,160,44) Green |
| 3 | rgb(152,78,163) Purple | rgb(214,39,40) Red |
| 4 | rgb(255,127,0) Orange | rgb(148,103,189) Purple |
| 5 | rgb(255,255,51) Yellow | rgb(140,86,75) Brown |
| 6 | rgb(166,86,40) Brown | rgb(227,119,194) Pink |
| 7 | rgb(247,129,191) Pink | rgb(127,127,127) Gray |

---

## Phase 5: Testing Expansion

### New Unit Tests (44 new tests)

| Module | New Tests | Coverage |
|--------|-----------|----------|
| `layout.rs` | 14 | is_all_polar/tile/dendrogram, categorical layout, error bar y-range, residual isolation, dendrogram ranges, map_x/map_y |
| `render.rs` | 11 | Axis suppression, violin bounds, boxen small groups, density2d clipping, annotation bg, error bars |
| `stats.rs` | 6 | letter_value_stats degenerate levels, empty/single values, determinism, group_by_category order |
| `spec.rs` | 4 | add_column new/replace/preserve/chain |
| `theme.rs` | 5 | Updated defaults, colors, grid width, dark theme |
| `color.rs` | 5 | Palette values, 8 colors, distinctness, with_alpha, lerp |

### New Integration Tests (`test_vizor_audit.rs`, 35 tests)

| Category | Tests |
|----------|-------|
| Categorical axis labels | 5 (box, violin, strip, swarm, boxen) |
| Axis suppression | 3 (pie, rose, radar) |
| Dendrogram rendering | 1 |
| Density2d clipping | 1 |
| Boxen degenerate fix | 1 |
| Error bar + add_column | 3 |
| Residual y-axis | 1 |
| Annotation stacking | 3 |
| Theme defaults | 2 |
| Color palette | 1 |
| Determinism | 7 (box, violin, errorbar, boxen, dendrogram, residual, tile) |
| SVG well-formedness | 1 (no NaN/Infinity in 4 chart types) |
| Parity (AST vs MIR) | 6 (add_column, errorbar, boxen, residual, dendrogram, tile) |

---

## Phase 6: Full Regression Run

```
Total workspace tests: 4025 passed, 0 failed, 34 ignored
Vizor unit tests:       123 passed (up from 79)
Gallery generation:      80 SVGs generated successfully
```

**Zero regressions. All tests pass.**

---

## Chart Quality Matrix (Post-Audit)

| Chart Type | Rendering | Axes | Clipping | Labels | Determinism |
|------------|-----------|------|----------|--------|-------------|
| Point/Scatter | OK | OK | OK | OK | OK |
| Line | OK | OK | OK | OK | OK |
| Bar | OK | OK | OK | OK | OK |
| Histogram | OK | OK | OK | OK | OK |
| Density | OK | OK | OK | OK | OK |
| Area | OK | OK | OK | OK | OK |
| Rug | OK | OK | OK | OK | OK |
| ECDF | OK | OK | OK | OK | OK |
| Box | FIXED | FIXED | OK | FIXED | OK |
| Violin | FIXED | FIXED | FIXED | FIXED | OK |
| Strip | OK | FIXED | OK | FIXED | OK |
| Swarm | OK | FIXED | OK | FIXED | OK |
| Boxen | FIXED | FIXED | OK | FIXED | OK |
| Tile/Heatmap | OK | FIXED | OK | OK | OK |
| Regression Line | OK | OK | OK | OK | OK |
| Residual | OK | FIXED | OK | OK | OK |
| Dendrogram | OK | FIXED | FIXED | OK | OK |
| Pie | OK | FIXED | OK | OK | OK |
| Rose | OK | FIXED | OK | OK | OK |
| Radar | OK | FIXED | OK | OK | OK |
| Density2d | OK | OK | FIXED | OK | OK |
| Contour | OK | OK | OK | OK | OK |
| Error Bar | FIXED | OK | FIXED | OK | OK |
| Step | OK | OK | OK | OK | OK |

**All 24 geom types: rendering correct, axes correct, clipping correct, labels correct, deterministic.**

---

## Files Modified

| File | Changes |
|------|---------|
| `crates/cjc-vizor/src/layout.rs` | Categorical axis detection, unique_categories(), error bar y-range, residual isolation, dendrogram ranges, is_all_polar/tile/dendrogram helpers |
| `crates/cjc-vizor/src/render.rs` | Axis suppression, density2d clipping, violin clamping, annotation bg+stacking |
| `crates/cjc-vizor/src/stats.rs` | letter_value_stats degenerate level filter |
| `crates/cjc-vizor/src/spec.rs` | add_column() method |
| `crates/cjc-vizor/src/dispatch.rs` | add_column dispatch wiring |
| `crates/cjc-vizor/src/docs.rs` | add_column documentation |
| `crates/cjc-types/src/effect_registry.rs` | add_column effect registration |
| `crates/cjc-vizor/src/theme.rs` | Updated default theme aesthetics |
| `crates/cjc-vizor/src/color.rs` | Updated categorical palette to muted ColorBrewer |
| `gallery/generate_all.cjc` | Error bar examples with add_column |
| `tests/test_vizor_audit.rs` | 35 new integration tests |
