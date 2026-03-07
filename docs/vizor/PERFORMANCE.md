# Vizor Performance Characteristics

## Design Goals

Vizor prioritizes **correctness and determinism** over raw speed. Since CJC
targets scientific workflows where reproducibility is paramount, Vizor makes
deliberate trade-offs:

| Priority     | Approach                                    |
|-------------|---------------------------------------------|
| Correctness  | Immutable PlotSpec, clone-on-modify          |
| Determinism  | BTreeMap ordering, `{:.2}` float formatting  |
| Simplicity   | Zero external dependencies                   |
| Speed        | Acceptable for interactive use               |

## Render Pipeline Cost Model

```
PlotSpec (data + config)
    |
    v
compute_layout()     O(n) -- axis bounds, tick computation
    |
    v
build_scene()        O(n) -- one SceneElement per data point
    |
    v
render_svg()         O(n) -- string concatenation
  or
render_bmp()         O(w*h + n) -- pixel buffer + rasterization
```

### SVG output

- **Time:** Proportional to data points + annotation count
- **Memory:** The SVG string grows ~100-200 bytes per data point
- **Bottleneck:** String formatting of float coordinates

### BMP output

- **Time:** Dominated by pixel buffer allocation (width * height * 3 bytes)
- **Memory:** Fixed at `width * height * 3` for the RGB buffer
- **Bottleneck:** `fill_circle` and `draw_line` use Bresenham algorithms (integer math)

## Benchmark Guidelines

For a 1000-point scatter plot at default 800x600:
- SVG generation: ~1ms (string building)
- BMP generation: ~5-10ms (pixel fill + line drawing)
- File I/O: depends on disk

## Clone-on-Modify Cost

Each builder method (`title()`, `geom_point()`, etc.) clones the entire
`PlotSpec`. For typical usage (5-10 builder calls), the clone overhead is
negligible. If building hundreds of plots in a loop, consider constructing
the spec incrementally.

## Memory Layout

```
PlotSpec {
    x_data: Vec<f64>,       // 24 bytes + n * 8
    y_data: Vec<f64>,       // 24 bytes + n * 8
    geoms: Vec<Geom>,       // 24 bytes + g * enum_size
    annotations: Vec<Ann>,  // 24 bytes + a * enum_size
    title: Option<String>,  // 24 bytes
    ...                     // ~200 bytes fixed overhead
}
```

Total per plot: ~250 bytes fixed + 16 bytes per data point.

## Optimization Opportunities (Future)

1. **Arena allocation** for scene elements (avoid per-element heap alloc)
2. **SVG path batching** (single `<path>` for all circles of same style)
3. **Incremental BMP** (render only changed regions for animation)
4. **SIMD tick computation** (vectorize nice-number algorithm)

These are not implemented in v0 because the current performance is adequate
for CJC's target workloads (datasets up to ~10K points).
