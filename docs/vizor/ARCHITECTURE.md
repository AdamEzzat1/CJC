# Vizor Library Architecture

## Overview

Vizor is CJC's built-in grammar-of-graphics visualization library. It provides
declarative plot construction with deterministic SVG and BMP output, following
CJC's zero-external-dependency philosophy for core functionality.

## Crate Structure

```
crates/cjc-vizor/src/
  lib.rs          -- Public API, CjcLibrary trait impl
  spec.rs         -- PlotSpec, PlotData, DataColumn, Layer, Geom types
  scene.rs        -- Scene graph (flat list of SceneElements)
  layout.rs       -- Coordinate mapping, tick generation, nice_ticks
  render.rs       -- PlotSpec -> Scene pipeline
  svg.rs          -- Scene -> SVG string serializer
  raster.rs       -- Scene -> PixelBuffer (Bresenham, bitmap font)
  bmp.rs          -- PixelBuffer -> BMP bytes (zero deps)
  png_export.rs   -- PixelBuffer -> PNG (behind feature flag)
  color.rs        -- Color type, 8-color palette
  theme.rs        -- Theme: margins, fonts, colors
  text.rs         -- Text measurement, tick formatting
  annotation.rs   -- Statistical annotations (p-value, R^2, CI, etc.)
  dispatch.rs     -- CJC runtime dispatch (builtin + method)
  docs.rs         -- IDE documentation metadata
```

## Integration Model

Vizor follows the **CjcLibrary** pattern established by `cjc-data`:

1. **Type-erased Value variant**: `Value::VizorPlot(Rc<dyn Any>)` holds a `PlotSpec`
2. **Import gating**: `import vizor` activates the library's builtins/methods
3. **Dispatch wiring**: Both `cjc-eval` and `cjc-mir-exec` call into
   `cjc_vizor::dispatch::{dispatch_vizor_builtin, dispatch_vizor_method}`
4. **Effect registry**: Vizor builtins registered in `cjc-types/src/effect_registry.rs`

## Pipeline

```
vizor_plot(x, y)  -->  PlotSpec (immutable builder)
  .geom_point()   -->  PlotSpec + Layer(Geom::Point)
  .title("...")    -->  PlotSpec + Labels.title
  .to_svg()        -->  compute_layout() -> build_scene() -> render_svg()
  .save("out.svg") -->  same pipeline + std::fs::write
```

## Determinism Guarantees

- All float formatting uses `{:.2}` (2 decimal places)
- BTreeMap for ordered iteration
- Bresenham integer line drawing (no floating-point rasterization)
- Nice-ticks algorithm produces identical output for identical input
- No RNG, no hash-order dependencies
- Both execution backends (AST-eval and MIR-exec) produce identical output

## Available Geoms

| Geom | Method | SVG Element |
|------|--------|-------------|
| Point (scatter) | `.geom_point()` | `<circle>` |
| Line | `.geom_line()` | `<polyline>` |
| Bar | `.geom_bar()` | `<rect>` |
| Histogram | `.geom_histogram(bins)` | `<rect>` |

## Annotation Types

- `annotate_text(text, x, y)` -- Free text at data coordinates
- `annotate_regression(eq, r2)` -- Regression equation + R^2
- `annotate_ci(level, lower, upper)` -- Confidence interval
- `annotate_pvalue(p)` -- P-value with smart formatting
- `annotate_event(x, label)` -- Vertical event marker
- `annotate_note(text)` -- Corner note
- `annotate_data_note(text)` -- Data provenance
- `annotate_inline_label(text, x, y)` -- Label near data point
