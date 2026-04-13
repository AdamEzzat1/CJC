---
title: Vizor
tags: [data, visualization]
status: Implemented
---

# Vizor

**Crate**: `cjc-vizor` — `crates/cjc-vizor/src/`.

**Docs**: `docs/vizor/ARCHITECTURE.md`, `DETERMINISM.md`, `API_REFERENCE.md`, `PERFORMANCE.md`, `QUICKSTART.md`, `TESTING.md`, `TOOLING.md`, `AUDIT.md`.

## Summary

A grammar-of-graphics visualization library, modeled on ggplot2. Takes a `PlotSpec` (declarative builder) and produces deterministic SVG or BMP output.

## Geometries supported

From the code survey:

- **Basic**: Point, Line, Bar, Histogram
- **Distribution**: Density, Area, Rug, Ecdf
- **Categorical**: Box, Violin, Strip, Swarm, Boxen
- **Matrix**: Tile (heatmap)
- **Regression**: RegressionLine, Residual, Dendrogram
- **Polar**: Pie, Rose, Radar
- **2D**: Density2d, Contour
- **Error**: ErrorBar, Step

README claims 80 chart types — the geoms above combine with stats, themes, and coordinate systems to produce that total.

## Coordinate systems

- Cartesian
- Flipped Cartesian
- Polar

## Builder API

```cjcl
vizor_plot(x, y)
    |> geom_point()
    |> title("My Plot")
    |> xlab("x axis")
    |> ylab("y axis")
    |> save("plot.svg")
```

Method names: `.geom_point()`, `.geom_line()`, `.geom_bar()`, `.geom_histogram()`, `.title()`, `.xlab()`, `.ylab()`, `.to_svg()`, `.to_bmp()`, `.save()`.

## Determinism

`docs/vizor/DETERMINISM.md` lays out the specific guarantees:
- Float formatting via `{:.2}` fixed-width.
- Bresenham integer rasterization (no floating-point scanline).
- Fixed 8-color palette.
- `BTreeMap` for ordered iteration.
- Concrete rolling/prefix ops for stat layers.

Tests verify: repeat runs, seed independence, and [[cjc-eval]] vs [[cjc-mir-exec]] parity at the *byte* level of the SVG/BMP output.

## Gallery

`gallery/` in the repo contains 80 Vizor-generated SVG files — one per chart type. Used as golden regression fixtures.

## Related

- [[DataFrame DSL]]
- [[Determinism Contract]]
- [[Binary Serialization]]
