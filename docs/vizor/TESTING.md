# Vizor Testing Strategy

## Test Files

| File | Tests | Description |
|------|-------|-------------|
| `crates/cjc-vizor/` (unit) | 45 | Rust unit tests for all modules |
| `tests/test_vizor.rs` | 26 | Integration tests via CJC eval |
| `tests/test_vizor_parity.rs` | 5 | AST-eval vs MIR-exec parity |
| `tests/test_vizor_determinism.rs` | 7 | Reproducibility guarantees |
| **Total** | **83** | |

## Test Categories

### Unit Tests (45)
Located in each source module's `#[cfg(test)]` block:
- Color: palette wrapping, hex parsing, SVG output
- Theme: default/minimal constructors
- Text: tick formatting, text measurement
- Spec: builder pattern, data column operations
- Scene: element push, construction
- Layout: nice_ticks, coordinate mapping
- Render: scene building for all geom types
- SVG: serialization, text escaping, determinism
- Raster: pixel buffer, Bresenham line, bitmap font
- BMP: header structure, roundtrip
- Dispatch: builtin routing, method dispatch

### Integration Tests (26)
End-to-end tests through the CJC interpreter:
- Import gating (with/without `import vizor`)
- Plot construction and type checks
- All 4 geom types
- Builder method chaining
- SVG content validation (circles, polylines, rects, titles, labels)
- BMP export and header validation
- All annotation types in SVG output
- File save (.svg and .bmp)
- Full pipeline tests (scatter + bar with all features)

### Parity Tests (5)
Ensure AST-eval and MIR-exec produce identical results:
- Print output comparison
- SVG file comparison
- BMP byte-level comparison

### Determinism Tests (7)
- Repeated SVG generation (3 runs, assert equal)
- Line, bar, annotated plot determinism
- BMP determinism
- Float precision verification
- Seed independence
