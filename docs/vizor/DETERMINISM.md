# Vizor Determinism Guarantees

## Principle

Vizor guarantees **bit-identical output** for identical input data, regardless
of execution backend (AST-eval or MIR-exec), interpreter seed, or platform.

## Implementation Details

### Float Formatting
All SVG coordinate values use `{:.2}` formatting (two decimal places). This
prevents platform-dependent float-to-string conversion differences.

### Ordering
- Data columns are processed in insertion order (Vec, not HashMap)
- Tick labels use the nice-ticks algorithm with deterministic rounding
- Annotations are rendered in the order they were added

### Rasterization (BMP)
- Line drawing uses Bresenham's algorithm (integer-only, no FP)
- Circle filling uses integer distance checks
- Built-in 6x10 bitmap font (95 ASCII glyphs, hardcoded glyph data)

### Color Palette
Fixed 8-color categorical palette (ggplot2-inspired), indexed with wrapping.
No random color generation.

### Testing

Three levels of determinism testing:

1. **Repeat runs**: Same source, multiple executions, assert identical SVG output
2. **Seed independence**: Different interpreter seeds produce identical plots
3. **Parity**: AST-eval and MIR-exec produce byte-identical SVG/BMP

Test files:
- `tests/test_vizor_determinism.rs` (7 tests)
- `tests/test_vizor_parity.rs` (5 tests)
