# Vizor API Reference

## Prerequisites

All Vizor functions require `import vizor` at the top of your CJC file.

## Plot Construction

### `vizor_plot(x, y) -> VizorPlot`
Create a new plot from x and y data arrays.
```cjc
import vizor
let p = vizor_plot([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]);
```

### `vizor_plot_xy(x, y) -> VizorPlot`
Alias for `vizor_plot`.

## Geometry Layers

### `.geom_point() -> VizorPlot`
Add a scatter (point) layer. Renders as circles in SVG.

### `.geom_line() -> VizorPlot`
Add a line layer connecting points in order. Renders as polyline in SVG.

### `.geom_bar() -> VizorPlot`
Add a bar layer. Renders as rectangles in SVG.

### `.geom_histogram(bins: i64) -> VizorPlot`
Add a histogram layer with the specified number of bins.

## Labels

### `.title(text: String) -> VizorPlot`
Set the plot title (centered above plot area).

### `.xlab(text: String) -> VizorPlot`
Set the x-axis label.

### `.ylab(text: String) -> VizorPlot`
Set the y-axis label.

## Scales

### `.xlim(min: f64, max: f64) -> VizorPlot`
Override x-axis limits.

### `.ylim(min: f64, max: f64) -> VizorPlot`
Override y-axis limits.

## Theme & Layout

### `.theme_minimal() -> VizorPlot`
Apply the minimal theme (lighter gridlines, more whitespace).

### `.coord_flip() -> VizorPlot`
Flip x and y coordinates.

### `.size(width: i64, height: i64) -> VizorPlot`
Set plot dimensions in pixels (default: 800x600).

## Annotations

### `.annotate_text(text, x, y) -> VizorPlot`
Free-form text at data coordinates.

### `.annotate_regression(equation, r_squared) -> VizorPlot`
Regression summary with equation and R-squared value.

### `.annotate_ci(level, lower, upper) -> VizorPlot`
Confidence interval annotation.

### `.annotate_pvalue(value) -> VizorPlot`
P-value with smart formatting (shows significance stars).

### `.annotate_event(x, label) -> VizorPlot`
Vertical event marker at x-position.

### `.annotate_note(text) -> VizorPlot`
Corner note annotation.

### `.annotate_data_note(text) -> VizorPlot`
Data provenance note.

### `.annotate_inline_label(text, x, y) -> VizorPlot`
Label placed near a specific data point.

## Export

### `.to_svg() -> String`
Render to SVG string.

### `.to_bmp() -> Bytes`
Render to BMP byte buffer.

### `.save(path: String) -> Void`
Save to file. Format inferred from extension:
- `.svg` -- SVG format
- `.bmp` -- BMP format (uncompressed, zero deps)
- `.png` -- PNG format (requires `png` feature flag)
