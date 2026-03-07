# Vizor Quick-Start Guide

## Overview

Vizor is CJC's built-in grammar-of-graphics visualization library. It produces
deterministic SVG and BMP output with zero external dependencies.

## Basic Usage

### 1. Import the library

```cjc
import vizor
```

> **Note:** `import vizor` has no semicolon. All Vizor functions require this
> import or you will get an "unknown function" error.

### 2. Create a plot from data

```cjc
let x = [1.0, 2.0, 3.0, 4.0, 5.0]
let y = [2.1, 4.0, 5.9, 8.1, 10.0]
let p = vizor_plot_xy(x, y)
```

### 3. Add a geometry layer

```cjc
let p = p.geom_point()    // scatter plot
// or
let p = p.geom_line()     // line chart
// or
let p = p.geom_bar()      // bar chart
```

### 4. Customize with builder methods

```cjc
let p = p.title("My Plot")
let p = p.xlab("Horizontal")
let p = p.ylab("Vertical")
let p = p.theme_minimal()
let p = p.size(800.0, 600.0)
```

### 5. Save to file

```cjc
p.save("output.svg")   // SVG (default for .svg extension)
p.save("output.bmp")   // BMP raster (default for .bmp extension)
```

## Complete Example

```cjc
import vizor

let temps = [15.0, 18.0, 22.0, 28.0, 33.0, 35.0, 34.0, 30.0, 25.0, 19.0, 15.0, 12.0]
let months = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]

let p = vizor_plot_xy(months, temps)
let p = p.geom_line()
let p = p.geom_point()
let p = p.title("Monthly Temperature")
let p = p.xlab("Month")
let p = p.ylab("Temp (C)")
let p = p.theme_minimal()

p.save("temperature.svg")
```

## API At a Glance

| Function                | Description                      |
|------------------------|----------------------------------|
| `vizor_plot_xy(x, y)`  | Create plot from x/y arrays      |
| `.geom_point()`        | Add scatter points               |
| `.geom_line()`         | Add connected line               |
| `.geom_bar()`          | Add vertical bars                |
| `.geom_histogram()`    | Add histogram bins               |
| `.title(s)`            | Set plot title                   |
| `.xlab(s)` / `.ylab(s)`| Set axis labels                 |
| `.xlim(lo, hi)`        | Set x-axis range                 |
| `.ylim(lo, hi)`        | Set y-axis range                 |
| `.size(w, h)`          | Set canvas dimensions            |
| `.theme_minimal()`     | Use minimal theme                |
| `.coord_flip()`        | Swap x and y axes                |
| `.save(path)`          | Write SVG or BMP to disk         |

## Annotations

```cjc
let p = p.annotate_text("Label", 3.0, 7.0)
let p = p.annotate_regression("y=2x+1", 0.98)
let p = p.annotate_ci(0.95, 4.2, 6.8)
let p = p.annotate_pvalue(0.003)
```

## In-memory export

```cjc
let svg_string = p.to_svg()   // returns SVG as string
let bmp_bytes  = p.to_bmp()   // returns BMP as byte array
```
