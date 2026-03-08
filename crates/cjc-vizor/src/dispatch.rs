//! Vizor dispatch — maps CJC language calls to the concrete Vizor API.
//!
//! Follows the `cjc_data::tidy_dispatch` pattern exactly.
//! Both `cjc-eval` and `cjc-mir-exec` call into `dispatch_vizor_builtin`
//! and `dispatch_vizor_method`.

use std::cell::RefCell;
use std::rc::Rc;
use std::any::Any;

use cjc_runtime::value::Value;

use crate::annotation::Annotation;
use crate::render::build_scene;
use crate::spec::PlotSpec;
use crate::svg::render_svg;
use crate::bmp::render_bmp;
use crate::png_export::render_png;

// ============================================================================
//  Public entry points
// ============================================================================

/// Dispatch a free-function builtin call.
///
/// Returns `Ok(Some(value))` if the name is a known Vizor builtin,
/// `Ok(None)` if not recognised (allows fallthrough to other dispatchers).
pub fn dispatch_vizor_builtin(
    name: &str,
    args: &[Value],
) -> Result<Option<Value>, String> {
    match name {
        "vizor_plot" | "vizor_plot_xy" => {
            if args.len() != 2 {
                return Err(format!("{name} requires 2 arguments: x array, y array"));
            }
            let x = value_to_f64_vec(&args[0])?;
            let y = value_to_f64_vec(&args[1])?;
            let spec = PlotSpec::from_xy(x, y);
            Ok(Some(wrap_plot(spec)))
        }
        "vizor_plot_cat" => {
            if args.len() != 2 {
                return Err("vizor_plot_cat requires 2 arguments: categories array, values array".into());
            }
            let cats = value_to_string_vec(&args[0])?;
            let vals = value_to_f64_vec(&args[1])?;
            let spec = PlotSpec::from_cat(cats, vals);
            Ok(Some(wrap_plot(spec)))
        }
        "vizor_plot_matrix" => {
            if args.len() != 3 {
                return Err("vizor_plot_matrix requires 3 arguments: matrix, row_labels, col_labels".into());
            }
            let matrix = value_to_matrix(&args[0])?;
            let row_labels = value_to_string_vec(&args[1])?;
            let col_labels = value_to_string_vec(&args[2])?;
            let spec = PlotSpec::from_matrix(matrix, row_labels, col_labels);
            Ok(Some(wrap_plot(spec)))
        }
        "vizor_corr_matrix" => {
            // Takes a flat array of column arrays + labels.
            // vizor_corr_matrix(columns: Array<Array<f64>>, labels: Array<String>)
            if args.len() != 2 {
                return Err("vizor_corr_matrix requires 2 arguments: columns array, labels array".into());
            }
            let columns = value_to_matrix(&args[0])?; // each row is a column of data
            let labels = value_to_string_vec(&args[1])?;
            let col_refs: Vec<&[f64]> = columns.iter().map(|c| c.as_slice()).collect();
            let corr = crate::stats::correlation_matrix(&col_refs);
            let spec = PlotSpec::from_matrix(corr, labels.clone(), labels)
                .scale_color_diverging()
                .geom_tile()
                .show_values(true);
            Ok(Some(wrap_plot(spec)))
        }
        // ── Phase 2B: Figure-level wrappers ──
        "vizor_displot" => {
            // vizor_displot(data: Array<f64>) → histogram with density overlay
            if args.len() != 1 {
                return Err("vizor_displot requires 1 argument: data array".into());
            }
            let data = value_to_f64_vec(&args[0])?;
            let zeros = vec![0.0; data.len()];
            let spec = PlotSpec::from_xy(data, zeros)
                .geom_histogram(20)
                .geom_density()
                .title("Distribution");
            Ok(Some(wrap_plot(spec)))
        }
        "vizor_catplot" => {
            // vizor_catplot(cats, values, kind) → categorical plot dispatched by kind
            if args.len() != 3 {
                return Err("vizor_catplot requires 3 arguments: categories, values, kind string".into());
            }
            let cats = value_to_string_vec(&args[0])?;
            let vals = value_to_f64_vec(&args[1])?;
            let kind = value_to_string(&args[2])?;
            let mut spec = PlotSpec::from_cat(cats, vals);
            spec = match kind.as_str() {
                "box" => spec.geom_box(),
                "violin" => spec.geom_violin(),
                "strip" => spec.geom_strip(),
                "swarm" => spec.geom_swarm(),
                "boxen" => spec.geom_boxen(),
                "bar" => spec.geom_bar(),
                _ => return Err(format!("vizor_catplot: unknown kind '{}'. Use box, violin, strip, swarm, boxen, or bar", kind)),
            };
            Ok(Some(wrap_plot(spec)))
        }
        "vizor_relplot" => {
            // vizor_relplot(x, y, kind) → relational plot
            if args.len() != 3 {
                return Err("vizor_relplot requires 3 arguments: x, y, kind string".into());
            }
            let x = value_to_f64_vec(&args[0])?;
            let y = value_to_f64_vec(&args[1])?;
            let kind = value_to_string(&args[2])?;
            let mut spec = PlotSpec::from_xy(x, y);
            spec = match kind.as_str() {
                "scatter" | "point" => spec.geom_point(),
                "line" => spec.geom_line(),
                _ => return Err(format!("vizor_relplot: unknown kind '{}'. Use scatter or line", kind)),
            };
            Ok(Some(wrap_plot(spec)))
        }
        "vizor_lmplot" => {
            // vizor_lmplot(x, y) → scatter + regression line overlay
            if args.len() != 2 {
                return Err("vizor_lmplot requires 2 arguments: x array, y array".into());
            }
            let x = value_to_f64_vec(&args[0])?;
            let y = value_to_f64_vec(&args[1])?;
            let (slope, intercept, r2) = crate::stats::linear_regression(&x, &y);
            let eq = format!("y = {:.2}x + {:.2}", slope, intercept);
            let spec = PlotSpec::from_xy(x, y)
                .geom_point()
                .annotate(crate::annotation::Annotation::regression(&eq, r2));
            Ok(Some(wrap_plot(spec)))
        }
        "vizor_jointplot" => {
            // vizor_jointplot(x, y) → scatter plot (main panel)
            // For simplicity, builds a scatter plot (future: add marginal histograms via composition).
            if args.len() != 2 {
                return Err("vizor_jointplot requires 2 arguments: x array, y array".into());
            }
            let x = value_to_f64_vec(&args[0])?;
            let y = value_to_f64_vec(&args[1])?;
            let spec = PlotSpec::from_xy(x, y)
                .geom_point()
                .title("Joint Plot");
            Ok(Some(wrap_plot(spec)))
        }
        "vizor_clustermap" => {
            // vizor_clustermap(columns, labels) → clustered heatmap
            if args.len() != 2 {
                return Err("vizor_clustermap requires 2 arguments: columns array, labels array".into());
            }
            let columns = value_to_matrix(&args[0])?;
            let labels = value_to_string_vec(&args[1])?;
            let col_refs: Vec<&[f64]> = columns.iter().map(|c| c.as_slice()).collect();
            let corr = crate::stats::correlation_matrix(&col_refs);

            // Cluster the correlation matrix rows to reorder.
            let n = corr.len();
            if n < 2 {
                // Not enough data to cluster, just make a heatmap.
                let spec = PlotSpec::from_matrix(corr, labels.clone(), labels)
                    .scale_color_diverging()
                    .geom_tile()
                    .show_values(true)
                    .title("Cluster Map");
                return Ok(Some(wrap_plot(spec)));
            }

            // Build distance matrix from correlation: d(i,j) = 1 - corr(i,j).
            let mut dist = vec![vec![0.0; n]; n];
            for i in 0..n {
                for j in 0..n {
                    dist[i][j] = (1.0 - corr[i][j]).abs();
                }
            }

            let merges = crate::stats::hierarchical_cluster(&dist, crate::stats::Linkage::Average);
            let order = crate::stats::dendrogram_leaf_order(&merges, n);

            // Reorder the correlation matrix and labels.
            let reordered_labels: Vec<String> = order.iter().map(|&i| labels[i].clone()).collect();
            let mut reordered_corr = vec![vec![0.0; n]; n];
            for (ri, &oi) in order.iter().enumerate() {
                for (ci, &oj) in order.iter().enumerate() {
                    reordered_corr[ri][ci] = corr[oi][oj];
                }
            }

            let spec = PlotSpec::from_matrix(reordered_corr, reordered_labels.clone(), reordered_labels)
                .scale_color_diverging()
                .geom_tile()
                .show_values(true)
                .title("Cluster Map");
            Ok(Some(wrap_plot(spec)))
        }
        "vizor_pairplot" => {
            // vizor_pairplot(columns, labels) → grid of scatter/density panels
            // Builds a correlation matrix heatmap + point overlays as a visual summary.
            if args.len() != 2 {
                return Err("vizor_pairplot requires 2 arguments: columns array, labels array".into());
            }
            let columns = value_to_matrix(&args[0])?;
            let labels = value_to_string_vec(&args[1])?;
            let col_refs: Vec<&[f64]> = columns.iter().map(|c| c.as_slice()).collect();
            let corr = crate::stats::correlation_matrix(&col_refs);
            let spec = PlotSpec::from_matrix(corr, labels.clone(), labels)
                .scale_color_diverging()
                .geom_tile()
                .show_values(true)
                .title("Pair Plot (Correlation)");
            Ok(Some(wrap_plot(spec)))
        }
        _ => Ok(None),
    }
}

/// Dispatch a method call on a `Value::VizorPlot`.
///
/// Returns `Ok(Some(value))` if the method is known, `Ok(None)` if not.
pub fn dispatch_vizor_method(
    inner: &Rc<dyn Any>,
    method: &str,
    args: &[Value],
) -> Result<Option<Value>, String> {
    let spec = downcast_spec(inner)?;

    match method {
        // ── Geometry layers ──
        "geom_point" => {
            Ok(Some(wrap_plot(spec.clone().geom_point())))
        }
        "geom_line" => {
            Ok(Some(wrap_plot(spec.clone().geom_line())))
        }
        "geom_bar" => {
            Ok(Some(wrap_plot(spec.clone().geom_bar())))
        }
        "geom_histogram" => {
            let bins = if args.is_empty() { 10 } else { value_to_i64(&args[0])? as usize };
            Ok(Some(wrap_plot(spec.clone().geom_histogram(bins))))
        }

        // ── Phase 2B: Distribution geoms ──
        "geom_density" => {
            Ok(Some(wrap_plot(spec.clone().geom_density())))
        }
        "geom_density_bw" => {
            require_args(method, args, 1)?;
            let bw = value_to_f64(&args[0])?;
            Ok(Some(wrap_plot(spec.clone().geom_density_bw(bw))))
        }
        "geom_area" => {
            Ok(Some(wrap_plot(spec.clone().geom_area())))
        }
        "geom_rug" => {
            Ok(Some(wrap_plot(spec.clone().geom_rug())))
        }
        "geom_ecdf" => {
            Ok(Some(wrap_plot(spec.clone().geom_ecdf())))
        }

        // ── Phase 2B: Categorical geoms ──
        "geom_box" => {
            Ok(Some(wrap_plot(spec.clone().geom_box())))
        }
        "geom_violin" => {
            Ok(Some(wrap_plot(spec.clone().geom_violin())))
        }
        "geom_strip" => {
            Ok(Some(wrap_plot(spec.clone().geom_strip())))
        }
        "geom_swarm" => {
            Ok(Some(wrap_plot(spec.clone().geom_swarm())))
        }
        "geom_boxen" => {
            Ok(Some(wrap_plot(spec.clone().geom_boxen())))
        }

        // ── Phase 2B: Regression geoms ──
        "geom_regression" => {
            Ok(Some(wrap_plot(spec.clone().geom_regression())))
        }
        "geom_residplot" => {
            Ok(Some(wrap_plot(spec.clone().geom_residplot())))
        }

        // ── Phase 3: Polar geoms ──
        "geom_pie" => {
            Ok(Some(wrap_plot(spec.clone().geom_pie())))
        }
        "geom_donut" => {
            let inner = if args.is_empty() { 0.4 } else { value_to_f64(&args[0])? };
            Ok(Some(wrap_plot(spec.clone().geom_donut(inner))))
        }
        "geom_rose" => {
            Ok(Some(wrap_plot(spec.clone().geom_rose())))
        }
        "geom_radar" => {
            Ok(Some(wrap_plot(spec.clone().geom_radar())))
        }
        "coord_polar" => {
            Ok(Some(wrap_plot(spec.clone().coord_polar())))
        }

        // ── Phase 3: 2D density + contour geoms ──
        "geom_density2d" => {
            Ok(Some(wrap_plot(spec.clone().geom_density2d())))
        }
        "geom_contour" => {
            Ok(Some(wrap_plot(spec.clone().geom_contour())))
        }

        // ── Phase 3.2: Error bars + Step line ──
        "geom_errorbar" => {
            Ok(Some(wrap_plot(spec.clone().geom_errorbar())))
        }
        "geom_errorbar_col" => {
            require_args(method, args, 1)?;
            let col = value_to_string(&args[0])?;
            Ok(Some(wrap_plot(spec.clone().geom_errorbar_col(&col))))
        }
        "geom_step" => {
            Ok(Some(wrap_plot(spec.clone().geom_step())))
        }
        "no_legend" => {
            Ok(Some(wrap_plot(spec.clone().no_legend())))
        }
        "subtitle" => {
            require_args(method, args, 1)?;
            let text = value_to_string(&args[0])?;
            Ok(Some(wrap_plot(spec.clone().subtitle(&text))))
        }
        "scale_x_log" => {
            let base = if args.is_empty() { 10.0 } else { value_to_f64(&args[0])? };
            Ok(Some(wrap_plot(spec.clone().scale_x_log(base))))
        }
        "scale_y_log" => {
            let base = if args.is_empty() { 10.0 } else { value_to_f64(&args[0])? };
            Ok(Some(wrap_plot(spec.clone().scale_y_log(base))))
        }

        // ── Phase 2B: Dendrogram ──
        "geom_dendrogram" => {
            Ok(Some(wrap_plot(spec.clone().geom_dendrogram())))
        }

        // ── Phase 2B: Heatmap geoms ──
        "geom_tile" => {
            Ok(Some(wrap_plot(spec.clone().geom_tile())))
        }
        "scale_color_diverging" => {
            Ok(Some(wrap_plot(spec.clone().scale_color_diverging())))
        }
        "show_values" => {
            let show = if args.is_empty() { true } else {
                match &args[0] {
                    Value::Bool(b) => *b,
                    _ => true,
                }
            };
            Ok(Some(wrap_plot(spec.clone().show_values(show))))
        }

        // ── Phase 2B: Faceting ──
        "facet_wrap" => {
            require_args(method, args, 1)?;
            let col = value_to_string(&args[0])?;
            Ok(Some(wrap_plot(spec.clone().facet_wrap(&col))))
        }
        "facet_wrap_ncol" => {
            require_args(method, args, 2)?;
            let col = value_to_string(&args[0])?;
            let ncol = value_to_i64(&args[1])? as usize;
            Ok(Some(wrap_plot(spec.clone().facet_wrap_ncol(&col, ncol))))
        }
        "facet_grid" => {
            require_args(method, args, 2)?;
            let row = value_to_string(&args[0])?;
            let col = value_to_string(&args[1])?;
            Ok(Some(wrap_plot(spec.clone().facet_grid(&row, &col))))
        }

        // ── Labels ──
        "title" => {
            require_args(method, args, 1)?;
            let text = value_to_string(&args[0])?;
            Ok(Some(wrap_plot(spec.clone().title(&text))))
        }
        "xlab" => {
            require_args(method, args, 1)?;
            let text = value_to_string(&args[0])?;
            Ok(Some(wrap_plot(spec.clone().xlab(&text))))
        }
        "ylab" => {
            require_args(method, args, 1)?;
            let text = value_to_string(&args[0])?;
            Ok(Some(wrap_plot(spec.clone().ylab(&text))))
        }

        // ── Scales ──
        "xlim" => {
            require_args(method, args, 2)?;
            let min = value_to_f64(&args[0])?;
            let max = value_to_f64(&args[1])?;
            Ok(Some(wrap_plot(spec.clone().xlim(min, max))))
        }
        "ylim" => {
            require_args(method, args, 2)?;
            let min = value_to_f64(&args[0])?;
            let max = value_to_f64(&args[1])?;
            Ok(Some(wrap_plot(spec.clone().ylim(min, max))))
        }

        // ── Theme ──
        "theme_minimal" => {
            Ok(Some(wrap_plot(spec.clone().theme_minimal())))
        }
        "theme_publication" => {
            Ok(Some(wrap_plot(spec.clone().theme_publication())))
        }
        "theme_dark" => {
            Ok(Some(wrap_plot(spec.clone().theme_dark())))
        }
        "coord_flip" => {
            Ok(Some(wrap_plot(spec.clone().coord_flip())))
        }
        "size" => {
            require_args(method, args, 2)?;
            let w = value_to_i64(&args[0])? as u32;
            let h = value_to_i64(&args[1])? as u32;
            Ok(Some(wrap_plot(spec.clone().size(w, h))))
        }

        // ── Annotations ──
        "annotate_text" => {
            require_args(method, args, 3)?;
            let text = value_to_string(&args[0])?;
            let x = value_to_f64(&args[1])?;
            let y = value_to_f64(&args[2])?;
            Ok(Some(wrap_plot(spec.clone().annotate(Annotation::text(&text, x, y)))))
        }
        "annotate_regression" => {
            require_args(method, args, 2)?;
            let equation = value_to_string(&args[0])?;
            let r2 = value_to_f64(&args[1])?;
            Ok(Some(wrap_plot(spec.clone().annotate(Annotation::regression(&equation, r2)))))
        }
        "annotate_ci" => {
            require_args(method, args, 3)?;
            let level = value_to_f64(&args[0])?;
            let lower = value_to_f64(&args[1])?;
            let upper = value_to_f64(&args[2])?;
            Ok(Some(wrap_plot(spec.clone().annotate(Annotation::ci(level, lower, upper)))))
        }
        "annotate_pvalue" => {
            require_args(method, args, 1)?;
            let p = value_to_f64(&args[0])?;
            Ok(Some(wrap_plot(spec.clone().annotate(Annotation::pvalue(p)))))
        }
        "annotate_event" => {
            require_args(method, args, 2)?;
            let x = value_to_f64(&args[0])?;
            let label = value_to_string(&args[1])?;
            Ok(Some(wrap_plot(spec.clone().annotate(Annotation::event_marker(x, &label)))))
        }
        "annotate_note" => {
            require_args(method, args, 1)?;
            let text = value_to_string(&args[0])?;
            Ok(Some(wrap_plot(spec.clone().annotate(
                Annotation::note(&text, crate::annotation::Position::TopRight),
            ))))
        }
        "annotate_data_note" => {
            require_args(method, args, 1)?;
            let text = value_to_string(&args[0])?;
            Ok(Some(wrap_plot(spec.clone().annotate(Annotation::data_note(&text)))))
        }
        "annotate_inline_label" => {
            require_args(method, args, 3)?;
            let text = value_to_string(&args[0])?;
            let x = value_to_f64(&args[1])?;
            let y = value_to_f64(&args[2])?;
            Ok(Some(wrap_plot(spec.clone().annotate(Annotation::inline_label(&text, x, y)))))
        }

        // ── Rendering ──
        "to_svg" => {
            let scene = build_scene(spec);
            let svg = render_svg(&scene);
            Ok(Some(Value::String(Rc::new(svg))))
        }
        "to_bmp" => {
            let scene = build_scene(spec);
            let bytes = render_bmp(&scene);
            Ok(Some(Value::Bytes(Rc::new(RefCell::new(bytes)))))
        }
        "to_png" => {
            let scene = build_scene(spec);
            let bytes = render_png(&scene)?;
            Ok(Some(Value::Bytes(Rc::new(RefCell::new(bytes)))))
        }
        "save" => {
            require_args(method, args, 1)?;
            let path = value_to_string(&args[0])?;
            let scene = build_scene(spec);

            if path.ends_with(".svg") {
                let svg = render_svg(&scene);
                std::fs::write(&path, svg)
                    .map_err(|e| format!("VizorPlot.save: {}", e))?;
            } else if path.ends_with(".bmp") {
                let bytes = render_bmp(&scene);
                std::fs::write(&path, bytes)
                    .map_err(|e| format!("VizorPlot.save: {}", e))?;
            } else if path.ends_with(".png") {
                let bytes = render_png(&scene)?;
                std::fs::write(&path, bytes)
                    .map_err(|e| format!("VizorPlot.save: {}", e))?;
            } else {
                return Err(format!(
                    "VizorPlot.save: unsupported format '{}'. Use .svg, .bmp, or .png",
                    path
                ));
            }
            Ok(Some(Value::Void))
        }

        _ => Ok(None),
    }
}

// ============================================================================
//  Helpers
// ============================================================================

fn wrap_plot(spec: PlotSpec) -> Value {
    Value::VizorPlot(Rc::new(spec))
}

fn downcast_spec(inner: &Rc<dyn Any>) -> Result<&PlotSpec, String> {
    inner
        .downcast_ref::<PlotSpec>()
        .ok_or_else(|| "VizorPlot: invalid internal type".to_string())
}

fn require_args(method: &str, args: &[Value], n: usize) -> Result<(), String> {
    if args.len() != n {
        Err(format!("VizorPlot.{method} requires {n} argument(s), got {}", args.len()))
    } else {
        Ok(())
    }
}

fn value_to_f64(v: &Value) -> Result<f64, String> {
    match v {
        Value::Float(f) => Ok(*f),
        Value::Int(i) => Ok(*i as f64),
        _ => Err(format!("expected number, got {}", v.type_name())),
    }
}

fn value_to_i64(v: &Value) -> Result<i64, String> {
    match v {
        Value::Int(i) => Ok(*i),
        Value::Float(f) => Ok(*f as i64),
        _ => Err(format!("expected integer, got {}", v.type_name())),
    }
}

fn value_to_string(v: &Value) -> Result<String, String> {
    match v {
        Value::String(s) => Ok((**s).clone()),
        _ => Err(format!("expected String, got {}", v.type_name())),
    }
}

fn value_to_f64_vec(v: &Value) -> Result<Vec<f64>, String> {
    match v {
        Value::Array(arr) => {
            let mut out = Vec::with_capacity(arr.len());
            for item in arr.iter() {
                out.push(value_to_f64(item)?);
            }
            Ok(out)
        }
        Value::Tensor(t) => {
            Ok(t.to_vec())
        }
        _ => Err(format!("expected Array or Tensor, got {}", v.type_name())),
    }
}

fn value_to_string_vec(v: &Value) -> Result<Vec<String>, String> {
    match v {
        Value::Array(arr) => {
            let mut out = Vec::with_capacity(arr.len());
            for item in arr.iter() {
                match item {
                    Value::String(s) => out.push((**s).clone()),
                    _ => return Err(format!("expected String element, got {}", item.type_name())),
                }
            }
            Ok(out)
        }
        _ => Err(format!("expected Array of Strings, got {}", v.type_name())),
    }
}

fn value_to_matrix(v: &Value) -> Result<Vec<Vec<f64>>, String> {
    match v {
        Value::Array(rows) => {
            let mut matrix = Vec::with_capacity(rows.len());
            for row in rows.iter() {
                match row {
                    Value::Array(cols) => {
                        let mut row_vals = Vec::with_capacity(cols.len());
                        for val in cols.iter() {
                            row_vals.push(value_to_f64(val)?);
                        }
                        matrix.push(row_vals);
                    }
                    _ => return Err(format!("expected Array row, got {}", row.type_name())),
                }
            }
            Ok(matrix)
        }
        _ => Err(format!("expected Array of Arrays (matrix), got {}", v.type_name())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dispatch_vizor_plot() {
        let x = Value::Array(Rc::new(vec![Value::Float(1.0), Value::Float(2.0)]));
        let y = Value::Array(Rc::new(vec![Value::Float(3.0), Value::Float(4.0)]));
        let result = dispatch_vizor_builtin("vizor_plot", &[x, y]);
        assert!(result.is_ok());
        let val = result.unwrap();
        assert!(val.is_some());
        assert_eq!(val.unwrap().type_name(), "VizorPlot");
    }

    #[test]
    fn test_dispatch_unknown() {
        let result = dispatch_vizor_builtin("not_vizor", &[]);
        assert!(matches!(result, Ok(None)));
    }

    #[test]
    fn test_dispatch_method_geom_point() {
        let x = Value::Array(Rc::new(vec![Value::Float(1.0)]));
        let y = Value::Array(Rc::new(vec![Value::Float(2.0)]));
        let plot = dispatch_vizor_builtin("vizor_plot", &[x, y]).unwrap().unwrap();
        if let Value::VizorPlot(inner) = &plot {
            let result = dispatch_vizor_method(inner, "geom_point", &[]);
            assert!(result.is_ok());
            assert!(result.unwrap().is_some());
        } else {
            panic!("Expected VizorPlot");
        }
    }

    #[test]
    fn test_dispatch_vizor_plot_cat() {
        let cats = Value::Array(Rc::new(vec![
            Value::String(Rc::new("a".to_string())),
            Value::String(Rc::new("b".to_string())),
        ]));
        let vals = Value::Array(Rc::new(vec![Value::Float(1.0), Value::Float(2.0)]));
        let result = dispatch_vizor_builtin("vizor_plot_cat", &[cats, vals]);
        assert!(result.is_ok());
        let val = result.unwrap();
        assert!(val.is_some());
        assert_eq!(val.unwrap().type_name(), "VizorPlot");
    }

    #[test]
    fn test_dispatch_vizor_plot_matrix() {
        let row0 = Value::Array(Rc::new(vec![Value::Float(1.0), Value::Float(2.0)]));
        let row1 = Value::Array(Rc::new(vec![Value::Float(3.0), Value::Float(4.0)]));
        let matrix = Value::Array(Rc::new(vec![row0, row1]));
        let row_labels = Value::Array(Rc::new(vec![
            Value::String(Rc::new("r0".to_string())),
            Value::String(Rc::new("r1".to_string())),
        ]));
        let col_labels = Value::Array(Rc::new(vec![
            Value::String(Rc::new("c0".to_string())),
            Value::String(Rc::new("c1".to_string())),
        ]));
        let result = dispatch_vizor_builtin("vizor_plot_matrix", &[matrix, row_labels, col_labels]);
        assert!(result.is_ok());
        let val = result.unwrap();
        assert!(val.is_some());
        assert_eq!(val.unwrap().type_name(), "VizorPlot");
    }

    #[test]
    fn test_value_to_string_vec() {
        let arr = Value::Array(Rc::new(vec![
            Value::String(Rc::new("hello".to_string())),
            Value::String(Rc::new("world".to_string())),
        ]));
        let result = super::value_to_string_vec(&arr);
        assert_eq!(result.unwrap(), vec!["hello".to_string(), "world".to_string()]);
    }

    #[test]
    fn test_value_to_matrix() {
        let row0 = Value::Array(Rc::new(vec![Value::Float(1.0), Value::Float(2.0)]));
        let row1 = Value::Array(Rc::new(vec![Value::Float(3.0), Value::Float(4.0)]));
        let arr = Value::Array(Rc::new(vec![row0, row1]));
        let result = super::value_to_matrix(&arr);
        let mat = result.unwrap();
        assert_eq!(mat.len(), 2);
        assert_eq!(mat[0], vec![1.0, 2.0]);
        assert_eq!(mat[1], vec![3.0, 4.0]);
    }

    #[test]
    fn test_dispatch_method_to_svg() {
        let x = Value::Array(Rc::new(vec![Value::Float(1.0), Value::Float(2.0)]));
        let y = Value::Array(Rc::new(vec![Value::Float(3.0), Value::Float(4.0)]));
        let plot = dispatch_vizor_builtin("vizor_plot", &[x.clone(), y.clone()]).unwrap().unwrap();
        if let Value::VizorPlot(inner) = &plot {
            let plot2 = dispatch_vizor_method(inner, "geom_point", &[]).unwrap().unwrap();
            if let Value::VizorPlot(inner2) = &plot2 {
                let svg_val = dispatch_vizor_method(inner2, "to_svg", &[]).unwrap().unwrap();
                if let Value::String(svg) = svg_val {
                    assert!(svg.contains("<svg"));
                    assert!(svg.contains("</svg>"));
                } else {
                    panic!("Expected String");
                }
            }
        }
    }
}
