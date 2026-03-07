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
