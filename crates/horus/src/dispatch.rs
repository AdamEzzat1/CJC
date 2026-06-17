//! Language-level dispatch for `seshat_*` builtins — the thin, **write-only**
//! `.cjcl`-facing surface that lets a CJC-Lang program annotate its own
//! execution into a Seshat trace.
//!
//! Routed from both `cjc-eval` and `cjc-mir-exec` *after* the shared
//! `cjc_runtime::dispatch_builtin` and the other satellites
//! (`dispatch_quantum`, `dispatch_grad_graph`, `dispatch_abng`, `dispatch_locke`)
//! — the exact precedent established by those crates, which keeps
//! `cjc-runtime → horus` from becoming a dependency cycle.
//!
//! # Determinism & the write-only rule
//!
//! Every builtin here records events into a per-thread sink and returns a
//! *deterministic* value (a sequential zone handle, an event count, or 0). None
//! returns trace-derived analysis back into program state, so the builtins
//! cannot perturb program control flow — the same guarantee the existing
//! `profile_zone_*` builtins provide. Zone handles come from the builder's
//! monotonic counter, so AST-eval and MIR-exec observe identical handles
//! (parity).

use std::cell::RefCell;

use cjc_runtime::value::Value;

use polytrace::trace::{FrameKind, OwnershipDomain, Trace, TraceBuilder};

thread_local! {
    /// The per-thread trace sink. Reset by `seshat_reset()`.
    static SINK: RefCell<TraceBuilder> = RefCell::new(TraceBuilder::new(0));
}

/// Reset the per-thread sink. Tests/`.cjcl` programs should call this at the top
/// of a recording, since thread-locals can outlive a single test function when
/// cargo runs tests on a thread pool.
pub fn reset() {
    SINK.with(|c| *c.borrow_mut() = TraceBuilder::new(0));
}

/// Run a closure with mutable access to the sink.
pub fn with_sink<R>(f: impl FnOnce(&mut TraceBuilder) -> R) -> R {
    SINK.with(|c| f(&mut c.borrow_mut()))
}

/// Snapshot the current sink as an immutable [`Trace`] without resetting it.
pub fn snapshot() -> Trace {
    SINK.with(|c| c.borrow().clone().finish())
}

// ─── arg helpers (mirror cjc-ad / cjc-quantum boundary hardening) ───────────

fn want(name: &str, args: &[Value], n: usize) -> Result<(), String> {
    if args.len() != n {
        Err(format!("{name}: expected {n} arguments, got {}", args.len()))
    } else {
        Ok(())
    }
}

fn as_int(name: &str, v: &Value) -> Result<i64, String> {
    match v {
        Value::Int(i) => Ok(*i),
        other => Err(format!("{name}: expected Int, got {}", type_name(other))),
    }
}

fn as_str(name: &str, v: &Value) -> Result<String, String> {
    match v {
        Value::String(s) => Ok(s.as_str().to_string()),
        other => Err(format!("{name}: expected String, got {}", type_name(other))),
    }
}

fn nonneg_bytes(name: &str, v: &Value) -> Result<u64, String> {
    let i = as_int(name, v)?;
    if i < 0 {
        Err(format!("{name}: byte count must be non-negative, got {i}"))
    } else {
        Ok(i as u64)
    }
}

fn type_name(v: &Value) -> &'static str {
    match v {
        Value::Int(_) => "Int",
        Value::Float(_) => "Float",
        Value::Bool(_) => "Bool",
        Value::String(_) => "String",
        _ => "<other>",
    }
}

/// Dispatch a `seshat_*` builtin. Returns `Ok(None)` if `name` is not a Seshat
/// builtin (so the caller falls through to the next dispatcher), `Ok(Some(v))`
/// on success, `Err` on a malformed call — never panics.
pub fn dispatch_seshat(name: &str, args: &[Value]) -> Result<Option<Value>, String> {
    if !name.starts_with("seshat_") {
        return Ok(None);
    }
    let v = match name {
        "seshat_reset" => {
            want(name, args, 0)?;
            reset();
            Value::Int(0)
        }
        "seshat_zone_start" => {
            want(name, args, 1)?;
            let zone = as_str(name, &args[0])?;
            let handle = with_sink(|b| b.zone_start(&zone));
            Value::Int(handle as i64)
        }
        "seshat_zone_stop" => {
            want(name, args, 1)?;
            let handle = as_int(name, &args[0])?;
            if handle < 0 {
                return Err(format!("{name}: handle must be non-negative, got {handle}"));
            }
            with_sink(|b| b.zone_stop(handle as u64));
            Value::Int(0)
        }
        "seshat_mark_boundary" => {
            want(name, args, 1)?;
            let bname = as_str(name, &args[0])?;
            with_sink(|b| {
                let f = b.intern_frame(FrameKind::FfiBoundary, &bname, "<cjcl>", 0);
                b.boundary_cross(f);
            });
            Value::Int(0)
        }
        "seshat_mark_copy" => {
            want(name, args, 3)?;
            let from = OwnershipDomain::from_str(&as_str(name, &args[0])?)
                .ok_or_else(|| format!("{name}: unknown source domain"))?;
            let to = OwnershipDomain::from_str(&as_str(name, &args[1])?)
                .ok_or_else(|| format!("{name}: unknown destination domain"))?;
            let bytes = nonneg_bytes(name, &args[2])?;
            with_sink(|b| {
                let f = b.intern_frame(FrameKind::Rust, "seshat_marker", "<cjcl>", 0);
                b.copy(from, to, bytes, f);
            });
            Value::Int(0)
        }
        "seshat_alloc_tag" => {
            want(name, args, 2)?;
            let domain = OwnershipDomain::from_str(&as_str(name, &args[0])?)
                .ok_or_else(|| format!("{name}: unknown ownership domain"))?;
            let bytes = nonneg_bytes(name, &args[1])?;
            with_sink(|b| {
                let f = b.intern_frame(FrameKind::Rust, "seshat_marker", "<cjcl>", 0);
                b.alloc(domain, bytes, f);
            });
            Value::Int(0)
        }
        "seshat_event_count" => {
            want(name, args, 0)?;
            let n = with_sink(|b| b.event_count());
            Value::Int(n as i64)
        }
        "seshat_dump_trace" => {
            want(name, args, 1)?;
            let path = as_str(name, &args[0])?;
            let trace = snapshot();
            let bytes = polytrace::serialize::serialize(&trace);
            std::fs::write(&path, &bytes)
                .map_err(|e| format!("{name}: failed to write '{path}': {e}"))?;
            // Return the (deterministic) event count, not the path/byte count, so
            // both executors agree regardless of filesystem.
            Value::Int(trace.num_events() as i64)
        }
        _ => return Err(format!("unknown seshat builtin: {name}")),
    };
    Ok(Some(v))
}

/// The set of names this dispatcher claims, for the executors' `is_known_builtin`
/// gates.
pub const SESHAT_BUILTINS: &[&str] = &[
    "seshat_reset",
    "seshat_zone_start",
    "seshat_zone_stop",
    "seshat_mark_boundary",
    "seshat_mark_copy",
    "seshat_alloc_tag",
    "seshat_event_count",
    "seshat_dump_trace",
];

/// Whether `name` is a Seshat builtin (used by both executors' classification).
pub fn is_seshat_builtin(name: &str) -> bool {
    SESHAT_BUILTINS.contains(&name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::rc::Rc;

    fn s(x: &str) -> Value {
        Value::String(Rc::new(x.to_string()))
    }

    // `Value` intentionally does not derive `PartialEq`, so unwrap the int.
    fn int(v: &Value) -> i64 {
        match v {
            Value::Int(i) => *i,
            other => panic!("expected Int, got {other:?}"),
        }
    }

    #[test]
    fn non_seshat_name_falls_through() {
        assert!(dispatch_seshat("matmul", &[]).unwrap().is_none());
    }

    #[test]
    fn zone_handles_are_sequential() {
        reset();
        let h1 = dispatch_seshat("seshat_zone_start", &[s("parse")]).unwrap().unwrap();
        let h2 = dispatch_seshat("seshat_zone_start", &[s("compute")]).unwrap().unwrap();
        assert_eq!(int(&h1), 1);
        assert_eq!(int(&h2), 2);
    }

    #[test]
    fn bad_domain_is_err_not_panic() {
        reset();
        let r = dispatch_seshat("seshat_alloc_tag", &[s("nonsense"), Value::Int(8)]);
        assert!(r.is_err());
    }

    #[test]
    fn wrong_arity_is_err() {
        reset();
        assert!(dispatch_seshat("seshat_zone_start", &[]).is_err());
    }

    #[test]
    fn negative_bytes_rejected() {
        reset();
        let r = dispatch_seshat("seshat_alloc_tag", &[s("rust"), Value::Int(-1)]);
        assert!(r.is_err());
    }

    #[test]
    fn records_into_sink() {
        reset();
        dispatch_seshat("seshat_alloc_tag", &[s("rust"), Value::Int(100)]).unwrap();
        dispatch_seshat("seshat_mark_copy", &[s("rust"), s("numpy"), Value::Int(100)]).unwrap();
        let t = snapshot();
        assert_eq!(t.num_events(), 2);
        let report = polytrace::analyze_trace(&t);
        assert_eq!(report.ownership.total_allocated, 100);
        assert_eq!(report.copy.total_bytes, 100);
        assert!(report.copy.flows[0].avoidable); // rust→numpy is zero-copy-compatible
    }
}
