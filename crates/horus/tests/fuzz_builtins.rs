//! Builtin-arg fuzz for the `.cjcl`-facing `seshat_*` surface: random args into
//! `dispatch_seshat` must return `Ok(Some)`/`Ok(None)`/`Err` — never panic.
//! (Relocated here from the engine crate when `dispatch_seshat` moved into
//! `horus`; the pure-engine fuzz targets stay in `polytrace`.)

use std::rc::Rc;

use cjc_runtime::value::Value;
use horus::dispatch_seshat;

#[test]
fn fuzz_builtin_args_never_panic() {
    bolero::check!()
        .with_type::<(u8, i64, i64)>()
        .for_each(|&(sel, x, y): &(u8, i64, i64)| {
            // NB: `seshat_dump_trace` is intentionally excluded — it performs
            // filesystem I/O, which has no place in a fuzz loop. Its arg
            // handling is covered by the unit tests.
            let names = [
                "seshat_reset",
                "seshat_zone_start",
                "seshat_zone_stop",
                "seshat_mark_boundary",
                "seshat_mark_copy",
                "seshat_alloc_tag",
                "seshat_event_count",
                "not_a_seshat_builtin",
            ];
            let name = names[(sel as usize) % names.len()];
            // a grab-bag of arg shapes the parser might hand us
            let argsets: Vec<Vec<Value>> = vec![
                vec![],
                vec![Value::Int(x)],
                vec![Value::Int(x), Value::Int(y)],
                vec![Value::String(Rc::new(format!("{x}")))],
                vec![
                    Value::String(Rc::new("rust".to_string())),
                    Value::String(Rc::new("numpy".to_string())),
                    Value::Int(y),
                ],
                vec![Value::Bool(true), Value::Float(1.5)],
            ];
            for args in &argsets {
                // never panic — Ok(Some)/Ok(None)/Err are all fine.
                let _ = dispatch_seshat(name, args);
            }
        });
}
