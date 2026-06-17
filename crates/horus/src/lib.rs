//! # Horus — CJC-Lang binding for the [Polytrace](polytrace) profiler
//!
//! `horus` is the thin glue that exposes the standalone `polytrace` profiler to
//! the CJC-Lang runtime: the write-only `seshat_*` `.cjcl` builtins
//! ([`dispatch_seshat`]) routed from `cjc-eval` and `cjc-mir-exec`. It is kept
//! separate from `polytrace` on purpose — so the **published** profiler crate
//! carries no `cjc-runtime` dependency and stays a clean, general-purpose tool.
//!
//! The full `polytrace` analysis surface (trace model, the analyses, `merge`,
//! `serialize`/`replay`, renderers) is re-exported here, so `horus` is a drop-in
//! for code that previously used the combined crate.

// Re-export the entire standalone engine so `horus::Trace`, `horus::analyze_trace`,
// etc. resolve exactly as the old combined crate did.
pub use polytrace::*;

pub mod dispatch;
pub use dispatch::{dispatch_seshat, is_seshat_builtin, SESHAT_BUILTINS};
