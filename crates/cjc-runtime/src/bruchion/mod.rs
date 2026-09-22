//! Bruchion native kernels: milestone 1 of the Bruchion-for-CJC integration.
//!
//! The Bruchion repository builds a kernel pack (`examples/cjc/kernels_f64.bru`) into a
//! C-ABI static archive whose exports transcribe CJC's own hot loops — CJC's Kahan
//! recurrence with its zero skip, `matmul_raw`'s order, `relu_raw`'s comparison,
//! `adam_step`'s arithmetic, `piml_heat_1d_train`'s physics loop with `powi` reproduced
//! as Rust computes it — so that a call into the pack is meant to give the same bits
//! the Rust body gives. "Meant to" is a claim, and [`dispatch`]'s `parity` tests are
//! what turn it into a fact on this machine: every kernel against its Rust fallback,
//! `to_bits` equal, on inputs that include exact zeros, signed zeros and NaN.
//!
//! Two switches, both off by default:
//! * the cargo feature `bruchion-kernels` links the archive (`build.rs`) and compiles
//!   the `extern "C"` block; without it nothing here references a foreign symbol;
//! * the runtime policy field `bruchion_kernels` routes calls to the kernels
//!   (`crate::runtime_policy::set_bruchion_kernels(true)`); with the feature on and
//!   the switch off, every function below is its Rust body — which is how one binary
//!   runs both paths for the parity test and the benchmark.
//!
//! No CJC-source builtin toggles the switch yet; that is the next milestone's wiring.

/// The generated FFI skeleton, included verbatim from the kernel directory named by
/// `BRUCHION_KERNELS_DIR` at build time (`build.rs` exports the path). Its `const _`
/// assertions fail the build if Rust's layout of a mirror differs from the layout
/// Bruchion emitted.
#[cfg(feature = "bruchion-kernels")]
#[allow(non_camel_case_types, dead_code, unused_imports, missing_docs, clippy::all)]
pub mod ffi {
    include!(env!("BRUCHION_KERNELS_RS"));
}

pub mod dispatch;
