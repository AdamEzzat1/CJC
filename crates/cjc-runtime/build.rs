//! Links the Bruchion native kernel pack when the `bruchion-kernels` feature is on.
//!
//! Off the feature this script does nothing, so the default build stays exactly what
//! it was: no external crate, no external file. With the feature, `BRUCHION_KERNELS_DIR`
//! must name the directory `bruchionc build-kernel examples/cjc/kernels_f64.bru
//! --require no_alloc,no_os --emit-archive` printed in the Bruchion repository. That
//! directory's name is the content hash of the kernel C, its flags and its header, so
//! pinning the variable pins the ABI; `metadata.json` beside the archive is what
//! `bruchionc abi-check` compares two such directories by.
//!
//! What is linked: `libkernels_f64.a` (the kernel object, plus libgcc's `_chkstk_ms.o`
//! for MSVC consumers, bundled by `--emit-archive`). On an MSVC target the Bruchion
//! runtime's assert path also needs `fprintf`, which the UCRT keeps in
//! `legacy_stdio_definitions.lib`; that library ships with every MSVC toolchain and is
//! added here. The generated `kernels_f64.rs` is included verbatim by
//! `src/bruchion/mod.rs` through `BRUCHION_KERNELS_RS`: its `const` assertions are the
//! layout check, at compile time.
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=BRUCHION_KERNELS_DIR");
    if std::env::var_os("CARGO_FEATURE_BRUCHION_KERNELS").is_none() {
        return;
    }
    let dir = match std::env::var("BRUCHION_KERNELS_DIR") {
        Ok(d) if !d.is_empty() => PathBuf::from(d),
        _ => panic!(
            "bruchion-kernels: set BRUCHION_KERNELS_DIR to the directory `bruchionc build-kernel \
             examples/cjc/kernels_f64.bru --require no_alloc,no_os --emit-archive` wrote \
             (it holds libkernels_f64.a, kernels_f64.rs and metadata.json)"
        ),
    };
    let rs = dir.join("kernels_f64.rs");
    let archive = dir.join("libkernels_f64.a");
    let meta = dir.join("metadata.json");
    for p in [&rs, &archive, &meta] {
        assert!(p.is_file(), "bruchion-kernels: `{}` is missing; BRUCHION_KERNELS_DIR must be a build-kernel output directory", p.display());
        println!("cargo:rerun-if-changed={}", p.display());
    }
    println!("cargo:rustc-link-search=native={}", dir.display());
    // `+verbatim`: the archive keeps its `lib<stem>.a` name on every linker, MSVC's included.
    println!("cargo:rustc-link-lib=static:+verbatim=libkernels_f64.a");
    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        println!("cargo:rustc-link-lib=legacy_stdio_definitions");
    }
    println!("cargo:rustc-env=BRUCHION_KERNELS_RS={}", rs.display());
}
