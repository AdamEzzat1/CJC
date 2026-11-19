//! CJC Language Server — entry point.
//!
//! Run with: `cjc-analyzer [--companion PATH]` (communicates over stdin/stdout).
//! `--companion PATH` mirrors the editor's buffer and cursor for a Postulate
//! HUD (see `cjc_analyzer::companion`).

/// Counts heap bytes so `postulate/runReport` can report peak memory.
#[global_allocator]
static ALLOC: cjc_analyzer::alloc_meter::CountingAlloc = cjc_analyzer::alloc_meter::CountingAlloc;

fn main() {
    let mut args = std::env::args().skip(1);
    let mut companion = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--companion" => match args.next() {
                Some(p) => companion = Some(std::path::PathBuf::from(p)),
                None => {
                    eprintln!("cjc-analyzer: --companion requires a path");
                    std::process::exit(2);
                }
            },
            // Editors commonly pass `--stdio`; stdio is the only transport.
            "--stdio" => {}
            other => {
                eprintln!("cjc-analyzer: unknown argument `{other}`");
                std::process::exit(2);
            }
        }
    }
    if let Err(e) = cjc_analyzer::server::run_server(companion) {
        eprintln!("cjc-analyzer error: {}", e);
        std::process::exit(1);
    }
}
