//! CJC Language Server — entry point.
//!
//! Run with: `cjc-analyzer` (communicates over stdin/stdout)

fn main() {
    if let Err(e) = cjc_analyzer::server::run_server() {
        eprintln!("cjc-analyzer error: {}", e);
        std::process::exit(1);
    }
}
