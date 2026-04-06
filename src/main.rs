//! CJC binary — enables `cargo install cjc`.
//!
//! Delegates to the CLI implementation in `cjc-cli`.

fn main() {
    cjc_cli::cli_main();
}
