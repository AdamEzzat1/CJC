//! CJC Analyzer — Language Server Protocol implementation for CJC.
//!
//! Architecture:
//! - `symbol_index`: Collects all known symbols (builtins + libraries + user-defined)
//! - `hover`: Provides hover documentation
//! - `completion`: Generates completion items
//! - `diagnostics`: Bridges CJC diagnostics to LSP format
//! - `postulate`: Postulate IDE requests (`postulate/*`) and tensor-shape inlay hints
//! - `server`: In-process [`server::Analyzer`] core plus the stdio LSP loop
//!
//! The analyzer is import-aware: `import vizor` activates Vizor symbols in
//! completion and hover. Other libraries follow the same pattern.

pub mod symbol_index;
pub mod hover;
pub mod completion;
pub mod diagnostics;
pub mod postulate;
pub mod server;
