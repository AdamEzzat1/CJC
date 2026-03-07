//! CJC Analyzer — Language Server Protocol implementation for CJC.
//!
//! Architecture:
//! - `symbol_index`: Collects all known symbols (builtins + libraries + user-defined)
//! - `hover`: Provides hover documentation
//! - `completion`: Generates completion items
//! - `diagnostics`: Bridges CJC diagnostics to LSP format
//! - `server`: Main LSP event loop (stdin/stdout JSON-RPC via `lsp-server`)
//!
//! The analyzer is import-aware: `import vizor` activates Vizor symbols in
//! completion and hover. Other libraries follow the same pattern.

pub mod symbol_index;
pub mod hover;
pub mod completion;
pub mod diagnostics;
pub mod server;
