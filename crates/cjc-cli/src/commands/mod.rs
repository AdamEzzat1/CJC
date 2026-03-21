//! CJC CLI subcommands.
//!
//! Each module implements a single `cjc <command>` subcommand.
//! All commands share the output formatting from `crate::output`.

pub mod view;
pub mod proof;
pub mod flow;
pub mod patch;
pub mod seek;
pub mod drift;
pub mod forge;
