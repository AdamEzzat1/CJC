//! CJC CLI subcommands.
//!
//! Each module implements a single `cjcl <command>` subcommand.
//! All commands share the output formatting from `crate::output`.

// Phase 1: execution + data + reproducibility
pub mod view;
pub mod proof;
pub mod flow;
pub mod patch;
pub mod seek;
pub mod drift;
pub mod forge;

// Phase 2: inspection + observability + validation + diagnostics
pub mod inspect;
pub mod schema;
pub mod check2;
pub mod trace;
pub mod mem;
pub mod bench;
pub mod pack;
pub mod doctor;

// Phase 3: compiler visibility + runtime analysis + numerical + reproducibility + CI
pub mod emit;
pub mod explain;
pub mod gc;
pub mod nogc;
pub mod audit;
pub mod precision;
pub mod lock;
pub mod parity;
pub mod test_cmd;
pub mod ci;
