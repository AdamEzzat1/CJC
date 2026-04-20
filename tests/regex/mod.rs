//! Organized regex test suite for features added in the engine upgrade.
//!
//! Covers POSIX classes, Unicode escapes, non-capturing groups, inline flags,
//! absolute anchors (\A \z \Z), \B, counted repetition {n}/{n,}/{n,m},
//! MatchResult API, regex_explain, safety limits, and composition helpers.
//!
//! Run with:
//!   cargo test --test test_regex_new

mod engine;
mod property;
mod tidyview;
mod fuzz;
mod captures;
