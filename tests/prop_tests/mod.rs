//\! Property-based tests for the CJC compiler using proptest.
//\!
//\! These tests use proptest to generate structured random inputs and verify
//\! invariants that unit tests cannot exhaustively cover.

mod parser_props;
mod type_checker_props;
mod round_trip_props;
mod complex_props;
pub mod cli_expansion_props;
