//! Nearest-neighbor matching on the logit of the propensity score.
//!
//! ## Algorithm (greedy, without-replacement)
//!
//! For each treated unit (iterated in ascending row index), scan all
//! currently-unmatched control units (in ascending row index), find the one
//! whose `|logit_t - logit_c|` is smallest *and* within `caliper`. If found,
//! commit the match and mark the control consumed. If not, the treated unit
//! is left unmatched.
//!
//! ## Tie-breaking — the load-bearing determinism contract
//!
//! When two control units have logit distances to the treated unit that are
//! within `f64::EPSILON` of each other, the **lower control row index wins**.
//! This is the cjc-causal determinism rule per ADR-0043 §determinism (point 1).
//! Implementation mechanism: the inner loop iterates controls in ascending
//! row order, and an equal-or-not-strictly-smaller distance does NOT replace
//! the incumbent best. The first eligible control survives as the match.
//!
//! ## Complexity
//!
//! O(n_treated × n_control) — acceptable for v0.1's "teaching/research" scope
//! (typical academic propensity-score studies have n < 50,000). v0.2 may
//! introduce a KD-tree or sorted-logit nearest-search if a real deployment
//! surfaces this as a bottleneck.

/// A single matched pair (treated row index, control row index).
pub type MatchedPair = (usize, usize);

/// Greedy nearest-neighbor matching with caliper restriction.
///
/// # Arguments
///
/// - `logits`: per-row logit(propensity), length `n_rows`. NaN logits make
///   that row unmatchable but do not error.
/// - `treatment`: per-row treatment indicator (1.0 = treated, 0.0 = control,
///   any other value treated as "not treated and not control" → skipped).
/// - `caliper`: maximum allowed `|logit_t - logit_c|` for a match. Must be
///   `> 0.0`; the caller is responsible for that check.
///
/// # Returns
///
/// `(matched_pairs, n_treated_unmatched)`:
///
/// - `matched_pairs`: `Vec<(treated_idx, control_idx)>` in ascending treated
///   index order. Length = number of successful matches.
/// - `n_treated_unmatched`: number of treated units for which no control was
///   within caliper. Surfaces in [`BalanceReport::n_treated_unmatched`] and
///   may trigger Locke finding E9103 (overlap failure).
///
/// # Determinism
///
/// Two calls with bit-identical `logits` + `treatment` + `caliper` produce
/// bit-identical `matched_pairs` and `n_treated_unmatched`. No randomness,
/// no hash-order dependency, ties broken by ascending row index.
pub fn nearest_neighbor_match(
    logits: &[f64],
    treatment: &[f64],
    caliper: f64,
) -> (Vec<MatchedPair>, u64) {
    debug_assert_eq!(logits.len(), treatment.len(), "logits and treatment must align");

    let n_rows = logits.len();
    let mut consumed = vec![false; n_rows];
    let mut pairs: Vec<MatchedPair> = Vec::new();
    let mut n_treated_unmatched: u64 = 0;

    for t_idx in 0..n_rows {
        // Skip non-treated rows in the outer loop.
        if treatment[t_idx] != 1.0 {
            continue;
        }
        // Treated row's logit must itself be finite to be matchable.
        let l_t = logits[t_idx];
        if !l_t.is_finite() {
            n_treated_unmatched += 1;
            continue;
        }

        let mut best_idx: Option<usize> = None;
        let mut best_dist: f64 = f64::INFINITY;

        for c_idx in 0..n_rows {
            // Skip non-controls, non-finite logits, and already-consumed controls.
            if treatment[c_idx] != 0.0 || consumed[c_idx] {
                continue;
            }
            let l_c = logits[c_idx];
            if !l_c.is_finite() {
                continue;
            }
            let dist = (l_t - l_c).abs();
            if dist > caliper {
                continue;
            }
            // Strictly-less: equal-distance ties keep the incumbent (which
            // has the lower row index because we iterate ascending). This
            // is the determinism contract.
            if dist < best_dist {
                best_dist = dist;
                best_idx = Some(c_idx);
            }
        }

        match best_idx {
            Some(c) => {
                consumed[c] = true;
                pairs.push((t_idx, c));
            }
            None => {
                n_treated_unmatched += 1;
            }
        }
    }

    (pairs, n_treated_unmatched)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_returns_empty_output() {
        let (pairs, unmatched) = nearest_neighbor_match(&[], &[], 0.5);
        assert!(pairs.is_empty());
        assert_eq!(unmatched, 0);
    }

    #[test]
    fn no_controls_means_all_treated_unmatched() {
        // Three treated, zero controls.
        let logits = vec![0.1, 0.2, 0.3];
        let treatment = vec![1.0, 1.0, 1.0];
        let (pairs, unmatched) = nearest_neighbor_match(&logits, &treatment, 1.0);
        assert!(pairs.is_empty());
        assert_eq!(unmatched, 3);
    }

    #[test]
    fn single_pair_matches_within_caliper() {
        // Row 0 treated at logit 0.0, row 1 control at logit 0.1 — within caliper 0.2.
        let logits = vec![0.0, 0.1];
        let treatment = vec![1.0, 0.0];
        let (pairs, unmatched) = nearest_neighbor_match(&logits, &treatment, 0.2);
        assert_eq!(pairs, vec![(0, 1)]);
        assert_eq!(unmatched, 0);
    }

    #[test]
    fn pair_outside_caliper_left_unmatched() {
        // Distance |0.0 - 0.5| = 0.5 > caliper 0.1.
        let logits = vec![0.0, 0.5];
        let treatment = vec![1.0, 0.0];
        let (pairs, unmatched) = nearest_neighbor_match(&logits, &treatment, 0.1);
        assert!(pairs.is_empty());
        assert_eq!(unmatched, 1);
    }

    #[test]
    fn ties_break_by_ascending_control_index() {
        // Treated at logit 0.5. Two controls at logits 0.0 and 1.0 — both
        // bit-exact in IEEE-754, so 0.5 - 0.0 == 1.0 - 0.5 == 0.5 EXACTLY.
        // Determinism rule says lower control index (1) must win.
        //
        // Subtle: do NOT use values like 0.3 / 0.7 here. Those are not
        // representable in binary, so their subtraction has different
        // rounding and the distances are not bit-equal — that probes a
        // different code path (strict < picks the smaller) and would
        // silently hide a tie-break bug.
        let logits: Vec<f64> = vec![0.5, 0.0, 1.0];
        let treatment: Vec<f64> = vec![1.0, 0.0, 0.0];
        assert_eq!(
            (logits[0] - logits[1]).abs().to_bits(),
            (logits[0] - logits[2]).abs().to_bits(),
            "test premise: distances must be bit-equal"
        );
        let (pairs, _) = nearest_neighbor_match(&logits, &treatment, 1.0);
        assert_eq!(pairs, vec![(0, 1)], "lower control index must win on tie");
    }

    #[test]
    fn tie_break_stable_across_permutations_of_equal_distance_controls() {
        // Four controls at bit-equal distances. Whichever order they appear
        // in row indices, the LOWEST index wins for each treated unit.
        // logits: t=0.0 at row 0; controls at rows 1..5 with bit-equal
        // distances by symmetry around 0.0.
        let logits = vec![0.0, -1.0, 1.0, -2.0, 2.0];
        let treatment = vec![1.0, 0.0, 0.0, 0.0, 0.0];
        let (pairs, _) = nearest_neighbor_match(&logits, &treatment, 1.0);
        // Treated 0 has two equidistant controls at distance 1.0 (rows 1, 2).
        // Tie-break: row 1 wins.
        assert_eq!(pairs, vec![(0, 1)]);
    }

    #[test]
    fn nan_control_logits_are_skipped() {
        // Two controls: one NaN-logit (unmatchable), one valid.
        let logits = vec![0.0, f64::NAN, 0.1];
        let treatment = vec![1.0, 0.0, 0.0];
        let (pairs, _) = nearest_neighbor_match(&logits, &treatment, 1.0);
        assert_eq!(pairs, vec![(0, 2)]);
    }

    #[test]
    fn nan_treated_logit_makes_row_unmatchable() {
        let logits = vec![f64::NAN, 0.1];
        let treatment = vec![1.0, 0.0];
        let (pairs, unmatched) = nearest_neighbor_match(&logits, &treatment, 1.0);
        assert!(pairs.is_empty());
        assert_eq!(unmatched, 1);
    }

    #[test]
    fn matching_is_without_replacement() {
        // Two treated competing for one control.
        // Treated 0 at logit 0.0, treated 2 at logit 0.5, control 1 at logit 0.1.
        // Treated 0 is processed first (ascending iteration); it claims control 1.
        // Treated 2 then finds no available control and is unmatched.
        let logits = vec![0.0, 0.1, 0.5];
        let treatment = vec![1.0, 0.0, 1.0];
        let (pairs, unmatched) = nearest_neighbor_match(&logits, &treatment, 1.0);
        assert_eq!(pairs, vec![(0, 1)]);
        assert_eq!(unmatched, 1);
    }

    #[test]
    fn non_binary_treatment_values_skipped_silently() {
        // treatment[3] is 0.5 (neither 1.0 nor 0.0). Should be ignored.
        // The orchestrator validates 0/1; this is a defensive test for matching itself.
        let logits = vec![0.0, 0.1, 0.2, 0.3];
        let treatment = vec![1.0, 0.0, 0.0, 0.5];
        let (pairs, _) = nearest_neighbor_match(&logits, &treatment, 1.0);
        assert_eq!(pairs.len(), 1);
        // Treated 0 picks the nearest control (1, distance 0.1).
        assert_eq!(pairs[0], (0, 1));
    }
}
