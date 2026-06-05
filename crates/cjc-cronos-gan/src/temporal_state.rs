//! [`TemporalState`] trait and the generic `TemporalTransition` /
//! `TemporalRollout` shapes both SSM and Liquid implementations produce.
//!
//! The trait is intentionally minimal — it captures only the
//! shape-invariant of "a vector of f64 values representing the hidden
//! state of a temporal model" — so both `StateSpaceState` and
//! `LiquidState` can share it without forcing one to drag in the other's
//! representation.

/// A hidden state of a temporal model: a fixed-dimension `f64` vector
/// that the model carries across timesteps.
pub trait TemporalState: Clone + std::fmt::Debug {
    /// Number of state dimensions (must match the owning model's
    /// `state_dim`).
    fn dim(&self) -> usize;

    /// View the underlying state vector. Always `dim()`-long.
    fn data(&self) -> &[f64];
}

/// Result of one forward step of a temporal model. Carries both the new
/// state and the model's output prediction at the current step.
#[derive(Clone, Debug)]
pub struct TemporalTransition<S: TemporalState> {
    pub prev_state: S,
    pub new_state: S,
    pub output: Vec<f64>,
}

/// Result of rolling a temporal model forward across a full sequence.
///
/// `states[0]` is the initial state, `states[t]` is the state after
/// observing input `t-1`. `outputs[t]` is the model's prediction at step
/// `t`. So `states.len() == outputs.len() + 1`.
#[derive(Clone, Debug)]
pub struct TemporalRollout<S: TemporalState> {
    pub states: Vec<S>,
    pub outputs: Vec<Vec<f64>>,
}

impl<S: TemporalState> TemporalRollout<S> {
    /// Number of input timesteps this rollout processed.
    pub fn n_steps(&self) -> usize {
        self.outputs.len()
    }

    /// View the final state — handy for chained inference where one
    /// rollout's output feeds the next.
    pub fn final_state(&self) -> Option<&S> {
        self.states.last()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test: confirm the trait can be implemented + used through a
    /// generic helper without dragging in either model.
    #[derive(Clone, Debug)]
    struct DummyState(Vec<f64>);
    impl TemporalState for DummyState {
        fn dim(&self) -> usize {
            self.0.len()
        }
        fn data(&self) -> &[f64] {
            &self.0
        }
    }

    #[test]
    fn rollout_invariants() {
        let r: TemporalRollout<DummyState> = TemporalRollout {
            states: vec![DummyState(vec![0.0]), DummyState(vec![1.0])],
            outputs: vec![vec![0.5]],
        };
        assert_eq!(r.n_steps(), 1);
        assert_eq!(r.final_state().unwrap().data(), &[1.0]);
        // states.len() == outputs.len() + 1
        assert_eq!(r.states.len(), r.outputs.len() + 1);
    }
}
