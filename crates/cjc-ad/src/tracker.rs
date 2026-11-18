//! Postulate Phase 1 — the executor-side half of the GradGraph hooks.
//!
//! Both executors (`cjc-eval`, `cjc-mir-exec`) own one [`Tracker`] and call it
//! at the same four points: entering/leaving a user function, before a call
//! dispatches (to set the call's span), and after a builtin or method call
//! returns. Keeping the logic here, not in each executor, is what makes the
//! two produce byte-identical exports.
//!
//! Spans are always stamped (one `Option<SourceSpan>` per node). Everything
//! else — function grouping, the trained-graph handle, off-tape detection —
//! runs only when [`Tracker::enabled`] is set, so the default path pays a
//! branch and nothing more. None of it reads or writes a tensor value.

use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

use cjc_runtime::Tensor;

use crate::export::{GraphExport, SourceSpan};
use crate::telemetry::Telemetry;
use crate::GradGraph;

/// Handle type the executors use for `Value::GradGraph`.
pub type GraphHandle = Rc<RefCell<dyn Any>>;

/// The synthetic entry function MIR lowers top-level statements into. The tree
/// walker has no such frame, so it is skipped when grouping.
const SYNTHETIC_ENTRY: &str = "__main";

#[derive(Default)]
pub struct Tracker {
    /// Capture function groups, the last trained graph, and off-tape ops.
    pub enabled: bool,
    /// Span of the call about to dispatch. Executors set it after the call's
    /// arguments are evaluated, so nested calls cannot leave a stale value.
    pub span: Option<SourceSpan>,
    fn_stack: Vec<String>,
    /// Graph most recently created by `GradGraph.new`.
    current: Option<GraphHandle>,
    /// Graph most recently passed to `backward`.
    trained: Option<GraphHandle>,
    /// Tensors handed to `current` through `input`/`parameter`, with their node.
    registered: Vec<(Tensor, usize)>,
    /// Opt-in per-step telemetry; independent of [`Self::enabled`].
    pub telemetry: Option<Telemetry>,
}

impl Tracker {
    /// The span recorded for a call at `start..end` (single-file: file 0).
    /// Both executors build call spans only through this.
    pub fn call_span(start: usize, end: usize) -> SourceSpan {
        SourceSpan::new(0, start, end)
    }

    pub fn set_span(&mut self, start: usize, end: usize) {
        self.span = Some(Self::call_span(start, end));
    }

    pub fn enter_fn(&mut self, name: &str) {
        if self.enabled {
            self.fn_stack.push(name.to_owned());
        }
    }

    pub fn exit_fn(&mut self) {
        if self.enabled {
            self.fn_stack.pop();
        }
    }

    fn fn_group(&self) -> Option<&str> {
        self.fn_stack
            .iter()
            .rev()
            .map(String::as_str)
            .find(|n| *n != SYNTHETIC_ENTRY)
    }

    /// A builtin returned a new graph.
    pub fn on_graph_new(&mut self, h: &GraphHandle) {
        if self.enabled {
            self.current = Some(h.clone());
            self.registered.clear();
        }
    }

    /// A method ran on graph `h`. `arg_tensor` is the tensor argument of
    /// `input`/`parameter`; `first_int_arg` is the call's first node-index
    /// argument (the loss, for `backward`); `result_node` is the node index the
    /// call returned.
    pub fn after_graph_method(
        &mut self,
        h: &GraphHandle,
        method: &str,
        arg_tensor: Option<&Tensor>,
        first_int_arg: Option<usize>,
        result_node: Option<usize>,
    ) {
        let group = self.fn_group().map(str::to_owned);
        if let Some(g) = h.borrow_mut().downcast_mut::<GradGraph>() {
            g.stamp(self.span, group.as_deref());
            if let (Some(t), "backward", Some(loss)) = (&mut self.telemetry, method, first_int_arg) {
                if loss < g.len() {
                    t.on_backward(g, loss);
                }
            }
        }
        if !self.enabled {
            return;
        }
        match (method, arg_tensor, result_node) {
            ("input" | "parameter", Some(t), Some(node)) => {
                if self.current.as_ref().is_some_and(|c| Rc::ptr_eq(c, h)) {
                    self.registered.push((t.clone(), node));
                }
            }
            ("backward" | "backward_collect", _, _) => self.trained = Some(h.clone()),
            _ => {}
        }
    }

    /// A builtin or tensor method named `op` returned a tensor. If one of its
    /// tensor operands is a value the current graph owns, the op ran off the
    /// tape: record it (export keeps it only if that node feeds nothing on
    /// the tape — i.e. the tape never used it, so it cannot be trained).
    pub fn after_tensor_op<'a>(&mut self, op: &str, operands: impl Iterator<Item = &'a Tensor>) {
        if !self.enabled || self.registered.is_empty() || op.starts_with("GradGraph.") {
            return;
        }
        let mut hits = Vec::new();
        for t in operands {
            for (r, node) in &self.registered {
                if r.buffer.ptr_eq(&t.buffer) && !hits.contains(node) {
                    hits.push(*node);
                }
            }
        }
        if hits.is_empty() {
            return;
        }
        if let Some(h) = &self.current {
            if let Some(g) = h.borrow_mut().downcast_mut::<GradGraph>() {
                for node in hits {
                    g.record_off_tape(op, self.span, node);
                }
            }
        }
    }

    fn last_graph<R>(&self, f: impl FnOnce(&GradGraph) -> R) -> Option<R> {
        let h = self.trained.as_ref().or(self.current.as_ref())?;
        let b = h.borrow();
        b.downcast_ref::<GradGraph>().map(f)
    }

    /// Export of the graph most recently trained (else most recently created).
    pub fn export(&self) -> Option<GraphExport> {
        self.last_graph(GradGraph::export)
    }

    /// [`GradGraph::state_bits`] of the same graph.
    pub fn state_bits(&self) -> Option<Vec<u64>> {
        self.last_graph(GradGraph::state_bits)
    }
}
