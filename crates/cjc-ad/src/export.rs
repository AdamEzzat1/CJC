//! Postulate Phase 1 — what a `GradGraph` can tell an IDE about itself.
//!
//! The tape is a flat arena, so the export is one record per arena slot, in
//! arena order. Each record carries the source span of the call that created
//! it (stamped by the executor that ran the call), the layer group it belongs
//! to, and — for parameters after a backward pass — its gradient norm.
//!
//! Nothing here feeds back into the numeric path: spans and groups live in
//! side vectors that no forward or backward arm reads, so stamping them cannot
//! change an output bit (`tests/postulate_phase1` holds that to the hash).

use crate::GradOp;

/// A byte range in a source file. `file` is the executor's file id; `0` is the
/// entry file of the run (multi-module runs do not number their files yet).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SourceSpan {
    pub file: u32,
    pub start: u32,
    pub end: u32,
}

impl SourceSpan {
    pub fn new(file: u32, start: usize, end: usize) -> Self {
        Self { file, start: start as u32, end: end as u32 }
    }
}

/// One arena slot of the tape.
#[derive(Clone, Debug, PartialEq)]
pub struct NodeExport {
    pub id: u32,
    /// The `GradOp` variant name, e.g. `MatMul`, `Parameter`, `MseMean`.
    pub op: String,
    /// Operand node ids, in the order the op names them.
    pub inputs: Vec<u32>,
    pub shape: Vec<usize>,
    pub span: Option<SourceSpan>,
    pub group: Option<u32>,
    /// L2 norm of the accumulated gradient — parameters only, `None` before backward
    /// has run or for non-parameter nodes. Summed with Kahan (determinism rule).
    pub grad_norm: Option<f64>,
}

/// A layer group: either a layer the program named (`g.layer("hidden")`) or,
/// failing that, the function whose body recorded the op.
#[derive(Clone, Debug, PartialEq)]
pub struct GroupExport {
    pub id: u32,
    pub name: String,
    pub nodes: Vec<u32>,
}

/// A tensor op that ran outside the tape on a value the tape owns, so no
/// gradient flows through it: "computed outside the graph — not trained".
#[derive(Clone, Debug, PartialEq)]
pub struct OffTape {
    /// The builtin that ran off the tape, e.g. `Tensor.conv2d`.
    pub op: String,
    pub span: Option<SourceSpan>,
    /// The tape node whose value it read.
    pub node: u32,
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct GraphExport {
    pub nodes: Vec<NodeExport>,
    pub groups: Vec<GroupExport>,
    pub off_tape: Vec<OffTape>,
}

/// Side tables the executors fill. Kept apart from the numeric arena so the
/// forward/backward code never sees them.
#[derive(Clone, Debug, Default)]
pub(crate) struct NodeMeta {
    pub spans: Vec<Option<SourceSpan>>,
    pub groups: Vec<Option<u32>>,
    pub group_names: Vec<String>,
    /// Layer named by the program; overrides the function grouping while set.
    pub layer: Option<String>,
    pub off_tape: Vec<OffTape>,
}

impl NodeMeta {
    /// Give every node recorded since the last stamp this span and group.
    pub fn stamp(&mut self, len: usize, span: Option<SourceSpan>, fn_group: Option<&str>) {
        if self.spans.len() >= len {
            return;
        }
        let name = self.layer.clone().or_else(|| fn_group.map(str::to_owned));
        let gid = name.map(|n| self.group_id(&n));
        self.spans.resize(len, span);
        self.groups.resize(len, gid);
    }

    fn group_id(&mut self, name: &str) -> u32 {
        match self.group_names.iter().position(|g| g == name) {
            Some(i) => i as u32,
            None => {
                self.group_names.push(name.to_owned());
                (self.group_names.len() - 1) as u32
            }
        }
    }
}

/// The `GradOp` variant name, taken from its `Debug` form (stable per variant).
pub(crate) fn op_name(op: &GradOp) -> String {
    let dbg = format!("{op:?}");
    let end = dbg.find(|c: char| !(c.is_ascii_alphanumeric() || c == '_')).unwrap_or(dbg.len());
    dbg[..end].to_owned()
}

/// Kahan-summed L2 norm.
pub(crate) fn l2_norm(data: &[f64]) -> f64 {
    let sq: Vec<f64> = data.iter().map(|x| x * x).collect();
    cjc_repro::kahan_sum_f64(&sq).sqrt()
}

// ── JSON ────────────────────────────────────────────────────────────────
//
// Plain JSON with a fixed key order, so equal exports are equal bytes. Floats
// use Rust's shortest round-trip form; non-finite values become strings, the
// same convention as `cjc_snap::json`.

fn push_str(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

pub(crate) fn push_f64(out: &mut String, x: f64) {
    if x.is_nan() {
        out.push_str("\"NaN\"");
    } else if x.is_infinite() {
        out.push_str(if x > 0.0 { "\"Infinity\"" } else { "\"-Infinity\"" });
    } else {
        let s = format!("{x}");
        out.push_str(&s);
        if !s.contains(['.', 'e', 'E']) {
            out.push_str(".0");
        }
    }
}

fn push_list<T>(out: &mut String, items: &[T], mut f: impl FnMut(&mut String, &T)) {
    out.push('[');
    for (i, it) in items.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        f(out, it);
    }
    out.push(']');
}

fn push_span(out: &mut String, s: &Option<SourceSpan>) {
    match s {
        Some(s) => out.push_str(&format!(
            "{{\"file\":{},\"start\":{},\"end\":{}}}",
            s.file, s.start, s.end
        )),
        None => out.push_str("null"),
    }
}

impl GraphExport {
    /// Deterministic JSON: equal exports serialise to equal bytes.
    pub fn to_json(&self) -> String {
        let mut o = String::with_capacity(64 + self.nodes.len() * 96);
        o.push_str("{\"nodes\":");
        push_list(&mut o, &self.nodes, |o, n| {
            o.push_str(&format!("{{\"id\":{},\"op\":", n.id));
            push_str(o, &n.op);
            o.push_str(",\"inputs\":");
            push_list(o, &n.inputs, |o, i| o.push_str(&i.to_string()));
            o.push_str(",\"shape\":");
            push_list(o, &n.shape, |o, d| o.push_str(&d.to_string()));
            o.push_str(",\"span\":");
            push_span(o, &n.span);
            o.push_str(",\"group\":");
            match n.group {
                Some(g) => o.push_str(&g.to_string()),
                None => o.push_str("null"),
            }
            o.push_str(",\"grad_norm\":");
            match n.grad_norm {
                Some(g) => push_f64(o, g),
                None => o.push_str("null"),
            }
            o.push('}');
        });
        o.push_str(",\"groups\":");
        push_list(&mut o, &self.groups, |o, g| {
            o.push_str(&format!("{{\"id\":{},\"name\":", g.id));
            push_str(o, &g.name);
            o.push_str(",\"nodes\":");
            push_list(o, &g.nodes, |o, i| o.push_str(&i.to_string()));
            o.push('}');
        });
        o.push_str(",\"off_tape\":");
        push_list(&mut o, &self.off_tape, |o, t| {
            o.push_str("{\"op\":");
            push_str(o, &t.op);
            o.push_str(",\"span\":");
            push_span(o, &t.span);
            o.push_str(&format!(",\"node\":{}}}", t.node));
        });
        o.push('}');
        o
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GradGraph;
    use cjc_runtime::Tensor;

    fn sp(a: usize, b: usize) -> Option<SourceSpan> {
        Some(SourceSpan::new(0, a, b))
    }

    #[test]
    fn export_is_arena_order_with_edges_spans_and_groups() {
        let mut g = GradGraph::new();
        let w = g.parameter(Tensor::from_vec_unchecked(vec![1.0, 2.0], &[2, 1]));
        g.stamp(sp(0, 5), Some("main"));
        let x = g.input(Tensor::from_vec_unchecked(vec![3.0, 4.0], &[1, 2]));
        g.stamp(sp(6, 9), Some("main"));
        g.set_layer(Some("head".into()));
        let y = g.matmul(x, w);
        g.stamp(sp(10, 20), Some("main"));
        g.set_layer(None);
        let l = g.sum(y);
        g.stamp(sp(21, 30), Some("loss_fn"));
        g.backward(l);

        let e = g.export();
        assert_eq!(e.nodes.len(), g.len());
        assert_eq!(e.nodes[2].op, "MatMul");
        assert_eq!(e.nodes[2].inputs, vec![x as u32, w as u32]);
        assert_eq!(e.nodes[2].span, sp(10, 20));
        let names: Vec<&str> = e.groups.iter().map(|g| g.name.as_str()).collect();
        assert_eq!(names, ["main", "head", "loss_fn"]);
        assert_eq!(e.groups[1].nodes, vec![2]);
        // d(sum(x@w))/dw = x^T = [3, 4] → norm 5.
        assert_eq!(e.nodes[0].grad_norm, Some(5.0));
        assert_eq!(e.nodes[1].grad_norm, None);
        assert_eq!(e.to_json(), g.export().to_json());
    }

    #[test]
    fn unstamped_nodes_export_without_span() {
        let mut g = GradGraph::new();
        g.input(Tensor::from_vec_unchecked(vec![1.0], &[1]));
        let e = g.export();
        assert_eq!(e.nodes[0].span, None);
        assert_eq!(e.nodes[0].group, None);
        assert!(e.to_json().contains("\"span\":null"));
    }

    #[test]
    fn op_names_are_variant_names() {
        assert_eq!(op_name(&GradOp::MatMul(0, 1)), "MatMul");
        assert_eq!(op_name(&GradOp::Parameter), "Parameter");
        assert_eq!(op_name(&GradOp::MseMean { sq: 1, diff: 0 }), "MseMean");
    }
}
