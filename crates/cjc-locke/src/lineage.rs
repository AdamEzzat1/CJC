//! Impressions, Ideas, and the deterministic lineage DAG.
//!
//! A `LockeImpression` is a raw observed fact about a dataset — schema,
//! a row, a column value, a cell. A `LockeIdea` is a derived value — a
//! filtered subset, a feature, a metric, a prediction. Every Idea has at
//! least one parent (Impression or Idea); Impressions have no parents.
//!
//! The lineage graph is built incrementally with `LineageBuilder` and
//! frozen into an immutable `LineageGraph` that's deterministic to emit.
//!
//! Audit events are sequenced inside a single run (monotonic `seq`) and
//! never use wall-clock timestamps — repeated runs produce bit-identical
//! audit chains.

use std::collections::{BTreeMap, BTreeSet};

use crate::id::{fingerprint, fingerprint_compose, fingerprint_str, FingerprintId, IdDomain};

/// Raw observed fact. Has no parents.
#[derive(Clone, Debug, PartialEq)]
pub struct LockeImpression {
    pub id: FingerprintId,
    pub source: String,
    pub kind: ImpressionKind,
    pub n_rows: u64,
    pub columns: Vec<String>,
    pub schema_fingerprint: FingerprintId,
}

/// Categorical tag distinguishing the kind of raw observation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ImpressionKind {
    Dataset,
    Column,
    Row,
    Schema,
}

impl ImpressionKind {
    fn tag(self) -> &'static str {
        match self {
            ImpressionKind::Dataset => "dataset",
            ImpressionKind::Column => "column",
            ImpressionKind::Row => "row",
            ImpressionKind::Schema => "schema",
        }
    }
}

impl LockeImpression {
    pub fn new(source: &str, kind: ImpressionKind, n_rows: u64, mut columns: Vec<String>) -> Self {
        columns.sort();
        let schema_fingerprint = {
            let joined = columns.join("\u{1f}");
            fingerprint_str(IdDomain::Impression, &joined)
        };
        let id_parts = [
            fingerprint_str(IdDomain::Impression, source),
            fingerprint_str(IdDomain::Impression, kind.tag()),
            fingerprint(IdDomain::Impression, &n_rows.to_le_bytes()),
            schema_fingerprint,
        ];
        let id = fingerprint_compose(IdDomain::Impression, "impression", &id_parts);
        Self {
            id,
            source: source.to_string(),
            kind,
            n_rows,
            columns,
            schema_fingerprint,
        }
    }
}

/// A derived value. At least one parent (impression or idea).
#[derive(Clone, Debug, PartialEq)]
pub struct LockeIdea {
    pub id: FingerprintId,
    pub name: String,
    pub transform: TransformationRecord,
    pub parents: Vec<FingerprintId>,
}

/// One transformation step. The `op_id` is the canonical name of the
/// operation; `params` are stable string-encoded parameters (sorted).
#[derive(Clone, Debug, PartialEq)]
pub struct TransformationRecord {
    pub op_id: String,
    pub params: BTreeMap<String, String>,
    pub seed: Option<u64>,
}

impl TransformationRecord {
    pub fn fingerprint(&self) -> FingerprintId {
        let mut buf = Vec::new();
        buf.extend_from_slice(self.op_id.as_bytes());
        buf.push(0);
        for (k, v) in &self.params {
            buf.extend_from_slice(k.as_bytes());
            buf.push(b'=');
            buf.extend_from_slice(v.as_bytes());
            buf.push(0x1e);
        }
        if let Some(seed) = self.seed {
            buf.extend_from_slice(&seed.to_le_bytes());
        }
        fingerprint(IdDomain::Idea, &buf)
    }
}

impl LockeIdea {
    pub fn new(name: &str, transform: TransformationRecord, parents: Vec<FingerprintId>) -> Self {
        // Sort parents so order of supply doesn't affect identity; but if the
        // caller cares about parent order (e.g. a binary join), they should
        // include that information in `transform.params`.
        let mut sorted_parents = parents;
        sorted_parents.sort();
        let mut id_parts = vec![
            fingerprint_str(IdDomain::Idea, name),
            transform.fingerprint(),
        ];
        id_parts.extend_from_slice(&sorted_parents);
        let id = fingerprint_compose(IdDomain::Idea, "idea", &id_parts);
        Self {
            id,
            name: name.to_string(),
            transform,
            parents: sorted_parents,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum LineageNode {
    Impression(LockeImpression),
    Idea(LockeIdea),
}

impl LineageNode {
    pub fn id(&self) -> FingerprintId {
        match self {
            LineageNode::Impression(i) => i.id,
            LineageNode::Idea(i) => i.id,
        }
    }
    pub fn is_impression(&self) -> bool {
        matches!(self, LineageNode::Impression(_))
    }
    pub fn label(&self) -> &str {
        match self {
            LineageNode::Impression(i) => &i.source,
            LineageNode::Idea(i) => &i.name,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct LineageEdge {
    pub from: FingerprintId,
    pub to: FingerprintId,
    pub label: String,
}

/// One step in the deterministic audit chain.
#[derive(Clone, Debug, PartialEq)]
pub struct AuditEvent {
    pub id: FingerprintId,
    pub run_label: String,
    pub seq: u64,
    pub kind: String,
    pub subject_id: FingerprintId,
    pub note: String,
}

impl AuditEvent {
    /// Construct a new audit event.
    ///
    /// **Monotonicity is the caller's responsibility.** Internal callers go
    /// through [`LineageBuilder`] which assigns `seq` via an internal
    /// counter, so the chain produced by `builder.finish()` is always
    /// monotonic. If you construct events outside `LineageBuilder` and
    /// want to trust the resulting [`LineageGraph`]'s audit chain for
    /// replay or deterministic-ID purposes, you MUST guarantee that
    /// successive events have strictly-increasing `seq` values within a
    /// single `run_label`.
    ///
    /// To verify a graph's audit chain after construction (e.g. when
    /// rebuilding from JSON via [`crate::parse_locke_report_json`] or
    /// from external sources), call
    /// [`LineageGraph::validate_audit_monotonic`].
    pub fn new(run_label: &str, seq: u64, kind: &str, subject_id: FingerprintId, note: &str) -> Self {
        let id_parts = [
            fingerprint_str(IdDomain::AuditEvent, run_label),
            fingerprint(IdDomain::AuditEvent, &seq.to_le_bytes()),
            fingerprint_str(IdDomain::AuditEvent, kind),
            subject_id,
            fingerprint_str(IdDomain::AuditEvent, note),
        ];
        let id = fingerprint_compose(IdDomain::AuditEvent, "audit", &id_parts);
        Self {
            id,
            run_label: run_label.to_string(),
            seq,
            kind: kind.to_string(),
            subject_id,
            note: note.to_string(),
        }
    }
}

/// An immutable, deterministic lineage graph + audit chain.
///
/// `LineageBuilder` constructs one of these; once built, it's read-only
/// and serializes byte-identically across runs that fed it the same
/// inputs in the same order.
#[derive(Clone, Debug, PartialEq)]
pub struct LineageGraph {
    pub run_label: String,
    /// Nodes keyed by id and emitted in ascending-id order.
    pub nodes: BTreeMap<FingerprintId, LineageNode>,
    /// Edges sorted by (from, to, label).
    pub edges: Vec<LineageEdge>,
    /// Audit chain in `seq` order.
    pub audit: Vec<AuditEvent>,
    pub root_fingerprint: FingerprintId,
}

impl LineageGraph {
    /// Returns true iff the graph is acyclic. Always true when constructed
    /// via `LineageBuilder` (which rejects cycles at insertion), but kept
    /// as a public invariant check for property tests.
    pub fn is_acyclic(&self) -> bool {
        // v0.7+ B4.2 perf-fix: previously the inner `for e in &self.edges`
        // scanned every edge per popped node — O(V·E). Materialise a
        // forward-adjacency map once (sorted edge iteration preserves
        // determinism) and look up successors in O(deg) per pop —
        // textbook Kahn's at O(V+E).
        let succ = forward_adjacency(&self.edges);
        let mut indeg: BTreeMap<FingerprintId, u64> = self.nodes.keys().map(|k| (*k, 0)).collect();
        for e in &self.edges {
            if let Some(d) = indeg.get_mut(&e.to) {
                *d += 1;
            } else {
                return false; // dangling edge
            }
            if !indeg.contains_key(&e.from) {
                return false;
            }
        }
        let mut queue: Vec<FingerprintId> = indeg
            .iter()
            .filter(|(_, d)| **d == 0)
            .map(|(k, _)| *k)
            .collect();
        queue.sort();
        let mut visited: u64 = 0;
        while let Some(n) = queue.pop() {
            visited += 1;
            let mut newly_zero: Vec<FingerprintId> = Vec::new();
            if let Some(succs) = succ.get(&n) {
                for &to in succs {
                    if let Some(d) = indeg.get_mut(&to) {
                        *d -= 1;
                        if *d == 0 {
                            newly_zero.push(to);
                        }
                    }
                }
            }
            newly_zero.sort();
            queue.extend(newly_zero);
        }
        visited as usize == self.nodes.len()
    }

    /// Reachable ancestor set of `id` (deterministic order).
    pub fn ancestors(&self, id: FingerprintId) -> BTreeSet<FingerprintId> {
        // v0.7+ B4.2 perf-fix: O(V·E) → O(V+E) via a reverse-adjacency
        // map computed once. `out` doubles as the visited set (insert
        // returns false for already-present, gating the push).
        let pred = reverse_adjacency(&self.edges);
        let mut out = BTreeSet::new();
        let mut stack = vec![id];
        while let Some(n) = stack.pop() {
            if let Some(parents) = pred.get(&n) {
                for &from in parents {
                    if out.insert(from) {
                        stack.push(from);
                    }
                }
            }
        }
        out
    }

    /// Validate that the audit chain is monotonic within each
    /// `run_label`: successive events must have strictly-increasing
    /// `seq` values. Graphs built via `LineageBuilder` always satisfy
    /// this — the builder assigns `seq` from an internal counter. The
    /// validator exists because [`AuditEvent::new`] is public and can
    /// be invoked with arbitrary `seq` values by external callers (for
    /// instance, when rebuilding a graph from external storage).
    ///
    /// Returns `Ok(())` if every run-label's audit subchain is strictly
    /// monotonic; `Err` describes the first violation found (the indices
    /// are deterministic given the same `audit` vector).
    pub fn validate_audit_monotonic(&self) -> Result<(), AuditMonotonicError> {
        let mut last_seq_per_label: BTreeMap<&str, (u64, usize)> = BTreeMap::new();
        for (i, ev) in self.audit.iter().enumerate() {
            match last_seq_per_label.get(ev.run_label.as_str()) {
                None => {
                    last_seq_per_label.insert(ev.run_label.as_str(), (ev.seq, i));
                }
                Some(&(last_seq, last_i)) => {
                    if ev.seq <= last_seq {
                        return Err(AuditMonotonicError {
                            run_label: ev.run_label.clone(),
                            prior_index: last_i,
                            prior_seq: last_seq,
                            offending_index: i,
                            offending_seq: ev.seq,
                        });
                    }
                    last_seq_per_label.insert(ev.run_label.as_str(), (ev.seq, i));
                }
            }
        }
        Ok(())
    }
}

/// Detail for a non-monotonic audit-chain violation surfaced by
/// [`LineageGraph::validate_audit_monotonic`]. All fields are owned so
/// the error survives the graph that produced it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuditMonotonicError {
    pub run_label: String,
    pub prior_index: usize,
    pub prior_seq: u64,
    pub offending_index: usize,
    pub offending_seq: u64,
}

impl std::fmt::Display for AuditMonotonicError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "non-monotonic audit chain in run `{}`: event #{} has seq={} but prior event #{} has seq={}",
            self.run_label, self.offending_index, self.offending_seq, self.prior_index, self.prior_seq,
        )
    }
}

impl std::error::Error for AuditMonotonicError {}

/// Build a forward-adjacency map `from → [to, ...]` from the sorted
/// `edges` list. Per-key Vec preserves edge-sorted order, so traversals
/// using this map produce deterministic results.
fn forward_adjacency(edges: &[LineageEdge]) -> BTreeMap<FingerprintId, Vec<FingerprintId>> {
    let mut adj: BTreeMap<FingerprintId, Vec<FingerprintId>> = BTreeMap::new();
    for e in edges {
        adj.entry(e.from).or_default().push(e.to);
    }
    adj
}

/// Build a reverse-adjacency map `to → [from, ...]`. Used for ancestor
/// traversal and cycle-introduction checks.
fn reverse_adjacency<I, E>(edges: I) -> BTreeMap<FingerprintId, Vec<FingerprintId>>
where
    I: IntoIterator<Item = E>,
    E: std::borrow::Borrow<LineageEdge>,
{
    let mut adj: BTreeMap<FingerprintId, Vec<FingerprintId>> = BTreeMap::new();
    for e in edges {
        let e = e.borrow();
        adj.entry(e.to).or_default().push(e.from);
    }
    adj
}

/// Construct a deterministic lineage graph incrementally.
#[derive(Debug, Clone)]
pub struct LineageBuilder {
    run_label: String,
    nodes: BTreeMap<FingerprintId, LineageNode>,
    edges: BTreeSet<LineageEdge>,
    audit: Vec<AuditEvent>,
    seq: u64,
}

#[derive(Debug, PartialEq)]
pub enum LineageError {
    UnknownParent(FingerprintId),
    CycleIntroduced { from: FingerprintId, to: FingerprintId },
    DuplicateNode(FingerprintId),
}

impl LineageBuilder {
    pub fn new(run_label: &str) -> Self {
        Self {
            run_label: run_label.to_string(),
            nodes: BTreeMap::new(),
            edges: BTreeSet::new(),
            audit: Vec::new(),
            seq: 0,
        }
    }

    fn next_seq(&mut self) -> u64 {
        let s = self.seq;
        self.seq += 1;
        s
    }

    pub fn add_impression(&mut self, imp: LockeImpression) -> Result<FingerprintId, LineageError> {
        let id = imp.id;
        if self.nodes.contains_key(&id) {
            // Idempotent re-add: silently accept the existing node.
            return Ok(id);
        }
        self.nodes.insert(id, LineageNode::Impression(imp));
        let seq = self.next_seq();
        let ev = AuditEvent::new(&self.run_label, seq, "impression", id, "added impression");
        self.audit.push(ev);
        Ok(id)
    }

    pub fn add_idea(&mut self, idea: LockeIdea) -> Result<FingerprintId, LineageError> {
        // Validate parents exist.
        for p in &idea.parents {
            if !self.nodes.contains_key(p) {
                return Err(LineageError::UnknownParent(*p));
            }
        }
        let id = idea.id;
        if self.nodes.contains_key(&id) {
            return Ok(id);
        }

        // Cycle check: idea's id must not appear in any parent's ancestor closure.
        for p in &idea.parents {
            if self.would_introduce_cycle(*p, id) {
                return Err(LineageError::CycleIntroduced { from: *p, to: id });
            }
        }

        for p in &idea.parents {
            self.edges.insert(LineageEdge {
                from: *p,
                to: id,
                label: idea.transform.op_id.clone(),
            });
        }
        self.nodes.insert(id, LineageNode::Idea(idea));
        let seq = self.next_seq();
        let ev = AuditEvent::new(&self.run_label, seq, "idea", id, "added idea");
        self.audit.push(ev);
        Ok(id)
    }

    pub fn audit_note(&mut self, kind: &str, subject: FingerprintId, note: &str) {
        let seq = self.next_seq();
        let ev = AuditEvent::new(&self.run_label, seq, kind, subject, note);
        self.audit.push(ev);
    }

    fn would_introduce_cycle(&self, parent: FingerprintId, child_to_be: FingerprintId) -> bool {
        // If `child_to_be` is already an ancestor of `parent`, adding the edge
        // parent → child_to_be would create a cycle.
        //
        // v0.7+ B4.2 perf-fix: previously the inner `for e in &self.edges`
        // scanned every edge per popped node — O(V·E) per call, O(N·V·E)
        // across N add_idea calls. Materialise the reverse-adjacency map
        // once per call so the traversal is O(V+E); each call is still
        // independent (cf. incremental-closure approach which would cache
        // across calls but is MEDIUM effort — deferred).
        let pred = reverse_adjacency(self.edges.iter());
        let mut stack = vec![parent];
        let mut seen: BTreeSet<FingerprintId> = BTreeSet::new();
        while let Some(n) = stack.pop() {
            if n == child_to_be {
                return true;
            }
            if !seen.insert(n) {
                continue;
            }
            if let Some(parents) = pred.get(&n) {
                for &from in parents {
                    stack.push(from);
                }
            }
        }
        false
    }

    pub fn finish(self) -> LineageGraph {
        let nodes = self.nodes;
        let mut edges: Vec<LineageEdge> = self.edges.into_iter().collect();
        edges.sort();
        let mut parts: Vec<FingerprintId> = nodes.keys().copied().collect();
        for e in &edges {
            parts.push(fingerprint_compose(
                IdDomain::LineageEdge,
                &e.label,
                &[e.from, e.to],
            ));
        }
        let root_fingerprint = fingerprint_compose(IdDomain::LineageNode, "graph", &parts);
        LineageGraph {
            run_label: self.run_label,
            nodes,
            edges,
            audit: self.audit,
            root_fingerprint,
        }
    }
}

/// Stable JSON-ish text emit. Not a real JSON parser — just a canonical
/// indented serialization that two equal `LineageGraph`s produce byte-
/// for-byte identically.
pub fn emit_lineage_text(g: &LineageGraph) -> String {
    let mut out = String::new();
    out.push_str("# Locke Lineage Graph\n");
    out.push_str(&format!("run_label: {}\n", g.run_label));
    out.push_str(&format!("root: {}\n", g.root_fingerprint));
    out.push_str("nodes:\n");
    for (id, node) in &g.nodes {
        match node {
            LineageNode::Impression(imp) => {
                out.push_str(&format!(
                    "  - id={} kind=impression source={} schema={} n_rows={}\n",
                    id, imp.source, imp.schema_fingerprint, imp.n_rows
                ));
            }
            LineageNode::Idea(idea) => {
                out.push_str(&format!(
                    "  - id={} kind=idea name={} op={}\n",
                    id, idea.name, idea.transform.op_id
                ));
            }
        }
    }
    out.push_str("edges:\n");
    for e in &g.edges {
        out.push_str(&format!("  - {} -> {} [{}]\n", e.from, e.to, e.label));
    }
    out.push_str("audit:\n");
    for a in &g.audit {
        out.push_str(&format!(
            "  - seq={} kind={} subject={} note={}\n",
            a.seq, a.kind, a.subject_id, a.note
        ));
    }
    out
}

/// Emit the lineage graph as a Quarto/Markdown-friendly Mermaid block.
///
/// The output is deterministic (nodes ordered by their content-addressed
/// id; edges by `(from, to)` order). Impressions are rendered as cylinder
/// nodes (`[(...)]`); ideas as rounded nodes (`(...)`).
///
/// Escapes embedded `"` to `&quot;` so labels are safe to drop straight
/// into a Mermaid block.
pub fn emit_lineage_mermaid(g: &LineageGraph) -> String {
    fn safe(s: &str) -> String {
        s.replace('"', "&quot;").replace('\n', " ")
    }
    fn short_id(id: &FingerprintId) -> String {
        // Mermaid node-id must be a valid identifier — use a prefix + hex.
        format!("n{:016x}", id.0)
    }
    let mut out = String::new();
    out.push_str("```{mermaid}\n");
    out.push_str(&format!("%%| fig-cap: \"Locke lineage graph for {}.\"\n", safe(&g.run_label)));
    out.push_str("flowchart LR\n");
    // Nodes — iterate in BTreeMap (sorted) order for determinism.
    for (id, node) in &g.nodes {
        let nid = short_id(id);
        match node {
            LineageNode::Impression(imp) => {
                out.push_str(&format!(
                    "    {}[(\"{}<br/>id={}<br/>n_rows={}\")]:::imp\n",
                    nid,
                    safe(&imp.source),
                    id,
                    imp.n_rows
                ));
            }
            LineageNode::Idea(idea) => {
                out.push_str(&format!(
                    "    {}([\"{}<br/>op={}<br/>id={}\"]):::idea\n",
                    nid,
                    safe(&idea.name),
                    safe(&idea.transform.op_id),
                    id
                ));
            }
        }
    }
    // Edges — iterate in vector order (already deterministic).
    for e in &g.edges {
        out.push_str(&format!(
            "    {} -->|\"{}\"| {}\n",
            short_id(&e.from),
            safe(&e.label),
            short_id(&e.to)
        ));
    }
    out.push_str("    classDef imp fill:#e8f4fd,stroke:#2196F3,stroke-width:2px\n");
    out.push_str("    classDef idea fill:#fff3e0,stroke:#FF9800,stroke-width:1px\n");
    out.push_str("```\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn imp(source: &str) -> LockeImpression {
        LockeImpression::new(
            source,
            ImpressionKind::Dataset,
            10,
            vec!["x".into(), "y".into()],
        )
    }

    fn idea(name: &str, parents: Vec<FingerprintId>, op: &str) -> LockeIdea {
        LockeIdea::new(
            name,
            TransformationRecord {
                op_id: op.into(),
                params: BTreeMap::new(),
                seed: None,
            },
            parents,
        )
    }

    #[test]
    fn impression_ids_are_deterministic_and_distinct() {
        let a1 = imp("train.csv");
        let a2 = imp("train.csv");
        let b = imp("test.csv");
        assert_eq!(a1.id, a2.id);
        assert_ne!(a1.id, b.id);
    }

    #[test]
    fn idea_with_unknown_parent_fails() {
        let mut b = LineageBuilder::new("run");
        let nonexistent = FingerprintId(0xDEADBEEF);
        let i = idea("filter", vec![nonexistent], "filter");
        assert!(matches!(
            b.add_idea(i),
            Err(LineageError::UnknownParent(_))
        ));
    }

    #[test]
    fn builder_emits_acyclic_graph() {
        let mut b = LineageBuilder::new("run");
        let p = b.add_impression(imp("train.csv")).unwrap();
        let _i = b.add_idea(idea("filter", vec![p], "filter")).unwrap();
        let g = b.finish();
        assert!(g.is_acyclic());
    }

    #[test]
    fn duplicate_add_is_idempotent() {
        let mut b = LineageBuilder::new("run");
        let p = b.add_impression(imp("train.csv")).unwrap();
        let p2 = b.add_impression(imp("train.csv")).unwrap();
        assert_eq!(p, p2);
        assert_eq!(b.finish().nodes.len(), 1);
    }

    #[test]
    fn audit_is_monotonically_sequenced() {
        let mut b = LineageBuilder::new("run");
        let p = b.add_impression(imp("train.csv")).unwrap();
        let _ = b.add_idea(idea("filter", vec![p], "filter")).unwrap();
        let g = b.finish();
        for w in g.audit.windows(2) {
            assert!(w[0].seq < w[1].seq);
        }
    }

    #[test]
    fn graph_root_is_stable_across_repeated_builds() {
        fn build_once() -> FingerprintId {
            let mut b = LineageBuilder::new("run");
            let p = b.add_impression(imp("train.csv")).unwrap();
            let _ = b.add_idea(idea("filter", vec![p], "filter")).unwrap();
            b.finish().root_fingerprint
        }
        assert_eq!(build_once(), build_once());
    }

    #[test]
    fn ancestors_includes_transitive() {
        let mut b = LineageBuilder::new("run");
        let p = b.add_impression(imp("train.csv")).unwrap();
        let f = b.add_idea(idea("filter", vec![p], "filter")).unwrap();
        let m = b.add_idea(idea("mean", vec![f], "mean")).unwrap();
        let g = b.finish();
        let anc = g.ancestors(m);
        assert!(anc.contains(&p));
        assert!(anc.contains(&f));
    }

    // ── Mermaid emit ────────────────────────────────────────────────────

    fn small_graph() -> LineageGraph {
        let mut b = LineageBuilder::new("test-run");
        let p = b.add_impression(imp("train.csv")).unwrap();
        let f = b.add_idea(idea("filter", vec![p], "filter")).unwrap();
        let _m = b.add_idea(idea("mean", vec![f], "mean")).unwrap();
        b.finish()
    }

    #[test]
    fn mermaid_emit_starts_with_fenced_block() {
        let g = small_graph();
        let s = emit_lineage_mermaid(&g);
        assert!(s.starts_with("```{mermaid}\n"));
        assert!(s.ends_with("```\n"));
    }

    #[test]
    fn mermaid_emit_contains_flowchart_keyword() {
        let g = small_graph();
        let s = emit_lineage_mermaid(&g);
        assert!(s.contains("flowchart LR"));
        assert!(s.contains("classDef imp"));
        assert!(s.contains("classDef idea"));
    }

    #[test]
    fn mermaid_emit_is_deterministic() {
        let a = emit_lineage_mermaid(&small_graph());
        let b = emit_lineage_mermaid(&small_graph());
        assert_eq!(a, b);
    }

    #[test]
    fn mermaid_emit_escapes_quotes_in_labels() {
        let mut b = LineageBuilder::new("test-run");
        let _p = b
            .add_impression(LockeImpression::new(
                "evil\"label",
                ImpressionKind::Dataset,
                1,
                vec!["x".into()],
            ))
            .unwrap();
        let g = b.finish();
        let s = emit_lineage_mermaid(&g);
        assert!(s.contains("evil&quot;label"));
        assert!(!s.contains("evil\"label"));
    }

    // ─── B5.2: audit monotonic validator ─────────────────────────────

    #[test]
    fn validate_audit_monotonic_passes_on_builder_constructed_graph() {
        // Graphs from the builder are always monotonic by construction.
        let mut b = LineageBuilder::new("run-1");
        let p1 = b.add_impression(imp("train.csv")).unwrap();
        let p2 = b.add_impression(imp("test.csv")).unwrap();
        b.add_idea(idea("filter", vec![p1], "filter")).unwrap();
        b.add_idea(idea("filter2", vec![p2], "filter")).unwrap();
        let g = b.finish();
        g.validate_audit_monotonic()
            .expect("builder-constructed graph must be monotonic");
    }

    #[test]
    fn validate_audit_monotonic_detects_external_seq_violation() {
        // Hand-construct a graph with a non-monotonic audit chain — the
        // kind of state external `AuditEvent::new` callers could produce.
        let imp1 = imp("train.csv");
        let id1 = imp1.id;
        let nodes: BTreeMap<FingerprintId, LineageNode> =
            [(id1, LineageNode::Impression(imp1))].into_iter().collect();
        let audit = vec![
            AuditEvent::new("run-1", 5, "impression", id1, "first"),
            // BUG: external caller resets seq to 0
            AuditEvent::new("run-1", 0, "audit-note", id1, "rolled-back seq"),
        ];
        let g = LineageGraph {
            run_label: "test".into(),
            nodes,
            edges: vec![],
            audit,
            root_fingerprint: id1,
        };
        let err = g
            .validate_audit_monotonic()
            .expect_err("non-monotonic chain must fail");
        assert_eq!(err.run_label, "run-1");
        assert_eq!(err.prior_seq, 5);
        assert_eq!(err.offending_seq, 0);
    }

    #[test]
    fn validate_audit_monotonic_treats_run_labels_independently() {
        // Two run-labels in the same graph each maintain their own
        // monotonic chain; reusing low seq under a fresh label is fine.
        let imp1 = imp("a.csv");
        let id1 = imp1.id;
        let nodes: BTreeMap<FingerprintId, LineageNode> =
            [(id1, LineageNode::Impression(imp1))].into_iter().collect();
        let audit = vec![
            AuditEvent::new("run-A", 10, "x", id1, ""),
            AuditEvent::new("run-B", 0, "x", id1, ""),
            AuditEvent::new("run-A", 11, "x", id1, ""),
            AuditEvent::new("run-B", 1, "x", id1, ""),
        ];
        let g = LineageGraph {
            run_label: "test".into(),
            nodes,
            edges: vec![],
            audit,
            root_fingerprint: id1,
        };
        g.validate_audit_monotonic()
            .expect("independent run-labels must each be monotonic");
    }
}
