//! `cjcl abng …` — ABNG CLI surface (Phase 0.4 Track A).
//!
//! Subcommands:
//! * `inspect <model.snap> [--node ID] [--audit] [--stats] [--tree]` —
//!   read-only viewer. Validates the audit chain, prints header
//!   summary, optionally drills into one node / the audit log / the
//!   tree topology.
//! * `replay <model.snap> [--verify]` — wrapper around
//!   `cjc_abng::serialize::replay`. Reports any `DecodeError` cleanly.
//! * `diff <a.snap> <b.snap>` — topology + per-node fingerprint diff.
//! * `explain <prediction.snap>` — lineage + abstain reason
//!   (requires `Routed` audit events; ships in a later G3.x commit).
//! * `train --config <x.toml> [--seed N]` — driver loop
//!   (observe → decide_step → checkpoint; ships in the final G3.x
//!   commit).
//!
//! All subcommands are read-only with respect to the input snapshot
//! files. `train` is the only writer; it consumes a config + seed and
//! produces `<out>.snap` + `<out>.audit.log`.

use std::fs;
use std::path::Path;
use std::process;

pub fn print_help() {
    println!("\
cjcl abng — ABNG (Adaptive Belief Network Graph) CLI

USAGE:
    cjcl abng <subcommand> [options]

SUBCOMMANDS:
    inspect <model.snap> [--node ID] [--audit] [--stats] [--tree]
        Read-only viewer for an ABNG snapshot. Validates the audit
        chain on load and prints the header summary; flags drill into
        per-node state, the audit log, or the arena topology.

    replay <model.snap> [--verify]
        Replay the snapshot through cjc_abng::serialize::replay and
        report whether the chain reconstructs cleanly.

    diff <a.snap> <b.snap>
        Topology and per-node fingerprint diff between two snapshots.

    explain <prediction.snap> [--model <model.snap>]
        Lineage + abstain reason for a prediction snapshot produced by
        abng_predict_snap. With --model, additionally verifies the
        prediction's chain_head + lineage hashes match the model.

    train [--config CFG.toml] [--seed N] [--n-obs N] [--obs-seed N]
          [--decide-every K] [--max-decide N] --out <PATH>
        Run a deterministic training driver loop (observe N times →
        decide_step every K obs → serialize). Accepts both an
        explicit-flag form and a TOML --config file (sections:
        [graph], [codebook], [leaf_head], [blr_prior], [density],
        [calibration], [decision_policy], [training], [output]).
        Explicit flags override TOML values. Use --help for the
        full TOML schema.

GLOBAL OPTIONS:
    --help, -h    Print this help message
    --json        Emit JSON output where supported (inspect, diff,
                  explain, replay)
");
}

pub fn run(args: &[String]) {
    if args.is_empty() {
        eprintln!("error: cjcl abng requires a subcommand");
        eprintln!("       run `cjcl abng --help` for available subcommands");
        process::exit(2);
    }

    let sub = args[0].as_str();
    let rest = &args[1..];

    match sub {
        "inspect" => inspect::run(rest),
        "replay" => replay::run(rest),
        "diff" => diff::run(rest),
        "explain" => explain::run(rest),
        "train" => train::run(rest),
        "--help" | "-h" => print_help(),
        other => {
            eprintln!("error: unknown abng subcommand `{}`", other);
            eprintln!("       run `cjcl abng --help` for available subcommands");
            process::exit(2);
        }
    }
}

// ── Shared helpers ─────────────────────────────────────────────────────

/// Read a snapshot file and replay it into an `AdaptiveBeliefGraph`.
/// Centralised so every subcommand reports `DecodeError` the same way.
fn load_snapshot(path: &Path) -> cjc_abng::graph::AdaptiveBeliefGraph {
    let bytes = match fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("error: could not read `{}`: {}", path.display(), e);
            process::exit(1);
        }
    };
    match cjc_abng::serialize::replay(&bytes) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("error: replay of `{}` failed: {:?}", path.display(), e);
            eprintln!(
                "       (the snapshot bytes either disagree with the magic-byte \
                 contract for v10 or contain a corrupted hash chain)"
            );
            process::exit(1);
        }
    }
}

/// SHA-256 hex of the raw file bytes — fingerprint independent of
/// the audit chain.
fn file_sha256_hex(bytes: &[u8]) -> String {
    let h = cjc_snap::hash::sha256(bytes);
    hex_of(&h)
}

fn hex_of(bytes: &[u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

// ── inspect ────────────────────────────────────────────────────────────

mod inspect {
    use super::*;
    use cjc_abng::audit::AuditKind;
    use std::collections::BTreeMap;

    struct Args {
        path: String,
        node: Option<u32>,
        audit: bool,
        stats: bool,
        tree: bool,
        json: bool,
    }

    fn parse_args(args: &[String]) -> Args {
        let mut path: Option<String> = None;
        let mut node: Option<u32> = None;
        let mut audit = false;
        let mut stats = false;
        let mut tree = false;
        let mut json = false;
        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "--audit" => audit = true,
                "--stats" => stats = true,
                "--tree" => tree = true,
                "--json" => json = true,
                "--node" => {
                    i += 1;
                    if i >= args.len() {
                        eprintln!("error: --node requires a numeric argument");
                        process::exit(2);
                    }
                    node = Some(args[i].parse().unwrap_or_else(|_| {
                        eprintln!("error: invalid --node value `{}`", args[i]);
                        process::exit(2);
                    }));
                }
                "--help" | "-h" => {
                    println!(
                        "cjcl abng inspect <model.snap> [--node ID] [--audit] \
                         [--stats] [--tree] [--json]"
                    );
                    process::exit(0);
                }
                other if other.starts_with("--") => {
                    eprintln!("error: unknown inspect flag `{}`", other);
                    process::exit(2);
                }
                other => {
                    if path.is_some() {
                        eprintln!(
                            "error: inspect takes one positional snapshot path; got `{}`",
                            other
                        );
                        process::exit(2);
                    }
                    path = Some(other.to_string());
                }
            }
            i += 1;
        }
        let path = path.unwrap_or_else(|| {
            eprintln!("error: cjcl abng inspect requires a snapshot path");
            process::exit(2);
        });
        Args { path, node, audit, stats, tree, json }
    }

    pub fn run(args: &[String]) {
        let a = parse_args(args);
        let p = Path::new(&a.path);
        if !p.exists() {
            eprintln!("error: file `{}` not found", a.path);
            process::exit(1);
        }
        let bytes = match fs::read(p) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("error: read `{}`: {}", a.path, e);
                process::exit(1);
            }
        };
        let g = match cjc_abng::serialize::replay(&bytes) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("error: replay failed: {:?}", e);
                process::exit(1);
            }
        };

        if a.json {
            print_json_summary(&a, &g, &bytes);
        } else {
            print_text_summary(&a, &g, &bytes);
        }
    }

    fn print_text_summary(
        a: &Args,
        g: &cjc_abng::graph::AdaptiveBeliefGraph,
        bytes: &[u8],
    ) {
        println!("ABNG snapshot: {}", a.path);
        println!("  file size:    {} bytes", bytes.len());
        println!("  file sha256:  {}", file_sha256_hex(bytes));
        println!("  magic:        ABNG\\x0A (v10)");
        println!("  seed:         {}", g.seed);
        println!("  epoch:        {}", g.epoch);
        println!("  nodes:        {}", g.nodes.len());
        println!("  audit events: {}", g.audit.len());
        println!("  chain head:   {}", hex_of(&g.chain_head));
        let c = &g.action_counts;
        println!(
            "  action_counts: [grow={} split={} merge={} prune={} compress={} freeze={}]",
            c[0], c[1], c[2], c[3], c[4], c[5]
        );

        if a.audit {
            println!();
            print_audit_summary(g);
        }
        if a.stats {
            println!();
            print_per_node_stats(g);
        }
        if a.tree {
            println!();
            print_tree(g);
        }
        if let Some(nid) = a.node {
            println!();
            print_one_node(g, nid);
        }
    }

    fn print_audit_summary(g: &cjc_abng::graph::AdaptiveBeliefGraph) {
        println!("audit-event histogram:");
        let mut counts: BTreeMap<&'static str, u64> = BTreeMap::new();
        for ev in &g.audit {
            let name = audit_kind_name(&ev.kind);
            *counts.entry(name).or_insert(0) += 1;
        }
        for (name, n) in &counts {
            println!("  {:<32} {}", name, n);
        }
    }

    fn print_per_node_stats(g: &cjc_abng::graph::AdaptiveBeliefGraph) {
        println!("per-node Welford stats (samples / mean / variance):");
        for (i, n) in g.nodes.iter().enumerate() {
            let v = if n.stats.n_seen >= 2 {
                n.stats.m2.finalize() / (n.stats.n_seen - 1) as f64
            } else {
                0.0
            };
            println!(
                "  node {:>3}:  n={:>6}  mean={:>+12.6e}  var={:>12.6e}",
                i, n.stats.n_seen, n.stats.mean, v
            );
        }
    }

    fn print_tree(g: &cjc_abng::graph::AdaptiveBeliefGraph) {
        println!("arena topology (parent → children):");
        for (i, n) in g.nodes.iter().enumerate() {
            let children: Vec<u32> = n
                .children
                .iter()
                .into_iter()
                .map(|(_, id)| id)
                .collect();
            let parent_str = match n.parent {
                Some(p) => format!("{}", p),
                None => "(root)".to_string(),
            };
            let frozen = if n.is_frozen { " [frozen]" } else { "" };
            let inactive = if !n.is_active { " [inactive]" } else { "" };
            println!(
                "  node {:>3}  parent={:>6}  children={:?}{}{}",
                i, parent_str, children, frozen, inactive
            );
        }
    }

    fn print_one_node(g: &cjc_abng::graph::AdaptiveBeliefGraph, nid: u32) {
        if (nid as usize) >= g.nodes.len() {
            eprintln!(
                "error: --node {} out of range (graph has {} nodes)",
                nid,
                g.nodes.len()
            );
            process::exit(1);
        }
        let n = &g.nodes[nid as usize];
        println!("node {}:", nid);
        println!("  parent:         {:?}", n.parent);
        println!("  is_frozen:      {}", n.is_frozen);
        println!("  is_active:      {}", n.is_active);
        println!("  samples_seen:   {}", n.stats.n_seen);
        println!("  mean:           {:+e}", n.stats.mean);
        if n.stats.n_seen >= 2 {
            let v = n.stats.m2.finalize() / (n.stats.n_seen - 1) as f64;
            println!("  variance:       {:+e}", v);
        }
        if let Some(blr) = &n.blr {
            println!("  blr.n_seen:     {}", blr.n_seen);
            println!("  blr.fvh:        {}", hex_of(&blr.feature_version_hash));
        } else {
            println!("  blr:            (none)");
        }
        match n.expected_epistemic {
            Some(v) => println!("  expected_epistemic: {:+e}", v),
            None => println!("  expected_epistemic: (uncaptured)"),
        }
        println!("  signature_stable_calls: {}", n.signature_stable_calls);
    }

    fn print_json_summary(
        a: &Args,
        g: &cjc_abng::graph::AdaptiveBeliefGraph,
        bytes: &[u8],
    ) {
        // Hand-rolled JSON — no external dep. The format is intentionally
        // simple and stable; downstream tooling can rely on it.
        let c = &g.action_counts;
        println!("{{");
        println!("  \"path\": \"{}\",", json_escape(&a.path));
        println!("  \"file_size\": {},", bytes.len());
        println!("  \"file_sha256\": \"{}\",", file_sha256_hex(bytes));
        println!("  \"magic_version\": 10,");
        println!("  \"seed\": {},", g.seed);
        println!("  \"epoch\": {},", g.epoch);
        println!("  \"n_nodes\": {},", g.nodes.len());
        println!("  \"n_events\": {},", g.audit.len());
        println!("  \"chain_head\": \"{}\",", hex_of(&g.chain_head));
        println!(
            "  \"action_counts\": {{\"grow\":{},\"split\":{},\"merge\":{},\"prune\":{},\"compress\":{},\"freeze\":{}}}",
            c[0], c[1], c[2], c[3], c[4], c[5]
        );
        println!("}}");
    }

    fn json_escape(s: &str) -> String {
        let mut out = String::with_capacity(s.len() + 2);
        for ch in s.chars() {
            match ch {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                c if (c as u32) < 0x20 => {
                    out.push_str(&format!("\\u{:04x}", c as u32));
                }
                c => out.push(c),
            }
        }
        out
    }

    /// Stable name for an `AuditKind` discriminant — used in the audit
    /// histogram and (eventually) JSON output.
    fn audit_kind_name(k: &AuditKind) -> &'static str {
        match k {
            AuditKind::Created => "Created",
            AuditKind::BeliefUpdate { .. } => "BeliefUpdate",
            AuditKind::NodeAdded { .. } => "NodeAdded",
            AuditKind::ChildrenPromoted { .. } => "ChildrenPromoted",
            AuditKind::CodebookFrozen { .. } => "CodebookFrozen",
            AuditKind::LeafHeadConfigured { .. } => "LeafHeadConfigured",
            AuditKind::LeafParamsInitialized { .. } => "LeafParamsInitialized",
            AuditKind::LeafParamsUpdated { .. } => "LeafParamsUpdated",
            AuditKind::BlrPriorConfigured { .. } => "BlrPriorConfigured",
            AuditKind::BlrInitialized { .. } => "BlrInitialized",
            AuditKind::BlrUpdated { .. } => "BlrUpdated",
            AuditKind::DensityTrackerInstalled { .. } => "DensityTrackerInstalled",
            AuditKind::DensityUpdated { .. } => "DensityUpdated",
            AuditKind::CalibrationInstalled { .. } => "CalibrationInstalled",
            AuditKind::CalibrationUpdated { .. } => "CalibrationUpdated",
            AuditKind::DriftBaselineFrozen { .. } => "DriftBaselineFrozen",
            AuditKind::Grow { .. } => "Grow",
            AuditKind::Split { .. } => "Split",
            AuditKind::Merge { .. } => "Merge",
            AuditKind::Prune { .. } => "Prune",
            AuditKind::Compress { .. } => "Compress",
            AuditKind::Freeze { .. } => "Freeze",
            AuditKind::Unfreeze { .. } => "Unfreeze",
            AuditKind::ExpectedEpistemicCaptured { .. } => "ExpectedEpistemicCaptured",
            AuditKind::BlrNumericalRescue { .. } => "BlrNumericalRescue",
            AuditKind::LeafParamsUpdatedBatch { .. } => "LeafParamsUpdatedBatch",
            AuditKind::Routed { .. } => "Routed",
            AuditKind::StatsSnapshot { .. } => "StatsSnapshot",
            AuditKind::ProvenanceStamped { .. } => "ProvenanceStamped",
        }
    }
}

// ── replay ─────────────────────────────────────────────────────────────

mod replay {
    use super::*;

    pub fn run(args: &[String]) {
        let mut path: Option<String> = None;
        let mut verify = false;
        let mut json = false;
        for a in args {
            match a.as_str() {
                "--verify" => verify = true,
                "--json" => json = true,
                "--help" | "-h" => {
                    println!("cjcl abng replay <model.snap> [--verify] [--json]");
                    process::exit(0);
                }
                other if other.starts_with("--") => {
                    eprintln!("error: unknown replay flag `{}`", other);
                    process::exit(2);
                }
                other => {
                    if path.is_some() {
                        eprintln!("error: replay takes one positional path; got `{}`", other);
                        process::exit(2);
                    }
                    path = Some(other.to_string());
                }
            }
        }
        let path = path.unwrap_or_else(|| {
            eprintln!("error: cjcl abng replay requires a snapshot path");
            process::exit(2);
        });
        let p = Path::new(&path);
        let bytes = match fs::read(p) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("error: read `{}`: {}", path, e);
                process::exit(1);
            }
        };
        match cjc_abng::serialize::replay(&bytes) {
            Ok(g) => {
                let chain_ok = g.verify_chain().is_ok();
                if json {
                    println!(
                        "{{\"path\":\"{}\",\"replay\":\"ok\",\"chain_verify\":{},\"chain_head\":\"{}\",\"n_events\":{}}}",
                        path, chain_ok, hex_of(&g.chain_head), g.audit.len()
                    );
                } else {
                    println!("replay ok");
                    println!("  path:           {}", path);
                    println!("  chain head:     {}", hex_of(&g.chain_head));
                    println!("  n_events:       {}", g.audit.len());
                    if verify {
                        println!(
                            "  chain verify:   {}",
                            if chain_ok { "OK" } else { "FAILED" }
                        );
                    }
                }
                if verify && !chain_ok {
                    process::exit(1);
                }
            }
            Err(e) => {
                if json {
                    println!(
                        "{{\"path\":\"{}\",\"replay\":\"err\",\"error\":\"{:?}\"}}",
                        path, e
                    );
                } else {
                    eprintln!("error: replay failed: {:?}", e);
                }
                process::exit(1);
            }
        }
    }
}

// ── diff ───────────────────────────────────────────────────────────────

mod diff {
    use super::*;

    pub fn run(args: &[String]) {
        let mut json = false;
        let mut paths: Vec<String> = Vec::new();
        for a in args {
            match a.as_str() {
                "--json" => json = true,
                "--help" | "-h" => {
                    println!("cjcl abng diff <a.snap> <b.snap> [--json]");
                    process::exit(0);
                }
                other if other.starts_with("--") => {
                    eprintln!("error: unknown diff flag `{}`", other);
                    process::exit(2);
                }
                other => paths.push(other.to_string()),
            }
        }
        if paths.len() != 2 {
            eprintln!("error: cjcl abng diff requires exactly two snapshot paths");
            process::exit(2);
        }
        let a = load_snapshot(Path::new(&paths[0]));
        let b = load_snapshot(Path::new(&paths[1]));
        compute_and_print_diff(&paths[0], &a, &paths[1], &b, json);
    }

    fn compute_and_print_diff(
        path_a: &str,
        a: &cjc_abng::graph::AdaptiveBeliefGraph,
        path_b: &str,
        b: &cjc_abng::graph::AdaptiveBeliefGraph,
        json: bool,
    ) {
        let chain_eq = a.chain_head == b.chain_head;
        let n_nodes_eq = a.nodes.len() == b.nodes.len();
        let n_events_eq = a.audit.len() == b.audit.len();
        let action_eq = a.action_counts == b.action_counts;

        // Per-node stats_chain_head fingerprint diff (covers the
        // per-node Welford state without needing the full state hash).
        let common = a.nodes.len().min(b.nodes.len());
        let mut differing_nodes: Vec<u32> = Vec::new();
        for i in 0..common {
            if a.nodes[i].stats_chain_head != b.nodes[i].stats_chain_head {
                differing_nodes.push(i as u32);
            }
        }

        if json {
            println!("{{");
            println!("  \"a\": \"{}\",", path_a);
            println!("  \"b\": \"{}\",", path_b);
            println!("  \"chain_head_equal\": {},", chain_eq);
            println!("  \"n_nodes_equal\": {},", n_nodes_eq);
            println!("  \"n_events_equal\": {},", n_events_eq);
            println!("  \"action_counts_equal\": {},", action_eq);
            println!("  \"differing_node_ids\": {:?}", differing_nodes);
            println!("}}");
        } else {
            println!("ABNG diff: `{}`  vs  `{}`", path_a, path_b);
            println!("  chain_head equal:    {}", yesno(chain_eq));
            println!(
                "  n_nodes:             {} vs {}  ({})",
                a.nodes.len(),
                b.nodes.len(),
                yesno(n_nodes_eq)
            );
            println!(
                "  n_events:            {} vs {}  ({})",
                a.audit.len(),
                b.audit.len(),
                yesno(n_events_eq)
            );
            println!("  action_counts equal: {}", yesno(action_eq));
            if differing_nodes.is_empty() {
                println!("  per-node stats_chain_head: identical");
            } else {
                println!(
                    "  per-node stats_chain_head differs at: {:?}",
                    differing_nodes
                );
            }
        }
        if !chain_eq {
            // Different chain_head → exit non-zero so scripts can gate
            // on bit-equality.
            process::exit(1);
        }
    }

    fn yesno(b: bool) -> &'static str {
        if b { "yes" } else { "NO" }
    }
}

// ── explain ────────────────────────────────────────────────────────────

mod explain {
    use super::*;
    use cjc_abng::predict_snap::{self, PredictionSnap};

    /// Threshold below which a prediction's evidence count is flagged
    /// as "low evidence — consider abstaining". Conservative default;
    /// future versions may make this configurable via DecisionPolicy.
    const LOW_EVIDENCE_N_THRESHOLD: u64 = 30;

    pub fn run(args: &[String]) {
        let mut path: Option<String> = None;
        let mut model_path: Option<String> = None;
        let mut json = false;
        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "--json" => json = true,
                "--model" => {
                    i += 1;
                    if i >= args.len() {
                        eprintln!("error: --model requires a path argument");
                        std::process::exit(2);
                    }
                    model_path = Some(args[i].clone());
                }
                "--help" | "-h" => {
                    println!(
                        "cjcl abng explain <prediction.snap> [--model <model.snap>] [--json]\n\
                         \n\
                         Reads a prediction snapshot produced by abng_predict_snap and \
                         explains the lineage + abstain recommendation. With --model, \
                         additionally verifies that the prediction's chain_head and \
                         lineage hashes match the model snapshot."
                    );
                    std::process::exit(0);
                }
                other if other.starts_with("--") => {
                    eprintln!("error: unknown explain flag `{}`", other);
                    std::process::exit(2);
                }
                other => {
                    if path.is_some() {
                        eprintln!(
                            "error: explain takes one positional prediction-snap path; \
                             got `{}`",
                            other
                        );
                        std::process::exit(2);
                    }
                    path = Some(other.to_string());
                }
            }
            i += 1;
        }
        let path = path.unwrap_or_else(|| {
            eprintln!("error: cjcl abng explain requires a prediction-snap path");
            std::process::exit(2);
        });

        let p = Path::new(&path);
        if !p.exists() {
            eprintln!("error: file `{}` not found", path);
            std::process::exit(1);
        }
        let bytes = match fs::read(p) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("error: read `{}`: {}", path, e);
                std::process::exit(1);
            }
        };
        let snap = match predict_snap::unpack(&bytes) {
            Ok(s) => s,
            Err(e) => {
                eprintln!(
                    "error: prediction-snap decode failed: {:?}\n\
                     (the file's first 10 bytes must be `ABNG-PRED\\x02`)",
                    e
                );
                std::process::exit(1);
            }
        };

        // Optional --model verification.
        let mut model_check: Option<ModelCheck> = None;
        if let Some(mp) = &model_path {
            let mp_path = Path::new(mp);
            if !mp_path.exists() {
                eprintln!("error: --model file `{}` not found", mp);
                std::process::exit(1);
            }
            let g = load_snapshot(mp_path);
            let ok_chain = g.chain_head == snap.model_chain_head;
            let ok_codebook = match &g.codebook {
                Some(cb) => cb.frozen_hash == snap.codebook_hash,
                None => snap.codebook_hash == [0u8; 32],
            };
            let ok_head = match &g.head {
                Some(h) => h.config_hash == snap.leaf_head_hash,
                None => snap.leaf_head_hash == [0u8; 32],
            };
            // Phase 0.5 Item 1 — provenance lineage check. The model's
            // current per-node `provenance_stamp_hash` for the
            // predicting node must match the value the prediction
            // recorded; if the prediction was unstamped (all zeros) we
            // still surface the live model's stamp for diagnosis but
            // count "no stamp on either side" as a match (= no claim).
            let provenance_match = if (snap.node_id as usize) < g.nodes.len() {
                Some(
                    g.nodes[snap.node_id as usize].provenance_stamp_hash
                        == snap.provenance_stamp_hash,
                )
            } else {
                Some(false)
            };
            model_check = Some(ModelCheck {
                model_path: mp.clone(),
                chain_head_match: ok_chain,
                codebook_match: ok_codebook,
                leaf_head_match: ok_head,
                provenance_match,
            });
        }

        let abstain = abstain_reason(&snap);
        if json {
            print_json(&path, &snap, &abstain, model_check.as_ref());
        } else {
            print_text(&path, &snap, &abstain, model_check.as_ref());
        }
        // Exit non-zero if model lineage check failed — gates scripts.
        if let Some(mc) = &model_check {
            let prov_ok = mc.provenance_match.unwrap_or(true);
            if !mc.chain_head_match || !mc.codebook_match || !mc.leaf_head_match
                || !prov_ok
            {
                std::process::exit(1);
            }
        }
    }

    /// Result of an optional `--model` lineage check.
    struct ModelCheck {
        model_path: String,
        chain_head_match: bool,
        codebook_match: bool,
        leaf_head_match: bool,
        /// Phase 0.5 Item 1 — `Some(true)` if the prediction's
        /// `provenance_stamp_hash` matches the live model's stored
        /// stamp on the predicting node; `Some(false)` if they
        /// disagree; `None` only if the predicting node is out of
        /// range on the supplied model (an even worse mismatch — the
        /// caller is asked to look at the chain_head divergence
        /// first).
        provenance_match: Option<bool>,
    }

    /// Categorical interpretation of the prediction's evidence /
    /// uncertainty profile. Drives the "abstain or trust" output.
    enum AbstainReason {
        /// `expected_epistemic` was uncaptured — no calibrated
        /// reference. Caller should call
        /// `abng_force_recapture_expected_epistemic` (or wait for
        /// auto-capture in `decide_step`) before relying on the OOD
        /// signal.
        Uncalibrated,
        /// `blr_n_seen < LOW_EVIDENCE_N_THRESHOLD` — posterior is at
        /// or near the prior; abstain is recommended.
        LowEvidence { n: u64 },
        /// `epistemic_leverage > expected_epistemic` — the calibrated
        /// OOD ratio saturates at 1.0; the input lies in a region the
        /// model has limited evidence for.
        OodSaturated { ratio: f64 },
        /// Posterior is well-trained and the input is within the
        /// known distribution; the prediction is supported by the
        /// evidence.
        Supported {
            n: u64,
            ood_ratio: f64,
        },
    }

    fn abstain_reason(snap: &PredictionSnap) -> AbstainReason {
        if !snap.expected_epistemic.is_finite() {
            return AbstainReason::Uncalibrated;
        }
        if snap.blr_n_seen < LOW_EVIDENCE_N_THRESHOLD {
            return AbstainReason::LowEvidence { n: snap.blr_n_seen };
        }
        let ratio = snap.epistemic_leverage / snap.expected_epistemic;
        if ratio >= 1.0 {
            return AbstainReason::OodSaturated { ratio };
        }
        AbstainReason::Supported {
            n: snap.blr_n_seen,
            ood_ratio: ratio,
        }
    }

    fn print_text(
        path: &str,
        snap: &PredictionSnap,
        abstain: &AbstainReason,
        check: Option<&ModelCheck>,
    ) {
        println!("ABNG prediction snapshot: {}", path);
        println!("  format magic:    ABNG-PRED\\x02");
        println!("  model chain_head: {}", hex_of(&snap.model_chain_head));
        println!("  predicting node:  {}", snap.node_id);
        println!("  codebook hash:    {}", hex_of(&snap.codebook_hash));
        println!("  leaf head hash:   {}", hex_of(&snap.leaf_head_hash));
        println!("  blr state hash:   {}", hex_of(&snap.blr_state_hash));
        println!("  blr n_seen:       {}", snap.blr_n_seen);
        if snap.provenance_stamp_hash == [0u8; 32] {
            println!("  provenance stamp: <unstamped>");
        } else {
            println!(
                "  provenance stamp: {}",
                hex_of(&snap.provenance_stamp_hash)
            );
        }
        println!("  phi (d={}):", snap.phi.len());
        for (i, x) in snap.phi.iter().enumerate() {
            println!("      phi[{:2}] = {:+e}", i, x);
        }
        println!();
        println!("prediction:");
        println!("  mean:              {:+e}", snap.mean);
        println!("  epistemic leverage: {:+e}", snap.epistemic_leverage);
        println!("  aleatoric variance: {:+e}", snap.aleatoric_var);
        if snap.expected_epistemic.is_finite() {
            println!(
                "  expected leverage:  {:+e}  (calibrated reference)",
                snap.expected_epistemic
            );
        } else {
            println!("  expected leverage:  uncaptured (NaN sentinel)");
        }

        println!();
        match abstain {
            AbstainReason::Uncalibrated => {
                println!(
                    "abstain: UNCALIBRATED — expected_epistemic was not captured at \
                     predict time. The OOD ratio cannot be computed; consider running \
                     abng_force_recapture_expected_epistemic on the predicting node \
                     and re-issuing the prediction."
                );
            }
            AbstainReason::LowEvidence { n } => {
                println!(
                    "abstain: LOW EVIDENCE — predicting node has only {} observations \
                     (threshold {}). Posterior is near the prior; predictions are \
                     dominated by the prior precision and may be unreliable.",
                    n, LOW_EVIDENCE_N_THRESHOLD
                );
            }
            AbstainReason::OodSaturated { ratio } => {
                println!(
                    "abstain: OOD SATURATED — calibrated OOD ratio (lev/expected) = \
                     {:.3} ≥ 1.0. The input lies outside the region the model has \
                     trained on; predictions may be unreliable.",
                    ratio
                );
            }
            AbstainReason::Supported { n, ood_ratio } => {
                println!(
                    "trust:   SUPPORTED — n_seen = {}, calibrated OOD ratio = {:.3} \
                     (< 1.0). The prediction is within the model's trained \
                     distribution.",
                    n, ood_ratio
                );
            }
        }

        if let Some(mc) = check {
            println!();
            println!("model lineage check (--model {}):", mc.model_path);
            println!(
                "  chain_head match:    {}",
                if mc.chain_head_match { "yes" } else { "NO" }
            );
            println!(
                "  codebook hash match: {}",
                if mc.codebook_match { "yes" } else { "NO" }
            );
            println!(
                "  leaf head match:     {}",
                if mc.leaf_head_match { "yes" } else { "NO" }
            );
            // Phase 0.5 Item 1 — provenance lineage row.
            match mc.provenance_match {
                Some(true) => println!("  provenance match:    yes"),
                Some(false) => println!("  provenance match:    NO"),
                None => println!("  provenance match:    n/a (predicting node out of range)"),
            }
            let prov_ok = mc.provenance_match.unwrap_or(true);
            if !mc.chain_head_match
                || !mc.codebook_match
                || !mc.leaf_head_match
                || !prov_ok
            {
                println!(
                    "  WARNING: one or more lineage hashes do NOT match the \
                     prediction snapshot. The prediction was made against a \
                     different model state — explain output may not reflect \
                     the model's current behaviour."
                );
            }
        }
    }

    fn print_json(
        path: &str,
        snap: &PredictionSnap,
        abstain: &AbstainReason,
        check: Option<&ModelCheck>,
    ) {
        let abstain_kind = match abstain {
            AbstainReason::Uncalibrated => "uncalibrated",
            AbstainReason::LowEvidence { .. } => "low_evidence",
            AbstainReason::OodSaturated { .. } => "ood_saturated",
            AbstainReason::Supported { .. } => "supported",
        };
        let ood_ratio_str = match abstain {
            AbstainReason::OodSaturated { ratio } => format!("{}", ratio),
            AbstainReason::Supported { ood_ratio, .. } => format!("{}", ood_ratio),
            _ => "null".to_string(),
        };
        println!("{{");
        println!("  \"path\": \"{}\",", path);
        println!("  \"format_magic\": \"ABNG-PRED\\u0002\",");
        println!("  \"chain_head\": \"{}\",", hex_of(&snap.model_chain_head));
        println!("  \"node_id\": {},", snap.node_id);
        println!("  \"codebook_hash\": \"{}\",", hex_of(&snap.codebook_hash));
        println!("  \"leaf_head_hash\": \"{}\",", hex_of(&snap.leaf_head_hash));
        println!("  \"blr_state_hash\": \"{}\",", hex_of(&snap.blr_state_hash));
        println!("  \"blr_n_seen\": {},", snap.blr_n_seen);
        // Phase 0.5 Item 1 — emit provenance stamp; null when unstamped.
        if snap.provenance_stamp_hash == [0u8; 32] {
            println!("  \"provenance_stamp_hash\": null,");
        } else {
            println!(
                "  \"provenance_stamp_hash\": \"{}\",",
                hex_of(&snap.provenance_stamp_hash)
            );
        }
        println!("  \"phi_dim\": {},", snap.phi.len());
        println!("  \"prediction\": {{");
        println!("    \"mean\": {},", snap.mean);
        println!(
            "    \"epistemic_leverage\": {},",
            snap.epistemic_leverage
        );
        println!("    \"aleatoric_var\": {}", snap.aleatoric_var);
        println!("  }},");
        if snap.expected_epistemic.is_finite() {
            println!("  \"expected_epistemic\": {},", snap.expected_epistemic);
        } else {
            println!("  \"expected_epistemic\": null,");
        }
        println!("  \"abstain\": \"{}\",", abstain_kind);
        println!("  \"ood_ratio\": {}", ood_ratio_str);
        if let Some(mc) = check {
            println!(",  \"model_check\": {{");
            println!("    \"path\": \"{}\",", mc.model_path);
            println!("    \"chain_head_match\": {},", mc.chain_head_match);
            println!("    \"codebook_match\": {},", mc.codebook_match);
            println!("    \"leaf_head_match\": {},", mc.leaf_head_match);
            match mc.provenance_match {
                Some(b) => println!("    \"provenance_match\": {}", b),
                None => println!("    \"provenance_match\": null"),
            }
            println!("  }}");
        }
        println!("}}");
    }
}

// ── train ──────────────────────────────────────────────────────────────

mod train {
    use super::*;
    use cjc_abng::graph::AdaptiveBeliefGraph;
    use cjc_abng::serialize;
    use cjc_ad::pinn::Activation;
    use cjc_repro::Rng;
    use crate::toml_min::{self, TomlDoc, TomlValue};

    /// Phase 0.5 Item 3 — buildable graph specification. Either
    /// constructed from the canary defaults (when no `--config` is
    /// passed) or from a TOML config file with optional CLI flag
    /// overlays.
    #[derive(Clone, Debug)]
    struct GraphSpec {
        /// `(n_dims, n_bins, boundaries)` for `set_codebook`.
        codebook: (usize, u16, Vec<f64>),
        /// `(input_dim, hidden_dims, output_dim, activation)` for
        /// `set_leaf_head`.
        leaf_head: (u32, Vec<u32>, u32, Activation),
        /// `(a, b, sigma_init)` for `set_blr_prior`.
        blr_prior: (f64, f64, f64),
        /// Whether to call `set_density_tracker`.
        density_enabled: bool,
        /// `n_bins` for `set_calibration`. None means skip.
        calibration_n_bins: Option<u8>,
        /// 14-element threshold tensor for `set_decision_policy`.
        decision_policy_thresholds: Vec<f64>,
        /// `(parent_id, key_byte)` pairs for `add_node`. Applied in
        /// declaration order; the canary's default is `[(0, 1), (0, 2)]`.
        add_nodes: Vec<(u32, u8)>,
    }

    impl GraphSpec {
        /// Defaults match the `decide_step_canary_tests` scenario so the
        /// out-of-the-box `cjcl abng train --out model.snap` produces a
        /// graph identical to what the canary's `chain_head` lock-in
        /// expects (when run with the same seed).
        fn canary_default() -> Self {
            Self {
                codebook: (1, 4, vec![-1.0, 0.0, 1.0]),
                leaf_head: (1, vec![2], 1, Activation::Tanh),
                blr_prior: (1.0, 1.5, 1.0),
                density_enabled: true,
                calibration_n_bins: Some(15),
                decision_policy_thresholds: vec![
                    0.5, 64.0, 128.0, 0.05, 0.02, 4.0, 0.1, 32.0, 10.0, 8.0, 20.0,
                    f64::MAX,
                    // v11 (post-Track-A) — ECE/sigma stability thresholds.
                    0.005, 1.05,
                ],
                add_nodes: vec![(0, 1), (0, 2)],
            }
        }
    }

    /// Phase 0.5 Item 3 — load a [`GraphSpec`] from a TOML config
    /// file. Missing sections fall back to the canary defaults; only
    /// keys explicitly present in the TOML override.
    fn graph_spec_from_toml(doc: &TomlDoc) -> Result<GraphSpec, String> {
        let mut spec = GraphSpec::canary_default();
        // [codebook]
        if let Some(t) = doc.table("codebook") {
            let n_dims = lookup_int(t, "n_dims")?;
            let n_bins = lookup_int(t, "n_bins")?;
            let boundaries = lookup_array_f64(t, "boundaries")?;
            spec.codebook = (
                n_dims as usize,
                n_bins as u16,
                boundaries,
            );
        }
        // [leaf_head]
        if let Some(t) = doc.table("leaf_head") {
            let input_dim = lookup_int(t, "input_dim")? as u32;
            let hidden_dims_raw = lookup_array_int(t, "hidden_dims")?;
            let hidden_dims: Vec<u32> = hidden_dims_raw.into_iter().map(|x| x as u32).collect();
            let output_dim = lookup_int(t, "output_dim")? as u32;
            let act_str = lookup_string(t, "activation")?;
            let activation = parse_activation(&act_str)?;
            spec.leaf_head = (input_dim, hidden_dims, output_dim, activation);
        }
        // [blr_prior]
        if let Some(t) = doc.table("blr_prior") {
            let a = lookup_f64(t, "a")?;
            let b = lookup_f64(t, "b")?;
            let sigma_init = lookup_f64(t, "sigma_init")?;
            spec.blr_prior = (a, b, sigma_init);
        }
        // [density]
        if let Some(t) = doc.table("density") {
            spec.density_enabled = lookup_bool(t, "enabled")?;
        }
        // [calibration]
        if let Some(t) = doc.table("calibration") {
            let n_bins = lookup_int(t, "n_bins")?;
            spec.calibration_n_bins = Some(n_bins as u8);
        }
        // [decision_policy]
        if let Some(t) = doc.table("decision_policy") {
            let thresholds = lookup_array_f64(t, "thresholds")?;
            if thresholds.len() != 14 {
                return Err(format!(
                    "[decision_policy].thresholds must have exactly 14 elements (got {})",
                    thresholds.len()
                ));
            }
            spec.decision_policy_thresholds = thresholds;
        }
        // [graph] add_nodes (array of int-pair arrays).
        if let Some(t) = doc.table("graph") {
            if let Some(v) = t.iter().find(|(k, _)| k == "add_nodes") {
                let outer = v
                    .1
                    .as_array()
                    .ok_or_else(|| format!("[graph].add_nodes must be array, got {}", v.1.type_name()))?;
                let mut pairs = Vec::with_capacity(outer.len());
                for (i, pair) in outer.iter().enumerate() {
                    let inner = pair.as_array().ok_or_else(|| {
                        format!(
                            "[graph].add_nodes[{i}] must be a [parent, key] pair, got {}",
                            pair.type_name()
                        )
                    })?;
                    if inner.len() != 2 {
                        return Err(format!(
                            "[graph].add_nodes[{i}] must have exactly 2 elements (got {})",
                            inner.len()
                        ));
                    }
                    let parent = inner[0]
                        .as_int()
                        .ok_or_else(|| format!("[graph].add_nodes[{i}][0] must be int"))?
                        as u32;
                    let key = inner[1]
                        .as_int()
                        .ok_or_else(|| format!("[graph].add_nodes[{i}][1] must be int"))?
                        as u8;
                    pairs.push((parent, key));
                }
                spec.add_nodes = pairs;
            }
        }
        Ok(spec)
    }

    fn lookup_int(t: &toml_min::TomlTable, key: &str) -> Result<i64, String> {
        t.iter()
            .find_map(|(k, v)| if k == key { Some(v) } else { None })
            .ok_or_else(|| format!("missing key `{key}` in TOML table"))?
            .as_int()
            .ok_or_else(|| format!("key `{key}` must be integer"))
    }
    fn lookup_f64(t: &toml_min::TomlTable, key: &str) -> Result<f64, String> {
        t.iter()
            .find_map(|(k, v)| if k == key { Some(v) } else { None })
            .ok_or_else(|| format!("missing key `{key}` in TOML table"))?
            .as_f64()
            .ok_or_else(|| format!("key `{key}` must be number"))
    }
    fn lookup_bool(t: &toml_min::TomlTable, key: &str) -> Result<bool, String> {
        t.iter()
            .find_map(|(k, v)| if k == key { Some(v) } else { None })
            .ok_or_else(|| format!("missing key `{key}` in TOML table"))?
            .as_bool()
            .ok_or_else(|| format!("key `{key}` must be bool"))
    }
    fn lookup_string(t: &toml_min::TomlTable, key: &str) -> Result<String, String> {
        t.iter()
            .find_map(|(k, v)| if k == key { Some(v) } else { None })
            .ok_or_else(|| format!("missing key `{key}` in TOML table"))?
            .as_str()
            .map(|s| s.to_string())
            .ok_or_else(|| format!("key `{key}` must be string"))
    }
    fn lookup_array_f64(t: &toml_min::TomlTable, key: &str) -> Result<Vec<f64>, String> {
        let arr = t
            .iter()
            .find_map(|(k, v)| if k == key { Some(v) } else { None })
            .ok_or_else(|| format!("missing key `{key}` in TOML table"))?
            .as_array()
            .ok_or_else(|| format!("key `{key}` must be array"))?;
        let mut out = Vec::with_capacity(arr.len());
        for (i, v) in arr.iter().enumerate() {
            out.push(
                v.as_f64()
                    .ok_or_else(|| format!("`{key}[{i}]` must be number, got {}", v.type_name()))?,
            );
        }
        Ok(out)
    }
    fn lookup_array_int(t: &toml_min::TomlTable, key: &str) -> Result<Vec<i64>, String> {
        let arr = t
            .iter()
            .find_map(|(k, v)| if k == key { Some(v) } else { None })
            .ok_or_else(|| format!("missing key `{key}` in TOML table"))?
            .as_array()
            .ok_or_else(|| format!("key `{key}` must be array"))?;
        let mut out = Vec::with_capacity(arr.len());
        for (i, v) in arr.iter().enumerate() {
            out.push(
                v.as_int()
                    .ok_or_else(|| format!("`{key}[{i}]` must be int, got {}", v.type_name()))?,
            );
        }
        Ok(out)
    }
    fn parse_activation(s: &str) -> Result<Activation, String> {
        match s.to_ascii_lowercase().as_str() {
            "tanh" => Ok(Activation::Tanh),
            "sigmoid" => Ok(Activation::Sigmoid),
            "relu" => Ok(Activation::Relu),
            "none" => Ok(Activation::None),
            "gelu" => Ok(Activation::Gelu),
            "silu" | "swish" => Ok(Activation::Silu),
            "elu" => Ok(Activation::Elu),
            "selu" => Ok(Activation::Selu),
            other => Err(format!(
                "unknown activation `{other}` (expected one of: tanh, sigmoid, relu, none, gelu, silu, elu, selu)"
            )),
        }
    }

    struct TrainArgs {
        seed: u64,
        n_observations: u64,
        observation_seed: u64,
        decide_step_every: u64,
        max_decide_steps: u64,
        out_path: String,
        spec: GraphSpec,
    }

    fn parse_args(args: &[String]) -> TrainArgs {
        // First-pass: load TOML config if `--config <path>` is
        // present. The TOML provides defaults for spec + scalar
        // fields; subsequent CLI flags overlay on top (back-compat
        // contract from the handoff).
        let mut config_path: Option<String> = None;
        {
            let mut j = 0;
            while j < args.len() {
                if args[j] == "--config" {
                    if j + 1 >= args.len() {
                        eprintln!("error: --config requires a path argument");
                        process::exit(2);
                    }
                    config_path = Some(args[j + 1].clone());
                    j += 2;
                } else {
                    j += 1;
                }
            }
        }
        // Defaults — either from TOML or from canary fixtures.
        let mut seed: u64 = 42;
        let mut n_observations: u64 = 100;
        let mut observation_seed: u64 = 42;
        let mut decide_step_every: u64 = 25;
        let mut max_decide_steps: u64 = u64::MAX;
        let mut out_path: Option<String> = None;
        let mut spec = GraphSpec::canary_default();
        if let Some(path) = &config_path {
            let p = Path::new(path);
            if !p.exists() {
                eprintln!("error: --config file `{path}` not found");
                process::exit(1);
            }
            let src = match fs::read_to_string(p) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("error: read `{path}`: {e}");
                    process::exit(1);
                }
            };
            let doc = match toml_min::parse(&src) {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("error: parse `{path}`: {e}");
                    process::exit(2);
                }
            };
            // Spec overrides (from [codebook], [leaf_head], etc.).
            spec = match graph_spec_from_toml(&doc) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("error: invalid config in `{path}`: {e}");
                    process::exit(2);
                }
            };
            // Scalar overrides.
            if let Some(v) = doc.get("graph", "seed") {
                seed = v.as_int().unwrap_or_else(|| {
                    eprintln!("error: [graph].seed must be int");
                    process::exit(2);
                }) as u64;
            }
            if let Some(v) = doc.get("training", "n_observations") {
                n_observations = v.as_int().unwrap_or(0) as u64;
            }
            if let Some(v) = doc.get("training", "observation_seed") {
                observation_seed = v.as_int().unwrap_or(0) as u64;
            }
            if let Some(v) = doc.get("training", "decide_step_every") {
                let val = v.as_int().unwrap_or(0);
                if val <= 0 {
                    eprintln!("error: [training].decide_step_every must be > 0");
                    process::exit(2);
                }
                decide_step_every = val as u64;
            }
            if let Some(v) = doc.get("training", "max_decide_steps") {
                max_decide_steps = v.as_int().unwrap_or(0) as u64;
            }
            if let Some(v) = doc.get("output", "path") {
                out_path = v.as_str().map(|s| s.to_string());
            }
        }
        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "--seed" => {
                    i += 1;
                    if i >= args.len() {
                        eprintln!("error: --seed requires a numeric argument");
                        process::exit(2);
                    }
                    seed = args[i].parse().unwrap_or_else(|_| {
                        eprintln!("error: invalid --seed value `{}`", args[i]);
                        process::exit(2);
                    });
                }
                "--n-obs" | "--n-observations" => {
                    i += 1;
                    if i >= args.len() {
                        eprintln!("error: --n-obs requires a numeric argument");
                        process::exit(2);
                    }
                    n_observations = args[i].parse().unwrap_or_else(|_| {
                        eprintln!("error: invalid --n-obs value `{}`", args[i]);
                        process::exit(2);
                    });
                }
                "--obs-seed" | "--observation-seed" => {
                    i += 1;
                    if i >= args.len() {
                        eprintln!("error: --obs-seed requires a numeric argument");
                        process::exit(2);
                    }
                    observation_seed = args[i].parse().unwrap_or_else(|_| {
                        eprintln!("error: invalid --obs-seed value `{}`", args[i]);
                        process::exit(2);
                    });
                }
                "--decide-every" => {
                    i += 1;
                    if i >= args.len() {
                        eprintln!("error: --decide-every requires a numeric argument");
                        process::exit(2);
                    }
                    decide_step_every = args[i].parse().unwrap_or_else(|_| {
                        eprintln!("error: invalid --decide-every value `{}`", args[i]);
                        process::exit(2);
                    });
                    if decide_step_every == 0 {
                        eprintln!("error: --decide-every must be > 0");
                        process::exit(2);
                    }
                }
                "--max-decide" => {
                    i += 1;
                    if i >= args.len() {
                        eprintln!("error: --max-decide requires a numeric argument");
                        process::exit(2);
                    }
                    max_decide_steps = args[i].parse().unwrap_or_else(|_| {
                        eprintln!("error: invalid --max-decide value `{}`", args[i]);
                        process::exit(2);
                    });
                }
                "--out" | "-o" => {
                    i += 1;
                    if i >= args.len() {
                        eprintln!("error: --out requires a path argument");
                        process::exit(2);
                    }
                    out_path = Some(args[i].clone());
                }
                "--config" => {
                    // Consumed by the pre-pass (TOML loader) above.
                    // We just need to skip the path argument here so
                    // it isn't re-parsed as a positional.
                    i += 1;
                    if i >= args.len() {
                        eprintln!("error: --config requires a path argument");
                        process::exit(2);
                    }
                }
                "--help" | "-h" => {
                    println!("\
cjcl abng train [OPTIONS]

Run a deterministic training driver loop:
  observe N times → decide_step every K observations → serialize.

OPTIONS:
    --config <PATH>       TOML config file. Sections (all optional):
                          [graph] seed, add_nodes
                          [codebook] n_dims, n_bins, boundaries
                          [leaf_head] input_dim, hidden_dims, output_dim, activation
                          [blr_prior] a, b, sigma_init
                          [density] enabled
                          [calibration] n_bins
                          [decision_policy] thresholds (length 14)
                          [training] n_observations, observation_seed,
                                     decide_step_every, max_decide_steps
                          [output] path
                          Explicit flags below override TOML values.
    --seed <N>            graph seed (default: 42)
    --n-obs <N>           number of observations (default: 100)
    --obs-seed <N>        observation-stream seed (default: 42)
    --decide-every <K>    run decide_step after every K observations
                          (default: 25)
    --max-decide <N>      cap on the total number of decide_step calls
                          (default: unbounded)
    --out, -o <PATH>      output snapshot path (REQUIRED)

DEFAULTS produce a graph matching the decide_step_canary_tests
fixture: 1-D codebook with 4 bins, 1→2→1 tanh head, BLR(1, 1.5, 1),
density tracker + 15-bin calibration + DecisionPolicy with
drift_unfreeze disabled, two children added with key bytes 1 and 2
before observation begins.
");
                    process::exit(0);
                }
                other if other.starts_with("--") => {
                    eprintln!("error: unknown train flag `{}`", other);
                    process::exit(2);
                }
                other => {
                    eprintln!("error: unexpected positional argument `{}`", other);
                    process::exit(2);
                }
            }
            i += 1;
        }
        let out_path = out_path.unwrap_or_else(|| {
            eprintln!("error: cjcl abng train requires --out <PATH>");
            process::exit(2);
        });
        TrainArgs {
            seed,
            n_observations,
            observation_seed,
            decide_step_every,
            max_decide_steps,
            out_path,
            spec,
        }
    }

    pub fn run(args: &[String]) {
        let a = parse_args(args);

        // Build the graph from the resolved spec (canary defaults +
        // optional [TOML] overrides + explicit flag overrides).
        let mut g = AdaptiveBeliefGraph::new(a.seed);
        let (cb_n_dims, cb_n_bins, cb_boundaries) = &a.spec.codebook;
        if let Err(e) = g.set_codebook(*cb_n_dims, *cb_n_bins, cb_boundaries) {
            eprintln!("error: set_codebook failed: {:?}", e);
            process::exit(1);
        }
        let (lh_in, lh_hidden, lh_out, lh_act) = &a.spec.leaf_head;
        if let Err(e) = g.set_leaf_head(*lh_in, lh_hidden.clone(), *lh_out, *lh_act) {
            eprintln!("error: set_leaf_head failed: {:?}", e);
            process::exit(1);
        }
        let (blr_a, blr_b, blr_sigma) = a.spec.blr_prior;
        if let Err(e) = g.set_blr_prior(blr_a, blr_b, blr_sigma) {
            eprintln!("error: set_blr_prior failed: {:?}", e);
            process::exit(1);
        }
        if a.spec.density_enabled {
            if let Err(e) = g.set_density_tracker() {
                eprintln!("error: set_density_tracker failed: {:?}", e);
                process::exit(1);
            }
        }
        if let Some(n_bins) = a.spec.calibration_n_bins {
            if let Err(e) = g.set_calibration(n_bins) {
                eprintln!("error: set_calibration failed: {:?}", e);
                process::exit(1);
            }
        }
        if let Err(e) = g.set_decision_policy(&a.spec.decision_policy_thresholds) {
            eprintln!("error: set_decision_policy failed: {:?}", e);
            process::exit(1);
        }
        for (parent, key) in &a.spec.add_nodes {
            if let Err(e) = g.add_node(*parent, *key) {
                eprintln!("error: add_node({parent}, {key}) failed: {:?}", e);
                process::exit(1);
            }
        }

        // Observation stream is a deterministic SplitMix64 sequence
        // mapped to [-1, 1]. Every (seed, n_observations) combination
        // produces the same byte sequence across runs and platforms.
        let mut rng = Rng::seeded(a.observation_seed);
        let mut decide_calls: u64 = 0;
        let mut total_actions = [0u64; 6];
        // Phase 0.5 Item 3 — observations now cycle across whatever
        // number of nodes the spec produces (was a hardcoded `% 3`).
        // For the default canary spec (root + 2 children) this is
        // bit-identical to the v11 behavior; for TOML-customized
        // specs with more or fewer children it round-robins through
        // them all.
        let n_nodes_for_obs = g.node_count();
        for k in 0..a.n_observations {
            let raw = rng.next_f64();          // [0, 1)
            let value = raw * 2.0 - 1.0;       // [-1, 1)
            let target_node = (k % n_nodes_for_obs as u64) as u32;
            if let Err(e) = g.observe(target_node, value) {
                eprintln!("error: observe(node={target_node}) failed: {:?}", e);
                process::exit(1);
            }
            // Run decide_step at the configured cadence.
            if (k + 1) % a.decide_step_every == 0
                && decide_calls < a.max_decide_steps
            {
                let counts = g.decide_step();
                for i in 0..6 {
                    total_actions[i] = total_actions[i].saturating_add(counts[i]);
                }
                decide_calls += 1;
            }
        }

        // Serialize + write.
        let blob = serialize::serialize(&g);
        if let Err(e) = std::fs::write(&a.out_path, &blob) {
            eprintln!("error: write `{}` failed: {}", a.out_path, e);
            process::exit(1);
        }

        // Summary to stdout (machine-friendly: each line is a single
        // key: value pair so shell scripts can grep/awk).
        println!("ok");
        println!("path:           {}", a.out_path);
        println!("size:           {} bytes", blob.len());
        println!("seed:           {}", a.seed);
        println!("n_observations: {}", a.n_observations);
        println!("decide_calls:   {}", decide_calls);
        println!("chain_head:     {}", hex_of(&g.chain_head));
        println!("n_nodes:        {}", g.nodes.len());
        println!("n_events:       {}", g.audit.len());
        println!(
            "action_counts:  grow={} split={} merge={} prune={} compress={} freeze={}",
            total_actions[0],
            total_actions[1],
            total_actions[2],
            total_actions[3],
            total_actions[4],
            total_actions[5]
        );
    }
}
