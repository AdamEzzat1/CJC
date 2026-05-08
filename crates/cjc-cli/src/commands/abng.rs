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

    explain <prediction.snap>
        Lineage + abstain reason for a prediction snapshot. Requires
        Routed audit events (Phase 0.4 Track A — 0x1B audit kind).

    train --config <x.toml> [--seed N]
        Run a training driver loop (observe → decide_step →
        checkpoint). Writes <out>.snap and <out>.audit.log.

GLOBAL OPTIONS:
    --help, -h    Print this help message
    --json        Emit JSON output where supported (inspect, diff)
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
                     (the file's first 10 bytes must be `ABNG-PRED\\x01`)",
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
            model_check = Some(ModelCheck {
                model_path: mp.clone(),
                chain_head_match: ok_chain,
                codebook_match: ok_codebook,
                leaf_head_match: ok_head,
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
            if !mc.chain_head_match || !mc.codebook_match || !mc.leaf_head_match {
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
        println!("  format magic:    ABNG-PRED\\x01");
        println!("  model chain_head: {}", hex_of(&snap.model_chain_head));
        println!("  predicting node:  {}", snap.node_id);
        println!("  codebook hash:    {}", hex_of(&snap.codebook_hash));
        println!("  leaf head hash:   {}", hex_of(&snap.leaf_head_hash));
        println!("  blr state hash:   {}", hex_of(&snap.blr_state_hash));
        println!("  blr n_seen:       {}", snap.blr_n_seen);
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
            if !mc.chain_head_match || !mc.codebook_match || !mc.leaf_head_match {
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
        println!("  \"format_magic\": \"ABNG-PRED\\u0001\",");
        println!("  \"chain_head\": \"{}\",", hex_of(&snap.model_chain_head));
        println!("  \"node_id\": {},", snap.node_id);
        println!("  \"codebook_hash\": \"{}\",", hex_of(&snap.codebook_hash));
        println!("  \"leaf_head_hash\": \"{}\",", hex_of(&snap.leaf_head_hash));
        println!("  \"blr_state_hash\": \"{}\",", hex_of(&snap.blr_state_hash));
        println!("  \"blr_n_seen\": {},", snap.blr_n_seen);
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
            println!("    \"leaf_head_match\": {}", mc.leaf_head_match);
            println!("  }}");
        }
        println!("}}");
    }
}

// ── train (stub for later G3.x) ────────────────────────────────────────

mod train {
    pub fn run(_args: &[String]) {
        eprintln!(
            "error: cjcl abng train is not yet shipped — Phase 0.4 Track A \
             G3.8. Build training driver loops in your own .cjcl source \
             using abng_observe / abng_decide_step / abng_serialize for \
             now."
        );
        std::process::exit(2);
    }
}
