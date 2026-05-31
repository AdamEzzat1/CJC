# ADR-0039 Locke v0.7+ A3 — Governance Workflows

- **Status:** Accepted (2026-05-29)
- **Crate:** `cjc-locke` (extended from ADRs 0028–0038)
- **Companion docs:** [[Locke Belief Reports]], [[Locke Roadmap]] §v0.7+ heavy A3, [[ADR-0038 Locke v0.7+ A2 — Per-Value Category Lineage]]

## Context

Locke produces many findings — **142+ on the diabetes-130 workload** at the §A3 brief's reference cadence. Most are valid concerns the team has *already triaged*: a known-rare category, a column the data-platform team is rebuilding, a column intentionally not yet curated. Without a way to acknowledge those triaged findings, owners cannot run Locke in CI: the alerting volume drowns real regressions.

A3 adds the missing governance layer. Three rule kinds, all declarative:

1. **Suppression** — drop findings matching `(code, column?)` with a recorded `reason`. Every suppression decision is content-addressed so the same suppression on the same finding produces a stable `decision_id` across runs.
2. **Owner annotation** — tag findings with a responsible team/person. Used to route alerts without modifying the finding itself.
3. **Required-finding policy** — assert findings of code X satisfy a threshold. Enforces contractual guarantees like "no E9004 duplicate keys, ever."

Plus an end-to-end gate integration: `cjcl locke gate <ref.json> <current> --policy file.toml` now honours requirement rules alongside the existing appeared-severity gate.

## Decisions

### 1. New module `crates/cjc-locke/src/policy.rs` (~750 LOC)

Public API:

```rust
pub enum RequirementOperator {
    EqZero, LessOrEqual, GreaterOrEqual, Less, Greater, Equal,
}

pub struct SuppressionRule {
    pub code: String,
    pub column: Option<String>,
    pub reason: String,
}

pub struct OwnerRule {
    pub team: String,
    pub column: Option<String>,
    pub code: Option<String>,
}

pub struct RequiredFindingRule {
    pub code: String,
    pub operator: RequirementOperator,
    pub threshold: u64,
    pub owner: Option<String>,
}

pub struct Policy {
    pub suppressions: Vec<SuppressionRule>,
    pub owners: Vec<OwnerRule>,
    pub requirements: Vec<RequiredFindingRule>,
}

pub struct SuppressionDecision {
    pub finding_id: FingerprintId,
    pub rule_index: usize,
    pub decision_id: FingerprintId,
    pub code: String,
    pub column: Option<String>,
    pub reason: String,
    pub severity: FindingSeverity,
}

pub struct OwnerAttribution {
    pub finding_id: FingerprintId,
    pub team: String,
    pub rule_index: usize,
}

pub struct RequirementResult {
    pub code: String,
    pub operator: RequirementOperator,
    pub threshold: u64,
    pub observed: u64,
    pub satisfied: bool,
    pub owner: Option<String>,
}

pub struct PolicyResult {
    pub suppressions: Vec<SuppressionDecision>,
    pub attributions: Vec<OwnerAttribution>,
    pub requirements: Vec<RequirementResult>,
    pub remaining_findings: Vec<ValidationFinding>,
    pub policy_fingerprint: FingerprintId,
}

pub fn apply_policy(report, policy) -> PolicyResult;
pub fn emit_policy_result_text(result) -> String;
```

All types + functions re-exported from the crate root.

### 2. Policy DSL — `.cjcl-locke.toml`

Standard TOML using array-of-tables for repeated entries:

```toml
[[suppress]]
code = "E9082"
column = "weight"           # optional
reason = "real distinction, not typo"

[[owner]]
team = "team-data-platform"
column = "patient_nbr"      # optional
code = "E9001"              # optional

[[require]]
code = "E9004"
operator = "==0"            # ==0 / <= / >= / < / > / == (or lte/gte/lt/gt/eq aliases)
threshold = 0
owner = "team-data-platform"   # optional
```

### 3. `toml_min.rs` extension

The cjc-cli zero-dep minimal TOML parser is extended with `[[name]]` array-of-tables syntax. The new `TomlDoc::array_tables: Vec<(String, TomlTable)>` field collects every `[[name]]` header in declaration order; `TomlDoc::array_tables(name)` returns all entries for a given name. Backward-compatible: existing `[name]` single-table syntax and configs continue to parse identically. `DuplicateTable` errors are scoped to `[name]`-style headers only.

### 4. Audit chain — content-addressed `decision_id`

Every `SuppressionDecision` carries a `decision_id: FingerprintId` computed as `fingerprint_compose("policy_suppress", [finding_id, rule_fingerprint])`. The rule fingerprint hashes `(code, column, reason)`, so:

- Same finding + same rule → same `decision_id`.
- Changing the suppression reason → new `decision_id` (the audit trail remembers the policy revision).
- Two runs of the same dataset against the same policy → byte-identical `decision_id`s.

This preserves Locke's content-addressed-trace property. Downstream consumers (CI gates, dashboards, governance audits) can reason about "is this the same suppression decision the team approved last week?" purely by `decision_id` comparison.

The `Policy::fingerprint()` is the compose of every rule's fingerprint in declaration order — order matters by design, because first-match-wins for suppression and owner rules.

### 5. Determinism

- `PolicyResult::suppressions` preserves the input report's canonical finding order (which is already content-addressed).
- `PolicyResult::attributions` follows the remaining findings' order.
- `PolicyResult::requirements` follows `Policy::requirements` declaration order.
- `attributions_by_team` returns a `BTreeMap` → sorted team iteration.

Bolero structural fuzz over arbitrary suppression / requirement parameters confirms `apply_policy` never panics and is byte-deterministic across consecutive runs.

### 6. Gate integration

`ReportDiff` gains an optional `policy_result: Option<PolicyResult>` field. Two builder methods on `ReportDiff`:

```rust
pub fn with_policy(self, policy: &Policy, current_report: &LockeReport) -> Self;
pub fn policy_gate_fails(&self) -> bool;
```

The CLI's `cjcl locke gate <ref> <current> --policy file.toml` reads the policy, attaches it via `with_policy`, and surfaces a single-line `policy: status=... fingerprint=...` summary in the diff emit. Worst severity escalates to `error` when the policy gate fails, so `--fail-on error` correctly triggers on requirement violations regardless of finding severity.

The five pre-A3 `ReportDiff` fields (appeared / disappeared / unchanged / ref_run_id / cur_run_id / belief_partial_order) are unchanged. Adding a policy is *purely additive*; existing tests pass without modification.

### 7. CLI surface

```
cjcl locke policy apply <data.csv> --policy <file.toml>
cjcl locke gate <ref.json> <current> [--policy <file.toml>] [--fail-on SEV]
```

`policy apply` runs `validate` + `apply_policy` + emits the result text. Worst severity is `error` if any requirement fails. The action is gated under `policy apply` (rather than top-level `cjcl locke policy ...`) so future actions (`lint`, `show`, `diff`) can layer on without breaking parsing.

## Use-case map

| Need | Rule kind | Example |
|---|---|---|
| "We know `weight` is 96.9% `?` and won't fix it this quarter" | suppress E9001 on `weight` | reason = "deferred until v2 ingestion" |
| "Route any `discharge_disposition_id` finding to the clinical team" | owner | team = "clinical", column = "discharge_disposition_id" |
| "There must never be duplicate `patient_nbr` rows in prod data" | require E9004 == 0 | hard guarantee in CI |
| "Allow up to 5 outlier findings before failing" | require E9040 <= 5 | soft guarantee |
| "Re-validate that v2.3's E9080 suppressions still apply post-feature-rollout" | suppress E9080 + decision_id hash | compare last-week's `decision_id` against this week's |

## Consequences

1. **Locke is now operationally usable at 142-finding scale.** Teams can adopt Locke in CI without alert fatigue.
2. **The audit trail is preserved through suppression.** `decision_id` content-addresses each suppression so changes to the policy file produce a visibly different audit trail rather than silently mutating decisions.
3. **Gate integration is additive.** `cjcl locke gate` without `--policy` works exactly as it did in v0.4; with `--policy`, requirements compose with the appeared-severity check.
4. **The DSL is intentionally exact-match.** Wildcards / patterns / suppression expiration / conditional policies are A3.2+ — deliberately scoped out to keep A3.1 reviewable.
5. **TOML parser is the bottleneck for future expansion.** The minimal hand-rolled parser doesn't support inline tables, dotted keys, or datetimes. If A3.2 needs those (e.g., suppression `valid_until = 2026-12-31`), they'll be added incrementally to `toml_min.rs`.

## Out of scope (v0.7+ A3.1)

Deferred to A3.2+:

- **Column wildcards / patterns** (`column = "diag_*"`).
- **Multi-policy file inheritance and merging** (a base policy + project-specific overrides).
- **Suppression expiration dates** (`valid_until = "2026-12-31"` → automatically expire stale acknowledgements).
- **Conditional policies** ("require E9004==0 only when `target_column = readmitted`").
- **`cjcl locke policy lint`** — schema validation + dead-rule warnings.
- **`cjcl locke policy diff`** — compare two policy files and show changed `decision_id`s.

## Test infrastructure

| Layer | Location | New count |
|---|---|---|
| Unit (in-module) | `policy.rs::tests` | 20 |
| Unit (gate integration) | `gate.rs::tests` (post-A1) | 6 |
| Unit (TOML extension) | `crates/cjc-cli/src/toml_min.rs::tests` | 6 |
| Integration | `tests/locke/policy_tests.rs` | 13 |
| Property (proptest) | `tests/locke/locke_proptest.rs` | 1 (determinism over arbitrary suppressions) |
| Bolero structural fuzz | `tests/locke/locke_fuzz.rs` | 1 (arbitrary rules × arbitrary data) |
| CLI parser | `crates/cjc-cli/src/commands/locke.rs::tests` | 12 (4 parser + 8 TOML policy parser) |

## Net delta after v0.7+ A3

- cjc-locke `--lib`: 309 → **335** (+26: 20 policy unit + 6 gate integration)
- tests/locke: 211 → **225** (+13 integration + 1 proptest = +14; bolero target counts inside the same locke_fuzz module)
- cjc-cli `--lib`: 157 → **175** (+12 locke parser + 6 toml_min extension)
- tests/abng: 629 unchanged (no ABNG surface touched)
- Workspace builds clean (2 pre-existing v0.7 part 1 warnings).

Related: [[ADR-0028 Locke Data Skepticism Layer]], [[ADR-0036 Locke v0.7 — Per-axis BeliefScore Composition Algebra]], [[ADR-0037 Locke v0.6.4 — Auto String-Sentinel Detection and Per-Level Leakage]], [[ADR-0038 Locke v0.7+ A2 — Per-Value Category Lineage]], [[Locke Belief Reports]], [[Locke Roadmap]].
