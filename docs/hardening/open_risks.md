# CJC v0.1 Hardening — Open Risks

## R1: Snap Decoder Unchecked Allocations (High)

**Description:** `cjc_snap::snap_decode()` reads length prefixes from the input byte stream and allocates memory based on them without validating against the remaining input size. Random or malicious input can trigger allocations of many exabytes, causing immediate OOM.

**Impact:** DoS via crafted snap payloads. Fuzz testing of snap_decode with arbitrary bytes is impossible.

**Mitigation:** Add length validation in snap_decode: `if length > remaining_bytes { return Err(...) }`.

**Workaround:** Fuzz tests use roundtrip strategy (encode then decode) or cap input to 8 bytes.

## R2: Builtin Wiring Gap (Medium)

**Description:** The runtime defines ~230 builtin functions, but only ~150 are whitelisted in the eval executor and ~140 in MIR-exec. Unregistered builtins produce "undefined function" errors.

**Impact:** Programs using advanced builtins may fail in one or both executors.

**Mitigation:** Audit `is_known_builtin()` in both executors against the full runtime catalog. Add missing entries.

## R3: Eval Closure Parity (Medium)

**Description:** The AST tree-walk interpreter (cjc-eval) cannot call closures stored in local variables when parsing from source text. MIR-exec handles this correctly.

**Impact:** Eval-MIR parity is broken for programs with closure variables.

**Mitigation:** Fix eval's variable lookup to check for closure values when resolving function calls.

## R4: HashMap Usage (Low)

**Description:** `HashMap`/`HashSet` found in 9+ files across the codebase. These have non-deterministic iteration order.

**Impact:** Potential non-determinism if iteration order of these maps affects output. Currently not triggered in tests.

**Mitigation:** Audit each HashMap usage. Replace with BTreeMap where iteration order matters.

## R5: Parser Semicolon After While (Low)

**Description:** Parser errors on `};` after `while {}` inside function bodies. Three chess RL debug tests are ignored due to this.

**Impact:** Some valid-looking programs fail to parse.

**Mitigation:** Fix parser to accept optional semicolons after block-terminated statements.
