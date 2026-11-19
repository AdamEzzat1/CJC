# Postulate Phase 2: what CJC added

Phase 2 (the companion HUD window) lives in its own workspace,
`C:\Users\adame\postulate`. It talks to CJC only through CJC's binaries. These
are the CJC-side changes it needed. CJC still has no dependency on Postulate.

## `cjcl run --telemetry PATH [--telemetry-every N]`

This appends the Phase 1 telemetry stream to `PATH`: one JSON record per
sampled `backward`, the same records `cjc_ad::telemetry` writes.

- **Sampling**: every 10 steps by default.
- **Flushing**: the CLI uses a ring of one record, so each record is flushed
  as soon as it's taken and a client tailing the file sees training live.
  `postulate-overhead` measured the default sampling rate with a 64-record
  ring, so the per-record write cost here is not part of that measurement.
- **Write-only**: the program's output doesn't change. On
  `examples/01_mlp_xor.cjcl`, output is byte-identical with telemetry on and
  off, and the 200 epochs write 20 records.
- **Scope**: only the default executor (`cjc-eval`). Combining the flag with
  `--multi-file`, `--mir-opt`, `--mir-mono` or `--pinn-weights` is a CLI
  error, not a silent no-op.
- **File mode**: the file is opened append-only (per the spec). A client that
  wants one run per file, like the HUD, chooses a fresh path for each run.

## `cjc-analyzer`

| Addition | What it answers |
|---|---|
| `postulate/telemetry` | The MIR run's telemetry records, parsed, plus `steps` and `every` |
| `postulate/cursorContext` | At a position: the identifier token (lexer), the enclosing `fn` (parser), the innermost enclosing `while`/`for` whose body calls `.backward(` (AST span plus lexer tokens), and the GradGraph nodes whose span contains the position. `nodes` is `null` until this version has been run, because a cursor move never starts a run. |
| `textDocument/documentHighlight` | Occurrences of the identifier under the cursor, matched by name. Editors send this on every cursor move, which is what makes cursor following work. |
| `executorAgreement.telemetry_parity` | The MIR run with telemetry on (sampled every 10 steps) printed the same output and left the same tape bits (`state_bits`) as a third MIR run with telemetry off |
| `runReport.peak_bytes` | Peak live heap bytes above the starting level during one MIR run, from `alloc_meter::CountingAlloc`, which only the `cjc-analyzer` binary installs. It's listed under `measured`, since the LSP transport threads allocate concurrently. A library or test without the meter gets `null`, never a made-up figure. |
| `--companion PATH` | Mirrors the editor's buffer to `PATH.doc` on `didOpen`/`didChange`, and the last cursor position to `PATH` on `hover`/`documentHighlight`/`completion`. Both files are replaced atomically, the doc before the cursor. It only observes messages and never changes a response. |

Each run of a document now takes three runs: eval, MIR, and MIR without
telemetry for the parity check. Each run also samples telemetry every 10
steps instead of never.

## Fix: `cjc-analyzer` outlived its client

`run_server` joined the LSP I/O threads while still holding the
`Connection`. The connection's sender keeps the writer thread alive, so the
join never returned. That meant a server whose client died without sending
`shutdown` (a crashed or killed editor, or a test harness calling
`process::exit`) never exited. Any parent waiting on pipes the orphan had
inherited hung with it. The Phase 2 poison runner hit this. The bug predates
Phase 2 (it is in Phase 1's `server.rs` too). The fix drops the connection
before the join. `crates/cjc-analyzer/tests/process_exit.rs` pins it: it
starts the real binary, initializes, closes stdin without `shutdown`, and
requires the process to exit within 10 s.

## Tests

- `crates/cjc-analyzer/tests/replay.rs`: the Phase 1 LSP replay gate now
  scripts `postulate/telemetry`, two `postulate/cursorContext` requests (one
  inside the training loop, one inside `forward`) and a `documentHighlight`.
  That's 21 messages, still byte-identical across 10 replays once the
  `measured` fields are masked. New assertions cover `telemetry_parity = true`,
  6 records for 60 steps, the training loop found from the AST, and
  `nodes[0].op` being `Sub` or `MatMul` under the cursor.
- `crates/cjc-analyzer/src/cursor.rs`: unit tests for
  - position↔offset round-trips over every byte of a fixture;
  - the training loop being the one that calls `backward`;
  - highlight counts.
- `crates/cjc-analyzer/src/companion.rs`: mirroring the buffer and cursor to
  disk.
- Postulate's `tests/companion.rs` drives the real `cjc-analyzer --companion`
  binary end to end.
