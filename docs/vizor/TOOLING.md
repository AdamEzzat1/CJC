> **Pre-v0.1.4 document.** Uses legacy naming: CJC (now CJC-Lang), `cjc` (now `cjcl`), `.cjc` (now `.cjcl`). Kept unmodified for historical accuracy. See [../REBRAND_NOTICE.md](../REBRAND_NOTICE.md) for the full mapping.

# Vizor IDE / Tooling Integration

## cjc-analyzer: LSP Server

The `cjc-analyzer` crate provides a Language Server Protocol (LSP) skeleton for
IDE integration with CJC code that uses Vizor.

### Features

| Feature        | Status     | Description                               |
|---------------|------------|-------------------------------------------|
| Diagnostics    | Working    | Parse errors shown as you type            |
| Hover          | Working    | Signature + docs for builtins and methods |
| Completion     | Working    | Prefix-based + dot-triggered completions  |
| Go-to-def      | Planned    | Jump to function definitions              |
| Rename         | Planned    | Rename symbols across file                |

### Running the LSP

```bash
cargo build --release -p cjc-analyzer
```

The binary communicates over **stdio** using the standard LSP JSON-RPC protocol.

### Editor Configuration

#### VS Code (manual)

Add to `.vscode/settings.json`:
```json
{
  "cjc.serverPath": "./target/release/cjc-analyzer"
}
```

Or configure a generic LSP client extension to launch `cjc-analyzer`.

#### Neovim (nvim-lspconfig)

```lua
vim.api.nvim_create_autocmd("FileType", {
  pattern = "cjc",
  callback = function()
    vim.lsp.start({
      name = "cjc-analyzer",
      cmd = { "path/to/cjc-analyzer" },
    })
  end,
})
```

### Import-Aware Completions

The analyzer scans source text for `import vizor` lines. When detected, it
populates the completion list with Vizor-specific symbols:

- `vizor_plot_xy` / `vizor_plot` constructors
- All `VizorPlot.*` method completions after typing a `.`
- Hover documentation sourced from `cjc_vizor::docs::vizor_docs()`

Without the import, only core CJC builtins appear.

### Architecture

```
cjc-analyzer/
  src/
    main.rs           -- binary entry point
    server.rs         -- LSP connection loop, message dispatch
    symbol_index.rs   -- BTreeMap-based symbol index (deterministic)
    hover.rs          -- Hover info formatting (markdown)
    completion.rs     -- CompletionItem generation
    diagnostics.rs    -- DiagnosticBag -> LSP Diagnostic conversion
    lib.rs            -- module declarations
```

### Dependencies

The analyzer is the **only** CJC crate allowed external dependencies:
- `lsp-server 0.7` -- JSON-RPC transport
- `lsp-types 0.97` -- LSP protocol types
- `serde_json` -- JSON serialization
- `serde` -- derive support

All other CJC crates remain zero-dependency.

## Future Tooling

### Vizor Preview Server (planned)

A local HTTP server that watches `.cjc` files and serves live SVG previews:
```
cjc-preview --watch examples/scatter.cjc --port 8080
```

### Plot REPL (planned)

Interactive mode where plot commands update a persistent display:
```
vizor> let p = vizor_plot_xy([1,2,3], [4,5,6])
vizor> p.geom_point().title("Interactive")
[SVG rendered in terminal via sixel/kitty protocol]
```
