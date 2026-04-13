---
title: Data Systems and CLI
tags: [data, cli, hub]
status: Implemented
---

# Data Systems and CLI

Hub for everything that is *not* the core numeric stack but is used for data wrangling, visualization, serialization, and tooling around CJC-Lang programs.

## Subsystems

| Subsystem | Crate | Note |
|---|---|---|
| DataFrame DSL | `cjc-data` | [[DataFrame DSL]] |
| Visualization | `cjc-vizor` | [[Vizor]] |
| Regex engine | `cjc-regex` | [[Regex Engine]] |
| Binary serialization | `cjc-snap` | [[Binary Serialization]] |
| CLI | `cjc-cli` | [[CLI Surfaces]] |
| REPL | `cjc-cli` | [[REPL]] |
| Language server | `cjc-analyzer` | [[Language Server]] (Experimental) |
| Module system | `cjc-module` | [[Module System]] (Partially implemented) |

## Related

- [[Runtime Architecture]]
- [[Data Systems Source Map]]
