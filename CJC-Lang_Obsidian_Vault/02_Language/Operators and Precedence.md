---
title: Operators and Precedence
tags: [language, operators]
status: Implemented
---

# Operators and Precedence

CJC-Lang uses a Pratt parser (see [[Parser]]) with explicit binding powers per operator.

## Operator families

| Family | Operators |
|---|---|
| Unary | `- !  ~` |
| Multiplicative | `* / %` |
| Additive | `+ -` |
| Power | `**` (right-associative) |
| Shift | `<< >>` |
| Bitwise | `& ^ \|` |
| Comparison | `< > <= >= == !=` |
| Logical | `&& \|\|` |
| Pipe | `\|>` |
| Regex match | `~=` `!~` |
| Assignment | `=`, `+=`, `-=`, `*=`, `/=` |

Exact precedence is defined by the Pratt binding-power table in `crates/cjc-parser/src/lib.rs`. **See source** for authoritative ordering.

## Overloading and dispatch

Operators route through [[Dispatch Layer]] for polymorphic resolution. The dispatcher uses a `Specificity` ranking (`None → Generic → Constrained → Concrete`) to pick the most specific overload. A coherence checker detects overlapping definitions.

For example, `a + b` where `a, b: Tensor` dispatches to the tensor-specific addition kernel in [[Tensor Runtime]]. Where `a: f64, b: Tensor`, dispatch picks the scalar-tensor broadcast kernel.

## Pipe operator

`data |> f() |> g()` is desugared in [[HIR]] to `g(f(data))`. This is *syntax only* — there is no special runtime semantics.

## Related

- [[Syntax]]
- [[Parser]]
- [[Dispatch Layer]]
- [[Tensor Runtime]]
