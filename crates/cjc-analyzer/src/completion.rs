//! Completion provider — generates completion items for the LSP.
//!
//! Completion sources:
//! 1. Builtins (always available)
//! 2. Library symbols (when `import <lib>` is in scope)
//! 3. User-defined functions/variables from the AST
//! 4. Method completions after `.` on known types

use lsp_types::{CompletionItem, CompletionItemKind};

use crate::symbol_index::{SymbolIndex, SymbolKind};

/// Generate completion items for a given prefix.
pub fn complete(index: &SymbolIndex, prefix: &str) -> Vec<CompletionItem> {
    index
        .completions(prefix)
        .into_iter()
        .map(|sym| {
            let kind = match sym.kind {
                SymbolKind::Function => Some(CompletionItemKind::FUNCTION),
                SymbolKind::Method => Some(CompletionItemKind::METHOD),
                SymbolKind::Variable => Some(CompletionItemKind::VARIABLE),
                SymbolKind::Type => Some(CompletionItemKind::CLASS),
                SymbolKind::Constant => Some(CompletionItemKind::CONSTANT),
                SymbolKind::Module => Some(CompletionItemKind::MODULE),
            };

            let detail = sym.signature.clone();
            let documentation = Some(lsp_types::Documentation::String(sym.description.clone()));

            CompletionItem {
                label: sym.name.clone(),
                kind,
                detail,
                documentation,
                ..Default::default()
            }
        })
        .collect()
}

/// Generate method completions for a VizorPlot value (after `.`).
pub fn vizor_plot_methods(index: &SymbolIndex) -> Vec<CompletionItem> {
    index
        .library_symbols("vizor")
        .into_iter()
        .filter(|s| s.kind == SymbolKind::Method)
        .map(|sym| CompletionItem {
            label: sym.name.clone(),
            kind: Some(CompletionItemKind::METHOD),
            detail: sym.signature.clone(),
            documentation: Some(lsp_types::Documentation::String(sym.description.clone())),
            ..Default::default()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_completion_prefix() {
        let mut idx = SymbolIndex::new();
        idx.populate_builtins();
        let items = complete(&idx, "pr");
        assert!(items.iter().any(|i| i.label == "print"));
    }

    #[test]
    fn test_vizor_method_completion() {
        let mut idx = SymbolIndex::new();
        idx.populate_vizor();
        let methods = vizor_plot_methods(&idx);
        assert!(methods.len() > 0);
    }
}
