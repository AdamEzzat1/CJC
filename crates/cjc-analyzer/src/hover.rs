//! Hover provider — returns documentation for symbols under the cursor.
//!
//! Currently provides:
//! - Builtin function signatures + descriptions
//! - Vizor library docs (when `import vizor` is active)
//! - User-defined function signatures

use crate::symbol_index::SymbolIndex;

/// Result of a hover query.
#[derive(Debug, Clone)]
pub struct HoverResult {
    pub contents: String,
    pub range: Option<(usize, usize)>,
}

/// Compute hover information for a given symbol name.
pub fn hover_for_symbol(index: &SymbolIndex, name: &str) -> Option<HoverResult> {
    let info = index.lookup(name)?;

    let mut markdown = String::new();

    // Signature block
    if let Some(sig) = &info.signature {
        markdown.push_str("```cjc\n");
        markdown.push_str(sig);
        markdown.push_str("\n```\n\n");
    }

    // Description
    markdown.push_str(&info.description);

    // Library badge
    if let Some(lib) = &info.library {
        markdown.push_str(&format!("\n\n*From library: `{}`*", lib));
    }

    Some(HoverResult {
        contents: markdown,
        range: None,
    })
}

/// Extract the word at a given byte offset in a source line.
pub fn word_at_offset(line: &str, offset: usize) -> Option<&str> {
    if offset >= line.len() {
        return None;
    }

    let bytes = line.as_bytes();
    if !is_ident_char(bytes[offset]) {
        return None;
    }

    let start = line[..offset]
        .rfind(|c: char| !c.is_alphanumeric() && c != '_')
        .map(|i| i + 1)
        .unwrap_or(0);

    let end = line[offset..]
        .find(|c: char| !c.is_alphanumeric() && c != '_')
        .map(|i| i + offset)
        .unwrap_or(line.len());

    Some(&line[start..end])
}

fn is_ident_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_word_at_offset() {
        assert_eq!(word_at_offset("let x = sqrt(42)", 8), Some("sqrt"));
        assert_eq!(word_at_offset("let x = sqrt(42)", 9), Some("sqrt"));
        assert_eq!(word_at_offset("let x = sqrt(42)", 12), None); // '(' char
        assert_eq!(word_at_offset("vizor_plot()", 0), Some("vizor_plot"));
    }

    #[test]
    fn test_hover_for_known_symbol() {
        let mut idx = crate::symbol_index::SymbolIndex::new();
        idx.populate_builtins();
        let result = hover_for_symbol(&idx, "sqrt").unwrap();
        assert!(result.contents.contains("sqrt"));
        assert!(result.contents.contains("Square root"));
    }

    #[test]
    fn test_hover_unknown_symbol() {
        let idx = crate::symbol_index::SymbolIndex::new();
        assert!(hover_for_symbol(&idx, "nonexistent").is_none());
    }
}
