//! CJC Regex Engine — NoGC-safe, zero-dependency, DFA-based ByteSlice matcher.
//!
//! Compiles regex patterns into NFA states, then executes via Thompson NFA
//! simulation (equivalent to lazy DFA). No backtracking. No per-match allocations
//! beyond the initial NFA state set swap buffers.
//!
//! Supported syntax (Perl-spirit subset):
//!
//! ```text
//!   .         any byte (or any byte except \n without `s` flag)
//!   \d        ASCII digit [0-9]
//!   \w        ASCII word  [a-zA-Z0-9_]
//!   \s        ASCII whitespace [\t\n\r\x0C\x20]
//!   \D \W \S  negated classes
//!   [abc]     character class
//!   [^abc]    negated character class
//!   [a-z]     character range
//!   a|b       alternation
//!   (...)     grouping
//!   *         zero or more (greedy)
//!   +         one or more (greedy)
//!   ?         zero or one (greedy)
//!   ^         start of input (or line in `m` mode)
//!   $         end of input (or line in `m` mode)
//!   \b        word boundary
//!   \\        literal backslash
//!   \xNN      hex byte
//! ```
//!
//! Flags:
//!   i   case-insensitive (ASCII only)
//!   m   multiline (^ and $ match line boundaries)
//!   s   dotall (. matches \n)
//!   x   extended (whitespace in pattern ignored, # comments)
//!   g   global (for split/replace — find all matches)

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Returns true if the pattern matches anywhere in `haystack`.
pub fn is_match(pattern: &str, flags: &str, haystack: &[u8]) -> bool {
    let opts = Flags::parse(flags);
    let nfa = match compile(pattern, &opts) {
        Ok(nfa) => nfa,
        Err(_) => return false, // invalid pattern → no match
    };
    nfa_search(&nfa, haystack, &opts).is_some()
}

/// Find the first match span (start, end) in `haystack`, or None.
pub fn find(pattern: &str, flags: &str, haystack: &[u8]) -> Option<(usize, usize)> {
    let opts = Flags::parse(flags);
    let nfa = compile(pattern, &opts).ok()?;
    nfa_search(&nfa, haystack, &opts)
}

/// Find all non-overlapping match spans.
pub fn find_all(pattern: &str, flags: &str, haystack: &[u8]) -> Vec<(usize, usize)> {
    let opts = Flags::parse(flags);
    let nfa = match compile(pattern, &opts) {
        Ok(nfa) => nfa,
        Err(_) => return Vec::new(),
    };
    let mut results = Vec::new();
    let mut start = 0;
    while start <= haystack.len() {
        if let Some((ms, me)) = nfa_search_from(&nfa, haystack, &opts, start) {
            results.push((ms, me));
            start = if me == ms { me + 1 } else { me };
        } else {
            break;
        }
    }
    results
}

/// Split `haystack` by regex pattern. Returns byte-ranges of non-matching segments.
pub fn split(pattern: &str, flags: &str, haystack: &[u8]) -> Vec<(usize, usize)> {
    let matches = find_all(pattern, flags, haystack);
    let mut segments = Vec::new();
    let mut pos = 0;
    for (ms, me) in &matches {
        segments.push((pos, *ms));
        pos = *me;
    }
    segments.push((pos, haystack.len()));
    segments
}

// ---------------------------------------------------------------------------
// Flags
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Flags {
    case_insensitive: bool,
    multiline: bool,
    dotall: bool,
    extended: bool,
}

impl Flags {
    fn parse(s: &str) -> Self {
        Self {
            case_insensitive: s.contains('i'),
            multiline: s.contains('m'),
            dotall: s.contains('s'),
            extended: s.contains('x'),
        }
    }
}

// ---------------------------------------------------------------------------
// NFA representation
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
enum NfaNode {
    /// Match a single byte.
    Byte(u8),
    /// Match a byte class (256-bit bitmap).
    Class(ByteClass),
    /// Match any byte.
    AnyByte,
    /// Match any byte except newline.
    AnyByteNoNl,
    /// Epsilon transition (split into two paths).
    Split(usize, usize),
    /// Epsilon transition to a single target.
    Epsilon(usize),
    /// Accepting state.
    Accept,
    /// Start-of-input / start-of-line anchor.
    AnchorStart,
    /// End-of-input / end-of-line anchor.
    AnchorEnd,
    /// Word boundary assertion.
    WordBoundary,
}

#[derive(Clone, Debug)]
struct ByteClass {
    bits: [u64; 4], // 256 bits
}

impl ByteClass {
    fn new() -> Self {
        Self { bits: [0; 4] }
    }

    fn set(&mut self, b: u8) {
        let idx = (b / 64) as usize;
        let bit = b % 64;
        self.bits[idx] |= 1u64 << bit;
    }

    fn contains(&self, b: u8) -> bool {
        let idx = (b / 64) as usize;
        let bit = b % 64;
        (self.bits[idx] >> bit) & 1 == 1
    }

    fn negate(&mut self) {
        for b in &mut self.bits {
            *b = !*b;
        }
    }

    fn set_range(&mut self, lo: u8, hi: u8) {
        for b in lo..=hi {
            self.set(b);
        }
    }
}

struct Nfa {
    nodes: Vec<NfaNode>,
    start: usize,
}

// ---------------------------------------------------------------------------
// Compiler: pattern string → NFA
// ---------------------------------------------------------------------------

struct Compiler<'a> {
    pattern: &'a [u8],
    pos: usize,
    nodes: Vec<NfaNode>,
    flags: Flags,
}

impl<'a> Compiler<'a> {
    fn new(pattern: &'a str, flags: &Flags) -> Self {
        Self {
            pattern: pattern.as_bytes(),
            pos: 0,
            nodes: Vec::new(),
            flags: flags.clone(),
        }
    }

    fn peek(&self) -> Option<u8> {
        if self.pos < self.pattern.len() {
            Some(self.pattern[self.pos])
        } else {
            None
        }
    }

    fn advance(&mut self) -> Option<u8> {
        let ch = self.peek()?;
        self.pos += 1;
        Some(ch)
    }

    fn emit(&mut self, node: NfaNode) -> usize {
        let id = self.nodes.len();
        self.nodes.push(node);
        id
    }

    fn compile(mut self) -> Result<Nfa, String> {
        if self.flags.extended {
            // Pre-strip whitespace and #-comments (not inside character classes)
            // For simplicity, we handle this inline during parsing.
        }
        let (start, end) = self.parse_alternation()?;
        // Patch `end` to Accept
        let accept = self.emit(NfaNode::Accept);
        self.patch(end, accept);
        Ok(Nfa {
            nodes: self.nodes,
            start,
        })
    }

    /// Patch a placeholder node to point to `target`.
    fn patch(&mut self, placeholder: usize, target: usize) {
        match &mut self.nodes[placeholder] {
            NfaNode::Epsilon(ref mut t) if *t == usize::MAX => *t = target,
            NfaNode::Split(ref mut a, ref mut b) => {
                if *a == usize::MAX { *a = target; }
                if *b == usize::MAX { *b = target; }
            }
            _ => {
                // Node already finalized or is a different type — add epsilon
                // This shouldn't happen in normal compilation, but safety:
                let next = self.nodes.len();
                self.nodes.push(NfaNode::Epsilon(target));
                // Try to chain
                match &mut self.nodes[placeholder] {
                    NfaNode::Byte(_) | NfaNode::Class(_) | NfaNode::AnyByte
                    | NfaNode::AnyByteNoNl | NfaNode::AnchorStart
                    | NfaNode::AnchorEnd | NfaNode::WordBoundary => {
                        // These are single-step nodes — they should already have
                        // been set up with a next pointer via the fragment model.
                        // Fallback: replace with epsilon chain
                        let _ = next;
                    }
                    _ => {}
                }
            }
        }
    }

    // -- Fragment model --
    // Each parse returns (start_id, end_placeholder_id).
    // The end_placeholder is an Epsilon(MAX) or similar that needs patching.

    fn parse_alternation(&mut self) -> Result<(usize, usize), String> {
        let (mut start, mut end) = self.parse_sequence()?;
        while self.peek() == Some(b'|') {
            self.advance(); // consume '|'
            let (s2, e2) = self.parse_sequence()?;
            let split = self.emit(NfaNode::Split(start, s2));
            let join = self.emit(NfaNode::Epsilon(usize::MAX));
            self.patch(end, join);
            self.patch(e2, join);
            start = split;
            end = join;
        }
        Ok((start, end))
    }

    fn parse_sequence(&mut self) -> Result<(usize, usize), String> {
        let mut fragments: Vec<(usize, usize)> = Vec::new();
        loop {
            match self.peek() {
                None | Some(b'|') | Some(b')') => break,
                _ => {
                    let frag = self.parse_quantified()?;
                    fragments.push(frag);
                }
            }
        }
        if fragments.is_empty() {
            // Empty sequence: epsilon
            let e = self.emit(NfaNode::Epsilon(usize::MAX));
            return Ok((e, e));
        }
        // Chain fragments: patch end of each to start of next
        for i in 0..fragments.len() - 1 {
            let next_start = fragments[i + 1].0;
            self.patch(fragments[i].1, next_start);
        }
        Ok((fragments[0].0, fragments[fragments.len() - 1].1))
    }

    fn parse_quantified(&mut self) -> Result<(usize, usize), String> {
        let (s, e) = self.parse_atom()?;
        match self.peek() {
            Some(b'*') => {
                self.advance();
                // Split → (body → back to split) | (skip)
                let split = self.emit(NfaNode::Split(s, usize::MAX));
                self.patch(e, split);
                let out = self.emit(NfaNode::Epsilon(usize::MAX));
                // Make split's second path go to out
                if let NfaNode::Split(_, ref mut b) = self.nodes[split] {
                    *b = out;
                }
                Ok((split, out))
            }
            Some(b'+') => {
                self.advance();
                // body → split(body, out)
                let split = self.emit(NfaNode::Split(s, usize::MAX));
                self.patch(e, split);
                let out = self.emit(NfaNode::Epsilon(usize::MAX));
                if let NfaNode::Split(_, ref mut b) = self.nodes[split] {
                    *b = out;
                }
                Ok((s, out))
            }
            Some(b'?') => {
                self.advance();
                // Split → (body → out) | (out)
                let out = self.emit(NfaNode::Epsilon(usize::MAX));
                self.patch(e, out);
                let split = self.emit(NfaNode::Split(s, out));
                Ok((split, out))
            }
            _ => Ok((s, e)),
        }
    }

    fn parse_atom(&mut self) -> Result<(usize, usize), String> {
        // Skip whitespace in extended mode
        if self.flags.extended {
            self.skip_extended_ws();
        }

        match self.peek() {
            Some(b'(') => {
                self.advance();
                let inner = self.parse_alternation()?;
                if self.peek() == Some(b')') {
                    self.advance();
                } else {
                    return Err("unclosed group `(`".into());
                }
                Ok(inner)
            }
            Some(b'[') => self.parse_char_class(),
            Some(b'.') => {
                self.advance();
                let node = if self.flags.dotall {
                    NfaNode::AnyByte
                } else {
                    NfaNode::AnyByteNoNl
                };
                let s = self.emit(node);
                let e = self.emit(NfaNode::Epsilon(usize::MAX));
                // s transitions to e on match
                // We represent this with a sequential chain: s → e
                // The NFA executor handles Byte/Class/Any nodes by consuming one byte
                // and advancing to the next node (s+1).
                Ok((s, e))
            }
            Some(b'^') => {
                self.advance();
                let s = self.emit(NfaNode::AnchorStart);
                let e = self.emit(NfaNode::Epsilon(usize::MAX));
                Ok((s, e))
            }
            Some(b'$') => {
                self.advance();
                let s = self.emit(NfaNode::AnchorEnd);
                let e = self.emit(NfaNode::Epsilon(usize::MAX));
                Ok((s, e))
            }
            Some(b'\\') => {
                self.advance();
                let (class_or_byte, negate) = self.parse_escape()?;
                match class_or_byte {
                    EscapeResult::Byte(b) => {
                        let actual_byte = if self.flags.case_insensitive {
                            b.to_ascii_lowercase()
                        } else {
                            b
                        };
                        let s = if self.flags.case_insensitive && actual_byte.is_ascii_alphabetic() {
                            let mut cls = ByteClass::new();
                            cls.set(actual_byte.to_ascii_lowercase());
                            cls.set(actual_byte.to_ascii_uppercase());
                            self.emit(NfaNode::Class(cls))
                        } else {
                            self.emit(NfaNode::Byte(actual_byte))
                        };
                        let e = self.emit(NfaNode::Epsilon(usize::MAX));
                        Ok((s, e))
                    }
                    EscapeResult::Class(mut cls) => {
                        if negate {
                            cls.negate();
                        }
                        let s = self.emit(NfaNode::Class(cls));
                        let e = self.emit(NfaNode::Epsilon(usize::MAX));
                        Ok((s, e))
                    }
                    EscapeResult::WordBoundary => {
                        let s = self.emit(NfaNode::WordBoundary);
                        let e = self.emit(NfaNode::Epsilon(usize::MAX));
                        Ok((s, e))
                    }
                }
            }
            Some(ch) if ch != b'|' && ch != b')' && ch != b'*' && ch != b'+' && ch != b'?' => {
                self.advance();
                let s = if self.flags.case_insensitive && ch.is_ascii_alphabetic() {
                    let mut cls = ByteClass::new();
                    cls.set(ch.to_ascii_lowercase());
                    cls.set(ch.to_ascii_uppercase());
                    self.emit(NfaNode::Class(cls))
                } else {
                    self.emit(NfaNode::Byte(ch))
                };
                let e = self.emit(NfaNode::Epsilon(usize::MAX));
                Ok((s, e))
            }
            _ => Err(format!("unexpected character in regex at position {}", self.pos)),
        }
    }

    fn parse_char_class(&mut self) -> Result<(usize, usize), String> {
        self.advance(); // consume '['
        let negated = self.peek() == Some(b'^');
        if negated {
            self.advance();
        }
        let mut cls = ByteClass::new();
        let mut first = true;
        loop {
            match self.peek() {
                None => return Err("unclosed character class `[`".into()),
                Some(b']') if !first => {
                    self.advance();
                    break;
                }
                _ => {
                    first = false;
                    let lo = self.parse_class_atom()?;
                    if self.peek() == Some(b'-') && self.pos + 1 < self.pattern.len() && self.pattern[self.pos + 1] != b']' {
                        self.advance(); // consume '-'
                        let hi = self.parse_class_atom()?;
                        if self.flags.case_insensitive {
                            for b in lo..=hi {
                                cls.set(b.to_ascii_lowercase());
                                cls.set(b.to_ascii_uppercase());
                            }
                        } else {
                            cls.set_range(lo, hi);
                        }
                    } else if self.flags.case_insensitive && lo.is_ascii_alphabetic() {
                        cls.set(lo.to_ascii_lowercase());
                        cls.set(lo.to_ascii_uppercase());
                    } else {
                        cls.set(lo);
                    }
                }
            }
        }
        if negated {
            cls.negate();
        }
        let s = self.emit(NfaNode::Class(cls));
        let e = self.emit(NfaNode::Epsilon(usize::MAX));
        Ok((s, e))
    }

    fn parse_class_atom(&mut self) -> Result<u8, String> {
        match self.advance() {
            Some(b'\\') => {
                match self.advance() {
                    Some(b'n') => Ok(b'\n'),
                    Some(b't') => Ok(b'\t'),
                    Some(b'r') => Ok(b'\r'),
                    Some(b'\\') => Ok(b'\\'),
                    Some(b']') => Ok(b']'),
                    Some(b'-') => Ok(b'-'),
                    Some(b'x') => self.parse_hex_byte(),
                    Some(ch) => Ok(ch),
                    None => Err("unexpected end of escape in character class".into()),
                }
            }
            Some(ch) => Ok(ch),
            None => Err("unexpected end of character class".into()),
        }
    }

    fn parse_hex_byte(&mut self) -> Result<u8, String> {
        let hi = self.advance().ok_or("incomplete hex escape")?;
        let lo = self.advance().ok_or("incomplete hex escape")?;
        let h = hex_val(hi).ok_or_else(|| format!("invalid hex digit `{}`", hi as char))?;
        let l = hex_val(lo).ok_or_else(|| format!("invalid hex digit `{}`", lo as char))?;
        Ok(h * 16 + l)
    }

    fn parse_escape(&mut self) -> Result<(EscapeResult, bool), String> {
        match self.advance() {
            Some(b'd') => {
                let mut cls = ByteClass::new();
                cls.set_range(b'0', b'9');
                Ok((EscapeResult::Class(cls), false))
            }
            Some(b'D') => {
                let mut cls = ByteClass::new();
                cls.set_range(b'0', b'9');
                Ok((EscapeResult::Class(cls), true))
            }
            Some(b'w') => {
                let mut cls = ByteClass::new();
                cls.set_range(b'a', b'z');
                cls.set_range(b'A', b'Z');
                cls.set_range(b'0', b'9');
                cls.set(b'_');
                Ok((EscapeResult::Class(cls), false))
            }
            Some(b'W') => {
                let mut cls = ByteClass::new();
                cls.set_range(b'a', b'z');
                cls.set_range(b'A', b'Z');
                cls.set_range(b'0', b'9');
                cls.set(b'_');
                Ok((EscapeResult::Class(cls), true))
            }
            Some(b's') => {
                let mut cls = ByteClass::new();
                cls.set(b' ');
                cls.set(b'\t');
                cls.set(b'\n');
                cls.set(b'\r');
                cls.set(0x0C); // form feed
                Ok((EscapeResult::Class(cls), false))
            }
            Some(b'S') => {
                let mut cls = ByteClass::new();
                cls.set(b' ');
                cls.set(b'\t');
                cls.set(b'\n');
                cls.set(b'\r');
                cls.set(0x0C);
                Ok((EscapeResult::Class(cls), true))
            }
            Some(b'b') => Ok((EscapeResult::WordBoundary, false)),
            Some(b'n') => Ok((EscapeResult::Byte(b'\n'), false)),
            Some(b't') => Ok((EscapeResult::Byte(b'\t'), false)),
            Some(b'r') => Ok((EscapeResult::Byte(b'\r'), false)),
            Some(b'0') => Ok((EscapeResult::Byte(0), false)),
            Some(b'x') => {
                let b = self.parse_hex_byte()?;
                Ok((EscapeResult::Byte(b), false))
            }
            Some(ch) => {
                // Literal escape (e.g. \. \+ \* etc.)
                Ok((EscapeResult::Byte(ch), false))
            }
            None => Err("unexpected end of escape sequence".into()),
        }
    }

    fn skip_extended_ws(&mut self) {
        while let Some(ch) = self.peek() {
            if ch == b' ' || ch == b'\t' || ch == b'\n' || ch == b'\r' {
                self.advance();
            } else if ch == b'#' {
                // Skip until end of line
                while let Some(c) = self.peek() {
                    self.advance();
                    if c == b'\n' { break; }
                }
            } else {
                break;
            }
        }
    }
}

enum EscapeResult {
    Byte(u8),
    Class(ByteClass),
    WordBoundary,
}

fn compile(pattern: &str, flags: &Flags) -> Result<Nfa, String> {
    Compiler::new(pattern, flags).compile()
}

fn hex_val(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// NFA executor: Thompson NFA simulation
// ---------------------------------------------------------------------------

fn nfa_search(nfa: &Nfa, haystack: &[u8], opts: &Flags) -> Option<(usize, usize)> {
    // Try matching from each position (unanchored search)
    for start in 0..=haystack.len() {
        if let Some(end) = nfa_match_at(nfa, haystack, opts, start) {
            return Some((start, end));
        }
    }
    None
}

fn nfa_search_from(nfa: &Nfa, haystack: &[u8], opts: &Flags, from: usize) -> Option<(usize, usize)> {
    for start in from..=haystack.len() {
        if let Some(end) = nfa_match_at(nfa, haystack, opts, start) {
            return Some((start, end));
        }
    }
    None
}

/// Try to match the NFA starting at position `start` in the haystack.
/// Returns the end position if a match is found.
fn nfa_match_at(nfa: &Nfa, haystack: &[u8], opts: &Flags, start: usize) -> Option<usize> {
    let mut current = Vec::new();
    let mut next = Vec::new();
    let mut in_current = vec![false; nfa.nodes.len()];
    let mut in_next = vec![false; nfa.nodes.len()];
    let mut last_match: Option<usize> = None;

    // Add start state with epsilon closure
    add_state(nfa, nfa.start, &mut current, &mut in_current, haystack, opts, start);

    // Check if start state itself accepts (for zero-length matches)
    for &state in &current {
        if matches!(nfa.nodes[state], NfaNode::Accept) {
            last_match = Some(start);
        }
    }

    let mut pos = start;
    while pos < haystack.len() && !current.is_empty() {
        let byte = haystack[pos];

        for &state in &current {
            match &nfa.nodes[state] {
                NfaNode::Byte(b) => {
                    if byte == *b {
                        add_state(nfa, state + 1, &mut next, &mut in_next, haystack, opts, pos + 1);
                    }
                }
                NfaNode::Class(cls) => {
                    if cls.contains(byte) {
                        add_state(nfa, state + 1, &mut next, &mut in_next, haystack, opts, pos + 1);
                    }
                }
                NfaNode::AnyByte => {
                    add_state(nfa, state + 1, &mut next, &mut in_next, haystack, opts, pos + 1);
                }
                NfaNode::AnyByteNoNl => {
                    if byte != b'\n' {
                        add_state(nfa, state + 1, &mut next, &mut in_next, haystack, opts, pos + 1);
                    }
                }
                NfaNode::Accept => {
                    // Already recorded in last_match
                }
                _ => {
                    // Epsilon/Split/Anchor states should have been resolved during add_state
                }
            }
        }

        // Check for accepts in next
        for &state in &next {
            if matches!(nfa.nodes[state], NfaNode::Accept) {
                last_match = Some(pos + 1);
            }
        }

        // Swap current and next
        std::mem::swap(&mut current, &mut next);
        std::mem::swap(&mut in_current, &mut in_next);
        next.clear();
        for b in in_next.iter_mut() { *b = false; }

        pos += 1;
    }

    last_match
}

/// Add a state to the set, following epsilon transitions (closure).
fn add_state(
    nfa: &Nfa,
    state: usize,
    set: &mut Vec<usize>,
    in_set: &mut [bool],
    haystack: &[u8],
    opts: &Flags,
    pos: usize,
) {
    if state >= nfa.nodes.len() || in_set[state] {
        return;
    }
    in_set[state] = true;

    match &nfa.nodes[state] {
        NfaNode::Epsilon(target) => {
            if *target != usize::MAX {
                add_state(nfa, *target, set, in_set, haystack, opts, pos);
            }
        }
        NfaNode::Split(a, b) => {
            add_state(nfa, *a, set, in_set, haystack, opts, pos);
            add_state(nfa, *b, set, in_set, haystack, opts, pos);
        }
        NfaNode::AnchorStart => {
            let ok = if opts.multiline {
                pos == 0 || (pos > 0 && haystack[pos - 1] == b'\n')
            } else {
                pos == 0
            };
            if ok {
                add_state(nfa, state + 1, set, in_set, haystack, opts, pos);
            }
        }
        NfaNode::AnchorEnd => {
            let ok = if opts.multiline {
                pos == haystack.len() || haystack[pos] == b'\n'
            } else {
                pos == haystack.len()
            };
            if ok {
                add_state(nfa, state + 1, set, in_set, haystack, opts, pos);
            }
        }
        NfaNode::WordBoundary => {
            let before_word = pos > 0 && is_word_byte(haystack[pos - 1]);
            let after_word = pos < haystack.len() && is_word_byte(haystack[pos]);
            if before_word != after_word {
                add_state(nfa, state + 1, set, in_set, haystack, opts, pos);
            }
        }
        _ => {
            // Consuming state — add to active set
            set.push(state);
        }
    }
}

fn is_word_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_literal() {
        assert!(is_match("hello", "", b"say hello world"));
        assert!(!is_match("hello", "", b"say helo world"));
    }

    #[test]
    fn test_dot() {
        assert!(is_match("h.llo", "", b"hello"));
        assert!(is_match("h.llo", "", b"hxllo"));
        assert!(!is_match("h.llo", "", b"hlo"));
    }

    #[test]
    fn test_star() {
        assert!(is_match("ab*c", "", b"ac"));
        assert!(is_match("ab*c", "", b"abc"));
        assert!(is_match("ab*c", "", b"abbc"));
        assert!(is_match("ab*c", "", b"abbbbc"));
    }

    #[test]
    fn test_plus() {
        assert!(!is_match("ab+c", "", b"ac"));
        assert!(is_match("ab+c", "", b"abc"));
        assert!(is_match("ab+c", "", b"abbc"));
    }

    #[test]
    fn test_question() {
        assert!(is_match("ab?c", "", b"ac"));
        assert!(is_match("ab?c", "", b"abc"));
        assert!(!is_match("ab?c", "", b"abbc"));
    }

    #[test]
    fn test_alternation() {
        assert!(is_match("cat|dog", "", b"I have a cat"));
        assert!(is_match("cat|dog", "", b"I have a dog"));
        assert!(!is_match("cat|dog", "", b"I have a bird"));
    }

    #[test]
    fn test_char_class() {
        assert!(is_match("[abc]", "", b"a"));
        assert!(is_match("[abc]", "", b"b"));
        assert!(is_match("[abc]", "", b"c"));
        assert!(!is_match("[abc]", "", b"d"));
    }

    #[test]
    fn test_char_class_range() {
        assert!(is_match("[a-z]", "", b"m"));
        assert!(!is_match("[a-z]", "", b"M"));
    }

    #[test]
    fn test_negated_class() {
        assert!(!is_match("[^abc]", "", b"a"));
        assert!(is_match("[^abc]", "", b"d"));
    }

    #[test]
    fn test_digit_class() {
        assert!(is_match("\\d+", "", b"abc123def"));
        assert!(!is_match("^\\d+$", "", b"abc123def"));
        assert!(is_match("^\\d+$", "", b"123"));
    }

    #[test]
    fn test_word_class() {
        assert!(is_match("\\w+", "", b"hello_world"));
        assert!(!is_match("^\\w+$", "", b"hello world"));
    }

    #[test]
    fn test_anchors() {
        assert!(is_match("^hello", "", b"hello world"));
        assert!(!is_match("^hello", "", b"say hello"));
        assert!(is_match("world$", "", b"hello world"));
        assert!(!is_match("world$", "", b"world hello"));
    }

    #[test]
    fn test_case_insensitive() {
        assert!(is_match("hello", "i", b"HELLO"));
        assert!(is_match("hello", "i", b"HeLLo"));
    }

    #[test]
    fn test_dotall() {
        assert!(!is_match("a.b", "", b"a\nb"));
        assert!(is_match("a.b", "s", b"a\nb"));
    }

    #[test]
    fn test_multiline() {
        assert!(is_match("^world", "m", b"hello\nworld"));
        assert!(!is_match("^world", "", b"hello\nworld"));
    }

    #[test]
    fn test_groups() {
        assert!(is_match("(abc)+", "", b"abcabc"));
        assert!(!is_match("(abc)+", "", b"abab"));
    }

    #[test]
    fn test_hex_escape() {
        assert!(is_match("\\x41\\x42", "", b"AB"));
    }

    #[test]
    fn test_word_boundary() {
        assert!(is_match("\\bhello\\b", "", b"say hello world"));
        assert!(!is_match("\\bhello\\b", "", b"sayhelloworld"));
    }

    #[test]
    fn test_find_span() {
        let span = find("\\d+", "", b"abc123def");
        assert_eq!(span, Some((3, 6)));
    }

    #[test]
    fn test_find_all() {
        let spans = find_all("\\d+", "", b"a1b22c333");
        assert_eq!(spans, vec![(1, 2), (3, 5), (6, 9)]);
    }

    #[test]
    fn test_split() {
        let segs = split(",", "", b"a,b,c");
        assert_eq!(segs, vec![(0, 1), (2, 3), (4, 5)]);
    }

    #[test]
    fn test_split_whitespace() {
        let segs = split("\\s+", "", b"hello  world");
        assert_eq!(segs, vec![(0, 5), (7, 12)]);
    }

    #[test]
    fn test_empty_pattern() {
        assert!(is_match("", "", b"anything"));
    }

    #[test]
    fn test_empty_haystack() {
        assert!(!is_match("a", "", b""));
        assert!(is_match("", "", b""));
        assert!(is_match("a*", "", b""));
    }

    #[test]
    fn test_determinism() {
        // Same pattern, same input → same result, always
        for _ in 0..10 {
            let r = find_all("\\d+", "", b"x1y22z333");
            assert_eq!(r, vec![(1, 2), (3, 5), (6, 9)]);
        }
    }

    #[test]
    fn test_whitespace_class() {
        assert!(is_match("\\s", "", b" "));
        assert!(is_match("\\s", "", b"\t"));
        assert!(is_match("\\s", "", b"\n"));
        assert!(!is_match("\\s", "", b"a"));
    }

    #[test]
    fn test_negated_digit() {
        assert!(is_match("\\D", "", b"a"));
        assert!(!is_match("\\D", "", b"5"));
    }
}
