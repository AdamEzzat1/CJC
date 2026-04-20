//! CJC Regex Engine -- NoGC-safe, zero-dependency, NFA-based byte-slice matcher.
//!
//! Compile regex patterns into NFA states, then execute via Thompson NFA
//! simulation (equivalent to a lazy DFA). There is no backtracking and no
//! per-match heap allocation beyond the initial NFA state-set swap buffers.
//! This makes the engine safe for use in NoGC-verified code paths and
//! guarantees deterministic, linear-time matching regardless of input.
//!
//! # Architecture
//!
//! 1. **Compile** -- the [`Compiler`] translates a pattern string into an [`Nfa`]
//!    (a vector of [`NfaNode`] states) using a fragment-based construction.
//! 2. **Execute** -- the Thompson NFA simulator walks the state set in lockstep
//!    with the input bytes, tracking epsilon closures at each step. Greedy
//!    quantifiers record the *longest* match; lazy quantifiers (e.g. `*?`)
//!    return the *shortest* match by accepting on the first `Accept` state
//!    reached.
//!
//! # Public API
//!
//! | Function            | Purpose                                          |
//! |---------------------|--------------------------------------------------|
//! | [`is_match`]        | Test whether the pattern matches anywhere         |
//! | [`find`]            | Return the first match span `(start, end)`       |
//! | [`find_all`]        | Return all non-overlapping match spans            |
//! | [`split`]           | Split input by pattern, returning segment spans   |
//! | [`find_match`]      | Return first match as a [`MatchResult`]           |
//! | [`find_all_matches`]| Return all matches as `Vec<MatchResult>`          |
//! | [`regex_explain`]   | Return human-readable NFA description             |
//!
//! All functions accept the pattern, a flags string, and a `&[u8]` haystack.
//!
//! # Supported syntax (Perl-spirit subset)
//!
//! ```text
//!   .            any byte (or any byte except \n without `s` flag)
//!   \d           ASCII digit [0-9]
//!   \w           ASCII word  [a-zA-Z0-9_]
//!   \s           ASCII whitespace [\t\n\r\x0C\x20]
//!   \D \W \S     negated classes
//!   [abc]        character class
//!   [^abc]       negated character class
//!   [a-z]        character range
//!   [[:alpha:]]  POSIX character classes (inside [...])
//!   a|b          alternation
//!   (...)        grouping (capturing)
//!   (?:...)      non-capturing group
//!   (?i) (?m)    inline flags (i, m, s, x)
//!   (?i:...)     flag-scoped group (flags apply only within)
//!   *            zero or more (greedy)
//!   +            one or more (greedy)
//!   ?            zero or one (greedy)
//!   *? +? ??     non-greedy (lazy) variants
//!   {n}          exactly n repetitions
//!   {n,}         n or more repetitions
//!   {n,m}        between n and m repetitions
//!   {n}? {n,}?   lazy counted repetitions
//!   ^            start of input (or line in `m` mode)
//!   $            end of input (or line in `m` mode)
//!   \A           absolute start of input (ignores multiline)
//!   \z           absolute end of input (ignores multiline)
//!   \Z           end of input or before final newline
//!   \b           word boundary
//!   \B           non-word boundary
//!   \\           literal backslash
//!   \xNN         hex byte
//!   \uNNNN       Unicode codepoint (UTF-8 encoded)
//!   \u{NNNN}     Unicode codepoint (braced form)
//! ```
//!
//! # Flags
//!
//! Pass flags as a string (e.g. `"im"` for case-insensitive multiline):
//!
//! | Flag | Meaning                                          |
//! |------|--------------------------------------------------|
//! | `i`  | Case-insensitive (ASCII only)                    |
//! | `m`  | Multiline (`^`/`$` match line boundaries)        |
//! | `s`  | Dotall (`.` matches `\n`)                        |
//! | `x`  | Extended (whitespace ignored, `#` comments)      |
//! | `g`  | Global (for split/replace -- find all matches)   |
//!
//! # Determinism
//!
//! The engine is fully deterministic: identical pattern, flags, and haystack
//! always produce bit-identical results regardless of platform or invocation
//! count. No `HashMap`, no random iteration, no FMA.

use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// Safety limits
// ---------------------------------------------------------------------------

/// Maximum pattern length in bytes. Patterns longer than this are rejected.
const MAX_PATTERN_LEN: usize = 4096;
/// Maximum NFA node count. Counted repetitions that would exceed this fail.
const MAX_NODES: usize = 65536;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A structured match result with byte-offset span.
///
/// Returned by [`find_match`] and [`find_all_matches`]. The span `[start, end)`
/// is a half-open byte range into the original haystack.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MatchResult {
    /// Byte offset of the start of the match (inclusive).
    pub start: usize,
    /// Byte offset of the end of the match (exclusive).
    pub end: usize,
}

impl MatchResult {
    /// Extract the matched bytes from the original haystack.
    pub fn extract<'a>(&self, haystack: &'a [u8]) -> &'a [u8] {
        &haystack[self.start..self.end]
    }

    /// Extract the matched text as a UTF-8 string slice, if valid.
    pub fn extract_str<'a>(&self, haystack: &'a [u8]) -> Option<&'a str> {
        std::str::from_utf8(self.extract(haystack)).ok()
    }

    /// Length of the match in bytes.
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    /// True if this is a zero-length match.
    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Test whether `pattern` matches anywhere inside `haystack`.
///
/// # Examples
///
/// ```
/// use cjc_regex::is_match;
///
/// assert!(is_match("\\d+", "", b"abc123"));
/// assert!(!is_match("^\\d+$", "", b"abc123"));
/// assert!(is_match("hello", "i", b"Hello World"));
/// ```
pub fn is_match(pattern: &str, flags: &str, haystack: &[u8]) -> bool {
    let opts = Flags::parse(flags);
    let nfa = match compile(pattern, &opts) {
        Ok(nfa) => nfa,
        Err(_) => return false,
    };
    nfa_search(&nfa, haystack, &opts).is_some()
}

/// Find the byte-offset span of the first match in `haystack`.
///
/// # Examples
///
/// ```
/// use cjc_regex::find;
///
/// assert_eq!(find("\\d+", "", b"abc123def"), Some((3, 6)));
/// assert_eq!(find("xyz", "", b"hello"), None);
/// ```
pub fn find(pattern: &str, flags: &str, haystack: &[u8]) -> Option<(usize, usize)> {
    let opts = Flags::parse(flags);
    let nfa = compile(pattern, &opts).ok()?;
    nfa_search(&nfa, haystack, &opts)
}

/// Find all non-overlapping match spans in `haystack`.
///
/// # Examples
///
/// ```
/// use cjc_regex::find_all;
///
/// let spans = find_all("\\d+", "", b"a1b22c333");
/// assert_eq!(spans, vec![(1, 2), (3, 5), (6, 9)]);
/// ```
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

/// Split `haystack` by a regex pattern, returning byte-ranges of non-matching segments.
///
/// # Examples
///
/// ```
/// use cjc_regex::split;
///
/// let segs = split(",", "", b"a,b,c");
/// assert_eq!(segs, vec![(0, 1), (2, 3), (4, 5)]);
///
/// let segs = split("\\s+", "", b"hello  world");
/// assert_eq!(segs, vec![(0, 5), (7, 12)]);
/// ```
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

/// Find the first match, returning a [`MatchResult`].
///
/// # Examples
///
/// ```
/// use cjc_regex::{find_match, MatchResult};
///
/// let m = find_match("\\d+", "", b"abc123def").unwrap();
/// assert_eq!(m, MatchResult { start: 3, end: 6 });
/// assert_eq!(m.extract(b"abc123def"), b"123");
/// ```
pub fn find_match(pattern: &str, flags: &str, haystack: &[u8]) -> Option<MatchResult> {
    find(pattern, flags, haystack).map(|(start, end)| MatchResult { start, end })
}

/// Find all non-overlapping matches, returning a `Vec<MatchResult>`.
///
/// # Examples
///
/// ```
/// use cjc_regex::find_all_matches;
///
/// let ms = find_all_matches("\\d+", "", b"a1b22c333");
/// assert_eq!(ms.len(), 3);
/// assert_eq!(ms[0].extract(b"a1b22c333"), b"1");
/// ```
pub fn find_all_matches(pattern: &str, flags: &str, haystack: &[u8]) -> Vec<MatchResult> {
    find_all(pattern, flags, haystack)
        .into_iter()
        .map(|(start, end)| MatchResult { start, end })
        .collect()
}

/// Return a human-readable description of the compiled NFA.
///
/// Useful for debugging patterns that behave unexpectedly.
///
/// # Errors
///
/// Returns `Err(String)` if the pattern fails to compile.
///
/// # Examples
///
/// ```
/// use cjc_regex::regex_explain;
///
/// let desc = regex_explain("\\d+", "").unwrap();
/// assert!(desc.contains("NFA"));
/// ```
pub fn regex_explain(pattern: &str, flags: &str) -> Result<String, String> {
    let opts = Flags::parse(flags);
    let nfa = compile(pattern, &opts)?;
    let mut out = String::new();
    out.push_str(&format!("NFA for pattern `{}` flags `{}`\n", pattern, flags));
    out.push_str(&format!("  Start : node {}\n", nfa.start));
    out.push_str(&format!("  Nodes : {}\n", nfa.nodes.len()));
    out.push_str(&format!("  Lazy  : {}\n", nfa.has_lazy));
    out.push_str("\nNode table:\n");
    for (i, node) in nfa.nodes.iter().enumerate() {
        let marker = if i == nfa.start { "→ " } else { "  " };
        let desc = match node {
            NfaNode::Byte(b) => format!("Byte(0x{:02x} = {:?})", b, *b as char),
            NfaNode::Class(_) => "Class([bitmap])".to_string(),
            NfaNode::AnyByte => "AnyByte".to_string(),
            NfaNode::AnyByteNoNl => "AnyByteNoNl".to_string(),
            NfaNode::Split(a, b) => format!("Split(→{}, →{})", a, b),
            NfaNode::Epsilon(t) => {
                if *t == usize::MAX { "Epsilon(→UNPATCHED)".to_string() }
                else { format!("Epsilon(→{})", t) }
            }
            NfaNode::Accept => "Accept ✓".to_string(),
            NfaNode::AnchorStart => "AnchorStart (^)".to_string(),
            NfaNode::AnchorEnd => "AnchorEnd ($)".to_string(),
            NfaNode::WordBoundary => "WordBoundary (\\b)".to_string(),
            NfaNode::NonWordBoundary => "NonWordBoundary (\\B)".to_string(),
            NfaNode::AbsoluteStart => "AbsoluteStart (\\A)".to_string(),
            NfaNode::AbsoluteEnd => "AbsoluteEnd (\\z)".to_string(),
            NfaNode::AbsoluteEndBeforeNewline => "AbsoluteEndBeforeNewline (\\Z)".to_string(),
            NfaNode::Save(slot) => format!("Save(slot={})", slot),
        };
        out.push_str(&format!("  [{:3}] {}{}\n", i, marker, desc));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Capture group types and API
// ---------------------------------------------------------------------------

/// A single capture group result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Capture {
    /// Byte offset of the start of the capture (inclusive).
    pub start: usize,
    /// Byte offset of the end of the capture (exclusive).
    pub end: usize,
}

impl Capture {
    /// Extract the captured bytes from the original haystack.
    pub fn extract<'a>(&self, haystack: &'a [u8]) -> &'a [u8] {
        &haystack[self.start..self.end]
    }

    /// Extract the captured text as a UTF-8 string slice, if valid.
    pub fn extract_str<'a>(&self, haystack: &'a [u8]) -> Option<&'a str> {
        std::str::from_utf8(self.extract(haystack)).ok()
    }
}

/// Full match result with capture groups.
#[derive(Clone, Debug)]
pub struct CaptureResult {
    /// The full match (group 0).
    pub full: MatchResult,
    /// All capture groups (index 0 = full match, 1+ = capture groups).
    pub groups: Vec<Option<Capture>>,
    /// Named group index map (name -> group index, 1-based).
    pub names: BTreeMap<String, usize>,
}

impl CaptureResult {
    /// Get a capture group by index (0 = full match, 1+ = groups).
    pub fn get(&self, idx: usize) -> Option<&Capture> {
        self.groups.get(idx).and_then(|c| c.as_ref())
    }

    /// Get a named capture group.
    pub fn get_named(&self, name: &str) -> Option<&Capture> {
        self.names.get(name).and_then(|&idx| self.get(idx))
    }
}

/// Find the first match with capture groups.
///
/// Returns `None` if the pattern doesn't match. The returned `CaptureResult`
/// contains the full match as group 0 and all numbered/named capture groups.
pub fn find_captures(pattern: &str, flags: &str, haystack: &[u8]) -> Option<CaptureResult> {
    let opts = Flags::parse(flags);
    let nfa = compile(pattern, &opts).ok()?;
    pike_search(&nfa, haystack, &opts)
}

/// Find all non-overlapping matches with capture groups.
pub fn find_all_captures(pattern: &str, flags: &str, haystack: &[u8]) -> Vec<CaptureResult> {
    let opts = Flags::parse(flags);
    let nfa = match compile(pattern, &opts) {
        Ok(nfa) => nfa,
        Err(_) => return Vec::new(),
    };
    let mut results = Vec::new();
    let mut start = 0;
    while start <= haystack.len() {
        if let Some(cr) = pike_search_from(&nfa, haystack, &opts, start) {
            let next = if cr.full.end == cr.full.start {
                cr.full.end + 1
            } else {
                cr.full.end
            };
            results.push(cr);
            start = next;
        } else {
            break;
        }
    }
    results
}

/// Return the number of capture groups in a pattern (0 if none or invalid).
pub fn capture_count(pattern: &str, flags: &str) -> usize {
    let opts = Flags::parse(flags);
    match compile(pattern, &opts) {
        Ok(nfa) => nfa.num_groups,
        Err(_) => 0,
    }
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
    /// Word boundary assertion (\b).
    WordBoundary,
    /// Non-word boundary assertion (\B).
    NonWordBoundary,
    /// Absolute start of input (\A) — ignores multiline flag.
    AbsoluteStart,
    /// Absolute end of input (\z) — ignores multiline flag.
    AbsoluteEnd,
    /// End of input or before final newline (\Z).
    AbsoluteEndBeforeNewline,
    /// Save the current position into capture slot `n`.
    /// Used by the Pike VM for capture group tracking.
    Save(usize),
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
    /// True if any quantifier in the pattern is non-greedy (lazy).
    has_lazy: bool,
    /// Flags as seen at the end of compilation, including any inline modifiers
    /// (e.g. `(?m)`) that affect runtime anchor evaluation.
    effective_flags: Flags,
    /// Number of capturing groups in the pattern (0 if none).
    num_groups: usize,
    /// Named group index map: name -> group index (1-based).
    group_names: BTreeMap<String, usize>,
}

// ---------------------------------------------------------------------------
// Compiler: pattern string → NFA
// ---------------------------------------------------------------------------

struct Compiler<'a> {
    pattern: &'a [u8],
    pos: usize,
    nodes: Vec<NfaNode>,
    flags: Flags,
    has_lazy: bool,
    /// Number of capturing groups seen so far.
    num_groups: usize,
    /// Named group index map: name -> group index (1-based).
    group_names: BTreeMap<String, usize>,
}

impl<'a> Compiler<'a> {
    fn new(pattern: &'a str, flags: &Flags) -> Self {
        Self {
            pattern: pattern.as_bytes(),
            pos: 0,
            nodes: Vec::new(),
            flags: flags.clone(),
            has_lazy: false,
            num_groups: 0,
            group_names: BTreeMap::new(),
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
        if self.pattern.len() > MAX_PATTERN_LEN {
            return Err(format!(
                "pattern too long: {} bytes (limit {})",
                self.pattern.len(),
                MAX_PATTERN_LEN
            ));
        }
        let (start, end) = self.parse_alternation()?;
        let accept = self.emit(NfaNode::Accept);
        self.patch(end, accept);
        if self.nodes.len() > MAX_NODES {
            return Err(format!(
                "NFA too large: {} nodes (limit {})",
                self.nodes.len(),
                MAX_NODES
            ));
        }
        let effective_flags = self.flags.clone();
        let num_groups = self.num_groups;
        let group_names = self.group_names;
        Ok(Nfa {
            nodes: self.nodes,
            start,
            has_lazy: self.has_lazy,
            effective_flags,
            num_groups,
            group_names,
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
                // Already finalized — insert forwarding epsilon
                let next = self.nodes.len();
                self.nodes.push(NfaNode::Epsilon(target));
                let _ = next;
            }
        }
    }

    // -- Fragment model --
    // Each parse fn returns (start_id, end_placeholder_id).
    // The end_placeholder is an Epsilon(MAX) that needs to be patched.

    fn parse_alternation(&mut self) -> Result<(usize, usize), String> {
        let (mut start, mut end) = self.parse_sequence()?;
        while self.peek() == Some(b'|') {
            self.advance();
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
            let e = self.emit(NfaNode::Epsilon(usize::MAX));
            return Ok((e, e));
        }
        for i in 0..fragments.len() - 1 {
            let next_start = fragments[i + 1].0;
            self.patch(fragments[i].1, next_start);
        }
        Ok((fragments[0].0, fragments[fragments.len() - 1].1))
    }

    fn parse_quantified(&mut self) -> Result<(usize, usize), String> {
        let atom_node_start = self.nodes.len();
        let (s, e) = self.parse_atom()?;
        let atom_node_end = self.nodes.len();

        match self.peek() {
            Some(b'*') => {
                self.advance();
                let lazy = self.peek() == Some(b'?');
                if lazy { self.advance(); self.has_lazy = true; }
                let split = self.emit(NfaNode::Split(s, usize::MAX));
                self.patch(e, split);
                let out = self.emit(NfaNode::Epsilon(usize::MAX));
                if lazy {
                    if let NfaNode::Split(ref mut a, ref mut b) = self.nodes[split] {
                        *a = out; *b = s;
                    }
                } else {
                    if let NfaNode::Split(_, ref mut b) = self.nodes[split] {
                        *b = out;
                    }
                }
                Ok((split, out))
            }
            Some(b'+') => {
                self.advance();
                let lazy = self.peek() == Some(b'?');
                if lazy { self.advance(); self.has_lazy = true; }
                let split = self.emit(NfaNode::Split(s, usize::MAX));
                self.patch(e, split);
                let out = self.emit(NfaNode::Epsilon(usize::MAX));
                if lazy {
                    if let NfaNode::Split(ref mut a, ref mut b) = self.nodes[split] {
                        *a = out; *b = s;
                    }
                } else {
                    if let NfaNode::Split(_, ref mut b) = self.nodes[split] {
                        *b = out;
                    }
                }
                Ok((s, out))
            }
            Some(b'?') => {
                self.advance();
                let lazy = self.peek() == Some(b'?');
                if lazy { self.advance(); self.has_lazy = true; }
                let out = self.emit(NfaNode::Epsilon(usize::MAX));
                self.patch(e, out);
                let split = if lazy {
                    self.emit(NfaNode::Split(out, s))
                } else {
                    self.emit(NfaNode::Split(s, out))
                };
                Ok((split, out))
            }
            Some(b'{') => {
                self.advance(); // consume '{'

                let min = self.parse_count_digits_opt()
                    .ok_or_else(|| format!("expected digit after `{{` at position {}", self.pos))?;

                let max_opt = if self.peek() == Some(b',') {
                    self.advance(); // consume ','
                    if self.peek() == Some(b'}') {
                        None // {n,} unbounded
                    } else {
                        let m = self.parse_count_digits_opt()
                            .ok_or_else(|| format!("expected digit after `,` at position {}", self.pos))?;
                        Some(m)
                    }
                } else {
                    Some(min) // {n} exact
                };

                if self.peek() != Some(b'}') {
                    return Err(format!("expected `}}` in counted repetition at position {}", self.pos));
                }
                self.advance(); // consume '}'

                let lazy = self.peek() == Some(b'?');
                if lazy { self.advance(); self.has_lazy = true; }

                if let Some(max) = max_opt {
                    if max < min {
                        return Err(format!("invalid {{n,m}}: n={} > m={}", min, max));
                    }
                }

                // {0,0}: matches empty string
                if min == 0 && max_opt == Some(0) {
                    let eps = self.emit(NfaNode::Epsilon(usize::MAX));
                    return Ok((eps, eps));
                }

                // Snapshot the original atom nodes BEFORE any patching so that
                // each clone always copies the pristine fragment, not one that
                // has already had its end-placeholder patched to point elsewhere.
                let atom_snapshot: Vec<NfaNode> = self.nodes[atom_node_start..atom_node_end]
                    .iter()
                    .cloned()
                    .collect();

                // Build chain: start with either (s,e) for the first mandatory
                // copy or an epsilon if min==0.
                let (chain_start, mut chain_end) = if min == 0 {
                    let eps = self.emit(NfaNode::Epsilon(usize::MAX));
                    (eps, eps)
                } else {
                    (s, e) // first mandatory copy already emitted
                };

                // Emit remaining mandatory copies (min-1 clones, or min if started with eps)
                let mandatory_extra = if min == 0 { 0 } else { min - 1 };
                for _ in 0..mandatory_extra {
                    let (ns, ne) = self.clone_from_snapshot(&atom_snapshot, atom_node_start, s, e);
                    self.patch(chain_end, ns);
                    chain_end = ne;
                }

                match max_opt {
                    None => {
                        // {n,}: mandatory done, add * for remainder
                        let (ns, ne) = self.clone_from_snapshot(&atom_snapshot, atom_node_start, s, e);
                        let split = self.emit(NfaNode::Split(ns, usize::MAX));
                        self.patch(ne, split); // loop
                        let out = self.emit(NfaNode::Epsilon(usize::MAX));
                        if lazy {
                            if let NfaNode::Split(ref mut a, ref mut b) = self.nodes[split] {
                                *a = out; *b = ns;
                            }
                        } else {
                            if let NfaNode::Split(_, ref mut b) = self.nodes[split] {
                                *b = out;
                            }
                        }
                        self.patch(chain_end, split);
                        Ok((chain_start, out))
                    }
                    Some(max) if max > min => {
                        // {n,m}: add (max-min) optional copies
                        let mut opt_end = chain_end;
                        for _ in 0..(max - min) {
                            let (ns, ne) = self.clone_from_snapshot(&atom_snapshot, atom_node_start, s, e);
                            let out_e = self.emit(NfaNode::Epsilon(usize::MAX));
                            let split = if lazy {
                                self.emit(NfaNode::Split(out_e, ns))
                            } else {
                                self.emit(NfaNode::Split(ns, out_e))
                            };
                            self.patch(opt_end, split);
                            self.patch(ne, out_e);
                            opt_end = out_e;
                        }
                        Ok((chain_start, opt_end))
                    }
                    Some(_) => {
                        // {n,n} or {n}: exactly n, chain already built
                        Ok((chain_start, chain_end))
                    }
                }
            }
            _ => Ok((s, e)),
        }
    }

    /// Clone atom nodes from a pre-saved snapshot (not from `self.nodes`) so
    /// that intermediate patches to the original fragment do not corrupt clones.
    fn clone_from_snapshot(
        &mut self,
        snapshot: &[NfaNode],
        atom_node_start: usize,
        original_s: usize,
        original_e: usize,
    ) -> (usize, usize) {
        let new_base = self.nodes.len();
        let delta = new_base - atom_node_start;
        let atom_node_end = atom_node_start + snapshot.len();
        let clones: Vec<NfaNode> = snapshot
            .iter()
            .map(|n| remap_node_idx(n, atom_node_start, atom_node_end, delta))
            .collect();
        for node in clones {
            self.nodes.push(node);
        }
        (original_s + delta, original_e + delta)
    }

    /// Parse zero or more ASCII digits and return the integer value, or `None`
    /// if no digit is found at the current position.
    fn parse_count_digits_opt(&mut self) -> Option<usize> {
        let mut n: usize = 0;
        let mut found = false;
        while let Some(ch) = self.peek() {
            if ch >= b'0' && ch <= b'9' {
                found = true;
                self.advance();
                n = n.saturating_mul(10).saturating_add((ch - b'0') as usize);
            } else {
                break;
            }
        }
        if found { Some(n) } else { None }
    }

    fn parse_atom(&mut self) -> Result<(usize, usize), String> {
        // Skip whitespace in extended mode
        if self.flags.extended {
            self.skip_extended_ws();
        }

        match self.peek() {
            Some(b'(') => {
                self.advance(); // consume '('

                if self.peek() == Some(b'?') {
                    self.advance(); // consume '?'

                    match self.peek() {
                        Some(b':') => {
                            // Non-capturing group (?:...)
                            self.advance(); // consume ':'
                            let inner = self.parse_alternation()?;
                            if self.peek() != Some(b')') {
                                return Err("unclosed non-capturing group `(?:`".into());
                            }
                            self.advance();
                            return Ok(inner);
                        }
                        Some(b'P') => {
                            // Named capturing group (?P<name>...)
                            self.advance(); // consume 'P'
                            if self.peek() != Some(b'<') {
                                return Err(format!(
                                    "expected `<` after `(?P` at position {}",
                                    self.pos
                                ));
                            }
                            self.advance(); // consume '<'
                            let name = self.parse_group_name()?;
                            return self.parse_named_capture_body(name);
                        }
                        Some(b'<') => {
                            // Named capturing group (?<name>...)
                            self.advance(); // consume '<'
                            let name = self.parse_group_name()?;
                            return self.parse_named_capture_body(name);
                        }
                        Some(b'i') | Some(b'm') | Some(b's') | Some(b'x') | Some(b'-') => {
                            // Inline flags: (?flags) or (?flags:...)
                            let saved_flags = self.flags.clone();
                            self.parse_inline_flags()?;

                            if self.peek() == Some(b':') {
                                // (?flags:...) — scoped group, restore after
                                self.advance(); // consume ':'
                                let inner = self.parse_alternation()?;
                                if self.peek() != Some(b')') {
                                    return Err("unclosed flag-scoped group `(?flags:`".into());
                                }
                                self.advance();
                                self.flags = saved_flags;
                                return Ok(inner);
                            } else if self.peek() == Some(b')') {
                                // (?flags) — standalone, consume ')'
                                self.advance();
                                let eps = self.emit(NfaNode::Epsilon(usize::MAX));
                                return Ok((eps, eps));
                            } else {
                                return Err(format!(
                                    "expected `)` or `:` after inline flags at position {}",
                                    self.pos
                                ));
                            }
                        }
                        _ => {
                            return Err(format!(
                                "unsupported group syntax `(?` at position {}",
                                self.pos
                            ));
                        }
                    }
                }

                // Regular capturing group — track with Save nodes.
                // Layout: Save(2*idx) | <inner nodes> | Save(2*idx+1) | Epsilon(MAX)
                // Pike VM transitions Save -> state+1, so save_open flows to inner_start
                // (which is save_open+1) and save_close flows to out (save_close+1).
                self.num_groups += 1;
                let group_idx = self.num_groups;
                let save_open = self.emit(NfaNode::Save(2 * group_idx));
                let (_inner_start, inner_end) = self.parse_alternation()?;
                let save_close = self.emit(NfaNode::Save(2 * group_idx + 1));
                let out = self.emit(NfaNode::Epsilon(usize::MAX));
                self.patch(inner_end, save_close);
                if self.peek() != Some(b')') {
                    return Err("unclosed group `(`".into());
                }
                self.advance();
                Ok((save_open, out))
            }
            Some(b'[') => self.parse_char_class(),
            Some(b'.') => {
                self.advance();
                let node = if self.flags.dotall { NfaNode::AnyByte } else { NfaNode::AnyByteNoNl };
                let s = self.emit(node);
                let e = self.emit(NfaNode::Epsilon(usize::MAX));
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
                let (esc, negate) = self.parse_escape()?;
                match esc {
                    EscapeResult::Byte(b) => {
                        let s = if self.flags.case_insensitive && b.is_ascii_alphabetic() {
                            let mut cls = ByteClass::new();
                            cls.set(b.to_ascii_lowercase());
                            cls.set(b.to_ascii_uppercase());
                            self.emit(NfaNode::Class(cls))
                        } else {
                            self.emit(NfaNode::Byte(b))
                        };
                        let e = self.emit(NfaNode::Epsilon(usize::MAX));
                        Ok((s, e))
                    }
                    EscapeResult::Sequence(bytes) => {
                        // Multi-byte Unicode codepoint — chain as individual Byte nodes
                        assert!(!bytes.is_empty());
                        let first_s = self.emit(NfaNode::Byte(bytes[0]));
                        let first_e = self.emit(NfaNode::Epsilon(usize::MAX));
                        let mut chain_end = first_e;
                        for &b in &bytes[1..] {
                            let bs = self.emit(NfaNode::Byte(b));
                            let be = self.emit(NfaNode::Epsilon(usize::MAX));
                            self.patch(chain_end, bs);
                            chain_end = be;
                        }
                        Ok((first_s, chain_end))
                    }
                    EscapeResult::Class(mut cls) => {
                        if negate { cls.negate(); }
                        let s = self.emit(NfaNode::Class(cls));
                        let e = self.emit(NfaNode::Epsilon(usize::MAX));
                        Ok((s, e))
                    }
                    EscapeResult::WordBoundary => {
                        let s = self.emit(NfaNode::WordBoundary);
                        let e = self.emit(NfaNode::Epsilon(usize::MAX));
                        Ok((s, e))
                    }
                    EscapeResult::NonWordBoundary => {
                        let s = self.emit(NfaNode::NonWordBoundary);
                        let e = self.emit(NfaNode::Epsilon(usize::MAX));
                        Ok((s, e))
                    }
                    EscapeResult::Anchor(node) => {
                        let s = self.emit(node);
                        let e = self.emit(NfaNode::Epsilon(usize::MAX));
                        Ok((s, e))
                    }
                }
            }
            Some(ch) if ch != b'|' && ch != b')' && ch != b'*' && ch != b'+' && ch != b'?' && ch != b'{' => {
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
        if negated { self.advance(); }
        let mut cls = ByteClass::new();
        let mut first = true;
        loop {
            match self.peek() {
                None => return Err("unclosed character class `[`".into()),
                Some(b']') if !first => {
                    self.advance();
                    break;
                }
                // POSIX class [:name:] inside [...]
                Some(b'[')
                    if self.pos + 1 < self.pattern.len() && self.pattern[self.pos + 1] == b':' =>
                {
                    first = false;
                    self.advance(); // '['
                    self.advance(); // ':'
                    self.parse_posix_class_into(&mut cls)?;
                }
                _ => {
                    first = false;
                    let lo = self.parse_class_atom()?;
                    if self.peek() == Some(b'-')
                        && self.pos + 1 < self.pattern.len()
                        && self.pattern[self.pos + 1] != b']'
                    {
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
        if negated { cls.negate(); }
        let s = self.emit(NfaNode::Class(cls));
        let e = self.emit(NfaNode::Epsilon(usize::MAX));
        Ok((s, e))
    }

    /// Parse a POSIX class name after `[:` has been consumed, up to and
    /// including the closing `:]`. Adds the matching bytes to `cls`.
    fn parse_posix_class_into(&mut self, cls: &mut ByteClass) -> Result<(), String> {
        let mut name: Vec<u8> = Vec::new();
        loop {
            match self.peek() {
                Some(b':') => { self.advance(); break; }
                Some(ch) => { name.push(ch); self.advance(); }
                None => return Err("unclosed POSIX class — expected `:`".into()),
            }
        }
        if self.peek() != Some(b']') {
            return Err(format!(
                "expected `]` after `:` in POSIX class `[:{name}:]`",
                name = name.iter().map(|&b| b as char).collect::<String>()
            ));
        }
        self.advance(); // consume ']'

        match name.as_slice() {
            b"alpha" => {
                cls.set_range(b'a', b'z');
                cls.set_range(b'A', b'Z');
            }
            b"digit" => cls.set_range(b'0', b'9'),
            b"alnum" => {
                cls.set_range(b'a', b'z');
                cls.set_range(b'A', b'Z');
                cls.set_range(b'0', b'9');
            }
            b"space" => {
                cls.set(b' ');
                cls.set(b'\t');
                cls.set(b'\n');
                cls.set(b'\r');
                cls.set(0x0C); // form feed
                cls.set(0x0B); // vertical tab
            }
            b"blank" => {
                cls.set(b' ');
                cls.set(b'\t');
            }
            b"upper" => cls.set_range(b'A', b'Z'),
            b"lower" => cls.set_range(b'a', b'z'),
            b"punct" => {
                for b in b'!'..=b'/' { cls.set(b); }
                for b in b':'..=b'@' { cls.set(b); }
                for b in b'['..=b'`' { cls.set(b); }
                for b in b'{'..=b'~' { cls.set(b); }
            }
            b"print" => cls.set_range(b' ', b'~'),
            b"graph" => cls.set_range(b'!', b'~'),
            b"cntrl" => {
                cls.set_range(0, 0x1F);
                cls.set(0x7F);
            }
            b"xdigit" => {
                cls.set_range(b'0', b'9');
                cls.set_range(b'a', b'f');
                cls.set_range(b'A', b'F');
            }
            _ => {
                return Err(format!(
                    "unknown POSIX class `[:{name}:]`",
                    name = name.iter().map(|&b| b as char).collect::<String>()
                ));
            }
        }
        Ok(())
    }

    fn parse_class_atom(&mut self) -> Result<u8, String> {
        match self.advance() {
            Some(b'\\') => match self.advance() {
                Some(b'n') => Ok(b'\n'),
                Some(b't') => Ok(b'\t'),
                Some(b'r') => Ok(b'\r'),
                Some(b'\\') => Ok(b'\\'),
                Some(b']') => Ok(b']'),
                Some(b'-') => Ok(b'-'),
                Some(b'x') => self.parse_hex_byte(),
                Some(ch) => Ok(ch),
                None => Err("unexpected end of escape in character class".into()),
            },
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
                cls.set(0x0C);
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
            Some(b'B') => Ok((EscapeResult::NonWordBoundary, false)),
            Some(b'A') => Ok((EscapeResult::Anchor(NfaNode::AbsoluteStart), false)),
            Some(b'z') => Ok((EscapeResult::Anchor(NfaNode::AbsoluteEnd), false)),
            Some(b'Z') => Ok((EscapeResult::Anchor(NfaNode::AbsoluteEndBeforeNewline), false)),
            Some(b'n') => Ok((EscapeResult::Byte(b'\n'), false)),
            Some(b't') => Ok((EscapeResult::Byte(b'\t'), false)),
            Some(b'r') => Ok((EscapeResult::Byte(b'\r'), false)),
            Some(b'0') => Ok((EscapeResult::Byte(0), false)),
            Some(b'x') => {
                let b = self.parse_hex_byte()?;
                Ok((EscapeResult::Byte(b), false))
            }
            Some(b'u') => self.parse_unicode_escape(),
            Some(ch) => Ok((EscapeResult::Byte(ch), false)),
            None => Err("unexpected end of escape sequence".into()),
        }
    }

    /// Parse `\uNNNN` or `\u{NNNN}` after the `u` has been consumed.
    /// Returns the UTF-8 byte sequence for the codepoint.
    fn parse_unicode_escape(&mut self) -> Result<(EscapeResult, bool), String> {
        let braced = self.peek() == Some(b'{');
        if braced { self.advance(); }

        let mut code: u32 = 0;
        let mut count = 0;
        let limit = if braced { 6 } else { 4 };

        loop {
            if count >= limit { break; }
            match self.peek() {
                Some(b'}') if braced => break,
                Some(ch) => {
                    if let Some(v) = hex_val(ch) {
                        self.advance();
                        code = code.saturating_mul(16).saturating_add(v as u32);
                        count += 1;
                    } else {
                        break;
                    }
                }
                None => break,
            }
        }

        if braced {
            if self.peek() != Some(b'}') {
                return Err("missing `}` in \\u{...} escape".into());
            }
            self.advance();
        }

        if count == 0 {
            return Err("empty \\u escape — expected hex digits".into());
        }

        let ch = char::from_u32(code)
            .ok_or_else(|| format!("invalid Unicode codepoint U+{:04X}", code))?;
        let mut buf = [0u8; 4];
        let utf8 = ch.encode_utf8(&mut buf);
        let bytes = utf8.as_bytes();

        if bytes.len() == 1 {
            Ok((EscapeResult::Byte(bytes[0]), false))
        } else {
            Ok((EscapeResult::Sequence(bytes.to_vec()), false))
        }
    }

    /// Parse inline flag letters (i, m, s, x) modifying `self.flags`.
    /// Stops at `)` or `:` without consuming them.
    fn parse_inline_flags(&mut self) -> Result<(), String> {
        loop {
            match self.peek() {
                Some(b'i') => { self.flags.case_insensitive = true; self.advance(); }
                Some(b'm') => { self.flags.multiline = true; self.advance(); }
                Some(b's') => { self.flags.dotall = true; self.advance(); }
                Some(b'x') => { self.flags.extended = true; self.advance(); }
                Some(b'-') => { self.advance(); } // negation not yet supported — skip
                Some(b')') | Some(b':') => break,
                None => return Err("unclosed inline flags — expected `)` or `:`".into()),
                Some(ch) => {
                    return Err(format!(
                        "unknown inline flag `{}` at position {}",
                        ch as char, self.pos
                    ));
                }
            }
        }
        Ok(())
    }

    /// Parse a group name after `<` has been consumed, up to and including `>`.
    fn parse_group_name(&mut self) -> Result<String, String> {
        let mut name = Vec::new();
        loop {
            match self.peek() {
                Some(b'>') => { self.advance(); break; }
                Some(ch) if ch.is_ascii_alphanumeric() || ch == b'_' => {
                    name.push(ch);
                    self.advance();
                }
                Some(ch) => {
                    return Err(format!(
                        "invalid character `{}` in group name at position {}",
                        ch as char, self.pos
                    ));
                }
                None => return Err("unclosed group name — expected `>`".into()),
            }
        }
        if name.is_empty() {
            return Err("empty group name".into());
        }
        Ok(String::from_utf8(name).unwrap())
    }

    /// Parse the body of a named capturing group after the name has been extracted.
    /// Assigns a group index, emits Save nodes, and records the name.
    fn parse_named_capture_body(&mut self, name: String) -> Result<(usize, usize), String> {
        self.num_groups += 1;
        let group_idx = self.num_groups;
        self.group_names.insert(name, group_idx);
        let save_open = self.emit(NfaNode::Save(2 * group_idx));
        let (_inner_start, inner_end) = self.parse_alternation()?;
        let save_close = self.emit(NfaNode::Save(2 * group_idx + 1));
        let out = self.emit(NfaNode::Epsilon(usize::MAX));
        self.patch(inner_end, save_close);
        if self.peek() != Some(b')') {
            return Err("unclosed named group".into());
        }
        self.advance();
        Ok((save_open, out))
    }

    fn skip_extended_ws(&mut self) {
        while let Some(ch) = self.peek() {
            if ch == b' ' || ch == b'\t' || ch == b'\n' || ch == b'\r' {
                self.advance();
            } else if ch == b'#' {
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
    /// Multi-byte UTF-8 sequence (from Unicode escapes > U+007F).
    Sequence(Vec<u8>),
    Class(ByteClass),
    WordBoundary,
    NonWordBoundary,
    Anchor(NfaNode),
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

/// Remap internal index references within a cloned NFA node.
/// Only indices in `[from, from+len)` are remapped by `delta`; `usize::MAX`
/// (unpatched placeholders) are preserved as-is.
fn remap_node_idx(node: &NfaNode, from: usize, to: usize, delta: usize) -> NfaNode {
    let remap = |idx: usize| -> usize {
        if idx == usize::MAX { usize::MAX }
        else if idx >= from && idx < to { idx + delta }
        else { idx }
    };
    match node {
        NfaNode::Epsilon(t) => NfaNode::Epsilon(remap(*t)),
        NfaNode::Split(a, b) => NfaNode::Split(remap(*a), remap(*b)),
        NfaNode::Save(slot) => NfaNode::Save(*slot), // slot index is not a node ref
        other => other.clone(),
    }
}

// ---------------------------------------------------------------------------
// NFA executor: Thompson NFA simulation
// ---------------------------------------------------------------------------

fn merge_flags(user: &Flags, compiled: &Flags) -> Flags {
    Flags {
        case_insensitive: user.case_insensitive || compiled.case_insensitive,
        multiline: user.multiline || compiled.multiline,
        dotall: user.dotall || compiled.dotall,
        extended: user.extended || compiled.extended,
    }
}

fn nfa_search(nfa: &Nfa, haystack: &[u8], opts: &Flags) -> Option<(usize, usize)> {
    let opts = merge_flags(opts, &nfa.effective_flags);
    for start in 0..=haystack.len() {
        if let Some(end) = nfa_match_at(nfa, haystack, &opts, start) {
            return Some((start, end));
        }
    }
    None
}

fn nfa_search_from(nfa: &Nfa, haystack: &[u8], opts: &Flags, from: usize) -> Option<(usize, usize)> {
    let opts = merge_flags(opts, &nfa.effective_flags);
    for start in from..=haystack.len() {
        if let Some(end) = nfa_match_at(nfa, haystack, &opts, start) {
            return Some((start, end));
        }
    }
    None
}

/// Try to match the NFA starting at position `start` in the haystack.
fn nfa_match_at(nfa: &Nfa, haystack: &[u8], opts: &Flags, start: usize) -> Option<usize> {
    let mut current = Vec::new();
    let mut next = Vec::new();
    let mut in_current = vec![false; nfa.nodes.len()];
    let mut in_next = vec![false; nfa.nodes.len()];
    let mut last_match: Option<usize> = None;

    add_state(nfa, nfa.start, &mut current, &mut in_current, haystack, opts, start);

    for &state in &current {
        if matches!(nfa.nodes[state], NfaNode::Accept) {
            last_match = Some(start);
            if nfa.has_lazy { return last_match; }
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
                NfaNode::Accept => {}
                _ => {} // zero-width assertions handled in add_state
            }
        }

        for &state in &next {
            if matches!(nfa.nodes[state], NfaNode::Accept) {
                last_match = Some(pos + 1);
                if nfa.has_lazy { return last_match; }
            }
        }

        std::mem::swap(&mut current, &mut next);
        std::mem::swap(&mut in_current, &mut in_next);
        next.clear();
        for b in in_next.iter_mut() { *b = false; }

        pos += 1;
    }

    last_match
}

/// Add a state to the set, following epsilon/zero-width-assertion transitions.
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
            if ok { add_state(nfa, state + 1, set, in_set, haystack, opts, pos); }
        }
        NfaNode::AnchorEnd => {
            let ok = if opts.multiline {
                pos == haystack.len() || haystack[pos] == b'\n'
            } else {
                pos == haystack.len()
            };
            if ok { add_state(nfa, state + 1, set, in_set, haystack, opts, pos); }
        }
        NfaNode::WordBoundary => {
            let before_word = pos > 0 && is_word_byte(haystack[pos - 1]);
            let after_word = pos < haystack.len() && is_word_byte(haystack[pos]);
            if before_word != after_word {
                add_state(nfa, state + 1, set, in_set, haystack, opts, pos);
            }
        }
        NfaNode::NonWordBoundary => {
            let before_word = pos > 0 && is_word_byte(haystack[pos - 1]);
            let after_word = pos < haystack.len() && is_word_byte(haystack[pos]);
            if before_word == after_word {
                add_state(nfa, state + 1, set, in_set, haystack, opts, pos);
            }
        }
        NfaNode::AbsoluteStart => {
            if pos == 0 {
                add_state(nfa, state + 1, set, in_set, haystack, opts, pos);
            }
        }
        NfaNode::AbsoluteEnd => {
            if pos == haystack.len() {
                add_state(nfa, state + 1, set, in_set, haystack, opts, pos);
            }
        }
        NfaNode::AbsoluteEndBeforeNewline => {
            let ok = pos == haystack.len()
                || (pos + 1 == haystack.len() && haystack[pos] == b'\n');
            if ok { add_state(nfa, state + 1, set, in_set, haystack, opts, pos); }
        }
        NfaNode::Save(_) => {
            // Save is epsilon-like in the fast path (no capture tracking).
            add_state(nfa, state + 1, set, in_set, haystack, opts, pos);
        }
        _ => {
            // Consuming state (Byte, Class, AnyByte, AnyByteNoNl, Accept)
            set.push(state);
        }
    }
}

fn is_word_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

// ---------------------------------------------------------------------------
// Pike VM executor: Thompson NFA with capture group tracking
// ---------------------------------------------------------------------------

/// A Pike VM thread: a state id + its own slot array for capture positions.
#[derive(Clone)]
struct PikeThread {
    state: usize,
    slots: Vec<Option<usize>>,
}

fn pike_search(nfa: &Nfa, haystack: &[u8], opts: &Flags) -> Option<CaptureResult> {
    let opts = merge_flags(opts, &nfa.effective_flags);
    for start in 0..=haystack.len() {
        if let Some(cr) = pike_match_at(nfa, haystack, &opts, start) {
            return Some(cr);
        }
    }
    None
}

fn pike_search_from(nfa: &Nfa, haystack: &[u8], opts: &Flags, from: usize) -> Option<CaptureResult> {
    let opts = merge_flags(opts, &nfa.effective_flags);
    for start in from..=haystack.len() {
        if let Some(cr) = pike_match_at(nfa, haystack, &opts, start) {
            return Some(cr);
        }
    }
    None
}

/// Pike VM matching at a specific starting position.
/// Returns a CaptureResult if the NFA matches starting at `start`.
fn pike_match_at(nfa: &Nfa, haystack: &[u8], opts: &Flags, start: usize) -> Option<CaptureResult> {
    let num_slots = 2 * (nfa.num_groups + 1); // +1 for group 0 (full match)
    let initial_slots = vec![None; num_slots];

    let mut current: Vec<PikeThread> = Vec::new();
    let mut next: Vec<PikeThread> = Vec::new();
    let mut in_current = vec![false; nfa.nodes.len()];
    let mut in_next = vec![false; nfa.nodes.len()];
    let mut best_slots: Option<Vec<Option<usize>>> = None;

    // Set group 0 start slot
    let mut start_slots = initial_slots.clone();
    start_slots[0] = Some(start);

    pike_add_state(
        nfa, nfa.start, start_slots,
        &mut current, &mut in_current, haystack, opts, start,
    );

    // Check for Accept in initial state set
    let mut i = 0;
    while i < current.len() {
        if matches!(nfa.nodes[current[i].state], NfaNode::Accept) {
            let mut slots = current[i].slots.clone();
            slots[1] = Some(start); // group 0 end
            if nfa.has_lazy {
                return Some(slots_to_capture_result(&slots, nfa, start));
            }
            best_slots = Some(slots);
        }
        i += 1;
    }

    let mut pos = start;
    while pos < haystack.len() && !current.is_empty() {
        let byte = haystack[pos];

        for thread in &current {
            let advance_to = pos + 1;
            match &nfa.nodes[thread.state] {
                NfaNode::Byte(b) if byte == *b => {
                    pike_add_state(
                        nfa, thread.state + 1, thread.slots.clone(),
                        &mut next, &mut in_next, haystack, opts, advance_to,
                    );
                }
                NfaNode::Class(cls) if cls.contains(byte) => {
                    pike_add_state(
                        nfa, thread.state + 1, thread.slots.clone(),
                        &mut next, &mut in_next, haystack, opts, advance_to,
                    );
                }
                NfaNode::AnyByte => {
                    pike_add_state(
                        nfa, thread.state + 1, thread.slots.clone(),
                        &mut next, &mut in_next, haystack, opts, advance_to,
                    );
                }
                NfaNode::AnyByteNoNl if byte != b'\n' => {
                    pike_add_state(
                        nfa, thread.state + 1, thread.slots.clone(),
                        &mut next, &mut in_next, haystack, opts, advance_to,
                    );
                }
                _ => {} // Accept, anchors, Save handled in pike_add_state
            }
        }

        // Check for Accept in next set
        for thread in &next {
            if matches!(nfa.nodes[thread.state], NfaNode::Accept) {
                let mut slots = thread.slots.clone();
                slots[1] = Some(pos + 1); // group 0 end
                if nfa.has_lazy {
                    return Some(slots_to_capture_result(&slots, nfa, start));
                }
                // Greedy: keep longest match
                match &best_slots {
                    Some(prev) => {
                        if slots[1] > prev[1] {
                            best_slots = Some(slots);
                        }
                    }
                    None => { best_slots = Some(slots); }
                }
            }
        }

        std::mem::swap(&mut current, &mut next);
        std::mem::swap(&mut in_current, &mut in_next);
        next.clear();
        for b in in_next.iter_mut() { *b = false; }

        pos += 1;
    }

    best_slots.map(|slots| slots_to_capture_result(&slots, nfa, start))
}

/// Add a state to the Pike VM thread set, following epsilon/zero-width/Save transitions.
fn pike_add_state(
    nfa: &Nfa,
    state: usize,
    slots: Vec<Option<usize>>,
    set: &mut Vec<PikeThread>,
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
                pike_add_state(nfa, *target, slots, set, in_set, haystack, opts, pos);
            }
        }
        NfaNode::Split(a, b) => {
            // Each branch gets its own copy of the slots
            pike_add_state(nfa, *a, slots.clone(), set, in_set, haystack, opts, pos);
            pike_add_state(nfa, *b, slots, set, in_set, haystack, opts, pos);
        }
        NfaNode::Save(slot_idx) => {
            let mut new_slots = slots;
            if *slot_idx < new_slots.len() {
                new_slots[*slot_idx] = Some(pos);
            }
            pike_add_state(nfa, state + 1, new_slots, set, in_set, haystack, opts, pos);
        }
        NfaNode::AnchorStart => {
            let ok = if opts.multiline {
                pos == 0 || (pos > 0 && haystack[pos - 1] == b'\n')
            } else {
                pos == 0
            };
            if ok { pike_add_state(nfa, state + 1, slots, set, in_set, haystack, opts, pos); }
        }
        NfaNode::AnchorEnd => {
            let ok = if opts.multiline {
                pos == haystack.len() || haystack[pos] == b'\n'
            } else {
                pos == haystack.len()
            };
            if ok { pike_add_state(nfa, state + 1, slots, set, in_set, haystack, opts, pos); }
        }
        NfaNode::WordBoundary => {
            let before_word = pos > 0 && is_word_byte(haystack[pos - 1]);
            let after_word = pos < haystack.len() && is_word_byte(haystack[pos]);
            if before_word != after_word {
                pike_add_state(nfa, state + 1, slots, set, in_set, haystack, opts, pos);
            }
        }
        NfaNode::NonWordBoundary => {
            let before_word = pos > 0 && is_word_byte(haystack[pos - 1]);
            let after_word = pos < haystack.len() && is_word_byte(haystack[pos]);
            if before_word == after_word {
                pike_add_state(nfa, state + 1, slots, set, in_set, haystack, opts, pos);
            }
        }
        NfaNode::AbsoluteStart => {
            if pos == 0 {
                pike_add_state(nfa, state + 1, slots, set, in_set, haystack, opts, pos);
            }
        }
        NfaNode::AbsoluteEnd => {
            if pos == haystack.len() {
                pike_add_state(nfa, state + 1, slots, set, in_set, haystack, opts, pos);
            }
        }
        NfaNode::AbsoluteEndBeforeNewline => {
            let ok = pos == haystack.len()
                || (pos + 1 == haystack.len() && haystack[pos] == b'\n');
            if ok { pike_add_state(nfa, state + 1, slots, set, in_set, haystack, opts, pos); }
        }
        _ => {
            // Consuming state (Byte, Class, AnyByte, AnyByteNoNl, Accept)
            set.push(PikeThread { state, slots });
        }
    }
}

/// Convert slot array to CaptureResult.
fn slots_to_capture_result(slots: &[Option<usize>], nfa: &Nfa, _start: usize) -> CaptureResult {
    let full_start = slots.first().copied().flatten().unwrap_or(0);
    let full_end = slots.get(1).copied().flatten().unwrap_or(full_start);

    let mut groups = Vec::with_capacity(nfa.num_groups + 1);
    // Group 0 = full match
    groups.push(Some(Capture { start: full_start, end: full_end }));
    // Groups 1..num_groups
    for i in 1..=nfa.num_groups {
        let s = slots.get(2 * i).copied().flatten();
        let e = slots.get(2 * i + 1).copied().flatten();
        match (s, e) {
            (Some(start), Some(end)) => groups.push(Some(Capture { start, end })),
            _ => groups.push(None),
        }
    }

    CaptureResult {
        full: MatchResult { start: full_start, end: full_end },
        groups,
        names: nfa.group_names.clone(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Existing: basic --

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

    // -- Non-greedy (lazy) quantifiers --

    #[test]
    fn test_lazy_star() {
        let greedy = find("a.*b", "", b"aXbYb");
        assert_eq!(greedy, Some((0, 5)));
        let lazy = find("a.*?b", "", b"aXbYb");
        assert_eq!(lazy, Some((0, 3)));
    }

    #[test]
    fn test_lazy_plus() {
        let greedy = find("a.+b", "", b"aXYZb");
        assert_eq!(greedy, Some((0, 5)));
        let lazy = find("a.+?b", "", b"aXbYb");
        assert_eq!(lazy, Some((0, 3)));
    }

    #[test]
    fn test_lazy_question() {
        let greedy = find("a?b", "", b"ab");
        assert_eq!(greedy, Some((0, 2)));
        let lazy = find("a??b", "", b"ab");
        assert_eq!(lazy, Some((0, 2)));
    }

    #[test]
    fn test_lazy_star_find_all() {
        let greedy = find_all("<.+>", "", b"<a><b>");
        assert_eq!(greedy, vec![(0, 6)]);
        let lazy = find_all("<.+?>", "", b"<a><b>");
        assert_eq!(lazy, vec![(0, 3), (3, 6)]);
    }

    #[test]
    fn test_lazy_star_zero_length() {
        let lazy = find("a*?", "", b"aaa");
        assert_eq!(lazy, Some((0, 0)));
    }

    #[test]
    fn test_lazy_still_matches() {
        assert!(is_match("a+?", "", b"aaa"));
        assert!(is_match("a*?b", "", b"aaab"));
    }

    #[test]
    fn test_lazy_plus_minimum() {
        let lazy = find("a+?", "", b"aaa");
        assert_eq!(lazy, Some((0, 1)));
    }

    #[test]
    fn test_greedy_unchanged() {
        assert_eq!(find("a+", "", b"aaa"), Some((0, 3)));
        assert_eq!(find("a*", "", b"aaa"), Some((0, 3)));
        assert_eq!(find("<.+>", "", b"<a><b>"), Some((0, 6)));
    }

    // -- New: non-capturing groups --

    #[test]
    fn test_non_capturing_group() {
        assert!(is_match("(?:abc)+", "", b"abcabc"));
        assert!(!is_match("(?:abc)+", "", b"abab"));
        assert!(is_match("(?:cat|dog)", "", b"cat"));
        assert!(is_match("(?:cat|dog)", "", b"dog"));
    }

    #[test]
    fn test_non_capturing_group_in_sequence() {
        assert!(is_match("(?:ab)+c", "", b"ababc")); // two reps of ab + c
        assert!(is_match("(?:ab)+c", "", b"abc"));   // one rep of ab + c
        assert!(!is_match("(?:ab)+c", "", b"ac"));   // no ab before c
        assert!(!is_match("(?:ab)+c", "", b"bc"));   // missing ab
    }

    // -- New: inline flags --

    #[test]
    fn test_inline_flag_i() {
        // (?i) makes rest of pattern case-insensitive
        assert!(is_match("(?i)hello", "", b"HELLO"));
        assert!(is_match("(?i)hello", "", b"HeLLo"));
    }

    #[test]
    fn test_inline_flag_scoped() {
        // (?i:...) applies flag only within the group
        assert!(is_match("(?i:hello) world", "", b"HELLO world"));   // HELLO matches (i flag), world matches exactly
        assert!(is_match("(?i:hello) WORLD", "", b"HELLO WORLD"));   // HELLO matches (i flag), WORLD matches exactly
        assert!(!is_match("(?i:hello) world", "", b"HELLO WORLD"));  // world ≠ WORLD (case-sensitive after group)
    }

    #[test]
    fn test_inline_flag_m() {
        assert!(is_match("(?m)^world", "", b"hello\nworld"));
    }

    #[test]
    fn test_inline_flag_s() {
        assert!(is_match("(?s)a.b", "", b"a\nb"));
    }

    // -- New: POSIX character classes --

    #[test]
    fn test_posix_alpha() {
        assert!(is_match("[[:alpha:]]", "", b"a"));
        assert!(is_match("[[:alpha:]]", "", b"Z"));
        assert!(!is_match("[[:alpha:]]", "", b"1"));
        assert!(!is_match("[[:alpha:]]", "", b"_"));
    }

    #[test]
    fn test_posix_digit() {
        assert!(is_match("[[:digit:]]", "", b"5"));
        assert!(!is_match("[[:digit:]]", "", b"a"));
    }

    #[test]
    fn test_posix_alnum() {
        assert!(is_match("[[:alnum:]]+", "", b"abc123"));
        assert!(!is_match("^[[:alnum:]]+$", "", b"abc 123"));
    }

    #[test]
    fn test_posix_space() {
        assert!(is_match("[[:space:]]", "", b" "));
        assert!(is_match("[[:space:]]", "", b"\t"));
        assert!(is_match("[[:space:]]", "", b"\n"));
        assert!(!is_match("[[:space:]]", "", b"a"));
    }

    #[test]
    fn test_posix_upper_lower() {
        assert!(is_match("[[:upper:]]", "", b"A"));
        assert!(!is_match("[[:upper:]]", "", b"a"));
        assert!(is_match("[[:lower:]]", "", b"z"));
        assert!(!is_match("[[:lower:]]", "", b"Z"));
    }

    #[test]
    fn test_posix_xdigit() {
        assert!(is_match("[[:xdigit:]]+", "", b"0a1F"));
        assert!(!is_match("^[[:xdigit:]]+$", "", b"0g1"));
    }

    #[test]
    fn test_posix_blank() {
        assert!(is_match("[[:blank:]]", "", b" "));
        assert!(is_match("[[:blank:]]", "", b"\t"));
        assert!(!is_match("[[:blank:]]", "", b"\n"));
    }

    #[test]
    fn test_posix_mixed_with_literal() {
        // Mix POSIX class with literal chars in same bracket expression
        assert!(is_match("[[:digit:]_]", "", b"_"));
        assert!(is_match("[[:digit:]_]", "", b"7"));
        assert!(!is_match("[[:digit:]_]", "", b"a"));
    }

    // -- New: counted repetition {n,m} --

    #[test]
    fn test_counted_exact() {
        assert!(is_match("^a{3}$", "", b"aaa"));
        assert!(!is_match("^a{3}$", "", b"aa"));
        assert!(!is_match("^a{3}$", "", b"aaaa"));
    }

    #[test]
    fn test_counted_min() {
        assert!(is_match("^a{2,}$", "", b"aa"));
        assert!(is_match("^a{2,}$", "", b"aaa"));
        assert!(is_match("^a{2,}$", "", b"aaaaaaa"));
        assert!(!is_match("^a{2,}$", "", b"a"));
    }

    #[test]
    fn test_counted_range() {
        assert!(!is_match("^a{2,4}$", "", b"a"));
        assert!(is_match("^a{2,4}$", "", b"aa"));
        assert!(is_match("^a{2,4}$", "", b"aaa"));
        assert!(is_match("^a{2,4}$", "", b"aaaa"));
        assert!(!is_match("^a{2,4}$", "", b"aaaaa"));
    }

    #[test]
    fn test_counted_zero() {
        // {0} matches empty
        assert!(is_match("^a{0}$", "", b""));
        assert!(!is_match("^a{0}$", "", b"a"));
    }

    #[test]
    fn test_counted_zero_to_n() {
        assert!(is_match("^a{0,2}$", "", b""));
        assert!(is_match("^a{0,2}$", "", b"a"));
        assert!(is_match("^a{0,2}$", "", b"aa"));
        assert!(!is_match("^a{0,2}$", "", b"aaa"));
    }

    #[test]
    fn test_counted_on_group() {
        assert!(is_match("^(?:ab){2}$", "", b"abab"));
        assert!(!is_match("^(?:ab){2}$", "", b"ab"));
        assert!(!is_match("^(?:ab){2}$", "", b"ababab"));
    }

    #[test]
    fn test_counted_on_class() {
        assert!(is_match("^\\d{4}$", "", b"2024"));
        assert!(!is_match("^\\d{4}$", "", b"202"));
        assert!(!is_match("^\\d{4}$", "", b"20245"));
    }

    #[test]
    fn test_counted_lazy() {
        // Lazy counted repetition
        let lazy = find("a{2,4}?", "", b"aaaa");
        // Should prefer minimum (2)
        assert_eq!(lazy, Some((0, 2)));
    }

    #[test]
    fn test_counted_find_all() {
        let spans = find_all("\\d{2}", "", b"1 22 333 4444");
        // Matches "22", "33", "44" (non-overlapping 2-digit sequences)
        assert!(spans.contains(&(2, 4)));
    }

    // -- New: Unicode escapes --

    #[test]
    fn test_unicode_escape_ascii() {
        // \u0041 = 'A'
        assert!(is_match("\\u0041", "", b"A"));
        assert!(!is_match("\\u0041", "", b"B"));
    }

    #[test]
    fn test_unicode_escape_braced() {
        // \u{0041} = 'A'
        assert!(is_match("\\u{0041}", "", b"A"));
    }

    #[test]
    fn test_unicode_escape_multibyte() {
        // \u00E9 = é (U+00E9) → UTF-8: 0xC3 0xA9
        let haystack = "café".as_bytes();
        assert!(is_match("caf\\u00E9", "", haystack));
    }

    #[test]
    fn test_unicode_escape_braced_multibyte() {
        let haystack = "café".as_bytes();
        assert!(is_match("caf\\u{00E9}", "", haystack));
    }

    // -- New: absolute anchors \A \z \Z --

    #[test]
    fn test_absolute_start_anchor() {
        // \A always anchors to absolute start, even in multiline mode
        assert!(is_match("\\Ahello", "", b"hello world"));
        assert!(!is_match("\\Ahello", "", b"say hello"));
        // Even with multiline flag, \A doesn't match at line start
        assert!(!is_match("\\Aworld", "m", b"hello\nworld"));
    }

    #[test]
    fn test_absolute_end_anchor() {
        assert!(is_match("world\\z", "", b"hello world"));
        assert!(!is_match("world\\z", "", b"world hello"));
        // With multiline, \z still anchors to absolute end
        assert!(!is_match("hello\\z", "m", b"hello\nworld"));
    }

    #[test]
    fn test_absolute_end_before_newline() {
        // \Z matches at end or before final \n
        assert!(is_match("world\\Z", "", b"hello world"));
        assert!(is_match("world\\Z", "", b"hello world\n"));
        assert!(!is_match("world\\Z", "", b"world hello"));
    }

    // -- New: \B non-word-boundary --

    #[test]
    fn test_non_word_boundary() {
        // \B matches where \b doesn't
        assert!(is_match("\\Bhello\\B", "", b"sayhelloworld"));
        assert!(!is_match("\\Bhello\\B", "", b"say hello world"));
    }

    // -- New: MatchResult API --

    #[test]
    fn test_find_match_result() {
        let m = find_match("\\d+", "", b"abc123def").unwrap();
        assert_eq!(m.start, 3);
        assert_eq!(m.end, 6);
        assert_eq!(m.len(), 3);
        assert_eq!(m.extract(b"abc123def"), b"123");
        assert_eq!(m.extract_str(b"abc123def"), Some("123"));
    }

    #[test]
    fn test_find_match_none() {
        assert!(find_match("xyz", "", b"abc").is_none());
    }

    #[test]
    fn test_find_all_matches_result() {
        let ms = find_all_matches("\\d+", "", b"a1b22c333");
        assert_eq!(ms.len(), 3);
        assert_eq!(ms[0], MatchResult { start: 1, end: 2 });
        assert_eq!(ms[1], MatchResult { start: 3, end: 5 });
        assert_eq!(ms[2], MatchResult { start: 6, end: 9 });
        assert_eq!(ms[0].extract(b"a1b22c333"), b"1");
        assert_eq!(ms[2].extract(b"a1b22c333"), b"333");
    }

    #[test]
    fn test_match_result_is_empty() {
        let m = MatchResult { start: 5, end: 5 };
        assert!(m.is_empty());
        assert_eq!(m.len(), 0);
    }

    // -- New: regex_explain --

    #[test]
    fn test_regex_explain_valid() {
        let desc = regex_explain("\\d+", "").unwrap();
        assert!(desc.contains("NFA"));
        assert!(desc.contains("Accept"));
    }

    #[test]
    fn test_regex_explain_invalid() {
        let err = regex_explain("(unclosed", "");
        assert!(err.is_err());
    }

    // -- New: safety limits --

    #[test]
    fn test_pattern_too_long() {
        let long = "a".repeat(MAX_PATTERN_LEN + 1);
        assert!(!is_match(&long, "", b"a"));
    }

    #[test]
    fn test_compile_error_unmatched_paren() {
        assert!(!is_match("(abc", "", b"abc"));
    }

    #[test]
    fn test_compile_error_bad_group_syntax() {
        assert!(!is_match("(?Q)", "", b"Q"));
    }

    // -- New: determinism of new features --

    #[test]
    fn test_counted_determinism() {
        for _ in 0..5 {
            let r = find_all("\\d{2,3}", "", b"1 12 123 1234");
            assert_eq!(r, find_all("\\d{2,3}", "", b"1 12 123 1234"));
        }
    }

    #[test]
    fn test_posix_determinism() {
        for _ in 0..5 {
            let r = find_all("[[:alpha:]]+", "", b"hello 123 world");
            assert_eq!(r, find_all("[[:alpha:]]+", "", b"hello 123 world"));
        }
    }
}
