//! Zero-dependency line editor for the CJC REPL.
//!
//! Provides:
//! - Raw terminal mode for character-by-character input
//! - Arrow key navigation (left/right within line, up/down for history)
//! - History stored in Vec<String>, saved to ~/.cjc_history
//! - Multi-line input: lines ending with `{` or `\` continue on next line
//! - Home/End key support
//! - Backspace and Delete
//!
//! Platform support:
//! - Windows: uses SetConsoleMode via FFI
//! - Unix: uses termios via libc FFI

use std::io::{self, Read, Write};

/// A minimal line editor with history and raw terminal support.
pub struct LineEditor {
    history: Vec<String>,
    history_file: Option<String>,
    max_history: usize,
    use_color: bool,
}

/// Result of a single line read operation.
pub enum ReadResult {
    /// A complete line of input.
    Line(String),
    /// EOF (Ctrl+D or Ctrl+Z).
    Eof,
}

impl LineEditor {
    pub fn new() -> Self {
        Self::new_with_color(true)
    }

    pub fn new_with_color(use_color: bool) -> Self {
        let history_file = dirs_home().map(|h| {
            let mut p = h;
            p.push_str("/.cjc_history");
            p
        });

        let mut editor = Self {
            history: Vec::new(),
            history_file,
            max_history: 1000,
            use_color,
        };
        editor.load_history();
        editor
    }

    /// Read a single (possibly multi-line) input from the user.
    pub fn read_line(&mut self, prompt: &str) -> ReadResult {
        let mut full_input = String::new();
        let mut continuation = false;

        loop {
            let p = if continuation { "... " } else { prompt };
            match self.read_raw_line(p) {
                ReadResult::Eof => {
                    if full_input.is_empty() {
                        return ReadResult::Eof;
                    }
                    // Return what we have so far
                    return ReadResult::Line(full_input);
                }
                ReadResult::Line(line) => {
                    if continuation {
                        full_input.push('\n');
                    }
                    full_input.push_str(&line);

                    let trimmed = line.trim_end();
                    if trimmed.ends_with('{') || trimmed.ends_with('\\') {
                        continuation = true;
                        continue;
                    }

                    // Check for unbalanced braces
                    let open = full_input.chars().filter(|&c| c == '{').count();
                    let close = full_input.chars().filter(|&c| c == '}').count();
                    if open > close {
                        continuation = true;
                        continue;
                    }

                    break;
                }
            }
        }

        let trimmed = full_input.trim().to_string();
        if !trimmed.is_empty() {
            // Avoid consecutive duplicates
            if self.history.last().map(|h| h.as_str()) != Some(&trimmed) {
                self.history.push(trimmed.clone());
                if self.history.len() > self.max_history {
                    self.history.remove(0);
                }
            }
            self.save_history();
        }

        ReadResult::Line(full_input)
    }

    fn read_raw_line(&self, prompt: &str) -> ReadResult {
        // Try raw terminal mode; fall back to cooked mode
        if let Some(result) = self.try_raw_line(prompt) {
            return result;
        }
        self.fallback_line(prompt)
    }

    fn try_raw_line(&self, prompt: &str) -> Option<ReadResult> {
        let mut state = RawLineState::new(prompt, &self.history, self.use_color)?;
        let result = state.run();
        Some(result)
    }

    fn fallback_line(&self, prompt: &str) -> ReadResult {
        use std::io::BufRead;
        eprint!("{}", prompt);
        io::stderr().flush().ok();
        let mut line = String::new();
        match io::stdin().lock().read_line(&mut line) {
            Ok(0) => ReadResult::Eof,
            Ok(_) => ReadResult::Line(line.trim_end_matches('\n').trim_end_matches('\r').to_string()),
            Err(_) => ReadResult::Eof,
        }
    }

    fn load_history(&mut self) {
        if let Some(ref path) = self.history_file {
            if let Ok(contents) = std::fs::read_to_string(path) {
                self.history = contents
                    .lines()
                    .filter(|l| !l.is_empty())
                    .map(|l| l.to_string())
                    .collect();
                // Trim to max
                while self.history.len() > self.max_history {
                    self.history.remove(0);
                }
            }
        }
    }

    fn save_history(&self) {
        if let Some(ref path) = self.history_file {
            let content = self.history.join("\n");
            let _ = std::fs::write(path, content);
        }
    }
}

/// Get the user's home directory (zero-dep).
fn dirs_home() -> Option<String> {
    #[cfg(windows)]
    {
        std::env::var("USERPROFILE").ok()
    }
    #[cfg(not(windows))]
    {
        std::env::var("HOME").ok()
    }
}

// ── Raw terminal input state machine ────────────────────────────

struct RawLineState<'a> {
    buf: Vec<char>,
    cursor: usize,
    prompt: &'a str,
    history: &'a [String],
    history_idx: usize,     // points past end = current input
    saved_input: String,    // what user typed before navigating history
    use_color: bool,
    _guard: RawModeGuard,
}

impl<'a> RawLineState<'a> {
    fn new(prompt: &'a str, history: &'a [String], use_color: bool) -> Option<Self> {
        let guard = RawModeGuard::enter()?;
        Some(Self {
            buf: Vec::new(),
            cursor: 0,
            prompt,
            history,
            history_idx: history.len(),
            saved_input: String::new(),
            use_color,
            _guard: guard,
        })
    }

    fn run(&mut self) -> ReadResult {
        self.draw_line();

        loop {
            match read_key() {
                Some(Key::Char(c)) if c == '\r' || c == '\n' => {
                    eprint!("\r\n");
                    io::stderr().flush().ok();
                    let line: String = self.buf.iter().collect();
                    return ReadResult::Line(line);
                }
                Some(Key::Char('\x03')) => {
                    // Ctrl+C
                    eprint!("^C\r\n");
                    io::stderr().flush().ok();
                    return ReadResult::Line(String::new());
                }
                Some(Key::Char('\x04')) => {
                    // Ctrl+D
                    if self.buf.is_empty() {
                        eprint!("\r\n");
                        io::stderr().flush().ok();
                        return ReadResult::Eof;
                    }
                    // Delete char at cursor (like Delete key)
                    if self.cursor < self.buf.len() {
                        self.buf.remove(self.cursor);
                        self.draw_line();
                    }
                }
                Some(Key::Char('\x7f')) | Some(Key::Backspace) => {
                    // Backspace
                    if self.cursor > 0 {
                        self.cursor -= 1;
                        self.buf.remove(self.cursor);
                        self.draw_line();
                    }
                }
                Some(Key::Char('\x15')) => {
                    // Ctrl+U: clear line
                    self.buf.clear();
                    self.cursor = 0;
                    self.draw_line();
                }
                Some(Key::Char('\x0b')) => {
                    // Ctrl+K: kill to end of line
                    self.buf.truncate(self.cursor);
                    self.draw_line();
                }
                Some(Key::Char('\x01')) => {
                    // Ctrl+A: go to start
                    self.cursor = 0;
                    self.draw_line();
                }
                Some(Key::Char('\x05')) => {
                    // Ctrl+E: go to end
                    self.cursor = self.buf.len();
                    self.draw_line();
                }
                Some(Key::Char('\x0c')) => {
                    // Ctrl+L: clear screen
                    eprint!("\x1b[2J\x1b[H");
                    self.draw_line();
                }
                Some(Key::Char(c)) if c >= ' ' => {
                    self.buf.insert(self.cursor, c);
                    self.cursor += 1;
                    self.draw_line();
                }
                Some(Key::Left) => {
                    if self.cursor > 0 {
                        self.cursor -= 1;
                        self.draw_line();
                    }
                }
                Some(Key::Right) => {
                    if self.cursor < self.buf.len() {
                        self.cursor += 1;
                        self.draw_line();
                    }
                }
                Some(Key::Up) => {
                    if self.history_idx > 0 {
                        if self.history_idx == self.history.len() {
                            self.saved_input = self.buf.iter().collect();
                        }
                        self.history_idx -= 1;
                        let entry = &self.history[self.history_idx];
                        self.buf = entry.chars().collect();
                        self.cursor = self.buf.len();
                        self.draw_line();
                    }
                }
                Some(Key::Down) => {
                    if self.history_idx < self.history.len() {
                        self.history_idx += 1;
                        if self.history_idx == self.history.len() {
                            self.buf = self.saved_input.chars().collect();
                        } else {
                            let entry = &self.history[self.history_idx];
                            self.buf = entry.chars().collect();
                        }
                        self.cursor = self.buf.len();
                        self.draw_line();
                    }
                }
                Some(Key::Home) => {
                    self.cursor = 0;
                    self.draw_line();
                }
                Some(Key::End) => {
                    self.cursor = self.buf.len();
                    self.draw_line();
                }
                Some(Key::Delete) => {
                    if self.cursor < self.buf.len() {
                        self.buf.remove(self.cursor);
                        self.draw_line();
                    }
                }
                _ => {}
            }
        }
    }

    fn draw_line(&self) {
        let line: String = self.buf.iter().collect();
        // Move to start of line, clear it, redraw with optional syntax highlighting
        let display = if self.use_color {
            crate::highlight::highlight(&line)
        } else {
            line
        };
        eprint!("\r\x1b[K{}{}", self.prompt, display);
        // Position cursor (use plain char count, not ANSI-inflated length)
        let cursor_pos = self.prompt.len() + self.cursor;
        eprint!("\r\x1b[{}C", cursor_pos);
        io::stderr().flush().ok();
    }
}

// ── Key reading ────────────────────────────────────────────────

enum Key {
    Char(char),
    Left,
    Right,
    Up,
    Down,
    Home,
    End,
    Delete,
    Backspace,
}

#[cfg(windows)]
fn read_key() -> Option<Key> {
    // On Windows, read from stdin in raw mode
    let mut buf = [0u8; 1];
    if io::stdin().read_exact(&mut buf).is_err() {
        return None;
    }
    match buf[0] {
        0x08 => Some(Key::Backspace), // Backspace on Windows
        0x1b => {
            // Escape sequence
            let mut seq = [0u8; 1];
            if io::stdin().read_exact(&mut seq).is_err() {
                return Some(Key::Char('\x1b'));
            }
            if seq[0] == b'[' {
                let mut code = [0u8; 1];
                if io::stdin().read_exact(&mut code).is_err() {
                    return None;
                }
                match code[0] {
                    b'A' => Some(Key::Up),
                    b'B' => Some(Key::Down),
                    b'C' => Some(Key::Right),
                    b'D' => Some(Key::Left),
                    b'H' => Some(Key::Home),
                    b'F' => Some(Key::End),
                    b'3' => {
                        // Delete key: ESC [ 3 ~
                        let mut tilde = [0u8; 1];
                        let _ = io::stdin().read_exact(&mut tilde);
                        Some(Key::Delete)
                    }
                    _ => None,
                }
            } else if seq[0] == b'O' {
                let mut code = [0u8; 1];
                if io::stdin().read_exact(&mut code).is_err() {
                    return None;
                }
                match code[0] {
                    b'H' => Some(Key::Home),
                    b'F' => Some(Key::End),
                    _ => None,
                }
            } else {
                None
            }
        }
        b if b < 0x20 => Some(Key::Char(b as char)),
        b => {
            // Handle UTF-8 multi-byte
            let ch = if b < 0x80 {
                b as char
            } else {
                // Read remaining UTF-8 bytes
                let extra = if b & 0xE0 == 0xC0 { 1 }
                    else if b & 0xF0 == 0xE0 { 2 }
                    else if b & 0xF8 == 0xF0 { 3 }
                    else { 0 };
                let mut utf8_buf = vec![b];
                for _ in 0..extra {
                    let mut next = [0u8; 1];
                    if io::stdin().read_exact(&mut next).is_err() {
                        return None;
                    }
                    utf8_buf.push(next[0]);
                }
                std::str::from_utf8(&utf8_buf)
                    .ok()
                    .and_then(|s| s.chars().next())
                    .unwrap_or('?')
            };
            Some(Key::Char(ch))
        }
    }
}

#[cfg(not(windows))]
fn read_key() -> Option<Key> {
    let mut buf = [0u8; 1];
    if io::stdin().read_exact(&mut buf).is_err() {
        return None;
    }
    match buf[0] {
        0x7f => Some(Key::Backspace),
        0x1b => {
            let mut seq = [0u8; 1];
            if io::stdin().read_exact(&mut seq).is_err() {
                return Some(Key::Char('\x1b'));
            }
            if seq[0] == b'[' {
                let mut code = [0u8; 1];
                if io::stdin().read_exact(&mut code).is_err() {
                    return None;
                }
                match code[0] {
                    b'A' => Some(Key::Up),
                    b'B' => Some(Key::Down),
                    b'C' => Some(Key::Right),
                    b'D' => Some(Key::Left),
                    b'H' => Some(Key::Home),
                    b'F' => Some(Key::End),
                    b'3' => {
                        let mut tilde = [0u8; 1];
                        let _ = io::stdin().read_exact(&mut tilde);
                        Some(Key::Delete)
                    }
                    _ => None,
                }
            } else if seq[0] == b'O' {
                let mut code = [0u8; 1];
                if io::stdin().read_exact(&mut code).is_err() {
                    return None;
                }
                match code[0] {
                    b'H' => Some(Key::Home),
                    b'F' => Some(Key::End),
                    _ => None,
                }
            } else {
                None
            }
        }
        b if b < 0x20 => Some(Key::Char(b as char)),
        b => {
            let ch = if b < 0x80 {
                b as char
            } else {
                let extra = if b & 0xE0 == 0xC0 { 1 }
                    else if b & 0xF0 == 0xE0 { 2 }
                    else if b & 0xF8 == 0xF0 { 3 }
                    else { 0 };
                let mut utf8_buf = vec![b];
                for _ in 0..extra {
                    let mut next = [0u8; 1];
                    if io::stdin().read_exact(&mut next).is_err() {
                        return None;
                    }
                    utf8_buf.push(next[0]);
                }
                std::str::from_utf8(&utf8_buf)
                    .ok()
                    .and_then(|s| s.chars().next())
                    .unwrap_or('?')
            };
            Some(Key::Char(ch))
        }
    }
}

// ── Raw terminal mode (platform-specific) ───────────────────────

struct RawModeGuard {
    #[cfg(windows)]
    original_mode: u32,
    #[cfg(not(windows))]
    original_termios: [u8; 60], // termios struct bytes
}

#[cfg(windows)]
mod platform {
    // Windows console API via FFI
    #[link(name = "kernel32")]
    extern "system" {
        fn GetConsoleMode(handle: *mut core::ffi::c_void, mode: *mut u32) -> i32;
        fn SetConsoleMode(handle: *mut core::ffi::c_void, mode: u32) -> i32;
        fn GetStdHandle(std_handle: u32) -> *mut core::ffi::c_void;
    }

    const STD_INPUT_HANDLE: u32 = 0xFFFF_FFF6;
    const ENABLE_ECHO_INPUT: u32 = 0x0004;
    const ENABLE_LINE_INPUT: u32 = 0x0002;
    const ENABLE_PROCESSED_INPUT: u32 = 0x0001;
    const ENABLE_VIRTUAL_TERMINAL_INPUT: u32 = 0x0200;

    const STD_ERROR_HANDLE: u32 = 0xFFFF_FFF4;
    const ENABLE_VIRTUAL_TERMINAL_PROCESSING: u32 = 0x0004;

    pub fn enter_raw_mode() -> Option<u32> {
        unsafe {
            let handle = GetStdHandle(STD_INPUT_HANDLE);
            if handle.is_null() {
                return None;
            }
            let mut original_mode: u32 = 0;
            if GetConsoleMode(handle, &mut original_mode) == 0 {
                return None;
            }

            // Disable line mode and echo, enable VT input
            let raw_mode = (original_mode
                & !(ENABLE_ECHO_INPUT | ENABLE_LINE_INPUT | ENABLE_PROCESSED_INPUT))
                | ENABLE_VIRTUAL_TERMINAL_INPUT;

            if SetConsoleMode(handle, raw_mode) == 0 {
                return None;
            }

            // Also enable VT processing on stderr for ANSI escape codes
            let err_handle = GetStdHandle(STD_ERROR_HANDLE);
            if !err_handle.is_null() {
                let mut err_mode: u32 = 0;
                if GetConsoleMode(err_handle, &mut err_mode) != 0 {
                    let _ = SetConsoleMode(err_handle, err_mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
                }
            }

            Some(original_mode)
        }
    }

    pub fn restore_mode(original_mode: u32) {
        unsafe {
            let handle = GetStdHandle(STD_INPUT_HANDLE);
            if !handle.is_null() {
                SetConsoleMode(handle, original_mode);
            }
        }
    }
}

#[cfg(not(windows))]
mod platform {
    use std::io;

    // Unix termios via libc FFI
    extern "C" {
        fn tcgetattr(fd: i32, termios: *mut u8) -> i32;
        fn tcsetattr(fd: i32, action: i32, termios: *const u8) -> i32;
    }

    const STDIN_FD: i32 = 0;
    const TCSAFLUSH: i32 = 2;

    // termios struct offsets (Linux x86_64)
    // c_iflag at offset 0 (4 bytes)
    // c_oflag at offset 4 (4 bytes)
    // c_cflag at offset 8 (4 bytes)
    // c_lflag at offset 12 (4 bytes)
    const LFLAG_OFFSET: usize = 12;
    const ICANON: u32 = 0x0000_0002;
    const ECHO: u32 = 0x0000_0008;
    const ISIG: u32 = 0x0000_0001;

    pub fn enter_raw_mode() -> Option<[u8; 60]> {
        let mut original = [0u8; 60];
        unsafe {
            if tcgetattr(STDIN_FD, original.as_mut_ptr()) != 0 {
                return None;
            }
        }

        let mut raw = original;
        // Clear ICANON, ECHO, ISIG in c_lflag
        let lflag = u32::from_ne_bytes([
            raw[LFLAG_OFFSET],
            raw[LFLAG_OFFSET + 1],
            raw[LFLAG_OFFSET + 2],
            raw[LFLAG_OFFSET + 3],
        ]);
        let new_lflag = lflag & !(ICANON | ECHO | ISIG);
        let bytes = new_lflag.to_ne_bytes();
        raw[LFLAG_OFFSET] = bytes[0];
        raw[LFLAG_OFFSET + 1] = bytes[1];
        raw[LFLAG_OFFSET + 2] = bytes[2];
        raw[LFLAG_OFFSET + 3] = bytes[3];

        unsafe {
            if tcsetattr(STDIN_FD, TCSAFLUSH, raw.as_ptr()) != 0 {
                return None;
            }
        }

        Some(original)
    }

    pub fn restore_mode(original: [u8; 60]) {
        unsafe {
            tcsetattr(STDIN_FD, TCSAFLUSH, original.as_ptr());
        }
    }
}

impl RawModeGuard {
    fn enter() -> Option<Self> {
        #[cfg(windows)]
        {
            let original_mode = platform::enter_raw_mode()?;
            Some(Self { original_mode })
        }
        #[cfg(not(windows))]
        {
            let original_termios = platform::enter_raw_mode()?;
            Some(Self { original_termios })
        }
    }
}

impl Drop for RawModeGuard {
    fn drop(&mut self) {
        #[cfg(windows)]
        {
            platform::restore_mode(self.original_mode);
        }
        #[cfg(not(windows))]
        {
            platform::restore_mode(self.original_termios);
        }
    }
}
