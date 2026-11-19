//! Postulate Phase 2 — the companion channel.
//!
//! An editor talks to *its* `cjc-analyzer`; the Postulate HUD is a separate
//! window with its own analyzer. Started as `cjc-analyzer --companion PATH`,
//! the editor's server mirrors what the editor has open into two small files
//! the HUD watches:
//!
//! - `PATH.doc`: `{"uri","version","text"}` — written on `didOpen` /
//!   `didChange`, so the HUD analyzes the editor's buffer, saved or not.
//! - `PATH`: `{"seq","uri","version","line","character"}` — the last cursor
//!   position the editor reported (`hover`, `documentHighlight`,
//!   `completion`, all carrying a position). Editors send `documentHighlight`
//!   on every cursor move once the server advertises it.
//!
//! Both are replaced atomically (write a temp file, then rename), the doc
//! before the cursor, so a reader that sees cursor version N finds doc N.
//! The channel only observes messages; it never changes a response.

use std::collections::BTreeMap;
use std::io;
use std::path::{Path, PathBuf};

use lsp_server::Message;
use serde_json::{json, Value as Json};

pub struct Companion {
    path: PathBuf,
    seq: u64,
    versions: BTreeMap<String, i64>,
}

fn write_atomic(path: &Path, body: &str) -> io::Result<()> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, body)?;
    std::fs::rename(&tmp, path)
}

impl Companion {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into(), seq: 0, versions: BTreeMap::new() }
    }

    pub fn doc_path(path: &Path) -> PathBuf {
        let mut p = path.as_os_str().to_owned();
        p.push(".doc");
        PathBuf::from(p)
    }

    /// Mirror one incoming message. I/O errors are returned to the caller,
    /// which logs and carries on: the companion must never stop the server.
    pub fn observe(&mut self, msg: &Message) -> io::Result<()> {
        let (method, params) = match msg {
            Message::Request(r) => (r.method.as_str(), &r.params),
            Message::Notification(n) => (n.method.as_str(), &n.params),
            Message::Response(_) => return Ok(()),
        };
        match method {
            "textDocument/didOpen" | "textDocument/didChange" => {
                let uri = params.pointer("/textDocument/uri").and_then(Json::as_str);
                let version = params.pointer("/textDocument/version").and_then(Json::as_i64);
                let text = if method == "textDocument/didOpen" {
                    params.pointer("/textDocument/text").and_then(Json::as_str)
                } else {
                    params
                        .get("contentChanges")
                        .and_then(Json::as_array)
                        .and_then(|c| c.last())
                        .and_then(|c| c.get("text"))
                        .and_then(Json::as_str)
                };
                if let (Some(uri), Some(version), Some(text)) = (uri, version, text) {
                    self.versions.insert(uri.to_owned(), version);
                    let doc = json!({ "uri": uri, "version": version, "text": text });
                    write_atomic(&Self::doc_path(&self.path), &doc.to_string())?;
                }
            }
            "textDocument/hover" | "textDocument/documentHighlight" | "textDocument/completion" => {
                let uri = params.pointer("/textDocument/uri").and_then(Json::as_str);
                let line = params.pointer("/position/line").and_then(Json::as_u64);
                let character = params.pointer("/position/character").and_then(Json::as_u64);
                if let (Some(uri), Some(line), Some(character)) = (uri, line, character) {
                    self.seq += 1;
                    let state = json!({
                        "seq": self.seq,
                        "uri": uri,
                        "version": self.versions.get(uri),
                        "line": line,
                        "character": character,
                    });
                    write_atomic(&self.path, &state.to_string())?;
                }
            }
            _ => {}
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lsp_server::{Notification, Request, RequestId};

    #[test]
    fn mirrors_the_buffer_and_the_cursor() {
        let dir = std::env::temp_dir().join(format!("cjc-companion-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("state.json");
        let mut c = Companion::new(&path);
        let uri = "file:///a.cjcl";
        c.observe(&Message::Notification(Notification::new(
            "textDocument/didOpen".into(),
            json!({ "textDocument": { "uri": uri, "languageId": "cjc", "version": 3, "text": "let x = 1;" } }),
        )))
        .unwrap();
        c.observe(&Message::Request(Request::new(
            RequestId::from(1),
            "textDocument/documentHighlight".into(),
            json!({ "textDocument": { "uri": uri }, "position": { "line": 0, "character": 4 } }),
        )))
        .unwrap();
        let doc: Json = serde_json::from_str(&std::fs::read_to_string(Companion::doc_path(&path)).unwrap()).unwrap();
        let cur: Json = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(doc, json!({ "uri": uri, "version": 3, "text": "let x = 1;" }));
        assert_eq!(cur, json!({ "seq": 1, "uri": uri, "version": 3, "line": 0, "character": 4 }));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
