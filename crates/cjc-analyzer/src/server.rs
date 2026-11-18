//! LSP server — the CJC language server.
//!
//! [`Analyzer`] is the whole server as a value: messages in, messages out, no
//! I/O. [`run_server`] is a thin stdio loop around it, and tests drive it
//! in-process (`tests/replay.rs`), which is what makes a scripted session
//! replayable.
//!
//! Supports:
//! - textDocument/didOpen, didChange, didClose
//! - textDocument/hover
//! - textDocument/completion
//! - textDocument/publishDiagnostics (on edit)
//! - textDocument/inlayHint (tensor shapes, Postulate)
//! - postulate/gradGraph, postulate/snapshotHash, postulate/executorAgreement,
//!   postulate/runReport (see [`crate::postulate`])

use std::collections::BTreeMap;
use std::rc::Rc;

use lsp_server::{Connection, Message, Notification, Request, RequestId, Response};
use lsp_types::notification::{DidChangeTextDocument, DidCloseTextDocument, DidOpenTextDocument, Notification as _};
use lsp_types::request::{Completion, HoverRequest, InlayHintRequest, Request as _};
use lsp_types::{
    CompletionOptions, CompletionParams, DidChangeTextDocumentParams,
    DidCloseTextDocumentParams, DidOpenTextDocumentParams, Hover, HoverContents,
    HoverParams, HoverProviderCapability, InitializeParams, InlayHintParams, MarkupContent,
    MarkupKind, OneOf, PublishDiagnosticsParams, ServerCapabilities, TextDocumentSyncCapability,
    TextDocumentSyncKind, Uri,
};
use serde_json::Value as Json;

use crate::completion;
use crate::diagnostics;
use crate::hover::{hover_for_symbol, word_at_offset};
use crate::postulate::{self, RunCapture};
use crate::symbol_index::SymbolIndex;

type BoxError = Box<dyn std::error::Error>;

/// Open document state.
struct DocumentState {
    text: String,
    version: i32,
}

/// The language server, without transport.
pub struct Analyzer {
    documents: BTreeMap<Uri, DocumentState>,
    index: SymbolIndex,
    /// One run per (uri, seed), valid for the document version it records.
    runs: BTreeMap<(String, u64), (i32, Rc<RunCapture>)>,
}

impl Default for Analyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl Analyzer {
    pub fn new() -> Self {
        let mut index = SymbolIndex::new();
        index.populate_builtins();
        Self { documents: BTreeMap::new(), index, runs: BTreeMap::new() }
    }

    pub fn capabilities() -> ServerCapabilities {
        ServerCapabilities {
            text_document_sync: Some(TextDocumentSyncCapability::Kind(TextDocumentSyncKind::FULL)),
            hover_provider: Some(HoverProviderCapability::Simple(true)),
            completion_provider: Some(CompletionOptions {
                trigger_characters: Some(vec![".".to_string()]),
                ..Default::default()
            }),
            inlay_hint_provider: Some(OneOf::Left(true)),
            ..Default::default()
        }
    }

    /// Handle one message; return what the server sends back, in order.
    pub fn handle(&mut self, msg: Message) -> Vec<Message> {
        match msg {
            Message::Request(req) => {
                let id = req.id.clone();
                let method = req.method.clone();
                match self.handle_request(req) {
                    Ok(Some(r)) => vec![Message::Response(r)],
                    Ok(None) => vec![Message::Response(Response::new_err(
                        id,
                        lsp_server::ErrorCode::MethodNotFound as i32,
                        format!("Unknown method: {method}"),
                    ))],
                    Err(e) => vec![Message::Response(Response::new_err(
                        id,
                        lsp_server::ErrorCode::InvalidParams as i32,
                        e.to_string(),
                    ))],
                }
            }
            Message::Notification(n) => self.handle_notification(n).unwrap_or_default(),
            Message::Response(_) => Vec::new(),
        }
    }

    fn handle_request(&mut self, req: Request) -> Result<Option<Response>, BoxError> {
        Ok(Some(match req.method.as_str() {
            HoverRequest::METHOD => {
                let (id, params): (RequestId, HoverParams) = extract_request(req)?;
                let uri = &params.text_document_position_params.text_document.uri;
                let pos = params.text_document_position_params.position;
                let result = self.documents.get(uri).and_then(|doc| {
                    let line = doc.text.lines().nth(pos.line as usize)?;
                    let word = word_at_offset(line, pos.character as usize)?;
                    let hover_info = hover_for_symbol(&self.index, word)?;
                    Some(Hover {
                        contents: HoverContents::Markup(MarkupContent {
                            kind: MarkupKind::Markdown,
                            value: hover_info.contents,
                        }),
                        range: None,
                    })
                });
                Response::new_ok(id, result)
            }
            Completion::METHOD => {
                let (id, params): (RequestId, CompletionParams) = extract_request(req)?;
                let uri = &params.text_document_position.text_document.uri;
                let pos = params.text_document_position.position;
                let items = self
                    .documents
                    .get(uri)
                    .map(|doc| {
                        let line = doc.text.lines().nth(pos.line as usize).unwrap_or("");
                        let col = (pos.character as usize).min(line.len());
                        let prefix_start = line[..col]
                            .rfind(|c: char| !c.is_alphanumeric() && c != '_')
                            .map(|i| i + 1)
                            .unwrap_or(0);
                        completion::complete(&self.index, &line[prefix_start..col])
                    })
                    .unwrap_or_default();
                Response::new_ok(id, items)
            }
            InlayHintRequest::METHOD => {
                let (id, params): (RequestId, InlayHintParams) = extract_request(req)?;
                let uri = params.text_document.uri;
                let hints = match self.run(&uri, postulate::DEFAULT_SEED) {
                    Some((src, run)) => postulate::shape_hints(&src, &run, params.range),
                    None => Vec::new(),
                };
                Response::new_ok(id, hints)
            }
            "postulate/gradGraph"
            | "postulate/snapshotHash"
            | "postulate/executorAgreement"
            | "postulate/runReport" => {
                let (uri, seed) = postulate_params(&req.params)?;
                let Some((src, run)) = self.run(&uri, seed) else {
                    return Err(format!("document not open: {}", uri.as_str()).into());
                };
                let body = match req.method.as_str() {
                    "postulate/gradGraph" => postulate::grad_graph_response(&src, &run),
                    "postulate/snapshotHash" => postulate::snapshot_hash_response(&run),
                    "postulate/executorAgreement" => postulate::executor_agreement_response(&run),
                    _ => postulate::run_report_response(&src, &run),
                };
                Response::new_ok(req.id, body)
            }
            _ => return Ok(None),
        }))
    }

    /// The document's text and its run at `seed`, running it if this version
    /// has not been run yet.
    fn run(&mut self, uri: &Uri, seed: u64) -> Option<(String, Rc<RunCapture>)> {
        let doc = self.documents.get(uri)?;
        let key = (uri.as_str().to_owned(), seed);
        if let Some((v, run)) = self.runs.get(&key) {
            if *v == doc.version {
                return Some((doc.text.clone(), run.clone()));
            }
        }
        let run = postulate::capture(&doc.text, seed);
        self.runs.insert(key, (doc.version, run.clone()));
        Some((doc.text.clone(), run))
    }

    fn forget_runs(&mut self, uri: &Uri) {
        self.runs.retain(|(u, _), _| u != uri.as_str());
    }

    fn handle_notification(&mut self, notif: Notification) -> Result<Vec<Message>, BoxError> {
        let mut out = Vec::new();
        match notif.method.as_str() {
            DidOpenTextDocument::METHOD => {
                let params: DidOpenTextDocumentParams = serde_json::from_value(notif.params)?;
                let uri = params.text_document.uri;
                let text = params.text_document.text;
                update_index_for_imports(&mut self.index, &text);
                out.push(diagnostics_notification(&uri, &text));
                self.forget_runs(&uri);
                self.documents.insert(uri, DocumentState { text, version: params.text_document.version });
            }
            DidChangeTextDocument::METHOD => {
                let params: DidChangeTextDocumentParams = serde_json::from_value(notif.params)?;
                let uri = params.text_document.uri;
                if let Some(change) = params.content_changes.into_iter().last() {
                    let text = change.text;
                    update_index_for_imports(&mut self.index, &text);
                    out.push(diagnostics_notification(&uri, &text));
                    self.forget_runs(&uri);
                    self.documents
                        .insert(uri, DocumentState { text, version: params.text_document.version });
                }
            }
            DidCloseTextDocument::METHOD => {
                let params: DidCloseTextDocumentParams = serde_json::from_value(notif.params)?;
                let uri = params.text_document.uri;
                self.documents.remove(&uri);
                self.forget_runs(&uri);
                // Clear diagnostics for closed document
                out.push(Message::Notification(Notification::new(
                    "textDocument/publishDiagnostics".to_string(),
                    PublishDiagnosticsParams { uri, diagnostics: vec![], version: None },
                )));
            }
            _ => {}
        }
        Ok(out)
    }
}

/// `{ "textDocument": { "uri" }, "seed"?: u64 }`
fn postulate_params(params: &Json) -> Result<(Uri, u64), BoxError> {
    let uri = params
        .pointer("/textDocument/uri")
        .and_then(Json::as_str)
        .ok_or("missing textDocument.uri")?;
    let uri: Uri = uri.parse().map_err(|e| format!("bad uri: {e:?}"))?;
    let seed = match params.get("seed") {
        None | Some(Json::Null) => postulate::DEFAULT_SEED,
        Some(s) => s.as_u64().ok_or("seed must be a non-negative integer")?,
    };
    Ok((uri, seed))
}

/// Run the LSP server. Blocks until the client disconnects.
pub fn run_server() -> Result<(), BoxError> {
    let (connection, io_threads) = Connection::stdio();

    let caps_json = serde_json::to_value(Analyzer::capabilities())?;
    let _init_params: InitializeParams = serde_json::from_value(connection.initialize(caps_json)?)?;

    main_loop(&connection)?;
    io_threads.join()?;
    Ok(())
}

/// The main message loop.
fn main_loop(conn: &Connection) -> Result<(), BoxError> {
    let mut analyzer = Analyzer::new();
    for msg in &conn.receiver {
        if let Message::Request(req) = &msg {
            if conn.handle_shutdown(req)? {
                return Ok(());
            }
        }
        for out in analyzer.handle(msg) {
            conn.sender.send(out)?;
        }
    }
    Ok(())
}

/// Scan source for `import vizor` and populate the index accordingly.
fn update_index_for_imports(index: &mut SymbolIndex, source: &str) {
    // Simple text scan for `import vizor` — no need for full parse
    let has_vizor = source.lines().any(|line| {
        let trimmed = line.trim();
        trimmed == "import vizor" || trimmed.starts_with("import vizor ")
    });

    if has_vizor {
        index.populate_vizor();
    }
}

/// Parse the source and build the diagnostics notification.
fn diagnostics_notification(uri: &Uri, source: &str) -> Message {
    let (_program, diags) = cjc_parser::parse_source(source);
    let params = PublishDiagnosticsParams {
        uri: uri.clone(),
        diagnostics: diagnostics::diagnostics_from_parse(source, &diags),
        version: None,
    };
    Message::Notification(Notification::new("textDocument/publishDiagnostics".to_string(), params))
}

/// Extract a typed request from a raw LSP Request.
fn extract_request<P: serde::de::DeserializeOwned>(req: Request) -> Result<(RequestId, P), BoxError> {
    let params: P = serde_json::from_value(req.params)?;
    Ok((req.id, params))
}
