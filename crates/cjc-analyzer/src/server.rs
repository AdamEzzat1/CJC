//! LSP server — the main event loop for the CJC language server.
//!
//! Uses `lsp-server` for JSON-RPC transport over stdin/stdout.
//! Supports:
//! - textDocument/didOpen, didChange, didClose
//! - textDocument/hover
//! - textDocument/completion
//! - textDocument/publishDiagnostics (on edit)

use std::collections::BTreeMap;

use lsp_server::{Connection, Message, Notification, Request, RequestId, Response};
use lsp_types::notification::{DidChangeTextDocument, DidCloseTextDocument, DidOpenTextDocument, Notification as _};
use lsp_types::request::{Completion, HoverRequest, Request as _};
use lsp_types::{
    CompletionOptions, CompletionParams, DidChangeTextDocumentParams,
    DidCloseTextDocumentParams, DidOpenTextDocumentParams, Hover, HoverContents,
    HoverParams, HoverProviderCapability, InitializeParams, MarkupContent, MarkupKind,
    PublishDiagnosticsParams, ServerCapabilities, TextDocumentSyncCapability,
    TextDocumentSyncKind, Uri,
};

use crate::completion;
use crate::diagnostics;
use crate::hover::{hover_for_symbol, word_at_offset};
use crate::symbol_index::SymbolIndex;

/// Open document state.
struct DocumentState {
    text: String,
    #[allow(dead_code)]
    version: i32,
}

/// Run the LSP server. Blocks until the client disconnects.
pub fn run_server() -> Result<(), Box<dyn std::error::Error>> {
    let (connection, io_threads) = Connection::stdio();

    let capabilities = ServerCapabilities {
        text_document_sync: Some(TextDocumentSyncCapability::Kind(
            TextDocumentSyncKind::FULL,
        )),
        hover_provider: Some(HoverProviderCapability::Simple(true)),
        completion_provider: Some(CompletionOptions {
            trigger_characters: Some(vec![".".to_string()]),
            ..Default::default()
        }),
        ..Default::default()
    };

    let caps_json = serde_json::to_value(capabilities)?;
    let _init_params: InitializeParams = serde_json::from_value(
        connection.initialize(caps_json)?,
    )?;

    main_loop(&connection)?;
    io_threads.join()?;
    Ok(())
}

/// The main message loop.
fn main_loop(conn: &Connection) -> Result<(), Box<dyn std::error::Error>> {
    let mut documents: BTreeMap<Uri, DocumentState> = BTreeMap::new();
    let mut index = SymbolIndex::new();
    index.populate_builtins();

    for msg in &conn.receiver {
        match msg {
            Message::Request(req) => {
                if conn.handle_shutdown(&req)? {
                    return Ok(());
                }
                handle_request(req, conn, &documents, &index)?;
            }
            Message::Notification(notif) => {
                handle_notification(notif, conn, &mut documents, &mut index)?;
            }
            Message::Response(_) => {}
        }
    }
    Ok(())
}

/// Handle incoming requests (hover, completion).
fn handle_request(
    req: Request,
    conn: &Connection,
    documents: &BTreeMap<Uri, DocumentState>,
    index: &SymbolIndex,
) -> Result<(), Box<dyn std::error::Error>> {
    match req.method.as_str() {
        HoverRequest::METHOD => {
            let (id, params): (RequestId, HoverParams) = extract_request(req)?;
            let uri = &params.text_document_position_params.text_document.uri;
            let pos = params.text_document_position_params.position;

            let result = documents.get(uri).and_then(|doc| {
                let line = doc.text.lines().nth(pos.line as usize)?;
                let word = word_at_offset(line, pos.character as usize)?;
                let hover_info = hover_for_symbol(index, word)?;
                Some(Hover {
                    contents: HoverContents::Markup(MarkupContent {
                        kind: MarkupKind::Markdown,
                        value: hover_info.contents,
                    }),
                    range: None,
                })
            });

            let resp = Response::new_ok(id, result);
            conn.sender.send(Message::Response(resp))?;
        }
        Completion::METHOD => {
            let (id, params): (RequestId, CompletionParams) = extract_request(req)?;
            let uri = &params.text_document_position.text_document.uri;
            let pos = params.text_document_position.position;

            let items = documents
                .get(uri)
                .map(|doc| {
                    let line = doc.text.lines().nth(pos.line as usize).unwrap_or("");
                    // Get prefix up to cursor
                    let col = pos.character as usize;
                    let prefix_start = line[..col]
                        .rfind(|c: char| !c.is_alphanumeric() && c != '_')
                        .map(|i| i + 1)
                        .unwrap_or(0);
                    let prefix = &line[prefix_start..col];
                    completion::complete(index, prefix)
                })
                .unwrap_or_default();

            let resp = Response::new_ok(id, items);
            conn.sender.send(Message::Response(resp))?;
        }
        _ => {
            // Unknown request — respond with method not found
            let resp = Response::new_err(
                req.id,
                lsp_server::ErrorCode::MethodNotFound as i32,
                format!("Unknown method: {}", req.method),
            );
            conn.sender.send(Message::Response(resp))?;
        }
    }
    Ok(())
}

/// Handle incoming notifications (didOpen, didChange, didClose).
fn handle_notification(
    notif: Notification,
    conn: &Connection,
    documents: &mut BTreeMap<Uri, DocumentState>,
    index: &mut SymbolIndex,
) -> Result<(), Box<dyn std::error::Error>> {
    match notif.method.as_str() {
        DidOpenTextDocument::METHOD => {
            let params: DidOpenTextDocumentParams = serde_json::from_value(notif.params)?;
            let uri = params.text_document.uri.clone();
            let text = params.text_document.text.clone();
            let version = params.text_document.version;

            // Check for vizor import and update index
            update_index_for_imports(index, &text);

            // Publish diagnostics
            publish_diagnostics(conn, &uri, &text)?;

            documents.insert(
                uri,
                DocumentState { text, version },
            );
        }
        DidChangeTextDocument::METHOD => {
            let params: DidChangeTextDocumentParams = serde_json::from_value(notif.params)?;
            let uri = params.text_document.uri.clone();

            if let Some(change) = params.content_changes.into_iter().last() {
                let text = change.text;
                let version = params.text_document.version;

                update_index_for_imports(index, &text);
                publish_diagnostics(conn, &uri, &text)?;

                documents.insert(
                    uri,
                    DocumentState { text, version },
                );
            }
        }
        DidCloseTextDocument::METHOD => {
            let params: DidCloseTextDocumentParams = serde_json::from_value(notif.params)?;
            documents.remove(&params.text_document.uri);

            // Clear diagnostics for closed document
            let diag_params = PublishDiagnosticsParams {
                uri: params.text_document.uri,
                diagnostics: vec![],
                version: None,
            };
            let notif = lsp_server::Notification::new(
                "textDocument/publishDiagnostics".to_string(),
                diag_params,
            );
            conn.sender.send(Message::Notification(notif))?;
        }
        _ => {}
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

/// Parse the source and publish diagnostics to the client.
fn publish_diagnostics(
    conn: &Connection,
    uri: &Uri,
    source: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let (_program, diags) = cjc_parser::parse_source(source);
    let lsp_diags = diagnostics::diagnostics_from_parse(source, &diags);

    let params = PublishDiagnosticsParams {
        uri: uri.clone(),
        diagnostics: lsp_diags,
        version: None,
    };

    let notif = lsp_server::Notification::new(
        "textDocument/publishDiagnostics".to_string(),
        params,
    );
    conn.sender.send(Message::Notification(notif))?;
    Ok(())
}

/// Extract a typed request from a raw LSP Request.
fn extract_request<P: serde::de::DeserializeOwned>(
    req: Request,
) -> Result<(RequestId, P), Box<dyn std::error::Error>> {
    let params: P = serde_json::from_value(req.params)?;
    Ok((req.id, params))
}
