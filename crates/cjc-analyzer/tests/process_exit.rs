//! The server process exits when its client goes away, with or without a
//! `shutdown` request. Before the fix, `run_server` joined the I/O threads
//! while still holding the connection, so the writer thread (and the
//! process) lived forever after the client died: orphaned servers, and any
//! parent waiting on inherited pipes hung with them.

use std::io::{Read, Write};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

fn frame(body: &str) -> Vec<u8> {
    format!("Content-Length: {}\r\n\r\n{body}", body.len()).into_bytes()
}

fn exits_within(mut child: std::process::Child, limit: Duration) -> bool {
    let t = Instant::now();
    while t.elapsed() < limit {
        if child.try_wait().unwrap().is_some() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    let _ = child.kill();
    false
}

#[test]
fn server_exits_when_the_client_closes_stdin_without_shutdown() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_cjc-analyzer"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    stdin.write_all(&frame(r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"capabilities":{}}}"#)).unwrap();
    stdin.write_all(&frame(r#"{"jsonrpc":"2.0","method":"initialized","params":{}}"#)).unwrap();
    stdin.flush().unwrap();
    // Read the initialize response so the server is in its main loop.
    let mut stdout = child.stdout.take().unwrap();
    let mut buf = [0u8; 64];
    let n = stdout.read(&mut buf).unwrap();
    assert!(n > 0 && buf.starts_with(b"Content-Length"), "no initialize response");
    // The client dies: stdin closes, no shutdown request.
    drop(stdin);
    assert!(exits_within(child, Duration::from_secs(10)), "cjc-analyzer outlived its client");
}
