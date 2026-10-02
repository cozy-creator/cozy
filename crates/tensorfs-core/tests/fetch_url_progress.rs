//! THE DECISIVE ARM for `tfs fetch url --progress`: a real transfer says what it is doing
//! WHILE it is doing it, on the STDERR event channel `tfs fetch` already reports on.
//!
//! The H3 ingest pulled 210 GB across 48 files onto a rented pod over 2h56m and was lost,
//! and the reason it was lost is that its only signal was a cumulative byte total at each
//! file's COMPLETION, ten to fifteen minutes apart. For that whole window "slow" and
//! "wedged" are the same observation. So the bar here is not that the flag exists: it is
//! that records with a PARTIAL total reach a reader's pipe before the process that prints
//! them has exited, against a real socket and a real store — and that NONE of them can be
//! mistaken for the result, because the result is on the other stream.
//!
//! Nothing in this file judges a duration. The origin drips on purpose, the reader reads
//! until the pipe closes, and every assertion is about a byte count or an ordering.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::Duration;

/// A loopback origin that declares the whole length and then hands the body over in pieces,
/// the way a real CDN does — an honestly-moving slow link.
fn drip_origin(body: Vec<u8>, chunk: usize, gap: Duration) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback origin");
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        let Ok((mut socket, _)) = listener.accept() else {
            return;
        };
        let mut request = [0u8; 4096];
        let _ = socket.read(&mut request);
        let head = format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        );
        if socket.write_all(head.as_bytes()).is_err() {
            return;
        }
        for piece in body.chunks(chunk) {
            if socket.write_all(piece).is_err() || socket.flush().is_err() {
                return;
            }
            std::thread::sleep(gap);
        }
        let _ = socket.shutdown(std::net::Shutdown::Both);
    });
    // Detached on purpose: a fetch that refuses before it dials leaves this thread parked
    // in accept(), and a join here would turn that refusal into a hang instead of into the
    // assertion below.
    format!("http://127.0.0.1:{port}")
}

/// Read one integer field out of an event line. The events are flat objects of numbers, so
/// this needs no JSON library to be exact about them.
fn number(line: &str, key: &str) -> u64 {
    let at = line
        .find(&format!("\"{key}\":"))
        .unwrap_or_else(|| panic!("{key} absent from {line}"))
        + key.len()
        + 3;
    line[at..]
        .chars()
        .take_while(char::is_ascii_digit)
        .collect::<String>()
        .parse()
        .unwrap_or_else(|_| panic!("{key} is not a number in {line}"))
}

struct Record {
    moved: u64,
    admitted: u64,
    /// Whether the fetch process was still running when this record was read off the pipe.
    in_flight: bool,
}

#[test]
fn a_real_transfer_reports_partial_totals_while_it_is_still_moving() {
    let root = std::env::temp_dir().join(format!("tfs-progress-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();

    // 16 MiB, dripped 512 KiB at a time over several of the one-second samples the fetch
    // reports at. The object is deterministic so the digest the store admits is the digest
    // this test computed, with nothing to fake.
    let body: Vec<u8> = (0..(16usize << 20)).map(|i| (i * 31 + 7) as u8).collect();
    let length = body.len() as u64;
    let sha = tensorfs_core::sha256::hex_digest(&body);
    let origin = drip_origin(body, 512 << 10, Duration::from_millis(150));

    // A REAL store, made the way the pod makes one.
    let init = Command::new(env!("CARGO_BIN_EXE_tfs"))
        .args(["store", "init", root.to_str().unwrap()])
        .output()
        .expect("tfs store init");
    assert!(
        init.status.success(),
        "store init: {}",
        String::from_utf8_lossy(&init.stderr)
    );

    let mut child = Command::new(env!("CARGO_BIN_EXE_tfs"))
        .args([
            "fetch",
            "url",
            root.to_str().unwrap(),
            &sha,
            &length.to_string(),
            &format!("{origin}/one/granted/object"),
            "--allow-local",
            "--allow-hosts",
            "127.0.0.1",
            "--progress",
            // The channel's RESOLUTION, stated so this proof does not need a 64 MiB object
            // to see more than one record. It is configuration; it is not a verdict, and
            // production states nothing and gets the constant.
            "--progress-bytes",
            "1048576",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn tfs");

    // EVENTS ARE STDERR. stdout is drained on its own thread and asserted on afterwards:
    // it must carry the result and nothing else.
    let stdout = child.stdout.take().unwrap();
    let result = std::thread::spawn(move || {
        let mut text = String::new();
        let _ = BufReader::new(stdout).read_to_string(&mut text);
        text
    });
    let stderr = child.stderr.take().unwrap();
    let (tx, rx) = mpsc::channel::<String>();
    let reader = std::thread::spawn(move || {
        for line in BufReader::new(stderr).lines() {
            let Ok(line) = line else { break };
            if tx.send(line).is_err() {
                break;
            }
        }
    });

    // Read the pipe while the child runs, marking each record with whether the child was
    // still alive when it arrived. A command that printed everything at exit produces
    // nothing marked in flight — which is exactly the defect this proves closed.
    let mut records: Vec<Record> = Vec::new();
    let mut lines: Vec<String> = Vec::new();
    while let Ok(line) = rx.recv() {
        let alive = child.try_wait().ok().flatten().is_none();
        if line.contains("\"event\":\"fetch.progress\"") {
            records.push(Record {
                moved: number(&line, "origin_bytes"),
                admitted: number(&line, "origin_objects"),
                in_flight: alive,
            });
        }
        // Every event is one JSON object with `event` first — the shape `cmd_pull` set.
        assert!(
            line.starts_with("{\"event\":\"") && line.ends_with('}'),
            "the channel carried a non-event line: {line}"
        );
        lines.push(line);
    }
    let status = child.wait().expect("wait for tfs");
    let stdout = result.join().expect("drain stdout");
    let _ = reader.join();
    let _ = std::fs::remove_dir_all(&root);

    assert!(
        status.success(),
        "the fetch refused:\n{stdout}\n{}",
        lines.join("\n")
    );

    // NO EVENT CAN BE PARSED AS A COMPLETION PROOF, and the guarantee is which stream it
    // arrived on. stdout carries this command's result — the disposition and the byte
    // count a caller reads — and carries nothing else.
    assert!(
        stdout.contains("admitted     ") && stdout.contains("transferred  "),
        "stdout does not carry the result: {stdout}"
    );
    assert!(
        !stdout.contains("\"event\""),
        "an event reached the result stream: {stdout}"
    );

    // THE WHOLE POINT: partial totals, on the reader's pipe, before the process exited.
    let live: Vec<&Record> = records
        .iter()
        .filter(|r| r.in_flight && r.moved > 0 && r.moved < length)
        .collect();
    assert!(
        live.len() >= 2,
        "the transfer reported {} partial total(s) while still moving; \
         a completion-only signal cannot tell slow from wedged.\n{}",
        live.len(),
        lines.join("\n")
    );

    // Cumulative and monotonic, so a reader that misses a record loses nothing.
    let mut previous = 0u64;
    for record in &records {
        assert!(
            record.moved >= previous,
            "the totals are not cumulative: {} after {previous}",
            record.moved
        );
        previous = record.moved;
    }

    // The last record closes the channel: every byte, and the object admitted.
    let last = records.last().expect("at least one record");
    assert_eq!(
        (last.moved, last.admitted),
        (length, 1),
        "the closing record is {} B / {} admitted",
        last.moved,
        last.admitted
    );

    // The URL is a bearer capability. It is not in the channel, at any point.
    for line in &lines {
        assert!(
            !line.contains("/one/granted/object"),
            "the granted URL reached the observation channel: {line}"
        );
    }
}
