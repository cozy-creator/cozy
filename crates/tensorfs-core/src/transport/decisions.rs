//! One line per dropped request and one per pull, appended to the Store's own bounded log
//! so a slow tail is visible after the fact: run 2322 and sasori left nothing on disk to
//! say what the transport did. Never per chunk, never fatal.
use crate::ids::ObjectRef;
use crate::store::Store;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::time::{SystemTime, UNIX_EPOCH};

/// Past this the log rotates to `transport.log.1`, so it holds at most twice this.
const ROTATE_BYTES: u64 = 4 << 20;

/// How a line names one object: its digest prefix and length.
pub(super) fn object(object: &ObjectRef) -> String {
    format!(
        "{} {}",
        &object.sha256[..object.sha256.len().min(16)],
        object.length
    )
}

pub(super) fn record(store: &Store, kind: &str, subject: &str, detail: &str) {
    let dir = store.root().join("logs");
    let path = dir.join("transport.log");
    let _ = fs::create_dir_all(&dir);
    if fs::metadata(&path).is_ok_and(|meta| meta.len() > ROTATE_BYTES) {
        let _ = fs::rename(&path, dir.join("transport.log.1"));
    }
    let at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|since| since.as_secs_f64())
        .unwrap_or(0.0);
    let line = format!("{at:.3} {kind} {subject} {detail}\n");
    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(&path) {
        let _ = file.write_all(line.as_bytes());
    }
}
