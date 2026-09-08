//! THE DECISIVE ARM for tfs-067: a fetched source carrier is STAGED, not admitted.
//!
//! The claim under test is a negative one — "this file is not a store object" — and a
//! negative is only worth anything if it is checked against the machinery that would notice.
//! So every assertion here is paired: the same bytes, the same length, the same digest, put
//! through `staging` and through `put_stream`, asked the same three questions.
//!
//!   - is it installed?                    `Store::contains`
//!   - would a collection take it?         `Census::gc_plan`, then a real `gc::collect`
//!   - does the catalog vouch for it?      `tensorfs_verified_blobs`
//!
//! **The red arm is the point.** Without a test that shows the CAS path DOES appear in
//! `gc_plan`, the suite asserts only that `staging/` is a directory nothing happens to look
//! in, which would pass just as well against a store with no census at all.
//!
//! Everything runs through the real `tfs` binary against a real loopback origin, in the
//! shape `tests/fetch_url_progress.rs` uses: a real socket, a real Store, a real digest.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use tensorfs_core::ids::ObjectRef;
use tensorfs_core::storage::Census;
use tensorfs_core::store::{Fault, Store};

// ---------------------------------------------------------------- the loopback origin

/// An origin that answers every GET with whatever `body` currently holds, and counts the
/// asks. The count is how "a re-ask moved zero bytes" is proved against the wire rather
/// than against a number the process under test printed about itself.
struct Origin {
    url: String,
    body: Arc<Mutex<Vec<u8>>>,
    /// Declared `Content-Length`. `None` is the honest one.
    declared: Arc<Mutex<Option<usize>>>,
    asks: Arc<AtomicUsize>,
}

impl Origin {
    fn asks(&self) -> usize {
        self.asks.load(Ordering::Relaxed)
    }
    fn serve(&self, bytes: Vec<u8>) {
        *self.body.lock().unwrap() = bytes;
    }
    fn declare(&self, length: Option<usize>) {
        *self.declared.lock().unwrap() = length;
    }
}

fn origin(body: Vec<u8>) -> Origin {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback origin");
    let port = listener.local_addr().unwrap().port();
    let body = Arc::new(Mutex::new(body));
    let declared: Arc<Mutex<Option<usize>>> = Arc::new(Mutex::new(None));
    let asks = Arc::new(AtomicUsize::new(0));
    let (b, d, a) = (body.clone(), declared.clone(), asks.clone());
    std::thread::spawn(move || {
        for socket in listener.incoming() {
            let Ok(mut socket) = socket else { return };
            let mut request = [0u8; 4096];
            let _ = socket.read(&mut request);
            a.fetch_add(1, Ordering::Relaxed);
            let payload = b.lock().unwrap().clone();
            let length = d.lock().unwrap().unwrap_or(payload.len());
            let head =
                format!("HTTP/1.1 200 OK\r\nContent-Length: {length}\r\nConnection: close\r\n\r\n");
            if socket.write_all(head.as_bytes()).is_err() {
                continue;
            }
            let _ = socket.write_all(&payload);
            let _ = socket.flush();
            let _ = socket.shutdown(std::net::Shutdown::Both);
        }
    });
    Origin {
        url: format!("http://127.0.0.1:{port}"),
        body,
        declared,
        asks,
    }
}

// ---------------------------------------------------------------- the harness

fn temporary(name: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!(
        "tfs-staging-{name}-{}-{:?}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    root
}

fn tfs(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_tfs"))
        .args(args)
        .output()
        .expect("run tfs")
}

fn store_at(root: &Path) -> PathBuf {
    let store = root.join("store");
    let out = tfs(&["store", "init", store.to_str().unwrap()]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    store
}

/// `tfs staging fetch`, with the loopback fence a declared-local deployment gets.
fn stage(store: &Path, object: &ObjectRef, origin: &Origin) -> Output {
    tfs(&[
        "staging",
        "fetch",
        store.to_str().unwrap(),
        &object.sha256,
        &object.length.to_string(),
        &format!("{}/a/carrier", origin.url),
        "--allow-local",
        "--allow-hosts",
        "127.0.0.1",
    ])
}

fn field(stdout: &str, key: &str) -> String {
    stdout
        .lines()
        .find_map(|line| line.strip_prefix(key))
        .unwrap_or_else(|| panic!("{key} absent from:\n{stdout}"))
        .trim()
        .to_string()
}

/// Every `stage-` temp still under `tmp/`. A refusal must leave none.
fn stage_temps(store: &Path) -> Vec<PathBuf> {
    std::fs::read_dir(store.join("tmp"))
        .map(|entries| {
            entries
                .flatten()
                .map(|entry| entry.path())
                .filter(|path| {
                    path.file_name()
                        .and_then(|name| name.to_str())
                        .is_some_and(|name| name.starts_with("stage-"))
                })
                .collect()
        })
        .unwrap_or_default()
}

fn vouched(store: &Path, sha256: &str) -> u32 {
    let connection = rusqlite::Connection::open(store.join("tensorfs.sqlite")).unwrap();
    connection
        .query_row(
            "SELECT count(*) FROM tensorfs_verified_blobs WHERE sha256=?1",
            [sha256],
            |row| row.get(0),
        )
        .unwrap()
}

fn planned(store: &Path) -> Vec<String> {
    Census::open(store)
        .unwrap()
        .gc_plan(&[])
        .unwrap()
        .into_iter()
        .map(|row| row.sha256)
        .collect()
}

fn payload(seed: usize, length: usize) -> Vec<u8> {
    (0..length)
        .map(|i| (i * 31 + seed * 7 + 11) as u8)
        .collect()
}

// Concurrent tests start real child processes. An inherited recovery descriptor
// may outlive this thread's writer until exec; STORE_BUSY is then the correct GC
// refusal. Await that transient liveness before asserting exact collection facts.
fn collect_when_idle(root: &Path) -> tensorfs_core::gc::Report {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        match tensorfs_core::gc::collect(root, false) {
            Ok(report) => return report,
            Err(error)
                if error.code == tensorfs_core::err::Code::STORE_BUSY
                    && std::time::Instant::now() < deadline =>
            {
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            Err(error) => panic!("idle staging collection failed: {error:?}"),
        }
    }
}

// ---------------------------------------------------------------- the proof

/// THE WHOLE ISSUE, in one test and its red arm.
///
/// A carrier is on disk, readable at an absolute path, and provably not a store object by
/// three independent checks — and the SAME BYTES admitted the ordinary way fail all three
/// checks in the opposite direction. The store is asked the same questions both times.
#[test]
fn a_staged_carrier_is_not_a_store_object_and_the_same_bytes_admitted_are() {
    let root = temporary("not-an-object");
    let store_root = store_at(&root);
    let bytes = payload(1, 1 << 20);
    let carrier = ObjectRef::of(&bytes);
    let origin = origin(bytes.clone());

    let out = stage(&store_root, &carrier, &origin);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    assert_eq!(field(&stdout, "staged"), carrier.id());
    assert_eq!(
        field(&stdout, "transferred"),
        format!("{} B", carrier.length)
    );

    // The path is PRINTED, not derived. That is the point of the verb: the supervisor stops
    // knowing TensorFS's internal shape.
    let path = PathBuf::from(field(&stdout, "path"));
    assert_eq!(path, store_root.join("staging").join(&carrier.sha256));
    assert!(path.is_absolute());
    let metadata = std::fs::symlink_metadata(&path).unwrap();
    assert!(metadata.is_file() && !metadata.file_type().is_symlink());
    assert_eq!(metadata.len(), carrier.length);
    assert_eq!(std::fs::read(&path).unwrap(), bytes);

    // ---- check 1: not installed.
    let store = Store::open(&store_root).unwrap();
    assert!(
        !store.contains(&carrier.sha256),
        "a staged carrier answered `contains`"
    );
    assert!(!store.blob_path(&carrier.sha256).exists());

    // ---- check 2: not a collection candidate. `Census` walks repos/, manifests/ and
    // blobs/, so this is structural rather than a rule anyone has to remember.
    assert!(
        !planned(&store_root).contains(&carrier.sha256),
        "a staged carrier reached the gc plan"
    );

    // ---- check 3: no verification record.
    assert_eq!(vouched(&store_root, &carrier.sha256), 0);

    // ---- and a REAL collection reclaims nothing and leaves it standing.
    let report = collect_when_idle(&store_root);
    assert_eq!((report.reclaimed_bytes, report.reclaimed_blobs), (0, 0));
    assert!(path.is_file(), "gc reclaimed a staged carrier");

    // ---- THE RED ARM. The same bytes through the CAS door answer every one of those the
    // other way, and a collection takes them. Without this the three assertions above hold
    // against any directory nothing happens to look in.
    let admitted = store
        .put_stream(&mut bytes.as_slice(), Some(&carrier), &Fault::default())
        .unwrap();
    assert!(admitted.admitted);
    assert!(store.contains(&carrier.sha256));
    assert_eq!(vouched(&store_root, &carrier.sha256), 1);
    assert!(
        planned(&store_root).contains(&carrier.sha256),
        "the CAS path did not reach the gc plan, so this suite proves nothing"
    );
    let report = collect_when_idle(&store_root);
    assert_eq!(report.reclaimed_bytes, carrier.length);
    assert_eq!(report.reclaimed_blobs, 1);
    assert!(!store.blob_path(&carrier.sha256).exists());
    // One name was collected and the other was not, at the same digest, in one pass.
    assert!(
        path.is_file(),
        "gc reclaimed the staged carrier alongside the blob"
    );
    assert_eq!(vouched(&store_root, &carrier.sha256), 0);

    let _ = std::fs::remove_dir_all(&root);
}

/// The digest is enforced on arrival, exactly as it is at the CAS door — and a refusal
/// leaves nothing behind, neither a staged name nor a multi-gigabyte temp.
#[test]
fn a_flipped_byte_refuses_and_stages_nothing() {
    let root = temporary("flipped");
    let store_root = store_at(&root);
    let bytes = payload(2, 512 << 10);
    let carrier = ObjectRef::of(&bytes);
    let mut wrong = bytes.clone();
    wrong[9_000] ^= 0xff;
    let origin = origin(wrong);

    let out = stage(&store_root, &carrier, &origin);
    assert!(!out.status.success(), "a wrong-bytes fetch succeeded");
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    assert!(
        stderr.contains("OBJECT_ID_MISMATCH"),
        "the refusal was not about the digest: {stderr}"
    );
    assert!(!store_root.join("staging").join(&carrier.sha256).exists());
    assert!(
        stage_temps(&store_root).is_empty(),
        "a refused fetch left its temp behind"
    );
    // Nothing reached the CAS either — the narrower door is not a back door.
    assert_eq!(vouched(&store_root, &carrier.sha256), 0);

    let _ = std::fs::remove_dir_all(&root);
}

/// A carrier the origin serves short refuses on LENGTH, before the digest can be reached.
#[test]
fn a_short_carrier_refuses_on_length() {
    let root = temporary("short");
    let store_root = store_at(&root);
    let bytes = payload(3, 256 << 10);
    let carrier = ObjectRef::of(&bytes);
    let origin = origin(bytes[..bytes.len() - 4_096].to_vec());
    // Declared honestly at the SHORT length, so the body arrives complete and it is the
    // door — not the socket — that has the last word.
    origin.declare(Some(bytes.len() - 4_096));

    let out = stage(&store_root, &carrier, &origin);
    assert!(!out.status.success(), "a short fetch succeeded");
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    assert!(
        stderr.contains("LENGTH_MISMATCH"),
        "the refusal was not about the length: {stderr}"
    );
    assert!(!store_root.join("staging").join(&carrier.sha256).exists());
    assert!(stage_temps(&store_root).is_empty());

    let _ = std::fs::remove_dir_all(&root);
}

/// Residency is answered by REHASH, and that is what makes a pod restart cheap.
///
/// A re-ask over a good staged file moves zero bytes and never opens a socket; a re-ask
/// over a corrupted one removes it and re-fetches. This preserves a behaviour `tfs fetch
/// url` has today, which a port to a new destination could silently have dropped — tfs-060's
/// shape exactly.
#[test]
fn a_re_ask_moves_zero_bytes_and_a_corrupt_carrier_re_fetches() {
    let root = temporary("re-ask");
    let store_root = store_at(&root);
    let bytes = payload(4, 1 << 20);
    let carrier = ObjectRef::of(&bytes);
    let origin = origin(bytes.clone());

    let out = stage(&store_root, &carrier, &origin);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(origin.asks(), 1);
    let path = store_root.join("staging").join(&carrier.sha256);

    // ---- the warm re-ask: no socket, no bytes.
    let out = stage(&store_root, &carrier, &origin);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    assert_eq!(field(&stdout, "present"), carrier.id());
    assert_eq!(field(&stdout, "transferred"), "0 B");
    assert_eq!(field(&stdout, "path"), path.display().to_string());
    assert_eq!(origin.asks(), 1, "a warm re-ask opened a socket");

    // ---- the corrupted one: same length, different bytes. A length check would pass it, so
    // this is the arm that says the answer is a rehash and not a stat.
    let mut rotted = bytes.clone();
    rotted[42] ^= 0xff;
    let mut permissions = std::fs::metadata(&path).unwrap().permissions();
    std::os::unix::fs::PermissionsExt::set_mode(&mut permissions, 0o644);
    std::fs::set_permissions(&path, permissions).unwrap();
    std::fs::write(&path, &rotted).unwrap();
    assert_eq!(std::fs::metadata(&path).unwrap().len(), carrier.length);

    let out = stage(&store_root, &carrier, &origin);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    assert_eq!(field(&stdout, "staged"), carrier.id());
    assert_eq!(
        field(&stdout, "transferred"),
        format!("{} B", carrier.length)
    );
    assert_eq!(origin.asks(), 2, "a corrupt carrier was not re-fetched");
    assert_eq!(std::fs::read(&path).unwrap(), bytes);

    // ---- a carrier at the WRONG LENGTH is removed on the stat, with no rehash to pay.
    let short = payload(5, 1 << 10);
    origin.serve(short.clone());
    origin.declare(None);
    let other = ObjectRef::of(&short);
    let out = stage(&store_root, &other, &origin);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let other_path = store_root.join("staging").join(&other.sha256);
    let mut permissions = std::fs::metadata(&other_path).unwrap().permissions();
    std::os::unix::fs::PermissionsExt::set_mode(&mut permissions, 0o644);
    std::fs::set_permissions(&other_path, permissions).unwrap();
    std::fs::write(&other_path, b"a stump").unwrap();
    let asks = origin.asks();
    let out = stage(&store_root, &other, &origin);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(origin.asks(), asks + 1);
    assert_eq!(std::fs::read(&other_path).unwrap(), short);

    let _ = std::fs::remove_dir_all(&root);
}

/// `drop` is one unlink and it is idempotent; `clear` sweeps the area and the orphan temps
/// a crashed fetch left, and touches nothing that is not the store's own staging scratch.
#[test]
fn drop_is_idempotent_and_clear_sweeps_carriers_and_orphan_temps() {
    let root = temporary("retire");
    let store_root = store_at(&root);
    let store = store_root.to_str().unwrap();

    // ---- an id that was never staged. The caller's intent is already true.
    let absent = "0".repeat(64);
    let out = tfs(&["staging", "drop", store, &absent]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(String::from_utf8_lossy(&out.stdout).contains("absent"));

    let first = payload(6, 128 << 10);
    let second = payload(7, 64 << 10);
    let one = ObjectRef::of(&first);
    let two = ObjectRef::of(&second);
    let origin = origin(first.clone());
    assert!(stage(&store_root, &one, &origin).status.success());
    origin.serve(second.clone());
    assert!(stage(&store_root, &two, &origin).status.success());

    // ---- list names both, at their lengths, at absolute paths.
    let out = tfs(&["staging", "list", store]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let listing = String::from_utf8_lossy(&out.stdout).to_string();
    for object in [&one, &two] {
        let path = store_root.join("staging").join(&object.sha256);
        assert!(
            listing.contains(&format!(
                "{} {} {}",
                object.id(),
                object.length,
                path.display()
            )),
            "list did not name {}: {listing}",
            object.id()
        );
    }

    // ---- drop one. The other is untouched: this is a targeted retirement with no plan
    // behind it and nothing to be atomic about.
    let out = tfs(&["staging", "drop", store, &one.sha256]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(String::from_utf8_lossy(&out.stdout).contains("dropped"));
    assert!(!store_root.join("staging").join(&one.sha256).exists());
    assert!(store_root.join("staging").join(&two.sha256).is_file());
    // ...and dropping it again is success, not a refusal.
    let out = tfs(&["staging", "drop", store, &one.sha256]);
    assert!(out.status.success());
    assert!(String::from_utf8_lossy(&out.stdout).contains("absent"));

    // ---- the crash case: a `stage-` temp whose writer is gone, and a `put-` temp that is
    // somebody else's business. `clear` is the end-of-operation sweep, not a garbage
    // collection, so it takes the first and leaves the second.
    let orphan = store_root.join("tmp/stage-99999-12345");
    std::fs::write(&orphan, vec![0u8; 4096]).unwrap();
    let foreign = store_root.join("tmp/put-99999-12345");
    std::fs::write(&foreign, vec![0u8; 16]).unwrap();

    let out = tfs(&["staging", "clear", store]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    assert_eq!(field(&stdout, "cleared"), "1 carrier(s)");
    assert_eq!(field(&stdout, "reclaimed"), format!("{} B", two.length));
    assert_eq!(field(&stdout, "temps"), "1 orphan(s)");
    assert!(!store_root.join("staging").join(&two.sha256).exists());
    assert!(!orphan.exists());
    assert!(foreign.is_file(), "clear took an admission temp");
    // The area itself survives an empty sweep: a concurrent staging writer is between a
    // create and a link, and removing the directory under it would fail that writer.
    assert!(store_root.join("staging").is_dir());

    let out = tfs(&["staging", "clear", store]);
    assert!(out.status.success());
    assert_eq!(
        field(&String::from_utf8_lossy(&out.stdout), "cleared"),
        "0 carrier(s)"
    );

    let _ = std::fs::remove_dir_all(&root);
}

/// A `gc` pass reaps the orphan temp of a crashed FETCH — the 5 GB leak nothing else would
/// ever name — and leaves every landed carrier exactly where it is.
#[test]
fn gc_reaps_an_orphan_stage_temp_and_never_a_landed_carrier() {
    let root = temporary("gc-temps");
    let store_root = store_at(&root);
    let bytes = payload(8, 96 << 10);
    let carrier = ObjectRef::of(&bytes);
    let origin = origin(bytes.clone());
    assert!(stage(&store_root, &carrier, &origin).status.success());

    let orphan = store_root.join("tmp/stage-77777-54321");
    std::fs::write(&orphan, vec![7u8; 8192]).unwrap();

    let report = collect_when_idle(&store_root);
    assert_eq!(report.reclaimed_bytes, 0);
    assert!(report.scratch_reaped >= 1, "gc left the orphan stage- temp");
    assert!(!orphan.exists());
    assert!(store_root.join("staging").join(&carrier.sha256).is_file());

    let _ = std::fs::remove_dir_all(&root);
}

/// The disk budget still means what it meant. A carrier that left the CAS must not leave
/// the arithmetic with it: before tfs-067 a fetched shard was an ordinary blob and
/// `occupancy` counted it, and a store with 200 GB of carriers on it reporting full
/// headroom is the stall this work exists to prevent.
#[test]
fn a_staged_carrier_is_still_counted_against_the_disk_budget() {
    let root = temporary("occupancy");
    let store_root = store_at(&root);
    let bytes = payload(9, 200 << 10);
    let carrier = ObjectRef::of(&bytes);
    let origin = origin(bytes.clone());

    let store = Store::open(&store_root).unwrap();
    assert_eq!(store.occupancy().unwrap(), 0);
    assert!(stage(&store_root, &carrier, &origin).status.success());
    assert_eq!(store.occupancy().unwrap(), carrier.length);

    // ...and it stops being counted the moment it is dropped, because it really is gone.
    assert!(tfs(&[
        "staging",
        "drop",
        store_root.to_str().unwrap(),
        &carrier.sha256
    ])
    .status
    .success());
    assert_eq!(store.occupancy().unwrap(), 0);

    let _ = std::fs::remove_dir_all(&root);
}
