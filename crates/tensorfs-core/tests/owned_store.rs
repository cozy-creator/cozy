//! The embedded owner role uses real store files, flock and process death. Ordinary
//! readers retain their existing recovery-lock behavior across the ownership transition.

use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::fs::symlink;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

use tensorfs_core::err::Code;
use tensorfs_core::ids::ObjectRef;
use tensorfs_core::meta::{now_nanos_unique, Meta};
use tensorfs_core::read;
use tensorfs_core::store::{Fault, Store};

struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "tensorfs-owned-store-{}-{}",
            std::process::id(),
            now_nanos_unique()
        ));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn store(&self) -> PathBuf {
        self.0.join("store")
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

struct Process(Child);

impl Process {
    fn start(root: &Path, mode: &str) -> Self {
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "owner_process", "--ignored", "--nocapture"])
            .env("TFS_OWNER_TEST_ROOT", root)
            .env("TFS_OWNER_TEST_MODE", mode)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        let mut output = BufReader::new(child.stdout.take().unwrap());
        let mut line = String::new();
        loop {
            line.clear();
            assert_ne!(
                output.read_line(&mut line).unwrap(),
                0,
                "child ended before ready"
            );
            if line.trim() == "OWNER_READY" {
                break;
            }
        }
        Self(child)
    }
}

impl Drop for Process {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn put(store: &Store, bytes: &[u8]) -> ObjectRef {
    store
        .put_stream(&mut &bytes[..], None, &Fault::default())
        .unwrap()
        .obj
}

#[test]
#[ignore = "subprocess fixture"]
fn owner_process() {
    let root = PathBuf::from(std::env::var_os("TFS_OWNER_TEST_ROOT").unwrap());
    let mode = std::env::var("TFS_OWNER_TEST_MODE").unwrap();
    let store = if mode == "owned" {
        Store::ensure_owned(&root).unwrap()
    } else {
        Store::ensure(&root).unwrap()
    };
    let meta = Meta::open(&store).unwrap();
    let object = put(&store, b"external reader");
    let (lease, _) = read::acquire(&store, &meta, "layout", vec![object]).unwrap();
    println!("OWNER_READY");
    std::io::stdout().flush().unwrap();
    let mut done = String::new();
    std::io::stdin().read_line(&mut done).unwrap();
    lease.release(&meta).unwrap();
}

#[test]
fn clones_and_reopened_handles_keep_the_claim() {
    let scratch = Scratch::new();
    let root = scratch.store();
    let first = Store::ensure_owned(&root).unwrap();
    let cloned = first.clone();
    let reopened = Store::open(&root).unwrap();
    drop(first);
    assert_eq!(
        Store::ensure_owned(&root).unwrap_err().code,
        Code::STORE_BUSY
    );
    drop(cloned);
    assert_eq!(
        Store::ensure_owned(&root).unwrap_err().code,
        Code::STORE_BUSY
    );
    drop(reopened);
    drop(Store::ensure_owned(&root).unwrap());
}

#[test]
fn the_existing_daemon_lock_and_library_claim_exclude_each_other() {
    let scratch = Scratch::new();
    let root = scratch.store();
    drop(Store::ensure(&root).unwrap());
    // TensorD builds preceding this API retain this file directly.
    let legacy = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(root.join("owner.lock"))
        .unwrap();
    fs2::FileExt::try_lock_exclusive(&legacy).unwrap();
    assert_eq!(
        Store::ensure_owned(&root).unwrap_err().code,
        Code::STORE_BUSY
    );
    drop(legacy);
    let owner = Store::ensure_owned(&root).unwrap();
    let another = fs::File::open(root.join("owner.lock")).unwrap();
    assert_eq!(
        fs2::FileExt::try_lock_exclusive(&another)
            .unwrap_err()
            .kind(),
        std::io::ErrorKind::WouldBlock
    );
    drop(owner);
    fs2::FileExt::try_lock_exclusive(&another).unwrap();
}

#[test]
fn a_read_lease_keeps_the_claim_and_gc_pins_without_a_store_handle() {
    let scratch = Scratch::new();
    let root = scratch.store();
    let store = Store::ensure_owned(&root).unwrap();
    let meta = Meta::open(&store).unwrap();
    let object = put(&store, b"owner reader");
    let garbage = put(&store, b"unreferenced bytes");
    let path = store.object_path(&object.sha256);
    let garbage_path = store.object_path(&garbage.sha256);
    let (lease, _) = read::acquire(&store, &meta, "layout", vec![object.clone()]).unwrap();
    drop(store);
    assert_eq!(
        Store::ensure_owned(&root).unwrap_err().code,
        Code::STORE_BUSY
    );
    tensorfs_core::gc::collect(&root, false).unwrap();
    assert!(path.is_file());
    assert!(!garbage_path.exists());
    lease.recheck(&meta).unwrap();
    let mut bytes = vec![0; object.length as usize];
    read::read_into(
        &lease,
        &read::ObjectRange {
            obj: object.clone(),
            off: 0,
            len: object.length,
        },
        &mut bytes,
    )
    .unwrap();
    assert_eq!(bytes, b"owner reader");
    lease.release(&meta).unwrap();

    // The old process-wide opt-in must not survive the last owner handle. This ordinary
    // lease uses the recovery lock again, even in the same process and at the same root.
    let shared = Store::open(&root).unwrap();
    let (lease, _) = read::acquire(&shared, &meta, "layout", vec![object]).unwrap();
    assert_eq!(
        tensorfs_core::gc::collect(&root, false).unwrap_err().code,
        Code::STORE_BUSY
    );
    lease.release(&meta).unwrap();
    drop(Store::ensure_owned(&root).unwrap());
    tensorfs_core::gc::collect(&root, false).unwrap();
    assert!(!path.exists());
}

#[test]
fn another_process_and_path_alias_cannot_claim_until_owner_death() {
    let scratch = Scratch::new();
    let root = scratch.store();
    let first = Process::start(&root, "owned");
    symlink(&scratch.0, scratch.0.join("alias")).unwrap();
    let alias = scratch.0.join("alias/store");
    assert_eq!(
        Store::ensure_owned(&alias).unwrap_err().code,
        Code::STORE_BUSY
    );
    // Ownership excludes another owner, not ordinary store access.
    drop(Store::ensure(&root).unwrap());
    drop(first); // real SIGKILL, then wait for kernel descriptor cleanup
    drop(Store::ensure_owned(&alias).unwrap());
}

#[test]
fn another_process_reader_still_excludes_owner_gc() {
    let scratch = Scratch::new();
    let root = scratch.store();
    let owner = Store::ensure_owned(&root).unwrap();
    let reader = Process::start(&root, "shared");
    assert_eq!(
        tensorfs_core::gc::collect(&root, false).unwrap_err().code,
        Code::STORE_BUSY
    );
    drop(reader);
    tensorfs_core::gc::collect(&root, false).unwrap();
    drop(owner);
}

#[test]
fn contention_refuses_before_inspecting_or_recovering_the_catalog() {
    let scratch = Scratch::new();
    let root = scratch.store();
    let owner = Store::ensure_owned(&root).unwrap();
    fs::remove_file(root.join("tensorfs.sqlite")).unwrap();
    assert_eq!(
        Store::ensure_owned(&root).unwrap_err().code,
        Code::STORE_BUSY
    );
    assert!(!root.join("tensorfs.sqlite").exists());
    drop(owner);
    assert_eq!(
        Store::ensure_owned(&root).unwrap_err().code,
        Code::DURABILITY_UNPROVEN
    );
    // Failed open relinquishes its claim as well.
    let lock = fs::File::open(root.join("owner.lock")).unwrap();
    fs2::FileExt::try_lock_exclusive(&lock).unwrap();
}

#[test]
fn ownership_does_not_initialize_foreign_or_partial_roots() {
    let scratch = Scratch::new();
    let foreign = scratch.0.join("foreign");
    fs::create_dir(&foreign).unwrap();
    fs::write(foreign.join("unrelated"), b"keep").unwrap();
    assert_eq!(
        Store::ensure_owned(&foreign).unwrap_err().code,
        Code::STORE_ERA
    );
    assert!(!foreign.join("owner.lock").exists());
    let partial = scratch.0.join("partial");
    fs::create_dir_all(partial.join("blobs")).unwrap();
    assert_eq!(
        Store::ensure_owned(&partial).unwrap_err().code,
        Code::STORE_ERA
    );
    assert!(!partial.join("owner.lock").exists());
    symlink(&foreign, scratch.store()).unwrap();
    assert_eq!(
        Store::ensure_owned(&scratch.store()).unwrap_err().code,
        Code::STORE_ERA
    );
}

#[test]
fn an_owner_lock_symlink_or_nonregular_file_is_refused() {
    let scratch = Scratch::new();
    let root = scratch.store();
    drop(Store::ensure(&root).unwrap());
    let other = scratch.0.join("other");
    fs::write(&other, b"leave intact").unwrap();
    symlink(&other, root.join("owner.lock")).unwrap();
    assert_eq!(
        Store::ensure_owned(&root).unwrap_err().code,
        Code::IO_FAILED
    );
    assert_eq!(fs::read(&other).unwrap(), b"leave intact");
    fs::remove_file(root.join("owner.lock")).unwrap();
    let fifo =
        std::ffi::CString::new(root.join("owner.lock").as_os_str().as_encoded_bytes()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(fifo.as_ptr(), 0o600) }, 0);
    assert_eq!(
        Store::ensure_owned(&root).unwrap_err().code,
        Code::NOT_REGULAR_FILE
    );
}
