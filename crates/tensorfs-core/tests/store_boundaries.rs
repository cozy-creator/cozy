use std::fs;
use std::os::unix::fs::{symlink, PermissionsExt};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use tensorfs_core::err::Code;
use tensorfs_core::header::{Body, Closure, Header, Part, Tensor};
use tensorfs_core::ids::ObjectRef;
use tensorfs_core::manifest::{Draft, Entry, Manifest};
use tensorfs_core::store::{Fault, Store};
use tensorfs_core::{dtype::Dtype, registry};

fn temporary(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "tensorfs-boundary-{name}-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}

fn mode(path: &Path) -> u32 {
    fs::symlink_metadata(path).unwrap().permissions().mode() & 0o7777
}

fn model_manifest(store: &Store, body: Body, shape: Vec<u64>, note: &[u8]) -> ObjectRef {
    let spec = registry::seeds()
        .into_iter()
        .find(|seed| seed.alias == "plain/1")
        .unwrap()
        .spec;
    let header = Header {
        configs: Vec::new(),
        assets: Vec::new(),
        encodings: vec![spec.clone()],
        components: vec![(
            "model".into(),
            vec![(
                "weight".into(),
                Tensor {
                    dtype: Dtype::F32,
                    shape: shape.clone(),
                    encoding: spec.object_id(),
                    parts: vec![(
                        "value".into(),
                        Part {
                            dtype: Dtype::F32,
                            shape,
                            body,
                        },
                    )],
                },
            )],
        )],
    };
    header.validate(&Closure::default()).unwrap();
    let header_bytes = header.canonical_bytes().unwrap();
    let header_ref = store
        .put_stream(
            &mut header_bytes.as_slice(),
            Some(&ObjectRef::of(&header_bytes)),
            &Fault::default(),
        )
        .unwrap()
        .obj;
    let manifest = Draft {
        entries: vec![
            ("README.txt".into(), Entry::File(ObjectRef::of(note))),
            ("model.cozytensors".into(), Entry::CozyTensors(header_ref)),
        ],
    }
    .seal()
    .unwrap();
    store.put_manifest(&manifest).unwrap().obj
}

/// Many processes ensuring one brand-new root at once: one creates it, the others open it,
/// and none is refused as "not empty".
#[test]
fn concurrent_ensures_of_a_new_root_all_succeed() {
    let root = temporary("concurrent-ensure");
    fs::create_dir_all(&root).unwrap();
    let empty = root.join("empty");
    fs::create_dir(&empty).unwrap();
    for target in [root.join("absent"), empty] {
        let children: Vec<_> = (0..16)
            .map(|_| {
                Command::new(env!("CARGO_BIN_EXE_tfs"))
                    .args(["store", "ensure", target.to_str().unwrap()])
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::piped())
                    .spawn()
                    .unwrap()
            })
            .collect();
        for child in children {
            let output = child.wait_with_output().unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        Store::open(&target).unwrap();
    }
    let _ = fs::remove_dir_all(root);
}

#[test]
fn ensure_initializes_only_absent_or_empty_real_roots() {
    let root = temporary("ensure");
    fs::create_dir_all(&root).unwrap();

    let absent = root.join("absent");
    let output = Command::new(env!("CARGO_BIN_EXE_tfs"))
        .args(["store", "ensure", absent.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8(output.stdout)
        .unwrap()
        .contains(&format!("root:       {}", absent.display())));
    Store::ensure(&absent).unwrap();
    assert_eq!(Store::init(&absent).unwrap_err().code, Code::STORE_ERA);
    Store::ensure(&absent).unwrap();

    let empty = root.join("empty");
    fs::create_dir(&empty).unwrap();
    Store::ensure(&empty).unwrap();

    let foreign = root.join("foreign");
    fs::create_dir(&foreign).unwrap();
    fs::write(foreign.join("somebody-elses-state"), b"keep").unwrap();
    assert_eq!(Store::ensure(&foreign).unwrap_err().code, Code::STORE_ERA);
    assert_eq!(Store::init(&foreign).unwrap_err().code, Code::STORE_ERA);
    assert!(!foreign.join("repos").exists());

    let partial = root.join("partial");
    fs::create_dir(&partial).unwrap();
    fs::create_dir(partial.join("repos")).unwrap();
    fs::create_dir(partial.join("manifests")).unwrap();
    assert_eq!(Store::ensure(&partial).unwrap_err().code, Code::STORE_ERA);
    assert!(!partial.join("blobs").exists());

    let missing_catalog = root.join("missing-catalog");
    for namespace in ["repos", "manifests", "blobs"] {
        fs::create_dir_all(missing_catalog.join(namespace)).unwrap();
    }
    assert_eq!(
        Store::ensure(&missing_catalog).unwrap_err().code,
        Code::DURABILITY_UNPROVEN
    );
    assert!(!missing_catalog.join("tmp").exists());

    let corrupt_catalog = root.join("corrupt-catalog");
    Store::init(&corrupt_catalog).unwrap();
    fs::remove_dir_all(corrupt_catalog.join("tmp")).unwrap();
    let connection = rusqlite::Connection::open(corrupt_catalog.join("tensorfs.sqlite")).unwrap();
    connection
        .execute_batch("ALTER TABLE tensorfs_verified_blobs DROP COLUMN mtime_ns")
        .unwrap();
    drop(connection);
    assert_eq!(
        Store::ensure(&corrupt_catalog).unwrap_err().code,
        Code::UNKNOWN_FORMAT
    );
    assert!(!corrupt_catalog.join("tmp").exists());

    let real = root.join("real");
    Store::init(&real).unwrap();
    let link = root.join("link");
    symlink(&real, &link).unwrap();
    assert_eq!(Store::ensure(&link).unwrap_err().code, Code::STORE_ERA);
    assert_eq!(Store::init(&link).unwrap_err().code, Code::STORE_ERA);
    assert_eq!(Store::open(&link).unwrap_err().code, Code::STORE_ERA);

    let missing_target = root.join("must-not-be-created");
    let dangling = root.join("dangling");
    symlink(&missing_target, &dangling).unwrap();
    assert_eq!(Store::init(&dangling).unwrap_err().code, Code::STORE_ERA);
    assert!(!missing_target.exists());

    for namespace in ["repos", "manifests", "blobs"] {
        let store_root = root.join(format!("symlink-{namespace}"));
        Store::init(&store_root).unwrap();
        let outside = root.join(format!("outside-{namespace}"));
        fs::create_dir(&outside).unwrap();
        fs::remove_dir(store_root.join(namespace)).unwrap();
        symlink(&outside, store_root.join(namespace)).unwrap();
        assert_eq!(
            Store::ensure(&store_root).unwrap_err().code,
            Code::STORE_ERA
        );
        assert_eq!(Store::open(&store_root).unwrap_err().code, Code::STORE_ERA);
    }
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn prepare_readers_owns_exact_modes_and_refuses_internal_symlinks() {
    let root = temporary("readers");
    let store_root = root.join("store");
    fs::create_dir_all(&root).unwrap();
    Store::ensure(&store_root).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_tfs"))
        .args(["store", "prepare-readers", store_root.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    for path in [
        "",
        "repos",
        "manifests",
        "blobs",
        "checkouts",
        // tfs-067: the staging area is TensorFS's to create at TensorFS's own mode, for
        // run-185's reason — a directory born under whichever uid touched the Store first
        // is one the pod's other identity cannot enter.
        "staging",
        "tmp",
        "tmp/writers",
    ] {
        assert_eq!(mode(&store_root.join(path)), 0o755, "{path}");
    }
    assert_eq!(mode(&store_root.join("tmp/leases")), 0o1733);
    assert_eq!(mode(&store_root.join("tmp/writers/recovery.lock")), 0o666);

    let outside = root.join("outside");
    fs::create_dir(&outside).unwrap();
    fs::set_permissions(&outside, fs::Permissions::from_mode(0o700)).unwrap();
    fs::remove_dir(store_root.join("tmp/leases")).unwrap();
    symlink(&outside, store_root.join("tmp/leases")).unwrap();
    let store = Store::open(&store_root).unwrap();
    assert!(store.prepare_readers().is_err());
    assert_eq!(mode(&outside), 0o700);

    fs::remove_file(store_root.join("tmp/leases")).unwrap();
    fs::create_dir(store_root.join("tmp/leases")).unwrap();
    let outside_file = root.join("outside-lock");
    fs::write(&outside_file, b"not the lock").unwrap();
    fs::set_permissions(&outside_file, fs::Permissions::from_mode(0o600)).unwrap();
    fs::remove_file(store_root.join("tmp/writers/recovery.lock")).unwrap();
    symlink(&outside_file, store_root.join("tmp/writers/recovery.lock")).unwrap();
    assert!(store.prepare_readers().is_err());
    assert_eq!(mode(&outside_file), 0o600);
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn complete_cozytensors_census_is_sorted_verified_and_runtime_only() {
    let root = temporary("census");
    let store = Store::ensure(&root).unwrap();
    let complete_a = model_manifest(&store, Body::Inline(vec![0; 4]), vec![1], b"missing-a");
    let complete_b = model_manifest(&store, Body::Inline(vec![1; 4]), vec![1], b"missing-b");
    let missing_part = ObjectRef::of(&vec![2; 260]);
    let _incomplete = model_manifest(
        &store,
        Body::Segments(vec![missing_part]),
        vec![65],
        b"missing-c",
    );
    let ordinary = Manifest::from_files(vec![(
        "dataset.jsonl".into(),
        ObjectRef::of(b"not resident"),
    )])
    .unwrap();
    store.put_manifest(&ordinary).unwrap();
    let corrupt = "11".repeat(32);
    let corrupt_path = store.manifest_path(&corrupt);
    fs::create_dir_all(corrupt_path.parent().unwrap()).unwrap();
    fs::write(corrupt_path, b"not canonical manifest bytes").unwrap();

    let mut expected = vec![complete_a.id(), complete_b.id()];
    expected.sort();
    assert_eq!(store.complete_cozytensors_manifests().unwrap(), expected);

    let output = Command::new(env!("CARGO_BIN_EXE_tfs"))
        .args(["store", "complete-cozytensors", root.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8(output.stdout)
            .unwrap()
            .lines()
            .map(str::to_string)
            .collect::<Vec<_>>(),
        expected
    );
    fs::remove_dir_all(root).unwrap();
}

/// THE BINDING IS A PROPERTY OF THE STORE, and this is the whole of what that claims.
///
/// It survives the process that wrote it (a pod boots, sets up once, and converts for
/// hours under a different handle in a different program); it is LOCAL, never on the cache,
/// because #612 gives the NFS root immutable Manifest and blob paths and nothing else; and
/// no shape of it can turn a Store that would open into a Store that will not, because a
/// cache is only ever allowed to change cost.
#[test]
fn the_repo_cache_binding_survives_reopen_and_can_never_refuse_a_store() {
    let root = temporary("repo-cache-binding");
    let store_root = root.join("store");
    let cache = root.join("repo-cache");
    let store = Store::init(&store_root).unwrap();
    assert!(
        store.repo_cache().is_none(),
        "a new Store is bound to nothing"
    );

    let store = store.bind_repo_cache(Some(&cache)).unwrap();
    assert_eq!(store.repo_cache().map(|c| c.root()), Some(&*cache));
    // A DIFFERENT HANDLE, made the way every later command makes one.
    assert_eq!(
        Store::open(&store_root)
            .unwrap()
            .repo_cache()
            .map(|c| c.root()),
        Some(&*cache)
    );
    assert_eq!(
        Store::ensure(&store_root)
            .unwrap()
            .repo_cache()
            .map(|c| c.root()),
        Some(&*cache)
    );
    // Recording the mount created nothing on it. `bind` performs no I/O on the cache path:
    // a stat on a hard NFS mount that has stopped answering never returns, and setup is not
    // allowed to hang on the one thing that may only cost latency.
    assert!(!cache.exists());

    // Rebinding replaces; the record is DECLARATIVE and never accumulates.
    let elsewhere = root.join("other-cache");
    let store = store.bind_repo_cache(Some(&elsewhere)).unwrap();
    assert_eq!(store.repo_cache().map(|c| c.root()), Some(&*elsewhere));
    let store = store.bind_repo_cache(None).unwrap();
    assert!(store.repo_cache().is_none());
    assert!(Store::open(&store_root).unwrap().repo_cache().is_none());
    // Clearing a binding that is already absent is not an error.
    store.bind_repo_cache(None).unwrap();

    // A cache INSIDE the Store, or a Store inside the cache, is refused at bind time —
    // where it is a caller's mistake and not weather.
    for overlap in [store_root.clone(), store_root.join("blobs"), root.clone()] {
        assert_eq!(
            Store::open(&store_root)
                .unwrap()
                .bind_repo_cache(Some(&overlap))
                .unwrap_err()
                .code,
            Code::STORE_ERA,
            "{} was accepted as a cache root",
            overlap.display()
        );
    }
    assert_eq!(
        Store::open(&store_root)
            .unwrap()
            .bind_repo_cache(Some(Path::new("relative/cache")))
            .unwrap_err()
            .code,
        Code::KEY_GRAMMAR
    );

    // AND THE PART THAT MATTERS MOST: garbage where the binding lives is NO BINDING, never
    // a refusal. A Store that would not open because a cache record was unreadable would be
    // a cache failure that cost availability, which is exactly what #612 forbids.
    let binding = store_root.join("repo-cache");
    for junk in [
        b"not-absolute\n".to_vec(),
        b"\n".to_vec(),
        Vec::new(),
        vec![0u8; 8],
        vec![b'/'; 9000],
    ] {
        fs::write(&binding, &junk).unwrap();
        let reopened = Store::open(&store_root).unwrap();
        assert!(
            reopened.repo_cache().is_none(),
            "{junk:?} was read as a mount point"
        );
        assert!(!reopened.objects().unwrap().is_empty() || reopened.objects().is_ok());
    }
    // A symlink where the binding lives is read through O_NOFOLLOW, so it is not a binding
    // either — and still not a refusal.
    fs::remove_file(&binding).unwrap();
    symlink(&cache, &binding).unwrap();
    assert!(Store::open(&store_root).unwrap().repo_cache().is_none());
    fs::remove_file(&binding).unwrap();

    // The binding is a plain local file beside the namespaces, readable by the pod's other
    // uid, and nothing about it reaches the cache.
    Store::open(&store_root)
        .unwrap()
        .bind_repo_cache(Some(&cache))
        .unwrap();
    assert_eq!(mode(&binding), 0o644);
    assert!(!cache.exists());
    let _ = fs::remove_dir_all(root);
}

// Run this arm as root as well as in the ordinary suite. Separate CLI processes
// exercise the same public Store boundary used by broker writes and Host downloads.
#[test]
fn privileged_writer_preserves_the_store_owners_public_directories() {
    use std::io::Write;
    use std::os::unix::fs::{chown, MetadataExt};
    use std::os::unix::process::CommandExt;
    use std::process::{Output, Stdio};

    let root = temporary("writer-owner");
    fs::create_dir_all(&root).unwrap();
    if fs::metadata(&root).unwrap().uid() != 0 {
        eprintln!("multi-uid arm requires root; CI runs this exact test through sudo too");
        fs::remove_dir_all(root).unwrap();
        return;
    }
    fs::set_permissions(&root, fs::Permissions::from_mode(0o755)).unwrap();
    // CI workspaces may have owner-only ancestors. The child identities need an
    // executable in this fixture's traversable root, not access to the runner home.
    let executable = root.join("tfs");
    fs::copy(env!("CARGO_BIN_EXE_tfs"), &executable).unwrap();
    fs::set_permissions(&executable, fs::Permissions::from_mode(0o755)).unwrap();
    // Final inode ownership alone misses SQLite's open -> fchown window. The
    // subprocess probe rejects a root-owned WAL/SHM inode at that exact boundary.
    // Root CI compiles the checked-in probe; isolated root-container proofs can
    // supply the same prebuilt library without requiring a compiler in the image.
    #[cfg(target_os = "linux")]
    let owner_probe = if let Some(probe) = std::env::var_os("TFS_TEST_CATALOG_WAL_PROBE") {
        PathBuf::from(probe)
    } else {
        let probe = root.join("catalog-owner-probe.so");
        let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/catalog_owner_probe.c");
        let compiled = Command::new("cc")
            .args(["-shared", "-fPIC", "-O2", "-Wall", "-Wextra"])
            .arg(&source)
            .args(["-ldl", "-o"])
            .arg(&probe)
            .output()
            .unwrap();
        assert!(
            compiled.status.success(),
            "{}",
            String::from_utf8_lossy(&compiled.stderr)
        );
        probe
    };
    let store_root = root.join("store");
    fs::create_dir(&store_root).unwrap();
    let uid = 65530;
    chown(&store_root, Some(uid), Some(uid)).unwrap();
    let run = |args: &[&str], as_uid: u32| -> Output {
        let mut command = Command::new(&executable);
        command.args(args).uid(as_uid).gid(as_uid);
        #[cfg(target_os = "linux")]
        command.env("LD_PRELOAD", &owner_probe);
        command.output().unwrap()
    };
    let ok = |out: Output| {
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
    };
    ok(run(&["store", "init", store_root.to_str().unwrap()], uid));
    ok(run(
        &["store", "prepare-readers", store_root.to_str().unwrap()],
        uid,
    ));
    let store = Store::open(&store_root).unwrap();
    let root_bytes = b"privileged writer payload";
    let root_ref = ObjectRef::of(root_bytes);
    let mut matching = Vec::new();
    for i in 0u64.. {
        let raw = format!("owner payload {i}").into_bytes();
        if ObjectRef::of(&raw).sha256[..4] == root_ref.sha256[..4] {
            matching.push(raw);
            if matching.len() == 2 {
                break;
            }
        }
    }
    let owner_bytes = &matching[0];
    store
        .put_stream(&mut root_bytes.as_slice(), None, &Fault::default())
        .unwrap();
    let path = store.blob_path(&root_ref.sha256);
    let directory = path.parent().unwrap();
    let before = fs::metadata(&path).unwrap();
    assert_eq!(before.uid(), 0); // immutable payload ownership is not rewritten
    assert_eq!(mode(&path), 0o444);
    assert_eq!(fs::metadata(directory).unwrap().uid(), uid);
    assert_eq!(fs::metadata(directory).unwrap().gid(), uid);
    assert_eq!(mode(directory), 0o755);
    let inode = fs::metadata(directory).unwrap().ino();
    let input = root.join("owner-input");
    fs::write(&input, owner_bytes).unwrap();
    ok(run(
        &["put", store_root.to_str().unwrap(), input.to_str().unwrap()],
        uid,
    ));
    assert_eq!(fs::metadata(directory).unwrap().ino(), inode);
    let after = fs::metadata(&path).unwrap();
    assert_eq!(
        (
            before.ino(),
            before.uid(),
            before.gid(),
            before.mode(),
            before.len(),
            before.mtime(),
            before.mtime_nsec()
        ),
        (
            after.ino(),
            after.uid(),
            after.gid(),
            after.mode(),
            after.len(),
            after.mtime(),
            after.mtime_nsec()
        )
    );
    assert_eq!(fs::read(&path).unwrap(), root_bytes);

    // Concurrent owners compete for previously absent first-level namespaces. The
    // parent-directory lock covers mkdir/owner assignment, never their payload reads.
    let mut inputs = Vec::new();
    let concurrent_prefix = ObjectRef::of(b"new concurrent shard").sha256[..4].to_string();
    assert_ne!(concurrent_prefix, root_ref.sha256[..4]);
    for i in 0u64.. {
        let raw = format!("concurrent directory fixture {i}");
        if ObjectRef::of(raw.as_bytes()).sha256[..4] != concurrent_prefix {
            continue;
        }
        let n = inputs.len();
        let file = root.join(format!("concurrent-{n}"));
        fs::write(&file, raw).unwrap();
        inputs.push(file);
        if inputs.len() == 12 {
            break;
        }
    }
    let mut children = Vec::new();
    for (n, file) in inputs.iter().enumerate() {
        let mut command = Command::new("/bin/sh");
        command.args([
            "-c",
            "read gate; exec \"$@\"",
            "owner-proof",
            executable.to_str().unwrap(),
            "put",
            store_root.to_str().unwrap(),
            file.to_str().unwrap(),
        ]);
        #[cfg(target_os = "linux")]
        command.env("LD_PRELOAD", &owner_probe);
        children.push(
            command
                .uid(if n % 2 == 0 { 0 } else { uid })
                .gid(if n % 2 == 0 { 0 } else { uid })
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap(),
        );
    }
    for child in &mut children {
        child.stdin.take().unwrap().write_all(b"go\n").unwrap();
    }
    for child in children {
        ok(child.wait_with_output().unwrap());
    }
    for first in fs::read_dir(store_root.join("blobs")).unwrap() {
        let first = first.unwrap().path();
        assert_eq!(fs::metadata(&first).unwrap().uid(), uid);
        for second in fs::read_dir(first).unwrap() {
            assert_eq!(fs::metadata(second.unwrap().path()).unwrap().uid(), uid);
        }
    }
    let manifest = Manifest::from_files(vec![("payload".into(), root_ref.clone())]).unwrap();
    let admitted = store.put_manifest(&manifest).unwrap();
    let manifest_path = store.manifest_path(&admitted.obj.sha256);
    assert_eq!(
        fs::metadata(manifest_path.parent().unwrap()).unwrap().uid(),
        uid
    );
    assert_eq!(mode(&manifest_path), 0o444);

    // The managed Runtime (root) publishes a repository, then the ordinary Host
    // (Store owner) reads it on the SAME Store. Every atomic replacement must
    // preserve this handoff; directory ownership alone did not protect 0600 files.
    use tensorfs_core::repository::{Mutation, ReleaseLane, RepositoryName};
    let repo = RepositoryName::new("paul", "reference-image").unwrap();
    store
        .apply_repository(
            None,
            &Mutation::PutCheckpoint {
                repo: repo.clone(),
                manifest: admitted.obj.clone(),
            },
            &Fault::default(),
        )
        .unwrap();
    let mut observed = Some(fs::read(store.repository_path(&repo)).unwrap());
    for version in ["1.0.0", "1.0.1", "1.0.2"] {
        store
            .apply_repository(
                observed.as_deref(),
                &Mutation::UpdateRelease {
                    expected_revision: 0,
                    repo: repo.clone(),
                    remove: vec![],
                    set: vec![ReleaseLane {
                        extra: Default::default(),
                        lane: "original".into(),
                        manifest: admitted.obj.clone(),
                    }],
                    version: version.into(),
                },
                &Fault::default(),
            )
            .unwrap();
        let path = store.repository_path(&repo);
        let metadata = fs::metadata(&path).unwrap();
        assert_eq!(
            (metadata.uid(), metadata.gid(), mode(&path)),
            (uid, uid, 0o600)
        );
        observed = Some(fs::read(&path).unwrap());
        let rows = store_root.join("repo-list.jsonl");
        ok(run(
            &[
                "repo",
                "list",
                store_root.to_str().unwrap(),
                "--rows",
                rows.to_str().unwrap(),
            ],
            uid,
        ));
        assert!(String::from_utf8(fs::read(rows).unwrap())
            .unwrap()
            .contains(version));
    }
    // An already-unreadable file is not silently repaired. Its typed native
    // refusal lets the Host end this preparation instead of retrying forever.
    let repo_path = store.repository_path(&repo);
    chown(&repo_path, Some(0), Some(0)).unwrap();
    let denied = run(
        &[
            "repo",
            "list",
            store_root.to_str().unwrap(),
            "--rows",
            store_root.join("denied.jsonl").to_str().unwrap(),
        ],
        uid,
    );
    assert!(!denied.status.success());
    assert!(
        String::from_utf8_lossy(&denied.stderr).contains("PERMISSION_DENIED"),
        "{}",
        String::from_utf8_lossy(&denied.stderr)
    );
    assert_eq!(fs::metadata(&repo_path).unwrap().uid(), 0);

    // Unprivileged readers cannot become public writers. The shared lease door is
    // separate from all public CAS directories, which remain owner-write only.
    assert!(!run(
        &["put", store_root.to_str().unwrap(), input.to_str().unwrap()],
        uid - 1
    )
    .status
    .success());

    // An existing unexpected owner needs an explicit repair, including when called
    // by root. Do not mutate its inode, metadata, or existing payload to make it pass.
    chown(directory, Some(uid - 1), Some(uid - 1)).unwrap();
    let wrong = fs::metadata(directory).unwrap();
    let refusal = store
        .put_stream(&mut matching[1].as_slice(), None, &Fault::default())
        .unwrap_err();
    assert_eq!(refusal.code, Code::STORE_ERA);
    assert_eq!(fs::metadata(directory).unwrap().uid(), wrong.uid());
    assert_eq!(fs::metadata(directory).unwrap().ino(), wrong.ino());
    assert_eq!(fs::read(&path).unwrap(), root_bytes);
    chown(directory, Some(uid), Some(uid)).unwrap();

    let outside = root.join("outside");
    fs::create_dir(&outside).unwrap();
    let raw = b"symlink must not redirect admission";
    let reference = ObjectRef::of(raw);
    let target = store.blob_path(&reference.sha256);
    fs::create_dir_all(target.parent().unwrap().parent().unwrap()).unwrap();
    let first = target.parent().unwrap().parent().unwrap();
    chown(first, Some(uid), Some(uid)).unwrap();
    symlink(&outside, target.parent().unwrap()).unwrap();
    assert!(store
        .put_stream(&mut raw.as_slice(), None, &Fault::default())
        .is_err());
    assert_eq!(fs::read_dir(&outside).unwrap().count(), 0);
    assert_eq!(fs::metadata(&outside).unwrap().uid(), 0);
    fs::remove_dir_all(root).unwrap();
}
