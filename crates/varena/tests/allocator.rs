//! The callback may use local completed resources while a peer is stopped under the ledger
//! heap lock. These are real processes and the OS's real range locks, not simulated lock state.
//! Unix: the set is loaded from TensorFS's sources.
#![cfg(unix)]
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};
use std::sync::{mpsc, Arc};
use std::time::Duration;

use tensorfs::format::ObjectRef;
use tensorfs::io::{ItemSource, ObjectRange, source::Source};
use varena::layout::{Item, Layout, Part, Region, REGION_ALIGN};
use varena::{Fill, Fingerprint, GpuId, HostBackend, Level, Memory, Plan, SetSpec};

fn gpu() -> GpuId {
    GpuId { uuid: "callback-heap-lock".into(), model: "host".into(), memory: 64 << 20 }
}

fn cold_page(dir: &std::path::Path) -> SetSpec {
    let path = dir.join("cold.bin");
    std::fs::write(&path, vec![7u8; REGION_ALIGN as usize]).unwrap();
    let object = ObjectRef { sha256: "callback-cold-page".into(), length: REGION_ALIGN };
    let layout = Layout {
        parts: vec![Part { what: "weight".into(), region: 0, offset: 0, nbytes: REGION_ALIGN }],
        regions: vec![Region { offset: 0, nbytes: REGION_ALIGN, span: REGION_ALIGN, items: vec![Item {
            offset: 0, len: REGION_ALIGN,
            source: ItemSource::Object(ObjectRange { obj: object.clone(), off: 0, len: REGION_ALIGN }), part_end: true,
        }] }], nbytes: REGION_ALIGN, digest: "callback-cold-layout".into(), names: vec![vec!["weight".into()]],
    };
    // Undeclared fills are never exported to custody: actual driver bytes are ours to free.
    let fingerprint = Fingerprint::new(&layout, &[], Fill::Undeclared, "1", "0/1", &gpu().uuid);
    let source = Source::files(vec![(object, std::fs::File::open(path).unwrap())], false, 0);
    SetSpec { name: "cold".into(), layout: Arc::new(layout), source: Arc::new(source), fingerprint }
}

struct Peer(Child);
impl Drop for Peer {
    fn drop(&mut self) { let _ = self.0.kill(); let _ = self.0.wait(); }
}

fn peer(role: &str, namespace: &str) -> Peer {
    let child = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "callback_peer", "--ignored", "--nocapture", "--test-threads=1"])
        .env("VARENA_CALLBACK_ROLE", role).env("VARENA_CALLBACK_NAMESPACE", namespace)
        .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::inherit()).spawn().unwrap();
    let mut child = Peer(child);
    let mut output = BufReader::new(child.0.stdout.take().unwrap());
    loop {
        let mut line = String::new();
        assert!(output.read_line(&mut line).unwrap() != 0, "peer {role} exited before ready");
        if line.contains("CALLBACK_READY") { break; }
    }
    child.0.stdout = Some(output.into_inner());
    child
}

/// Its claim file goes when the test does, passed or failed.
struct Claims;

impl Drop for Claims {
    fn drop(&mut self) {
        varena::tier::remove_claims();
    }
}

#[test]
#[ignore = "subprocess role, invoked by the_hooks_free_read_does_not_wait_for_a_peer_heap_lock"]
fn callback_peer() {
    let role = std::env::var("VARENA_CALLBACK_ROLE").unwrap();
    assert!(varena::os::namespace(&std::env::var("VARENA_CALLBACK_NAMESPACE").unwrap()));
    if role == "dead-row" {
        let _memory = Memory::open(gpu(), None, Box::new(HostBackend::new(gpu().memory))).unwrap();
        println!("CALLBACK_READY"); std::io::stdout().flush().unwrap();
        loop { std::thread::park(); }
    }
    assert_eq!(role, "heap-lock");
    let grant = varena::Grant::open(&gpu(), None).unwrap();
    // The ledger's heap lock: the table's first 8 bytes, through the seam the ledger takes it by.
    assert!(platform::memory::lock(&grant.table, 0, 8, true).unwrap());
    println!("CALLBACK_READY"); std::io::stdout().flush().unwrap();
    let mut release = String::new();
    std::io::stdin().read_line(&mut release).unwrap();
    assert_eq!(release.trim(), "release");
}

#[test]
fn the_hooks_free_read_does_not_wait_for_a_peer_heap_lock() {
    let namespace = format!("allocator-test-{}-", std::process::id());
    assert!(varena::os::namespace(&namespace));
    let _claims = Claims;
    let memory = Arc::new(Memory::open(gpu(), None, Box::new(HostBackend::new(gpu().memory))).unwrap());
    let dir = tempfile::tempdir().unwrap();
    let set = memory.load(cold_page(dir.path()), Level::Gpu).unwrap();
    let pin = memory.bind(set, 0, 0).unwrap();
    pin.filled();
    drop(pin);
    let request = memory.request("callback", Plan::default()).unwrap();
    let mut dead = peer("dead-row", &namespace);
    dead.0.kill().unwrap();
    assert!(!dead.0.wait().unwrap().success(), "leave an unreaped row by killing its owner");
    let mut locker = peer("heap-lock", &namespace);
    let (answer, received) = mpsc::channel();
    let hook = std::thread::spawn({
        let memory = memory.clone();
        move || {
            let _ = answer.send(memory.free());
        }
    });
    // Hang protection only. The assertion is causal: the read returned BEFORE the controller
    // releases C's real heap lock. On failure release first, then join; no child is stranded.
    let result = received.recv_timeout(Duration::from_secs(5));
    locker.0.stdin.as_mut().unwrap().write_all(b"release\n").unwrap();
    assert!(locker.0.wait().unwrap().success());
    hook.join().unwrap();
    assert!(result.is_ok(), "the hook's free read waited for a peer heap lock");
    drop(request);
}

// Append to crates/varena/tests/allocator.rs (it uses that file's gpu, cold_page, peer and Claims).

/// Reviewer (step 1): the hook's make_room, not only its free read, while a stopped peer holds
/// the ledger's heap lock. make_room refreshes the rows under that lock: the allocation waits
/// for the peer.
#[test]
fn step1_the_hooks_make_room_waits_for_a_peer_heap_lock() {
    let namespace = format!("allocator-test-{}-", std::process::id());
    assert!(varena::os::namespace(&namespace));
    let _claims = Claims;
    let memory = Arc::new(Memory::open(gpu(), None, Box::new(HostBackend::new(gpu().memory))).unwrap());
    let dir = tempfile::tempdir().unwrap();
    let set = memory.load(cold_page(dir.path()), Level::Gpu).unwrap();
    let pin = memory.bind(set, 0, 0).unwrap();
    pin.filled();
    drop(pin);
    let request = memory.request("callback", Plan::default()).unwrap();
    let mut locker = peer("heap-lock", &namespace);
    let need = memory.free() + REGION_ALIGN;
    let (answer, received) = mpsc::channel();
    let hook = std::thread::spawn({
        let memory = memory.clone();
        move || {
            let _ = answer.send(memory.make_room(need).map(drop).map_err(|e| e.code()));
        }
    });
    let result = received.recv_timeout(Duration::from_secs(5));
    locker.0.stdin.as_mut().unwrap().write_all(b"release
").unwrap();
    assert!(locker.0.wait().unwrap().success());
    let after = if result.is_err() { received.recv_timeout(Duration::from_secs(5)).ok() } else { None };
    hook.join().unwrap();
    println!("make_room with a stopped peer holding the heap lock: within 5 s {result:?}; after the release {after:?}");
    assert!(result.is_ok(), "the hook's make_room waited for a peer heap lock");
    drop(request);
}
