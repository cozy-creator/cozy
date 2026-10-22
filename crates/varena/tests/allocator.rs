//! The callback may use local completed resources while a peer is stopped under the ledger
//! heap lock. These are real processes/OFD locks, not simulated lock state.
use std::io::{BufRead, BufReader, Write};
use std::os::fd::AsRawFd;
use std::process::{Child, Command, Stdio};
use std::sync::{mpsc, Arc};
use std::time::Duration;

use tensorfs::format::ObjectRef;
use tensorfs::io::{ItemSource, ObjectRange, source::Source};
use varena::layout::{Item, Layout, Part, Region, REGION_ALIGN};
use varena::{Fill, Fingerprint, GpuId, HostBackend, Level, Memory, Plan, SetSpec, Torch};

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

#[test]
#[ignore = "subprocess role, invoked by callback_does_not_wait_for_a_peer_heap_lock"]
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
    let mut lock: libc::flock = unsafe { std::mem::zeroed() };
    lock.l_type = libc::F_WRLCK as i16;
    lock.l_whence = libc::SEEK_SET as i16;
    lock.l_start = 0; lock.l_len = 8;
    assert_eq!(unsafe { libc::fcntl(grant.table.as_raw_fd(), libc::F_OFD_SETLKW, &lock) }, 0);
    println!("CALLBACK_READY"); std::io::stdout().flush().unwrap();
    let mut release = String::new();
    std::io::stdin().read_line(&mut release).unwrap();
    assert_eq!(release.trim(), "release");
}

#[test]
fn callback_does_not_wait_for_a_peer_heap_lock() {
    let namespace = format!("allocator-test-{}-", std::process::id());
    assert!(varena::os::namespace(&namespace));
    let memory = Memory::open(gpu(), None, Box::new(HostBackend::new(gpu().memory))).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let set = memory.load(cold_page(dir.path()), Level::Gpu).unwrap();
    let pin = memory.bind(set, 0, 0).unwrap();
    pin.filled();
    drop(pin);
    let request = Arc::new(memory.request("callback", Plan::default()).unwrap());
    memory.observe(Torch::default()).unwrap();
    let mut dead = peer("dead-row", &namespace);
    dead.0.kill().unwrap();
    assert!(!dead.0.wait().unwrap().success(), "leave an unreaped row by killing its owner");
    let mut locker = peer("heap-lock", &namespace);
    let (answer, received) = mpsc::channel();
    let active = request.clone();
    let callback = std::thread::spawn(move || {
        let until = std::time::Instant::now() + Duration::from_secs(1);
        let result = loop {
            let result = active.reclaim_allocation(None, gpu().memory, Torch::default());
            if !matches!(result, Ok(0)) || std::time::Instant::now() >= until { break result; }
            std::thread::yield_now();
        };
        let _ = answer.send(result);
    });
    // Hang protection only. The assertion is causal: callback returned BEFORE the controller
    // releases C's real heap lock. On failure release first, then join; no child is stranded.
    let result = received.recv_timeout(Duration::from_secs(5));
    locker.0.stdin.as_mut().unwrap().write_all(b"release\n").unwrap();
    assert!(locker.0.wait().unwrap().success());
    callback.join().unwrap();
    assert!(result.is_ok(), "native callback waited for a peer heap lock");
    assert!(result.unwrap().unwrap() >= REGION_ALIGN, "the callback actually reclaimed a completed local page while C still held the peer heap lock");
    drop(request);
    drop(memory);
}
