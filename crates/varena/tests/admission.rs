//! Two real processes must not spend the same shared cap. The driver-read barrier forces
//! the old snapshot/apply interleaving; it never substitutes for the final admission invariant.
#![cfg(target_os = "linux")]
use std::{io::{Read, Write}, os::{fd::{AsRawFd, FromRawFd}, unix::{net::UnixStream, process::CommandExt}},
    process::{Child, Command}, sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}}, time::Duration};
use varena::{Access, Backend, Event, GpuId, HostBackend, Ledger, Memory, Physical, Torch, Va};
use platform::handles::Owned;
use tensorfs::io::Dst;

const M: u64 = 1 << 20;
fn gpu() -> GpuId { GpuId { uuid: "shared-admission".into(), model: "host".into(), memory: 32 * M } }

struct Gated {
    host: HostBackend,
    armed: Arc<AtomicBool>,
    control: Mutex<UnixStream>,
}
impl Backend for Gated {
    fn capacity(&self) -> u64 { self.host.capacity() }
    fn free(&self) -> varena::Result<u64> {
        if self.armed.swap(false, Ordering::SeqCst) {
            let mut control = self.control.lock().unwrap();
            control.write_all(b"S").unwrap();
            let mut next = [0];
            control.read_exact(&mut next).unwrap();
            assert_eq!(&next, b"G");
        }
        self.host.free()
    }
    fn process_used(&self) -> varena::Result<u64> { self.host.process_used() }
    fn opened_context(&self) -> u64 { self.host.opened_context() }
    fn granularity(&self) -> u64 { self.host.granularity() }
    fn reserve(&self, len: u64) -> varena::Result<Va> { self.host.reserve(len) }
    fn release(&self, va: Va) { self.host.release(va) }
    fn create(&self, len: u64) -> varena::Result<Physical> { self.host.create(len) }
    fn destroy(&self, page: Physical) { self.host.destroy(page) }
    fn map(&self, va: Va, off: u64, page: &Physical, access: Access) -> varena::Result<()> { self.host.map(va, off, page, access) }
    fn protect(&self, va: Va, off: u64, len: u64, access: Access) -> varena::Result<()> { self.host.protect(va, off, len, access) }
    fn unmap(&self, va: Va, off: u64, len: u64) -> varena::Result<()> { self.host.unmap(va, off, len) }
    fn export(&self, page: &Physical) -> varena::Result<Owned> { self.host.export(page) }
    fn import(&self, handle: Owned, len: u64) -> varena::Result<Physical> { self.host.import(handle, len) }
    fn target(&self, va: Va, off: u64, len: u64) -> Dst { self.host.target(va, off, len) }
    fn save(&self, va: Va, off: u64, host: &mut [u8], after: Event) -> varena::Result<()> { self.host.save(va, off, host, after) }
    fn fence(&self) -> varena::Result<Event> { self.host.fence() }
    fn record(&self, stream: usize) -> varena::Result<Event> { self.host.record(stream) }
    fn query(&self, event: Event) -> varena::Result<bool> { self.host.query(event) }
    fn elapsed(&self, from: Event, to: Event) -> varena::Result<Option<u64>> { self.host.elapsed(from, to) }
    fn wait(&self, event: Event, stream: usize) -> varena::Result<()> { self.host.wait(event, stream) }
    fn synchronize(&self, stream: usize) -> varena::Result<()> { self.host.synchronize(stream) }
    fn finish(&self, event: Event) -> varena::Result<()> { self.host.finish(event) }
    fn register_host(&self, ptr: u64, len: u64) -> varena::Result<()> { self.host.register_host(ptr, len) }
    fn unregister_host(&self, ptr: u64) -> varena::Result<()> { self.host.unregister_host(ptr) }
    fn retire(&self, event: Event) { self.host.retire(event) }
    fn drain(&self) -> varena::Result<()> { self.host.drain() }
    fn drain_save(&self) -> varena::Result<()> { self.host.drain_save() }
}

#[test]
#[ignore = "child role for concurrent_workspace_admissions_share_one_cap"]
fn admission_child() {
    let namespace = std::env::var("VARENA_ADMISSION_NAMESPACE").unwrap();
    assert!(varena::os::namespace(&namespace));
    // SAFETY: the parent passed this one descriptor across exec specifically for this role.
    let mut control = unsafe { UnixStream::from_raw_fd(3) };
    control.set_read_timeout(Some(Duration::from_secs(15))).unwrap();
    let armed = Arc::new(AtomicBool::new(false));
    let memory = Memory::open(gpu(), None, Box::new(Gated {
        host: HostBackend::new(gpu().memory), armed: armed.clone(), control: Mutex::new(control.try_clone().unwrap()),
    })).unwrap();
    control.write_all(b"R").unwrap();
    let mut next = [0];
    control.read_exact(&mut next).unwrap();
    assert_eq!(&next, b"B");
    armed.store(true, Ordering::SeqCst);
    let claim = memory.workspace(6 * M, Torch::default());
    control.write_all(if claim.is_ok() { b"Y" } else { b"N" }).unwrap();
    control.read_exact(&mut next).unwrap();
    assert_eq!(&next, b"E");
    drop(claim);
    drop(memory);
}

struct Peer { child: Child, control: UnixStream }
impl Drop for Peer {
    fn drop(&mut self) { let _ = self.child.kill(); let _ = self.child.wait(); }
}
impl Peer {
    fn spawn(namespace: &str) -> Self {
        let (control, inherited) = UnixStream::pair().unwrap();
        control.set_read_timeout(Some(Duration::from_secs(15))).unwrap();
        let fd = inherited.as_raw_fd();
        let mut command = Command::new(std::env::current_exe().unwrap());
        command.args(["admission_child", "--exact", "--ignored", "--nocapture", "--test-threads=1"])
            .env("VARENA_ADMISSION_NAMESPACE", namespace);
        // SAFETY: only async-signal-safe descriptor operations between fork and exec.
        unsafe { command.pre_exec(move || {
            if libc::dup2(fd, 3) < 0 || libc::fcntl(3, libc::F_SETFD, 0) < 0 { return Err(std::io::Error::last_os_error()); }
            Ok(())
        }); }
        let child = command.spawn().unwrap();
        drop(inherited);
        Self { child, control }
    }
    fn receive(&mut self) -> u8 {
        let mut answer = [0];
        self.control.read_exact(&mut answer).expect("child failed or blocked before the expected phase");
        answer[0]
    }
    fn send(&mut self, message: u8) { self.control.write_all(&[message]).unwrap(); }
}

#[test]
fn concurrent_workspace_admissions_share_one_cap() {
    let namespace = format!("admission-test-{}-", std::process::id());
    assert!(varena::os::namespace(&namespace));
    let ledger = Ledger::create(&gpu(), Some(8 * M)).unwrap();
    let mut first = Peer::spawn(&namespace);
    let mut second = Peer::spawn(&namespace);
    assert_eq!(first.receive(), b'R');
    assert_eq!(second.receive(), b'R');
    first.send(b'B'); second.send(b'B');
    // Both callbacks are paused at a real driver sampling boundary. The old implementation
    // has already snapshotted peer rows here; the corrected gate must resnapshot afterward.
    assert_eq!(first.receive(), b'S');
    assert_eq!(second.receive(), b'S');
    first.send(b'G'); second.send(b'G');
    let answers = [first.receive(), second.receive()];
    let admitted = answers.iter().filter(|a| **a == b'Y').count();
    let status = ledger.status();
    let promised: u64 = status.rows.iter().map(|r| r.torch).sum();
    // Cleanly end both roles before asserting, including the known red case. Socket deadlines
    // only fail a stuck test; they never authorize a claim or make this invariant pass.
    first.send(b'E'); second.send(b'E');
    assert!(first.child.wait().unwrap().success());
    assert!(second.child.wait().unwrap().success());
    assert!(admitted <= 1 && promised <= status.capacity,
        "two independent admissions spent one shared cap: answers={answers:?}, admitted={admitted}, promised={promised}, cap={}", status.capacity);
}
