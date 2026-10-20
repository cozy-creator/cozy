//! Real processes, OFD claims, eventfd wakeups and descriptor custody. No CUDA emulator.
#![cfg(target_os = "linux")]

use std::io::{BufRead, BufReader, Read, Write};
use std::os::fd::{AsFd, AsRawFd, FromRawFd};
use std::os::unix::net::UnixStream;
use std::os::unix::process::CommandExt;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use varena::ledger::{Channel, Grant, Ledger, Owner, PageKey, Running, Snapshot, State};
use varena::plan::Class;

const CAPACITY: u64 = 1024;
const KEY: PageKey = PageKey {
    fingerprint: [7; 32],
    page: 1,
};
static NEXT: AtomicU64 = AtomicU64::new(0);

struct Fixture {
    ledger: Ledger,
    name: String,
}
impl Fixture {
    fn new() -> Self {
        let name = format!("test-{}-{}", std::process::id(), NEXT.fetch_add(1, Relaxed));
        Self {
            ledger: Ledger::create(&name, CAPACITY).unwrap(),
            name,
        }
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        // Only this test's uniquely named ledger. The mapping remains valid until drop.
        let uid = unsafe { libc::geteuid() };
        let _ = std::fs::remove_file(format!("/dev/shm/varena-{}-{uid}", self.name));
    }
}
struct ChildRow {
    process: Child,
    input: ChildStdin,
    output: BufReader<ChildStdout>,
    owner: Owner,
}
impl ChildRow {
    fn spawn(ledger: &Ledger) -> Self {
        // grant authenticates the row's PID after exec; the PID is not known before spawn.
        let (ours, theirs) = UnixStream::pair().unwrap();
        let inherited = theirs.as_raw_fd();
        let mut cmd = Command::new(std::env::current_exe().unwrap());
        cmd.args([
            "--exact",
            "child",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped());
        unsafe {
            cmd.pre_exec(move || {
                if libc::dup2(inherited, 9) < 0 {
                    return Err(std::io::Error::last_os_error());
                }
                // dup2 clears CLOEXEC only when the descriptors differ. Under
                // parallel tests the socket itself can already be descriptor9.
                if libc::fcntl(9, libc::F_SETFD, 0) < 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
        let mut process = cmd.spawn().unwrap();
        drop(theirs);
        let Grant {
            table,
            events,
            channel,
            ..
        } = ledger.grant(process.id()).unwrap();
        for fd in [&table, &events, &channel] {
            platform::handles::send_fd(&ours, fd).unwrap();
        }
        drop((table, events, channel, ours));
        let input = process.stdin.take().unwrap();
        let output = BufReader::new(process.stdout.take().unwrap());
        let mut child = Self {
            process,
            input,
            output,
            owner: Owner {
                row: 0,
                generation: 0,
            },
        };
        let fields = child.reply();
        let values: Vec<_> = fields.split_whitespace().collect();
        child.owner = Owner {
            row: values[0].parse().unwrap(),
            generation: values[1].parse().unwrap(),
        };
        child
    }
    fn reply(&mut self) -> String {
        loop {
            let mut line = String::new();
            assert!(
                self.output.read_line(&mut line).unwrap() > 0,
                "child exited early"
            );
            if let Some((_, reply)) = line.split_once('@') {
                return reply.trim().to_owned();
            }
        }
    }
    fn send(&mut self, command: &str) {
        writeln!(self.input, "{command}").unwrap();
        self.input.flush().unwrap();
    }
    fn call(&mut self, command: &str) -> String {
        self.send(command);
        self.reply()
    }
    fn kill(&mut self) {
        self.process.kill().unwrap();
        self.process.wait().unwrap();
    }
}
impl Drop for ChildRow {
    fn drop(&mut self) {
        let _ = self.process.kill();
        let _ = self.process.wait();
    }
}

#[test]
#[ignore]
fn child() {
    let control = unsafe { UnixStream::from_raw_fd(9) };
    let table = platform::handles::recv_fd(&control).unwrap();
    let events = platform::handles::recv_fd(&control).unwrap();
    let channel = Channel::new(platform::handles::recv_fd(&control).unwrap());
    let ledger = Ledger::adopt(table.as_fd(), events.as_fd()).unwrap();
    let row = ledger.claim(std::process::id()).unwrap();
    channel.bind(row.owner()).unwrap();
    println!("@{} {}", row.owner().row, row.owner().generation);
    let mut containers = Vec::new();
    for command in std::io::stdin().lock().lines().map(Result::unwrap) {
        let args: Vec<_> = command.split_whitespace().collect();
        match args[0] {
            "publish" => {
                let n = args[1].parse().unwrap();
                row.publish(
                    &Snapshot {
                        classes: [n; 7],
                        context: n,
                        requests: Vec::new(),
                    },
                    n,
                )
                .unwrap();
                println!("@ok");
            }
            "running" => {
                let left = args[1].parse().ok();
                row.publish(
                    &Snapshot {
                        classes: [0; 7],
                        context: 16,
                        requests: vec![Running {
                            id: 42,
                            remaining: left,
                        }],
                    },
                    1,
                )
                .unwrap();
                println!("@ok");
            }
            "ask" => {
                let ask = row
                    .wait_ask(&std::sync::atomic::AtomicBool::new(false))
                    .unwrap()
                    .unwrap();
                row.ack(ask).unwrap();
                println!("@{} {}", ask.ceiling.as_usize(), ask.bytes);
            }
            "keep" => {
                let mut file =
                    std::fs::File::from(platform::shm::memfd_named("ledger-test-page").unwrap());
                file.write_all(b"page-data").unwrap();
                let (tx, rx) = UnixStream::pair().unwrap();
                platform::handles::send_fd(&tx, &file).unwrap();
                drop(tx);
                assert!(channel.keep(row.owner(), KEY, &[9], rx.as_fd()).unwrap());
                containers.push(rx);
                println!("@ok");
            }
            "unmap" => {
                for rx in containers.drain(..) {
                    drop(platform::handles::recv_fd(&rx).unwrap());
                }
                println!("@ok");
            }
            "take" | "wrong" => {
                let key = if args[0] == "wrong" {
                    PageKey {
                        fingerprint: [8; 32],
                        ..KEY
                    }
                } else {
                    KEY
                };
                let handles = channel.take(row.owner(), key).unwrap();
                let mut bytes = String::new();
                for (span, fd) in &handles {
                    assert_eq!(*span, 9);
                    let mut file = std::fs::File::from(fd.try_clone().unwrap());
                    std::io::Seek::rewind(&mut file).unwrap();
                    file.read_to_string(&mut bytes).unwrap();
                }
                println!("@{} {}", handles.len(), bytes);
            }
            _ => panic!("unknown test command"),
        }
        std::io::stdout().flush().unwrap();
    }
}

#[test]
fn killed_claim_is_reaped_without_touching_another_process() {
    let fixture = Fixture::new();
    let mut killed = ChildRow::spawn(&fixture.ledger);
    let mut alive = ChildRow::spawn(&fixture.ledger);
    assert_eq!(killed.call("publish 1"), "ok");
    assert_eq!(alive.call("publish 2"), "ok");
    killed.kill();
    fixture.ledger.reap(killed.process.id()).unwrap();
    let rows = fixture.ledger.status().unwrap().rows;
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].owner, alive.owner);
    assert_eq!(rows[0].snapshot.classes, [2; 7]);
    assert_eq!(rows[0].state, State::Live);
    assert!(fixture.ledger.reap(alive.process.id()).is_err());
}

#[test]
fn class_ask_is_exact_and_old_generations_cannot_target_a_reused_row() {
    let fixture = Fixture::new();
    let mut child = ChildRow::spawn(&fixture.ledger);
    child.send("ask");
    let ask = fixture
        .ledger
        .ask(child.owner, 12, Class::Declared)
        .unwrap();
    assert_eq!(child.reply(), "4 12");
    assert!(fixture.ledger.answered(child.owner, ask).unwrap());
    child.kill();
    fixture.ledger.reap(child.process.id()).unwrap();
    let next = ChildRow::spawn(&fixture.ledger);
    assert_ne!(next.owner, child.owner);
    assert!(fixture.ledger.ask(child.owner, 1, Class::Unused).is_err());
}

#[test]
fn unmeasured_request_runs_alone_then_measured_admission_uses_remaining_sets() {
    let fixture = Fixture::new();
    assert!(fixture.ledger.fits("package", "new-shape").unwrap());
    let mut child = ChildRow::spawn(&fixture.ledger);
    child.call("running unknown");
    assert!(!fixture.ledger.fits("package", "new-shape").unwrap());
    fixture.ledger.record_peak("package", "shape", 400).unwrap();
    child.call("running 600");
    assert!(fixture.ledger.fits("package", "shape").unwrap());
    child.call("running 700");
    assert!(!fixture.ledger.fits("package", "shape").unwrap());
    assert!(fixture.ledger.stalled().unwrap().is_empty());
}

#[test]
fn named_table_and_measured_peaks_survive_coordinator_restart() {
    let fixture = Fixture::new();
    let mut child = ChildRow::spawn(&fixture.ledger);
    fixture.ledger.record_peak("package", "shape", 600).unwrap();
    let restarted = Ledger::create(&fixture.name, CAPACITY).unwrap();
    child.call("running 500");
    assert!(!restarted.fits("package", "shape").unwrap());
    assert_eq!(restarted.status().unwrap().rows[0].owner, child.owner);
    assert_eq!(restarted.peaks(), vec![("package|shape".into(), 600)]);
}

#[test]
fn a_new_cap_requires_every_executor_to_release_its_claim() {
    let fixture = Fixture::new();
    fixture.ledger.record_peak("package", "shape", 900).unwrap();
    let mut child = ChildRow::spawn(&fixture.ledger);
    assert!(Ledger::create(&fixture.name, CAPACITY / 2).is_err());
    child.kill();
    fixture.ledger.reap(child.process.id()).unwrap();
    let resized = Ledger::create(&fixture.name, CAPACITY / 2).unwrap();
    assert_eq!(resized.capacity(), CAPACITY / 2);
    assert_eq!(fixture.ledger.capacity(), CAPACITY / 2);
    assert!(resized.peaks().is_empty());
}

#[test]
fn a_normal_close_can_reopen_without_reaping_the_live_process() {
    let fixture = Fixture::new();
    let row = fixture.ledger.claim(std::process::id()).unwrap();
    let old = row.owner();
    drop(row);
    assert!(fixture.ledger.status().unwrap().rows.is_empty());
    let next = fixture.ledger.claim(std::process::id()).unwrap();
    assert_ne!(next.owner(), old);
    assert!(fixture.ledger.ask(old, 1, Class::Unused).is_err());
    assert_eq!(fixture.ledger.status().unwrap().rows.len(), 1);
}

#[test]
fn killed_exporter_leaves_one_importable_page_and_live_eviction_leaves_none() {
    let fixture = Fixture::new();
    let mut first = ChildRow::spawn(&fixture.ledger);
    first.call("keep");
    first.kill();
    fixture.ledger.reap(first.process.id()).unwrap();
    let mut second = ChildRow::spawn(&fixture.ledger);
    assert_eq!(second.call("wrong"), "0");
    assert_eq!(second.call("take"), "1 page-data");
    assert_eq!(second.call("take"), "0");
    second.call("keep");
    second.call("unmap");
    second.kill();
    fixture.ledger.reap(second.process.id()).unwrap();
    let mut third = ChildRow::spawn(&fixture.ledger);
    assert_eq!(third.call("take"), "0");
}

#[test]
fn ram_service_fills_through_tensorfs_and_retains_a_readers_layout() {
    use std::sync::Arc;
    use tensorfs::io::{source::Item, ItemSource, ObjectRange};
    use varena::ledger::RamRegion;
    let fixture = Fixture::new();
    let row = fixture.ledger.claim(std::process::id()).unwrap();
    let grant = fixture.ledger.grant(std::process::id()).unwrap();
    let channel = Channel::new(grant.channel);
    let layout = [9; 32];
    let object = tensorfs::format::ObjectRef {
        sha256: "3a6eb0790f39ac87c94f3856b2dd2c5d110e6811602261a9a923d3bb23adc8b7".into(),
        length: 4,
    };
    let files = vec![(
        object.clone(),
        platform::shm::frozen("RAM-source-test", b"data")
            .unwrap()
            .into(),
    )];
    let regions = vec![
        RamRegion {
            offset: 0,
            len: 4,
            items: vec![Item {
                offset: 0,
                len: 4,
                source: ItemSource::Inline(Arc::from(*b"data")),
                part_end: true,
            }],
        },
        RamRegion {
            offset: 4,
            len: 4,
            items: vec![Item {
                offset: 4,
                len: 4,
                source: ItemSource::Object(ObjectRange {
                    obj: object,
                    off: 0,
                    len: 4,
                }),
                part_end: true,
            }],
        },
    ];
    assert!(channel
        .cache_ram(row.owner(), layout, &files, &regions)
        .unwrap());
    let view = loop {
        if let Some(view) = channel.ram(row.owner(), layout, 1).unwrap() {
            break view;
        }
        let mut poll = libc::pollfd {
            fd: fixture.ledger.events().as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        };
        assert_eq!(
            unsafe { libc::poll(&mut poll, 1, 3000) },
            1,
            "RAM worker did not signal progress"
        );
        let mut bytes = [0u8; 8];
        let read = unsafe {
            libc::read(
                fixture.ledger.events().as_raw_fd(),
                bytes.as_mut_ptr().cast(),
                bytes.len(),
            )
        };
        assert_eq!(read, 8);
    };
    assert_eq!((view.offset, view.len), (4, 4));
    let mut data = String::new();
    std::fs::File::from(view.file.try_clone().unwrap())
        .read_to_string(&mut data)
        .unwrap();
    assert_eq!(data, "datadata");
    assert!(!fixture.ledger.release_ram(layout));
    drop(view);
    // Region readiness can precede completion of the final fill bookkeeping.
    // A busy return is followed by the ledger's completion wake, not a timer poll.
    while !fixture.ledger.release_ram(layout) {
        let mut pending = libc::pollfd {
            fd: fixture.ledger.events().as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        };
        assert_eq!(
            unsafe { libc::poll(&mut pending, 1, 3000) },
            1,
            "fill did not finish"
        );
        let mut word = [0u8; 8];
        assert_eq!(
            unsafe { libc::read(pending.fd, word.as_mut_ptr().cast(), word.len()) },
            8
        );
    }
}

#[test]
fn reopened_coordinator_gets_real_eventfd_wakeups() {
    let fixture = Fixture::new();
    let mut child = ChildRow::spawn(&fixture.ledger);
    let reopened = Ledger::create(&fixture.name, CAPACITY).unwrap();
    child.call("publish 3");
    let mut poll = libc::pollfd {
        fd: reopened.events().as_raw_fd(),
        events: libc::POLLIN,
        revents: 0,
    };
    assert_eq!(
        unsafe { libc::poll(&mut poll, 1, 3000) },
        1,
        "reopened coordinator missed shared progress"
    );
    assert_eq!(reopened.status().unwrap().rows[0].snapshot.classes, [3; 7]);
}
