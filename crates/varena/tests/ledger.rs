//! Real processes, OFD claims, eventfd wakeups and descriptor custody. No CUDA emulator.
#![cfg(target_os = "linux")]

use std::io::{BufRead, BufReader, Read, Write};
use std::os::fd::{AsFd, AsRawFd, FromRawFd};
use std::os::unix::fs::FileExt;
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
        Self::with_host(std::sync::Arc::new(varena::ledger::HostCache::default()))
    }
    fn with_host(host: std::sync::Arc<varena::ledger::HostCache>) -> Self {
        let name = format!("test-{}-{}", std::process::id(), NEXT.fetch_add(1, Relaxed));
        Self {
            ledger: Ledger::create_with_host(&name, CAPACITY, host).unwrap(),
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
            "limit" => {
                let capacity = args[1].parse().unwrap();
                println!("@{}", ledger.limit_capacity(capacity).unwrap());
            }
            "capacity" => println!("@{}", ledger.capacity()),
            "capacity_event" => {
                row.wait_event(&mut 0, &std::sync::atomic::AtomicBool::new(false))
                    .unwrap();
                println!("@{}", ledger.capacity());
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
            "warm" => {
                let worker = channel.fork(row.owner()).unwrap();
                worker
                    .cache_ram(
                        row.owner(),
                        [19; 32],
                        &[],
                        &[varena::ledger::RamRegion {
                            offset: 0,
                            len: 4,
                            items: vec![tensorfs::io::source::Item {
                                offset: 0,
                                len: 4,
                                part_end: true,
                                source: tensorfs::io::ItemSource::Inline(b"warm".to_vec().into()),
                            }],
                        }],
                    )
                    .unwrap();
                println!("@ok");
            }
            "forget_ram" => {
                channel.forget_ram(row.owner(), [19; 32], 0).unwrap();
                println!("@ok");
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
fn restart_collects_unknown_dead_rows_without_losing_peaks_or_existing_custody() {
    let fixture = Fixture::new();
    fixture.ledger.record_peak("package", "shape", 800).unwrap();
    let mut child = ChildRow::spawn(&fixture.ledger);
    child.call("publish 300");
    child.call("keep");
    child.kill();
    assert_eq!(fixture.ledger.status().unwrap().rows[0].state, State::Dead);
    let restarted = Ledger::create(&fixture.name, CAPACITY).unwrap();
    assert!(
        restarted.status().unwrap().rows.is_empty(),
        "startup must collect owners it never tracked"
    );
    assert!(restarted.fits("package", "shape").unwrap());
    assert_eq!(restarted.peaks(), vec![("package|shape".into(), 800)]);
    // A second coordinator's collection must not destroy another service's
    // still-held page containers. Reuse recognizes the old exact owner as orphaned.
    let mut next = ChildRow::spawn(&fixture.ledger);
    assert_eq!(next.owner.row, child.owner.row);
    assert!(next.owner.generation > child.owner.generation);
    assert_eq!(next.call("take"), "1 page-data");
}

#[test]
fn admission_collects_an_unknown_owner_that_dies_after_restart() {
    let fixture = Fixture::new();
    let mut child = ChildRow::spawn(&fixture.ledger);
    child.call("running unknown");
    let restarted = Ledger::create(&fixture.name, CAPACITY).unwrap();
    assert!(!restarted.fits("package", "shape").unwrap());
    child.kill();
    assert!(restarted.fits("package", "shape").unwrap());
    assert!(restarted.status().unwrap().rows.is_empty());
}

#[test]
fn live_kernel_claims_survive_restart_and_pid_reuse_even_with_nonexistent_pid_metadata() {
    let fixture = Fixture::new();
    let pid = u32::MAX;
    let mut old = ChildRow::spawn(&fixture.ledger);
    old.kill();
    fixture.ledger.reap(old.process.id()).unwrap();
    let mut live = ChildRow::spawn(&fixture.ledger);
    assert_eq!(live.owner.row, old.owner.row);
    assert!(live.owner.generation > old.owner.generation);
    live.call("publish 300");
    // Change only this uniquely named fixture's PID hint, after its child has
    // claimed normally. The checked Slot ABI keeps pid at header64 + row576 +8.
    // The live child's independent OFD must win over misleading PID metadata.
    let table = std::fs::File::from(fixture.ledger.fd().try_clone_to_owned().unwrap());
    table
        .write_all_at(&pid.to_ne_bytes(), 64 + u64::from(live.owner.row) * 576 + 8)
        .unwrap();
    fixture.ledger.record_peak("package", "shape", 800).unwrap();
    let restarted = Ledger::create(&fixture.name, CAPACITY).unwrap();
    assert_eq!(restarted.status().unwrap().rows[0].state, State::Live);
    std::thread::scope(|scope| {
        for _ in 0..2 {
            let ledger = &restarted;
            scope.spawn(move || {
                for _ in 0..32 {
                    assert!(
                        ledger.reap(pid).is_err(),
                        "a PID hint cannot override a live OFD claim"
                    );
                    assert!(!ledger.fits("package", "shape").unwrap());
                }
            });
        }
        assert_eq!(live.call("publish 300"), "ok");
    });
    let rows = restarted.status().unwrap().rows;
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].owner, live.owner);
    assert_eq!(rows[0].snapshot.classes, [300; 7]);
    assert_eq!(rows[0].pid, pid);
    assert_eq!(rows[0].state, State::Live);
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
fn a_child_corrects_shared_capacity_and_wakes_owners_and_admission() {
    let fixture = Fixture::new();
    let mut owner = ChildRow::spawn(&fixture.ledger);
    let mut correcting = ChildRow::spawn(&fixture.ledger);
    fixture.ledger.record_peak("package", "shape", 400).unwrap();
    owner.call("running 500");
    assert!(fixture.ledger.fits("package", "shape").unwrap());

    // Its event bridge starts after all earlier publications, so this eventfd
    // wake must follow the child's clamp rather than an earlier snapshot.
    let reopened = Ledger::create(&fixture.name, CAPACITY).unwrap();
    owner.send("capacity_event");
    assert_eq!(correcting.call("limit 900"), "900");
    let mut ready = libc::pollfd {
        fd: owner.output.get_ref().as_raw_fd(),
        events: libc::POLLIN,
        revents: 0,
    };
    assert_eq!(unsafe { libc::poll(&mut ready, 1, 3000) }, 1, "owner wake");
    assert_eq!(
        owner.reply(),
        "900",
        "an existing owner missed its row wake"
    );
    let mut poll = libc::pollfd {
        fd: reopened.events().as_raw_fd(),
        events: libc::POLLIN,
        revents: 0,
    };
    assert_eq!(unsafe { libc::poll(&mut poll, 1, 3000) }, 1);
    assert_eq!(fixture.ledger.capacity(), 900);
    assert_eq!(reopened.status().unwrap().capacity, 900);
    assert!(!reopened.fits("package", "shape").unwrap());
    assert_eq!(owner.call("capacity"), "900");

    // Different ranks observe one usable total, not one deduction per process.
    assert_eq!(owner.call("limit 900"), "900");
    assert_eq!(correcting.call("limit 1024"), "900");
    let restarted = Ledger::create(&fixture.name, CAPACITY).unwrap();
    assert_eq!(restarted.capacity(), 900);
    assert_eq!(restarted.peaks(), vec![("package|shape".into(), 400)]);
    assert!(Ledger::create(&fixture.name, 800).is_err());
    assert!(fixture.ledger.limit_capacity(0).is_err());
    assert_eq!(fixture.ledger.capacity(), 900);
}

#[test]
fn usable_capacity_preserves_a_smaller_authorized_cap() {
    let fixture = Fixture::new();
    let capped = Ledger::create(&fixture.name, CAPACITY / 2).unwrap();
    let mut owner = ChildRow::spawn(&capped);
    capped.record_peak("package", "shape", 400).unwrap();
    assert_eq!(owner.call("limit 900"), "512");
    assert_eq!(fixture.ledger.capacity(), 512);
    assert_eq!(capped.peaks(), vec![("package|shape".into(), 400)]);
    drop(owner);
    let raised = Ledger::create(&fixture.name, CAPACITY).unwrap();
    assert_eq!(raised.capacity(), CAPACITY);
    assert!(raised.peaks().is_empty());
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
    assert_eq!((view.offset, view.len), (0, 4));
    let mut data = String::new();
    std::fs::File::from(view.file.try_clone().unwrap())
        .read_to_string(&mut data)
        .unwrap();
    assert_eq!(
        data, "data",
        "each independently reclaimable region has its own memfd"
    );
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

fn observe(ledger: &Ledger, mut ready: impl FnMut() -> bool) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while !ready() {
        let remaining = deadline.saturating_duration_since(std::time::Instant::now());
        assert!(!remaining.is_zero(), "RAM state did not progress");
        let mut poll = libc::pollfd {
            fd: ledger.events().as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        };
        assert_eq!(
            unsafe { libc::poll(&mut poll, 1, remaining.as_millis() as i32) },
            1
        );
        let mut word = [0u8; 8];
        assert_eq!(
            unsafe { libc::read(poll.fd, word.as_mut_ptr().cast(), word.len()) },
            8
        );
    }
}

#[test]
fn forked_cross_process_registration_uses_the_original_service_custody() {
    let fixture = Fixture::new();
    let mut child = ChildRow::spawn(&fixture.ledger);
    assert_eq!(child.call("warm"), "ok");
    observe(&fixture.ledger, || {
        fixture
            .ledger
            .ram_status([19; 32])
            .is_some_and(|s| !s.filling)
    });
    let status = fixture.ledger.ram_status([19; 32]).unwrap();
    assert_eq!(status.ready, vec![true]);
    let row = fixture.ledger.claim(std::process::id()).unwrap();
    let channel = Channel::new(fixture.ledger.grant(std::process::id()).unwrap().channel);
    let view = channel.ram(row.owner(), [19; 32], 0).unwrap().unwrap();
    let mut bytes = String::new();
    std::fs::File::from(view.file)
        .read_to_string(&mut bytes)
        .unwrap();
    assert_eq!(bytes, "warm");
}

#[test]
fn ram_generations_and_tagged_cancellation_follow_actual_custody() {
    let fixture = Fixture::new();
    let row = fixture.ledger.claim(std::process::id()).unwrap();
    let channel = Channel::new(fixture.ledger.grant(std::process::id()).unwrap().channel);
    let regions = [varena::ledger::RamRegion {
        offset: 0,
        len: 4,
        items: vec![tensorfs::io::source::Item {
            offset: 0,
            len: 4,
            part_end: true,
            source: tensorfs::io::ItemSource::Inline(b"warm".to_vec().into()),
        }],
    }];
    channel
        .cache_ram_tagged(row.owner(), [20; 32], &[], &regions, 1)
        .unwrap();
    observe(&fixture.ledger, || {
        channel
            .ram_status(row.owner(), [20; 32])
            .unwrap()
            .is_some_and(|s| !s.filling)
    });
    let first = channel.ram_status(row.owner(), [20; 32]).unwrap().unwrap();
    assert_eq!(first.ready, vec![true]);
    assert!(!channel
        .cache_ram_tagged(row.owner(), [20; 32], &[], &regions, 2)
        .unwrap());
    channel.forget_ram(row.owner(), [20; 32], 1).unwrap();
    assert_eq!(
        channel.ram_status(row.owner(), [20; 32]).unwrap(),
        Some(first.clone())
    );
    channel.forget_ram(row.owner(), [20; 32], 2).unwrap();
    assert!(channel.ram_status(row.owner(), [20; 32]).unwrap().is_none());
    assert!(channel.ram(row.owner(), [20; 32], 0).unwrap().is_none());
    channel
        .cache_ram_tagged(row.owner(), [20; 32], &[], &regions, 3)
        .unwrap();
    observe(&fixture.ledger, || {
        channel
            .ram_status(row.owner(), [20; 32])
            .unwrap()
            .is_some_and(|s| !s.filling)
    });
    assert!(
        channel
            .ram_status(row.owner(), [20; 32])
            .unwrap()
            .unwrap()
            .generation
            > first.generation
    );
    assert!(fixture.ledger.release_ram([20; 32]));
    assert!(channel.ram_status(row.owner(), [20; 32]).unwrap().is_none());
}

#[test]
fn two_gpu_ledgers_share_raw_ram_but_close_and_reap_keep_device_owners_separate() {
    use std::os::unix::fs::MetadataExt;
    let host = std::sync::Arc::new(varena::ledger::HostCache::default());
    let first = Fixture::with_host(host.clone());
    let second = Fixture::with_host(host);
    let mut child = ChildRow::spawn(&first.ledger);
    assert_eq!(child.call("warm"), "ok");
    observe(&first.ledger, || {
        first
            .ledger
            .ram_status([19; 32])
            .is_some_and(|s| !s.filling)
    });
    let row = second.ledger.claim(std::process::id()).unwrap();
    assert_eq!(
        row.owner(),
        child.owner,
        "same row/generation is distinct on another device"
    );
    let channel = Channel::new(second.ledger.grant(std::process::id()).unwrap().channel);
    let regions = [varena::ledger::RamRegion {
        offset: 0,
        len: 4,
        items: vec![tensorfs::io::source::Item {
            offset: 0,
            len: 4,
            part_end: true,
            source: tensorfs::io::ItemSource::Inline(b"warm".to_vec().into()),
        }],
    }];
    assert!(!channel
        .cache_ram(row.owner(), [19; 32], &[], &regions)
        .unwrap());
    let original = first.ledger.ram_status([19; 32]).unwrap();
    assert_eq!(
        channel.ram_status(row.owner(), [19; 32]).unwrap(),
        Some(original.clone())
    );
    assert_eq!(first.ledger.status().unwrap().ram_bytes, 4);
    assert_eq!(
        second.ledger.status().unwrap().ram_bytes,
        4,
        "one shared layout is counted by each view"
    );
    let first_row = first.ledger.claim(std::process::id()).unwrap();
    let first_channel = Channel::new(first.ledger.grant(std::process::id()).unwrap().channel);
    let a = first_channel
        .ram(first_row.owner(), [19; 32], 0)
        .unwrap()
        .unwrap();
    let b = channel.ram(row.owner(), [19; 32], 0).unwrap().unwrap();
    assert_eq!(
        std::fs::File::from(a.file.try_clone().unwrap())
            .metadata()
            .unwrap()
            .ino(),
        std::fs::File::from(b.file.try_clone().unwrap())
            .metadata()
            .unwrap()
            .ino(),
        "both devices receive the exact same memfd"
    );
    assert_eq!(child.call("forget_ram"), "ok");
    assert_eq!(
        channel.ram_status(row.owner(), [19; 32]).unwrap(),
        Some(original.clone()),
        "device A's close removed device B"
    );
    assert_eq!(child.call("warm"), "ok");
    child.kill();
    first.ledger.reap(child.process.id()).unwrap();
    // This ordered RPC also proves channel B finished its earlier lease handoff:
    // recv_fd alone can return before the sender drops its local descriptor.
    assert_eq!(
        channel.ram_status(row.owner(), [19; 32]).unwrap(),
        Some(original.clone()),
        "device A's reap removed device B"
    );
    assert!(
        !first.ledger.release_unused_ram(),
        "pressure cannot reclaim a live reader on either device"
    );
    // Channel A is independent: B's RPC cannot order A's descriptor handoff.
    // Acknowledge A too before testing release of the clients' last lease copies.
    assert_eq!(
        first_channel
            .ram_status(first_row.owner(), [19; 32])
            .unwrap(),
        Some(original)
    );
    drop((a, b));
    assert!(
        first.ledger.release_unused_ram(),
        "pressure on one device sheds the shared idle host tier"
    );
    assert!(channel.ram_status(row.owner(), [19; 32]).unwrap().is_none());
    assert!(channel.ram(row.owner(), [19; 32], 0).unwrap().is_none());
}

#[test]
fn explicitly_owned_host_cache_drops_with_its_last_ledger() {
    let host = std::sync::Arc::new(varena::ledger::HostCache::default());
    let lifetime = std::sync::Arc::downgrade(&host);
    let first = Fixture::with_host(host.clone());
    let second = Fixture::with_host(host);
    drop(first);
    assert!(lifetime.upgrade().is_some());
    drop(second);
    assert!(
        lifetime.upgrade().is_none(),
        "there is no global owner retaining host custody"
    );
}

#[test]
fn pressure_reclaims_one_region_while_another_reader_stays_live_then_refills_the_hole() {
    let fixture = Fixture::new();
    let row = fixture.ledger.claim(std::process::id()).unwrap();
    let channel = Channel::new(fixture.ledger.grant(std::process::id()).unwrap().channel);
    let regions = (0..2)
        .map(|index| varena::ledger::RamRegion {
            offset: index * 4096,
            len: 4096,
            items: vec![tensorfs::io::source::Item {
                offset: index * 4096,
                len: 4,
                part_end: true,
                source: tensorfs::io::ItemSource::Inline(b"warm".to_vec().into()),
            }],
        })
        .collect::<Vec<_>>();
    assert!(channel
        .cache_ram(row.owner(), [30; 32], &[], &regions)
        .unwrap());
    observe(&fixture.ledger, || {
        channel
            .ram_status(row.owner(), [30; 32])
            .unwrap()
            .is_some_and(|s| !s.filling)
    });
    let first = channel.ram_status(row.owner(), [30; 32]).unwrap().unwrap();
    assert_eq!(first.ready, vec![true, true]);
    let live = channel.ram(row.owner(), [30; 32], 1).unwrap().unwrap();
    assert!(fixture.ledger.release_unused_ram());
    let partial = channel.ram_status(row.owner(), [30; 32]).unwrap().unwrap();
    assert_eq!(partial.ready, vec![false, true]);
    assert!(partial.generation > first.generation);
    assert!(channel.ram(row.owner(), [30; 32], 0).unwrap().is_none());
    assert!(
        !fixture.ledger.release_unused_ram(),
        "only the reader's own region stays protected"
    );
    assert!(
        channel
            .cache_ram(row.owner(), [30; 32], &[], &regions)
            .unwrap(),
        "a later hint refills only the missing region"
    );
    observe(&fixture.ledger, || {
        channel
            .ram_status(row.owner(), [30; 32])
            .unwrap()
            .is_some_and(|s| !s.filling)
    });
    assert_eq!(
        channel
            .ram_status(row.owner(), [30; 32])
            .unwrap()
            .unwrap()
            .ready,
        vec![true, true]
    );
    let mut bytes = [0; 4];
    std::fs::File::from(live.file.try_clone().unwrap())
        .read_exact(&mut bytes)
        .unwrap();
    assert_eq!(&bytes, b"warm");
    drop(live);
}

#[test]
fn unmeasured_admission_sheds_cached_regions_and_suppresses_optional_warms_until_finished() {
    let fixture = Fixture::new();
    let row = fixture.ledger.claim(std::process::id()).unwrap();
    let channel = Channel::new(fixture.ledger.grant(std::process::id()).unwrap().channel);
    let regions = [varena::ledger::RamRegion {
        offset: 0,
        len: 4096,
        items: vec![tensorfs::io::source::Item {
            offset: 0,
            len: 4,
            part_end: true,
            source: tensorfs::io::ItemSource::Inline(b"warm".to_vec().into()),
        }],
    }];
    channel
        .cache_ram(row.owner(), [31; 32], &[], &regions)
        .unwrap();
    observe(&fixture.ledger, || {
        channel
            .ram_status(row.owner(), [31; 32])
            .unwrap()
            .is_some_and(|s| !s.filling)
    });
    fixture.ledger.prepare_host(1, None);
    assert!(channel.ram_status(row.owner(), [31; 32]).unwrap().is_none());
    assert!(fixture.ledger.fits("unknown-host-peak", "new").unwrap());
    channel
        .cache_ram(row.owner(), [31; 32], &[], &regions)
        .unwrap();
    observe(&fixture.ledger, || {
        channel
            .ram_status(row.owner(), [31; 32])
            .unwrap()
            .is_none_or(|s| !s.filling)
    });
    assert!(channel.ram_status(row.owner(), [31; 32]).unwrap().is_none());
    fixture.ledger.finish_host(1);
    assert!(channel
        .cache_ram(row.owner(), [31; 32], &[], &regions)
        .unwrap());
    observe(&fixture.ledger, || {
        channel
            .ram_status(row.owner(), [31; 32])
            .unwrap()
            .is_some_and(|s| !s.filling)
    });
    assert_eq!(
        channel
            .ram_status(row.owner(), [31; 32])
            .unwrap()
            .unwrap()
            .ready,
        vec![true]
    );
}

#[test]
#[ignore = "child of the delegated-cgroup pressure test"]
fn pressure_child() {
    let fixture = Fixture::new();
    let row = fixture.ledger.claim(std::process::id()).unwrap();
    let channel = Channel::new(fixture.ledger.grant(std::process::id()).unwrap().channel);
    let warm = 20 << 20;
    for key in [[22; 32], [23; 32]] {
        assert!(channel
            .cache_ram(
                row.owner(),
                key,
                &[],
                &[varena::ledger::RamRegion {
                    offset: 0,
                    len: warm,
                    items: vec![],
                }]
            )
            .unwrap());
    }
    observe(&fixture.ledger, || {
        [[22; 32], [23; 32]].into_iter().all(|key| {
            channel
                .ram_status(row.owner(), key)
                .unwrap()
                .is_some_and(|status| !status.filling)
        })
    });
    assert_eq!(
        [[22; 32], [23; 32]]
            .into_iter()
            .filter(|key| {
                channel
                    .ram_status(row.owner(), *key)
                    .unwrap()
                    .unwrap()
                    .ready
                    == [true]
            })
            .count(),
        1,
        "concurrent cache hints cannot spend the same cgroup headroom twice"
    );
    let len = 64 << 20;
    assert!(platform::contain::host_memory::read().available < len as i64);
    assert!(channel
        .cache_ram(
            row.owner(),
            [21; 32],
            &[],
            &[varena::ledger::RamRegion {
                offset: 0,
                len,
                items: vec![],
            }]
        )
        .unwrap());
    observe(&fixture.ledger, || {
        channel
            .ram_status(row.owner(), [21; 32])
            .unwrap()
            .is_some_and(|s| !s.filling)
    });
    assert_eq!(
        channel
            .ram_status(row.owner(), [21; 32])
            .unwrap()
            .unwrap()
            .ready,
        vec![false]
    );
    assert!(channel.ram(row.owner(), [21; 32], 0).unwrap().is_none());
    assert!(
        fixture
            .ledger
            .fits("unmeasured", "starts-under-pressure")
            .unwrap(),
        "host pressure never blocks admission"
    );
}

#[test]
fn cgroup_pressure_refuses_warming_without_claiming_readiness_or_blocking_admission() {
    if std::env::var_os("GITHUB_ACTIONS").is_none() {
        eprintln!("real pressure proof runs in the hosted systemd fixture");
        return;
    }
    // The delegated service has no child memory controller enabled. Let systemd
    // create one owned, finite-memory service rather than alter the test runner's
    // hierarchy or put unrelated test processes under its limit.
    let unit = format!(
        "varena-ram-pressure-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Relaxed)
    );
    let output = Command::new("sudo")
        .args([
            "-n",
            "systemd-run",
            "--quiet",
            "--wait",
            "--pipe",
            "--collect",
        ])
        .arg(format!("--unit={unit}"))
        .arg(format!("--uid={}", unsafe { libc::getuid() }))
        .args(["--property=MemoryMax=32M", "--property=MemorySwapMax=0"])
        .arg(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "pressure_child",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "pressure child failed: {}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
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
