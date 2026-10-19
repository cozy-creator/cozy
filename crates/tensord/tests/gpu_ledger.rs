//! TensorD on varena's ledger (design memory-varena §2, §5.6), with real processes as tenants
//! and a fake `libnvidia-ml.so.1` as the only stand-in: the GPU. Each test runs in a child of this
//! binary under that NVML, whose memory is the bytes each file in `FAKE_NVML_DIR` charges its
//! pid. A tenant (`tenant`, another child) is an executor's ledger half: it claims its own row,
//! publishes its facts and running requests, charges its bytes, frees, and answers asks by
//! evicting cold bytes (or never). Its death leaves its charge until the test takes the file
//! away, as a driver gives a dead process's memory back.
#![cfg(target_os = "linux")]
use std::{
    fs,
    io::{BufRead, BufReader, Write},
    os::{
        fd::{AsRawFd, BorrowedFd},
        unix::process::CommandExt,
    },
    path::{Path, PathBuf},
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex, OnceLock,
    },
    time::Duration,
};
use tensord::{
    memory::{GpuLedger, MemoryLog},
    process::Liveness,
};
use varena::ledger::{Facts, Ledger};

const GIB: u64 = 1 << 30;

const FAKE_NVML: &str = r#"
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
static int gpu;
typedef struct { unsigned version; unsigned long long total, reserved, free, used; } memory_v2;
typedef struct { unsigned pid; unsigned long long used; unsigned gpu_instance, compute_instance; } process;
static unsigned scan(process *out, unsigned room, unsigned long long *sum) {
    const char *root = getenv("FAKE_NVML_DIR");
    DIR *dir = root ? opendir(root) : 0;
    struct dirent *entry;
    char path[4096];
    unsigned n = 0;
    *sum = 0;
    while (dir && (entry = readdir(dir))) {
        if (entry->d_name[0] == '.') continue;
        snprintf(path, sizeof path, "%s/%s", root, entry->d_name);
        FILE *file = fopen(path, "r");
        unsigned long long bytes = 0;
        if (!file) continue;
        if (fscanf(file, "%llu", &bytes) != 1) bytes = 0;
        fclose(file);
        *sum += bytes;
        if (out && n < room) {
            out[n].pid = strtoul(entry->d_name, 0, 10);
            out[n].used = bytes;
            out[n].gpu_instance = out[n].compute_instance = 0xffffffff;
        }
        n++;
    }
    if (dir) closedir(dir);
    return n;
}
int nvmlInit_v2(void) { return 0; }
int nvmlInit_v3(unsigned flags) { return 0; }
int nvmlShutdown(void) { return 0; }
int nvmlDeviceGetCount_v2(unsigned *n) { *n = 1; return 0; }
int nvmlDeviceGetHandleByIndex_v2(unsigned i, void **d) { if (i) return 2; *d = &gpu; return 0; }
int nvmlDeviceGetHandleByUUID(const char *u, void **d) { *d = &gpu; return 0; }
int nvmlDeviceGetIndex(void *d, unsigned *i) { *i = 0; return 0; }
int nvmlDeviceGetUUID(void *d, char *b, unsigned n) { snprintf(b, n, "GPU-fake-ledger"); return 0; }
int nvmlDeviceGetName(void *d, char *b, unsigned n) { snprintf(b, n, "NVIDIA Fake"); return 0; }
int nvmlSystemGetDriverVersion(char *b, unsigned n) { snprintf(b, n, "595.84"); return 0; }
int nvmlDeviceGetDisplayActive(void *d, int *s) { *s = 0; return 0; }
int nvmlDeviceGetMemoryInfo_v2(void *d, memory_v2 *m) {
    unsigned long long used;
    scan(0, 0, &used);
    m->total = strtoull(getenv("FAKE_NVML_TOTAL"), 0, 10);
    m->reserved = 0;
    m->used = used;
    m->free = m->total > used ? m->total - used : 0;
    return 0;
}
int nvmlDeviceGetComputeRunningProcesses_v3(void *d, unsigned *count, process *infos) {
    unsigned long long used;
    unsigned n;
    if (!getenv("FAKE_NVML_PROCESSES")) return 3;
    n = scan(infos, infos ? *count : 0, &used);
    if (!infos || *count < n) { *count = n; return n ? 7 : 0; }
    *count = n;
    return 0;
}
"#;

/// The fake NVML's directory, built once per process.
fn fake_nvml() -> &'static Path {
    static BUILT: OnceLock<PathBuf> = OnceLock::new();
    BUILT.get_or_init(|| {
        let dir = std::env::temp_dir().join(format!("fake-nvml-ledger-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("nvml.c"), FAKE_NVML).unwrap();
        let built = Command::new("cc")
            .args(["-shared", "-fPIC", "-o"])
            .arg(dir.join("libnvidia-ml.so.1"))
            .arg(dir.join("nvml.c"))
            .status()
            .unwrap();
        assert!(built.success());
        dir
    })
}

/// Runs `inner` in a child under the fake NVML with a GPU of `total` bytes; `processes`: NVML
/// charges each process its bytes (outside a container).
fn under_fake_nvml(inner: &str, total: u64, processes: bool) {
    let state = tempfile::tempdir().unwrap();
    let mut command = Command::new(std::env::current_exe().unwrap());
    command
        .args(["--exact", inner, "--ignored", "--nocapture", "--test-threads=1"])
        .env("LD_LIBRARY_PATH", fake_nvml())
        .env("FAKE_NVML_DIR", state.path())
        .env("FAKE_NVML_TOTAL", total.to_string());
    if processes {
        command.env("FAKE_NVML_PROCESSES", "1");
    }
    let out = command.output().unwrap();
    let said = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    assert!(out.status.success() && said.contains("1 passed"), "{said}");
}

fn charges() -> PathBuf {
    PathBuf::from(std::env::var("FAKE_NVML_DIR").unwrap())
}

/// The ledger TensorD keeps for the fake GPU.
fn ledger(liveness: Liveness) -> GpuLedger {
    GpuLedger::start("0", false, 0, None, None, liveness, MemoryLog::default()).unwrap().expect("a ledger")
}

/// The driver gives `pid`'s memory back.
fn release(pid: u32) {
    let _ = fs::remove_file(charges().join(pid.to_string()));
}

/// A process charging the GPU without a row (another program).
fn foreign(bytes: u64) {
    fs::write(charges().join("1"), bytes.to_string()).unwrap();
}

/// A tenant process with its own row, driven over stdin: `publish context resident cold torch`,
/// `running [id remaining]…`, `freed`; each answered `tenant-ok`.
struct Tenant {
    child: Child,
    input: ChildStdin,
    output: BufReader<ChildStdout>,
    pid: u32,
}

impl Tenant {
    /// Its bytes are charged in `charges` (the fake NVML's directory, or any).
    fn start(ledger: &GpuLedger, charges: &Path, mode: &str) -> Self {
        let [memfd, events] = ledger.grant().unwrap();
        let (fd, events_fd) = (memfd.as_raw_fd(), events.as_raw_fd());
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args(["--exact", "tenant", "--ignored", "--nocapture", "--test-threads=1"])
            .env("LEDGER_FD", fd.to_string())
            .env("EVENTS_FD", events_fd.to_string())
            .env("TENANT_MODE", mode)
            .env("FAKE_NVML_DIR", charges)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped());
        // SAFETY: only fcntl on descriptors this process keeps open through the spawn.
        unsafe {
            command.pre_exec(move || match (libc::fcntl(fd, libc::F_SETFD, 0), libc::fcntl(events_fd, libc::F_SETFD, 0)) {
                (0, 0) => Ok(()),
                _ => Err(std::io::Error::last_os_error()),
            });
        }
        let mut child = command.spawn().unwrap();
        drop((memfd, events));
        let (input, output) = (child.stdin.take().unwrap(), BufReader::new(child.stdout.take().unwrap()));
        let pid = child.id();
        let mut tenant = Self { child, input, output, pid };
        tenant.answer();
        tenant
    }

    /// libtest prints `test tenant ... ` ahead of what the test prints on the same line.
    fn answer(&mut self) {
        let mut line = String::new();
        loop {
            line.clear();
            assert!(self.output.read_line(&mut line).unwrap() > 0, "tenant {} exited", self.pid);
            if line.trim_end().ends_with("tenant-ok") {
                return;
            }
        }
    }

    fn call(&mut self, line: &str) {
        writeln!(self.input, "{line}").unwrap();
        self.input.flush().unwrap();
        self.answer();
    }

    /// `kill -9`: its row drops at once; its bytes stay charged until the driver lets them go.
    fn kill(&mut self) {
        self.child.kill().unwrap();
        self.child.wait().unwrap();
    }
}

impl Drop for Tenant {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// A tenant: claims a row in the ledger it inherited and serves the commands above; in `answer`
/// mode a thread answers asks by evicting cold bytes, in `stall` mode nothing does.
#[test]
#[ignore = "a tenant process the ledger tests start"]
fn tenant() {
    let fd: i32 = std::env::var("LEDGER_FD").unwrap().parse().unwrap();
    let events: i32 = std::env::var("EVENTS_FD").unwrap().parse().unwrap();
    // SAFETY: the descriptors this process inherited, open for its whole life.
    let ledger = Ledger::adopt(unsafe { BorrowedFd::borrow_raw(fd) }, Some(unsafe { BorrowedFd::borrow_raw(events) })).unwrap();
    let ledger = Arc::new(ledger);
    let row = Arc::new(ledger.claim(std::process::id()).unwrap());
    let facts = Arc::new(Mutex::new(Facts { last_use: 1, ..Facts::default() }));
    let charge = |facts: &Facts| fs::write(charges().join(std::process::id().to_string()), facts.held().to_string()).unwrap();
    if std::env::var("TENANT_MODE").unwrap() == "answer" {
        let (row, facts) = (row.clone(), facts.clone());
        std::thread::spawn(move || loop {
            let want = row.wait_want();
            let mut f = facts.lock().unwrap();
            let freed = want.min(f.cold);
            (f.cold, f.resident) = (f.cold - freed, f.resident - freed);
            row.publish(&f);
            charge(&f);
            row.ack(freed);
        });
    }
    println!("tenant-ok");
    for line in std::io::stdin().lock().lines() {
        let line = line.unwrap();
        let words: Vec<&str> = line.split_whitespace().collect();
        match words[0] {
            "publish" => {
                let n: Vec<u64> = words[1..].iter().map(|w| w.parse().unwrap()).collect();
                let mut f = facts.lock().unwrap();
                (f.context, f.resident, f.cold, f.torch_allocated, f.torch_reserved) = (n[0], n[1], n[2], n[3], n[3]);
                row.publish(&f);
                charge(&f);
            }
            "running" => {
                let pairs = words[1..].chunks(2).map(|p| (p[0].to_string(), p[1].parse().unwrap())).collect::<Vec<(String, i64)>>();
                row.running(&pairs).unwrap();
            }
            "freed" => ledger.freed(),
            other => panic!("unknown command {other}"),
        }
        println!("tenant-ok");
    }
}

/// `fits` through TensorD's ledger: a measured peak beside every row's unevictable bytes, every
/// running request's remaining peak and what NVML shows that no row explains; a shape never
/// measured only alone. A request ending frees bytes, and the pool hears it.
#[test]
fn a_request_starts_only_when_its_measured_peak_fits_and_freed_bytes_wake_the_pool() {
    under_fake_nvml("inner_a_request_starts_only_when_it_fits", 24 * GIB, false);
}

#[test]
#[ignore = "run under the fake NVML by its outer test"]
fn inner_a_request_starts_only_when_it_fits() {
    let ledger = ledger(Liveness::default());
    let woken = Arc::new(AtomicU64::new(0));
    let heard = woken.clone();
    ledger.on_freed(Box::new(move || {
        heard.fetch_add(1, Ordering::SeqCst);
    }));
    let mut running = Tenant::start(&ledger, &charges(), "answer");
    running.call(&format!("publish {} {} {} {}", GIB / 2, 4 * GIB, 4 * GIB, 3 * GIB / 2));
    ledger.now();
    assert!(ledger.fits("plan", Some("s"), false), "never measured, and nothing runs");
    running.call(&format!("running r1 {}", 12 * GIB));
    assert!(!ledger.fits("plan", Some("s"), false), "never measured beside a running request");
    let [memfd, _] = ledger.grant().unwrap();
    let executor = Ledger::adopt(std::os::fd::AsFd::as_fd(&memfd), None).unwrap();
    executor.set_peak("plan|s", 8 * GIB).unwrap();
    assert!(ledger.fits("plan", Some("s"), false), "2 unevictable + 12 remaining + 8 of 24");
    foreign(3 * GIB);
    ledger.now();
    assert!(!ledger.fits("plan", Some("s"), false), "another program's 3 GiB count");
    release(1);
    ledger.now();
    assert!(ledger.fits("plan", Some("s"), false));
    let before = woken.load(Ordering::SeqCst);
    running.call("running");
    running.call("freed");
    for _ in 0..2 {
        ledger.now();
    }
    assert!(woken.load(Ordering::SeqCst) > before, "freed bytes woke the pool");
    executor.set_peak("plan|s", 23 * GIB).unwrap();
    assert!(!ledger.fits("plan", Some("s"), false), "2 unevictable + 23 of 24");
    assert!(ledger.fits("plan", Some("other"), false), "a shape never measured, alone");
}

/// A 24 GiB card limited to 8 GiB (`machine.gpu_memory_limit`): the ledger executors adopt
/// says 8, and `fits` measures against 8, never the card; a limit above the card is the card.
#[test]
fn a_gpu_memory_limit_is_every_ledgers_capacity() {
    under_fake_nvml("inner_a_gpu_memory_limit_is_every_ledgers_capacity", 24 * GIB, false);
}

#[test]
#[ignore = "run under the fake NVML by its outer test"]
fn inner_a_gpu_memory_limit_is_every_ledgers_capacity() {
    let limited = |limit| GpuLedger::start("0", false, 0, None, Some(limit), Liveness::default(), MemoryLog::default()).unwrap().unwrap();
    assert_eq!(limited(32 * GIB).status().unwrap().capacity, 24 * GIB, "never above the card");
    let ledger = limited(8 * GIB);
    let [memfd, _] = ledger.grant().unwrap();
    let executor = Ledger::adopt(std::os::fd::AsFd::as_fd(&memfd), None).unwrap();
    assert_eq!(executor.capacity(), 8 * GIB, "what an executor reads");
    let mut tenant = Tenant::start(&ledger, &charges(), "answer");
    tenant.call(&format!("publish {} {} 0 {}", GIB / 2, 2 * GIB, GIB));
    ledger.now();
    executor.set_peak("plan|s", 6 * GIB).unwrap();
    assert!(!ledger.fits("plan", Some("s"), false), "3.5 unevictable + 6 of 8, though the card has 24");
    executor.set_peak("plan|s", 4 * GIB).unwrap();
    assert!(ledger.fits("plan", Some("s"), false), "3.5 + 4 of 8");
}

/// The 11405 shape: an executor holding 7 GiB of an 8 GiB card in torch is killed. Its bytes
/// are counted until NVML shows them back, and nothing fits beside them meanwhile.
#[test]
fn a_killed_executors_bytes_stay_counted_until_nvml_shows_them_back() {
    under_fake_nvml("inner_a_killed_executors_bytes_stay_counted", 8 * GIB, true);
}

/// The same in a container, where NVML charges no process its bytes.
#[test]
fn a_killed_executors_bytes_stay_counted_where_nvml_cannot_attribute_them() {
    under_fake_nvml("inner_a_killed_executors_bytes_stay_counted", 8 * GIB, false);
}

#[test]
#[ignore = "run under the fake NVML by its outer test"]
fn inner_a_killed_executors_bytes_stay_counted() {
    let ledger = ledger(Liveness::default());
    let mut held = Tenant::start(&ledger, &charges(), "answer");
    held.call(&format!("publish {} 0 0 {}", GIB / 2, 6 * GIB + GIB / 2));
    let [memfd, _] = ledger.grant().unwrap();
    Ledger::adopt(std::os::fd::AsFd::as_fd(&memfd), None).unwrap().set_peak("plan|s", 2 * GIB).unwrap();
    ledger.now();
    assert!(!ledger.fits("plan", Some("s"), false), "7 held + 2 of 8");
    held.kill();
    ledger.settle();
    assert!(!ledger.fits("plan", Some("s"), false), "nothing beside a dead executor's bytes");
    // Where NVML charges each process, its row stays reaping meanwhile: not even a request never
    // measured starts beside it. In a container the bytes are TensorD's row's, unexplained.
    let attributed = std::env::var_os("FAKE_NVML_PROCESSES").is_some();
    assert_eq!(ledger.fits("plan", Some("never"), false), !attributed);
    release(held.pid);
    ledger.settle();
    assert!(ledger.fits("plan", Some("s"), false));
}

/// A victim that never answers an ask is named stalled once its progress has not moved for a
/// wedge floor, and ended; the asker's wait then returns.
#[test]
fn a_stalled_victim_is_ended_and_its_asker_goes_on() {
    under_fake_nvml("inner_a_stalled_victim_is_ended", 8 * GIB, false);
}

#[test]
#[ignore = "run under the fake NVML by its outer test"]
fn inner_a_stalled_victim_is_ended() {
    let ledger = ledger(Liveness { sample: Duration::from_millis(50) });
    let mut stalled = Tenant::start(&ledger, &charges(), "stall");
    stalled.call(&format!("publish {} {} {} 0", GIB / 2, 6 * GIB + GIB / 2, 6 * GIB));
    let ended: Arc<Mutex<Vec<u32>>> = Arc::default();
    let seen = ended.clone();
    ledger.on_stalled(Box::new(move |pid| {
        seen.lock().unwrap().push(pid);
        // SAFETY: a signal to the tenant this test started, alive until its Tenant drops.
        unsafe { libc::kill(pid as i32, libc::SIGKILL) };
        release(pid);
    }));
    let [memfd, _] = ledger.grant().unwrap();
    let asker = Ledger::adopt(std::os::fd::AsFd::as_fd(&memfd), None).unwrap();
    // A call in flight: executors ask each other for bytes, and TensorD watches for stalls.
    let _call = ledger.active();
    for victim in asker.ask(None, 3 * GIB).unwrap() {
        asker.wait(&victim).unwrap();
    }
    assert_eq!(*ended.lock().unwrap(), [stalled.pid]);
}

/// A group's ranks each hold a row in their own GPU's ledger (no rank cells): a follower that
/// dies is reaped on its GPU alone. No NVML: each ledger's capacity is the configured one.
#[test]
fn each_rank_of_a_group_holds_its_own_gpus_row() {
    let ledgers: Vec<GpuLedger> = ["GPU-rank-0", "GPU-rank-1"]
        .iter()
        .map(|entry| GpuLedger::start(entry, false, 0, Some(8 * GIB), None, Liveness::default(), MemoryLog::default()).unwrap().unwrap())
        .collect();
    let dir = tempfile::tempdir().unwrap();
    let mut ranks: Vec<Tenant> = ledgers.iter().map(|l| Tenant::start(l, dir.path(), "answer")).collect();
    for rank in &mut ranks {
        rank.call(&format!("publish {} {} 0 {}", GIB / 2, 2 * GIB, GIB));
        rank.call("running req-1 0");
    }
    for ledger in &ledgers {
        ledger.now();
        assert!(ledger.running());
    }
    let [rank0, mut rank1]: [Tenant; 2] = ranks.try_into().ok().unwrap();
    assert!(ledgers[1].resident(rank1.pid) && !ledgers[1].resident(rank0.pid));
    rank1.kill();
    ledgers[1].settle();
    assert!(!ledgers[1].running(), "the follower's GPU has its row back");
    assert!(ledgers[0].running() && ledgers[0].resident(rank0.pid), "rank 0's GPU is untouched");
}

/// Kept GPU pages (B8): while their executor lives they are its own; once it is gone they are
/// TensorD's, counted in its row as kept, handed whole to the next executor of the layout, and
/// closed when another row asks for room. Memfds in containers stand in for the VMM fds.
#[test]
fn kept_pages_outlive_their_executor_and_go_on_an_ask() {
    use std::{fs::File, io::Read, os::fd::{AsFd, OwnedFd}, os::unix::net::UnixStream};
    use tensord::device_executor::Page;
    let ledger = GpuLedger::start("GPU-kept", false, 0, Some(8 * GIB), None, Liveness::default(), MemoryLog::default()).unwrap().unwrap();
    let page = |region: u32, text: &str| -> (Page, File) {
        let mut fd = File::from(platform::shm::memfd_named("page").unwrap());
        fd.write_all(text.as_bytes()).unwrap();
        let (tx, rx) = UnixStream::pair().unwrap();
        platform::handles::send_fd(&tx, &fd).unwrap();
        (Page { region, nbytes: GIB }, File::from(OwnedFd::from(rx)))
    };
    let kept = |l: &GpuLedger| {
        l.now();
        let rows = l.status().unwrap().rows;
        rows.iter().find(|r| r.pid == std::process::id()).unwrap().facts.kept
    };
    let mut executor = Command::new("sleep").arg("1000").spawn().unwrap();
    ledger.keep("layout-a", "fp-a", executor.id(), vec![page(0, "zero"), page(1, "one")]);
    ledger.keep("layout-b", "fp-b", executor.id(), vec![page(0, "b")]);
    assert!(ledger.take_kept("layout-a").is_none() && kept(&ledger) == 0, "a live executor's pages are its own");
    executor.kill().unwrap();
    executor.wait().unwrap();
    assert_eq!(kept(&ledger), 3 * GIB, "a gone executor's pages are TensorD's");
    let (fingerprint, pages) = ledger.take_kept("layout-a").expect("handed to the next executor");
    let read = |mut f: &File| {
        let mut text = String::new();
        std::io::Seek::rewind(&mut f).unwrap();
        f.read_to_string(&mut text).unwrap();
        text
    };
    assert_eq!((fingerprint.as_str(), pages.iter().map(|(p, f)| (p.region, read(f))).collect::<Vec<_>>()), ("fp-a", vec![(0, "zero".into()), (1, "one".into())]));
    assert_eq!(kept(&ledger), GIB);
    let [table, events] = ledger.grant().unwrap();
    let asker = Ledger::adopt(table.as_fd(), Some(events.as_fd())).unwrap();
    let victims = asker.ask(None, GIB).unwrap();
    assert_eq!(victims.iter().map(|v| (v.pid, v.bytes)).collect::<Vec<_>>(), [(std::process::id(), GIB)]);
    for victim in &victims {
        asker.wait(victim).unwrap();
    }
    assert!(kept(&ledger) == 0 && ledger.take_kept("layout-b").is_none(), "an ask closed them");
}
