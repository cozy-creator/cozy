//! TensorD on varena's ledger (design memory-varena §5.6), with real processes as tenants and
//! a fake `libnvidia-ml.so.1` as the only stand-in: the GPU. Each test runs in a child of this
//! binary under that NVML, whose memory is the bytes each file in `FAKE_NVML_DIR` charges its
//! pid; a tenant (`tenant`, another child) claims its own row, publishes its facts, charges
//! its bytes and answers asks by evicting cold bytes. Its death leaves its charge until the
//! test takes the file away, as a driver gives a dead process's memory back.
#![cfg(target_os = "linux")]
use std::{
    fs,
    io::{BufRead, BufReader},
    os::{
        fd::{AsRawFd, BorrowedFd},
        unix::process::CommandExt,
    },
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{Arc, Mutex, OnceLock},
    time::Duration,
};
use tensord::{
    memory::{Admitted, GpuLedger, MemoryLog},
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
    GpuLedger::start("0", false, 0, None, liveness, MemoryLog::default()).unwrap().expect("a ledger")
}

/// A tenant process with its own row in `ledger`.
struct Tenant {
    child: Child,
    pid: u32,
}

impl Tenant {
    /// Its bytes are charged in `charges` (the fake NVML's directory, or any).
    fn start(ledger: &GpuLedger, charges: &Path, facts: &str, mode: &str) -> Self {
        let fd = ledger.grant().unwrap();
        let raw = fd.as_raw_fd();
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args(["--exact", "tenant", "--ignored", "--nocapture", "--test-threads=1"])
            .env("LEDGER_FD", raw.to_string())
            .env("TENANT_FACTS", facts)
            .env("TENANT_MODE", mode)
            .env("FAKE_NVML_DIR", charges)
            .stdout(Stdio::piped());
        // SAFETY: only fcntl on a descriptor this process keeps open through the spawn.
        unsafe {
            command.pre_exec(move || match libc::fcntl(raw, libc::F_SETFD, 0) {
                0 => Ok(()),
                _ => Err(std::io::Error::last_os_error()),
            });
        }
        let mut child = command.spawn().unwrap();
        drop(fd);
        let mut lines = BufReader::new(child.stdout.take().unwrap()).lines();
        // libtest prints `test tenant ... ` ahead of what the test prints on the same line.
        assert!(lines.any(|line| line.unwrap().ends_with("tenant-ready")), "the tenant never claimed its row");
        let pid = child.id();
        Self { child, pid }
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

/// The driver gives `pid`'s memory back.
fn release(pid: u32) {
    let _ = fs::remove_file(charges().join(pid.to_string()));
}

/// A tenant: claims a row in the ledger it inherited, publishes `TENANT_FACTS`
/// (`context,resident,cold,torch_allocated` bytes), charges them to its pid, then
/// answers asks by evicting cold bytes (`answer`) or never (`stall`). It says `tenant-ready`.
#[test]
#[ignore = "a tenant process the ledger tests start"]
fn tenant() {
    let fd: i32 = std::env::var("LEDGER_FD").unwrap().parse().unwrap();
    // SAFETY: the descriptor this process inherited, open for its whole life.
    let ledger = Ledger::adopt(unsafe { BorrowedFd::borrow_raw(fd) }).unwrap();
    let row = ledger.claim(std::process::id()).unwrap();
    let bytes: Vec<u64> = std::env::var("TENANT_FACTS").unwrap().split(',').map(|n| n.parse().unwrap()).collect();
    let mut facts = Facts {
        context: bytes[0],
        resident: bytes[1],
        cold: bytes[2],
        torch_allocated: bytes[3],
        torch_reserved: bytes[3],
        last_use: 1,
        ..Facts::default()
    };
    let charge = |facts: &Facts| fs::write(charges().join(std::process::id().to_string()), facts.held().to_string()).unwrap();
    row.publish(&facts);
    charge(&facts);
    println!("tenant-ready");
    if std::env::var("TENANT_MODE").unwrap() == "stall" {
        loop {
            std::thread::park();
        }
    }
    loop {
        let freed = row.wait_want().min(facts.cold);
        facts.cold -= freed;
        facts.resident -= freed;
        row.publish(&facts);
        charge(&facts);
        row.ack(freed);
    }
}

#[test]
fn a_second_executor_is_admitted_once_the_firsts_cold_weights_are_asked_for_and_freed() {
    under_fake_nvml("inner_a_second_executor_waits_for_the_firsts_cold_weights", 24 * GIB, false);
}

#[test]
#[ignore = "run under the fake NVML by its outer test"]
fn inner_a_second_executor_waits_for_the_firsts_cold_weights() {
    let ledger = ledger(Liveness::default());
    let first = Tenant::start(&ledger, &charges(), &format!("{},{},{},0", GIB / 2, 19 * GIB + GIB / 2, 19 * GIB), "answer");
    ledger.now();
    assert_eq!(ledger.free().unwrap(), 4 * GIB);
    assert!(!ledger.fits(8 * GIB, true), "8 GiB is not free beside the first executor");
    assert!(ledger.fits(8 * GIB, false), "its cold weights make the room");
    let admitted = ledger.admit(8 * GIB, &mut || panic!("cold weights are asked for, no executor ended")).unwrap();
    assert_eq!(admitted, Admitted::Now);
    assert!(ledger.free().unwrap() >= 8 * GIB);
    let (_, resident) = ledger.row(first.pid).expect("the first executor lives on");
    assert_eq!(resident, 15 * GIB + GIB / 2, "it gave exactly what was missing");
}

/// The 11405 shape: an executor holding 7 GiB of an 8 GiB card in torch is killed. Its bytes
/// are counted until NVML shows them back, and nothing is admitted beside them meanwhile.
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
    let mut held = Tenant::start(&ledger, &charges(), &format!("{},0,0,{}", GIB / 2, 6 * GIB + GIB / 2), "answer");
    ledger.now();
    assert_eq!(ledger.free().unwrap(), GIB);
    held.kill();
    let ended = std::cell::Cell::new(0);
    let admitted = ledger.admit(2 * GIB, &mut || {
        ended.set(ended.get() + 1);
        false
    });
    assert_eq!(admitted.unwrap(), Admitted::Short(GIB), "nothing beside a dead executor's bytes");
    assert_eq!(ended.get(), 1, "an idle executor would have been ended for the room");
    assert_eq!(ledger.free().unwrap(), GIB);
    release(held.pid);
    assert_eq!(ledger.admit(2 * GIB, &mut || false).unwrap(), Admitted::Now);
    assert_eq!(ledger.free().unwrap(), 8 * GIB);
}

/// A victim that never answers is named stalled once its progress has not moved for a wedge
/// floor, and ended; the start it held up then goes ahead.
#[test]
fn a_stalled_victim_is_ended_and_the_start_goes_ahead() {
    under_fake_nvml("inner_a_stalled_victim_is_ended", 8 * GIB, false);
}

#[test]
#[ignore = "run under the fake NVML by its outer test"]
fn inner_a_stalled_victim_is_ended() {
    let ledger = ledger(Liveness { sample: Duration::from_millis(50) });
    let stalled = Tenant::start(&ledger, &charges(), &format!("{},{},{},0", GIB / 2, 6 * GIB + GIB / 2, 6 * GIB), "stall");
    let ended: Arc<Mutex<Vec<u32>>> = Arc::default();
    let seen = ended.clone();
    ledger.on_stalled(Box::new(move |pid| {
        seen.lock().unwrap().push(pid);
        // SAFETY: a signal to the tenant this test started, alive until its Tenant drops.
        unsafe { libc::kill(pid as i32, libc::SIGKILL) };
        release(pid);
    }));
    ledger.now();
    assert_eq!(ledger.free().unwrap(), GIB);
    let admitted = ledger.admit(4 * GIB, &mut || false).unwrap();
    assert_eq!(admitted, Admitted::Now);
    assert_eq!(*ended.lock().unwrap(), [stalled.pid]);
}

/// A group's ranks each hold a row in their own GPU's ledger (no rank cells): a follower that
/// dies is reaped on its GPU alone. No NVML: each ledger's capacity is the configured one.
#[test]
fn each_rank_of_a_group_holds_its_own_gpus_row() {
    let ledgers: Vec<GpuLedger> = ["GPU-rank-0", "GPU-rank-1"]
        .iter()
        .map(|entry| GpuLedger::start(entry, false, 0, Some(8 * GIB), Liveness::default(), MemoryLog::default()).unwrap().unwrap())
        .collect();
    let dir = tempfile::tempdir().unwrap();
    let ranks: Vec<Tenant> = ledgers.iter().map(|l| Tenant::start(l, dir.path(), &format!("{},{},0,{}", GIB / 2, 2 * GIB, GIB), "answer")).collect();
    for ledger in &ledgers {
        ledger.now();
        assert_eq!(ledger.free().unwrap(), 8 * GIB - GIB / 2 - 3 * GIB);
    }
    assert!(ledgers[0].row(ranks[0].pid).is_some() && ledgers[0].row(ranks[1].pid).is_none());
    assert!(ledgers[1].row(ranks[1].pid).is_some() && ledgers[1].row(ranks[0].pid).is_none());
    let [rank0, mut rank1]: [Tenant; 2] = ranks.try_into().ok().unwrap();
    rank1.kill();
    ledgers[1].now();
    assert_eq!(ledgers[1].free().unwrap(), 8 * GIB, "the follower's GPU has its bytes back");
    ledgers[0].now();
    assert_eq!(ledgers[0].free().unwrap(), 8 * GIB - GIB / 2 - 3 * GIB, "rank 0's GPU is untouched");
    drop(rank0);
}
