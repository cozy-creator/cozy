//! Live host memory: what this process can still allocate before its tightest limit. Read on
//! every decision, never cached: a budget fixed at boot describes a machine nobody else is on.
//! Linux ports Runtime `proctree.host_memory` (cgroup v1 and v2, every parent, and
//! `MemAvailable`); macOS reads the VM statistics, Windows the global memory status.
use serde::Serialize;
use std::io;

#[cfg_attr(target_os = "linux", path = "host_memory_linux.rs")]
#[cfg_attr(target_os = "macos", path = "host_memory_macos.rs")]
#[cfg_attr(windows, path = "host_memory_windows.rs")]
mod sys;

#[cfg(target_os = "linux")]
pub(crate) use sys::{cgroups, number};

#[derive(Clone, Copy, Debug, Default, Serialize, PartialEq, Eq)]
pub struct HostMemory {
    /// Bytes left before the tightest limit on this process's cgroup path, or the host's
    /// available memory when that is tighter. -1: unreadable (never 0).
    pub available: i64,
    /// Shared memory (memfd/tmpfs) charged where that limit is: sealed and pinned weight tiers.
    /// 0 where weights are never held in shared memory (macOS, Windows).
    pub shmem: i64,
    /// The host's available memory (`MemAvailable` on Linux), for the record; -1 when unreadable.
    pub mem_available: i64,
    /// The limit `available` is measured against: the tightest cgroup's, else physical memory.
    pub bound: i64,
}

/// Physical memory; -1 when unreadable.
pub fn total() -> i64 {
    sys::total()
}

/// The live budget: the tightest of the host's available memory and, on Linux, every cgroup
/// on this process's path.
pub fn read() -> HostMemory {
    sys::read()
}

/// One process's resident and proportional set sizes: Linux's PSS counts a shared sealed layout
/// once across every process that maps it; macOS gives its physical footprint and Windows its
/// private commit there, with no shared part.
#[derive(Clone, Copy, Debug, Default, Serialize, PartialEq, Eq)]
pub struct ProcessMemory {
    pub rss: u64,
    pub pss: u64,
    pub pss_shmem: u64,
}

pub fn process(pid: u32) -> io::Result<ProcessMemory> {
    sys::process(pid)
}

/// One process's anonymous bytes now, and a conservative upper bound on their high-water
/// since it started or since [`reset_peak`]. Linux exposes a total resident peak, not a separate
/// anonymous peak; current file/shared pages cannot be subtracted from that historical value.
/// Linux only (the host tier's OS); elsewhere `Unsupported`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Anon {
    pub now: u64,
    pub peak: u64,
}

pub fn anon(pid: u32) -> io::Result<Anon> {
    sys::anon(pid)
}

/// Starts `pid`'s high-water over from its use now.
pub fn reset_peak(pid: u32) -> io::Result<()> {
    sys::reset_peak(pid)
}

/// Private bytes (PSS less shared pages) of `pid` and every process descended from it: what a
/// process tree holds that no one else counts.
pub fn tree_private(pid: u32) -> u64 {
    std::iter::once(pid)
        .chain(crate::process::descendants(pid))
        .filter_map(|pid| process(pid).ok())
        .map(|m| m.pss.saturating_sub(m.pss_shmem))
        .sum()
}

#[cfg(test)]
mod tests {
    #[test]
    fn every_os_reads_a_budget_within_its_memory_and_a_tree_holds_its_children() {
        let (facts, total) = (super::read(), super::total());
        assert!(total > 0, "{total}");
        assert!(
            facts.available > 0 && facts.available <= total,
            "{facts:?} of {total}"
        );
        assert!(facts.mem_available >= facts.available, "{facts:?}");
        let own = super::process(std::process::id()).unwrap();
        assert!(own.rss > 0 && own.pss > 0, "{own:?}");
        let mut child = std::process::Command::new(if cfg!(windows) { "cmd" } else { "sleep" })
            .args(if cfg!(windows) {
                &["/C", "ping -n 30 127.0.0.1 >NUL"][..]
            } else {
                &["30"][..]
            })
            .spawn()
            .unwrap();
        let child_private = super::process(child.id()).map(|m| m.pss - m.pss_shmem);
        let tree = super::tree_private(std::process::id());
        child.kill().unwrap();
        child.wait().unwrap();
        assert!(child_private.unwrap() > 0);
        assert!(
            tree > own.pss.saturating_sub(own.pss_shmem),
            "a child counts too: {tree}"
        );
    }

    /// A process's anonymous high-water holds what it touched and freed, until it starts over.
    #[cfg(target_os = "linux")]
    #[test]
    fn an_anonymous_high_water_holds_a_freed_peak_until_reset() {
        let script = "b = bytearray(64 << 20); b[::4096] = b'\\1' * len(b[::4096]); del b\nprint('freed', flush=True)\nimport time; time.sleep(60)";
        let mut child = std::process::Command::new("python3")
            .args(["-c", script])
            .stdout(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        let mut line = String::new();
        std::io::BufRead::read_line(&mut std::io::BufReader::new(child.stdout.take().unwrap()), &mut line).unwrap();
        let held = super::anon(child.id()).unwrap();
        assert!(held.peak >= 64 << 20 && held.now < 64 << 20, "{held:?}");
        super::reset_peak(child.id()).unwrap();
        let reset = super::anon(child.id()).unwrap();
        child.kill().unwrap();
        child.wait().unwrap();
        assert!(reset.peak < 64 << 20 && reset.peak >= reset.now, "{reset:?}");
    }

    /// Later file residency must not erase an earlier anonymous peak. Both observations use
    /// a real child process; its current RSS composition changes while its high-water remains.
    #[cfg(target_os = "linux")]
    #[test]
    fn file_pages_do_not_subtract_from_an_earlier_anonymous_peak() {
        use std::io::{BufRead, Write};
        let script = r#"
import mmap, sys, tempfile
anonymous = bytearray(64 << 20)
print('anonymous', flush=True)
sys.stdin.readline()
del anonymous
file = tempfile.TemporaryFile()
file.truncate(48 << 20)
mapped = mmap.mmap(file.fileno(), 0)
for at in range(0, len(mapped), 4096):
    mapped[at] = 1
print('file', flush=True)
sys.stdin.readline()
"#;
        let mut child = std::process::Command::new("python3")
            .args(["-c", script])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn().unwrap();
        let mut input = child.stdin.take().unwrap();
        let mut output = std::io::BufReader::new(child.stdout.take().unwrap());
        let mut line = String::new();
        output.read_line(&mut line).unwrap();
        let anonymous = super::anon(child.id()).unwrap();
        input.write_all(b"next\n").unwrap();
        line.clear();
        output.read_line(&mut line).unwrap();
        let file_backed = super::anon(child.id()).unwrap();
        let mapped = super::process(child.id()).unwrap();
        drop(input);
        assert!(child.wait().unwrap().success());
        assert!(anonymous.now >= 64 << 20, "{anonymous:?}");
        assert!(file_backed.now < 32 << 20 && mapped.rss > 48 << 20, "{file_backed:?}, {mapped:?}");
        assert!(file_backed.peak >= anonymous.now, "later file residency must preserve the anonymous peak: {anonymous:?} -> {file_backed:?}");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn the_live_budget_is_read_and_never_above_mem_available() {
        let facts = super::read();
        assert!(facts.mem_available > 0, "{facts:?}");
        assert!(
            facts.available > 0 && facts.available <= facts.mem_available,
            "{facts:?}"
        );
        let own = super::process(std::process::id()).unwrap();
        assert!(own.rss > 0 && own.pss > 0 && own.pss <= own.rss, "{own:?}");
        // A quiet tree, not this test process, whose other threads move its memory between
        // two reads: a shell waiting on the `sleep` it started.
        let mut shell = std::process::Command::new("sh")
            .args(["-c", "sleep 60 & echo $!; wait"])
            .stdout(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        let mut pid = String::new();
        std::io::BufRead::read_line(
            &mut std::io::BufReader::new(shell.stdout.take().unwrap()),
            &mut pid,
        )
        .unwrap();
        let sleeper: u32 = pid.trim().parse().unwrap();
        // Until the child has become `sleep`, it shares the shell's pages and their sizes move.
        while std::fs::read_to_string(format!("/proc/{sleeper}/comm"))
            .is_ok_and(|c| c.trim() != "sleep")
        {
            std::thread::yield_now();
        }
        let private = |pid: u32| {
            super::process(pid)
                .map(|m| m.pss.saturating_sub(m.pss_shmem))
                .unwrap()
        };
        let (alone, child) = (private(shell.id()), private(sleeper));
        let tree = super::tree_private(shell.id());
        if let Ok(Some(sleeping)) =
            crate::process::Birth::of(sleeper).and_then(|birth| crate::process::Exact::open(&birth))
        {
            let _ = sleeping.kill();
        }
        shell.wait().unwrap();
        // Shared library pages split their size among every process mapping them, so a quiet
        // process still moves by a few hundred bytes; the child's own pages are far more.
        assert!(
            child > 0 && tree > alone,
            "a child counts too: {tree} vs {alone} alone, {child} child"
        );
    }
}
