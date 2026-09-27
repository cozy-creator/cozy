//! How many transfer streams fit in the memory this process can actually use. There is no
//! assumed host: a 4 GB pod runs the same pull, source download or upload window with fewer
//! streams, never with N × buffers it does not have.
use std::fs;
use std::path::{Path, PathBuf};

/// The most memory one transfer stream holds, whatever its object's size: one 1 MiB body
/// chunk (the read, write, hash or upload buffer), the 64 KiB framing buffer, and rustls's
/// bounded record buffers, rounded up to a whole 2 MiB.
pub const STREAM_MEMORY: u64 = 2 << 20;

/// Bytes this process can still use now: the tightest of its cgroup's limit less the usage
/// that cannot be reclaimed (v2 at every ancestor, else v1) and the host's `MemAvailable`.
/// `None` when none of them is readable.
pub fn memory_available() -> Option<u64> {
    [cgroup_v2(), cgroup_v1(), mem_available()]
        .into_iter()
        .flatten()
        .min()
}

/// Concurrent transfer streams the memory available now allows: at most `wanted`, at
/// least one.
pub fn transfer_streams(wanted: usize) -> usize {
    streams_within(wanted, memory_available())
}

/// `transfer_streams` against a stated budget.
pub fn streams_within(wanted: usize, available: Option<u64>) -> usize {
    let fits = available.map_or(usize::MAX, |bytes| {
        usize::try_from(bytes / STREAM_MEMORY).unwrap_or(usize::MAX)
    });
    wanted.min(fits).max(1)
}

fn number(path: &Path) -> Option<u64> {
    fs::read_to_string(path).ok()?.trim().parse().ok()
}

/// `limit − (usage − page cache)`: page cache is reclaimed under pressure, so bytes a
/// download has written do not count against the next stream.
fn headroom(dir: &Path, limit: &str, usage: &str, cache: &str) -> Option<u64> {
    let limit = number(&dir.join(limit))?;
    let usage = number(&dir.join(usage))?;
    let cached = fs::read_to_string(dir.join("memory.stat"))
        .ok()
        .and_then(|stat| {
            stat.lines()
                .find_map(|line| line.strip_prefix(cache)?.strip_prefix(' ')?.parse().ok())
        })
        .unwrap_or(0);
    Some(limit.saturating_sub(usage.saturating_sub(cached)))
}

fn cgroup_v2() -> Option<u64> {
    let root = Path::new("/sys/fs/cgroup");
    let own = fs::read_to_string("/proc/self/cgroup").ok()?;
    let own = own.lines().find_map(|line| line.strip_prefix("0::"))?;
    let mut dir = root.join(own.trim().trim_start_matches('/'));
    let mut tightest: Option<u64> = None;
    loop {
        if let Some(left) = headroom(&dir, "memory.max", "memory.current", "file") {
            tightest = Some(tightest.map_or(left, |held| held.min(left)));
        }
        if dir == root || !dir.pop() || !dir.starts_with(root) {
            return tightest;
        }
    }
}

fn cgroup_v1() -> Option<u64> {
    let dir = PathBuf::from("/sys/fs/cgroup/memory");
    headroom(
        &dir,
        "memory.limit_in_bytes",
        "memory.usage_in_bytes",
        "total_cache",
    )
}

fn mem_available() -> Option<u64> {
    let text = fs::read_to_string("/proc/meminfo").ok()?;
    let kib: u64 = text
        .lines()
        .find_map(|line| line.strip_prefix("MemAvailable:"))?
        .trim()
        .strip_suffix("kB")?
        .trim()
        .parse()
        .ok()?;
    kib.checked_mul(1024)
}
