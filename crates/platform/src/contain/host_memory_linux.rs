//! Linux: `/proc/meminfo`, every cgroup on this process's path, and `smaps_rollup`.
use super::{HostMemory, ProcessMemory};
use std::{
    fs,
    io::{self, BufRead},
    path::{Path, PathBuf},
};

/// The host's `MemTotal`; -1 when unreadable.
pub(super) fn total() -> i64 {
    meminfo("MemTotal")
}

fn meminfo(field: &str) -> i64 {
    let Ok(file) = fs::File::open("/proc/meminfo") else {
        return -1;
    };
    for line in io::BufReader::new(file).lines().map_while(Result::ok) {
        if let Some(rest) = line.strip_prefix(field).and_then(|r| r.strip_prefix(':')) {
            return rest
                .split_whitespace()
                .next()
                .and_then(|v| v.parse::<i64>().ok())
                .map_or(-1, |kib| kib * 1024);
        }
    }
    -1
}

pub(crate) fn number(path: &Path) -> Option<u64> {
    let text = fs::read_to_string(path).ok()?;
    let text = text.trim();
    // cgroup v2 spells "unlimited" `max`; v1 a page-rounded value near LONG_MAX
    text.parse::<u64>().ok().filter(|&n| n > 0 && n < (1 << 60))
}

/// (reclaimable, `shmem`) from a cgroup's `memory.stat`: the page cache the kernel gives back
/// before it OOM-kills is its inactive file pages less those still dirty or under writeback (active
/// ones may not go in time, dirty ones wait on the disk); v1 states its subtree's under `total_`.
fn stat(cgroup: &Path) -> Option<(u64, u64)> {
    let text = fs::read_to_string(cgroup.join("memory.stat")).ok()?;
    let row = |names: &[&str]| -> u64 {
        let find = |key: &str| {
            text.lines().find_map(|l| {
                l.strip_prefix(key)
                    .and_then(|r| r.strip_prefix(' '))
                    .and_then(|v| v.trim().parse().ok())
            })
        };
        names.iter().find_map(|name| find(&format!("total_{name}")).or_else(|| find(name))).unwrap_or(0)
    };
    let unwritten = row(&["file_dirty", "dirty"]) + row(&["file_writeback", "writeback"]);
    Some((row(&["inactive_file"]).saturating_sub(unwritten), row(&["shmem"])))
}

/// A process's memory cgroup and every parent up to the controller root, innermost first, with
/// the limit files and the usage file of its version.
pub(crate) type Cgroups = (Vec<PathBuf>, &'static [&'static str], &'static str);

/// This process's [`Cgroups`].
pub(crate) fn cgroups() -> Option<Cgroups> {
    let memberships = fs::read_to_string("/proc/self/cgroup").ok()?;
    cgroups_in(Path::new("/sys/fs/cgroup"), &memberships, || fs::read_to_string("/proc/self/mountinfo").ok())
}

/// The [`Cgroups`] of `memberships` (a `/proc/<pid>/cgroup`) under the cgroup mount `root`;
/// v1's memory controller is found in `mounts` (a `/proc/<pid>/mountinfo`).
fn cgroups_in(root: &Path, memberships: &str, mounts: impl FnOnce() -> Option<String>) -> Option<Cgroups> {
    let (leaf, top, limits, usage): (PathBuf, PathBuf, &[&str], &str) =
        if root.join("cgroup.controllers").is_file() {
            let relative = memberships.lines().find_map(|l| l.strip_prefix("0::"))?;
            (
                root.join(relative.trim_start_matches('/')),
                root.to_path_buf(),
                &["memory.high", "memory.max"],
                "memory.current",
            )
        } else {
            let relative = memberships.lines().find_map(|l| {
                let mut fields = l.splitn(3, ':');
                let (_, names, path) = (fields.next()?, fields.next()?, fields.next()?);
                names.split(',').any(|n| n == "memory").then_some(path)
            })?;
            // The memory controller's mount: its root (mountinfo field 4) is the part of the
            // membership path the mount hides (a container sees its own cgroup at the mount).
            let mounts = mounts()?;
            let (mount_root, point) = mounts.lines().find_map(|l| {
                let fields: Vec<&str> = l.split(' ').collect();
                let dash = fields.iter().position(|f| *f == "-")?;
                (fields.get(dash + 1) == Some(&"cgroup")
                    && fields.get(dash + 3)?.split(',').any(|o| o == "memory"))
                .then(|| (fields[3].to_string(), PathBuf::from(fields[4])))
            })?;
            let inside = relative
                .strip_prefix(mount_root.trim_end_matches('/'))
                .unwrap_or(relative);
            (
                point.join(inside.trim_start_matches('/')),
                point,
                &["memory.limit_in_bytes"],
                "memory.usage_in_bytes",
            )
        };
    let mut path = Vec::new();
    let mut at = Some(leaf.as_path());
    while let Some(cgroup) = at {
        path.push(cgroup.to_path_buf());
        if cgroup == top {
            break;
        }
        at = cgroup.parent().filter(|p| p.starts_with(&top));
    }
    Some((path, limits, usage))
}

/// The tightest limit set on any cgroup of `path`: None where none is.
fn tightest(path: &[PathBuf], limits: &[&str]) -> Option<u64> {
    path.iter().flat_map(|cgroup| limits.iter().filter_map(|name| number(&cgroup.join(name)))).min()
}

/// The tightest cgroup limit on this process's path.
pub(super) fn limit() -> Option<u64> {
    cgroups().and_then(|(path, limits, _)| tightest(&path, limits))
}

/// The live budget: the tightest of every cgroup on this process's path (each `memory.high`
/// and `memory.max`, less what that cgroup uses and cannot give back; inactive page cache is
/// reclaimable; a missing limit file is no limit, and a limit whose use is unreadable is all
/// room) and the host's `MemAvailable`.
pub(super) fn read() -> HostMemory {
    let mem_available = meminfo("MemAvailable");
    let mut best = HostMemory {
        available: mem_available,
        shmem: meminfo("Shmem").max(0),
        mem_available,
        bound: total(),
    };
    let Some((path, limits, usage)) = cgroups() else {
        return best;
    };
    for cgroup in path {
        let Some(bound) = tightest(std::slice::from_ref(&cgroup), limits) else {
            continue;
        };
        let used = number(&cgroup.join(usage)).unwrap_or(0);
        let (cache, shmem) = stat(&cgroup).unwrap_or((0, 0));
        let room = bound.saturating_sub(used.saturating_sub(cache)) as i64;
        if best.available < 0 || room < best.available {
            best = HostMemory {
                available: room,
                shmem: shmem as i64,
                mem_available,
                bound: bound as i64,
            };
        }
    }
    best
}

/// `/proc/<pid>/status`: `RssAnon`, `RssFile`, `RssShmem` and `VmHWM`.
pub(super) fn resident(pid: u32) -> io::Result<super::Resident> {
    let text = fs::read_to_string(format!("/proc/{pid}/status"))?;
    let field = |name: &str| {
        text.lines()
            .find_map(|l| l.strip_prefix(name)?.strip_prefix(':')?.split_whitespace().next()?.parse::<u64>().ok())
            .map(|kib| kib * 1024)
            .ok_or_else(|| io::Error::other(format!("no {name} in /proc/{pid}/status")))
    };
    let (anon, file, shmem) = (field("RssAnon")?, field("RssFile")?, field("RssShmem")?);
    Ok(super::Resident { anon, file, shmem, peak: field("VmHWM")?.max(anon + file + shmem) })
}

/// `clear_refs` 5: the resident high-water becomes the process's use now.
pub(super) fn reset_peak(pid: u32) -> io::Result<()> {
    fs::write(format!("/proc/{pid}/clear_refs"), b"5")
}

pub(super) fn process(pid: u32) -> io::Result<ProcessMemory> {
    let text = fs::read_to_string(format!("/proc/{pid}/smaps_rollup"))?;
    let field = |name: &str| {
        text.lines()
            .find_map(|l| {
                l.strip_prefix(name)?
                    .strip_prefix(':')?
                    .split_whitespace()
                    .next()?
                    .parse::<u64>()
                    .ok()
            })
            .map_or(0, |kib| kib * 1024)
    };
    Ok(ProcessMemory {
        rss: field("Rss"),
        pss: field("Pss"),
        pss_shmem: field("Pss_Shmem"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    /// A fake cgroup mount under the temp directory: `(file, contents)` rows.
    fn tree(name: &str, files: &[(&str, String)]) -> PathBuf {
        let root = std::env::temp_dir().join(format!("cozy-cgroup-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        for (file, text) in files {
            let path = root.join(file);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, text).unwrap();
        }
        root
    }

    fn limit_in(root: &Path, memberships: &str, mounts: &str) -> Option<u64> {
        cgroups_in(root, memberships, || Some(mounts.to_string())).and_then(|(path, limits, _)| tightest(&path, limits))
    }

    #[test]
    fn v2_takes_the_tightest_max_or_set_high_on_the_path() {
        let root = tree("v2", &[
            ("cgroup.controllers", "cpu memory".into()),
            ("ci/memory.max", (32 * GIB).to_string()),
            ("ci/job/memory.max", "max".into()),
            ("ci/job/memory.high", "max".into()),
        ]);
        assert_eq!(limit_in(&root, "0::/ci/job\n", ""), Some(32 * GIB), "a parent's max bounds its child");
        fs::write(root.join("ci/job/memory.high"), (24 * GIB).to_string()).unwrap();
        assert_eq!(limit_in(&root, "0::/ci/job\n", ""), Some(24 * GIB), "a set high is tighter");
        assert_eq!(limit_in(&root, "0::/\n", ""), None, "the root's own files are no limit");
        // A container's namespace: its cgroup is the mount.
        fs::write(root.join("memory.max"), (16 * GIB).to_string()).unwrap();
        assert_eq!(limit_in(&root, "0::/\n", ""), Some(16 * GIB));
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn v1_reads_the_memory_controllers_mount_and_no_unlimited_value() {
        let unlimited = "9223372036854771712".to_string();
        let root = tree("v1", &[
            ("memory/memory.limit_in_bytes", unlimited.clone()),
            ("memory/docker/abc/memory.limit_in_bytes", (32 * GIB).to_string()),
            ("container/memory.limit_in_bytes", (8 * GIB).to_string()),
        ]);
        let memberships = "12:cpu,cpuacct:/docker/abc\n9:memory:/docker/abc\n";
        let mount = |mount_root: &str, point: &Path| format!("30 25 0:27 {mount_root} {} rw,nosuid shared:9 - cgroup cgroup rw,memory\n", point.display());
        // The host's view: the whole hierarchy is mounted.
        assert_eq!(limit_in(&root, memberships, &mount("/", &root.join("memory"))), Some(32 * GIB));
        // A container's view: the mount is its own cgroup.
        assert_eq!(limit_in(&root, memberships, &mount("/docker/abc", &root.join("container"))), Some(8 * GIB));
        // No memory controller mounted: no limit read.
        assert_eq!(limit_in(&root, memberships, ""), None);
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn the_limit_is_never_above_physical_memory() {
        let physical = u64::try_from(total()).unwrap();
        assert!(super::super::limit().is_some_and(|limit| limit <= physical));
    }
}
