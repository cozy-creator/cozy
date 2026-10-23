//! Host layouts executors deposit (design memory-varena §5.4): sealed memfds varena filled in an
//! executor, kept by TensorD as opaque handles under (manifest, layout), so a later executor of
//! the same weights maps them from RAM (D9's refill path). A layout goes back to the host only
//! once no live executor maps or holds it (releasing one still in use frees nothing), the least
//! recently granted first. An admitted layout reserves its bytes until filled.
use platform::process::Birth;
use std::{collections::BTreeMap, fs::File, io, sync::Mutex, time::Instant};

#[derive(Default)]
pub struct Deposits {
    held: Mutex<BTreeMap<(String, String), Held>>,
    /// Layouts a `call#rank` seals and has not deposited yet, with the bytes admitted for it.
    /// This lock also serializes admissions and the handoff to the deposited, filling layout.
    sealing: Mutex<BTreeMap<(String, String), Pending>>,
}

struct Pending {
    who: String,
    bytes: u64,
}

/// One kept layout: when it was last granted and every executor process it was handed to.
struct Held {
    file: File,
    used: Instant,
    holders: Vec<Birth>,
}

impl Held {
    /// Whether a live executor it was handed to still maps it or holds it open.
    fn in_use(&self) -> bool {
        self.holders.iter().any(|birth| birth.ended().is_ok_and(|ended| !ended) && uses(birth.pid, &self.file))
    }

    /// What its filler has still to fill: an admitted fill's reservation.
    fn unfilled(&self) -> u64 {
        match reopen(&self.file).is_some_and(|file| state(&file).0) {
            true => self.file.metadata().map_or(0, |m| m.len()).saturating_sub(allocated(&self.file)),
            false => 0,
        }
    }
}

fn key(manifest: &str, layout: &str) -> (String, String) {
    (manifest.trim_start_matches("sha256:").into(), layout.into())
}

fn call_of(who: &str) -> &str {
    who.rsplit_once('#').map_or(who, |(call, _)| call)
}

impl Deposits {
    /// Keeps `layout` that the executors (`holders`) deposited, refused unless sealed against
    /// writes and resizing; a second deposit of one layout replaces the first.
    pub fn deposit(&self, manifest: &str, layout: &str, file: File, holders: &[Birth]) -> io::Result<()> {
        sealed(&file)?;
        let mut sealing = self.sealing.lock().unwrap();
        let held = Held { file, used: Instant::now(), holders: holders.to_vec() };
        self.held.lock().unwrap().insert(key(manifest, layout), held);
        sealing.remove(&key(manifest, layout));
        Ok(())
    }

    /// A read-only description of the deposited layout, if one is held, for `holders`, and whether
    /// its filler still fills it: an executor deposits a layout as it opens it, before its bytes.
    /// One its filler left unfinished (it died) goes, and the next asker seals anew.
    pub fn grant(&self, manifest: &str, layout: &str, holders: &[Birth]) -> Option<(File, bool)> {
        let mut held = self.held.lock().unwrap();
        let kept = held.get_mut(&key(manifest, layout))?;
        let file = reopen(&kept.file)?;
        let (filling, complete) = state(&file);
        if !filling && !complete {
            held.remove(&key(manifest, layout));
            return None;
        }
        kept.used = Instant::now();
        for birth in holders {
            if !kept.holders.contains(birth) {
                kept.holders.push(birth.clone());
            }
        }
        Some((file, filling))
    }

    /// Whether `who` (`call#rank`) seals `layout` of `bytes` (its checkpoint's): `admit` decides,
    /// beside the bytes admitted fills of other checkpoints have still to fill, under this lock. One copy per layout: while
    /// another seals it, that one is named and the asker waits for its deposit. False: not
    /// admitted, or a deposit landed since the caller looked (it should ask `grant` again).
    /// `regardless`: the asker seals whatever the answer (an older Runtime or TensorFS), so it is
    /// the layout's one sealer anyway and its bytes are reserved. `admit` may inspect and release
    /// held deposits, but must not re-enter sealing or deposit.
    pub fn seal(&self, manifest: &str, layout: &str, who: &str, bytes: u64, regardless: bool, admit: impl FnOnce(u64) -> bool) -> Result<bool, String> {
        let mut sealing = self.sealing.lock().unwrap();
        if let Some(other) = sealing.get(&key(manifest, layout)) {
            return if other.who == who { Ok(true) } else { Err(other.who.clone()) };
        }
        let held = self.held.lock().unwrap();
        if held.contains_key(&key(manifest, layout)) {
            return Ok(false);
        }
        let mut reserved = reserved(&sealing, &held);
        reserved.remove(&key(manifest, layout).0);
        drop(held);
        if !admit(reserved.values().sum()) && !regardless {
            return Ok(false);
        }
        sealing.insert(key(manifest, layout), Pending { who: who.into(), bytes });
        Ok(true)
    }

    /// Bytes admitted fills have still to fill (`reserved`).
    pub fn reserved(&self) -> u64 {
        let sealing = self.sealing.lock().unwrap();
        reserved(&sealing, &self.held.lock().unwrap()).values().sum()
    }

    /// Call `call` ended: a layout it was sealing and never deposited is nobody's to wait for.
    pub fn ended(&self, call: &str) {
        self.sealing.lock().unwrap().retain(|_, p| call_of(&p.who) != call);
    }

    /// Whether a layout of this digest is held.
    pub fn holds(&self, layout: &str) -> bool {
        self.held.lock().unwrap().keys().any(|(_, held)| held == layout)
    }

    /// Bytes of the layouts no live executor uses: what releasing them gives back.
    pub fn idle_bytes(&self) -> u64 {
        self.held.lock().unwrap().values().filter(|h| !h.in_use()).map(|h| allocated(&h.file)).sum()
    }

    /// Lets the least recently granted layout no live executor uses go; whether one went.
    pub fn release_one(&self) -> bool {
        let mut held = self.held.lock().unwrap();
        let idle = held.iter().filter(|(_, h)| !h.in_use());
        let oldest = idle.min_by_key(|(_, h)| h.used).map(|(key, _)| key.clone());
        oldest.and_then(|key| held.remove(&key)).is_some()
    }
}

/// What admitted fills have still to fill, by manifest: a layout sealing and not yet deposited
/// reserves its checkpoint's selected bytes, which cover every layout of that checkpoint, else
/// the deposited layouts' fillers what they have still to fill.
fn reserved(sealing: &BTreeMap<(String, String), Pending>, held: &BTreeMap<(String, String), Held>) -> BTreeMap<String, u64> {
    let mut by: BTreeMap<String, u64> = BTreeMap::new();
    for ((manifest, _), layout) in held {
        *by.entry(manifest.clone()).or_default() += layout.unfilled();
    }
    for ((manifest, _), pending) in sealing {
        let bytes = by.entry(manifest.clone()).or_default();
        *bytes = (*bytes).max(pending.bytes);
    }
    by
}

/// Whether `pid` maps `file` or holds it open (its /proc entries name the same inode). A process
/// whose /proc this one cannot read counts as using it.
#[cfg(target_os = "linux")]
fn uses(pid: u32, file: &File) -> bool {
    use std::os::unix::fs::MetadataExt;
    let Ok(of) = file.metadata() else { return true };
    let (Ok(maps), Ok(fds)) = (std::fs::read_to_string(format!("/proc/{pid}/maps")), std::fs::read_dir(format!("/proc/{pid}/fd"))) else {
        return true;
    };
    let device = format!("{:02x}:{:02x}", libc::major(of.dev()), libc::minor(of.dev()));
    let inode = of.ino().to_string();
    maps.lines().any(|line| line.split_whitespace().skip(3).take(2).eq([device.as_str(), inode.as_str()]))
        || fds.flatten().any(|fd| std::fs::metadata(fd.path()).is_ok_and(|m| m.dev() == of.dev() && m.ino() == of.ino()))
}
#[cfg(not(target_os = "linux"))]
fn uses(_: u32, _: &File) -> bool {
    true
}

/// The bytes a layout holds in RAM: a sparse memfd holds only what was filled.
#[cfg(target_os = "linux")]
fn allocated(file: &File) -> u64 {
    file.metadata().map_or(0, |m| std::os::unix::fs::MetadataExt::blocks(&m) * 512)
}
#[cfg(not(target_os = "linux"))]
fn allocated(_: &File) -> u64 {
    0
}
/// The seals varena's `adopt_sealed` asks of a layout: resize and seal, and write or future-write
/// (varena seals a layout it filled with future-write, its own mapping still open).
#[cfg(target_os = "linux")]
fn sealed(file: &File) -> io::Result<()> {
    use platform::shm::{RESIZE_SEALS, WRITE_SEALS};
    let seals = platform::shm::seals(std::os::fd::AsFd::as_fd(file))?;
    match seals & RESIZE_SEALS == RESIZE_SEALS && seals & WRITE_SEALS != 0 {
        true => Ok(()),
        false => Err(io::Error::new(io::ErrorKind::InvalidInput, "a deposited layout must be sealed against writes and resizing")),
    }
}
#[cfg(not(target_os = "linux"))]
fn sealed(_: &File) -> io::Result<()> {
    Err(io::Error::new(io::ErrorKind::Unsupported, "sealed host layouts are Linux's"))
}

/// Whether its filler still fills the layout, and whether every region is in (a file that is
/// no varena layout counts as complete).
#[cfg(target_os = "linux")]
fn state(file: &File) -> (bool, bool) {
    let fd = std::os::fd::AsFd::as_fd(file);
    (varena::host::filling(fd).unwrap_or(false), varena::host::complete(fd).unwrap_or(true))
}
#[cfg(not(target_os = "linux"))]
fn state(_: &File) -> (bool, bool) {
    (false, true)
}

#[cfg(target_os = "linux")]
fn reopen(file: &File) -> Option<File> {
    File::open(format!("/proc/self/fd/{}", std::os::fd::AsRawFd::as_raw_fd(file))).ok()
}
#[cfg(not(target_os = "linux"))]
fn reopen(_: &File) -> Option<File> {
    None
}
