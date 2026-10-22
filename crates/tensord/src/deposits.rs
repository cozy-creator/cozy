//! Host layouts executors deposit (design memory-varena §5.4): sealed memfds varena filled in an
//! executor, kept by TensorD as opaque handles under (manifest, layout), so a later executor of
//! the same weights maps them from RAM (D9's refill path). A layout goes back to the host only
//! once no live executor it was handed to maps it (releasing one still mapped frees nothing),
//! the least recently granted first.
use platform::process::Birth;
use std::{collections::BTreeMap, fs::File, io, sync::Mutex, time::Instant};

#[derive(Default)]
pub struct Deposits {
    held: Mutex<BTreeMap<(String, String), Held>>,
    /// Layouts a call's rank was admitted to seal, by `call#rank`: until its deposit lands (or the
    /// call ends), any other asker waits for that one copy instead of a second.
    sealing: Mutex<BTreeMap<(String, String), String>>,
}

/// One kept layout: when it was last granted, the call that deposited it, and every executor
/// process it was handed to.
struct Held {
    file: File,
    used: Instant,
    call: String,
    holders: Vec<Birth>,
}

impl Held {
    fn mapped(&self) -> bool {
        self.holders.iter().any(|birth| birth.ended().is_ok_and(|ended| !ended))
    }
}

fn key(manifest: &str, layout: &str) -> (String, String) {
    (manifest.trim_start_matches("sha256:").into(), layout.into())
}

fn call_of(who: &str) -> &str {
    who.rsplit_once('#').map_or(who, |(call, _)| call)
}

impl Deposits {
    /// Keeps `layout` that `call`'s executor (`holders`) deposited, refused unless sealed against
    /// writes and resizing; a second deposit of one layout replaces the first.
    pub fn deposit(&self, manifest: &str, layout: &str, file: File, call: &str, holders: &[Birth]) -> io::Result<()> {
        self.sealing.lock().unwrap().remove(&key(manifest, layout));
        sealed(&file)?;
        let held = Held { file, used: Instant::now(), call: call.into(), holders: holders.to_vec() };
        self.held.lock().unwrap().insert(key(manifest, layout), held);
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

    /// `who` (`call#rank`) seals `layout` now when `admitted`, unless another still does: that
    /// one, named. Whether `who` seals it.
    pub fn seal(&self, manifest: &str, layout: &str, who: &str, admitted: bool) -> Result<bool, String> {
        let mut sealing = self.sealing.lock().unwrap();
        match sealing.get(&key(manifest, layout)) {
            Some(other) if other != who => Err(other.clone()),
            _ if admitted => {
                sealing.insert(key(manifest, layout), who.into());
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    /// Call `call` ended: a layout it was sealing and never deposited is nobody's to wait for.
    pub fn ended(&self, call: &str) {
        self.sealing.lock().unwrap().retain(|_, who| call_of(who) != call);
    }

    /// Whether a layout of another call than `call` is being sealed or filled: one call's tier
    /// fill at a time, so no two are admitted on the same surplus.
    pub fn filling_beside(&self, call: &str) -> bool {
        self.sealing.lock().unwrap().values().any(|who| call_of(who) != call)
            || self.held.lock().unwrap().values().any(|h| h.call != call && reopen(&h.file).is_some_and(|f| state(&f).0))
    }

    /// Bytes of the layouts no live executor maps: what releasing them gives back.
    pub fn idle_bytes(&self) -> u64 {
        self.held.lock().unwrap().values().filter(|h| !h.mapped()).map(|h| allocated(&h.file)).sum()
    }

    /// Lets the least recently granted layout no live executor maps go; whether one went.
    pub fn release_one(&self) -> bool {
        let mut held = self.held.lock().unwrap();
        let idle = held.iter().filter(|(_, h)| !h.mapped());
        let oldest = idle.min_by_key(|(_, h)| h.used).map(|(key, _)| key.clone());
        oldest.and_then(|key| held.remove(&key)).is_some()
    }
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

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;

    fn layout(bytes: u64) -> File {
        let layout = File::from(platform::shm::memfd_named("layout").unwrap());
        layout.set_len(bytes).unwrap();
        io::Write::write_all(&mut &layout, &vec![1u8; bytes as usize]).unwrap();
        platform::shm::seal(std::os::fd::AsFd::as_fd(&layout)).unwrap();
        layout
    }

    /// varena seals a layout it filled with future-write: kept, and granted read-only. An unsealed
    /// one is refused.
    #[test]
    fn a_layout_varena_sealed_is_kept_and_an_unsealed_one_refused() {
        let deposits = Deposits::default();
        let unsealed = File::from(platform::shm::memfd_named("unsealed").unwrap());
        assert!(deposits.deposit("sha256:m", "l", unsealed, "run-a", &[]).is_err());
        deposits.deposit("sha256:m", "l", layout(4096), "run-a", &[]).unwrap();
        let (granted, filling) = deposits.grant("m", "l", &[]).expect("held");
        assert!(!filling, "no filler holds it");
        assert_eq!(granted.metadata().unwrap().len(), 4096);
        let path = format!("/proc/self/fd/{}", std::os::fd::AsRawFd::as_raw_fd(&granted));
        let written = std::fs::OpenOptions::new().write(true).open(path).and_then(|mut f| io::Write::write_all(&mut f, b"x"));
        assert!(written.is_err(), "nobody writes a deposited layout");
    }

    /// One copy per layout: while a call's admitted rank seals it, another asker is told who, until
    /// the deposit lands or that call ends; the sealer itself may ask again. An asker not admitted
    /// reserves nothing.
    #[test]
    fn a_layout_has_one_sealer_until_its_deposit_or_its_call_ends() {
        let deposits = Deposits::default();
        assert_eq!(deposits.seal("sha256:m", "l", "run-z#0", false), Ok(false));
        assert_eq!(deposits.seal("sha256:m", "l", "run-a#0", true), Ok(true));
        assert_eq!(deposits.seal("m", "l", "run-a#1", true), Err("run-a#0".into()), "rank 1 waits for rank 0");
        assert_eq!(deposits.seal("m", "l", "run-b#0", false), Err("run-a#0".into()), "another call waits too");
        assert!(deposits.filling_beside("run-b") && !deposits.filling_beside("run-a"));
        deposits.seal("m", "other", "run-b#0", true).unwrap();
        deposits.deposit("m", "l", layout(4096), "run-a", &[]).unwrap();
        deposits.seal("m", "l", "run-c#0", true).unwrap();
        deposits.ended("run-c");
        deposits.seal("m", "l", "run-d#1", true).unwrap();
        assert_eq!(deposits.seal("m", "other", "run-d#1", true), Err("run-b#0".into()));
        deposits.ended("run-b");
        deposits.seal("m", "other", "run-d#1", true).unwrap();
    }

    /// A layout a live executor maps stays (releasing it frees nothing); once every executor it was
    /// handed to has ended, it goes back, the least recently granted first.
    #[test]
    fn only_a_layout_no_live_executor_maps_goes_back() {
        let deposits = Deposits::default();
        let mut executor = std::process::Command::new("sleep").arg("600").spawn().unwrap();
        let live = Birth::of(executor.id()).unwrap();
        deposits.deposit("m", "old", layout(8192), "run-a", &[]).unwrap();
        deposits.deposit("m", "mapped", layout(4096), "run-b", std::slice::from_ref(&live)).unwrap();
        deposits.grant("m", "old", &[]).unwrap();
        assert_eq!(deposits.idle_bytes(), 8192);
        assert!(deposits.release_one());
        assert!(!deposits.release_one(), "a live executor maps the other");
        assert_eq!(deposits.idle_bytes(), 0);
        executor.kill().unwrap();
        executor.wait().unwrap();
        assert_eq!(deposits.idle_bytes(), 4096);
        assert!(deposits.release_one());
        assert!(deposits.grant("m", "mapped", &[]).is_none());
    }
}
