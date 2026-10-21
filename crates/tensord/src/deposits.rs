//! Host layouts executors deposit (design memory-varena §5.4): sealed memfds varena filled in an
//! executor, kept by TensorD as opaque handles under (manifest, layout), so a later executor of
//! the same weights maps them from RAM (D9's refill path). No byte counts: under host pressure
//! the least recently granted goes first.
use std::{collections::BTreeMap, fs::File, io, sync::Mutex, time::Instant};

#[derive(Default)]
pub struct Deposits {
    held: Mutex<BTreeMap<(String, String), (File, Instant)>>,
    /// Layouts a call's rank was handed chunk files to seal, by `call#rank`: until its deposit
    /// lands (or the call ends), any other asker waits for that one copy instead of a second.
    sealing: Mutex<BTreeMap<(String, String), String>>,
}

fn key(manifest: &str, layout: &str) -> (String, String) {
    (manifest.trim_start_matches("sha256:").into(), layout.into())
}

impl Deposits {
    /// Keeps `layout`, refused unless sealed against writes and resizing; a second deposit of
    /// one layout replaces the first.
    pub fn deposit(&self, manifest: &str, layout: &str, file: File) -> io::Result<()> {
        self.sealing.lock().unwrap().remove(&key(manifest, layout));
        sealed(&file)?;
        self.held.lock().unwrap().insert(key(manifest, layout), (file, Instant::now()));
        Ok(())
    }

    /// A read-only description of the deposited layout, if one is held, and whether its filler
    /// still fills it: an executor deposits a layout as it opens it, before its bytes. One its
    /// filler left unfinished (it died) goes, and the next asker seals anew.
    pub fn grant(&self, manifest: &str, layout: &str) -> Option<(File, bool)> {
        let mut held = self.held.lock().unwrap();
        let (file, used) = held.get_mut(&key(manifest, layout))?;
        *used = Instant::now();
        let file = reopen(file)?;
        let (filling, complete) = state(&file);
        if !filling && !complete {
            held.remove(&key(manifest, layout));
            return None;
        }
        Some((file, filling))
    }

    /// `who` (`call#rank`) seals `layout` now, unless another still does: that one, named.
    pub fn seal(&self, manifest: &str, layout: &str, who: &str) -> Result<(), String> {
        let mut sealing = self.sealing.lock().unwrap();
        match sealing.get(&key(manifest, layout)) {
            Some(other) if other != who => Err(other.clone()),
            _ => {
                sealing.insert(key(manifest, layout), who.into());
                Ok(())
            }
        }
    }

    /// Call `call` ended: a layout it was sealing and never deposited is nobody's to wait for.
    pub fn ended(&self, call: &str) {
        self.sealing.lock().unwrap().retain(|_, who| who.rsplit_once('#').is_none_or(|(c, _)| c != call));
    }

    /// Lets the least recently granted layout go; whether one went.
    pub fn release_one(&self) -> bool {
        let mut held = self.held.lock().unwrap();
        let oldest = held.iter().min_by_key(|(_, (_, used))| *used).map(|(key, _)| key.clone());
        oldest.and_then(|key| held.remove(&key)).is_some()
    }
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

/// The SHA-256 (hex) of chunk objects in order, one `<sha256>:<length>\n` line each: what a
/// `read_grant` of chunk files names in place of the list.
pub fn objects_digest(objects: &[tensorfs::format::ObjectRef]) -> String {
    let lines: String = objects.iter().map(|o| format!("{}:{}\n", o.sha256, o.length)).collect();
    tensorfs::sha256::hex_digest(lines.as_bytes())
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;

    /// varena seals a layout it filled with future-write: kept, and granted read-only. An unsealed
    /// one is refused.
    #[test]
    fn a_layout_varena_sealed_is_kept_and_an_unsealed_one_refused() {
        let deposits = Deposits::default();
        let unsealed = File::from(platform::shm::memfd_named("unsealed").unwrap());
        assert!(deposits.deposit("sha256:m", "l", unsealed).is_err());
        let layout = File::from(platform::shm::memfd_named("layout").unwrap());
        layout.set_len(4096).unwrap();
        platform::shm::seal(std::os::fd::AsFd::as_fd(&layout)).unwrap();
        deposits.deposit("sha256:m", "l", layout).unwrap();
        let (granted, filling) = deposits.grant("m", "l").expect("held");
        assert!(!filling, "no filler holds it");
        assert_eq!(granted.metadata().unwrap().len(), 4096);
        let path = format!("/proc/self/fd/{}", std::os::fd::AsRawFd::as_raw_fd(&granted));
        let written = std::fs::OpenOptions::new().write(true).open(path).and_then(|mut f| io::Write::write_all(&mut f, b"x"));
        assert!(written.is_err(), "nobody writes a deposited layout");
    }

    /// One copy per layout: while a call's rank seals it, another asker is told who, until the
    /// deposit lands or that call ends; the sealer itself may ask again.
    #[test]
    fn a_layout_has_one_sealer_until_its_deposit_or_its_call_ends() {
        let deposits = Deposits::default();
        deposits.seal("sha256:m", "l", "run-a#0").unwrap();
        assert_eq!(deposits.seal("m", "l", "run-a#1"), Err("run-a#0".into()), "rank 1 waits for rank 0");
        assert_eq!(deposits.seal("m", "l", "run-b#0"), Err("run-a#0".into()), "another call waits too");
        deposits.seal("m", "l", "run-a#0").unwrap();
        deposits.seal("m", "other", "run-b#0").unwrap();
        let layout = File::from(platform::shm::memfd_named("layout").unwrap());
        layout.set_len(4096).unwrap();
        platform::shm::seal(std::os::fd::AsFd::as_fd(&layout)).unwrap();
        deposits.deposit("m", "l", layout).unwrap();
        deposits.seal("m", "l", "run-c#0").unwrap();
        deposits.ended("run-c");
        deposits.seal("m", "l", "run-d#1").unwrap();
        assert_eq!(deposits.seal("m", "other", "run-d#1"), Err("run-b#0".into()));
        deposits.ended("run-b");
        deposits.seal("m", "other", "run-d#1").unwrap();
    }
}
