//! Host layouts executors deposit (design memory-varena §5.4): sealed memfds varena filled in an
//! executor, kept by TensorD as opaque handles under (manifest, layout), so a later executor of
//! the same weights maps them from RAM (D9's refill path). No byte counts: under host pressure
//! the least recently granted goes first.
use std::{collections::BTreeMap, fs::File, io, sync::Mutex, time::Instant};

#[derive(Default)]
pub struct Deposits {
    held: Mutex<BTreeMap<(String, String), (File, Instant)>>,
}

fn key(manifest: &str, layout: &str) -> (String, String) {
    (manifest.trim_start_matches("sha256:").into(), layout.into())
}

impl Deposits {
    /// Keeps `layout`, refused unless sealed against writes and resizing; a second deposit of
    /// one layout replaces the first.
    pub fn deposit(&self, manifest: &str, layout: &str, file: File) -> io::Result<()> {
        sealed(&file)?;
        self.held.lock().unwrap().insert(key(manifest, layout), (file, Instant::now()));
        Ok(())
    }

    /// A read-only description of the deposited layout, if one is held.
    pub fn grant(&self, manifest: &str, layout: &str) -> Option<File> {
        let mut held = self.held.lock().unwrap();
        let (file, used) = held.get_mut(&key(manifest, layout))?;
        *used = Instant::now();
        reopen(file)
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
    use cozy_platform::shm::{RESIZE_SEALS, WRITE_SEALS};
    let seals = cozy_platform::shm::seals(std::os::fd::AsFd::as_fd(file))?;
    match seals & RESIZE_SEALS == RESIZE_SEALS && seals & WRITE_SEALS != 0 {
        true => Ok(()),
        false => Err(io::Error::new(io::ErrorKind::InvalidInput, "a deposited layout must be sealed against writes and resizing")),
    }
}
#[cfg(not(target_os = "linux"))]
fn sealed(_: &File) -> io::Result<()> {
    Err(io::Error::new(io::ErrorKind::Unsupported, "sealed host layouts are Linux's"))
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
pub fn objects_digest(objects: &[tensorfs_core::format::ObjectRef]) -> String {
    let lines: String = objects.iter().map(|o| format!("{}:{}\n", o.sha256, o.length)).collect();
    tensorfs_core::sha256::hex_digest(lines.as_bytes())
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;

    /// varena seals a layout it filled with future-write: kept, and granted read-only. An unsealed
    /// one is refused.
    #[test]
    fn a_layout_varena_sealed_is_kept_and_an_unsealed_one_refused() {
        let deposits = Deposits::default();
        let unsealed = File::from(cozy_platform::shm::memfd_named("unsealed").unwrap());
        assert!(deposits.deposit("sha256:m", "l", unsealed).is_err());
        let layout = File::from(cozy_platform::shm::memfd_named("layout").unwrap());
        layout.set_len(4096).unwrap();
        cozy_platform::shm::seal(std::os::fd::AsFd::as_fd(&layout)).unwrap();
        deposits.deposit("sha256:m", "l", layout).unwrap();
        let granted = deposits.grant("m", "l").expect("held");
        assert_eq!(granted.metadata().unwrap().len(), 4096);
        let path = format!("/proc/self/fd/{}", std::os::fd::AsRawFd::as_raw_fd(&granted));
        let written = std::fs::OpenOptions::new().write(true).open(path).and_then(|mut f| io::Write::write_all(&mut f, b"x"));
        assert!(written.is_err(), "nobody writes a deposited layout");
    }
}
