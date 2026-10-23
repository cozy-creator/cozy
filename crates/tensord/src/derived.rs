//! The derived-weights disk cache (design varena-leases §6): weights a GPU transform produced (a
//! merged LoRA, a quantization from another source), kept on disk so a refill after a reclaim reads
//! them through the page cache rather than deriving them again. Each entry is a file unlinked from
//! birth (`O_TMPFILE`, else open and unlink at once; Windows `FILE_FLAG_DELETE_ON_CLOSE`), so its
//! holders are its open descriptors: TensorD's while it keeps the entry, and each process it hands
//! one to. The OS frees it at the last close, so a crash leaves nothing behind: no sweep, no TTL,
//! no size cap. It is written nothing below the disk's reserve, and under it the cache goes first:
//! TensorD lets its own holders go, least recently used first. Entries live per trust domain
//! (package generation and user) and are never shared across domains.
use crate::reclaim::Disk;
use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io,
    path::{Path, PathBuf},
    sync::Mutex,
    time::Instant,
};

pub struct Derived {
    root: PathBuf,
    entries: Mutex<BTreeMap<(String, String), Entry>>,
}

struct Entry {
    file: File,
    bytes: u64,
    used: Instant,
    /// Its writer reported it whole: only then is it read.
    written: bool,
}

impl Derived {
    pub fn at(root: PathBuf) -> Derived {
        Derived { root, entries: Mutex::default() }
    }

    /// A new entry of `domain` named `key` (its base content, transform and parameters, LoRAs and
    /// strengths, lane) for `bytes`, held by TensorD: the descriptor the executor that derived it
    /// writes, in that domain's directory (TensorD's alone). None while the disk could not keep it
    /// free of its reserve.
    pub fn create(&self, domain: &str, key: &str, bytes: u64, disk: &Disk) -> io::Result<Option<File>> {
        if disk.free() < bytes {
            return Ok(None);
        }
        let dir = self.root.join(tensorfs::sha256::hex_digest(domain.as_bytes()));
        fs::create_dir_all(&dir)?;
        #[cfg(unix)]
        fs::set_permissions(&dir, std::os::unix::fs::PermissionsExt::from_mode(0o700))?;
        let file = unlinked(&dir)?;
        file.set_len(bytes)?;
        let writer = file.try_clone()?;
        let entry = Entry { file, bytes, used: Instant::now(), written: false };
        self.entries().insert((domain.into(), key.into()), entry);
        Ok(Some(writer))
    }

    /// Its writer wrote it whole: from now on it is read.
    pub fn written(&self, domain: &str, key: &str) {
        if let Some(entry) = self.entries().get_mut(&(domain.into(), key.into())) {
            entry.written = true;
        }
    }

    /// A written entry's descriptor for a process of its domain that reads it, while TensorD
    /// keeps it.
    pub fn open(&self, domain: &str, key: &str) -> Option<File> {
        let mut entries = self.entries();
        let entry = entries.get_mut(&(domain.into(), key.into())).filter(|entry| entry.written)?;
        entry.used = Instant::now();
        entry.file.try_clone().ok()
    }

    /// Lets TensorD's holders go, least recently used first, until `bytes` are let go (a low
    /// disk): an entry is freed once the processes holding it close it too. What it let go.
    pub fn shrink(&self, bytes: u64) -> u64 {
        let mut entries = self.entries();
        let mut oldest: Vec<((String, String), Instant)> = entries.iter().map(|(key, entry)| (key.clone(), entry.used)).collect();
        oldest.sort_by_key(|(_, used)| *used);
        let mut let_go = 0;
        for (key, _) in oldest {
            if let_go >= bytes {
                break;
            }
            let_go += entries.remove(&key).map_or(0, |entry| entry.bytes);
        }
        let_go
    }

    /// The bytes TensorD keeps: unlinked, so no directory listing shows them (statvfs does).
    pub fn bytes(&self) -> u64 {
        self.entries().values().map(|entry| entry.bytes).sum()
    }

    fn entries(&self) -> std::sync::MutexGuard<'_, BTreeMap<(String, String), Entry>> {
        self.entries.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// A new file in `dir` no name reaches: `O_TMPFILE` where the filesystem has it.
#[cfg(target_os = "linux")]
fn unlinked(dir: &Path) -> io::Result<File> {
    use std::os::unix::fs::OpenOptionsExt;
    match OpenOptions::new().read(true).write(true).custom_flags(libc::O_TMPFILE).mode(0o600).open(dir) {
        Ok(file) => Ok(file),
        Err(error) if matches!(error.raw_os_error(), Some(libc::EOPNOTSUPP | libc::EISDIR | libc::EINVAL)) => named_then_unlinked(dir),
        Err(error) => Err(error),
    }
}

#[cfg(all(unix, not(target_os = "linux")))]
fn unlinked(dir: &Path) -> io::Result<File> {
    named_then_unlinked(dir)
}

/// Windows: deleted at its last handle's close; its name is never reopened.
#[cfg(windows)]
fn unlinked(dir: &Path) -> io::Result<File> {
    use std::os::windows::fs::OpenOptionsExt;
    const FILE_FLAG_DELETE_ON_CLOSE: u32 = 0x0400_0000;
    const SHARE_ALL: u32 = 0x7;
    let path = dir.join(uuid::Uuid::new_v4().simple().to_string());
    OpenOptions::new().read(true).write(true).create_new(true).custom_flags(FILE_FLAG_DELETE_ON_CLOSE).share_mode(SHARE_ALL).open(path)
}

/// Created under a fresh name and unlinked at once (a filesystem without `O_TMPFILE`).
#[cfg(unix)]
fn named_then_unlinked(dir: &Path) -> io::Result<File> {
    use std::os::unix::fs::OpenOptionsExt;
    let path = dir.join(format!(".derived-{}", uuid::Uuid::new_v4().simple()));
    let file = OpenOptions::new().read(true).write(true).create_new(true).mode(0o600).open(&path)?;
    fs::remove_file(&path)?;
    Ok(file)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Seek, SeekFrom, Write};

    fn roomy() -> Disk {
        Disk { capacity: 1 << 40, available: 1 << 39, inodes: 0, available_inodes: 0 }
    }

    #[test]
    fn an_entry_is_held_by_its_descriptors_alone_and_goes_first_under_the_reserve() {
        let dir = tempfile::tempdir().unwrap();
        let derived = Derived::at(dir.path().join("derived"));
        let mut writer = derived.create("gen/1000", "lora-a", 4, &roomy()).unwrap().expect("room");
        writer.write_all(b"bake").unwrap();
        assert!(derived.open("gen/1000", "lora-a").is_none(), "unwritten: never read");
        derived.written("gen/1000", "lora-a");
        let listed = |root: &Path| fs::read_dir(root).unwrap().flatten().flat_map(|d| fs::read_dir(d.path()).unwrap().flatten()).count();
        assert_eq!(listed(&dir.path().join("derived")), 0, "no name reaches an entry");
        let mut reader = derived.open("gen/1000", "lora-a").expect("TensorD keeps it");
        let mut bytes = vec![];
        reader.seek(SeekFrom::Start(0)).unwrap();
        reader.read_to_end(&mut bytes).unwrap();
        assert_eq!(bytes, b"bake");
        assert!(derived.open("gen/2000", "lora-a").is_none(), "another trust domain never reads it");
        assert_eq!(derived.bytes(), 4);

        derived.create("gen/1000", "lora-b", 8, &roomy()).unwrap().unwrap();
        derived.written("gen/1000", "lora-b");
        derived.open("gen/1000", "lora-a");
        assert_eq!(derived.shrink(1), 8, "the least recently used goes first");
        assert!(derived.open("gen/1000", "lora-b").is_none() && derived.open("gen/1000", "lora-a").is_some());
        assert_eq!(derived.shrink(u64::MAX), 4);
        // A holder outlives TensorD's letting go: the bytes stay until it closes.
        reader.seek(SeekFrom::Start(0)).unwrap();
        let mut again = vec![];
        reader.read_to_end(&mut again).unwrap();
        assert_eq!(again, b"bake");

        let low = Disk { capacity: 100 << 30, available: 2 << 30, inodes: 0, available_inodes: 0 };
        assert!(derived.create("gen/1000", "lora-c", 1 << 30, &low).unwrap().is_none(), "nothing is written below the reserve");
    }
}
