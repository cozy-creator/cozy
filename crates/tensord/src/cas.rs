//! TensorD's run objects (inputs, outputs, trees, products, local package sources), apart from
//! the TensorFS store: `<root>/<aa>/<sha256>`, written to a temp file, hashed and renamed in,
//! never changed after. An object's mtime is its last use. A hold (`holds/<id>.json`) keeps its
//! roots until its owner releases it; a sweep removes unheld objects unused for its age.
use std::{
    collections::BTreeSet,
    fs::{self, File},
    io::{self, Read, Write},
    path::{Path, PathBuf},
    sync::RwLock,
    time::{Duration, SystemTime},
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tensorfs::format::ObjectRef;

/// A sweep decides and deletes while no use renews or holds an object.
static GATE: RwLock<()> = RwLock::new(());

#[derive(Clone, Debug)]
pub struct Cas {
    root: PathBuf,
}

/// What a hold keeps, and for whom.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Hold {
    pub owner: String,
    pub roots: Vec<ObjectRef>,
}

fn is_hex(text: &str) -> bool {
    text.len() == 64 && text.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

fn invalid(why: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, why)
}

fn old(path: &Path, age: Duration) -> bool {
    let modified = fs::symlink_metadata(path).and_then(|m| m.modified());
    modified.is_ok_and(|at| SystemTime::now().duration_since(at).is_ok_and(|unused| unused >= age))
}

impl Cas {
    pub fn new(root: impl Into<PathBuf>) -> Cas {
        Cas { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Where `object` lives; a malformed digest has no path.
    pub fn path(&self, object: &ObjectRef) -> io::Result<PathBuf> {
        match is_hex(&object.sha256) {
            true => Ok(self.root.join(&object.sha256[..2]).join(&object.sha256)),
            false => Err(invalid(format!("{:?} is not a sha256 digest", object.sha256))),
        }
    }

    /// Whether `object` is here; a use renews it.
    pub fn reuse(&self, object: &ObjectRef) -> io::Result<bool> {
        let _gate = GATE.read().unwrap();
        self.reuse_held(object)
    }

    /// Admits what `src` yields. With `expect`, one already here is reused unread and other
    /// bytes are refused (`InvalidData`).
    pub fn put(&self, mut src: impl Read, expect: Option<&ObjectRef>) -> io::Result<ObjectRef> {
        if let Some(expect) = expect {
            self.path(expect)?;
            if self.reuse(expect)? {
                return Ok(expect.clone());
            }
        }
        let tmp = self.root.join("tmp");
        fs::create_dir_all(&tmp)?;
        let temp = tmp.join(uuid::Uuid::new_v4().to_string());
        let written = (|| {
            let mut file = File::create_new(&temp)?;
            let (mut hash, mut length, mut buffer) = (Sha256::new(), 0u64, vec![0; 1 << 20]);
            loop {
                let n = match src.read(&mut buffer) {
                    Ok(0) => break,
                    Ok(n) => n,
                    Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                    Err(error) => return Err(error),
                };
                file.write_all(&buffer[..n])?;
                hash.update(&buffer[..n]);
                length += n as u64;
            }
            file.sync_data()?;
            let got = ObjectRef { sha256: format!("{:x}", hash.finalize()), length };
            if expect.is_some_and(|expect| *expect != got) {
                let expect = expect.expect("checked");
                return Err(io::Error::new(io::ErrorKind::InvalidData,
                    format!("object_digest_mismatch: expected {} ({} bytes), got {} ({length} bytes)", expect.sha256, expect.length, got.sha256)));
            }
            Ok(got)
        })();
        let got = match written {
            Ok(got) => got,
            Err(error) => {
                let _ = fs::remove_file(&temp);
                return Err(error);
            }
        };
        let path = self.path(&got)?;
        let dir = path.parent().expect("an object has a directory");
        fs::create_dir_all(dir)?;
        platform::durable::set_mode(&temp, 0o444).map_err(io::Error::from)?;
        let _gate = GATE.read().unwrap();
        fs::rename(&temp, &path)?;
        platform::durable::sync_dir(dir).map_err(io::Error::from)?;
        Ok(got)
    }

    pub fn put_bytes(&self, bytes: &[u8]) -> io::Result<ObjectRef> {
        self.put(bytes, None)
    }

    /// `object` to read, its length checked.
    pub fn file(&self, object: &ObjectRef) -> io::Result<File> {
        let file = File::open(self.path(object)?).map_err(|error| match error.kind() {
            io::ErrorKind::NotFound => io::Error::new(io::ErrorKind::NotFound, format!("object {} is absent", object.id())),
            _ => error,
        })?;
        match file.metadata()?.len() == object.length {
            true => Ok(file),
            false => Err(io::Error::new(io::ErrorKind::InvalidData, format!("object {} has another length here", object.id()))),
        }
    }

    pub fn read(&self, object: &ObjectRef) -> io::Result<Vec<u8>> {
        let mut bytes = Vec::with_capacity(object.length as usize);
        self.file(object)?.read_to_end(&mut bytes)?;
        Ok(bytes)
    }

    fn hold_path(&self, id: &str) -> io::Result<PathBuf> {
        let named = !id.is_empty() && id.len() <= 200 && id.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_'));
        match named {
            true => Ok(self.root.join("holds").join(format!("{id}.json"))),
            false => Err(invalid(format!("{id:?} is not a hold id"))),
        }
    }

    /// Keeps `hold`'s roots (each here) under `id`, replacing what it kept.
    pub fn hold(&self, id: &str, hold: &Hold) -> io::Result<()> {
        let path = self.hold_path(id)?;
        let bytes = serde_json::to_vec(hold)?;
        let _gate = GATE.read().unwrap();
        for root in &hold.roots {
            if !self.reuse_held(root)? {
                return Err(io::Error::new(io::ErrorKind::NotFound, format!("object {} is absent", root.id())));
            }
        }
        fs::create_dir_all(path.parent().expect("holds"))?;
        platform::durable::replace(&path, &bytes, 0o600).map_err(io::Error::from)
    }

    /// `reuse` under a gate the caller holds.
    fn reuse_held(&self, object: &ObjectRef) -> io::Result<bool> {
        let path = self.path(object)?;
        match fs::metadata(&path) {
            Ok(meta) if meta.len() == object.length => {
                let now = fs::FileTimes::new().set_modified(SystemTime::now());
                platform::durable::set_times(&path, now).map(|()| true).map_err(io::Error::from)
            }
            Ok(_) => Err(io::Error::new(io::ErrorKind::InvalidData, format!("object {} has another length here", object.id()))),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
            Err(error) => Err(error),
        }
    }

    pub fn held(&self, id: &str) -> io::Result<Option<Hold>> {
        match fs::read(self.hold_path(id)?) {
            Ok(bytes) => serde_json::from_slice(&bytes).map(Some).map_err(io::Error::other),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error),
        }
    }

    /// Drops hold `id`: its objects age from their last use. Whether it was held.
    pub fn release(&self, id: &str) -> io::Result<bool> {
        match fs::remove_file(self.hold_path(id)?) {
            Ok(()) => Ok(true),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
            Err(error) => Err(error),
        }
    }

    /// Removes each unheld object unused for `age` that `doomed` agrees to, and temp files as
    /// old: the digests removed.
    pub fn sweep(&self, age: Duration, doomed: &mut dyn FnMut(&str) -> io::Result<bool>) -> io::Result<Vec<String>> {
        let _gate = GATE.write().unwrap();
        let mut held = BTreeSet::new();
        for entry in fs::read_dir(self.root.join("holds")).into_iter().flatten().flatten() {
            let hold: Hold = serde_json::from_slice(&fs::read(entry.path())?).map_err(io::Error::other)?;
            held.extend(hold.roots.into_iter().map(|root| root.sha256));
        }
        let mut removed = vec![];
        for dir in fs::read_dir(&self.root).into_iter().flatten().flatten() {
            let name = dir.file_name().to_string_lossy().into_owned();
            if name == "tmp" {
                for temp in fs::read_dir(dir.path())?.flatten() {
                    if old(&temp.path(), age) {
                        let _ = fs::remove_file(temp.path());
                    }
                }
                continue;
            }
            if name.len() != 2 {
                continue;
            }
            for entry in fs::read_dir(dir.path())?.flatten() {
                let sha256 = entry.file_name().to_string_lossy().into_owned();
                if !is_hex(&sha256) || held.contains(&sha256) || !old(&entry.path(), age) || !doomed(&sha256)? {
                    continue;
                }
                fs::remove_file(entry.path())?;
                removed.push(sha256);
            }
        }
        Ok(removed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn objects_verify_once_and_go_only_unheld_and_unused() {
        let root = std::env::temp_dir().join(format!("td-cas-{}", uuid::Uuid::new_v4()));
        let cas = Cas::new(&root);
        let object = cas.put_bytes(b"a result's bytes").unwrap();
        assert_eq!(object, ObjectRef::of(b"a result's bytes"));
        assert_eq!(cas.read(&object).unwrap(), b"a result's bytes");
        let wrong = ObjectRef::of(b"other");
        assert_eq!(cas.put(&b"bytes"[..], Some(&wrong)).unwrap_err().kind(), io::ErrorKind::InvalidData);
        assert!(!cas.reuse(&wrong).unwrap() && fs::read_dir(root.join("tmp")).unwrap().count() == 0);
        let tree = cas.put_bytes(b"{}").unwrap();
        cas.hold("tree-1", &Hold { owner: "run-1".into(), roots: vec![tree.clone(), object.clone()] }).unwrap();
        assert!(cas.sweep(Duration::ZERO, &mut |_| Ok(true)).unwrap().is_empty(), "held objects stay");
        assert!(cas.release("tree-1").unwrap());
        assert!(cas.sweep(Duration::from_secs(3600), &mut |_| Ok(true)).unwrap().is_empty(), "just used");
        let mut removed = cas.sweep(Duration::ZERO, &mut |_| Ok(true)).unwrap();
        removed.sort();
        let mut both = vec![object.sha256.clone(), tree.sha256.clone()];
        both.sort();
        assert_eq!(removed, both);
        assert!(!cas.reuse(&object).unwrap());
        let _ = fs::remove_dir_all(root);
    }
}
