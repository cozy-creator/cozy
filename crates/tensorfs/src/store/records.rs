//! Refs name models. Each is one JSON file, changed under its directory's lock and replaced by a
//! temp and a rename: a reader sees the old record or the new.
//!
//! The lock is `flock` on an `O_CLOEXEC` descriptor: held by its open file description, waited
//! for, never inspected. A child forked meanwhile holds it only until it execs. A short shared
//! acquisition fence also spans root publication so GC cannot delete bytes between reading
//! metadata and naming them.

use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use platform::lock;

use super::{fail, invalid, platform, Error, Result, Store};
use crate::format::{repo_name, ObjectRef, Repo};

/// ASCII letters, digits, `.`, `_` and `-`, not leading `.`.
pub(crate) fn is_name(text: &str, max: usize) -> bool {
    !text.is_empty()
        && text.len() <= max
        && !text.starts_with('.')
        && text
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
}

fn ref_segment(segment: &str) -> String {
    let folded = segment.to_ascii_lowercase();
    let stem = folded.split('.').next().unwrap_or("");
    let device = matches!(stem, "con" | "prn" | "aux" | "nul")
        || (stem.len() == 4
            && (stem.starts_with("com") || stem.starts_with("lpt"))
            && matches!(stem.as_bytes()[3], b'1'..=b'9'));
    if is_name(&folded, 128) && !folded.ends_with('.') && !device {
        folded
    } else {
        // '%' was never a legal legacy segment. Hashing also bounds Unicode and long
        // names, without changing the protocol's raw repo_key or its case folding.
        format!("%{}", ObjectRef::of(folded.as_bytes()).sha256)
    }
}

impl Store {
    pub(super) fn lock(&self, dir: &str) -> Result<lock::DirLock> {
        let path = self.root.join(dir);
        lock::dir(&path).map_err(platform("lock", path))
    }

    /// Makes every object the new `roots` reach young, then everything this handle admitted
    /// durable, so a record never names what GC or a crash could take: each root and what it
    /// reaches. A root must be here; what it reaches need not be (a pull may take some
    /// components only).
    fn retain<'a>(&self, roots: impl IntoIterator<Item = &'a ObjectRef>) -> Result<Vec<(String, Vec<ObjectRef>)>> {
        let mut reached = Vec::new();
        for root in roots {
            if !self.reuse(root)? {
                return Err(Error::Absent(root.sha256.clone()));
            }
            let closure = self.closure(root)?;
            for object in &closure {
                self.reuse(object)?;
            }
            reached.push((root.sha256.clone(), closure));
        }
        self.sync()?;
        Ok(reached)
    }

    /// Case-folded local filenames; nonportable segments use a prefix absent from legacy names.
    fn ref_path(&self, name: &str) -> Result<PathBuf> {
        // The format's repository name, as the public Ref takes it; is_name remains the
        // stricter grammar for writer keys.
        match name.split_once('/') {
            Some((org, model)) if repo_name(name) => Ok(self
                .root
                .join("refs")
                .join(ref_segment(org))
                .join(format!("{}.json", ref_segment(model)))),
            _ => Err(invalid(
                self.root.join("refs"),
                format!("{name:?} is not org/name"),
            )),
        }
    }

    fn legacy_ref_path(&self, name: &str, canonical: &Path) -> Option<PathBuf> {
        // Windows must never probe a device path such as CON.json. Only POSIX could
        // have stored these old names; reading them does not migrate user metadata.
        if cfg!(windows) {
            return None;
        }
        let (org, model) = name.split_once('/')?;
        if !is_name(org, 128) || !is_name(model, 128) {
            return None;
        }
        let path = self
            .root
            .join("refs")
            .join(org.to_ascii_lowercase())
            .join(format!("{}.json", model.to_ascii_lowercase()));
        (path != canonical).then_some(path)
    }

    fn matching_legacy_ref(&self, name: &str, canonical: &Path) -> Result<Option<(PathBuf, Repo)>> {
        let Some(path) = self.legacy_ref_path(name, canonical) else {
            return Ok(None);
        };
        Ok(Self::parse_ref(&path)?
            .filter(|repo| repo.name.eq_ignore_ascii_case(name))
            .map(|repo| (path, repo)))
    }

    fn canonical_ref(path: &Path, name: &str) -> Result<Option<Repo>> {
        let repo = Self::parse_ref(path)?;
        if repo
            .as_ref()
            .is_some_and(|repo| !repo.name.eq_ignore_ascii_case(name))
        {
            return Err(invalid(path, format!("ref does not name {name:?}")));
        }
        Ok(repo)
    }

    fn parse_ref(path: &Path) -> Result<Option<Repo>> {
        match fs::read(path) {
            Ok(bytes) => Repo::from_json(&bytes)
                .map(Some)
                .map_err(|error| invalid(path, error)),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(fail("read", path)(error)),
        }
    }

    pub fn get_ref(&self, name: &str) -> Result<Option<Repo>> {
        let path = self.ref_path(name)?;
        if let Some(repo) = Self::canonical_ref(&path, name)? {
            return Ok(Some(repo));
        }
        if let Some((_, repo)) = self.matching_legacy_ref(name, &path)? {
            return Ok(Some(repo));
        }
        // A concurrent explicit update may have published canonical then removed legacy.
        Self::canonical_ref(&path, name)
    }

    /// Changes ref `name` to what `change` makes of it; `None` removes it. Naming a tree is a use
    /// of it, and a `local/` ref roots it.
    pub fn update_ref(
        &self,
        name: &str,
        change: impl FnOnce(Option<Repo>) -> Option<Repo>,
    ) -> Result<Option<Repo>> {
        let path = self.ref_path(name)?;
        let _acquisition = super::retention::acquire(self)?;
        let _lock = self.lock("refs")?;
        let legacy = self.matching_legacy_ref(name, &path)?;
        let old = Self::canonical_ref(&path, name)?
            .or_else(|| legacy.as_ref().map(|(_, repo)| repo.clone()));
        let lanes = |repo: &Option<Repo>| -> Vec<ObjectRef> {
            repo.iter().flat_map(Repo::roots).cloned().collect()
        };
        let before = lanes(&old);
        let Some(new) = change(old) else {
            // Remove a matching old record first, so no crash can expose it after
            // deleting a newer canonical record. Unrelated metadata is untouched.
            if let Some((legacy, _)) = &legacy {
                self.remove_record(legacy)?;
            }
            self.remove_record(&path)?;
            return Ok(None);
        };
        if !new.name.eq_ignore_ascii_case(name) {
            return Err(invalid(
                &path,
                format!("repo {:?} cannot be ref {name}", new.name),
            ));
        }
        let bytes = new.to_json().map_err(|error| invalid(&path, error))?;
        let after = lanes(&Some(new.clone()));
        let reached = self.retain(after.iter().filter(|lane| !before.contains(lane)))?;
        self.commit(&path, &bytes, true)?;
        for (root, closure) in &reached {
            self.used(root, closure)?;
        }
        // Only an explicit mutation promotes a legacy record, after durable publish.
        if let Some((legacy, _)) = &legacy {
            self.remove_record(legacy)?;
        }
        Ok(Some(new))
    }

    pub fn refs(&self) -> Result<Vec<Repo>> {
        let dir = self.root.join("refs");
        let mut repos = Vec::new();
        let mut named: BTreeMap<String, (bool, Repo)> = BTreeMap::new();
        for org in fs::read_dir(&dir).map_err(fail("list", &dir))? {
            let org = org.map_err(fail("list", &dir))?.path();
            if !org.is_dir() {
                continue;
            }
            for entry in fs::read_dir(&org).map_err(fail("list", &org))? {
                let path = entry.map_err(fail("list", &org))?.path();
                if path.extension().is_some_and(|ext| ext == "json") {
                    if let Some(repo) = Self::parse_ref(&path)? {
                        let canonical = self.ref_path(&repo.name)?;
                        let is_canonical = path == canonical;
                        if is_canonical
                            || self.legacy_ref_path(&repo.name, &canonical).as_ref() == Some(&path)
                        {
                            let key = repo.name.to_ascii_lowercase();
                            if !named.get(&key).is_some_and(|(preferred, _)| *preferred) {
                                named.insert(key, (is_canonical, repo));
                            }
                        } else {
                            // Preserve independently named metadata instead of treating an
                            // arbitrary filename as a stale copy eligible for deletion.
                            repos.push(repo);
                        }
                    }
                }
            }
        }
        repos.extend(named.into_values().map(|(_, repo)| repo));
        repos.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(repos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct Scratch(PathBuf);

    impl Scratch {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            loop {
                let root = std::env::temp_dir().join(format!(
                    "tensorfs-ref-names-{}-{}",
                    std::process::id(),
                    NEXT.fetch_add(1, Ordering::Relaxed)
                ));
                match fs::create_dir(&root) {
                    Ok(()) => return Self(root),
                    Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
                    Err(error) => panic!("create scratch: {error}"),
                }
            }
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[cfg(not(windows))]
    fn write_repo(path: &Path, repo: &Repo) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, repo.to_json().unwrap()).unwrap();
    }

    #[test]
    fn portable_refs_keep_safe_legacy_paths_and_ascii_case_folding() {
        let scratch = Scratch::new();
        let store = Store::open(&scratch.0).unwrap();
        assert_eq!(
            store.ref_path("AcMe/Model.v1").unwrap(),
            scratch.0.join("refs/acme/model.v1.json")
        );
        assert_eq!(
            store.ref_path("AcMe Space/MODEL%?").unwrap(),
            store.ref_path("acme space/model%?").unwrap()
        );
        assert_ne!(
            store.ref_path("org/É").unwrap(),
            store.ref_path("org/é").unwrap()
        );
        assert!(
            !is_name("%a", 200) && !is_name(".hidden", 200),
            "name grammar changed"
        );
    }

    #[test]
    fn portable_refs_roundtrip_unicode_reserved_hidden_and_long_segments() {
        let scratch = Scratch::new();
        let store = Store::open(&scratch.0).unwrap();
        let mut names: Vec<String> = [
            "AcMe Space/MoDEL%?#É",
            "Org%2F/.Hidden",
            "組/名",
            "CON/NUL.txt",
            "COM1/LPT9.anything",
            "org/trailing.",
            "org/trailing ",
            "org/:*?|<>\"",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect();
        names.push(format!("{}/{}", "a".repeat(200), "b".repeat(300)));
        for name in &names {
            let path = store.ref_path(name).unwrap();
            for part in path.strip_prefix(scratch.0.join("refs")).unwrap().iter() {
                let text = part.to_str().unwrap();
                assert!(text.len() <= 133 && text.is_ascii());
                assert!(
                    !text.ends_with(['.', ' '])
                        && !text.contains([':', '*', '?', '|', '<', '>', '"'])
                );
            }
            let repo = Repo::new(name);
            store.update_ref(name, |_| Some(repo.clone())).unwrap();
            assert_eq!(store.get_ref(name).unwrap(), Some(repo));
        }
        assert_eq!(store.refs().unwrap().len(), names.len());
        for device in ["con", "con.txt", "prn", "aux", "nul", "com1", "lpt9.log"] {
            assert!(
                ref_segment(device).starts_with('%'),
                "device name was not encoded: {device}"
            );
        }
    }

    #[test]
    fn portable_refs_reject_path_traversal() {
        let scratch = Scratch::new();
        let store = Store::open(&scratch.0).unwrap();
        for name in [
            "org", "/model", "org/", "org/..", "./model", "org/a/b", "org/a\\b", "org/a\0b",
        ] {
            assert!(store.ref_path(name).is_err(), "accepted {name:?}");
        }
    }

    #[cfg(not(windows))]
    #[test]
    fn portable_refs_read_legacy_without_mutation_then_promote_only_on_update() {
        let scratch = Scratch::new();
        let store = Store::open(&scratch.0).unwrap();
        let name = "CON/Model.";
        let legacy = scratch.0.join("refs/con/model..json");
        let canonical = store.ref_path(name).unwrap();
        let old = Repo::new(name);
        write_repo(&legacy, &old);
        let before = fs::read(&legacy).unwrap();
        assert_eq!(store.get_ref(name).unwrap(), Some(old.clone()));
        assert_eq!(fs::read(&legacy).unwrap(), before);
        assert!(!canonical.exists());
        let new = Repo::new("con/MODEL.");
        store
            .update_ref(name, |found| {
                assert_eq!(found, Some(old));
                Some(new.clone())
            })
            .unwrap();
        assert!(!legacy.exists() && canonical.exists());
        assert_eq!(store.get_ref(name).unwrap(), Some(new.clone()));
        assert_eq!(store.refs().unwrap(), vec![new]);
        store.update_ref(name, |_| None).unwrap();
        assert_eq!(store.get_ref(name).unwrap(), None);
        assert!(!legacy.exists() && !canonical.exists());
    }

    #[cfg(not(windows))]
    #[test]
    fn portable_refs_prefer_canonical_crash_duplicate_and_delete_without_resurrection() {
        let scratch = Scratch::new();
        let store = Store::open(&scratch.0).unwrap();
        let name = "CON/Model.";
        let legacy = scratch.0.join("refs/con/model..json");
        let canonical = store.ref_path(name).unwrap();
        write_repo(&legacy, &Repo::new(name));
        let new = Repo::new("con/MODEL.");
        write_repo(&canonical, &new);
        assert_eq!(store.get_ref(name).unwrap(), Some(new.clone()));
        assert_eq!(store.refs().unwrap(), vec![new]);
        store.update_ref(name, |_| None).unwrap();
        assert!(!legacy.exists() && !canonical.exists());
        assert_eq!(store.get_ref(name).unwrap(), None);
        assert!(store.refs().unwrap().is_empty());
        write_repo(&legacy, &Repo::new(name));
        store.update_ref(name, |_| None).unwrap();
        assert_eq!(
            store.get_ref(name).unwrap(),
            None,
            "legacy-only deletion resurrected"
        );
    }

    #[cfg(not(windows))]
    #[test]
    fn portable_refs_never_remove_unrelated_metadata_at_the_legacy_path() {
        let scratch = Scratch::new();
        let store = Store::open(&scratch.0).unwrap();
        let name = "CON/Model.";
        let legacy = scratch.0.join("refs/con/model..json");
        let unrelated = Repo::new("other/owned");
        write_repo(&legacy, &unrelated);
        let before = fs::read(&legacy).unwrap();
        store.update_ref(name, |_| Some(Repo::new(name))).unwrap();
        store.update_ref(name, |_| None).unwrap();
        assert_eq!(fs::read(&legacy).unwrap(), before);
        assert_eq!(store.get_ref(name).unwrap(), None);
        assert_eq!(store.refs().unwrap(), vec![unrelated]);
    }

    #[cfg(windows)]
    #[test]
    fn portable_refs_never_probe_legacy_windows_devices() {
        let scratch = Scratch::new();
        let store = Store::open(&scratch.0).unwrap();
        let path = store.ref_path("CON/NUL").unwrap();
        assert!(store.legacy_ref_path("CON/NUL", &path).is_none());
    }
}
