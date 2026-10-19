//! Refs name models; pins are the one retention record. Each is one JSON file, changed under its
//! directory's lock and replaced by a temp and a rename: a reader sees the old record or the new.
//!
//! The lock is `flock` on an `O_CLOEXEC` descriptor: held by its open file description, waited
//! for, never inspected. A child forked meanwhile holds it only until it execs. Nothing else in
//! the store locks; a short shared acquisition fence also spans root publication so GC cannot
//! delete bytes between metadata read and ownership publication.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use platform::lock;
use serde::{Deserialize, Serialize};

use super::{fail, invalid, now_secs, platform, Error, Result, Store};
use crate::format::{ObjectRef, Repo};

/// `pins/<id>.json`: what keeps objects alive, and when it was last used (unix seconds).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Pin {
    pub last_use: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub owner: Option<String>,
    pub roots: Vec<ObjectRef>,
}

impl Pin {
    pub fn new(roots: Vec<ObjectRef>) -> Pin {
        Pin {
            last_use: now_secs(),
            owner: None,
            roots,
        }
    }
}

/// ASCII letters, digits, `.`, `_` and `-`, not leading `.`.
pub(crate) fn is_name(text: &str, max: usize) -> bool {
    !text.is_empty()
        && text.len() <= max
        && !text.starts_with('.')
        && text
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
}

fn parse<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<Option<T>> {
    match fs::read(path) {
        Ok(bytes) => serde_json::from_slice(&bytes)
            .map(Some)
            .map_err(|error| invalid(path, error)),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(fail("read", path)(error)),
    }
}

fn json(path: &Path, value: &impl Serialize) -> Result<Vec<u8>> {
    serde_json::to_vec(value).map_err(|error| invalid(path, error))
}

impl Store {
    pub(super) fn lock(&self, dir: &str) -> Result<lock::DirLock> {
        let path = self.root.join(dir);
        lock::dir(&path).map_err(platform("lock", path))
    }

    /// Makes every object the new `roots` reach young, then everything this handle admitted
    /// durable, so a record never names what GC or a crash could take. A root must be here;
    /// what it reaches need not be (a pull may take some components only).
    fn retain<'a>(&self, roots: impl IntoIterator<Item = &'a ObjectRef>) -> Result<()> {
        for root in roots {
            if !self.reuse(root)? {
                return Err(Error::Absent(root.sha256.clone()));
            }
            for object in self.closure(root)? {
                self.reuse(&object)?;
            }
        }
        self.sync()
    }

    fn pin_path(&self, id: &str) -> Result<PathBuf> {
        if !is_name(id, 200) {
            return Err(invalid(
                self.root.join("pins"),
                format!("{id:?} is not a pin id"),
            ));
        }
        Ok(self.root.join("pins").join(format!("{id}.json")))
    }

    /// Creates or replaces pin `id`.
    pub fn pin(&self, id: &str, pin: &Pin) -> Result<()> {
        let path = self.pin_path(id)?;
        let _lock = self.lock("pins")?;
        let old = parse::<Pin>(&path)?.map_or_else(Vec::new, |old| old.roots);
        self.retain(pin.roots.iter().filter(|root| !old.contains(root)))?;
        self.commit(&path, &json(&path, pin)?, true)
    }

    pub fn get_pin(&self, id: &str) -> Result<Option<Pin>> {
        parse(&self.pin_path(id)?)
    }

    /// Sets pin `id`'s `last_use` to now; false when there is no such pin.
    pub fn touch_pin(&self, id: &str) -> Result<bool> {
        let path = self.pin_path(id)?;
        let _lock = self.lock("pins")?;
        let Some(mut pin) = parse::<Pin>(&path)? else {
            return Ok(false);
        };
        pin.last_use = now_secs();
        // A lost touch costs only recency, so it is never flushed.
        self.commit(&path, &json(&path, &pin)?, false)?;
        Ok(true)
    }

    pub fn unpin(&self, id: &str) -> Result<bool> {
        let path = self.pin_path(id)?;
        let _lock = self.lock("pins")?;
        if super::retention::protected(self, id)? {
            return Ok(false);
        }
        self.remove_record(&path)
    }

    /// Only the retention reaper calls this after proving the hold fence is unowned.
    pub(super) fn remove_hold_pin(&self, id: &str) -> Result<bool> {
        if !id.starts_with("hold-") {
            return Err(invalid(self.root.join("pins"), "not a retention hold"));
        }
        let path = self.pin_path(id)?;
        let _lock = self.lock("pins")?;
        self.remove_record(&path)
    }

    /// Removes pin `id` only while it still reads `last_use`: one used meanwhile stays.
    pub fn unpin_unused(&self, id: &str, last_use: u64) -> Result<bool> {
        // `id` names a file `pins()` listed, so it needs no grammar check.
        let path = self.root.join("pins").join(format!("{id}.json"));
        let _lock = self.lock("pins")?;
        if super::retention::protected(self, id)? {
            return Ok(false);
        }
        match parse::<Pin>(&path)? {
            Some(pin) if pin.last_use == last_use => self.remove_record(&path),
            _ => Ok(false),
        }
    }

    pub fn pins(&self) -> Result<Vec<(String, Pin)>> {
        let dir = self.root.join("pins");
        let mut pins = Vec::new();
        for entry in fs::read_dir(&dir).map_err(fail("list", &dir))? {
            let path = entry.map_err(fail("list", &dir))?.path();
            // Every record counts for GC, even one whose name `pin` would refuse.
            let id = path
                .file_name()
                .and_then(|name| name.to_str()?.strip_suffix(".json"));
            let Some(id) = id.map(str::to_owned) else {
                continue;
            };
            if let Some(pin) = parse(&path)? {
                pins.push((id, pin));
            }
        }
        pins.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(pins)
    }

    /// `org/name` → `refs/<org>/<name>.json`. Names are case-insensitive: one identity.
    fn ref_path(&self, name: &str) -> Result<PathBuf> {
        match name.split_once('/') {
            Some((org, model)) if is_name(org, 128) && is_name(model, 128) => Ok(self
                .root
                .join("refs")
                .join(org.to_ascii_lowercase())
                .join(format!("{}.json", model.to_ascii_lowercase()))),
            _ => Err(invalid(
                self.root.join("refs"),
                format!("{name:?} is not org/name"),
            )),
        }
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
        Self::parse_ref(&self.ref_path(name)?)
    }

    /// Changes ref `name` to what `change` makes of it; `None` removes it.
    pub fn update_ref(
        &self,
        name: &str,
        change: impl FnOnce(Option<Repo>) -> Option<Repo>,
    ) -> Result<Option<Repo>> {
        let path = self.ref_path(name)?;
        let _acquisition = super::retention::acquire(self)?;
        let _lock = self.lock("refs")?;
        let promoted = self.clear_cached_ref(name)?;
        let old = Self::parse_ref(&path)?;
        let lanes = |repo: &Option<Repo>| -> Vec<ObjectRef> {
            repo.iter().flat_map(Repo::roots).cloned().collect()
        };
        let before = lanes(&old);
        let Some(new) = change(old) else {
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
        self.retain(
            after
                .iter()
                .filter(|lane| promoted || !before.contains(lane)),
        )?;
        self.commit(&path, &bytes, true)?;
        Ok(Some(new))
    }

    pub fn refs(&self) -> Result<Vec<Repo>> {
        let dir = self.root.join("refs");
        let mut repos = Vec::new();
        for org in fs::read_dir(&dir).map_err(fail("list", &dir))? {
            let org = org.map_err(fail("list", &dir))?.path();
            if !org.is_dir() {
                continue;
            }
            for entry in fs::read_dir(&org).map_err(fail("list", &org))? {
                let path = entry.map_err(fail("list", &org))?.path();
                if path.extension().is_some_and(|ext| ext == "json") {
                    repos.extend(Self::parse_ref(&path)?);
                }
            }
        }
        repos.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(repos)
    }
}
