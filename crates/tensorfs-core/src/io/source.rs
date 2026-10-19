//! Owned byte sources for placement. TensorFS keeps verified descriptors and checkpoint
//! source context, then runs every transfer through the same host/device read engine.

use std::collections::HashMap;
use std::fs::File;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
#[cfg(feature = "store")]
use std::sync::RwLock;
use std::sync::{Arc, Mutex};

use super::{Copy, Dst, ReadOpts, ReadStats, Src};
use crate::failure::{Failure, Kind};
use crate::ids::ObjectRef;
use crate::read::ObjectRange;
#[cfg(feature = "store")]
use crate::{meta::Meta, read::ReadLease, store::Store};

#[derive(Debug, Clone)]
pub struct Item {
    pub offset: u64,
    pub len: u64,
    pub source: ItemSource,
    pub part_end: bool,
}

#[derive(Debug, Clone)]
pub enum ItemSource {
    Object(ObjectRange),
    Inline(Arc<[u8]>),
}

#[derive(Debug, Default)]
pub struct IoTally {
    pub cached_bytes: AtomicU64,
    pub direct_bytes: AtomicU64,
    pub buffered_bytes: AtomicU64,
    pub inline_bytes: AtomicU64,
    pub direct_refused: AtomicU64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Cache,
    Buffered,
    Cached,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Source(#[from] Failure),
    #[error(transparent)]
    Read(#[from] super::Error),
}
pub type Result<T> = std::result::Result<T, Error>;
fn invalid(detail: impl Into<String>) -> Error {
    Failure {
        kind: Kind::Invalid,
        detail: detail.into(),
    }
    .into()
}

pub struct Source {
    #[cfg(feature = "store")]
    lease: RwLock<Option<ReadLease>>,
    #[cfg(feature = "store")]
    meta: Option<Arc<Meta>>,
    #[cfg(feature = "store")]
    store: Option<Arc<Store>>,
    #[cfg(feature = "store")]
    _view: Option<super::view::View>,
    files: Mutex<HashMap<String, Arc<File>>>,
    direct_io: bool,
}

impl Source {
    #[cfg(feature = "store")]
    pub fn new(
        store: Arc<Store>,
        meta: Arc<Meta>,
        lease: ReadLease,
        direct_io: bool,
        _plane: usize,
    ) -> Source {
        Source {
            lease: RwLock::new(Some(lease)),
            meta: Some(meta),
            store: Some(store),
            _view: None,
            files: Mutex::default(),
            direct_io,
        }
    }

    pub fn files(files: Vec<(ObjectRef, File)>, direct_io: bool, _plane: usize) -> Source {
        Source {
            #[cfg(feature = "store")]
            lease: RwLock::new(None),
            #[cfg(feature = "store")]
            meta: None,
            #[cfg(feature = "store")]
            store: None,
            #[cfg(feature = "store")]
            _view: None,
            files: Mutex::new(
                files
                    .into_iter()
                    .map(|(o, f)| (o.sha256, Arc::new(f)))
                    .collect(),
            ),
            direct_io,
        }
    }

    /// Retain the checkpoint source and callbacks together with the opened chunk descriptors.
    #[cfg(feature = "store")]
    pub fn checkpoint(
        view: &super::view::View,
        chunks: &[usize],
        direct_io: bool,
    ) -> Result<Source> {
        let mut source = Self::files(view.files(chunks)?, direct_io, 0);
        source._view = Some(view.clone());
        Ok(source)
    }

    fn file(&self, object: &ObjectRef) -> Result<Arc<File>> {
        if let Some(file) = self.files.lock().unwrap().get(&object.sha256).cloned() {
            return Ok(file);
        }
        #[cfg(feature = "store")]
        if let Some(store) = &self.store {
            let file = store.open_verified(&object.sha256).map_err(Failure::from)?;
            if file.len() != object.length {
                return Err(invalid(format!(
                    "{}: verified length {} differs from {}",
                    object.sha256,
                    file.len(),
                    object.length
                )));
            }
            let file = Arc::new(file.into_file());
            self.files
                .lock()
                .unwrap()
                .insert(object.sha256.clone(), file.clone());
            return Ok(file);
        }
        Err(invalid(format!(
            "{}: source has no verified file",
            object.sha256
        )))
    }

    pub fn close_object(&self, sha256: &str) {
        #[cfg(feature = "store")]
        if self.store.is_some() {
            self.files.lock().unwrap().remove(sha256);
        }
        #[cfg(not(feature = "store"))]
        let _ = sha256;
    }

    pub fn resident(&self, obj: &ObjectRef, _tally: &IoTally) -> Result<bool> {
        let file = self.file(obj)?;
        Ok(platform::durable::cached(&file, 0, obj.length).unwrap_or(false))
    }

    /// Transfer a placement range. Padding is zeroed by its allocator and does not carry data.
    ///
    /// # Safety
    /// `dst` is exclusively writable for its length; the selected item ranges lie inside it.
    pub unsafe fn read_items(
        &self,
        items: &[Item],
        origin: u64,
        dst: Dst,
        threads: usize,
        mode: Mode,
        tally: &IoTally,
    ) -> Result<ReadStats> {
        #[cfg(feature = "store")]
        let lease = self.lease.read().unwrap();
        let mut files = HashMap::new();
        for item in items {
            if let ItemSource::Object(range) = &item.source {
                if range.len != item.len
                    || range
                        .off
                        .checked_add(range.len)
                        .is_none_or(|n| n > range.obj.length)
                {
                    return Err(invalid("object range exceeds its verified source"));
                }
                #[cfg(feature = "store")]
                if let Some(lease) = lease.as_ref() {
                    lease.covers(range).map_err(Failure::from)?;
                }
                files
                    .entry(range.obj.sha256.clone())
                    .or_insert(self.file(&range.obj)?);
            }
        }
        let mut copies = Vec::with_capacity(items.len());
        let mut inline = 0;
        for item in items {
            let offset = item
                .offset
                .checked_sub(origin)
                .ok_or_else(|| invalid("item begins before destination"))?;
            let end = offset
                .checked_add(item.len)
                .ok_or_else(|| invalid("item range overflows"))?;
            if end > dst.len() as u64 {
                return Err(invalid("item exceeds destination"));
            }
            let src = match &item.source {
                ItemSource::Object(range) => Src::File(&files[&range.obj.sha256], range.off),
                ItemSource::Inline(bytes) => {
                    if bytes.len() as u64 != item.len {
                        return Err(invalid("inline length differs from item"));
                    }
                    inline += item.len;
                    Src::Host(bytes.as_ptr())
                }
            };
            copies.push(Copy {
                src,
                dst: dst.slice(offset as usize, item.len as usize),
            });
        }
        let opts = ReadOpts {
            threads,
            direct: self.direct_io && mode == Mode::Cache,
            ..ReadOpts::default()
        };
        // SAFETY: item ranges and destination bounds checked above; caller supplies exclusivity.
        let stats = unsafe { super::read(&copies, &opts) }?;
        tally.direct_bytes.fetch_add(stats.direct, Relaxed);
        tally.inline_bytes.fetch_add(inline, Relaxed);
        if mode == Mode::Buffered {
            tally.buffered_bytes.fetch_add(stats.cached, Relaxed);
        } else {
            tally.cached_bytes.fetch_add(stats.cached, Relaxed);
        }
        if matches!(dst, Dst::Host { .. }) && stats.bytes > 0 {
            super::metrics::record(super::metrics::Copy {
                kind: "disk",
                gpu: -1,
                bytes: stats.bytes,
                ns: stats.ns,
                cached: stats.cached,
                ceiling: super::metrics::disk(
                    &files.values().map(AsRef::as_ref).collect::<Vec<_>>(),
                )
                .unwrap_or(0.0),
                ..Default::default()
            });
        }
        Ok(stats)
    }

    /// # Safety
    /// `dst` is exclusively writable for `item.len` bytes.
    pub unsafe fn read_item(
        &self,
        item: &Item,
        dst: *mut u8,
        mode: Mode,
        tally: &IoTally,
    ) -> Result<()> {
        // SAFETY: inherited from caller.
        unsafe {
            self.read_items(
                std::slice::from_ref(item),
                item.offset,
                Dst::Host {
                    ptr: dst,
                    len: item.len as usize,
                },
                1,
                mode,
                tally,
            )
        }
        .map(|_| ())
    }
}

impl Drop for Source {
    fn drop(&mut self) {
        #[cfg(feature = "store")]
        if let (Some(lease), Some(meta)) = (self.lease.get_mut().unwrap().take(), &self.meta) {
            if let Err(error) = lease.release(meta) {
                tracing::warn!("tensorfs: read lease release failed: {error}");
            }
        }
    }
}
