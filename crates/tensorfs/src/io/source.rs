//! Owned byte sources for placement. TensorFS keeps verified descriptors and checkpoint
//! source context, then runs every transfer through the same host/device read engine.

use std::collections::HashMap;
use std::fs::File;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex};

pub use super::{ItemSource, ObjectRange};
use super::{Copy, Dst, ReadOpts, ReadStats, Src};
use crate::failure::{Failure, Kind};
use crate::format::ObjectRef;

#[derive(Debug, Clone)]
pub struct Item {
    pub offset: u64,
    pub len: u64,
    pub source: ItemSource,
    pub part_end: bool,
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
    view: Option<super::view::View>,
    files: Mutex<HashMap<String, Arc<File>>>,
    direct_io: bool,
}

impl Source {
    pub fn files(files: Vec<(ObjectRef, File)>, direct_io: bool, _plane: usize) -> Source {
        Source {
            #[cfg(feature = "store")]
            view: None,
            files: Mutex::new(
                files
                    .into_iter()
                    .map(|(o, f)| (o.sha256, Arc::new(f)))
                    .collect(),
            ),
            direct_io,
        }
    }

    /// Retain metadata and the source context. No chunk opens or downloads until a read.
    #[cfg(feature = "store")]
    pub(crate) fn checkpoint(view: &super::view::View, direct_io: bool) -> Source {
        let mut source = Self::files(Vec::new(), direct_io, 0);
        source.view = Some(view.clone());
        source
    }

    #[cfg(feature = "store")]
    fn load(&self, objects: &[ObjectRef]) -> Result<()> {
        if let Some(view) = &self.view {
            let missing = {
                let files = self.files.lock().unwrap();
                objects
                    .iter()
                    .filter(|o| !files.contains_key(&o.sha256))
                    .cloned()
                    .collect::<Vec<_>>()
            };
            if !missing.is_empty() {
                let opened = view.files_for(&missing)?;
                self.files
                    .lock()
                    .unwrap()
                    .extend(opened.into_iter().map(|(o, f)| (o.sha256, Arc::new(f))));
            }
        }
        Ok(())
    }

    fn file(&self, object: &ObjectRef) -> Result<Arc<File>> {
        if let Some(file) = self.files.lock().unwrap().get(&object.sha256).cloned() {
            return Ok(file);
        }
        Err(invalid(format!(
            "{}: source has no verified file",
            object.sha256
        )))
    }

    pub fn resident(&self, obj: &ObjectRef, _tally: &IoTally) -> Result<bool> {
        #[cfg(feature = "store")]
        self.load(std::slice::from_ref(obj))?;
        let file = self.file(obj)?;
        Ok(platform::durable::cached(&file, 0, obj.length).unwrap_or(false))
    }

    /// Transfer a placement range. Uncovered padding is zeroed through the same engine.
    ///
    /// # Safety
    /// `dst` is exclusively writable for its length; the selected item ranges lie inside it.
    /// On CUDA error, retain device destination owners until `super::drain` proves completion.
    /// Inline host bytes use owned engine staging and are not borrowed by pending GPU work.
    pub unsafe fn read_items(
        &self,
        items: &[Item],
        origin: u64,
        dst: Dst,
        threads: usize,
        mode: Mode,
        tally: &IoTally,
    ) -> Result<ReadStats> {
        let mut intervals = items
            .iter()
            .map(|item| {
                if let ItemSource::Object(range) = &item.source {
                    if range.len != item.len
                        || range
                            .off
                            .checked_add(range.len)
                            .is_none_or(|n| n > range.obj.length)
                    {
                        return Err(invalid("object range exceeds its verified source"));
                    }
                }
                if let ItemSource::Inline(bytes) = &item.source {
                    if bytes.len() as u64 != item.len {
                        return Err(invalid("inline length differs from item"));
                    }
                }
                let start = item
                    .offset
                    .checked_sub(origin)
                    .ok_or_else(|| invalid("item begins before destination"))?;
                let end = start
                    .checked_add(item.len)
                    .ok_or_else(|| invalid("item range overflows"))?;
                if end > dst.len() as u64 {
                    return Err(invalid("item exceeds destination"));
                }
                Ok((start, end))
            })
            .collect::<Result<Vec<_>>>()?;
        intervals.sort_unstable();
        let mut gaps = Vec::new();
        let mut cursor = 0;
        for (start, end) in intervals {
            if start == end {
                continue;
            }
            if start < cursor {
                return Err(invalid("placement items overlap"));
            }
            if start > cursor {
                gaps.push((cursor, start - cursor));
            }
            cursor = end;
        }
        if cursor < dst.len() as u64 {
            gaps.push((cursor, dst.len() as u64 - cursor));
        }
        #[cfg(feature = "store")]
        {
            let wanted = items
                .iter()
                .filter_map(|item| match &item.source {
                    ItemSource::Object(range) => {
                        Some((range.obj.sha256.clone(), range.obj.clone()))
                    }
                    ItemSource::Inline(_) => None,
                })
                .collect::<std::collections::BTreeMap<_, _>>()
                .into_values()
                .collect::<Vec<_>>();
            self.load(&wanted)?;
        }
        let mut files = HashMap::new();
        for item in items {
            if let ItemSource::Object(range) = &item.source {
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
                    inline += item.len;
                    Src::Host(bytes.as_ptr())
                }
            };
            copies.push(Copy {
                src,
                dst: dst.slice(offset as usize, item.len as usize),
            });
        }
        for (start, len) in gaps {
            copies.push(Copy {
                src: Src::Zero,
                dst: dst.slice(start as usize, len as usize),
            });
        }
        let opts = ReadOpts {
            threads,
            direct: self.direct_io && mode == Mode::Cache,
            ..ReadOpts::default()
        };
        // SAFETY: item ranges and destination bounds checked above; caller supplies exclusivity.
        let mut stats = unsafe { super::read_staged(&copies, &opts) }?;
        stats.bytes = items.iter().map(|item| item.len).sum();
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
