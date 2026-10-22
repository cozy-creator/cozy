//! Owned byte sources for placement. TensorFS keeps verified descriptors and checkpoint
//! source context, then runs every transfer through the same host/device read engine.

use std::collections::HashMap;
use std::fs::File;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::{AcqRel, Relaxed}};
use std::sync::{Arc, Mutex};

pub use super::{ItemSource, ObjectRange};
use super::{Copy, Dst, Issued, ReadOpts, ReadStats, Src};
use crate::failure::{Failure, Kind};
use crate::format::ObjectRef;

#[derive(Debug, Clone)]
pub struct Item {
    pub offset: u64,
    pub len: u64,
    pub source: ItemSource,
    pub part_end: bool,
}

/// One destination's items, each at its offset less `origin`.
#[derive(Clone, Copy, Debug)]
pub struct Placement<'a> {
    pub items: &'a [Item],
    pub origin: u64,
    pub dst: Dst,
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

    /// Asks the kernel to read `items`' object bytes into the page cache ahead of their fills
    /// (`POSIX_FADV_WILLNEED`, Linux; elsewhere nothing): host RAM is the page cache. Best effort.
    pub fn readahead(&self, items: &[Item]) -> Result<()> {
        #[cfg(feature = "store")]
        {
            let objects: std::collections::BTreeMap<_, _> = items.iter().filter_map(|i| match &i.source {
                ItemSource::Object(range) => Some((range.obj.sha256.clone(), range.obj.clone())),
                ItemSource::Inline(_) => None,
            }).collect();
            self.load(&objects.into_values().collect::<Vec<_>>())?;
        }
        for item in items {
            if let ItemSource::Object(range) = &item.source {
                let file = self.file(&range.obj)?;
                #[cfg(target_os = "linux")]
                // SAFETY: advice on a descriptor this source holds open.
                unsafe {
                    use std::os::fd::AsRawFd;
                    libc::posix_fadvise(file.as_raw_fd(), range.off as libc::off_t, range.len as libc::off_t, libc::POSIX_FADV_WILLNEED);
                }
                #[cfg(not(target_os = "linux"))]
                let _ = file;
            }
        }
        Ok(())
    }

    /// Transfer a placement range. Uncovered padding is zeroed through the same engine.
    ///
    /// # Safety
    /// As [`Source::read_placements`].
    pub unsafe fn read_items(
        &self,
        items: &[Item],
        origin: u64,
        dst: Dst,
        threads: usize,
        mode: Mode,
        tally: &IoTally,
    ) -> Result<ReadStats> {
        // SAFETY: inherited from caller.
        unsafe {
            self.read_placements(&[Placement { items, origin, dst }], threads, mode, tally, None)
        }
    }

    /// Transfer many placement ranges in one engine read, so it pipelines across all of them.
    /// Uncovered padding is zeroed through the same engine. `issued` hears of each device
    /// placement (its index and the engine's streams) once all its copies are issued, before
    /// the read returns.
    ///
    /// # Safety
    /// Each `dst` is exclusively writable for its length; its selected item ranges lie inside it.
    /// On CUDA error, retain device destination owners until `super::drain` proves completion.
    /// Inline host bytes use owned engine staging and are not borrowed by pending GPU work.
    pub unsafe fn read_placements(
        &self,
        placements: &[Placement],
        threads: usize,
        mode: Mode,
        tally: &IoTally,
        issued: Option<Issued>,
    ) -> Result<ReadStats> {
        #[cfg(feature = "store")]
        {
            let wanted = placements
                .iter()
                .flat_map(|p| p.items)
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
        for item in placements.iter().flat_map(|p| p.items) {
            if let ItemSource::Object(range) = &item.source {
                files
                    .entry(range.obj.sha256.clone())
                    .or_insert(self.file(&range.obj)?);
            }
        }
        let mut copies = Vec::new();
        // Each copy's placement.
        let mut of = Vec::new();
        let mut inline = 0;
        for (p, &Placement { items, origin, dst }) in placements.iter().enumerate() {
            // Each placement's copies in destination order, gaps included, so contiguous runs
            // merge (`coalesce`).
            let mut ordered: Vec<(u64, Copy)> = Vec::with_capacity(items.len() + 1);
            let mut intervals = Vec::with_capacity(items.len());
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
                intervals.push((start, end));
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
                ordered.push((start, Copy {
                    src,
                    dst: dst.slice(start as usize, item.len as usize),
                }));
            }
            intervals.sort_unstable();
            let mut cursor = 0;
            for (start, end) in intervals {
                if start == end {
                    continue;
                }
                if start < cursor {
                    return Err(invalid("placement items overlap"));
                }
                if start > cursor {
                    ordered.push((cursor, Copy {
                        src: Src::Zero,
                        dst: dst.slice(cursor as usize, (start - cursor) as usize),
                    }));
                }
                cursor = end;
            }
            if cursor < dst.len() as u64 {
                ordered.push((cursor, Copy {
                    src: Src::Zero,
                    dst: dst.slice(cursor as usize, dst.len() - cursor as usize),
                }));
            }
            ordered.sort_by_key(|(at, _)| *at);
            of.resize(of.len() + ordered.len(), p);
            copies.extend(ordered.into_iter().map(|(_, c)| c));
        }
        let (copies, firsts) = super::coalesce(copies);
        // Each run's placements, and the runs each placement waits for.
        let runs: Vec<Vec<usize>> = (0..firsts.len())
            .map(|r| {
                let mut ps = of[firsts[r]..firsts.get(r + 1).copied().unwrap_or(of.len())].to_vec();
                ps.dedup();
                ps
            })
            .collect();
        let left: Vec<AtomicUsize> = (0..placements.len()).map(|_| AtomicUsize::new(0)).collect();
        runs.iter().flatten().for_each(|&p| _ = left[p].fetch_add(1, Relaxed));
        let run_issued = |r: usize, streams: &[usize]| {
            for &p in &runs[r] {
                if left[p].fetch_sub(1, AcqRel) == 1 {
                    issued.expect("only asked with a listener")(p, streams);
                }
            }
        };
        let opts = ReadOpts {
            threads,
            direct: self.direct_io && mode == Mode::Cache,
            ..ReadOpts::default()
        };
        // SAFETY: item ranges and destination bounds checked above; caller supplies exclusivity.
        let mut stats = unsafe { super::read_staged(&copies, &opts, issued.map(|_| &run_issued as Issued)) }?;
        stats.bytes = placements
            .iter()
            .flat_map(|p| p.items)
            .map(|item| item.len)
            .sum();
        tally.direct_bytes.fetch_add(stats.direct, Relaxed);
        tally.inline_bytes.fetch_add(inline, Relaxed);
        if mode == Mode::Buffered {
            tally.buffered_bytes.fetch_add(stats.cached, Relaxed);
        } else {
            tally.cached_bytes.fetch_add(stats.cached, Relaxed);
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
