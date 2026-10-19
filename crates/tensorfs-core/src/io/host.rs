//! Host destinations: positioned reads on a few threads, each copy cut on the request grid.

use std::alloc::{self, Layout};
use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock};

use platform::durable;

use super::{Copy, Dst, Error, ReadOpts, ReadStats, Src};
use crate::format::layout::ALIGN;

/// O_DIRECT's alignment of offsets, lengths and addresses.
const A: u64 = ALIGN;

/// Neighbouring reads of one file this close merge into one read (through a bounce buffer when
/// it goes O_DIRECT): a pack's small tensors cost one request, not one each.
const MERGE_GAP: u64 = 64 << 10;

/// One copy's share of a piece.
struct Seg {
    copy: usize,
    offset: u64,
    len: usize,
    dst: usize,
}

/// One read: segments of one file in file order, within one request (or one host range).
struct Piece {
    file: Option<usize>,
    segs: Range<usize>,
}

#[derive(Default)]
struct Tally {
    direct: AtomicU64,
    cached: AtomicU64,
    bounced: AtomicU64,
}

/// # Safety
/// As [`super::read`], for host destinations only.
pub(super) unsafe fn fill(copies: &[Copy], opts: &ReadOpts) -> Result<ReadStats, Error> {
    let grid = (opts.request as u64).max(A).next_multiple_of(A);
    let mut index: HashMap<*const File, usize> = HashMap::new();
    let mut files: Vec<&File> = Vec::new();
    let (mut segs, mut pieces) = (Vec::<Seg>::new(), Vec::<Piece>::new());
    for (i, c) in copies.iter().enumerate() {
        let Dst::Host { ptr, len } = c.dst else { unreachable!("host copies only") };
        let (file, offset) = match c.src {
            Src::File(f, offset) => {
                let i = *index.entry(f as *const File).or_insert_with(|| {
                    files.push(f);
                    files.len() - 1
                });
                (Some(i), offset)
            }
            Src::Host(p) => (None, p as u64),
        };
        let (mut at, end) = (offset, offset + len as u64);
        while at < end {
            let next = ((at / grid + 1) * grid).min(end);
            let joins = file.is_some()
                && pieces.last().is_some_and(|p| {
                    let (first, last) = (&segs[p.segs.start], &segs[p.segs.end - 1]);
                    let last_end = last.offset + last.len as u64;
                    p.file == file && at >= last_end && at - last_end <= MERGE_GAP && next - first.offset <= grid
                });
            segs.push(Seg { copy: i, offset: at, len: (next - at) as usize, dst: ptr as usize + (at - offset) as usize });
            match joins {
                true => pieces.last_mut().unwrap().segs.end += 1,
                false => pieces.push(Piece { file, segs: segs.len() - 1..segs.len() }),
            }
            at = next;
        }
    }
    let direct: Vec<OnceLock<Option<File>>> = files.iter().map(|_| OnceLock::new()).collect();
    let (tally, next, failed, error) = (Tally::default(), AtomicUsize::new(0), AtomicBool::new(false), Mutex::new(None));
    let work = || {
        let mut bounce = Bounce::default();
        while !failed.load(Relaxed) {
            let Some(p) = pieces.get(next.fetch_add(1, Relaxed)) else { return };
            let segs = &segs[p.segs.clone()];
            // SAFETY: a segment lies inside its copy's destination, which the caller lends us.
            let dst = |s: &Seg| unsafe { std::slice::from_raw_parts_mut(s.dst as *mut u8, s.len) };
            let Some(i) = p.file else {
                // SAFETY: and inside its host source.
                dst(&segs[0]).copy_from_slice(unsafe { std::slice::from_raw_parts(segs[0].offset as *const u8, segs[0].len) });
                continue;
            };
            let (file, last) = (files[i], &segs[segs.len() - 1]);
            let (from, to) = (segs[0].offset, last.offset + last.len as u64);
            let cold = opts.direct && segs.iter().any(|s| !durable::cached(file, s.offset, s.len as u64).unwrap_or(false));
            let d = cold.then(|| direct[i].get_or_init(|| durable::open_direct(file).ok().flatten()).as_ref());
            let r = match (d.flatten(), segs) {
                (Some(d), [s]) => direct_piece(d, s.offset, dst(s), &mut bounce, &tally),
                (Some(d), _) => bounce.fill(d, from, (to - from) as usize).map(|cover| {
                    for s in segs {
                        let at = (s.offset - from) as usize;
                        dst(s).copy_from_slice(&cover[at..at + s.len]);
                    }
                    let n = segs.iter().map(|s| s.len as u64).sum();
                    tally.direct.fetch_add(n, Relaxed);
                    tally.bounced.fetch_add(n, Relaxed);
                }),
                (None, _) => segs.iter().try_for_each(|s| {
                    read_exact_at(file, dst(s), s.offset)?;
                    tally.cached.fetch_add(s.len as u64, Relaxed);
                    Ok(())
                }),
            };
            if let Err(source) = r {
                error.lock().unwrap().get_or_insert(Error { copy: segs[0].copy, source });
                failed.store(true, Relaxed);
            }
        }
    };
    match opts.threads.clamp(1, pieces.len().max(1)) {
        1 => work(),
        n => std::thread::scope(|s| {
            for _ in 1..n {
                s.spawn(work);
            }
            work();
        }),
    }
    if let Some(e) = error.into_inner().unwrap() {
        return Err(e);
    }
    Ok(ReadStats {
        direct: tally.direct.into_inner(),
        cached: tally.cached.into_inner(),
        bounced: tally.bounced.into_inner(),
        ..ReadStats::default()
    })
}

/// `dst.len()` bytes at `at` past the page cache: the aligned middle straight into `dst` when
/// its address agrees with the file offset, the rest through `bounce`.
fn direct_piece(d: &File, at: u64, dst: &mut [u8], bounce: &mut Bounce, tally: &Tally) -> io::Result<()> {
    let end = at + dst.len() as u64;
    let (lo, hi) = (at.next_multiple_of(A), end - end % A);
    let mut bounced = dst.len() as u64;
    if (dst.as_ptr() as u64) % A == at % A && lo < hi {
        let (head, rest) = dst.split_at_mut((lo - at) as usize);
        let (middle, tail) = rest.split_at_mut((hi - lo) as usize);
        read_direct(d, middle, lo, middle.len())?;
        bounce.read(d, at, head)?;
        bounce.read(d, hi, tail)?;
        bounced = (head.len() + tail.len()) as u64;
    } else {
        bounce.read(d, at, dst)?;
    }
    tally.direct.fetch_add(dst.len() as u64, Relaxed);
    tally.bounced.fetch_add(bounced, Relaxed);
    Ok(())
}

/// At least `need` bytes at `at` into `buf`, both aligned, by O_DIRECT reads. The file may end
/// inside `buf` but not before `need`.
fn read_direct(d: &File, buf: &mut [u8], at: u64, need: usize) -> io::Result<()> {
    let mut done = 0;
    while done < need {
        let n = match read_at(d, &mut buf[done..], at + done as u64) {
            Ok(n) => n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        };
        // A short read resumes on the alignment; one that ends before `need` is the file's end.
        let next = done + n;
        let next = if next < need { next - next % A as usize } else { next };
        if next <= done {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, format!("O_DIRECT read ended at {}", at + done as u64)));
        }
        done = next;
    }
    Ok(())
}

fn read_exact_at(f: &File, mut buf: &mut [u8], mut at: u64) -> io::Result<()> {
    while !buf.is_empty() {
        match read_at(f, buf, at) {
            Ok(0) => return Err(io::Error::new(io::ErrorKind::UnexpectedEof, format!("the file ends at {at}"))),
            Ok(n) => (buf, at) = (&mut buf[n..], at + n as u64),
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

#[cfg(unix)]
pub(super) fn read_at(f: &File, buf: &mut [u8], at: u64) -> io::Result<usize> {
    std::os::unix::fs::FileExt::read_at(f, buf, at)
}

#[cfg(windows)]
pub(super) fn read_at(f: &File, buf: &mut [u8], at: u64) -> io::Result<usize> {
    std::os::windows::fs::FileExt::seek_read(f, buf, at)
}

/// One thread's aligned buffer for O_DIRECT reads its destination cannot take.
#[derive(Default)]
pub(super) struct Bounce {
    buf: Option<(*mut u8, Layout)>,
}

impl Bounce {
    /// `out.len()` bytes at `at`.
    fn read(&mut self, d: &File, at: u64, out: &mut [u8]) -> io::Result<()> {
        if !out.is_empty() {
            out.copy_from_slice(self.fill(d, at, out.len())?);
        }
        Ok(())
    }

    /// `len` bytes at `at`, read through the aligned cover of that range.
    fn fill(&mut self, d: &File, at: u64, len: usize) -> io::Result<&[u8]> {
        let from = at - at % A;
        let need = (at - from) as usize + len;
        let buf = self.get(need.next_multiple_of(A as usize));
        read_direct(d, buf, from, need)?;
        Ok(&buf[(at - from) as usize..need])
    }

    pub(super) fn get(&mut self, len: usize) -> &mut [u8] {
        if self.buf.is_none_or(|(_, l)| l.size() < len) {
            self.free();
            let layout = Layout::from_size_align(len, A as usize).expect("a bounce buffer layout");
            // SAFETY: a non-zero size; freed in `free`.
            let ptr = unsafe { alloc::alloc_zeroed(layout) };
            if ptr.is_null() {
                alloc::handle_alloc_error(layout);
            }
            self.buf = Some((ptr, layout));
        }
        let (ptr, _) = self.buf.unwrap();
        // SAFETY: ours, initialized, `layout.size() >= len`.
        unsafe { std::slice::from_raw_parts_mut(ptr, len) }
    }

    fn free(&mut self) {
        if let Some((ptr, layout)) = self.buf.take() {
            // SAFETY: allocated in `get` with this layout.
            unsafe { alloc::dealloc(ptr, layout) };
        }
    }
}

impl Drop for Bounce {
    fn drop(&mut self) {
        self.free();
    }
}
