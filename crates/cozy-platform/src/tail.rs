//! A log read as a tail or followed as it grows (`tail -F`): across a rotation and a daily roll,
//! no line is missed or read twice. TensorD's logs and Calcifer's.
use std::{
    fmt,
    fs::{self, File},
    io::{self, Read as _, Seek as _, SeekFrom},
    path::PathBuf,
    str::FromStr,
    time::Duration,
};

#[cfg(windows)]
use std::os::windows::io::AsRawHandle as _;

#[cfg(windows)]
use windows_sys::Win32::Storage::FileSystem::{
    BY_HANDLE_FILE_INFORMATION, GetFileInformationByHandle,
};

use crate::watch::DirWatch;

/// At most this much is read at once.
const BATCH: usize = 64 << 10;
/// How long a caught-up follower sleeps before it looks whether its reader left.
const IDLE: Duration = Duration::from_secs(1);

/// A log's files in one directory, the newest the one written.
#[derive(Clone, Debug)]
pub struct Log {
    dir: PathBuf,
    files: Files,
}

#[derive(Clone, Copy, Debug)]
enum Files {
    /// `name`, which rotates by a rename to `name.1`.
    Rotated(&'static str),
    /// `<prefix>.<date>.log`, a file a day; the newest two.
    Daily(&'static str),
}

/// A place in a log: a file, by an identity a rename keeps, and an offset in it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cursor {
    file: u64,
    offset: u64,
}

impl fmt::Display for Cursor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.file, self.offset)
    }
}

impl FromStr for Cursor {
    type Err = String;
    fn from_str(text: &str) -> Result<Self, String> {
        let parse = || {
            let (file, offset) = text.split_once(':')?;
            Some(Self {
                file: file.parse().ok()?,
                offset: offset.parse().ok()?,
            })
        };
        parse().ok_or_else(|| format!("{text:?} is not a log cursor"))
    }
}

/// Whole lines read from a log, and the place after them.
#[derive(Debug)]
pub struct Lines {
    pub text: String,
    pub after: Cursor,
}

/// Where a read begins: at the newest N whole lines (0: all it keeps), or after a cursor.
#[derive(Clone, Copy, Debug)]
pub enum Start {
    Last(u64),
    After(Cursor),
}

/// One of the log's files as listed.
struct Part {
    file: u64,
    length: u64,
    path: PathBuf,
}

/// A log being followed: watched, then opened at its start, so nothing appended after is missed
/// and its first file stays readable whatever rotates before the first read.
pub struct Follow {
    log: Log,
    reader: Reader,
    watch: DirWatch,
}

impl Log {
    pub fn rotated(dir: PathBuf, name: &'static str) -> Self {
        Self {
            dir,
            files: Files::Rotated(name),
        }
    }

    pub fn daily(dir: PathBuf, prefix: &'static str) -> Self {
        Self {
            dir,
            files: Files::Daily(prefix),
        }
    }

    /// The whole lines from `start` to the end now.
    pub fn read(&self, start: Start) -> io::Result<String> {
        let mut text = String::new();
        self.lines(
            self.open(start)?,
            |lines| {
                text.push_str(&lines.text);
                true
            },
            || Ok(false),
        )?;
        Ok(text)
    }

    /// Unsupported where this OS cannot watch a directory.
    pub fn follow(self, start: Start) -> crate::Result<Follow> {
        fs::create_dir_all(&self.dir).map_err(crate::Error::from)?;
        let watch = DirWatch::new(&self.dir)?;
        let reader = self.open(start)?;
        Ok(Follow {
            log: self,
            reader,
            watch,
        })
    }

    /// Hands `send` each batch of whole lines from `reader` on; at the end of the newest file,
    /// reads on while `idle` says to. Ends when either says stop.
    fn lines(
        &self,
        mut reader: Reader,
        mut send: impl FnMut(Lines) -> bool,
        mut idle: impl FnMut() -> io::Result<bool>,
    ) -> io::Result<()> {
        loop {
            // Newer first, then drained: what was written before a rotation is read before moving on.
            let newer = self.newer(reader.file);
            while let Some(lines) = reader.next()? {
                if !send(lines) {
                    return Ok(());
                }
            }
            match newer {
                Some(part) => {
                    let Some(next) = Reader::open(&part, 0)? else {
                        continue;
                    };
                    if let Some(rest) = reader.rest()
                        && !send(rest)
                    {
                        return Ok(());
                    }
                    reader = next;
                }
                None if !idle()? => return Ok(()),
                None => {}
            }
        }
    }

    fn open(&self, start: Start) -> io::Result<Reader> {
        loop {
            let parts = self.parts();
            let (index, offset) = match start {
                Start::After(cursor) => parts
                    .iter()
                    .position(|p| p.file == cursor.file && cursor.offset <= p.length)
                    .map_or((0, 0), |at| (at, cursor.offset)),
                Start::Last(0) => (0, 0),
                Start::Last(n) => match last(&parts, n)? {
                    Some(at) => at,
                    None => continue,
                },
            };
            let Some(part) = parts.get(index) else {
                return Ok(Reader::empty());
            };
            if let Some(reader) = Reader::open(part, offset)? {
                return Ok(reader);
            }
        }
    }

    /// The file after `file`, or the oldest when `file` is gone (or there was none).
    fn newer(&self, file: Option<u64>) -> Option<Part> {
        let mut parts = self.parts();
        let next = match parts.iter().position(|p| Some(p.file) == file) {
            Some(at) => at + 1,
            None => 0,
        };
        (next < parts.len()).then(|| parts.swap_remove(next))
    }

    /// The log's files now, oldest first.
    fn parts(&self) -> Vec<Part> {
        let names = match self.files {
            Files::Rotated(name) => vec![format!("{name}.1"), name.to_owned()],
            Files::Daily(prefix) => {
                let mut days: Vec<String> = fs::read_dir(&self.dir)
                    .map(|dir| {
                        dir.flatten()
                            .filter_map(|e| e.file_name().into_string().ok())
                            .collect()
                    })
                    .unwrap_or_default();
                days.retain(|name| {
                    name.strip_prefix(prefix)
                        .is_some_and(|rest| rest.starts_with('.'))
                        && name.ends_with(".log")
                });
                days.sort();
                days.split_off(days.len().saturating_sub(2))
            }
        };
        names
            .into_iter()
            .filter_map(|name| {
                let path = self.dir.join(name);
                let file = File::open(&path).ok()?;
                let length = file.metadata().ok()?.len();
                Some(Part {
                    file: identity(&file).ok()?,
                    length,
                    path,
                })
            })
            .collect()
    }
}

impl Follow {
    /// Hands `send` the lines from its start, then each batch as it is appended, until `send`
    /// says stop or, caught up, `wanted` says nobody reads any more.
    pub fn run(
        mut self,
        send: impl FnMut(Lines) -> bool,
        wanted: impl Fn() -> bool,
    ) -> io::Result<()> {
        let watch = &mut self.watch;
        self.log.lines(self.reader, send, || {
            watch.wait(IDLE)?;
            Ok(wanted())
        })
    }
}

/// Where the newest `n` whole lines of `parts` begin; None when a file rotated as it was read.
fn last(parts: &[Part], n: u64) -> io::Result<Option<(usize, u64)>> {
    let mut ends = 0;
    for (index, part) in parts.iter().enumerate().rev() {
        let Some(mut file) = open(part)? else {
            return Ok(None);
        };
        let mut end = part.length;
        while end > 0 {
            let from = end.saturating_sub(BATCH as u64);
            let mut block = vec![0; (end - from) as usize];
            file.seek(SeekFrom::Start(from))?;
            file.read_exact(&mut block)?;
            // The first line end back from the end closes the newest whole line.
            for (at, _) in block.iter().enumerate().rev().filter(|(_, b)| **b == b'\n') {
                ends += 1;
                if ends > n {
                    return Ok(Some((index, from + at as u64 + 1)));
                }
            }
            end = from;
        }
    }
    Ok(Some((0, 0)))
}

/// `part`'s file; None when its name now holds another: it rotated since it was listed.
fn open(part: &Part) -> io::Result<Option<File>> {
    let file = match File::open(&part.path) {
        Ok(file) => file,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e),
    };
    Ok((identity(&file)? == part.file).then_some(file))
}

/// One file read from a place on, handing out whole lines.
struct Reader {
    file: Option<u64>,
    handle: Option<File>,
    /// The offset of the next byte read.
    at: u64,
    pending: Vec<u8>,
}

impl Reader {
    fn empty() -> Self {
        Self {
            file: None,
            handle: None,
            at: 0,
            pending: vec![],
        }
    }

    fn open(part: &Part, at: u64) -> io::Result<Option<Self>> {
        let Some(mut handle) = open(part)? else {
            return Ok(None);
        };
        handle.seek(SeekFrom::Start(at))?;
        Ok(Some(Self {
            file: Some(part.file),
            handle: Some(handle),
            at,
            pending: vec![],
        }))
    }

    /// The next whole lines the file holds now.
    fn next(&mut self) -> io::Result<Option<Lines>> {
        let (Some(handle), Some(file)) = (self.handle.as_mut(), self.file) else {
            return Ok(None);
        };
        loop {
            let mut chunk = vec![0; BATCH];
            let read = handle.read(&mut chunk)?;
            if read == 0 {
                return Ok(None);
            }
            self.at += read as u64;
            self.pending.extend_from_slice(&chunk[..read]);
            if let Some(end) = self.pending.iter().rposition(|b| *b == b'\n') {
                let text = String::from_utf8_lossy(&self.pending[..=end]).into_owned();
                self.pending.drain(..=end);
                let after = Cursor {
                    file,
                    offset: self.at - self.pending.len() as u64,
                };
                return Ok(Some(Lines { text, after }));
            }
        }
    }

    /// A last line its writer left unended, once a newer file took over.
    fn rest(&mut self) -> Option<Lines> {
        let file = self.file?;
        (!self.pending.is_empty()).then(|| {
            let mut text = String::from_utf8_lossy(&std::mem::take(&mut self.pending)).into_owned();
            text.push('\n');
            Lines {
                text,
                after: Cursor {
                    file,
                    offset: self.at,
                },
            }
        })
    }
}

/// What names a file across a rename: its inode, or on Windows its file index (a creation time
/// is no name there: a file made under a just-renamed name inherits the old one's).
#[cfg(unix)]
fn identity(file: &File) -> io::Result<u64> {
    Ok(std::os::unix::fs::MetadataExt::ino(&file.metadata()?))
}

#[cfg(windows)]
fn identity(file: &File) -> io::Result<u64> {
    let mut info = BY_HANDLE_FILE_INFORMATION::default();
    // SAFETY: an open file's handle and a writable information block.
    if unsafe { GetFileInformationByHandle(file.as_raw_handle(), &mut info) } == 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(u64::from(info.nFileIndexHigh) << 32 | u64::from(info.nFileIndexLow))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    };

    /// Lines written while a follower reads, rotated twice partway, arrive once each and in order;
    /// a follow resumed from a cursor carries on with the next line.
    #[test]
    fn a_follow_crosses_rotations_without_missing_or_repeating_a_line() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("transport.log");
        use std::io::Write as _;
        let write = |n: usize| {
            if n > 0 && n.is_multiple_of(400) {
                fs::rename(&path, dir.path().join("transport.log.1")).unwrap();
            }
            let mut file = fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .unwrap();
            file.write_all(format!("line {n}\n").as_bytes()).unwrap();
        };
        write(0);
        let (sender, received) = mpsc::sync_channel(4);
        let done = Arc::new(AtomicBool::new(false));
        let follow = Log::rotated(dir.path().into(), "transport.log")
            .follow(Start::Last(0))
            .unwrap();
        let reading = {
            let done = done.clone();
            std::thread::spawn(move || {
                follow.run(
                    |lines| sender.send(lines).is_ok(),
                    || !done.load(Ordering::Relaxed),
                )
            })
        };
        (1..1000).for_each(write);
        let (mut text, mut last) = (String::new(), None);
        while text.lines().count() < 1000 {
            let lines = received.recv().expect("the follower keeps reading");
            text.push_str(&lines.text);
            last = Some(lines.after);
        }
        let expected: String = (0..1000).map(|n| format!("line {n}\n")).collect();
        assert_eq!(text, expected);
        done.store(true, Ordering::Relaxed);
        drop(received);
        reading.join().unwrap().unwrap();

        // Resumed after the last line read: only what came after it.
        write(1000);
        let log = Log::rotated(dir.path().into(), "transport.log");
        assert_eq!(
            log.read(Start::After(last.unwrap())).unwrap(),
            "line 1000\n"
        );
        assert_eq!(log.read(Start::Last(2)).unwrap(), "line 999\nline 1000\n");
        // The newest lines reach back into the rotated file; a line still being written waits.
        write(1001);
        fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(b"line 10")
            .unwrap();
        let newest: String = (799..1002).map(|n| format!("line {n}\n")).collect();
        assert_eq!(log.read(Start::Last(203)).unwrap(), newest);
    }
}
