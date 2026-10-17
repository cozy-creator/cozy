//! The logs a machine keeps, read as a tail or followed as they grow (`tail -F`): across a
//! rotation and the daily roll, no line is missed or read twice.
use std::{
    fmt,
    fs::{self, File},
    io::{self, Read as _, Seek as _, SeekFrom},
    path::PathBuf,
    str::FromStr,
    time::Duration,
};

use cozy_platform::watch::DirWatch;
use tokio::sync::mpsc;

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
            Some(Self { file: file.parse().ok()?, offset: offset.parse().ok()? })
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
        Self { dir, files: Files::Rotated(name) }
    }

    pub fn daily(dir: PathBuf, prefix: &'static str) -> Self {
        Self { dir, files: Files::Daily(prefix) }
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
    pub fn follow(self, start: Start) -> cozy_platform::Result<Follow> {
        fs::create_dir_all(&self.dir).map_err(cozy_platform::Error::from)?;
        let watch = DirWatch::new(&self.dir)?;
        let reader = self.open(start)?;
        Ok(Follow { log: self, reader, watch })
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
                    let Some(next) = Reader::open(&part, 0)? else { continue };
                    if let Some(rest) = reader.rest() {
                        if !send(rest) {
                            return Ok(());
                        }
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
            let Some(part) = parts.get(index) else { return Ok(Reader::empty()) };
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
                    .map(|dir| dir.flatten().filter_map(|e| e.file_name().into_string().ok()).collect())
                    .unwrap_or_default();
                days.retain(|name| name.strip_prefix(prefix).is_some_and(|rest| rest.starts_with('.')) && name.ends_with(".log"));
                days.sort();
                days.split_off(days.len().saturating_sub(2))
            }
        };
        names
            .into_iter()
            .filter_map(|name| {
                let path = self.dir.join(name);
                let meta = fs::metadata(&path).ok()?;
                Some(Part { file: identity(&meta), length: meta.len(), path })
            })
            .collect()
    }
}

impl Follow {
    /// Sends the lines from its start, then each batch as it is appended, until `lines` closes;
    /// a failed read is the last thing sent.
    pub fn run(mut self, lines: mpsc::Sender<io::Result<Lines>>) {
        let watch = &mut self.watch;
        let read = self.log.lines(
            self.reader,
            |batch| lines.blocking_send(Ok(batch)).is_ok(),
            || {
                watch.wait(IDLE)?;
                Ok(!lines.is_closed())
            },
        );
        if let Err(e) = read {
            let _ = lines.blocking_send(Err(e));
        }
    }
}

/// Where the newest `n` whole lines of `parts` begin; None when a file rotated as it was read.
fn last(parts: &[Part], n: u64) -> io::Result<Option<(usize, u64)>> {
    let mut ends = 0;
    for (index, part) in parts.iter().enumerate().rev() {
        let Some(mut file) = open(part)? else { return Ok(None) };
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
    Ok((identity(&file.metadata()?) == part.file).then_some(file))
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
        Self { file: None, handle: None, at: 0, pending: vec![] }
    }

    fn open(part: &Part, at: u64) -> io::Result<Option<Self>> {
        let Some(mut handle) = open(part)? else { return Ok(None) };
        handle.seek(SeekFrom::Start(at))?;
        Ok(Some(Self { file: Some(part.file), handle: Some(handle), at, pending: vec![] }))
    }

    /// The next whole lines the file holds now.
    fn next(&mut self) -> io::Result<Option<Lines>> {
        let (Some(handle), Some(file)) = (self.handle.as_mut(), self.file) else { return Ok(None) };
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
                let after = Cursor { file, offset: self.at - self.pending.len() as u64 };
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
            Lines { text, after: Cursor { file, offset: self.at } }
        })
    }
}

/// What names a file across a rename.
#[cfg(unix)]
fn identity(meta: &fs::Metadata) -> u64 {
    std::os::unix::fs::MetadataExt::ino(meta)
}

#[cfg(not(unix))]
fn identity(meta: &fs::Metadata) -> u64 {
    meta.created()
        .ok()
        .and_then(|at| at.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0, |at| at.as_nanos() as u64)
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;

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
            let mut file = fs::OpenOptions::new().create(true).append(true).open(&path).unwrap();
            file.write_all(format!("line {n}\n").as_bytes()).unwrap();
        };
        write(0);
        let (sender, mut received) = mpsc::channel(4);
        let follow = Log::rotated(dir.path().into(), "transport.log").follow(Start::Last(0)).unwrap();
        let reading = std::thread::spawn(move || follow.run(sender));
        (1..1000).for_each(write);
        let (mut text, mut last) = (String::new(), None);
        while text.lines().count() < 1000 {
            let lines = received.blocking_recv().expect("the follower keeps reading").unwrap();
            text.push_str(&lines.text);
            last = Some(lines.after);
        }
        let expected: String = (0..1000).map(|n| format!("line {n}\n")).collect();
        assert_eq!(text, expected);
        drop(received);
        reading.join().unwrap();

        // Resumed after the last line read: only what came after it.
        write(1000);
        let log = Log::rotated(dir.path().into(), "transport.log");
        assert_eq!(log.read(Start::After(last.unwrap())).unwrap(), "line 1000\n");
        assert_eq!(log.read(Start::Last(2)).unwrap(), "line 999\nline 1000\n");
        // The newest lines reach back into the rotated file; a line still being written waits.
        write(1001);
        fs::OpenOptions::new().append(true).open(&path).unwrap().write_all(b"line 10").unwrap();
        let newest: String = (799..1002).map(|n| format!("line {n}\n")).collect();
        assert_eq!(log.read(Start::Last(203)).unwrap(), newest);
    }
}
