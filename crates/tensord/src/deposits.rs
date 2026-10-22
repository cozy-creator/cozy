//! Host layouts executors deposit (design memory-varena §5.4): sealed memfds varena filled in an
//! executor, kept by TensorD as opaque handles under (manifest, layout), so a later executor of
//! the same weights maps them from RAM (D9's refill path). A layout goes back to the host only
//! once no live executor maps or holds it (releasing one still in use frees nothing), the least
//! recently granted first. An admitted layout reserves its bytes until filled.
use platform::process::Birth;
use std::{collections::BTreeMap, fs::File, io, sync::Mutex, time::Instant};

#[derive(Default)]
pub struct Deposits {
    held: Mutex<BTreeMap<(String, String), Held>>,
    /// Layouts a `call#rank` seals and has not deposited yet, with the bytes admitted for it.
    /// This lock also serializes admissions and the handoff to the deposited, filling layout.
    sealing: Mutex<BTreeMap<(String, String), Pending>>,
}

struct Pending {
    who: String,
    bytes: u64,
}

/// One kept layout: when it was last granted and every executor process it was handed to.
struct Held {
    file: File,
    used: Instant,
    holders: Vec<Birth>,
}

impl Held {
    /// Whether a live executor it was handed to still maps it or holds it open.
    fn in_use(&self) -> bool {
        self.holders.iter().any(|birth| birth.ended().is_ok_and(|ended| !ended) && uses(birth.pid, &self.file))
    }

    /// What its filler has still to fill: an admitted fill's reservation.
    fn unfilled(&self) -> u64 {
        match reopen(&self.file).is_some_and(|file| state(&file).0) {
            true => self.file.metadata().map_or(0, |m| m.len()).saturating_sub(allocated(&self.file)),
            false => 0,
        }
    }
}

fn key(manifest: &str, layout: &str) -> (String, String) {
    (manifest.trim_start_matches("sha256:").into(), layout.into())
}

fn call_of(who: &str) -> &str {
    who.rsplit_once('#').map_or(who, |(call, _)| call)
}

impl Deposits {
    /// Keeps `layout` that the executors (`holders`) deposited, refused unless sealed against
    /// writes and resizing; a second deposit of one layout replaces the first.
    pub fn deposit(&self, manifest: &str, layout: &str, file: File, holders: &[Birth]) -> io::Result<()> {
        sealed(&file)?;
        let mut sealing = self.sealing.lock().unwrap();
        let held = Held { file, used: Instant::now(), holders: holders.to_vec() };
        self.held.lock().unwrap().insert(key(manifest, layout), held);
        sealing.remove(&key(manifest, layout));
        Ok(())
    }

    /// A read-only description of the deposited layout, if one is held, for `holders`, and whether
    /// its filler still fills it: an executor deposits a layout as it opens it, before its bytes.
    /// One its filler left unfinished (it died) goes, and the next asker seals anew.
    pub fn grant(&self, manifest: &str, layout: &str, holders: &[Birth]) -> Option<(File, bool)> {
        let mut held = self.held.lock().unwrap();
        let kept = held.get_mut(&key(manifest, layout))?;
        let file = reopen(&kept.file)?;
        let (filling, complete) = state(&file);
        if !filling && !complete {
            held.remove(&key(manifest, layout));
            return None;
        }
        kept.used = Instant::now();
        for birth in holders {
            if !kept.holders.contains(birth) {
                kept.holders.push(birth.clone());
            }
        }
        Some((file, filling))
    }

    /// Whether `who` (`call#rank`) seals `layout` of `bytes`: `admit` decides, beside the bytes
    /// admitted fills have still to fill, under this one lock. One copy per layout: while
    /// another seals it, that one is named and the asker waits for its deposit. False: not
    /// admitted, or a deposit landed since the caller looked (it should ask `grant` again).
    /// `regardless`: the asker seals whatever the answer (an older Runtime or TensorFS), so it is
    /// the layout's one sealer anyway and its bytes are reserved. `admit` may inspect and release
    /// held deposits, but must not re-enter sealing or deposit.
    pub fn seal(&self, manifest: &str, layout: &str, who: &str, bytes: u64, regardless: bool, admit: impl FnOnce(u64) -> bool) -> Result<bool, String> {
        let mut sealing = self.sealing.lock().unwrap();
        if let Some(other) = sealing.get(&key(manifest, layout)) {
            return if other.who == who { Ok(true) } else { Err(other.who.clone()) };
        }
        let held = self.held.lock().unwrap();
        if held.contains_key(&key(manifest, layout)) {
            return Ok(false);
        }
        let reserved = sealing.values().map(|p| p.bytes).sum::<u64>() + held.values().map(Held::unfilled).sum::<u64>();
        drop(held);
        if !admit(reserved) && !regardless {
            return Ok(false);
        }
        sealing.insert(key(manifest, layout), Pending { who: who.into(), bytes });
        Ok(true)
    }

    /// Bytes admitted fills have still to fill: layouts sealing and not yet deposited, and
    /// deposited ones their fillers still fill.
    pub fn reserved(&self) -> u64 {
        let sealing = self.sealing.lock().unwrap();
        sealing.values().map(|p| p.bytes).sum::<u64>() + self.held.lock().unwrap().values().map(Held::unfilled).sum::<u64>()
    }

    /// Call `call` ended: a layout it was sealing and never deposited is nobody's to wait for.
    pub fn ended(&self, call: &str) {
        self.sealing.lock().unwrap().retain(|_, p| call_of(&p.who) != call);
    }

    /// Whether a layout of this digest is held.
    pub fn holds(&self, layout: &str) -> bool {
        self.held.lock().unwrap().keys().any(|(_, held)| held == layout)
    }

    /// Bytes of the layouts no live executor uses: what releasing them gives back.
    pub fn idle_bytes(&self) -> u64 {
        self.held.lock().unwrap().values().filter(|h| !h.in_use()).map(|h| allocated(&h.file)).sum()
    }

    /// Lets the least recently granted layout no live executor uses go; whether one went.
    pub fn release_one(&self) -> bool {
        let mut held = self.held.lock().unwrap();
        let idle = held.iter().filter(|(_, h)| !h.in_use());
        let oldest = idle.min_by_key(|(_, h)| h.used).map(|(key, _)| key.clone());
        oldest.and_then(|key| held.remove(&key)).is_some()
    }
}

/// Whether `pid` maps `file` or holds it open (its /proc entries name the same inode). A process
/// whose /proc this one cannot read counts as using it.
#[cfg(target_os = "linux")]
fn uses(pid: u32, file: &File) -> bool {
    use std::os::unix::fs::MetadataExt;
    let Ok(of) = file.metadata() else { return true };
    let (Ok(maps), Ok(fds)) = (std::fs::read_to_string(format!("/proc/{pid}/maps")), std::fs::read_dir(format!("/proc/{pid}/fd"))) else {
        return true;
    };
    let device = format!("{:02x}:{:02x}", libc::major(of.dev()), libc::minor(of.dev()));
    let inode = of.ino().to_string();
    maps.lines().any(|line| line.split_whitespace().skip(3).take(2).eq([device.as_str(), inode.as_str()]))
        || fds.flatten().any(|fd| std::fs::metadata(fd.path()).is_ok_and(|m| m.dev() == of.dev() && m.ino() == of.ino()))
}
#[cfg(not(target_os = "linux"))]
fn uses(_: u32, _: &File) -> bool {
    true
}

/// The bytes a layout holds in RAM: a sparse memfd holds only what was filled.
#[cfg(target_os = "linux")]
fn allocated(file: &File) -> u64 {
    file.metadata().map_or(0, |m| std::os::unix::fs::MetadataExt::blocks(&m) * 512)
}
#[cfg(not(target_os = "linux"))]
fn allocated(_: &File) -> u64 {
    0
}
/// The seals varena's `adopt_sealed` asks of a layout: resize and seal, and write or future-write
/// (varena seals a layout it filled with future-write, its own mapping still open).
#[cfg(target_os = "linux")]
fn sealed(file: &File) -> io::Result<()> {
    use platform::shm::{RESIZE_SEALS, WRITE_SEALS};
    let seals = platform::shm::seals(std::os::fd::AsFd::as_fd(file))?;
    match seals & RESIZE_SEALS == RESIZE_SEALS && seals & WRITE_SEALS != 0 {
        true => Ok(()),
        false => Err(io::Error::new(io::ErrorKind::InvalidInput, "a deposited layout must be sealed against writes and resizing")),
    }
}
#[cfg(not(target_os = "linux"))]
fn sealed(_: &File) -> io::Result<()> {
    Err(io::Error::new(io::ErrorKind::Unsupported, "sealed host layouts are Linux's"))
}

/// Whether its filler still fills the layout, and whether every region is in (a file that is
/// no varena layout counts as complete).
#[cfg(target_os = "linux")]
fn state(file: &File) -> (bool, bool) {
    let fd = std::os::fd::AsFd::as_fd(file);
    (varena::host::filling(fd).unwrap_or(false), varena::host::complete(fd).unwrap_or(true))
}
#[cfg(not(target_os = "linux"))]
fn state(_: &File) -> (bool, bool) {
    (false, true)
}

#[cfg(target_os = "linux")]
fn reopen(file: &File) -> Option<File> {
    File::open(format!("/proc/self/fd/{}", std::os::fd::AsRawFd::as_raw_fd(file))).ok()
}
#[cfg(not(target_os = "linux"))]
fn reopen(_: &File) -> Option<File> {
    None
}

#[cfg(all(test, target_os = "linux"))]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn layout(bytes: u64) -> File {
        let layout = File::from(platform::shm::memfd_named("layout").unwrap());
        layout.set_len(bytes).unwrap();
        io::Write::write_all(&mut &layout, &vec![1u8; bytes as usize]).unwrap();
        platform::shm::seal(std::os::fd::AsFd::as_fd(&layout)).unwrap();
        layout
    }

    /// A process that maps `file` read-only, touches every page and closes its descriptor: it
    /// holds the layout by its mapping alone, until killed.
    pub(crate) fn mapper(file: &File) -> std::process::Child {
        let path = format!("/proc/{}/fd/{}", std::process::id(), std::os::fd::AsRawFd::as_raw_fd(file));
        let script = format!(
            "import mmap, sys\nwith open('{path}', 'rb') as f:\n    m = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)\n\
             s = sum(m[i] for i in range(0, len(m), 4096))\nprint('mapped', flush=True)\nsys.stdin.read()"
        );
        let mut child = std::process::Command::new("python3")
            .args(["-c", &script])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        let mut line = String::new();
        io::BufRead::read_line(&mut io::BufReader::new(child.stdout.as_mut().unwrap()), &mut line).unwrap();
        assert_eq!(line.trim(), "mapped");
        child
    }

    /// varena seals a layout it filled with future-write: kept, and granted read-only. An unsealed
    /// one is refused.
    #[test]
    fn a_layout_varena_sealed_is_kept_and_an_unsealed_one_refused() {
        let deposits = Deposits::default();
        let unsealed = File::from(platform::shm::memfd_named("unsealed").unwrap());
        assert!(deposits.deposit("sha256:m", "l", unsealed, &[]).is_err());
        deposits.deposit("sha256:m", "l", layout(4096), &[]).unwrap();
        let (granted, filling) = deposits.grant("m", "l", &[]).expect("held");
        assert!(!filling, "no filler holds it");
        assert_eq!(granted.metadata().unwrap().len(), 4096);
        let path = format!("/proc/self/fd/{}", std::os::fd::AsRawFd::as_raw_fd(&granted));
        let written = std::fs::OpenOptions::new().write(true).open(path).and_then(|mut f| io::Write::write_all(&mut f, b"x"));
        assert!(written.is_err(), "nobody writes a deposited layout");
    }

    /// One copy per layout: while a call's admitted rank seals it, another asker is told who, until
    /// the deposit lands or that call ends; the sealer itself may ask again. An asker not admitted
    /// reserves nothing, and another layout seals beside a pending one where admitted.
    #[test]
    fn a_layout_has_one_sealer_until_its_deposit_or_its_call_ends() {
        let deposits = Deposits::default();
        assert_eq!(deposits.seal("sha256:m", "l", "run-z#0", 4096, false, |_| false), Ok(false));
        assert_eq!(deposits.seal("sha256:m", "l", "run-a#0", 4096, false, |reserved| reserved == 0), Ok(true));
        assert_eq!(deposits.seal("m", "l", "run-a#0", 4096, false, |_| panic!("a retry keeps its admission")), Ok(true));
        assert_eq!(deposits.seal("m", "l", "run-a#1", 4096, false, |_| true), Err("run-a#0".into()), "rank 1 waits for rank 0");
        assert_eq!(deposits.seal("m", "l", "run-b#0", 4096, false, |_| false), Err("run-a#0".into()), "another call waits too");
        assert_eq!(deposits.seal("m", "other", "run-b#0", 10, false, |reserved| reserved == 4096), Ok(true));
        assert_eq!(deposits.reserved(), 4096 + 10);
        deposits.deposit("m", "l", layout(4096), &[]).unwrap();
        assert_eq!(deposits.reserved(), 10, "a filled deposit reserves nothing");
        assert_eq!(deposits.seal("m", "l", "run-c#0", 4096, false, |_| panic!("the deposit already landed")), Ok(false));
        assert!(deposits.grant("m", "l", &[]).is_some());
        deposits.ended("run-b");
        assert_eq!(deposits.reserved(), 0);
        assert_eq!(deposits.seal("m", "other", "run-d#1", 10, false, |_| true), Ok(true));
    }

    /// An older Runtime or TensorFS seals whatever the answer: it is the layout's one sealer even
    /// unadmitted, so a second asker waits for its copy instead of filling another.
    #[test]
    fn an_older_peer_seals_once_and_the_next_waits() {
        let deposits = Deposits::default();
        assert_eq!(deposits.seal("m", "l", "run-a#0", 8192, true, |_| false), Ok(true));
        assert_eq!(deposits.seal("m", "l", "run-a#1", 8192, true, |_| false), Err("run-a#0".into()));
        assert_eq!(deposits.reserved(), 8192);
    }

    /// Concurrent admissions are one at a time, and each sees what the others admitted.
    #[test]
    fn concurrent_sealers_see_each_others_reservation() {
        use std::sync::{atomic::{AtomicUsize, Ordering}, Barrier};
        let deposits = Deposits::default();
        let start = Barrier::new(2);
        let checks = AtomicUsize::new(0);
        let answers = std::thread::scope(|scope| {
            let launch = |manifest, who| {
                let (deposits, start, checks) = (&deposits, &start, &checks);
                scope.spawn(move || {
                    start.wait();
                    deposits.seal(manifest, manifest, who, 10, false, |reserved| {
                        checks.fetch_add(1, Ordering::Relaxed);
                        reserved + 10 <= 15
                    }).unwrap()
                })
            };
            let a = launch("m-a", "call-a#0");
            let b = launch("m-b", "call-b#0");
            [a.join().unwrap(), b.join().unwrap()]
        });
        assert_eq!(answers.into_iter().filter(|admitted| *admitted).count(), 1);
        assert_eq!(checks.load(Ordering::Relaxed), 2);
    }

    /// The pending reservation hands over to the deposited memfd's fill: what it has still to
    /// fill stays reserved until its bytes land.
    #[test]
    fn a_deposited_layout_reserves_its_unfilled_bytes_until_filled() {
        use tensorfs::io::{source::{IoTally, Source}, ItemSource, Span};
        let deposits = Deposits::default();
        let spans = [Span { what: "model/weight#data".into(), len: 1 << 20, source: ItemSource::Inline(vec![7; 1 << 20].into()) }];
        let layout = varena::layout::Layout::build(&spans, &[vec!["model/weight".into()]]).unwrap();
        assert_eq!(deposits.seal("m", "l", "run-a#0", 1 << 20, false, |_| true), Ok(true));
        let (file, fill) = varena::host::open_sealed("admission", &layout, None, None).unwrap();
        deposits.deposit("m", "l", file.into(), &[]).unwrap();
        assert!(deposits.grant("m", "l", &[]).unwrap().1);
        assert!(deposits.reserved() >= 1 << 20, "{}", deposits.reserved());
        fill.run(&layout, &Source::files(vec![], false, 0), 1, &IoTally::default()).unwrap();
        assert!(!deposits.grant("m", "l", &[]).unwrap().1);
        assert_eq!(deposits.reserved(), 0);
        assert_eq!(deposits.seal("m-other", "other", "run-a#0", 1, false, |_| {
            // Admission may inspect and release completed deposits without a lock cycle.
            assert!(deposits.idle_bytes() >= 1 << 20);
            assert!(deposits.release_one());
            true
        }), Ok(true));
    }

    /// A layout a live executor maps stays (releasing it frees nothing); one whose live holders
    /// no longer map it, or whose holders all ended, goes back, the least recently granted first.
    #[test]
    fn only_a_layout_no_live_executor_maps_goes_back() {
        let deposits = Deposits::default();
        let (mapped, unmapped) = (layout(4096), layout(8192));
        let mut mapping = mapper(&mapped);
        let mut idle = std::process::Command::new("sleep").arg("600").spawn().unwrap();
        let (mapping_birth, idle_birth) = (Birth::of(mapping.id()).unwrap(), Birth::of(idle.id()).unwrap());
        deposits.deposit("m", "mapped", mapped, std::slice::from_ref(&mapping_birth)).unwrap();
        deposits.deposit("m", "unmapped", unmapped, std::slice::from_ref(&idle_birth)).unwrap();
        assert!(deposits.holds("mapped") && !deposits.holds("never"));
        assert_eq!(deposits.idle_bytes(), 8192, "a live holder that maps nothing pins nothing");
        assert!(deposits.release_one());
        assert!(!deposits.release_one(), "a live executor maps the other");
        mapping.kill().unwrap();
        mapping.wait().unwrap();
        assert_eq!(deposits.idle_bytes(), 4096);
        assert!(deposits.release_one());
        assert!(deposits.grant("m", "mapped", &[]).is_none());
        idle.kill().unwrap();
        idle.wait().unwrap();
    }
}
