//! Host layouts executors deposit (design memory-varena §5.4): sealed memfds varena filled in an
//! executor, kept by TensorD as opaque handles under (manifest, layout), so a later executor of
//! the same weights maps them from RAM (D9's refill path). A layout goes back to the host only
//! once no live executor it was handed to maps it (releasing one still mapped frees nothing),
//! the least recently granted first.
use platform::process::Birth;
use std::{collections::BTreeMap, fs::File, io, sync::Mutex, time::Instant};

#[derive(Default)]
pub struct Deposits {
    held: Mutex<BTreeMap<(String, String), Held>>,
    /// The pending layout and its admitted `call#rank`. This lock also serializes headroom
    /// checks with admission and the handoff to a deposited, possibly still filling layout.
    sealing: Mutex<BTreeMap<(String, String), String>>,
}

/// One kept layout: when it was last granted and every executor process it was handed to.
struct Held {
    file: File,
    used: Instant,
    holders: Vec<Birth>,
}

impl Held {
    fn mapped(&self) -> bool {
        self.holders.iter().any(|birth| birth.ended().is_ok_and(|ended| !ended))
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

    /// Claim a seal and check fresh headroom under one lock. Only one layout may be sealing or
    /// filling at a time, including within one call: its not-yet-resident bytes already own
    /// that surplus. A different layout can still run directly from the store without sealing.
    /// False also covers a deposit that arrived since the caller looked; it should ask `grant`
    /// again. The existing sealer keeps its permission without another headroom check.
    /// `admit` may inspect/release held deposits, but must not re-enter sealing or deposit.
    pub fn seal(&self, manifest: &str, layout: &str, who: &str, admit: impl FnOnce() -> bool) -> Result<bool, String> {
        let mut sealing = self.sealing.lock().unwrap();
        if let Some(other) = sealing.get(&key(manifest, layout)) {
            return if other == who { Ok(true) } else { Err(other.clone()) };
        }
        if !sealing.is_empty() {
            return Ok(false);
        }
        let held = self.held.lock().unwrap();
        if held.contains_key(&key(manifest, layout))
            || held.values().any(|h| reopen(&h.file).is_none_or(|f| state(&f).0))
        {
            return Ok(false);
        }
        drop(held);
        if !admit() {
            return Ok(false);
        }
        sealing.insert(key(manifest, layout), who.into());
        Ok(true)
    }

    /// Call `call` ended: a layout it was sealing and never deposited is nobody's to wait for.
    pub fn ended(&self, call: &str) {
        self.sealing.lock().unwrap().retain(|_, who| call_of(who) != call);
    }

    /// Whether a layout of this digest is held.
    pub fn holds(&self, layout: &str) -> bool {
        self.held.lock().unwrap().keys().any(|(_, held)| held == layout)
    }

    /// Bytes of the layouts no live executor maps: what releasing them gives back.
    pub fn idle_bytes(&self) -> u64 {
        self.held.lock().unwrap().values().filter(|h| !h.mapped()).map(|h| allocated(&h.file)).sum()
    }

    /// Lets the least recently granted layout no live executor maps go; whether one went.
    pub fn release_one(&self) -> bool {
        let mut held = self.held.lock().unwrap();
        let idle = held.iter().filter(|(_, h)| !h.mapped());
        let oldest = idle.min_by_key(|(_, h)| h.used).map(|(key, _)| key.clone());
        oldest.and_then(|key| held.remove(&key)).is_some()
    }
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
mod tests {
    use super::*;

    fn layout(bytes: u64) -> File {
        let layout = File::from(platform::shm::memfd_named("layout").unwrap());
        layout.set_len(bytes).unwrap();
        io::Write::write_all(&mut &layout, &vec![1u8; bytes as usize]).unwrap();
        platform::shm::seal(std::os::fd::AsFd::as_fd(&layout)).unwrap();
        layout
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
    /// reserves nothing.
    #[test]
    fn a_layout_has_one_sealer_until_its_deposit_or_its_call_ends() {
        let deposits = Deposits::default();
        assert_eq!(deposits.seal("sha256:m", "l", "run-z#0", || false), Ok(false));
        assert_eq!(deposits.seal("sha256:m", "l", "run-a#0", || true), Ok(true));
        assert_eq!(deposits.seal("m", "l", "run-a#0", || panic!("a retry keeps its admission")), Ok(true));
        assert_eq!(deposits.seal("m", "l", "run-a#1", || true), Err("run-a#0".into()), "rank 1 waits for rank 0");
        assert_eq!(deposits.seal("m", "l", "run-b#0", || false), Err("run-a#0".into()), "another call waits too");
        for who in ["run-a#0", "run-b#0"] {
            assert_eq!(deposits.seal("other-manifest", "other", who, || panic!("pending bytes already own the surplus")), Ok(false));
        }
        deposits.deposit("m", "l", layout(4096), &[]).unwrap();
        assert_eq!(deposits.seal("m", "l", "run-c#0", || panic!("the deposit already landed")), Ok(false));
        assert!(deposits.grant("m", "l", &[]).is_some());
        assert_eq!(deposits.seal("m", "other", "run-c#0", || true), Ok(true));
        deposits.ended("run-c");
        assert_eq!(deposits.seal("m", "other", "run-d#1", || true), Ok(true));
    }

    /// Distinct calls/checkpoints cannot both spend the same surplus before either fills it.
    #[test]
    fn concurrent_sealers_share_one_admission() {
        use std::sync::{atomic::{AtomicUsize, Ordering}, Barrier};
        let deposits = Deposits::default();
        let start = Barrier::new(2);
        let checks = AtomicUsize::new(0);
        let answers = std::thread::scope(|scope| {
            let launch = |manifest, who| {
                let (deposits, start, checks) = (&deposits, &start, &checks);
                scope.spawn(move || {
                    start.wait();
                    deposits.seal(manifest, manifest, who, || {
                        checks.fetch_add(1, Ordering::Relaxed);
                        true
                    }).unwrap()
                })
            };
            let a = launch("m-a", "call-a#0");
            let b = launch("m-b", "call-b#0");
            [a.join().unwrap(), b.join().unwrap()]
        });
        assert_eq!(answers.into_iter().filter(|admitted| *admitted).count(), 1);
        assert_eq!(checks.load(Ordering::Relaxed), 1);
    }

    /// The pending claim hands over to the real memfd's fill lock without opening a second
    /// admission; after its bytes land another checkpoint can use the remaining surplus.
    #[test]
    fn a_deposited_layout_keeps_its_admission_until_filled() {
        use tensorfs::io::{source::{IoTally, Source}, ItemSource, Span};
        let deposits = Deposits::default();
        let spans = [Span { what: "weight".into(), len: 4096, source: ItemSource::Inline(vec![7; 4096].into()) }];
        let layout = varena::layout::Layout::build(&spans, &[vec!["weight".into()]]).unwrap();
        assert_eq!(deposits.seal("m", "l", "run-a#0", || true), Ok(true));
        let (file, fill) = varena::host::open_sealed("admission", &layout, None, None).unwrap();
        deposits.deposit("m", "l", file.into(), &[]).unwrap();
        assert!(deposits.grant("m", "l", &[]).unwrap().1);
        assert_eq!(deposits.seal("m-other", "other", "run-a#0", || panic!("the first fill still owns the surplus")), Ok(false));
        fill.run(&layout, &Source::files(vec![], false, 0), 1, &IoTally::default()).unwrap();
        assert!(!deposits.grant("m", "l", &[]).unwrap().1);
        assert_eq!(deposits.seal("m-other", "other", "run-a#0", || {
            // Admission may inspect and release completed deposits without a lock cycle.
            assert!(deposits.idle_bytes() >= 4096);
            assert!(deposits.release_one());
            true
        }), Ok(true));
    }

    /// A layout a live executor maps stays (releasing it frees nothing); once every executor it was
    /// handed to has ended, it goes back, the least recently granted first.
    #[test]
    fn only_a_layout_no_live_executor_maps_goes_back() {
        let deposits = Deposits::default();
        let mut executor = std::process::Command::new("sleep").arg("600").spawn().unwrap();
        let live = Birth::of(executor.id()).unwrap();
        deposits.deposit("m", "old", layout(8192), &[]).unwrap();
        deposits.deposit("m", "mapped", layout(4096), std::slice::from_ref(&live)).unwrap();
        deposits.grant("m", "old", &[]).unwrap();
        assert!(deposits.holds("old") && !deposits.holds("never"));
        assert_eq!(deposits.idle_bytes(), 8192);
        assert!(deposits.release_one());
        assert!(!deposits.release_one(), "a live executor maps the other");
        assert_eq!(deposits.idle_bytes(), 0);
        executor.kill().unwrap();
        executor.wait().unwrap();
        assert_eq!(deposits.idle_bytes(), 4096);
        assert!(deposits.release_one());
        assert!(deposits.grant("m", "mapped", &[]).is_none());
    }
}
