//! The disk rung: a weight set larger than the host may hold streams through a streamed layout
//! (`HostMem` with a window), a sealed memfd of a few slots. Its machine stages each region an
//! adopter claims, reads ahead in order into slots nobody claims, and refills a slot only after
//! its region's last claim ends. Adopters wait per region exactly as for a layout still filling
//! (`HostMem::with_region`); none of them reads the store.

use std::os::fd::{AsFd, OwnedFd};
use std::sync::atomic::{fence, AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::Duration;

use crate::host::{self, HostMem};
use crate::io::{IoTally, Mode, Source};
use crate::layout::Layout;
use crate::{Error, Result};

/// The machine's side of a streamed layout: its writable mapping and the staging loop.
pub struct OpenWindow {
    host: HostMem,
}

/// Create `layout`'s streamed layout with `slots` slots (at least one; two keep a refill beside
/// a read) and seal it now; returns the memfd to hand out and the window to `serve`.
pub fn open_window(name: &str, layout: &Layout, slots: usize) -> Result<(OwnedFd, OpenWindow)> {
    let host = HostMem::create_window(name, layout, slots)?;
    let fd = host::reopen(host.raw_fd())?;
    // Future-write leaves our mapping writable.
    cozy_platform::shm::seal(fd.as_fd()).map_err(crate::os("seal window"))?;
    Ok((fd, OpenWindow { host }))
}

impl OpenWindow {
    /// The memfd's size: what the window charges once every slot has been used.
    pub fn bytes(&self) -> u64 {
        self.host.map_len() as u64
    }

    pub fn geometry(&self) -> host::Window {
        self.host.window().expect("a window")
    }

    /// Stage regions until `stop`: every claimed region first (lowest first), then the regions
    /// after the furthest claimed one, in order (wrapping once adopters have wrapped), into slots
    /// whose region nobody claims. Reads go through the verified path on `threads` threads
    /// (page-cache copies where resident, O_DIRECT for the rest). A region that cannot be read
    /// is marked failed: its readers get the error. Polls the lock table: 50 µs while staging,
    /// up to 2 ms idle.
    pub fn serve(&self, layout: &Layout, source: &Source, threads: usize, tally: &IoTally, stop: &AtomicBool) -> Result<()> {
        let h = &self.host;
        let w = h.window().expect("a window");
        let n = layout.regions.len() as u32;
        let mut held: Vec<Option<u32>> = vec![None; w.slots];
        let mut frontier: Option<u32> = None;
        let mut wrapped = false;
        let mut pause = Duration::from_micros(50);
        while !stop.load(Ordering::Acquire) {
            let claimed = h.claimed()?;
            if let (Some(f), Some(&low)) = (frontier, claimed.first()) {
                wrapped |= low < f && !claimed.contains(&f);
            }
            if let Some(&top) = claimed.last() {
                frontier = Some(top);
            }
            // What should be staged: the claimed regions, then the next ones in order.
            let mut want: Vec<u32> = claimed.iter().copied().filter(|&r| !h.failed(r)).collect();
            let start = frontier.map_or(0, |f| f + 1);
            let ahead = if wrapped { n } else { n.saturating_sub(start) };
            for i in 0..ahead {
                if want.len() >= w.slots {
                    break;
                }
                let r = (start + i) % n;
                if !want.contains(&r) && !h.failed(r) {
                    want.push(r);
                }
            }
            let mut staged_any = false;
            for &r in &want {
                if held.contains(&Some(r)) {
                    continue;
                }
                let urgent = claimed.contains(&r);
                let Some(s) = self.victim(&mut held, &want, &claimed, urgent) else { continue };
                self.stage(s, r, layout, source, threads, tally);
                held[s] = Some(r);
                staged_any = true;
            }
            if staged_any {
                pause = Duration::from_micros(50);
            } else {
                std::thread::sleep(pause);
                pause = (pause * 2).min(Duration::from_millis(2));
            }
        }
        Ok(())
    }

    /// A slot to stage into: an empty one, else one whose region is not wanted (for a claimed
    /// region, also one holding a region only read ahead). Its region's claim is checked after
    /// its word is cleared, so a reader that found it either holds a claim the check sees or saw
    /// the cleared word; a claimed slot is restored and skipped.
    fn victim(&self, held: &mut [Option<u32>], want: &[u32], claimed: &[u32], urgent: bool) -> Option<usize> {
        if let Some(s) = held.iter().position(Option::is_none) {
            return Some(s);
        }
        crate::faults::stall("window-victim"); // tests: claims arrive after the loop's snapshot
        let region = |s: usize| held[s].unwrap();
        let mut order: Vec<usize> = (0..held.len()).filter(|&s| !want.contains(&region(s))).collect();
        if urgent {
            let ahead: Vec<usize> = (0..held.len())
                .filter(|&s| want.contains(&region(s)) && !claimed.contains(&region(s)))
                .collect();
            order.extend(ahead);
        }
        for s in order {
            let q = held[s].unwrap();
            let word = self.host.slot_word(s);
            word.store((q as u64 + 1) << 32, Ordering::SeqCst);
            fence(Ordering::SeqCst);
            match self.host.claimed() {
                Ok(now) if !now.contains(&q) => {
                    held[s] = None;
                    return Some(s);
                }
                _ => word.store(host::staged(q), Ordering::SeqCst),
            }
        }
        None
    }

    /// Read region `r` into slot `s` (its word already cleared) and publish it.
    fn stage(&self, s: usize, r: u32, layout: &Layout, source: &Source, threads: usize, tally: &IoTally) {
        let h = &self.host;
        let w = h.window().expect("a window");
        let region = &layout.regions[r as usize];
        let base = h.base() as usize + s * w.slot_span as usize;
        let next = std::sync::atomic::AtomicUsize::new(0);
        let failed: Mutex<Option<Error>> = Mutex::new(None);
        std::thread::scope(|scope| {
            for _ in 0..threads.clamp(1, region.items.len().max(1)) {
                scope.spawn(|| loop {
                    let k = next.fetch_add(1, Ordering::Relaxed);
                    let Some(it) = region.items.get(k) else { return };
                    let dst = (base + (it.offset - region.offset) as usize) as *mut u8;
                    // SAFETY: each item's range (and its padding) lies inside slot `s`, which is
                    // this thread's alone until its word is published; items are disjoint.
                    if let Err(e) = unsafe { source.read_item(it, dst, Mode::Cache, tally) } {
                        failed.lock().unwrap().get_or_insert(e);
                        next.store(region.items.len(), Ordering::Relaxed);
                        return;
                    }
                });
            }
        });
        match failed.into_inner().unwrap() {
            None => h.slot_word(s).store(host::staged(r), Ordering::SeqCst),
            Some(_) => {
                h.slot_word(s).store(0, Ordering::SeqCst);
                h.mark_failed(r);
            }
        }
    }
}

impl Drop for OpenWindow {
    /// A window that stops serving fails every region: no adopter waits for it forever.
    fn drop(&mut self) {
        for r in 0..self.host.regions_len() as u32 {
            self.host.mark_failed(r);
        }
    }
}
