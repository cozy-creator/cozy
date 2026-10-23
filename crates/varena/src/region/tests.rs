//! The region frames end to end on a real agent and a real Memory: a stand-in manager on a socket
//! pair welcomes the agent with a table, reads the Region and RegionGone frames, and sends Reclaim,
//! which reaches the region through the agent's reader and the mechanism thread.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::os::unix::net::UnixStream;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use crate::arena::CHUNK;
use crate::lease::table::{self, Table};
use crate::lease::wire::{End, RegionFacts, ToAgent, ToManager};
use crate::region::{Kind, Reclaimable};
use crate::{os, GpuId, HostBackend, Memory};

const MB: u64 = 1 << 20;
const A: usize = 0xa00;

enum Heard {
    Region(RegionFacts),
    Gone(u64),
}

/// Welcomes the agent, then forwards every region frame it reads.
fn standin(stream: UnixStream) -> (std::sync::Arc<End>, mpsc::Receiver<Heard>) {
    let end = std::sync::Arc::new(End::new(stream));
    let (tx, rx) = mpsc::channel();
    let reader = end.clone();
    std::thread::spawn(move || {
        let Ok(Some((ToManager::Hello { birth, .. }, _))) = reader.recv::<ToManager>() else { return };
        let file = os::sealed_page(table::bytes(1) as u64).unwrap();
        let table = Table::create(os::map(&file, table::bytes(1), true).unwrap(), 1, std::process::id(), birth).unwrap();
        reader.send(&ToAgent::Welcome { epoch: table.attach() }, Some(os::lend(&file).unwrap())).unwrap();
        while let Ok(Some((m, _))) = reader.recv::<ToManager>() {
            let heard = match m {
                ToManager::Region(f) => Heard::Region(f),
                ToManager::RegionGone { id } => Heard::Gone(id),
                _ => continue,
            };
            if tx.send(heard).is_err() {
                return;
            }
        }
        drop(table);
    });
    (end, rx)
}

fn until(what: &str, f: impl Fn() -> bool) {
    let start = Instant::now();
    while !f() {
        assert!(start.elapsed() < Duration::from_secs(20), "never: {what}");
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// The next frame about `id`.
fn next(rx: &mpsc::Receiver<Heard>, id: u64) -> Heard {
    loop {
        match rx.recv_timeout(Duration::from_secs(20)).expect("no region frame") {
            Heard::Region(f) if f.id != id => {}
            Heard::Gone(g) if g != id => {}
            h => return h,
        }
    }
}

#[test]
fn a_managers_reclaim_reaches_the_region_and_its_facts_follow() {
    let card = 4 << 30;
    let uuid = format!("region-frames{}", std::process::id());
    let g = GpuId { uuid: uuid.clone(), model: "host".into(), memory: card };
    assert!(os::namespace(&format!("test{}-", std::process::id())));
    let _ledger = crate::Ledger::create(&g, None).unwrap();
    let mem = Memory::open(g, None, Box::new(HostBackend::new(card))).unwrap();
    let (theirs, ours) = UnixStream::pair().unwrap();
    let (manager, heard) = standin(theirs);
    assert_eq!(mem.lease_over(ours).unwrap(), None);
    mem.serve_standalone_torch().unwrap();

    let region = mem.region("kv", Kind::Kv, &[], false).unwrap();
    let id = region.id();
    let Heard::Region(created) = next(&heard, id) else { panic!("no Region frame at its create") };
    assert_eq!((created.kind, created.next_use), (Reclaimable::Discardable, u64::MAX));
    let p = {
        let _filling = region.allocating().unwrap();
        mem.torch_alloc(3 * MB, A, None).unwrap()
    };
    assert!(matches!(next(&heard, id), Heard::Region(_)), "no Region frame once filled");
    region.next_use(Some(4));
    let Heard::Region(advised) = next(&heard, id) else { panic!("no Region frame for its advice") };
    assert_eq!(advised.next_use, 4);

    // Pinned-then-unpinned ordering is tests/region.rs's; here the frame's path: reader, mechanism thread, region.
    let bytes = || mem.region_records().iter().find(|r| r.id == id).map_or(u64::MAX, |r| r.bytes);
    assert_eq!(bytes(), CHUNK);
    manager.send(&ToAgent::Reclaim { region: id, swap: false }, None).unwrap();
    until("the manager's reclaim discards the region", || bytes() == 0);
    assert!(matches!(next(&heard, id), Heard::Region(_)), "no Region frame after the reclaim");
    assert!(region.pin().unwrap().fresh());
    mem.torch_free(p, 3 * MB, A);
    drop(region);
    loop {
        if let Heard::Gone(_) = next(&heard, id) {
            break;
        }
    }
    drop(mem);
    os::remove_table(&uuid);
}
