use super::*;
use std::sync::Barrier;

struct Host;
impl Hosting for Host {
    fn identify(&self, _: &Peer) -> Option<Identity> { None }
    fn free(&self) -> Option<u64> { Some(100) }
}

fn manager() -> Arc<Manager> {
    Arc::new(Manager {
        open: None, opening: Mutex::new(()), gpu_memory: 100, limit: None,
        hosting: Arc::new(Host), retired: AtomicBool::new(false),
        books: Mutex::new(Books { card: 100, ..Books::default() }),
        looking: Condvar::new(), events: Signal::new().unwrap(), born: Instant::now(),
    })
}

fn peak(manager: &Manager, key: &str, bytes: u64) {
    manager.restore(&[Peak::Request { key: key.into(), bytes }]).unwrap();
}

#[test]
fn descendants_can_start_on_first_use_without_admitting_siblings_over_them() {
    let m = manager();
    assert!(Manager::admit(&[&m], "parent", "parent", "shape", &[]).unwrap());
    assert!(!Manager::admit(&[&m], "child", "child", "shape", &[]).unwrap());
    let ancestors = [Ancestor { request: "parent".into(), pids: vec![1] }];
    assert!(Manager::admit(&[&m], "child", "child", "shape", &ancestors).unwrap());
    assert!(!Manager::admit(&[&m], "sibling", "sibling", "shape", &ancestors).unwrap());
    m.undispatch("child");
    peak(&m, "parent|shape", 90);
    peak(&m, "child|shape", 90);
    assert!(Manager::admit(&[&m], "child", "child", "shape", &ancestors).unwrap());
    assert_eq!(m.holdings().dispatched.len(), 2, "yielding never forgets the parent's identity");
}

#[test]
fn a_refused_group_reserves_none_of_its_gpus() {
    let a = manager();
    let b = manager();
    peak(&a, "child|shape", 70);
    peak(&b, "child|shape", 70);
    peak(&b, "busy|shape", 70);
    b.dispatch("busy", "busy", "shape").unwrap();
    assert!(!Manager::admit(&[&a, &b], "child", "child", "shape", &[]).unwrap());
    assert!(a.holdings().dispatched.is_empty());
    assert_eq!(b.holdings().dispatched, [("busy".into(), "busy|shape".into())]);
}

#[test]
fn competing_reversed_groups_make_one_atomic_reservation() {
    for _ in 0..20 {
        let a = manager();
        let b = manager();
        peak(&a, "child|shape", 70);
        peak(&b, "child|shape", 70);
        let go = Arc::new(Barrier::new(3));
        let threads: Vec<_> = [false, true].into_iter().map(|reverse| {
            let (a, b, go) = (a.clone(), b.clone(), go.clone());
            std::thread::spawn(move || {
                go.wait();
                let group = if reverse { [&*b, &*a] } else { [&*a, &*b] };
                Manager::admit(&group, if reverse { "two" } else { "one" }, "child", "shape", &[]).unwrap()
            })
        }).collect();
        go.wait();
        assert_eq!(threads.into_iter().map(|t| t.join().unwrap() as u32).sum::<u32>(), 1);
        assert_eq!(a.holdings().dispatched, b.holdings().dispatched);
    }
}

#[cfg(unix)]
fn executor(m: &Manager, pid: u32, bytes: u64) -> Arc<Table> {
    let fd = os::sealed_page(table::bytes(1) as u64).unwrap();
    let table = Arc::new(Table::create(os::map(&fd, table::bytes(1), true).unwrap(), 1, pid, 1).unwrap());
    assert!(table.grant(0, 1, bytes, None, false).unwrap());
    assert!(table.answer(0, 1, true).unwrap());
    let (stream, _) = std::os::unix::net::UnixStream::pair().unwrap();
    m.books.lock().executors.insert(1, Exec {
        pid, identity: Identity { executor: "parent".into(), domain: "test".into() },
        table: table.clone(), end: Arc::new(End::new(stream)), free: BTreeSet::new(),
        leased: [(0, Leased { phys: Physical { id: 1, len: bytes }, seq: 1, region: None, index: 0, landed: false, fill: None })].into(),
        waiting: VecDeque::new(), asked: BTreeMap::new(), progress: 0, seen: (0, 0), still: 0,
        stalled: false, reported: BTreeSet::new(), waits_on: BTreeSet::new(), regions: BTreeMap::new(), reclaiming: BTreeSet::new(),
    });
    table
}

#[cfg(unix)]
#[test]
fn a_yielded_ancestors_pins_still_count_beside_a_sibling() {
    let m = manager();
    let table = executor(&m, 1, 60);
    assert!(table.pin(0).unwrap());
    peak(&m, "child|shape", 30);
    peak(&m, "sibling|shape", 20);
    m.dispatch("parent", "parent", "shape").unwrap();
    m.dispatch("sibling", "sibling", "shape").unwrap();
    let ancestors = [Ancestor { request: "parent".into(), pids: vec![1, 1] }];
    assert!(!Manager::admit(&[&m], "child", "child", "shape", &ancestors).unwrap());
    table.unpin(0).unwrap();
    assert!(Manager::admit(&[&m], "child", "child", "shape", &ancestors).unwrap());
}

#[cfg(unix)]
#[test]
fn final_reply_records_peaks_before_a_delayed_lease_measurement() {
    let m = manager();
    let _table = executor(&m, 1, 1);
    m.dispatch("call", "package", "shape").unwrap();
    let stats = crate::Stats { request: "call".into(), peak: 70, compute: 30, host: Some(80), ..Default::default() };
    m.complete("call", Some(&stats));
    // The report sent before Invoke completed reaches its other socket afterward.
    m.measured(1, Measured::Request { request: "call".into(), working: 70, compute: 30, host: 80 });
    assert_eq!(m.host_peak("package", "shape"), Some(80));
    assert!(m.peaks().contains(&Peak::Request { key: "package|shape".into(), bytes: 70 }));
    assert!(m.peaks().contains(&Peak::Compute { key: "package|shape".into(), bytes: 30 }));
    assert!(m.holdings().dispatched.is_empty(), "no completion tombstones accumulate");
    m.dispatch("other", "package", "other").unwrap();
    m.complete("other", Some(&stats));
    assert_eq!(m.host_peak("package", "other"), None, "one request cannot train another's key");
}

#[test]
fn restored_peaks_merge_maxima_in_either_gpu_order() {
    for reverse in [false, true] {
        let m = manager();
        let mut peaks = vec![
            Peak::Request { key: "package|shape".into(), bytes: 70 },
            Peak::Request { key: "package|shape".into(), bytes: 30 },
            Peak::Compute { key: "package|shape".into(), bytes: 20 },
            Peak::Compute { key: "package|shape".into(), bytes: 50 },
        ];
        if reverse { peaks.reverse(); }
        m.restore(&peaks).unwrap();
        assert!(m.peaks().contains(&Peak::Request { key: "package|shape".into(), bytes: 70 }));
        assert!(m.peaks().contains(&Peak::Compute { key: "package|shape".into(), bytes: 50 }));
    }
}
