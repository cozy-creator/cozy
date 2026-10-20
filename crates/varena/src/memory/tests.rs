//! CPU ownership controls, not CUDA execution. Handles below are inert metadata: the tests
//! enter production failure helpers before any driver use. Actual post-enqueue GPU proof is
//! separate. Retained objects deliberately live until this test process exits.

use super::*;

#[cfg(all(feature = "faults", target_os = "linux"))]
mod gpu;

fn error(what: &str) -> Error {
    Error::Io(what.into())
}

fn weights_with_host(with_host: bool) -> (Arc<Weights>, i32) {
    let file = std::fs::File::open("/dev/null").unwrap();
    let fd = file.as_raw_fd();
    let source = Source::files(vec![(tensorfs::ids::ObjectRef::of(&[]), file)], false, 0);
    let layout = Layout {
        parts: vec![],
        regions: vec![crate::layout::Region {
            offset: 0,
            nbytes: 0,
            span: REGION_ALIGN,
            items: vec![],
        }],
        nbytes: REGION_ALIGN,
        digest: "0".repeat(64),
        names: vec![vec![]],
    };
    let host = with_host.then(|| HostMem::create("ownership-control", &layout).unwrap());
    (
        Arc::new(Weights {
            name: "ownership".into(),
            layout,
            parts: vec![],
            source,
            host,
        }),
        fd,
    )
}

fn gpu_with_host(with_host: bool) -> (Arc<Gpu>, Weak<Weights>, i32) {
    let (w, fd) = weights_with_host(with_host);
    let weak = Arc::downgrade(&w);
    let z = Counters::default();
    let set = Set {
        w,
        pending: vec![],
        base: 0,
        regs: vec![Reg {
            home: true,
            map: Some(Mapping {
                pins: 1,
                ..Default::default()
            }),
        }],
        step: z,
        last: z,
        total: z,
        used: 0,
        warm: true,
        busy: false,
        shared: false,
    };
    let inner = Arc::new(Inner {
        ordinal: 0,
        ctx: std::ptr::null_mut(),
        pci: String::new(),
        ceiling: 0.0,
        prop: AllocationProp::device(0, 0),
        shared: None,
        chunk: 64 << 20,
        events: Arc::new(Events {
            ctx: std::ptr::null_mut(),
            free: Mutex::default(),
        }),
        copy: std::ptr::null_mut(),
        readers: Readers::new(1),
        tally: IoTally::default(),
        staging: Staging::new(vec![]),
        jobs: Default::default(),
        st: Mutex::new(State {
            sets: vec![Some(set)],
            ..Default::default()
        }),
        owner: Mutex::default(),
        custody: Mutex::default(),
    });
    let gpu = Arc::new(Gpu {
        inner: inner.clone(),
        mover: None,
    });
    *inner.owner.lock().unwrap() = Arc::downgrade(&gpu);
    (gpu, weak, fd)
}

struct Fixture {
    gpu: Arc<Gpu>,
    ws: Option<Arc<WeightSet>>,
    fd: i32,
}

impl Fixture {
    fn new() -> Self {
        Self::with_host(false)
    }

    fn with_host(with_host: bool) -> Self {
        let (gpu, w, fd) = gpu_with_host(with_host);
        let ws = Arc::new(WeightSet {
            gpu: gpu.clone(),
            id: 0,
            base: 0,
            weights: w.upgrade().unwrap(),
        });
        Self {
            gpu,
            ws: Some(ws),
            fd,
        }
    }

    fn ws(&self) -> &Arc<WeightSet> {
        self.ws.as_ref().unwrap()
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        // Synthetic VA/stream metadata must never enter successful driver teardown.
        self.gpu
            .inner
            .poison(error("CPU ownership fixture finished"));
        self.gpu.inner.readers.stop();
    }
}

fn fd_is_open(fd: i32) -> bool {
    // SAFETY: F_GETFD only queries this process's descriptor table.
    (unsafe { libc::fcntl(fd, libc::F_GETFD) }) >= 0
}

fn buffer() -> Buf {
    Buf {
        ptr: std::ptr::null_mut(),
        last: None,
    }
}

#[test]
fn unknown_fill_retains_sources_and_refuses_bind_prefetch_and_eviction() {
    let mut f = Fixture::new();
    let g = &f.gpu.inner;
    let fill = Arc::new(Fill {
        state: Mutex::new(None),
        cv: Condvar::new(),
    });
    fill.set(g.failed_fill(error("copy submitted"), Err(error("drain failed"))));
    {
        let mut st = g.lock();
        let m = st.reg(0, 0).unwrap().map.as_mut().unwrap();
        m.pins = 0;
        m.fill = Some(fill.clone());
        assert!(!m.dead());
        assert!(!m.settled());
        assert!(!m.quiet().unwrap());
    }
    assert!(fill.wait().is_err());
    assert!(f.ws().bind(0, 0).is_err());
    assert!(f.ws().prefetch(0).is_err());
    assert!(f.ws().evict(None).is_err());
    let weights = Arc::downgrade(&f.ws().weights);
    drop(f.ws.take());
    assert!(weights.upgrade().is_some());
    assert!(g.lock().sets[0].is_some());
    assert!(fd_is_open(f.fd));
}

#[test]
fn ordinary_copy_error_with_confirmed_drain_stays_reclaimable() {
    let f = Fixture::new();
    let g = &f.gpu.inner;
    let fill = Arc::new(Fill {
        state: Mutex::new(None),
        cv: Condvar::new(),
    });
    fill.set(g.failed_fill(error("read failed"), Ok(())));
    let mut m = Mapping {
        fill: Some(fill),
        ..Default::default()
    };
    assert!(m.dead());
    assert!(m.settled());
    assert!(m.quiet().unwrap());
    assert!(g.check().is_ok());
}

#[test]
fn a_successful_copy_drain_cannot_cover_an_unproven_consumer_dependency() {
    let f = Fixture::new();
    let g = &f.gpu.inner;
    let release = Arc::new(Ev {
        raw: std::ptr::null_mut(),
        timing: false,
        pool: g.events.clone(),
    });
    let weak = Arc::downgrade(&release);
    let error = g.failed_dependencies(
        error("wait enqueue failed"),
        std::slice::from_ref(&release),
        Err(error("consumer fence failed")),
    );
    let filled = g.failed_fill(error, Ok(()));
    let fill = Arc::new(Fill {
        state: Mutex::new(Some(filled)),
        cv: Condvar::new(),
    });
    let mut map = Mapping {
        fill: Some(fill),
        ..Default::default()
    };
    assert!(!map.dead());
    assert!(!map.settled());
    assert!(!map.quiet().unwrap());
    g.retain_waits(std::slice::from_ref(&release));
    assert_eq!(g.custody.lock().unwrap().releases.len(), 1);
    drop(release);
    assert!(weak.upgrade().is_some());
}

#[test]
fn failed_wait_with_independent_consumer_completion_keeps_normal_recovery() {
    let f = Fixture::new();
    let g = &f.gpu.inner;
    let error = g.failed_dependencies(error("wait enqueue failed"), &[], Ok(()));
    let filled = g.failed_fill(error, Ok(()));
    assert!(matches!(filled, Filled::Failed(_)));
    assert!(g.check().is_ok());
    assert!(g.custody.lock().unwrap().releases.is_empty());
}

#[test]
fn source_completion_snapshot_survives_streaming_slot_reassignment() {
    let mut f = Fixture::new();
    let ready = Arc::new(Ev {
        raw: std::ptr::null_mut(),
        timing: false,
        pool: f.gpu.inner.events.clone(),
    });
    let fill = Arc::new(Fill {
        state: Mutex::new(Some(Filled::Ready(ready))),
        cv: Condvar::new(),
    });
    {
        let mut st = f.gpu.inner.lock();
        st.set(0).track_fill(fill.clone());
        st.slots.push(Some(Slot {
            m: Mapping {
                fill: Some(fill.clone()),
                ..Default::default()
            },
            holds: Some((0, 0)),
            pending: false,
            used: 0,
        }));
        // Simulate only the metadata handoff; no fake successful CUDA mapping or fence.
        st.slot(0).holds = Some((1, 0));
        st.slot(0).m.fill = None;
        let (closing, _) = st.close_fences(0).unwrap();
        assert_eq!(closing.len(), 1);
        assert!(Arc::ptr_eq(&closing[0], &fill));
    }
    let source = Arc::downgrade(&f.ws().weights);
    f.gpu
        .inner
        .poison(error("historical producer completion unknown"));
    drop(f.ws.take());
    assert!(source.upgrade().is_some());
    assert!(fd_is_open(f.fd));
    assert_eq!(
        f.gpu.inner.lock().sets[0].as_ref().unwrap().pending.len(),
        1
    );
}

#[test]
fn confirmed_completed_fill_history_retires_during_existing_activity() {
    let f = Fixture::new();
    let fill = Arc::new(Fill {
        state: Mutex::new(Some(Filled::Failed(error("confirmed drained failure")))),
        cv: Condvar::new(),
    });
    let weak = Arc::downgrade(&fill);
    let mut st = f.gpu.inner.lock();
    st.set(0).track_fill(fill);
    f.gpu.inner.reap_fills(&mut st, 0).unwrap();
    assert!(st.set(0).pending.is_empty());
    assert!(weak.upgrade().is_none());
}

#[cfg(feature = "faults")]
#[test]
fn failed_completion_query_retains_pending_source_and_event() {
    let f = Fixture::new();
    let ready = Arc::new(Ev {
        raw: std::ptr::null_mut(),
        timing: false,
        pool: f.gpu.inner.events.clone(),
    });
    let weak = Arc::downgrade(&ready);
    let fill = Arc::new(Fill {
        state: Mutex::new(Some(Filled::Ready(ready))),
        cv: Condvar::new(),
    });
    let mut st = f.gpu.inner.lock();
    st.set(0).track_fill(fill);
    crate::faults::arm("fill-query", 1);
    assert!(f.gpu.inner.reap_fills(&mut st, 0).is_err());
    assert_eq!(st.set(0).pending.len(), 1);
    assert!(weak.upgrade().is_some());
    assert!(fd_is_open(f.fd));
}

#[test]
fn unproven_staging_buffer_never_reenters_pool_and_readers_finish_with_error() {
    let f = Fixture::new();
    let (tx, rx) = mpsc::channel();
    let waiter = f.gpu.clone();
    let thread = std::thread::spawn(move || tx.send(waiter.inner.staging.take().is_err()).unwrap());
    let previous = Arc::new(Ev {
        raw: std::ptr::null_mut(),
        timing: false,
        pool: f.gpu.inner.events.clone(),
    });
    let retained = Arc::downgrade(&previous);
    let res = f.gpu.inner.finish_piece(
        buffer(),
        Some(previous),
        Err(error("prior event synchronize failed")),
        Some(Err(error("drain failed"))),
    );
    assert!(res.is_err());
    assert!(rx.recv_timeout(std::time::Duration::from_secs(2)).unwrap());
    thread.join().unwrap();
    let b = f.gpu.inner.staging.buffers.lock().unwrap();
    assert!(b.free.is_empty());
    assert_eq!(b.held.len(), 1);
    assert!(b.error.is_some());
    assert!(b.held[0].last.is_some());
    assert!(retained.upgrade().is_some());
}

#[test]
fn unproven_post_enqueue_buffer_without_an_event_stays_unavailable() {
    let f = Fixture::new();
    assert!(f
        .gpu
        .inner
        .finish_piece(
            buffer(),
            None,
            Err(error("post-enqueue record failed")),
            Some(Err(error("drain failed")))
        )
        .is_err());
    let b = f.gpu.inner.staging.buffers.lock().unwrap();
    assert!(b.free.is_empty());
    assert_eq!(b.held.len(), 1);
    assert!(b.held[0].last.is_none());
}

#[test]
fn staging_error_with_confirmed_drain_or_no_submission_returns_its_buffer() {
    for drained in [Some(Ok(())), None] {
        let f = Fixture::new();
        assert!(f
            .gpu
            .inner
            .finish_piece(buffer(), None, Err(error("copy or read failed")), drained)
            .is_err());
        assert!(f.gpu.inner.check().is_ok());
        assert!(f.gpu.inner.staging.take().is_ok());
    }
}

#[test]
fn failed_final_pin_retains_actual_weight_set_gpu_and_source_once() {
    let mut f = Fixture::new();
    let ws = f.ws.take().unwrap();
    let weak_ws = Arc::downgrade(&ws);
    let weak_gpu = Arc::downgrade(&f.gpu);
    for _ in 0..3 {
        ws.finish_pin(
            At::Home,
            0,
            vec![],
            Some(error("release record failed")),
            || Err(error("consumer drain failed")),
        );
    }
    assert_eq!(f.gpu.inner.custody.lock().unwrap().pins.len(), 1);
    assert_eq!(
        f.gpu
            .inner
            .lock()
            .reg(0, 0)
            .unwrap()
            .map
            .as_ref()
            .unwrap()
            .pins,
        1
    );
    let fd = f.fd;
    drop(ws);
    drop(f);
    assert!(weak_ws.upgrade().is_some());
    assert!(weak_gpu.upgrade().is_some());
    assert!(fd_is_open(fd));
}

#[test]
fn failed_release_record_with_confirmed_consumer_drain_releases_the_pin() {
    let f = Fixture::new();
    f.ws().finish_pin(
        At::Home,
        0,
        vec![],
        Some(error("release record failed")),
        || Ok(()),
    );
    assert_eq!(
        f.gpu
            .inner
            .lock()
            .reg(0, 0)
            .unwrap()
            .map
            .as_ref()
            .unwrap()
            .pins,
        0
    );
    assert!(f.gpu.inner.check().is_ok());
    assert!(f.gpu.inner.custody.lock().unwrap().pins.is_empty());
}

#[test]
fn a_retained_pin_refuses_even_a_same_stream_wait_after_poison() {
    let f = Fixture::new();
    let ready = Arc::new(Ev {
        raw: std::ptr::null_mut(),
        timing: false,
        pool: f.gpu.inner.events.clone(),
    });
    let pin = Pinned {
        ws: f.ws().clone(),
        at: At::Home,
        region: 0,
        ptr: 0,
        nbytes: 0,
        fresh: false,
        writable: true,
        stream: ConsumerStream::current(7),
        consumers: Mutex::new(vec![ConsumerStream::current(7)]),
        ready,
    };
    f.gpu.inner.poison(error("completion unknown"));
    assert!(pin.wait_on(7).is_err());
    drop(pin);
}

#[cfg(feature = "faults")]
#[test]
fn final_pinned_drop_keeps_the_last_weight_set_arc_after_context_and_drain_failure() {
    let mut f = Fixture::new();
    let ws = f.ws.take().unwrap();
    let weak = Arc::downgrade(&ws);
    let ready = Arc::new(Ev {
        raw: std::ptr::null_mut(),
        timing: false,
        pool: f.gpu.inner.events.clone(),
    });
    let pin = Pinned {
        ws,
        at: At::Home,
        region: 0,
        ptr: 0,
        nbytes: 0,
        fresh: false,
        writable: true,
        stream: ConsumerStream::current(0),
        consumers: Mutex::new(vec![ConsumerStream::current(0)]),
        ready,
    };
    crate::faults::arm("pin-release-context", 1);
    crate::faults::arm("pin-release-drain", 1);
    drop(pin);
    assert!(weak.upgrade().is_some());
    assert_eq!(f.gpu.inner.custody.lock().unwrap().pins.len(), 1);
    assert!(fd_is_open(f.fd));
}

#[cfg(feature = "faults")]
#[test]
fn failed_weight_set_context_entry_retains_its_set_and_source() {
    let mut f = Fixture::new();
    crate::faults::arm("set-drop-context", 1);
    let ws = f.ws.take().unwrap();
    let source = Arc::downgrade(&ws.weights);
    drop(ws);
    assert!(f.gpu.inner.check().is_err());
    assert!(f.gpu.inner.lock().sets[0].is_some());
    assert!(source.upgrade().is_some());
    assert!(fd_is_open(f.fd));
}

#[cfg(feature = "faults")]
#[test]
fn failed_final_driver_entry_retains_actual_mapping_source_and_queued_job() {
    let (gpu, source, fd) = gpu_with_host(false);
    let (w, job_fd) = weights_with_host(false);
    let job_source = Arc::downgrade(&w);
    let fill = Arc::new(Fill {
        state: Mutex::new(None),
        cv: Condvar::new(),
    });
    gpu.inner.jobs.0.lock().unwrap().q.push_back(Job {
        set: 0,
        w,
        region: 0,
        dst: 0,
        waits: vec![],
        fill,
    });
    crate::faults::arm("inner-drop-driver", 1);
    drop(gpu);
    assert!(source.upgrade().is_some());
    assert!(job_source.upgrade().is_some());
    assert!(fd_is_open(fd));
    assert!(fd_is_open(job_fd));
}

#[cfg(all(feature = "faults", target_os = "linux"))]
#[test]
fn failed_unregister_retains_the_actual_host_mapping_and_source() {
    let mut f = Fixture::with_host(true);
    let ws = f.ws.take().unwrap();
    let host = ws.weights.host.as_ref().unwrap();
    let (host_fd, base) = (host.fd(), host.base());
    let source = Arc::downgrade(&ws.weights);
    crate::faults::arm("set-unregister", 1);
    // The production post-fence helper, with failure injected before any CUDA call. The
    // host memfd/mapping are real CPU objects, rather than simulated owner counters.
    assert!(!ws.unregister_host());
    drop(ws);
    drop(f);
    assert!(source.upgrade().is_some());
    assert!(fd_is_open(host_fd));
    let mut resident = 0u8;
    // SAFETY: mincore only queries residency; EFAULT would expose a lost mapping.
    assert_eq!(
        unsafe { libc::mincore(base.cast(), PART_ALIGN as usize, &mut resident) },
        0
    );
}

fn consumer_pin(f: &Fixture) -> Pinned {
    Pinned {
        ws: f.ws().clone(),
        at: At::Home,
        region: 0,
        ptr: 0,
        nbytes: 0,
        fresh: false,
        writable: true,
        stream: ConsumerStream::current(7),
        consumers: Mutex::new(vec![ConsumerStream::current(7)]),
        ready: inert_event(f),
    }
}

fn inert_event(f: &Fixture) -> EvRef {
    Arc::new(Ev {
        raw: std::ptr::null_mut(),
        timing: false,
        pool: f.gpu.inner.events.clone(),
    })
}

#[test]
fn completed_fill_still_admits_and_deduplicates_secondary_consumers() {
    let f = Fixture::new();
    let pin = consumer_pin(&f);
    for stream in [7, 8, 8, 9, 7, 9] {
        pin.admit_stream(
            stream,
            || Ok(true),
            || panic!("completed fill needs no wait"),
        )
        .unwrap();
    }
    assert_eq!(consumer_ids(&pin), [7, 8, 9]);
    drop(f); // Poison inert handles before the real Pinned destructor.
    drop(pin);
}

#[test]
fn unfinished_fill_admission_waits_before_returning_the_consumer() {
    let f = Fixture::new();
    let pin = consumer_pin(&f);
    let mut waited = false;
    pin.admit_stream(
        8,
        || Ok(false),
        || {
            waited = true;
            Ok(())
        },
    )
    .unwrap();
    assert!(waited);
    assert_eq!(consumer_ids(&pin), [7, 8]);
    drop(f);
    drop(pin);
}

#[test]
fn failed_readiness_query_or_wait_keeps_candidate_consumers_for_final_fencing() {
    let f = Fixture::new();
    let pin = consumer_pin(&f);
    assert!(pin
        .admit_stream(8, || Err(error("query unknown")), || panic!("query failed"))
        .is_err());
    assert!(pin
        .admit_stream(9, || Ok(false), || Err(error("wait enqueue unknown")))
        .is_err());
    assert_eq!(consumer_ids(&pin), [7, 8, 9]);
    f.gpu.inner.poison(error("completion unknown"));
    assert!(pin
        .admit_stream(10, || panic!("poisoned"), || panic!("poisoned"))
        .is_err());
    assert_eq!(consumer_ids(&pin), [7, 8, 9]);
    drop(f);
    drop(pin);
}

#[test]
fn all_consumer_release_events_transfer_before_exactly_one_pin_decrement() {
    let f = Fixture::new();
    let pin = consumer_pin(&f);
    for stream in [8, 9] {
        pin.admit_stream(stream, || Ok(true), || unreachable!())
            .unwrap();
    }
    let mut recorded = vec![];
    pin.finish_streams(
        |stream| {
            recorded.push(stream);
            Ok(inert_event(&f))
        },
        |_| panic!("all recorded"),
    );
    assert_eq!(recorded, [7, 8, 9]);
    let mut st = f.gpu.inner.lock();
    let map = st.reg(0, 0).unwrap().map.as_ref().unwrap();
    assert_eq!(map.pins, 0);
    assert_eq!(map.released.len(), 3);
    drop(st);
    assert!(f.gpu.inner.check().is_ok());
    drop(f);
    drop(pin);
}

#[test]
fn partial_records_and_mixed_drains_retain_actual_events_and_last_owners() {
    let mut f = Fixture::new();
    let pin = consumer_pin(&f);
    for stream in [8, 9] {
        pin.admit_stream(stream, || Ok(true), || unreachable!())
            .unwrap();
    }
    let event = inert_event(&f);
    let held_event = Arc::downgrade(&event);
    let mut recorded = vec![];
    let mut drained = vec![];
    pin.finish_streams(
        |stream| {
            recorded.push(stream);
            if stream == 7 {
                Ok(event.clone())
            } else {
                Err(error("record failed"))
            }
        },
        |stream| {
            drained.push(stream);
            if stream == 8 {
                Err(error("secondary consumer unknown"))
            } else {
                Ok(())
            }
        },
    );
    drop(event);
    assert_eq!(recorded, [7, 8]);
    assert_eq!(drained, [7, 8, 9]);
    assert!(held_event.upgrade().is_some());
    assert_eq!(f.gpu.inner.custody.lock().unwrap().releases.len(), 1);
    assert_eq!(f.gpu.inner.custody.lock().unwrap().pins.len(), 1);
    assert_eq!(
        f.gpu
            .inner
            .lock()
            .reg(0, 0)
            .unwrap()
            .map
            .as_ref()
            .unwrap()
            .pins,
        1
    );
    let ws = f.ws.take().unwrap();
    let held_ws = Arc::downgrade(&ws);
    let held_gpu = Arc::downgrade(&f.gpu);
    let held_source = Arc::downgrade(&ws.weights);
    let fd = f.fd;
    drop(ws);
    drop(pin); // The actual destructor observes poison and keeps its owner, without CUDA.
    drop(f);
    assert!(held_ws.upgrade().is_some() && held_gpu.upgrade().is_some());
    assert!(held_source.upgrade().is_some() && held_event.upgrade().is_some());
    assert!(fd_is_open(fd));
}

#[test]
fn first_record_failure_still_checks_every_consumer_drain() {
    let f = Fixture::new();
    let pin = consumer_pin(&f);
    pin.admit_stream(8, || Ok(true), || unreachable!()).unwrap();
    pin.admit_stream(9, || Ok(true), || unreachable!()).unwrap();
    let mut drained = vec![];
    pin.finish_streams(
        |_| Err(error("context or first record failed")),
        |stream| {
            drained.push(stream);
            if stream != 8 {
                Err(error("drain failed"))
            } else {
                Ok(())
            }
        },
    );
    assert_eq!(drained, [7, 8, 9]);
    assert_eq!(
        f.gpu
            .inner
            .lock()
            .reg(0, 0)
            .unwrap()
            .map
            .as_ref()
            .unwrap()
            .pins,
        1
    );
    assert!(f.gpu.inner.custody.lock().unwrap().releases.is_empty());
    assert_eq!(f.gpu.inner.custody.lock().unwrap().pins.len(), 1);
    drop(f);
    drop(pin);
}

#[test]
fn partial_record_failure_with_all_drains_confirmed_releases_once_without_poison() {
    let f = Fixture::new();
    let pin = consumer_pin(&f);
    pin.admit_stream(8, || Ok(true), || unreachable!()).unwrap();
    let event = inert_event(&f);
    let held_event = Arc::downgrade(&event);
    let mut drained = vec![];
    pin.finish_streams(
        |stream| {
            if stream == 7 {
                Ok(event.clone())
            } else {
                Err(error("secondary record failed"))
            }
        },
        |stream| {
            drained.push(stream);
            Ok(())
        },
    );
    drop(event);
    assert_eq!(drained, [7, 8]);
    assert!(held_event.upgrade().is_none());
    assert!(f.gpu.inner.check().is_ok());
    assert!(f.gpu.inner.custody.lock().unwrap().pins.is_empty());
    assert_eq!(
        f.gpu
            .inner
            .lock()
            .reg(0, 0)
            .unwrap()
            .map
            .as_ref()
            .unwrap()
            .pins,
        0
    );
    drop(f);
    drop(pin);
}

fn consumer_ids(pin: &Pinned) -> Vec<u64> {
    pin.consumers
        .lock()
        .unwrap()
        .iter()
        .map(|stream| stream.raw)
        .collect()
}

#[test]
fn same_thread_default_consumer_is_fenced_normally() {
    let f = Fixture::new();
    let pin = consumer_pin(&f);
    pin.admit_stream(2, || Ok(true), || unreachable!()).unwrap();
    pin.admit_stream(2, || Ok(true), || unreachable!()).unwrap();
    assert_eq!(consumer_ids(&pin), [7, 2]);
    let mut recorded = vec![];
    pin.finish_streams(
        |stream| {
            recorded.push(stream);
            Ok(inert_event(&f))
        },
        |_| panic!("all records known"),
    );
    assert_eq!(recorded, [7, 2]);
    assert_eq!(
        f.gpu
            .inner
            .lock()
            .reg(0, 0)
            .unwrap()
            .map
            .as_ref()
            .unwrap()
            .pins,
        0
    );
    assert!(f.gpu.inner.check().is_ok());
    drop(f);
    drop(pin);
}

#[test]
fn cross_thread_default_consumer_keeps_owners_without_fencing_the_wrong_stream() {
    let mut f = Fixture::new();
    let mut pin = consumer_pin(&f);
    pin.stream = ConsumerStream::current(2);
    *pin.consumers.lock().unwrap() = vec![pin.stream];
    pin.admit_stream(2, || panic!("same bind identity"), || unreachable!())
        .unwrap();
    let pin = Arc::new(pin);
    let other = pin.clone();
    std::thread::spawn(move || {
        let mut waited = false;
        other
            .admit_stream(
                2,
                || Ok(false),
                || {
                    waited = true;
                    Ok(())
                },
            )
            .unwrap();
        assert!(
            waited,
            "per-thread default on another thread is not the bind identity"
        );
        other
            .admit_stream(2, || Ok(true), || unreachable!())
            .unwrap();
    })
    .join()
    .unwrap();
    assert_eq!(consumer_ids(&pin), [2, 2]);
    let event = inert_event(&f);
    let held_event = Arc::downgrade(&event);
    let mut recorded = vec![];
    let mut drained = vec![];
    pin.finish_streams(
        |stream| {
            recorded.push(stream);
            Ok(event.clone())
        },
        |stream| {
            drained.push(stream);
            Ok(())
        },
    );
    assert_eq!(
        recorded,
        [2],
        "record callback must not see the other thread's sentinel"
    );
    assert_eq!(
        drained,
        [2],
        "drain callback must not see the other thread's sentinel"
    );
    drop(event);
    let ws = f.ws.take().unwrap();
    let held_ws = Arc::downgrade(&ws);
    let held_gpu = Arc::downgrade(&f.gpu);
    let held_source = Arc::downgrade(&ws.weights);
    let fd = f.fd;
    assert_eq!(
        f.gpu
            .inner
            .lock()
            .reg(0, 0)
            .unwrap()
            .map
            .as_ref()
            .unwrap()
            .pins,
        1
    );
    drop(ws);
    drop(pin); // Real final Drop follows the existing poisoned owner-retention path.
    drop(f);
    assert!(held_ws.upgrade().is_some() && held_gpu.upgrade().is_some());
    assert!(held_source.upgrade().is_some() && held_event.upgrade().is_some());
    assert!(fd_is_open(fd));
}

#[test]
fn final_drop_on_another_thread_retains_per_thread_default_owners_without_cuda() {
    let mut f = Fixture::new();
    let ws = f.ws.take().unwrap();
    let held_ws = Arc::downgrade(&ws);
    let held_source = Arc::downgrade(&ws.weights);
    let held_gpu = Arc::downgrade(&f.gpu);
    let fd = f.fd;
    let pin = std::thread::spawn(move || {
        let stream = ConsumerStream::current(2);
        let ready = Arc::new(Ev {
            raw: std::ptr::null_mut(),
            timing: false,
            pool: ws.gpu.inner.events.clone(),
        });
        Pinned {
            ws,
            at: At::Home,
            region: 0,
            ptr: 0,
            nbytes: 0,
            fresh: false,
            writable: true,
            stream,
            consumers: Mutex::new(vec![stream]),
            ready,
        }
    })
    .join()
    .unwrap();
    drop(pin); // Identity refusal occurs before either real CUDA callback can be invoked.
    assert_eq!(f.gpu.inner.custody.lock().unwrap().pins.len(), 1);
    assert_eq!(
        f.gpu
            .inner
            .lock()
            .reg(0, 0)
            .unwrap()
            .map
            .as_ref()
            .unwrap()
            .pins,
        1
    );
    drop(f);
    assert!(held_ws.upgrade().is_some() && held_gpu.upgrade().is_some());
    assert!(held_source.upgrade().is_some() && fd_is_open(fd));
}
