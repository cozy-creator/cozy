//! Run only on an explicitly rented GPU. This is the real driver/read engine, not a simulator.
use std::{fs::OpenOptions, io::Write, sync::Arc};
use tensorfs::io::source::{Item, ItemSource, Source};
use tensorfs::{
    format::ObjectRef,
    io::{self, Copy, Dst, ObjectRange, ReadOpts, Src},
};
use varena::gpu::Gpu;

fn bytes(pin: &varena::gpu::Pin) -> Vec<u8> {
    let mut out = vec![0; pin.bytes as usize];
    pin.wait_on(0).expect("consumer stream");
    unsafe {
        io::read(
            &[Copy {
                src: Src::Device {
                    device: 0,
                    ptr: pin.ptr,
                    stream: 0,
                },
                dst: Dst::host(&mut out),
            }],
            &ReadOpts::default(),
        )
    }
    .expect("TensorFS device read");
    out
}

#[test]
#[ignore = "requires an owned rental; never run on the local GPU"]
fn pages_pin_evict_refill_and_survive_clean_owner_exit() {
    let data: Vec<u8> = (0..2 * 1024 * 1024).map(|n| (n % 251) as u8).collect();
    let path = std::env::temp_dir().join(format!("varena-slim-gpu-{}", std::process::id()));
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .read(true)
        .open(&path)
        .expect("source");
    file.write_all(&data).expect("write source");
    let object = ObjectRef::of(&data);
    let source = Arc::new(Source::files(vec![(object.clone(), file)], false, 0));
    let item = Item {
        offset: 0,
        len: data.len() as u64,
        source: ItemSource::Object(ObjectRange {
            obj: object,
            off: 0,
            len: data.len() as u64,
        }),
        part_end: true,
    };
    let gpu = Gpu::open(0).expect("real CUDA device");
    let reserved = gpu
        .reserve(2 * data.len() as u64)
        .expect("stable addresses");
    for (id, offset) in [(1, 0), (2, data.len() as u64)] {
        gpu.register(
            id,
            reserved.clone(),
            offset,
            data.len() as u64,
            source.clone(),
            vec![item.clone()],
            0,
            "same-layout/encoding/transform/rank/device".into(),
        )
        .expect("register page");
        gpu.fill(id).expect("real TensorFS disk fill");
    }
    let pin = gpu.bind(1, 0).expect("bind");
    let _busy_custody = gpu.export(1).expect("read-only custody while pinned");
    assert_eq!(bytes(&pin), data);
    assert!(!gpu.unmap(1).expect("pinned page cannot evict"));
    assert!(gpu.unmap(2).expect("other page independently evicts"));
    let fault = gpu.fill_with(2, |_, _, _| panic!("injected page transform panic"));
    assert!(matches!(fault, Err(varena::Error::Internal(_))));
    gpu.recover().expect("failed request recovers in process");
    gpu.fill_from_ram(2, &data)
        .expect("next request can refill");
    assert_eq!(bytes(&pin), data);
    drop(pin);
    io::drain(0).expect("device completion");
    // A release event may be newly recorded; its producer has completed above.
    gpu.poll().expect("deferred pin release");
    io::drain(0).expect("release completion");
    if !gpu.unmap(1).expect("finished pin") {
        // Deterministic completion via the real driver, not a timeout or polling loop.
        let (_, context) = io::cuda::primary_context(0).expect("context");
        let _guard = io::cuda::CtxGuard::enter(context).expect("enter");
        io::cuda::check("cuCtxSynchronize", unsafe {
            (io::cuda::driver().expect("driver").ctx_synchronize)()
        })
        .expect("sync");
        assert!(gpu.unmap(1).expect("release now complete"));
    }
    gpu.fill_from_ram(1, &data)
        .expect("real TensorFS RAM refill");
    assert_eq!(bytes(&gpu.bind(1, 0).expect("refill bind")), data);
    let (_, context) = io::cuda::primary_context(0).expect("context");
    let _guard = io::cuda::CtxGuard::enter(context).expect("enter");
    io::cuda::check("cuCtxSynchronize", unsafe {
        (io::cuda::driver().expect("driver").ctx_synchronize)()
    })
    .expect("sync");
    gpu.poll().expect("release before export");
    io::cuda::check("cuCtxSynchronize", unsafe {
        (io::cuda::driver().expect("driver").ctx_synchronize)()
    })
    .expect("sync");
    let (spans, container) = gpu.export(1).expect("read-only page custody");
    drop(reserved);
    drop(gpu); // Ledger-held container retains the physical pages across a clean owner exit.
    let next = Gpu::open(0).expect("replacement owner");
    let reserved = next
        .reserve(data.len() as u64)
        .expect("replacement addresses");
    next.register(
        1,
        reserved,
        0,
        data.len() as u64,
        source,
        vec![item],
        0,
        "same-layout/encoding/transform/rank/device".into(),
    )
    .expect("replacement metadata");
    assert!(next
        .import(
            1,
            "same-layout/encoding/transform/rank/device",
            &spans,
            container
        )
        .expect("import"));
    let retained = next.bind(1, 0).expect("imported bind");
    assert_eq!(bytes(&retained), data);
    assert!(!next.unregister(1).expect("close defers a live view"));
    assert_eq!(bytes(&retained), data);
    drop(retained);
    next.wait_pending()
        .expect("wait for real consumer completion");
    next.poll().expect("retire the last view");
    assert!(next.unregister(1).expect("closed model now has no pages"));
    std::fs::remove_file(path).expect("remove source");
    println!("real VMM: independent pages, pin protection, RAM refill and kept-page import passed");
}
