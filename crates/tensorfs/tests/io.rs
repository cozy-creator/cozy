//! The read engine on real chunk files: every tensor of a tiny HF SDXL through `read_into`
//! equals `read`; locations spanning chunk ends, read whole and in pieces; many destinations at
//! once from a writer's folder, cold (O_DIRECT) and warm; the disk probe.
#![cfg(unix)]

use std::collections::BTreeMap;
use std::fs;
#[cfg(target_os = "linux")]
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use tensorfs::format::{Body, Component, Dtype, Encoding, Header, Location, ObjectRef, Part, Tensor, CHUNKS_DIR};
use tensorfs::import::{Access, Request};
use tensorfs::io::{self, Copy, Dst, ReadOpts, ReadStats, Src, Target};
use tensorfs::store::{Checkpoint, Store};
use tensorfs::writer::{Data, PartData, Source, Writer};

const MIB: usize = 1 << 20;

struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Scratch {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let path = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("io-{name}-{}-{n}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        Scratch(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn noise(seed: u64, n: usize) -> Vec<u8> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut out = Vec::with_capacity(n + 8);
    while out.len() < n {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        out.extend_from_slice(&state.to_le_bytes());
    }
    out.truncate(n);
    out
}

#[test]
fn zero_source_changes_only_the_lent_host_gap() {
    let mut bytes = [0xa5; 37];
    let copy = Copy { src: Src::Zero, dst: Dst::host(&mut bytes[7..29]) };
    let stats = unsafe { io::read(&[copy], &ReadOpts::default()) }.unwrap();
    assert_eq!(stats.bytes, 22);
    assert_eq!(&bytes[..7], &[0xa5; 7]);
    assert_eq!(&bytes[7..29], &[0; 22]);
    assert_eq!(&bytes[29..], &[0xa5; 8]);
}

#[test]
#[ignore = "requires an allocated CUDA GPU; no mock driver"]
fn cuda_staging_pins_device_copies_and_chunk_spans_keep_exact_bytes() {
    let scratch = Scratch::new("cuda");
    let (ck, _, _) = spanning(&scratch.0.join("folder"));
    let opts = ReadOpts { streams: 4, ..ReadOpts::default() };
    for (component, key, role, location) in ranges(&ck) {
        let expected = stored(&ck, &location);
        let gpu = io::DeviceBuffer::new(0, expected.len()).unwrap();
        let target = Target { component: &component, key: &key, role: &role, offset: 0,
            dst: Dst::Device { device: 0, ptr: gpu.ptr(), len: gpu.len(), stream: 0 } };
        unsafe { ck.read_into(&[target], &opts) }.unwrap();
        let mut bytes = vec![0; expected.len()];
        unsafe { io::read(&[Copy { src: Src::Device { device: 0, ptr: gpu.ptr(), stream: 0 }, dst: Dst::host(&mut bytes) }], &opts) }.unwrap();
        assert_eq!(bytes, expected, "{component}/{key}#{role}");
    }
    // Unaligned bytes larger than the staging piece, then source reuse and an externally
    // registered page-aligned range. The engine must not unregister the caller's pin.
    let mut host = Buf::new(5 * MIB + 17, 0);
    let expected = noise(73, host.len);
    host.mem[host.at..host.at + host.len].copy_from_slice(&expected);
    let gpu = io::DeviceBuffer::new(0, host.len).unwrap();
    let other = io::DeviceBuffer::new(0, host.len).unwrap();
    let source = host.bytes().as_ptr();
    let copy = Copy { src: Src::Host(source), dst: Dst::Device { device: 0, ptr: gpu.ptr(), len: gpu.len(), stream: 0 } };
    for _ in 0..2 { unsafe { io::read(&[copy], &opts) }.unwrap(); }
    let _context = io::cuda::CtxGuard::enter(gpu.context()).unwrap();
    assert!(!unsafe { io::cuda::registered(source) }.unwrap(), "the completed reuse copy retained its temporary registration");
    let mut pin = unsafe { io::cuda::pin_region(source.cast_mut(), host.len, true) }.unwrap();
    assert!(unsafe { io::cuda::registered(source) }.unwrap());
    // A second guard borrows the first registration. Neither explicit release nor an
    // unreleased borrowed guard's Drop may unregister the live owner's mapping.
    let mut borrowed = unsafe { io::cuda::pin_region(source.cast_mut(), host.len, true) }.unwrap();
    borrowed.release().unwrap();
    borrowed.release().unwrap();
    assert!(unsafe { io::cuda::registered(source) }.unwrap(), "a borrowed release unregistered its owner's memory");
    drop(borrowed);
    assert!(unsafe { io::cuda::registered(source) }.unwrap());
    let borrowed = unsafe { io::cuda::pin_region(source.cast_mut(), host.len, true) }.unwrap();
    drop(borrowed);
    assert!(unsafe { io::cuda::registered(source) }.unwrap(), "a borrowed Drop unregistered its owner's memory");
    unsafe { io::read(&[copy], &opts) }.unwrap();
    assert!(unsafe { io::cuda::registered(source) }.unwrap());
    unsafe { io::read(&[Copy { src: Src::Device { device: 0, ptr: gpu.ptr(), stream: 0 }, dst: Dst::Device { device: 0, ptr: other.ptr(), len: other.len(), stream: 0 } }], &opts) }.unwrap();
    let mut bytes = vec![0; other.len()];
    unsafe { io::read(&[Copy { src: Src::Device { device: 0, ptr: other.ptr(), stream: 0 }, dst: Dst::host(&mut bytes) }], &opts) }.unwrap();
    assert_eq!(bytes, expected);
    let zero = Copy { src: Src::Zero, dst: Dst::Device { device: 0, ptr: other.ptr() + 17, len: 9, stream: 0 } };
    unsafe { io::read(&[zero], &opts) }.unwrap();
    unsafe { io::read(&[Copy { src: Src::Device { device: 0, ptr: other.ptr(), stream: 0 }, dst: Dst::host(&mut bytes) }], &opts) }.unwrap();
    assert_eq!(&bytes[..17], &expected[..17]);
    assert_eq!(&bytes[17..26], &[0; 9]);
    assert_eq!(&bytes[26..], &expected[26..]);
    let assert_no_current_context = || {
        let mut current = std::ptr::null_mut();
        io::cuda::check("cuCtxGetCurrent", unsafe { (io::cuda::driver().unwrap().ctx_get_current)(&mut current) }).unwrap();
        assert!(current.is_null(), "a HostPin release changed the caller's CUDA context");
    };
    {
        // Release enters the captured context and restores the caller's unbound state.
        // The buffers above still retain the primary context and all copies have drained.
        let _unbound = io::cuda::CtxGuard::enter(std::ptr::null_mut()).unwrap();
        pin.release().unwrap();
        pin.release().unwrap();
        assert_no_current_context();
    }
    assert!(!unsafe { io::cuda::registered(source) }.unwrap());
    drop(pin);
    let pin = unsafe { io::cuda::pin_region(source.cast_mut(), host.len, true) }.unwrap();
    assert!(unsafe { io::cuda::registered(source) }.unwrap());
    {
        let _unbound = io::cuda::CtxGuard::enter(std::ptr::null_mut()).unwrap();
        drop(pin);
        assert_no_current_context();
    }
    assert!(!unsafe { io::cuda::registered(source) }.unwrap());
    println!("HostPin: owned explicit/implicit unregister, borrowed explicit/implicit preservation, engine preservation and context restoration passed");
}

/// A destination `skew` bytes past a page boundary (0: O_DIRECT lands in place).
struct Buf {
    mem: Vec<u8>,
    at: usize,
    len: usize,
}

impl Buf {
    fn new(len: usize, skew: usize) -> Buf {
        let mem = vec![0xA5; len + 8192 + skew];
        let at = (4096 - mem.as_ptr() as usize % 4096) % 4096 + skew;
        Buf { mem, at, len }
    }
    fn bytes(&self) -> &[u8] {
        &self.mem[self.at..self.at + self.len]
    }
    fn dst(&mut self) -> Dst {
        Dst::host(&mut self.mem[self.at..self.at + self.len])
    }
}

/// Drops `ck`'s chunks from the page cache: the next read is cold.
fn evict(ck: &Checkpoint) {
    #[cfg(target_os = "linux")]
    for i in 0..ck.header().chunks.len() {
        let f = ck.chunk(i).unwrap();
        f.sync_all().unwrap();
        // SAFETY: an advisory call on our own descriptor.
        assert_eq!(unsafe { libc::posix_fadvise(f.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) }, 0);
    }
    #[cfg(not(target_os = "linux"))]
    let _ = ck;
}

/// Pages of `ck`'s chunk files not in the page cache (a loaded host evicts some).
#[cfg(target_os = "linux")]
fn evicted_pages(ck: &Checkpoint) -> u64 {
    let mut evicted = 0;
    for i in 0..ck.header().chunks.len() {
        let f = ck.chunk(i).unwrap();
        let len = f.metadata().unwrap().len() as usize;
        if len == 0 {
            continue;
        }
        // SAFETY: a read-only shared mapping of our own file, unmapped below.
        let p = unsafe { libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ, libc::MAP_SHARED, f.as_raw_fd(), 0) };
        assert_ne!(p, libc::MAP_FAILED);
        let mut v = vec![0u8; len.div_ceil(4096)];
        // SAFETY: the vector holds one byte per page of the mapping.
        assert_eq!(unsafe { libc::mincore(p, len, v.as_mut_ptr()) }, 0);
        // SAFETY: our mapping.
        unsafe { libc::munmap(p, len) };
        evicted += v.iter().filter(|b| **b & 1 == 0).count() as u64;
    }
    evicted
}

#[cfg(not(target_os = "linux"))]
fn evicted_pages(_: &Checkpoint) -> u64 {
    0
}

fn direct_ok(ck: &Checkpoint) -> bool {
    platform::durable::open_direct(&ck.chunk(0).unwrap()).unwrap().is_some()
}

/// Every stored range of `ck`: (component, key, role), a plain tensor's data with role `""`.
fn ranges(ck: &Checkpoint) -> Vec<(String, String, String, Location)> {
    let mut out = Vec::new();
    for (c, t) in ck.tensors() {
        for (role, at) in t.locations() {
            out.push((c.to_owned(), t.key.clone(), role.to_owned(), at));
        }
    }
    out
}

fn stored(ck: &Checkpoint, at: &Location) -> Vec<u8> {
    let mut want = vec![0; at.length as usize];
    ck.bytes_into(at, &mut want).unwrap();
    want
}

/// Reads every range of `ck` in one engine call into `skew`ed destinations and checks each
/// against `read` (plain) or the part's stored bytes.
fn read_everything(ck: &Checkpoint, skew: usize, opts: &ReadOpts) -> ReadStats {
    let all = ranges(ck);
    let mut bufs: Vec<Buf> = all.iter().map(|r| Buf::new(r.3.length as usize, skew)).collect();
    let targets: Vec<Target> = (all.iter().zip(&mut bufs))
        .map(|((c, k, role, _), b)| Target { component: c, key: k, role, offset: 0, dst: b.dst() })
        .collect();
    // SAFETY: each destination is its own live buffer.
    let stats = unsafe { ck.read_into(&targets, opts) }.unwrap();
    for ((c, k, role, at), b) in all.iter().zip(&bufs) {
        let want = if role.is_empty() { ck.read(c, k).unwrap() } else { stored(ck, at) };
        assert!(b.bytes() == want, "{c}/{k}#{role} skew {skew} {opts:?}");
    }
    let total: u64 = all.iter().map(|r| r.3.length).sum();
    assert_eq!((stats.bytes, stats.direct + stats.cached), (total, total), "{stats:?}");
    stats
}

fn sweep(ck: &Checkpoint, what: &str) {
    let small = ReadOpts { threads: 3, request: 64 << 10, ..ReadOpts::default() };
    let buffered = ReadOpts { direct: false, ..ReadOpts::default() };
    for opts in [ReadOpts::default(), small, buffered.clone()] {
        for skew in [0, 1, 4095] {
            evict(ck);
            let cold = read_everything(ck, skew, &opts);
            // The engine reads past the page cache, so a buffered read warms it.
            read_everything(ck, skew, &buffered);
            let warm = read_everything(ck, skew, &opts);
            // Counted after: a page the host evicts during the warm read stays out (O_DIRECT
            // does not fill the cache), so this covers every page that read found missing.
            let evicted = evicted_pages(ck);
            println!("{what} skew {skew} {opts:?}\n  cold {cold:?}\n  warm {warm:?}");
            if opts.direct && direct_ok(ck) {
                assert!(cold.direct > 0, "a cold read goes O_DIRECT: {cold:?}");
                // Only a piece with a page the host evicted since the warm-up goes direct.
                let allowed = evicted * opts.request.max(4096) as u64;
                assert!(warm.direct <= allowed, "a warm read stays in the page cache ({evicted} pages evicted): {warm:?}");
            } else {
                assert_eq!(cold.direct, 0);
            }
        }
    }
}

/// network: Hugging Face.
#[test]
fn every_tensor_of_a_tiny_sdxl_reads_through_the_engine_as_read_does() {
    let scratch = Scratch::new("tiny-sdxl");
    let store = Store::open(scratch.0.join("store")).unwrap();
    let sources = ["hf://hf-internal-testing/tiny-sdxl-pipe".to_string()];
    let access = Access::default();
    let request = Request {
        sources: &sources,
        converter: None,
        source_variant: None,
        access: &access,
        tmp: &store.root().join("tmp"),
        progress: &|_, _, _| {},
        stop: None,
    };
    let (header, _) = store.import(&request, Some("io-tiny-sdxl")).unwrap();
    let ck = store.checkpoint(&header).unwrap();
    assert!(ck.tensors().count() > 100);
    sweep(&ck, "tiny-sdxl");
}

/// A folder whose chunks have odd lengths, with locations that start mid-chunk (on 4 KiB) and run
/// on into the following chunks; `stream` is the chunks end to end.
fn spanning(dir: &Path) -> (Checkpoint, Vec<u8>, BTreeMap<String, u64>) {
    let chunks = [noise(1, 10_000), noise(2, 3 * 4096 + 17), noise(3, 70_001)];
    let starts = [0u64, 10_000, 10_000 + 3 * 4096 + 17];
    fs::create_dir_all(dir.join(CHUNKS_DIR)).unwrap();
    let refs: Vec<ObjectRef> = chunks.iter().map(|c| ObjectRef::of(c)).collect();
    for (c, r) in chunks.iter().zip(&refs) {
        fs::write(dir.join(CHUNKS_DIR).join(&r.sha256), c).unwrap();
    }
    let at = |chunk: u64, offset: u64, length: u64| Location { chunk, offset, length };
    let plain = |key: &str, l: Location| Tensor { key: key.into(), dtype: Dtype::U8, shape: vec![l.length], body: Body::Data(l) };
    let part = |role: &str, l: Location| Part { role: role.into(), dtype: Dtype::U8, shape: vec![l.length], chunk: l.chunk, offset: l.offset, length: l.length };
    let tensors = vec![
        plain("a", at(0, 4096, 20_000)),
        plain("b", at(1, 12_288, 100)),
        Tensor {
            key: "c".into(),
            dtype: Dtype::U8,
            shape: vec![38_041],
            body: Body::Encoded {
                encoding: Encoding { scheme: "acme-q".into(), params: Default::default() },
                parts: vec![part("w", at(2, 40_960, 29_041)), part("s", at(0, 8_192, 9_000))],
            },
        },
        plain("z", at(0, 0, 0)),
    ];
    let component = Component { name: "m".into(), tensors, ..Component::default() };
    let header = Header::new(refs, None, vec![component], Default::default()).unwrap();
    fs::write(dir.join("model.cozytensors"), header.to_cbor().unwrap()).unwrap();
    let ck = Checkpoint::folder(dir, None, None).unwrap();
    let mut first = BTreeMap::new();
    for (_, k, role, l) in ranges(&ck) {
        first.insert(format!("{k}#{role}"), starts[l.chunk as usize] + l.offset);
    }
    (ck, chunks.concat(), first)
}

#[test]
fn locations_spanning_chunk_ends_read_whole_and_in_pieces() {
    let scratch = Scratch::new("spanning");
    let (ck, stream, first) = spanning(&scratch.0);
    let a = ck.tensor("m", "a").unwrap();
    let Body::Data(a_at) = a.body else { unreachable!() };
    assert_eq!(ck.header().spans(&a_at).len(), 3, "a runs through all three chunks");
    sweep(&ck, "spanning");

    // Pieces of ranges, across chunk ends, all in one call.
    let asks = [("a", "", 1_900, 7_000), ("a", "", 17_000, 3_000), ("c", "s", 1_805, 6), ("c", "w", 0, 29_041), ("b", "", 3, 97), ("z", "", 0, 0)];
    for skew in [0, 7] {
        for opts in [ReadOpts::default(), ReadOpts { threads: 4, request: 4096, ..ReadOpts::default() }] {
            evict(&ck);
            let mut bufs: Vec<Buf> = asks.iter().map(|a| Buf::new(a.3, skew)).collect();
            let targets: Vec<Target> = (asks.iter().zip(&mut bufs))
                .map(|(&(key, role, offset, _), b)| Target { component: "m", key, role, offset, dst: b.dst() })
                .collect();
            // SAFETY: each destination is its own live buffer.
            unsafe { ck.read_into(&targets, &opts) }.unwrap();
            for (&(key, role, offset, len), b) in asks.iter().zip(&bufs) {
                let from = (first[&format!("{key}#{role}")] + offset) as usize;
                assert!(b.bytes() == &stream[from..from + len], "{key}#{role} {offset}+{len}");
            }
        }
    }

    // What a target cannot name.
    for (key, role, offset) in [("a", "", 19_995), ("c", "", 0), ("c", "x", 0), ("nope", "", 0)] {
        let mut buf = Buf::new(10, 0);
        let t = Target { component: "m", key, role, offset, dst: buf.dst() };
        // SAFETY: a live buffer.
        let e = unsafe { ck.read_into(&[t], &ReadOpts::default()) }.unwrap_err();
        assert_eq!(e.code(), "invalid", "{key}#{role}: {e}");
    }
    let device = Target { component: "m", key: "a", role: "", offset: 0, dst: Dst::Device { device: 0, ptr: 0, len: 10, stream: 0 } };
    // SAFETY: refused before any byte moves.
    let e = unsafe { ck.read_into(&[device], &ReadOpts::default()) }.unwrap_err();
    assert!(e.to_string().contains("CUDA"), "{e}");

    // Host memory to host memory: the refill path's source, no file.
    let mut out = Buf::new(stream.len() - 3, 1);
    let copy = Copy { src: Src::Host(stream[3..].as_ptr()), dst: out.dst() };
    // SAFETY: both live; the source outlives the call.
    let s = unsafe { io::read(&[copy], &ReadOpts { request: 4096, ..ReadOpts::default() }) }.unwrap();
    assert!(out.bytes() == &stream[3..] && s.bytes == stream.len() as u64 - 3);
}

#[test]
fn overflowing_file_ranges_and_request_grids_refuse_before_writing() {
    let scratch = Scratch::new("overflow");
    let path = scratch.0.join("bytes");
    fs::write(&path, b"real file bytes").unwrap();
    let file = fs::File::open(path).unwrap();
    for (offset, opts) in [(u64::MAX, ReadOpts::default()), (0, ReadOpts { request: usize::MAX, ..ReadOpts::default() })] {
        let mut out = [0xa5];
        let copy = Copy { src: Src::File(&file, offset), dst: Dst::host(&mut out) };
        // SAFETY: a live non-overlapping destination; file offsets and options are validated.
        let error = unsafe { io::read(&[copy], &opts) }.unwrap_err();
        assert_eq!(error.source.kind(), std::io::ErrorKind::InvalidInput);
        assert_eq!(out, [0xa5]);
    }
}

/// A writer's folder: `n` small tensors of varied sizes, an encoded tensor, and one standalone
/// tensor past the 64 MiB grid (two chunks).
fn written(dir: &Path, n: usize) -> Checkpoint {
    let mut w = Writer::folder(dir, "", None, false).unwrap();
    let mut c = w.component("unet", "diffusers.UNet2DConditionModel", Default::default(), None).unwrap();
    for i in 0..n {
        let bytes = noise(i as u64, 1 + (i * 7919) % (300 << 10));
        c.put(&format!("blocks.{}.w{i}", i / 16), &Dtype::U8, &[bytes.len() as u64], Data::Plain(Source::Read(&mut &bytes[..]))).unwrap();
    }
    let (w8, scale) = (noise(7, 3 * MIB + 5), noise(8, 4096 * 3));
    let (mut w8_bytes, mut scale_bytes) = (&w8[..], &scale[..]);
    let parts = vec![
        PartData { role: "weight".into(), dtype: Dtype::U8, shape: vec![w8.len() as u64], data: Source::Read(&mut w8_bytes) },
        PartData { role: "scale".into(), dtype: Dtype::U8, shape: vec![scale.len() as u64], data: Source::Read(&mut scale_bytes) },
    ];
    let encoding = Encoding { scheme: "acme-q".into(), params: Default::default() };
    c.put("quantized", &Dtype::U8, &[1], Data::Encoded { encoding, parts }).unwrap();
    let big = noise(9, 64 * MIB + 12_345);
    c.put("big", &Dtype::U8, &[big.len() as u64], Data::Plain(Source::Read(&mut &big[..]))).unwrap();
    w.commit().unwrap();
    Checkpoint::folder(dir, None, None).unwrap()
}

#[test]
fn many_destinations_at_once_from_a_writers_folder() {
    let scratch = Scratch::new("many");
    let ck = written(&scratch.0, 400);
    let Body::Data(big) = ck.tensor("unet", "big").unwrap().body else { unreachable!() };
    assert_eq!(ck.header().spans(&big).len(), 2, "past the grid, big spans two chunks");
    sweep(&ck, "many");
    evict(&ck);
    let cold = read_everything(&ck, 0, &ReadOpts::default());
    if direct_ok(&ck) {
        assert!(cold.direct - cold.bounced > big.length - MIB as u64, "big lands in place: {cold:?}");
    }

    let (gbps, direct) = io::disk(&scratch.0).unwrap();
    println!("disk probe: {gbps:.2} GB/s, direct {direct}");
    assert!(gbps > 0.0 && direct == direct_ok(&ck));
    assert!(io::ceiling(None, &scratch.0).unwrap().disk_gbps > 0.0);
    assert_eq!(io::ceiling(Some(0), &scratch.0).unwrap_err().kind(), std::io::ErrorKind::Unsupported);
}

/// What `Tensors.read` does with many tensors: every range in one allocation at its `pack`
/// offset, in one engine call, cold and warm. Each range lands whole; the padding stays untouched;
/// on the page grid, cold ranges land in place.
#[test]
fn packed_ranges_share_one_allocation_in_one_call() {
    assert_eq!(io::pack([], 256), Some((vec![], 0)));
    assert_eq!(io::pack([3, 0, 256, 1, 0], 256), Some((vec![0, 256, 256, 512, 768], 768)));
    assert_eq!(io::pack([3, 4097], 4096), Some((vec![0, 4096], 8193)));
    assert_eq!(io::pack([usize::MAX - 100, 1], 256), None);
    let scratch = Scratch::new("packed");
    let ck = written(&scratch.0, 400);
    let Body::Data(big) = ck.tensor("unet", "big").unwrap().body else { unreachable!() };
    let all = ranges(&ck);
    for (align, cold) in [(io::DEVICE_ALIGN, true), (io::DEVICE_ALIGN, false), (4096, true)] {
        let (at, len) = io::pack(all.iter().map(|r| r.3.length as usize), align).unwrap();
        for (i, (&a, r)) in at.iter().zip(&all).enumerate() {
            let end = if i == 0 { 0 } else { at[i - 1] + all[i - 1].3.length as usize };
            assert!(a % align == 0 && a >= end && a < end + align, "{}/{}#{}", r.0, r.1, r.2);
        }
        assert_eq!(len, at.last().unwrap() + all.last().unwrap().3.length as usize);
        if cold {
            evict(&ck);
        }
        let mut slab = Buf::new(len, 0);
        let base = slab.mem[slab.at..].as_mut_ptr();
        let targets: Vec<Target> = (all.iter().zip(&at))
            .map(|((c, k, role, l), &a)| Target {
                component: c,
                key: k,
                role,
                offset: 0,
                dst: Dst::Host { ptr: base.wrapping_add(a), len: l.length as usize },
            })
            .collect();
        // SAFETY: disjoint ranges of one live buffer.
        let stats = unsafe { ck.read_into(&targets, &ReadOpts::default()) }.unwrap();
        assert_eq!(stats.bytes, all.iter().map(|r| r.3.length).sum::<u64>());
        if align == 4096 && direct_ok(&ck) {
            assert!(stats.direct - stats.bounced > big.length - MIB as u64, "big lands in place: {stats:?}");
        }
        let bytes = slab.bytes();
        let mut end = 0;
        for ((c, k, role, l), &a) in all.iter().zip(&at) {
            assert!(bytes[end..a].iter().all(|&b| b == 0xA5), "padding before {c}/{k}#{role}");
            end = a + l.length as usize;
            let want = if role.is_empty() { ck.read(c, k).unwrap() } else { stored(&ck, l) };
            assert!(bytes[a..end] == want, "{c}/{k}#{role} cold {cold}");
        }
    }
}

/// Host reads, by tensor or by stored location, go through the engine and log a disk run. A run
/// the page cache served is unmeasured; a cold one that beats the kept ceiling raises it.
#[test]
fn host_reads_run_through_the_engine_and_log_a_disk_run() {
    // The copy log is process-wide and bounded, and the ceiling table is this child's own.
    if std::env::var_os("TENSORFS_IO_LOG_CHILD").is_none() {
        let home = Scratch::new("logged-home");
        let status = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "host_reads_run_through_the_engine_and_log_a_disk_run", "--nocapture"])
            .env("TENSORFS_IO_LOG_CHILD", "1")
            .env("TENSORFS_HOME", &home.0)
            .status()
            .unwrap();
        assert!(status.success());
        return;
    }
    let scratch = Scratch::new("logged");
    let ck = written(&scratch.0, 4);
    let Body::Data(big) = ck.tensor("unet", "big").unwrap().body else { unreachable!() };
    io::metrics::take();
    assert_eq!(ck.read("unet", "big").unwrap(), noise(9, big.length as usize));
    let runs = io::metrics::take();
    let run = runs.iter().find(|c| c.kind == "disk" && c.bytes == big.length);
    assert!(run.is_some_and(|c| c.gpu == -1 && c.ns > 0), "{runs:?}");
    assert!(runs.iter().all(|c| c.cached == 0 || c.ceiling == 0.0), "{runs:?}");
    if !cfg!(target_os = "linux") || !direct_ok(&ck) {
        return;
    }
    let boot = fs::read_to_string("/proc/sys/kernel/random/boot_id").unwrap();
    let volume = platform::durable::device(&scratch.0).unwrap();
    let table = PathBuf::from(std::env::var_os("TENSORFS_HOME").unwrap()).join("tmp/ceilings/table");
    fs::create_dir_all(table.parent().unwrap()).unwrap();
    fs::write(&table, format!("{} disk {volume}\t0.000001\t1\n", boot.trim())).unwrap();
    evict(&ck);
    assert_eq!(ck.read("unet", "big").unwrap(), noise(9, big.length as usize));
    let runs = io::metrics::take();
    // A loaded host may keep a page through the eviction; that run is unmeasured.
    let Some(cold) = runs.iter().find(|c| c.kind == "disk" && c.cached == 0) else { return };
    let rate = cold.bytes as f64 / cold.ns as f64;
    assert!((cold.ceiling - rate).abs() < 1e-9, "a run faster than its ceiling raises it: {cold:?}");
    assert_eq!(fs::read_to_string(&table).unwrap().lines().count(), 2);
}

/// Host read GB/s on this machine against `cat` and `dd iflag=direct` over the same chunk files
/// (`TENSORFS_IO_BENCH_GB`, default 4). Run with `--ignored --nocapture`.
#[test]
#[ignore]
fn host_read_speed() {
    let gb: usize = std::env::var("TENSORFS_IO_BENCH_GB").map_or(4, |v| v.parse().unwrap());
    let scratch = Scratch::new("speed");
    let mut w = Writer::folder(&scratch.0, "", None, false).unwrap();
    let mut c = w.component("transformer", "", Default::default(), None).unwrap();
    let block = noise(1, 48 * MIB);
    for i in 0..gb * 1024 / 48 {
        let mut bytes = block.clone();
        bytes[..8].copy_from_slice(&(i as u64).to_le_bytes());
        c.put(&format!("blocks.{i}.weight"), &Dtype::U8, &[bytes.len() as u64], Data::Plain(Source::Read(&mut &bytes[..]))).unwrap();
        for j in 0..8 {
            let small = noise((i * 8 + j) as u64, 512 << 10);
            c.put(&format!("blocks.{i}.norm{j}"), &Dtype::U8, &[small.len() as u64], Data::Plain(Source::Read(&mut &small[..]))).unwrap();
        }
    }
    w.commit().unwrap();
    let ck = Checkpoint::folder(&scratch.0, None, None).unwrap();
    let all = ranges(&ck);
    let total: u64 = all.iter().map(|r| r.3.length).sum();
    let files: Vec<String> = (0..ck.header().chunks.len()).map(|i| ck.chunk_path(i).display().to_string()).collect();
    let rate = |secs: f64| total as f64 / secs / 1e9;
    let shell = |cmd: &str| {
        evict(&ck);
        let t = Instant::now();
        assert!(Command::new("bash").arg("-c").arg(cmd).status().unwrap().success(), "{cmd}");
        t.elapsed().as_secs_f64()
    };
    let list = files.join(" ");
    println!("bench: {:.2} GB in {} chunks", total as f64 / 1e9, files.len());
    println!("cat (cold): {:.2} GB/s", rate(shell(&format!("cat {list} > /dev/null"))));
    println!("dd iflag=direct bs=16M, one file at a time (cold): {:.2} GB/s", rate(shell(&format!("for f in {list}; do dd if=$f of=/dev/null bs=16M iflag=direct status=none; done"))));
    println!("disk probe: {:?}", io::disk(&scratch.0).unwrap());
    let mut bufs: Vec<Buf> = all.iter().map(|r| Buf::new(r.3.length as usize, 0)).collect();
    for (threads, request) in [(16, MIB), (16, 512 << 10), (16, 4 * MIB), (16, 16 * MIB), (8, MIB), (32, MIB), (4, MIB)] {
        let opts = ReadOpts { threads, request, ..ReadOpts::default() };
        let targets: Vec<Target> = (all.iter().zip(&mut bufs))
            .map(|((c, k, role, _), b)| Target { component: c, key: k, role, offset: 0, dst: b.dst() })
            .collect();
        for cell in ["cold", "warm"] {
            if cell == "cold" {
                evict(&ck);
            }
            // SAFETY: each destination is its own live buffer.
            let s = unsafe { ck.read_into(&targets, &opts) }.unwrap();
            println!("engine {cell} threads {threads} request {} KiB: {:.2} GB/s ({s:?})", request >> 10, s.gbps());
        }
    }
    for (b, (c, k, _, _)) in bufs.iter().zip(&all).step_by(97) {
        assert!(b.bytes() == ck.read(c, k).unwrap(), "{c}/{k}");
    }
}
