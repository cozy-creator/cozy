//! Refill speeds for the audit's step 3 (host RAM = the OS page cache). The same bytes go onto
//! the GPU from five sources:
//! - pinned host memory (the RAM tier's path);
//! - the page cache, by buffered reads through TensorFS's pinned staging (`warm`);
//! - the page cache with TensorFS's default O_DIRECT reads (`warm_default`);
//! - disk by buffered reads (`cold`, no O_DIRECT);
//! - disk with O_DIRECT (`cold_direct`).
//!
//! One copy per 32 MiB block, as a fill issues them, and every mode's bytes are read back.
//! `REFILL_GIB` (default 4) GiB in 1 GiB files under `REFILL_DIR` (default the temp dir). The
//! test uses only `io::read`, so the same file builds against the shipping TensorFS (0.13.6).
#![cfg(target_os = "linux")]

use std::fs::{self, File};
use std::io::Write;
use std::os::fd::AsRawFd;
use std::os::unix::fs::FileExt;
use std::path::PathBuf;
use std::time::Instant;

use tensorfs::io::{self, Copy, Dst, ReadOpts, ReadStats, Src};

const MIB: usize = 1 << 20;
const GIB: usize = 1 << 30;
const BLOCK: usize = 32 * MIB;

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

/// Drops `files` from the page cache: the next read is cold.
fn evict(files: &[File]) {
    for f in files {
        f.sync_all().unwrap();
        // SAFETY: an advisory call on our own descriptor.
        assert_eq!(unsafe { libc::posix_fadvise(f.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) }, 0);
    }
}

/// Reads `files` whole by plain reads at their offsets: the page cache holds them.
fn warm(files: &[File], buf: &mut [u8]) {
    for f in files {
        let mut at = 0;
        while let n @ 1.. = f.read_at(buf, at).unwrap() {
            at += n as u64;
        }
    }
}

#[test]
#[ignore = "requires an allocated CUDA GPU; no mock driver"]
fn page_cache_refill_against_the_pinned_tier() {
    let gib: usize = std::env::var("REFILL_GIB").ok().and_then(|v| v.parse().ok()).unwrap_or(4);
    let dir = std::env::var("REFILL_DIR").map(PathBuf::from).unwrap_or_else(|_| std::env::temp_dir()).join(format!("refill-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let total = gib * GIB;
    let started = Instant::now();
    let paths: Vec<PathBuf> = (0..gib).map(|i| dir.join(format!("part-{i}"))).collect();
    for (i, path) in paths.iter().enumerate() {
        let mut f = File::create(path).unwrap();
        f.write_all(&noise(i as u64 + 1, GIB)).unwrap();
        f.sync_all().unwrap();
    }
    let files: Vec<File> = paths.iter().map(|p| File::open(p).unwrap()).collect();
    println!("STEP {} {}", line!(), started.elapsed().as_millis());
    let gpu = io::DeviceBuffer::new(0, total).unwrap();
    let uuid = io::cuda::uuid(0).unwrap();
    let device = |at: usize, len: usize| Dst::Device { device: 0, ptr: gpu.ptr() + at as u64, len, stream: 0 };
    let blocks: Vec<(usize, usize)> = (0..gib).flat_map(|f| (0..GIB).step_by(BLOCK).map(move |o| (f, o))).collect();
    let from_files: Vec<Copy> = blocks.iter().map(|&(f, o)| Copy { src: Src::File(&files[f], o as u64), dst: device(f * GIB + o, BLOCK) }).collect();
    // The first and last block of the range, as each mode left them on the GPU.
    let check = |mode: &str| {
        for &(f, o) in [blocks[0], blocks[blocks.len() - 1]].iter() {
            let mut bytes = vec![0; BLOCK];
            unsafe { io::read(&[Copy { src: Src::Device { device: 0, ptr: gpu.ptr() + (f * GIB + o) as u64, stream: 0 }, dst: Dst::host(&mut bytes) }], &ReadOpts::default()) }.unwrap();
            assert!(bytes == noise(f as u64 + 1, GIB)[o..o + BLOCK], "{mode}: block {f}/{o} differs");
        }
        unsafe { io::read(&[Copy { src: Src::Zero, dst: device(0, total) }], &ReadOpts::default()) }.unwrap();
    };
    let rate = |t: std::time::Duration| total as f64 / t.as_secs_f64() / 1e9;
    let mut rows = vec![];
    let mut run = |mode: &str, copies: &[Copy], opts: ReadOpts| {
        let start = Instant::now();
        let stats: ReadStats = unsafe { io::read(copies, &opts) }.unwrap();
        let gbps = rate(start.elapsed());
        check(mode);
        println!("STEP {} {}", line!(), started.elapsed().as_millis());
        rows.push(format!("{mode} {gbps:.2} GB/s (direct {} cached {} staged {} MiB)", stats.direct / MIB as u64, stats.cached / MIB as u64, stats.staged / MIB as u64));
    };
    // The RAM tier's path: the same bytes in registered host memory.
    let mut host = vec![0u8; total + 4096];
    let at = (4096 - host.as_ptr() as usize % 4096) % 4096;
    for (i, f) in files.iter().enumerate() {
        f.read_exact_at(&mut host[at + i * GIB..at + (i + 1) * GIB], 0).unwrap();
    }
    let source = host[at..].as_ptr();
    {
        let _context = io::cuda::CtxGuard::enter(gpu.context()).unwrap();
        let _pin = unsafe { io::cuda::pin_region(source.cast_mut(), total, true) }.unwrap();
        // SAFETY: every block lies inside the registered `total` bytes at `source`.
        let pinned: Vec<Copy> = blocks.iter().map(|&(f, o)| Copy { src: Src::Host(unsafe { source.add(f * GIB + o) }), dst: device(f * GIB + o, BLOCK) }).collect();
        run("pinned", &pinned, ReadOpts::default());
    }
    drop(host);
    let mut scratch = vec![0u8; 64 * MIB];
    warm(&files, &mut scratch);
    run("warm", &from_files, ReadOpts { direct: false, ..ReadOpts::default() });
    warm(&files, &mut scratch);
    run("warm_default", &from_files, ReadOpts::default());
    evict(&files);
    run("cold", &from_files, ReadOpts { direct: false, ..ReadOpts::default() });
    evict(&files);
    run("cold_direct", &from_files, ReadOpts::default());
    let fs_type = std::process::Command::new("stat").args(["-f", "-c", "%T"]).arg(&dir).output().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_default();
    let ceiling = io::ceiling(Some(0), &dir).map(|c| format!("{c:?}")).unwrap_or_else(|e| format!("unmeasured: {e}"));
    println!("TIMINGS {gib} GiB in {} blocks of 32 MiB on {fs_type}: {}; ceiling {ceiling}", blocks.len(), rows.join("; "));
    drop(files);
    let _ = fs::remove_dir_all(&dir);
    println!("PASS page-cache refill measured {uuid}");
}
