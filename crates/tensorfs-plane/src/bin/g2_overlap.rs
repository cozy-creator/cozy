//! G2a: copy-stream H2D bandwidth from cuMemHostAlloc, registered memfd and pageable sources
//! into cuMemAlloc and VMM destinations, alone and against a concurrent GEMM loop; per-call
//! enqueue cost; host pinning cost. Markdown, then one `JSON:` line.

#[macro_use]
#[path = "gpu_bench/mod.rs"]
mod gpu_bench;

use std::ffi::c_void;
use std::time::Instant;

use gpu_bench::*;

struct Bench<'a> {
    gm: &'a Gemm,
    cs: CUstream,
    gemms: usize,
    ahead: usize,
    copy_bytes: usize,
}

fn main() -> Result<()> {
    let a = Args::parse();
    let device: i32 = a.get("device", 0);
    let cublas: String = a.get("cublas", DEFAULT_CUBLAS.to_string());
    let m: usize = a.get("m", 4096);
    let src_len = a.get("src-mib", 1024usize) * MIB;
    let chunks: Vec<usize> = a.list::<usize>("chunks-mib", "2,16,64,256").iter().map(|c| c * MIB).collect();
    let sources: Vec<String> = a.list("sources", "hostalloc,memfd,memfd_huge,pageable");
    let dests: Vec<String> = a.list("dests", "alloc,vmm");
    let copy_bytes = a.get("copy-mib", 2048usize) * MIB;
    let gemms: usize = a.get("gemms", 100);
    let ahead = a.get("ahead-mib", 256usize) * MIB;
    let handle = a.get("vmm-handle-mib", 64usize) * MIB;
    let reg_reps = a.get("reg-reps", 3usize).max(1);
    a.done("g2_overlap: H2D copy-stream bandwidth vs a concurrent GEMM loop, by host source and destination");

    if chunks.iter().any(|&c| !src_len.is_multiple_of(c)) || !src_len.is_multiple_of(handle) {
        return Err(Error::Invalid("--chunks-mib and --vmm-handle-mib must divide --src-mib".into()));
    }
    let kinds: Vec<HostKind> = sources
        .iter()
        .filter_map(|s| HostKind::parse(s).or_else(|| warn("source", s)))
        .collect();
    for s in dests.iter().filter(|s| !matches!(s.as_str(), "alloc" | "vmm")) {
        warn::<()>("destination", s);
    }

    println!("# G2a: copy overlap\n");
    let g = Gpu::open(device)?;
    let info = g.info()?;
    let exportable = cuda::attribute(g.dev, cuda::ATTR_POSIX_FD_SUPPORTED)? == 1;
    let (cs, ks) = (g.stream()?, g.stream()?);
    let gm = Gemm::new(&cublas, m, ks)?;
    let want = |n: &str| dests.iter().any(|s| s == n);
    let alloc = want("alloc").then(|| DevBuf::new(src_len)).transpose()?;
    let prop = AllocationProp::device(g.dev, exportable);
    let vmm = want("vmm").then(|| VmmRange::new(&prop, src_len, handle)).transpose()?;
    let mut dsts = vec![];
    if let Some(b) = &alloc {
        dsts.push(("alloc", b.0));
    }
    if let Some(r) = &vmm {
        dsts.push(("vmm", r.va));
    }
    let free_mib = g.free_mib()?;

    let mut ev = Events::new(gemms + 1)?;
    for _ in 0..2 {
        gm.batch(&mut ev, gemms)?; // first pass is warm-up: clocks settle
        ev.sync(gemms)?;
    }
    let base = stats(&gm.times(&ev, gemms)?);
    let thp = thp_shmem_setting();
    println!(
        "- GEMM bf16 {m}³ alone: {} ms median ({:.1} TFLOPS); {gemms} GEMMs per window",
        n3(base.median),
        gm.tflops(base.median)
    );
    println!("- source {} MiB; copy stream ≤ {} MiB in flight; VMM dest in {} MiB handles (exportable={exportable})", src_len / MIB, ahead / MIB, handle / MIB);
    println!("- THP shmem_enabled: {thp}");
    println!("- free device memory with destinations + GEMM allocated: {free_mib} MiB\n");
    println!("GB/s = 1e9 B/s. \"w/ GEMM\" copy GB/s counts chunks that ran entirely inside the GEMM window; \"w/ GEMM\" GEMM ms counts GEMMs entirely inside the copy window.\n");

    let b = Bench { gm: &gm, cs, gemms, ahead, copy_bytes };
    let mut src_json = vec![];
    for kind in kinds {
        let mut setup = Samples::default();
        let mut buf: Option<HostBuf> = None;
        for _ in 0..reg_reps {
            drop(buf.take()); // free the previous one first: one source resident at a time
            let mut hb = HostBuf::new(kind, src_len)?;
            setup.merge(&hb.setup);
            if matches!(kind, HostKind::Memfd { .. }) {
                setup.time("register", || hb.register())?;
            }
            buf = Some(hb);
        }
        let src = buf.expect("reg_reps >= 1");
        let pin = match kind {
            HostKind::HostAlloc => Some(setup.stats("alloc").median),
            HostKind::Memfd { .. } => Some(setup.stats("register").median),
            HostKind::Pageable => None,
        };
        let pin_gbps = pin.map(|us| src_len as f64 / (us * 1e3));
        let steps: Vec<String> = setup.0.iter().map(|(k, v)| format!("{k} {} ms", n3(stats(v).median / 1e3))).collect();
        println!("## {}\n", kind.name());
        println!("- setup (median of {reg_reps}): {}", steps.join(", "));
        if let Some(r) = pin_gbps {
            println!("- pinning rate: {} GB/s", n3(r));
        }
        println!("- shmem mapped by 2 MiB PMDs after touch: {} MiB\n", src.pmd_kib / 1024);
        table(&[
            "dest",
            "chunk MiB",
            "copy alone GB/s",
            "copy w/ GEMM GB/s",
            "copy Δ %",
            "GEMM alone ms",
            "GEMM w/ copy ms",
            "GEMM slowdown %",
            "GEMM w/ copy p90",
            "enqueue µs alone med/p90",
            "enqueue µs w/ GEMM med/p90",
        ]);
        let mut cases = vec![];
        for &(name, dst) in &dsts {
            for &c in &chunks {
                cases.push(case(&b, &src, name, dst, c)?);
            }
        }
        println!();
        src_json.push(obj! {"kind" => kind.name(), "setup_us" => setup.json(), "pin_gbps" => pin_gbps,
            "pmd_mib" => src.pmd_kib / 1024, "cases" => cases});
    }

    let cfg = obj! {"m" => m, "src_mib" => src_len / MIB, "copy_mib" => copy_bytes / MIB, "gemms" => gemms,
        "ahead_mib" => ahead / MIB, "vmm_handle_mib" => handle / MIB, "reg_reps" => reg_reps,
        "exportable" => exportable, "thp_shmem" => thp};
    println!("JSON: {}", obj! {"bench" => "g2_overlap", "device" => info, "config" => cfg,
        "gemm_alone_ms" => base, "gemm_tflops" => gm.tflops(base.median), "sources" => src_json});
    Ok(())
}

fn warn<T>(what: &str, s: &str) -> Option<T> {
    eprintln!("warning: unknown {what} {s:?} skipped");
    None
}

struct Copies {
    ev: Events, // ev[k]..ev[k + 1] brackets chunk k
    n: usize,
    enqueue_us: Vec<f64>,
}

/// Chunked H2D copies of `src` into `dst` on the copy stream, cycling over the buffer with at
/// most `ahead` bytes in flight: `copy_bytes` in total, or until `until` completes.
fn copies(b: &Bench, src: &HostBuf, dst: CUdeviceptr, chunk: usize, until: Option<CUevent>) -> Result<Copies> {
    let d = cuda::driver()?;
    let (per, window) = (src.len / chunk, (b.ahead / chunk).max(2));
    let mut c = Copies { ev: Events::new(per + 1)?, n: 0, enqueue_us: vec![] };
    c.ev.record(0, b.cs)?;
    while match until {
        Some(e) => !event_done(e)?,
        None => c.n * chunk < b.copy_bytes,
    } {
        let k = c.n;
        if k >= window {
            c.ev.sync(k + 1 - window)?; // chunk k - window done
        }
        let off = (k % per) * chunk;
        let t = Instant::now();
        cu!(d.memcpy_htod_async(dst + off as u64, src.ptr.add(off) as *const c_void, chunk, b.cs))?;
        c.enqueue_us.push(us(t));
        c.ev.record(k + 1, b.cs)?;
        c.n += 1;
    }
    cu!(d.stream_synchronize(b.cs))?;
    Ok(c)
}

/// (i) copies alone, (ii) GEMMs alone, (iii) GEMMs queued then copies until they finish.
fn case(b: &Bench, src: &HostBuf, dst_name: &str, dst: CUdeviceptr, chunk: usize) -> Result<J> {
    let n = b.gemms;
    let alone = copies(b, src, dst, chunk, None)?;
    let alone_gbps = gbps(alone.n * chunk, alone.ev.ms(0, alone.n)?);
    let mut ev = Events::new(n + 1)?;
    b.gm.batch(&mut ev, n)?;
    ev.sync(n)?;
    let gemm_alone = stats(&b.gm.times(&ev, n)?);
    b.gm.batch(&mut ev, n)?;
    let both = copies(b, src, dst, chunk, Some(ev.e[n]))?;
    ev.sync(n)?;

    let rel = |e: CUevent| elapsed(ev.e[0], e); // ms after the first GEMM's start event
    let g_end = ev.ms(0, n)?;
    let (mut bytes, mut t0, mut t1) = (0, f64::MAX, f64::MIN);
    for k in 0..both.n {
        let (s, e) = (rel(both.ev.e[k])?, rel(both.ev.e[k + 1])?);
        if s >= 0.0 && e <= g_end {
            bytes += chunk;
            t0 = t0.min(s);
            t1 = t1.max(e);
        }
    }
    let both_gbps = (bytes > 0).then(|| gbps(bytes, t1 - t0));
    let (c0, c1) = (rel(both.ev.e[0])?, rel(both.ev.e[both.n])?);
    let mut in_copy = vec![];
    for i in 0..n {
        let (s, e) = (ev.ms(0, i)?, ev.ms(0, i + 1)?);
        if s >= c0 && e <= c1 {
            in_copy.push(e - s);
        }
    }
    let gemm_both = stats(&in_copy);
    let slowdown = (gemm_both.median / gemm_alone.median - 1.0) * 100.0;
    let copy_delta = both_gbps.map(|x| (x / alone_gbps - 1.0) * 100.0);
    let (ea, eb) = (stats(&alone.enqueue_us), stats(&both.enqueue_us));
    let opt = |x: Option<f64>| x.map_or("n/a".to_string(), n3);
    row!(
        dst_name,
        chunk / MIB,
        n3(alone_gbps),
        opt(both_gbps),
        opt(copy_delta),
        n3(gemm_alone.median),
        n3(gemm_both.median),
        n3(slowdown),
        n3(gemm_both.p90),
        format!("{} / {}", n3(ea.median), n3(ea.p90)),
        format!("{} / {}", n3(eb.median), n3(eb.p90)),
    );
    Ok(obj! {"dest" => dst_name, "chunk_mib" => chunk / MIB, "copy_alone_gbps" => alone_gbps,
        "copy_with_gemm_gbps" => both_gbps, "copy_with_gemm_chunks" => bytes / chunk,
        "gemm_alone_ms" => gemm_alone, "gemm_with_copy_ms" => gemm_both, "gemm_slowdown_pct" => slowdown,
        "enqueue_alone_us" => ea, "enqueue_with_gemm_us" => eb})
}
