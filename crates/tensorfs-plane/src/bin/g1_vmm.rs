//! G1 (design study §7.2): VMM call costs by handle size, handles vs bytes, remap-per-step
//! cost, and whether VMM calls or host unregister stall a concurrent GEMM. Markdown, then one
//! `JSON:` line.

#[macro_use]
#[path = "gpu_bench/mod.rs"]
mod gpu_bench;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Instant;

use gpu_bench::*;

fn main() -> Result<()> {
    let a = Args::parse();
    let device: i32 = a.get("device", 0);
    let reps: usize = a.get("reps", 20);
    let sizes: Vec<usize> = a.list::<usize>("sizes-mib", "2,64,256,1024").iter().map(|s| s * MIB).collect();
    let batch = a.get("batch-mib", 1024usize) * MIB;
    let va_len = a.get("va-gib", 64usize) * GIB;
    let cublas: String = a.get("cublas", DEFAULT_CUBLAS.to_string());
    let m: usize = a.get("m", 4096);
    let gemms: usize = a.get("gemms", 200);
    let depth = a.get("depth", 32usize).max(1);
    let stall: Vec<usize> = a.list::<usize>("stall-sizes-mib", "64,1024").iter().map(|s| s * MIB).collect();
    let host_reg = a.get("host-reg-mib", 256usize) * MIB;
    a.done("g1_vmm: CUDA VMM call costs and the GEMM stall test (design study §7.2)");

    println!("# G1: VMM costs\n");
    let g = Gpu::open(device)?;
    let d = g.d;
    let info = g.info()?;
    let g_free = g.free_mib()?;
    if cuda::attribute(g.dev, cuda::ATTR_VMM_SUPPORTED)? != 1 {
        return Err(Error::Invalid("device lacks VMM support".into()));
    }
    let exportable = cuda::attribute(g.dev, cuda::ATTR_POSIX_FD_SUPPORTED)? == 1;
    let prop = AllocationProp::device(g.dev, exportable);
    let exports: &[bool] = if exportable { &[false, true] } else { &[false] };

    println!("## Granularity\n");
    table(&["exportable", "minimum", "recommended"]);
    let (mut gran, mut min_gran) = (vec![], 0);
    for &e in exports {
        let p = AllocationProp::device(g.dev, e);
        let mut v = [0usize; 2];
        cu!(d.mem_get_allocation_granularity(&mut v[0], &p, cuda::MEM_GRANULARITY_MINIMUM))?;
        cu!(d.mem_get_allocation_granularity(&mut v[1], &p, GRANULARITY_RECOMMENDED))?;
        min_gran = min_gran.max(v[0]);
        row!(e, v[0], v[1]);
        gran.push(obj! {"exportable" => e, "minimum" => v[0], "recommended" => v[1]});
    }
    let bad = |s: usize| s == 0 || !s.is_multiple_of(min_gran);
    if sizes.iter().any(|&s| bad(s) || !batch.is_multiple_of(s)) || stall.iter().any(|&s| bad(s)) {
        return Err(Error::Invalid(format!("sizes must be {min_gran} B multiples dividing --batch-mib")));
    }

    println!("\n## Export flag: create/release of 64 MiB ({reps} reps)\n");
    table(&["exportable", "create µs median", "p90", "release µs median", "p90"]);
    let mut export_cost = vec![];
    for &e in exports {
        let p = AllocationProp::device(g.dev, e);
        let s = repeat(reps, |t| {
            let mut h = 0;
            t.time("create", || cu!(d.mem_create(&mut h, 64 * MIB, &p, 0)))?;
            t.time("release", || cu!(d.mem_release(h)))
        })?;
        let (c, r) = (s.stats("create"), s.stats("release"));
        row!(e, n3(c.median), n3(c.p90), n3(r.median), n3(r.p90));
        export_cost.push(obj! {"exportable" => e, "calls_us" => s.json()});
    }

    println!("\n## VA reserve/free of {} GiB ({reps} reps)\n", va_len / GIB);
    let va = repeat(reps, |t| {
        let mut p = 0;
        t.time("reserve", || cu!(d.mem_address_reserve(&mut p, va_len, 0, 0, 0)))?;
        t.time("free", || cu!(d.mem_address_free(p, va_len)))
    })?;
    table(&["call", "median µs", "p90", "max"]);
    for (k, v) in &va.0 {
        let s = stats(v);
        row!(k, n3(s.median), n3(s.p90), n3(s.max));
    }

    println!("\n## Lifecycle per call (exportable={exportable}, {} MiB per rep, {reps} reps)\n", batch / MIB);
    table(&["size MiB", "handles/rep", "call", "median µs", "p90", "max", "µs per GiB"]);
    let (mut life, mut totals) = (vec![], vec![]);
    for &s in &sizes {
        let (per, tot) = lifecycle(&g, &prop, s, batch, reps)?;
        per_call_rows(s, batch / s, &per);
        life.push(obj! {"size_mib" => s / MIB, "handles" => batch / s, "calls_us" => per.json(),
            "per_gib_us" => per_gib(&per, s), "totals_us" => tot.json()});
        totals.push((s, tot));
    }
    println!("\n## Handles vs bytes: {} MiB as N handles (create+map+one set_access; unmap+release)\n", batch / MIB);
    table(&["layout", "build ms median", "p90", "teardown ms median", "p90", "set_access(span) µs"]);
    for (s, t) in &totals {
        let (b, td, sa) = (t.stats("build"), t.stats("teardown"), t.stats("set_access_span"));
        let layout = format!("{} × {} MiB", batch / s, s / MIB);
        row!(layout, n3(b.median / 1e3), n3(b.p90 / 1e3), n3(td.median / 1e3), n3(td.p90 / 1e3), n3(sa.median));
    }

    println!("\n## Recycle: unmap + map + set_access of a live handle ({reps} reps)\n");
    table(&["size MiB", "handles/rep", "call", "median µs", "p90", "max", "µs per GiB"]);
    let mut recycled = vec![];
    for &s in &sizes {
        let r = recycle(&prop, s, batch, reps)?;
        per_call_rows(s, batch / s, &r);
        recycled.push(obj! {"size_mib" => s / MIB, "handles" => batch / s, "calls_us" => r.json(),
            "per_gib_us" => per_gib(&r, s)});
    }

    // Stall test.
    let cs = g.stream()?;
    let gm = Gemm::new(&cublas, m, cs)?;
    let mut ops: Vec<Op> = stall.iter().map(|&s| Op::Churn(s)).collect();
    ops.extend(stall.iter().map(|&s| Op::Recycle(s)));
    ops.push(Op::HostReg(host_reg));
    let mut idle = vec![];
    for &op in &ops {
        idle.push(repeat_worker(op, g.dev, exportable, reps)?);
    }
    phase(&g, &gm, gemms, depth, None, exportable)?; // warm-up: clocks settle
    let mut phases = vec![phase(&g, &gm, gemms, depth, None, exportable)?];
    for &op in &ops {
        phases.push(phase(&g, &gm, gemms, depth, Some(op), exportable)?);
    }
    phases.push(phase(&g, &gm, gemms, depth, None, exportable)?);
    let min_free = phases.iter().map(|p| p.free_mib).min().unwrap_or(0);

    let base = stats(&phases[0].gemm_ms).median;
    let drain = depth as f64 * base;
    println!("\n## Stall test: {gemms} GEMMs (bf16 {m}³, {:.1} TFLOPS), {depth} kept queued ≈ {drain:.0} ms of work\n", gm.tflops(base));
    println!("A host call that waits for the device to idle takes up to ~{drain:.0} ms in flight.\n");
    table(&["phase", "GEMM ms median", "p90", "max", "launch µs median", "p90", "max", "worker cycles"]);
    let labels: Vec<String> = std::iter::once("A baseline".to_string())
        .chain(ops.iter().map(|o| o.label()))
        .chain(std::iter::once("A baseline (end)".to_string()))
        .collect();
    for (p, l) in phases.iter().zip(&labels) {
        let (gs, ls) = (stats(&p.gemm_ms), stats(&p.launch_us));
        row!(l, n3(gs.median), n3(gs.p90), n3(gs.max), n3(ls.median), n3(ls.p90), n3(ls.max), p.cycles);
    }
    println!();
    table(&["phase", "call", "idle µs median", "in flight median", "p90", "max", "max / idle median"]);
    let mut stall_json = vec![];
    for (i, (p, l)) in phases.iter().zip(&labels).enumerate() {
        let quiet = if i > 0 && i <= ops.len() { Some(&idle[i - 1]) } else { None };
        let mut calls = vec![];
        for (k, v) in &p.ops.0 {
            let (b, q) = (stats(v), quiet.map(|q| q.stats(k)).unwrap_or_default());
            row!(l, k, n3(q.median), n3(b.median), n3(b.p90), n3(b.max), n3(b.max / q.median));
            calls.push(obj! {"call" => *k, "idle_us" => q, "in_flight_us" => b});
        }
        stall_json.push(obj! {"phase" => l.as_str(), "gemm_ms" => stats(&p.gemm_ms),
            "launch_us" => stats(&p.launch_us), "cycles" => p.cycles, "calls" => calls});
    }
    println!("\nFree device memory: {} MiB at start, {min_free} MiB minimum during the stall phases.", g_free);

    let cfg = obj! {"reps" => reps, "batch_mib" => batch / MIB, "va_gib" => va_len / GIB, "m" => m,
        "gemms" => gemms, "depth" => depth, "host_reg_mib" => host_reg / MIB, "exportable" => exportable,
        "free_mib_start" => g_free, "free_mib_min" => min_free, "gemm_device_mib" => gm.device_bytes() / MIB};
    println!("\nJSON: {}", obj! {"bench" => "g1_vmm", "device" => info, "config" => cfg,
        "granularity" => gran, "export_cost" => export_cost, "va_reserve_us" => va.json(),
        "lifecycle" => life, "recycle" => recycled,
        "stall" => obj! {"gemm_tflops" => gm.tflops(base), "drain_ms" => drain, "phases" => stall_json}});
    Ok(())
}

/// One warm-up, then `reps` timed runs of `f`.
fn repeat(reps: usize, mut f: impl FnMut(&mut Samples) -> Result<()>) -> Result<Samples> {
    f(&mut Samples::default())?;
    let mut s = Samples::default();
    for _ in 0..reps {
        f(&mut s)?;
    }
    Ok(s)
}

fn per_gib(s: &Samples, size: usize) -> J {
    let k = GIB as f64 / size as f64;
    J::O(s.0.iter().map(|(n, v)| (n.to_string(), J::from(stats(v).median * k))).collect())
}

fn per_call_rows(size: usize, handles: usize, s: &Samples) {
    for (k, v) in &s.0 {
        let st = stats(v);
        let per_gib = st.median * GIB as f64 / size as f64;
        row!(size / MIB, handles, k, n3(st.median), n3(st.p90), n3(st.max), n3(per_gib));
    }
}

/// Per rep: `batch / s` handles through create, map, set_access, unmap, release (each call
/// timed), then the same batch built with one set_access over the span (totals).
fn lifecycle(g: &Gpu, prop: &AllocationProp, s: usize, batch: usize, reps: usize) -> Result<(Samples, Samples)> {
    let (d, acc, n) = (g.d, access(g.dev), batch / s);
    let va = reserve(batch)?;
    let at = |i: usize| va + (i * s) as u64;
    let mut hs = vec![0 as Handle; n];
    let (mut per, mut tot) = (Samples::default(), Samples::default());
    for rep in 0..=reps {
        let (mut p, mut t) = (Samples::default(), Samples::default());
        for h in hs.iter_mut() {
            p.time("create", || cu!(d.mem_create(h, s, prop, 0)))?;
        }
        for (i, &h) in hs.iter().enumerate() {
            p.time("map", || cu!(d.mem_map(at(i), s, 0, h, 0)))?;
        }
        for i in 0..n {
            p.time("set_access", || cu!(d.mem_set_access(at(i), s, &acc, 1)))?;
        }
        for i in 0..n {
            p.time("unmap", || cu!(d.mem_unmap(at(i), s)))?;
        }
        for &h in &hs {
            p.time("release", || cu!(d.mem_release(h)))?;
        }
        let t0 = Instant::now();
        for h in hs.iter_mut() {
            cu!(d.mem_create(h, s, prop, 0))?;
        }
        for (i, &h) in hs.iter().enumerate() {
            cu!(d.mem_map(at(i), s, 0, h, 0))?;
        }
        t.time("set_access_span", || cu!(d.mem_set_access(va, batch, &acc, 1)))?;
        t.add("build", us(t0));
        let t1 = Instant::now();
        for i in 0..n {
            cu!(d.mem_unmap(at(i), s))?;
        }
        for &h in &hs {
            cu!(d.mem_release(h))?;
        }
        t.add("teardown", us(t1));
        if rep > 0 {
            per.merge(&p);
            tot.merge(&t);
        }
    }
    cu!(d.mem_address_free(va, batch))?;
    Ok((per, tot))
}

fn recycle(prop: &AllocationProp, s: usize, batch: usize, reps: usize) -> Result<Samples> {
    let d = cuda::driver()?;
    let (r, acc) = (VmmRange::new(prop, batch, s)?, access(prop.location.id));
    repeat(reps, |t| {
        for (i, &h) in r.hs.iter().enumerate() {
            let (a, t0) = (r.va + (i * s) as u64, Instant::now());
            t.time("unmap", || cu!(d.mem_unmap(a, s)))?;
            t.time("map", || cu!(d.mem_map(a, s, 0, h, 0)))?;
            t.time("set_access", || cu!(d.mem_set_access(a, s, &acc, 1)))?;
            t.add("cycle", us(t0));
        }
        Ok(())
    })
}

#[derive(Clone, Copy)]
enum Op {
    /// create + map + set_access + unmap + release of one handle.
    Churn(usize),
    /// unmap + map + set_access of one live handle.
    Recycle(usize),
    /// cuMemHostRegister + cuMemHostUnregister of a touched memfd mapping.
    HostReg(usize),
}

impl Op {
    fn label(self) -> String {
        match self {
            Op::Churn(s) => format!("B churn {} MiB", s / MIB),
            Op::Recycle(s) => format!("C recycle {} MiB", s / MIB),
            Op::HostReg(s) => format!("D host register {} MiB memfd", s / MIB),
        }
    }
}

/// Owns the memory one `Op` cycles; memory the GEMM never touches.
struct Worker {
    op: Op,
    prop: AllocationProp,
    acc: AccessDesc,
    va: CUdeviceptr,
    h: Handle,
    host: Option<HostBuf>,
}

impl Worker {
    fn new(op: Op, dev: CUdevice, exportable: bool) -> Result<Worker> {
        let d = cuda::driver()?;
        let (prop, acc) = (AllocationProp::device(dev, exportable), access(dev));
        let mut w = Worker { op, prop, acc, va: 0, h: 0, host: None };
        match op {
            Op::Churn(s) => w.va = reserve(s)?,
            Op::Recycle(s) => {
                w.va = reserve(s)?;
                cu!(d.mem_create(&mut w.h, s, &prop, 0))?;
                cu!(d.mem_map(w.va, s, 0, w.h, 0))?;
                cu!(d.mem_set_access(w.va, s, &acc, 1))?;
            }
            Op::HostReg(s) => w.host = Some(HostBuf::new(HostKind::Memfd { huge: false }, s)?),
        }
        Ok(w)
    }

    fn step(&mut self, t: &mut Samples) -> Result<()> {
        let d = cuda::driver()?;
        let (va, h, prop, acc) = (self.va, self.h, self.prop, self.acc);
        match self.op {
            Op::Churn(s) => {
                let mut h = 0;
                t.time("create", || cu!(d.mem_create(&mut h, s, &prop, 0)))?;
                t.time("map", || cu!(d.mem_map(va, s, 0, h, 0)))?;
                t.time("set_access", || cu!(d.mem_set_access(va, s, &acc, 1)))?;
                t.time("unmap", || cu!(d.mem_unmap(va, s)))?;
                t.time("release", || cu!(d.mem_release(h)))
            }
            Op::Recycle(s) => {
                t.time("unmap", || cu!(d.mem_unmap(va, s)))?;
                t.time("map", || cu!(d.mem_map(va, s, 0, h, 0)))?;
                t.time("set_access", || cu!(d.mem_set_access(va, s, &acc, 1)))
            }
            Op::HostReg(_) => {
                let b = self.host.as_mut().expect("HostReg owns a buffer");
                t.time("register", || b.register())?;
                t.time("unregister", || b.unregister())
            }
        }
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        let (Ok(d), Op::Churn(s) | Op::Recycle(s)) = (cuda::driver(), self.op) else { return };
        // SAFETY: tearing down what `new` built.
        unsafe {
            if self.h != 0 {
                (d.mem_unmap)(self.va, s);
                (d.mem_release)(self.h);
            }
            (d.mem_address_free)(self.va, s);
        }
    }
}

fn repeat_worker(op: Op, dev: CUdevice, exportable: bool, reps: usize) -> Result<Samples> {
    let mut w = Worker::new(op, dev, exportable)?;
    repeat(reps, |t| w.step(t))
}

struct Phase {
    gemm_ms: Vec<f64>,
    launch_us: Vec<f64>,
    ops: Samples,
    cycles: usize,
    free_mib: usize, // after priming: worker memory, GEMM buffers and the queue all live
}

/// `n` GEMMs with at most `depth` in flight; once the queue is full a worker thread cycles
/// `op` until the last GEMM is enqueued (so every cycle overlaps queued GEMMs).
fn phase(g: &Gpu, gm: &Gemm, n: usize, depth: usize, op: Option<Op>, exportable: bool) -> Result<Phase> {
    let mut ev = Events::new(n + 1)?;
    let (stop, go) = (Arc::new(AtomicBool::new(false)), Arc::new(Barrier::new(2)));
    let (ctx, dev) = (g.ctx as usize, g.dev);
    let worker = op.map(|op| {
        let (stop, go) = (stop.clone(), go.clone());
        thread::spawn(move || -> Result<(Samples, usize)> {
            let setup = CtxGuard::enter(ctx as CUcontext).and_then(|c| Ok((c, Worker::new(op, dev, exportable)?)));
            go.wait();
            let (bound, w) = setup?;
            let _bound = bound;
            let mut w = w;
            let (mut t, mut cycles) = (Samples::default(), 0);
            while !stop.load(Ordering::Acquire) {
                w.step(&mut t)?;
                cycles += 1;
            }
            Ok((t, cycles))
        })
    });
    let (mut launch_us, mut free_mib) = (Vec::with_capacity(n), 0);
    ev.record(0, gm.stream)?;
    for i in 0..n {
        if i >= depth {
            ev.sync(i + 1 - depth)?; // GEMM i - depth done
        }
        let t = Instant::now();
        gm.enqueue()?;
        ev.record(i + 1, gm.stream)?;
        launch_us.push(us(t));
        if i + 1 == depth.min(n) {
            if worker.is_some() {
                go.wait();
            }
            free_mib = g.free_mib()?;
        }
    }
    stop.store(true, Ordering::Release);
    ev.sync(n)?;
    let (ops, cycles) = match worker {
        Some(w) => w.join().map_err(|_| Error::Invalid("worker panicked".into()))??,
        None => (Samples::default(), 0),
    };
    Ok(Phase { gemm_ms: gm.times(&ev, n)?, launch_us, ops, cycles, free_mib })
}
