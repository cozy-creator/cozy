//! The read/projection CLI (tfs-005). Every verb here drives the real store: there are no
//! simulated reads, and the red arms are planted into real bytes.

use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use tensorfs_core::checkpoint;
use tensorfs_core::err::{Code, Refusal};
use tensorfs_core::fill::{guard_block_granular, Class, Xform};
use tensorfs_core::header::Header;
use tensorfs_core::ids::ObjectRef;
use tensorfs_core::manifest::{Entry, Manifest};
use tensorfs_core::meta::Meta;
use tensorfs_core::project;
use tensorfs_core::read::{self, ObjectRange, Order, PlanItem, ReadPlan, Source as ReadSource};
use tensorfs_core::sha256::{self, Sha256};
use tensorfs_core::store::Store;

use super::{bail, flag, flag_num, flag_on, Flags};

macro_rules! ok {
    ($e:expr) => {
        match $e {
            Ok(v) => v,
            Err(e) => return bail(e),
        }
    };
}

fn load(root: &Path, hex: &str) -> Result<(Store, Manifest, Header), Refusal> {
    let s = Store::open(root)?;
    let hex = hex.trim_start_matches("sha256:");
    let m = checkpoint::load_manifest(
        &s,
        &ObjectRef {
            sha256: hex.to_string(),
            length: std::fs::metadata(s.manifest_path(hex))
                .map(|m| m.len())
                .unwrap_or(0),
        },
    )?;
    let h = checkpoint::load_header(&s, m.header().unwrap())?;
    Ok((s, m, h))
}

/// Every component this header carries. The `tfs` CLI's own plans are whole-artifact by
/// construction — it has no tensor requirements to narrow by — so it declares the full set
/// (#570b). A caller that binds a subset declares the subset.
fn all_components(h: &Header) -> Vec<String> {
    h.components.iter().map(|(c, _)| c.clone()).collect()
}

fn construction_order(h: &Header) -> Vec<(String, String)> {
    h.tensors()
        .map(|(c, k, _)| (c.clone(), k.clone()))
        .collect()
}

// ---------------------------------------------------------------- lease

pub fn cmd_lease(root: &Path, hex: &str, f: &Flags) -> ExitCode {
    let (s, m, _) = ok!(load(root, hex));
    let meta = ok!(Meta::open(&s));
    let t0 = Instant::now();
    let (lease, rx) = ok!(read::acquire_cozytensors(&s, &meta, &m));
    println!("manifest:   sha256:{hex}");
    println!("hold:       {}", lease.hold_id());
    println!(
        "verified:   {} objects, {} B in {} ms",
        lease.objects().len(),
        lease.bytes(),
        t0.elapsed().as_millis()
    );
    println!("rehashed:   {}", rx.count("record-invalidated-rehash"));
    rx.print();
    let secs: u64 = flag_num(f, "hold-secs", 0);
    if secs > 0 {
        if let Some(p) = flag(f, "ready") {
            let _ = std::fs::write(p, format!("{}\n", lease.hold_id()));
        }
        println!(
            "HOLDING {} for {secs}s pid={}",
            lease.hold_id(),
            std::process::id()
        );
        std::thread::sleep(std::time::Duration::from_secs(secs));
    }
    if flag_on(f, "drop-unreleased") {
        // The lease is DROPPED, not released: the row must survive for the reaper and the
        // drop must be announced. Nothing here releases it.
        drop(lease);
        println!("dropped without release — the row is kept, the lock is not");
        return ExitCode::SUCCESS;
    }
    ok!(lease.release(&meta));
    println!("released:   explicitly");
    ExitCode::SUCCESS
}

// ---------------------------------------------------------------- plan

fn plan_of(h: &Header, f: &Flags) -> Result<ReadPlan, Refusal> {
    let window: u64 = flag_num(f, "window", 16 << 20);
    read::plan_for_traversal(h, &construction_order(h), &all_components(h), window)
}

pub fn cmd_plan(root: &Path, hex: &str, f: &Flags) -> ExitCode {
    let (_, _, h) = ok!(load(root, hex));
    let window: u64 = flag_num(f, "window", 16 << 20);
    let dest = ok!(read::plan_for_traversal(
        &h,
        &construction_order(&h),
        &all_components(&h),
        window
    ));
    println!(
        "items:      {} covering {} B ({} B disk leg)",
        dest.items.len(),
        dest.bytes,
        dest.io_bytes
    );
    println!(
        "inline:     {} parts, {} B out of the header — zero I/O, still in the plan and in order",
        dest.inline_parts,
        dest.bytes - dest.io_bytes
    );
    println!("order:      {}", dest.order.as_str());

    // The contract, asserted rather than assumed: the emitted plan follows the traversal it
    // was given, in that exact order. Proved against the REVERSED traversal too, because a
    // planner that quietly re-sorted would pass the identity case and fail this one.
    let mut rev = construction_order(&h);
    rev.reverse();
    println!(
        "contract:   reversed caller traversal plans in its own order: {}",
        match read::plan_for_traversal(&h, &rev, &all_components(&h), window) {
            Ok(p)
                if p.items.first().is_some_and(|it| it
                    .what
                    .starts_with(&format!("{}/{}#", rev[0].0, rev[0].1))) =>
                "YES",
            _ => "NO — DEFECT",
        }
    );
    for (n, it) in dest.items.iter().take(6).enumerate() {
        println!("  dest[{n}] {} +{} {} B", it.what, it.dest_off, it.len);
    }
    ExitCode::SUCCESS
}

// ---------------------------------------------------------------- fill

pub fn cmd_fill(root: &Path, hex: &str, f: &Flags) -> ExitCode {
    let (s, m, h) = ok!(load(root, hex));
    let meta = ok!(Meta::open(&s));
    let plan = ok!(plan_of(&h, f));
    let workers: usize = flag_num(f, "workers", 4);
    let ring: u64 = flag_num(f, "ring", 256 << 20);
    let (lease, mut rx) = ok!(read::acquire_cozytensors(&s, &meta, &m));
    if flag_on(f, "direct") {
        // The caller asked for O_DIRECT and did not get it. tensorfs.md §4 keeps direct I/O
        // deliberately non-default and NEVER a silent fallback where unsupported — so the
        // fallback emits a receipt and the caller can see what it actually bought.
        rx.emit(tensorfs_core::receipt::Receipt::new(
            "read_fill",
            "buffered-instead-of-direct",
            hex,
            "O_DIRECT requested; the v1 buffered baselines are unbeaten and the page cache is \
             what makes warm reloads nearly free — served buffered"
                .to_string(),
        ));
    }
    rx.print();

    let batches = plan.batches(ring);
    let mut buf = vec![0u8; ring as usize];
    let mut hasher = Sha256::new();
    let digest_it = !flag_on(f, "no-digest");
    let (mut bytes, mut io, mut items, mut ms, mut hash_ms) = (0u64, 0u64, 0usize, 0u128, 0u128);
    for b in &batches {
        let dest = &mut buf[..b.bytes as usize];
        // The CLOCK covers the reads and nothing else. Hashing 4.8 GiB with our own sha256
        // costs ~28 s at 190 MiB/s and would have reported the hash as the read.
        let t0 = Instant::now();
        let st = match read::execute(&lease, b, dest, workers) {
            Ok(st) => st,
            Err(e) => {
                let _ = lease.release(&meta);
                return bail(e);
            }
        };
        ms += t0.elapsed().as_millis();
        bytes += st.bytes;
        io += st.io_bytes;
        items += st.items;
        if digest_it {
            let t1 = Instant::now();
            hasher.update(dest);
            hash_ms += t1.elapsed().as_millis();
        }
    }
    let digest = sha256::hex(&hasher.finish());
    ok!(lease.release(&meta));

    println!("order:      {}", plan.order.as_str());
    println!(
        "workers:    {workers}   window {} B   ring {ring} B",
        flag_num::<u64>(f, "window", 16 << 20)
    );
    println!("batches:    {}", batches.len());
    println!("filled:     {items} items, {bytes} B ({io} B disk leg) in {ms} ms");
    println!(
        "throughput: {:.3} GB/s ({:.0} MiB/s) over the disk leg",
        io as f64 / 1e9 / (ms as f64 / 1000.0),
        io as f64 / 1048576.0 / (ms as f64 / 1000.0)
    );
    println!("peak buf:   {ring} B (caller-owned ring — TensorFS allocated no destination)");
    if digest_it {
        println!("digest:     {hash_ms} ms OUTSIDE the read clock (our sha256, not the read)");
        println!("fill sha256:{digest}");
    }
    ExitCode::SUCCESS
}

/// `tfs read stream` — the SAME plan through the continuous slot ring (`read::pump`).
/// This exists so the ring can be measured against `read fill`'s per-batch `execute` on the
/// same bytes in the same minutes, and so the Python facade's number has a Rust arm running
/// the identical mechanism rather than a differently-shaped one.
pub fn cmd_stream(root: &Path, hex: &str, f: &Flags) -> ExitCode {
    let (s, m, h) = ok!(load(root, hex));
    let meta = ok!(Meta::open(&s));
    let plan = ok!(plan_of(&h, f));
    let workers: usize = flag_num(f, "workers", 4);
    let nslots: usize = flag_num(f, "slots", 4);
    let slot_bytes: usize = flag_num(f, "slot-bytes", 16 << 20);
    let (lease, rx) = ok!(read::acquire_cozytensors(&s, &meta, &m));
    rx.print();

    let mut owned: Vec<Vec<u8>> = (0..nslots).map(|_| vec![0u8; slot_bytes]).collect();
    let slots: Vec<&mut [u8]> = owned.iter_mut().map(|v| &mut v[..]).collect();
    let digest_it = !flag_on(f, "no-digest");
    let mut hasher = Sha256::new();
    let mut batches = 0usize;
    let mut hash_ms = 0u128;
    let t0 = Instant::now();
    let r = read::pump(&lease, &plan, slots, workers, |fl, slot| {
        batches += 1;
        if digest_it {
            // The consumer reads its slot BEFORE releasing it: an explicit release is what
            // keeps these bytes valid across an asynchronous consumer.
            let t1 = Instant::now();
            hasher.update(fl.bytes);
            hash_ms += t1.elapsed().as_millis();
        }
        slot.release();
        Ok(())
    });
    let wall = t0.elapsed().as_millis();
    let ms = wall.saturating_sub(hash_ms);
    match r {
        Ok(st) => {
            ok!(lease.release(&meta));
            println!("order:      {}", plan.order.as_str());
            println!(
                "ring:       {nslots} slots x {slot_bytes} B, {workers} readers, {batches} batches"
            );
            println!(
                "filled:     {} items, {} B ({} B disk leg) in {ms} ms",
                st.items, st.bytes, st.io_bytes
            );
            println!(
                "throughput: {:.3} GB/s ({:.0} MiB/s) over the disk leg",
                st.io_bytes as f64 / 1e9 / (ms as f64 / 1000.0),
                st.io_bytes as f64 / 1048576.0 / (ms as f64 / 1000.0)
            );
            println!("peak buf:   {} B (caller-owned ring)", nslots * slot_bytes);
            if digest_it {
                println!(
                    "digest:     {hash_ms} ms OUTSIDE the read clock (our sha256, not the read)"
                );
                println!("fill sha256:{}", sha256::hex(&hasher.finish()));
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            let _ = lease.release(&meta);
            bail(e)
        }
    }
}

/// The independent check: the same bytes, in the same order, reconstructed through the
/// SEQUENTIAL document path (`checkpoint::materialize`) instead of the pooled range path.
/// Two code paths over one arithmetic; a disagreement is a real defect, not a flaky test.
pub fn cmd_verify_fill(root: &Path, hex: &str, f: &Flags) -> ExitCode {
    let (s, _, h) = ok!(load(root, hex));
    let order = match flag(f, "order") {
        Some("reverse") => {
            let mut v = construction_order(&h);
            v.reverse();
            v
        }
        _ => construction_order(&h),
    };
    let mut hasher = Sha256::new();
    let t0 = Instant::now();
    let mut n = 0u64;
    for (c, k) in &order {
        let t = h
            .components
            .iter()
            .find(|(x, _)| x == c)
            .and_then(|(_, ts)| ts.iter().find(|(y, _)| y == k))
            .map(|(_, t)| t)
            .unwrap();
        for (role, part) in &t.parts {
            let mut v = Vec::new();
            ok!(checkpoint::materialize(
                &s,
                &format!("{c}/{k}#{role}"),
                part,
                &mut v
            ));
            n += v.len() as u64;
            hasher.update(&v);
        }
    }
    println!("sequential: {n} B in {} ms", t0.elapsed().as_millis());
    println!("fill sha256:{}", sha256::hex(&hasher.finish()));
    ExitCode::SUCCESS
}

// ---------------------------------------------------------------- span

pub fn cmd_span(root: &Path, hex: &str, ck: [&str; 3], off: u64, len: u64) -> ExitCode {
    let (s, m, h) = ok!(load(root, hex));
    let meta = ok!(Meta::open(&s));
    let t = h
        .components
        .iter()
        .find(|(c, _)| c == ck[0])
        .and_then(|(_, ts)| ts.iter().find(|(k, _)| k == ck[1]));
    let t = match t {
        Some((_, t)) => t,
        None => {
            eprintln!("REFUSED MISSING_TENSOR: no {}/{}", ck[0], ck[1]);
            return ExitCode::FAILURE;
        }
    };
    let part = match t.parts.iter().find(|(r, _)| r == ck[2]) {
        Some((_, p)) => p,
        None => {
            eprintln!("REFUSED ROLE_SET_MISMATCH: no role {}", ck[2]);
            return ExitCode::FAILURE;
        }
    };
    let (lease, _) = ok!(read::acquire_cozytensors(&s, &meta, &m));
    let mut buf = vec![0u8; len as usize];
    let what = format!("{}/{}#{}", ck[0], ck[1], ck[2]);
    let r = read::read_part_into(&lease, &what, part, off, len, &mut buf);
    ok!(lease.release(&meta));
    ok!(r);
    println!("{what} [{off}, {}) -> {len} B", off + len);
    println!("sha256:     {}", sha256::hex_digest(&buf));
    ExitCode::SUCCESS
}

// ---------------------------------------------------------------- projection

pub fn cmd_checkout(root: &Path, hex: &str, dest: &Path, f: &Flags) -> ExitCode {
    let (s, m, _) = ok!(load(root, hex));
    let p = ok!(project::checkout(&s, &m, dest, flag_on(f, "no-symlink")));
    println!("dest:       {}", dest.display());
    println!(
        "checkout:   {} dirs, {} symlinks, {} copies, {} B",
        p.dirs, p.links, p.copies, p.bytes
    );
    println!(
        "cozytensors: {} validated header{}; payload segments remain CAS-only",
        p.cozytensors_headers,
        if p.cozytensors_headers == 1 { "" } else { "s" }
    );
    if let Some((path, Entry::CozyTensors(_))) = m
        .entries()
        .iter()
        .find(|(_, entry)| matches!(entry, Entry::CozyTensors(_)))
    {
        println!("inspect:    tfs cbor check {}", dest.join(path).display());
    }
    ExitCode::SUCCESS
}

pub fn cmd_materialize(root: &Path, manifest: &str, path: &str, dest: &Path) -> ExitCode {
    let s = ok!(Store::open(root));
    let hex = manifest.trim_start_matches("sha256:");
    let manifest_path = s.manifest_path(hex);
    let length = ok!(std::fs::metadata(&manifest_path)
        .map(|metadata| metadata.len())
        .map_err(|error| Refusal {
            code: Code::OBJECT_ABSENT,
            detail: format!("manifest sha256:{hex} is absent: {error}"),
        }));
    let m = ok!(s.read_manifest(&ObjectRef {
        sha256: hex.to_string(),
        length
    }));
    let blob = ok!(project::ordinary_file(&m, path));
    let t0 = Instant::now();
    let n = ok!(project::materialize(&s, blob, dest));
    println!(
        "materialized {n} B to {} in {} ms (O(one segment) resident, atomic)",
        dest.display(),
        t0.elapsed().as_millis()
    );
    ExitCode::SUCCESS
}

pub fn cmd_render(root: &Path, hex: &str) -> ExitCode {
    let s = ok!(Store::open(root));
    let bytes = match std::fs::read(s.object_path(hex)) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("REFUSED OBJECT_ABSENT: {e}");
            return ExitCode::FAILURE;
        }
    };
    let pretty = ok!(project::render(&bytes, 1 << 26));
    println!("{pretty}");
    println!("--- projection only ---");
    println!(
        "canonical:  sha256:{} ({} B)",
        sha256::hex_digest(&bytes),
        bytes.len()
    );
    println!(
        "pretty:     sha256:{} ({} B) — a different object, and no decoder admits it",
        sha256::hex_digest(pretty.as_bytes()),
        pretty.len()
    );
    ExitCode::SUCCESS
}

// ---------------------------------------------------------------- the transform

pub fn cmd_transform() -> ExitCode {
    println!("the ONE fill-transform implementation — decomposition is arithmetic, not a walk\n");
    let cases: [(&str, Xform, u64); 3] = [
        (
            // THE REAL GEOMETRY (proto-001 #322). This case used to carry unit_bytes 256 —
            // a 128-wide row — and printed `class gather`, which is where the whole
            // "the fill-transform rides the staging copy because a gather is expensive"
            // prior came from. The real `blocks.{i}.attn.qkv_proj.weight` is [21504, 5376]
            // BF16: the unit is 128 rows x 10,752 B = 1,376,256 B, 5,376x longer, and the
            // seam is squarely LONG-RUN.
            "h3 fused qkv, 56 head-major groups, [21504, 5376] bf16",
            Xform::QkvSplit {
                groups: 56,
                shares: [1, 1, 1],
                take: 0,
                unit_bytes: 128 * 5376 * 2,
            },
            1,
        ),
        (
            "contiguous 1x1-conv to_qkv (groups=1)",
            Xform::QkvSplit {
                groups: 1,
                shares: [1, 1, 1],
                take: 2,
                unit_bytes: 1 << 20,
            },
            1,
        ),
        (
            "per-element gather class (unit 2 B)",
            Xform::QkvSplit {
                groups: 8192,
                shares: [1, 1, 2],
                take: 1,
                unit_bytes: 2,
            },
            1,
        ),
    ];
    for (name, x, _) in &cases {
        let d = x.runs();
        println!(
            "{name}\n  runs {:6}  bytes {:9}  mean run {:8} B  class {}",
            d.runs.len(),
            d.bytes,
            d.mean_run(),
            d.class.as_str()
        );
        if d.class == Class::Gather {
            println!(
                "     gather-class: planned as real work (v1 measured 0.12 GiB/s), never free"
            );
        }
    }
    println!("\nblock-granularity boundary rule:");
    for (unit, block) in [(32u64, 32u64), (64, 32), (16, 32)] {
        match guard_block_granular("permute", unit, block) {
            Ok(()) => println!("  unit {unit:3} el over {block}-el blocks: fill-applicable"),
            Err(e) => println!(
                "  unit {unit:3} el over {block}-el blocks: REFUSED {}",
                e.code.as_str()
            ),
        }
    }
    ExitCode::SUCCESS
}

// ---------------------------------------------------------------- independent evidence

/// Reconstructed bytes against evidence the READ PATH did not produce: every delivered
/// buffer is hashed and compared to the digest the HEADER declares for that segment — a fact
/// computed at admission, from the source stream, by different code. This deliberately does
/// not consult the store's verification records: those are what the read path trusts, so
/// using them here would be checking a claim against itself.
pub fn cmd_prove(root: &Path, hex: &str, f: &Flags) -> ExitCode {
    let (s, m, h) = ok!(load(root, hex));
    let meta = ok!(Meta::open(&s));
    let sample: usize = flag_num(f, "sample", 64);
    let (lease, _) = ok!(read::acquire_cozytensors(&s, &meta, &m));

    let parts: Vec<(String, &tensorfs_core::header::Part)> = h
        .tensors()
        .flat_map(|(c, k, t)| {
            t.parts
                .iter()
                .map(move |(r, p)| (format!("{c}/{k}#{r}"), p))
        })
        .collect();
    let step = (parts.len() / sample.max(1)).max(1);
    let (mut checked, mut segs, mut bytes, mut bad) = (0usize, 0usize, 0u64, 0usize);
    for (what, part) in parts.iter().step_by(step) {
        checked += 1;
        for o in part.segments() {
            let mut buf = vec![0u8; o.length as usize];
            if let Err(e) = read::read_into(
                &lease,
                &ObjectRange {
                    obj: o.clone(),
                    off: 0,
                    len: o.length,
                },
                &mut buf,
            ) {
                let _ = lease.release(&meta);
                return bail(e);
            }
            let got = sha256::hex_digest(&buf);
            segs += 1;
            bytes += o.length;
            if got != o.sha256 {
                bad += 1;
                println!(
                    "  MISMATCH {what}: delivered sha256:{got}, header declares {}",
                    o.id()
                );
            }
        }
    }
    ok!(lease.release(&meta));
    println!(
        "sampled:    {checked} of {} parts (every {step}th)",
        parts.len()
    );
    println!("segments:   {segs}, {bytes} B rehashed from the DELIVERED buffers");
    println!(
        "identity:   {} — reconstructed bytes hash to the digest the header declares",
        if bad == 0 {
            "0 mismatches"
        } else {
            "MISMATCHES"
        }
    );
    if bad > 0 {
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

// ---------------------------------------------------------------- torn storage

/// The torn-read arm, planted into REAL bytes: verify the object, take the lease, THEN tear
/// the file underneath the reader. That is the only ordering that tests the read path — tear
/// it before the lease and admission-time verification catches it, which is a different
/// (already armed) claim.
///
/// Refuses outright on a store whose objects carry more than one link: this command mutates
/// object bytes, and a hardlinked working copy shares inodes with whatever it was copied
/// from. A destructive arm that can reach the quarry is not an arm, it is an accident.
/// `tfs read swap` — tfs-021's red arm: a SAME-LENGTH replacement installed at an object's
/// pathname AFTER acquire must never reach the reader. The lease pinned the verified
/// descriptor, so the swap is not caught — it is INEFFECTIVE, and the verified bytes are
/// what the read returns. (The pre-fix tree, which re-resolved the pathname per read with
/// only a size check, was observed serving the replaced bytes.)
pub fn cmd_swap(root: &Path, hex: &str) -> ExitCode {
    use std::os::unix::fs::MetadataExt;
    let (s, m, h) = ok!(load(root, hex));
    let meta = ok!(Meta::open(&s));
    let seg = match h
        .tensors()
        .flat_map(|(_, _, t)| t.parts.iter())
        .flat_map(|(_, p)| p.segments())
        .min_by_key(|o| o.length)
        .cloned()
    {
        Some(o) => o,
        None => {
            eprintln!("REFUSED MISSING_TENSOR: no segmented part to swap");
            return ExitCode::FAILURE;
        }
    };
    let path = s.object_path(&seg.sha256);
    let md = match std::fs::metadata(&path) {
        Ok(md) => md,
        Err(e) => {
            eprintln!("REFUSED IO_FAILED: {e}");
            return ExitCode::FAILURE;
        }
    };
    if md.nlink() > 1 {
        eprintln!(
            "REFUSED NOT_REGULAR_FILE: {} has {} links — this arm mutates bytes and will not \
             run against a store that shares inodes with another tree",
            seg.id(),
            md.nlink()
        );
        return ExitCode::FAILURE;
    }

    let (lease, _) = ok!(read::acquire_cozytensors(&s, &meta, &m));
    let keep = std::fs::read(&path).unwrap_or_default();

    // The swap, as an attacker would install it: same length, different bytes, atomic
    // rename over the pathname. Every fact a pathname-bound record checks (size, an mtime
    // the writer controls) can be made to agree.
    let mut fake = keep.clone();
    for b in fake.iter_mut() {
        *b ^= 0xff;
    }
    let tmp = path.with_extension("swap");
    let _ = std::fs::write(&tmp, &fake);
    let _ = std::fs::rename(&tmp, &path);

    let mut buf = vec![0u8; seg.length as usize];
    let r = read::read_into(
        &lease,
        &ObjectRange {
            obj: seg.clone(),
            off: 0,
            len: seg.length,
        },
        &mut buf,
    );

    // Put the verified bytes back before reporting, so a failed arm cannot leave a broken
    // store, then judge what the reader saw.
    let mut perms = std::fs::metadata(&path).unwrap().permissions();
    std::os::unix::fs::PermissionsExt::set_mode(&mut perms, 0o644);
    let _ = std::fs::set_permissions(&path, perms);
    let _ = std::fs::write(&path, &keep);
    let mut perms = std::fs::metadata(&path).unwrap().permissions();
    std::os::unix::fs::PermissionsExt::set_mode(&mut perms, 0o444);
    let _ = std::fs::set_permissions(&path, perms);
    ok!(lease.release(&meta));

    println!(
        "swapped object: {} ({} B, same-length replacement renamed over the path after \
         acquire)",
        seg.id(),
        seg.length
    );
    match r {
        Ok(()) if buf == keep => {
            println!("  FIRED  the reader returned the VERIFIED bytes — the swap never reached it");
            println!("         (the pinned descriptor is the only door; the pathname was not re-resolved)");
            ExitCode::SUCCESS
        }
        Ok(()) => {
            println!("  MISSED the reader served the REPLACED bytes — the v1 defect, live");
            ExitCode::FAILURE
        }
        Err(e) => {
            println!("  WRONG  the read refused ({e}) — the swap must be ineffective, not fatal");
            ExitCode::FAILURE
        }
    }
}

pub fn cmd_tear(root: &Path, hex: &str) -> ExitCode {
    use std::os::unix::fs::MetadataExt;
    let (s, m, h) = ok!(load(root, hex));
    let meta = ok!(Meta::open(&s));
    let seg = match h
        .tensors()
        .flat_map(|(_, _, t)| t.parts.iter())
        .flat_map(|(_, p)| p.segments())
        .min_by_key(|o| o.length)
        .cloned()
    {
        Some(o) => o,
        None => {
            eprintln!("REFUSED MISSING_TENSOR: no segmented part to tear");
            return ExitCode::FAILURE;
        }
    };
    let path = s.object_path(&seg.sha256);
    let md = match std::fs::metadata(&path) {
        Ok(md) => md,
        Err(e) => {
            eprintln!("REFUSED IO_FAILED: {e}");
            return ExitCode::FAILURE;
        }
    };
    if md.nlink() > 1 {
        eprintln!(
            "REFUSED NOT_REGULAR_FILE: {} has {} links — this arm mutates bytes and will not \
             run against a store that shares inodes with another tree",
            seg.id(),
            md.nlink()
        );
        return ExitCode::FAILURE;
    }

    let (lease, _) = ok!(read::acquire_cozytensors(&s, &meta, &m));
    let keep = std::fs::read(&path).unwrap_or_default();
    let mut perms = md.permissions();
    std::os::unix::fs::PermissionsExt::set_mode(&mut perms, 0o644);
    let _ = std::fs::set_permissions(&path, perms);
    let _ = std::fs::write(&path, &keep[..keep.len() - 1]);

    let mut buf = vec![0u8; seg.length as usize];
    let torn = read::read_into(
        &lease,
        &ObjectRange {
            obj: seg.clone(),
            off: 0,
            len: seg.length,
        },
        &mut buf,
    );
    let leaked = buf.iter().any(|b| *b != 0);

    // Put the bytes back before reporting, so a failed arm cannot leave a broken store.
    let _ = std::fs::write(&path, &keep);
    let mut perms = std::fs::metadata(&path).unwrap().permissions();
    std::os::unix::fs::PermissionsExt::set_mode(&mut perms, 0o444);
    let _ = std::fs::set_permissions(&path, perms);
    ok!(lease.release(&meta));

    println!(
        "torn object: {} ({} B, truncated by 1)",
        seg.id(),
        seg.length
    );
    match torn {
        Err(e) if e.code == Code::SHORT_READ => {
            println!("  FIRED  torn read: SHORT_READ — {}", e.detail);
            println!(
                "  buffer untouched: {} (no partial success is exposed)",
                if leaked { "NO — DEFECT" } else { "YES" }
            );
            if leaked {
                return ExitCode::FAILURE;
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            println!("  WRONG  wanted SHORT_READ, got {e}");
            ExitCode::FAILURE
        }
        Ok(()) => {
            println!("  MISSED torn read SUCCEEDED");
            ExitCode::FAILURE
        }
    }
}

// ---------------------------------------------------------------- red arms

struct Arms {
    fired: usize,
    missed: usize,
}

impl Arms {
    fn arm<T>(&mut self, name: &str, want: Code, r: Result<T, Refusal>) {
        match r {
            Err(e) if e.code == want => {
                self.fired += 1;
                println!("  FIRED  {name}: {}", e.code.as_str());
            }
            Err(e) => {
                self.missed += 1;
                println!("  WRONG  {name}: wanted {}, got {e}", want.as_str());
            }
            Ok(_) => {
                self.missed += 1;
                println!(
                    "  MISSED {name}: wanted {}, the call SUCCEEDED",
                    want.as_str()
                );
            }
        }
    }

    fn malformed_plan(&mut self, name: &str, lease: &read::ReadLease, plan: ReadPlan) {
        let mut destination = vec![0xa5; plan.bytes as usize];
        let before = destination.clone();
        match read::execute(lease, &plan, &mut destination, 4) {
            Err(e) if e.code == Code::RANGE_BOUNDS && destination == before => {
                self.fired += 1;
                println!("  FIRED  {name}: RANGE_BOUNDS, destination untouched");
            }
            Err(e) if e.code == Code::RANGE_BOUNDS => {
                self.missed += 1;
                println!("  WRONG  {name}: RANGE_BOUNDS after mutating the destination");
            }
            Err(e) => {
                self.missed += 1;
                println!("  WRONG  {name}: wanted RANGE_BOUNDS, got {e}");
            }
            Ok(_) => {
                self.missed += 1;
                println!("  MISSED {name}: malformed plan SUCCEEDED");
            }
        }
    }
}

fn inline_plan(items: &[(u64, &[u8])], bytes: u64) -> ReadPlan {
    ReadPlan {
        items: items
            .iter()
            .enumerate()
            .map(|(index, (dest_off, value))| PlanItem {
                what: format!("planted#{index}"),
                dest_off: *dest_off,
                len: value.len() as u64,
                source: ReadSource::Inline(value.to_vec()),
            })
            .collect(),
        bytes,
        io_bytes: 0,
        inline_parts: items.len(),
        order: Order::Destination,
    }
}

pub fn cmd_arms(root: &Path, hex: &str) -> ExitCode {
    let (s, m, h) = ok!(load(root, hex));
    let meta = ok!(Meta::open(&s));
    let mut a = Arms {
        fired: 0,
        missed: 0,
    };
    let (lease, _) = ok!(read::acquire_cozytensors(&s, &meta, &m));
    let first = lease.objects()[0].clone();

    println!("read-path red arms, planted live:");

    // 1. the caller's buffer must be exactly the range
    let mut small = vec![0u8; 8];
    a.arm(
        "buffer smaller than the range",
        Code::BUFFER_SIZE,
        read::read_into(
            &lease,
            &ObjectRange {
                obj: first.clone(),
                off: 0,
                len: 16,
            },
            &mut small,
        ),
    );

    // 2. a range past the DECLARED end refuses by arithmetic, before any byte moves
    let mut buf = vec![0u8; 16];
    a.arm(
        "range past the object's declared end",
        Code::RANGE_BOUNDS,
        read::read_into(
            &lease,
            &ObjectRange {
                obj: first.clone(),
                off: first.length,
                len: 16,
            },
            &mut buf,
        ),
    );

    // 3. an object the lease never verified
    a.arm(
        "object outside the lease",
        Code::LEASE_NOT_COVERED,
        read::read_into(
            &lease,
            &ObjectRange {
                obj: ObjectRef {
                    sha256: "0".repeat(64),
                    length: 16,
                },
                off: 0,
                len: 16,
            },
            &mut buf,
        ),
    );

    // 4. the SAME digest at a lying length — full-ObjectRef coverage, not digest coverage
    a.arm(
        "lease digest with a different declared length",
        Code::LEASE_NOT_COVERED,
        read::read_into(
            &lease,
            &ObjectRange {
                obj: ObjectRef {
                    sha256: first.sha256.clone(),
                    length: first.length + 1,
                },
                off: 0,
                len: 16,
            },
            &mut buf,
        ),
    );

    // 5. A public ReadPlan is validated as one exact destination partition before any
    // reader starts. Gap, overlap, trailing bytes and a span beyond the destination all
    // refuse typed and leave the caller's sentinel bytes untouched.
    a.malformed_plan(
        "construction-order destination gap",
        &lease,
        inline_plan(&[(0, b"ab"), (3, b"d")], 4),
    );
    a.malformed_plan(
        "construction-order destination overlap",
        &lease,
        inline_plan(&[(0, b"ab"), (1, b"cd")], 4),
    );
    a.malformed_plan(
        "construction-order trailing destination byte",
        &lease,
        inline_plan(&[(0, b"abc")], 4),
    );
    a.malformed_plan(
        "construction-order item past destination",
        &lease,
        inline_plan(&[(0, b"abcde")], 4),
    );
    a.malformed_plan(
        "construction-order item with unaddressable length",
        &lease,
        ReadPlan {
            items: vec![PlanItem {
                what: "planted-overflow".to_string(),
                dest_off: 0,
                len: u64::MAX,
                source: ReadSource::Inline(Vec::new()),
            }],
            bytes: 0,
            io_bytes: 0,
            inline_parts: 1,
            order: Order::Destination,
        },
    );

    let valid = inline_plan(&[(0, b"ab"), (2, b"cd")], 4);
    let mut valid_destination = vec![0u8; 4];
    match read::execute(&lease, &valid, &mut valid_destination, 4) {
        Ok(stats) if valid_destination == b"abcd" && stats.items == 2 => {
            println!("  GREEN  one issue cursor with concurrent disjoint completions covers abcd")
        }
        Ok(stats) => {
            a.missed += 1;
            println!(
                "  MISSED valid disjoint plan: destination {valid_destination:?}, items {}",
                stats.items
            );
        }
        Err(e) => {
            a.missed += 1;
            println!("  MISSED valid disjoint plan refused: {e}");
        }
    }

    // 6. a traversal that does not name every tensor
    let mut short = construction_order(&h);
    short.pop();
    a.arm(
        "traversal missing a tensor",
        Code::TRAVERSAL_INCOMPLETE,
        read::plan_for_traversal(&h, &short, &all_components(&h), 1 << 20),
    );

    // 5a-5d. COMPONENT-SCOPED COMPLETENESS (#570b). The scope moved from the manifest to
    // the declared component set; the hole it forbids did not move at all.
    let comps = all_components(&h);
    let one = comps[0].clone();
    let only_one = std::slice::from_ref(&one);
    let scoped: Vec<(String, String)> = construction_order(&h)
        .into_iter()
        .filter(|(c, _)| c == &one)
        .collect();

    // GREEN, and it is the whole point: 1-of-N components, complete for that one, admits.
    match read::plan_for_traversal(&h, &scoped, only_one, 1 << 20) {
        Ok(p) => println!(
            "  GREEN  a component-complete traversal of 1 of {} components admits: \
             {} item(s), {} B",
            comps.len(),
            p.items.len(),
            p.bytes
        ),
        Err(e) => {
            a.missed += 1;
            println!("  MISSED component-scoped plan should admit, got {e}");
        }
    }

    // A hole WITHIN a requested component still refuses — tfs-007's rule, unmoved.
    if scoped.len() > 1 {
        let mut holed = scoped.clone();
        holed.pop();
        a.arm(
            "a tensor missing WITHIN a requested component",
            Code::TRAVERSAL_INCOMPLETE,
            read::plan_for_traversal(&h, &holed, only_one, 1 << 20),
        );
    }

    // NO SMUGGLING: naming a tensor outside the declared scope refuses. Planted from a REAL
    // sibling component when the fixture has one, and from a synthetic name when it does
    // not — the scope check runs before the header lookup, so both reach it, and the arm
    // must exist on a single-component fixture too.
    let mut smuggled = scoped.clone();
    match construction_order(&h).into_iter().find(|(c, _)| c != &one) {
        Some(foreign) => smuggled.push(foreign),
        None => smuggled.push(("a_sibling_component".to_string(), scoped[0].1.clone())),
    }
    a.arm(
        "a tensor from a component the plan did not request",
        Code::TRAVERSAL_INCOMPLETE,
        read::plan_for_traversal(&h, &smuggled, only_one, 1 << 20),
    );

    // A requested component the header does not carry refuses rather than reading nothing.
    a.arm(
        "a requested component the header does not carry",
        Code::TRAVERSAL_INCOMPLETE,
        read::plan_for_traversal(&h, &scoped, &["not_a_component".to_string()], 1 << 20),
    );

    // An EMPTY declared scope is not "everything": it names nothing to be complete about.
    a.arm(
        "an empty requested component set",
        Code::TRAVERSAL_INCOMPLETE,
        read::plan_for_traversal(&h, &scoped, &[], 1 << 20),
    );

    // 6. element-level permute over a block-quantized carrier
    a.arm(
        "element-level permute over 32-element blocks",
        Code::PERMUTE_NOT_BLOCK_GRANULAR,
        guard_block_granular("qkv", 16, 32),
    );

    // 7. a pretty twin is a projection, never a stored document
    let hb = ok!(checkpoint::read_object(&s, m.header().unwrap(), 1 << 26));
    let pretty = ok!(project::render(&hb, 1 << 26));
    a.arm(
        "pretty header decoded as a document",
        Code::MALFORMED_CBOR,
        Header::parse(pretty.as_bytes()),
    );
    println!(
        "         canonical sha256:{} vs pretty sha256:{} — different objects",
        &sha256::hex_digest(&hb)[..12],
        &sha256::hex_digest(pretty.as_bytes())[..12]
    );

    ok!(lease.release(&meta));

    // 8. materializing a carrier the store never retained
    let fake = ObjectRef::of(b"a carrier this store has never seen");
    a.arm(
        "materialize a non-retained carrier",
        Code::OBJECT_ABSENT,
        project::materialize(&s, &fake, &std::env::temp_dir().join("tfs-arm-blob")),
    );

    println!("\n{} fired, {} missed", a.fired, a.missed);
    if a.missed > 0 {
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}
