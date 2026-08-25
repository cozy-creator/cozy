//! `tfs snapshot` — the live snapshot plane (tfs-004).
//!
//! The transitive root is walked against a REAL store: SnapshotManifest → CozyTensorsHeader →
//! contained documents → tensor rules → storage-part object runs. "Every and only" is proved
//! by set-differencing the walk against the store's own object census, not asserted.

use std::io::Write;
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use tensorfs_core::checkpoint as ck;
use tensorfs_core::ids::{Doc, ObjectRef};
use tensorfs_core::manifest::{BlobDescriptor, Entry, SnapshotManifest};
use tensorfs_core::sha256;
use tensorfs_core::store::Store;

use crate::{bail, flag_num, rss_kb, Flags};

fn opened(root: &Path, hex: &str) -> Result<(Store, ObjectRef), ExitCode> {
    let store = match Store::open(root) {
        Ok(s) => s,
        Err(e) => return Err(bail(e)),
    };
    let len = match std::fs::metadata(store.object_path(hex)) {
        Ok(m) => m.len(),
        Err(e) => {
            eprintln!("REFUSED OBJECT_ABSENT: {hex}: {e}");
            return Err(ExitCode::FAILURE);
        }
    };
    Ok((
        store,
        ObjectRef {
            sha256: hex.to_string(),
            length: len,
        },
    ))
}

macro_rules! got {
    ($e:expr) => {
        match $e {
            Ok(v) => v,
            Err(c) => return c,
        }
    };
}

macro_rules! ok {
    ($e:expr) => {
        match $e {
            Ok(v) => v,
            Err(e) => return bail(e),
        }
    };
}

// ------------------------------------------------------------------ show

pub fn cmd_show(root: &Path, hex: &str) -> ExitCode {
    let (store, sref) = got!(opened(root, hex));
    let bytes = ok!(ck::read_object(&store, &sref, 1 << 26));
    let t = Instant::now();
    let snap = ok!(SnapshotManifest::parse(&bytes));
    let parse_us = t.elapsed().as_secs_f64() * 1e6;

    // The codec's OWN acceptance: decode → re-encode must be the same bytes. `Doc::parse`
    // already enforces it; printing it makes the claim observable rather than implied.
    let re = snap.canonical_bytes();
    println!("snapshot {} ({} B)", sref.id(), sref.length);
    println!(
        "  decode → re-encode  {} ({} B in, {} B out), strict parse {parse_us:.0} µs",
        if re == bytes {
            "BYTE-IDENTICAL"
        } else {
            "DIVERGED"
        },
        bytes.len(),
        re.len()
    );
    println!("  snapshot_id     {}", snap.snapshot_id());
    match snap.checkpoint_id() {
        Ok(c) => println!("  checkpoint_id   {c}   (= snapshot_id, unconditional)"),
        Err(e) => println!("  checkpoint_id   —    {e}"),
    }
    if let Some(h) = &snap.header {
        println!("  attachment      {} (TYPED, not by pathname)", h.id());
    }
    println!("  {} entries", snap.entries.len());
    for (p, e) in &snap.entries {
        match e {
            Entry::Dir => println!("    dir   {p}/"),
            Entry::File(o) => println!("    file  {p}  {} ({} B)", o.id(), o.length),
            Entry::Blob(b) => println!(
                "    blob  {p}  sha256:{} ({} B, {} segments, {})",
                b.whole_sha256,
                b.logical_size,
                b.segments.len(),
                tensorfs_core::manifest::SEGMENT_PROFILE
            ),
        }
    }
    if re != bytes {
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

// ------------------------------------------------------------------ walk

pub fn cmd_walk(root: &Path, hex: &str) -> ExitCode {
    let (store, sref) = got!(opened(root, hex));
    let snap = ok!(ck::load_manifest(&store, &sref));

    let t = Instant::now();
    let w = ok!(ck::walk(&store, &snap));
    let walk_ms = t.elapsed().as_secs_f64() * 1000.0;

    println!("transitive root of {}", sref.id());
    println!(
        "  manifest → header → {} spec + {} vectors + {} config → {} tensors / {} parts → {} part runs",
        w.count("spec"),
        w.count("vectors"),
        w.count("config"),
        w.tensors,
        w.parts,
        w.count("part")
    );
    let distinct = w.distinct().len();
    println!(
        "  reached {} references, {distinct} distinct objects, {:.1} MiB declared, in {walk_ms:.1} ms",
        w.objects.len(),
        w.bytes() as f64 / (1024.0 * 1024.0)
    );
    if !w.siblings.is_empty() {
        println!("  sibling assets ({}): {:?}", w.siblings.len(), w.siblings);
    }

    // EVERY and ONLY. The snapshot object itself is a root, not something a root reaches.
    let census = ok!(store.objects());
    let mut reached: Vec<String> = w.distinct().iter().map(|o| o.sha256.clone()).collect();
    reached.push(sref.sha256.clone());
    reached.sort();
    reached.dedup();
    let unreached: Vec<&String> = census.iter().filter(|h| !reached.contains(h)).collect();
    let missing: Vec<&String> = reached.iter().filter(|h| !census.contains(h)).collect();
    println!(
        "  store census {} objects · reached+snapshot {} · unreached {} · reached-but-absent {}",
        census.len(),
        reached.len(),
        unreached.len(),
        missing.len()
    );
    for h in unreached.iter().take(5) {
        println!("    UNREACHED sha256:{h}");
    }
    for h in missing.iter().take(5) {
        println!("    ABSENT    sha256:{h}");
    }
    println!("  peak RSS {:.1} MiB", rss_kb("VmHWM") as f64 / 1024.0);
    if !unreached.is_empty() || !missing.is_empty() {
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

// ------------------------------------------------------------------ verify

struct Sink(u64);
impl Write for Sink {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        self.0 += b.len() as u64;
        Ok(b.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// The full-checkpoint verify walk: every declared byte of every part and every blob is
/// fetched through the real store and hashed against the record that promised it.
pub fn cmd_verify(root: &Path, hex: &str) -> ExitCode {
    let (store, sref) = got!(opened(root, hex));
    let snap = ok!(ck::load_manifest(&store, &sref));
    let w = ok!(ck::walk(&store, &snap));
    let header = ok!(ck::load_header(&store, snap.header.as_ref().unwrap()));

    let t = Instant::now();
    let (mut bytes, mut parts, mut blobs) = (0u64, 0usize, 0usize);
    for (comp, key, tensor) in header.tensors() {
        for (role, part) in &tensor.parts {
            let what = format!("{comp}/{key}#{role}");
            let mut sink = Sink(0);
            ok!(ck::materialize(&store, &what, part, &mut sink));
            bytes += sink.0;
            parts += 1;
        }
    }
    for (path, e) in &snap.entries {
        match e {
            Entry::Blob(b) => {
                let mut sink = Sink(0);
                ok!(ck::read_blob(&store, path, b, &mut sink));
                bytes += sink.0;
                blobs += 1;
            }
            Entry::File(o) => {
                let mut sink = Sink(0);
                let n = ok!(store.read_into(&o.sha256, &mut sink));
                bytes += n;
            }
            Entry::Dir => {}
        }
    }
    let secs = t.elapsed().as_secs_f64();
    let mb = bytes as f64 / (1024.0 * 1024.0);
    println!("full-checkpoint verify walk over {}", sref.id());
    println!(
        "  {} tensors, {parts} parts, {blobs} blobs, {} distinct objects",
        w.tensors,
        w.distinct().len()
    );
    println!(
        "  {bytes} B ({mb:.1} MiB) verified in {secs:.2}s = {:.1} MiB/s",
        mb / secs
    );
    println!("  peak RSS {:.1} MiB", rss_kb("VmHWM") as f64 / 1024.0);
    ExitCode::SUCCESS
}

// ------------------------------------------------------------------ blobs

/// The deterministic descriptor of a real file, derived in ONE streaming pass: boundaries
/// are profile-fixed, so this never depends on the store, the reader, or the machine.
fn describe(
    src: &Path,
    store: Option<&Store>,
) -> tensorfs_core::err::Result<(BlobDescriptor, u64)> {
    let mut f = std::fs::File::open(src).map_err(|e| tensorfs_core::err::Refusal {
        code: tensorfs_core::err::Code::IO_FAILED,
        detail: format!("{}: {e}", src.display()),
    })?;
    let mut whole = sha256::Sha256::new();
    let mut segments = Vec::new();
    let mut logical_size = 0u64;
    let mut admitted = 0u64;
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let mut seg = sha256::Sha256::new();
        let mut got = 0u64;
        while got < GRID {
            let want = ((GRID - got) as usize).min(buf.len());
            let n = std::io::Read::read(&mut f, &mut buf[..want]).map_err(|e| {
                tensorfs_core::err::Refusal {
                    code: tensorfs_core::err::Code::IO_FAILED,
                    detail: format!("{}: {e}", src.display()),
                }
            })?;
            if n == 0 {
                break;
            }
            whole.update(&buf[..n]);
            seg.update(&buf[..n]);
            got += n as u64;
        }
        if got == 0 {
            break;
        }
        let r = ObjectRef {
            sha256: sha256::hex(&seg.finish()),
            length: got,
        };
        if let Some(st) = store {
            let at = logical_size;
            let mut src2 = std::fs::File::open(src).map_err(|e| tensorfs_core::err::Refusal {
                code: tensorfs_core::err::Code::IO_FAILED,
                detail: format!("{}: {e}", src.display()),
            })?;
            std::io::Seek::seek(&mut src2, std::io::SeekFrom::Start(at)).ok();
            let put = st.put_stream(
                &mut std::io::Read::take(src2, got),
                Some(&r),
                &Default::default(),
            )?;
            if put.admitted {
                admitted += got;
            }
        }
        logical_size += got;
        segments.push(r);
        if got < GRID {
            break;
        }
    }
    Ok((
        BlobDescriptor {
            logical_size,
            whole_sha256: sha256::hex(&whole.finish()),
            segments,
        },
        admitted,
    ))
}

const GRID: u64 = 64 * 1024 * 1024;

/// Admit one real file as a blob.
pub fn cmd_blob_put(root: &Path, src: &Path) -> ExitCode {
    let store = ok!(Store::init(root));
    let t = Instant::now();
    let (d, admitted) = ok!(describe(src, Some(&store)));
    let secs = t.elapsed().as_secs_f64();
    let mb = d.logical_size as f64 / (1024.0 * 1024.0);
    println!("blob {}", src.display());
    println!("  whole_sha256   sha256:{}", d.whole_sha256);
    println!("  logical_size   {} B ({mb:.1} MiB)", d.logical_size);
    println!(
        "  segments       {} on {} ({} B admitted, {} B already present) in {secs:.2}s = {:.1} MiB/s",
        d.segments.len(),
        tensorfs_core::manifest::SEGMENT_PROFILE,
        admitted,
        d.logical_size - admitted,
        mb / secs
    );
    for (i, s) in d.segments.iter().enumerate() {
        println!("    [{i}] sha256:{} {} B", s.sha256, s.length);
    }
    println!("  peak RSS {:.1} MiB", rss_kb("VmHWM") as f64 / 1024.0);
    ExitCode::SUCCESS
}

/// Read a blob back through a descriptor RE-DERIVED from the source file, so verification
/// compares the store against the file rather than against itself.
pub fn cmd_blob_read(root: &Path, src: &Path, f: &Flags) -> ExitCode {
    let store = ok!(Store::open(root));
    let (mut d, _) = ok!(describe(src, None));
    // --whole plants a lie IN THE DESCRIPTOR rather than in the store: every segment digest
    // still verifies, so only the whole digest can catch it. That is the whole digest's job.
    if let Some(h) = crate::flag(f, "whole") {
        d.whole_sha256 = h.to_string();
    }
    let mut sink = Sink(0);
    let t = Instant::now();
    ok!(ck::read_blob(&store, "blob", &d, &mut sink));
    let secs = t.elapsed().as_secs_f64();
    let mb = sink.0 as f64 / (1024.0 * 1024.0);
    println!(
        "blob verified: {} B over {} segments in {secs:.2}s = {:.1} MiB/s (whole sha256:{})",
        sink.0,
        d.segments.len(),
        mb / secs,
        d.whole_sha256
    );
    println!("  peak RSS {:.1} MiB", rss_kb("VmHWM") as f64 / 1024.0);
    ExitCode::SUCCESS
}

/// Repair exactly ONE segment — the whole reason a large file is not one object.
pub fn cmd_blob_repair(root: &Path, src: &Path, index: usize) -> ExitCode {
    let store = ok!(Store::open(root));
    let (d, _) = ok!(describe(src, None));
    if index >= d.segments.len() {
        eprintln!("no segment {index} in a {}-segment blob", d.segments.len());
        return ExitCode::FAILURE;
    }
    let at: u64 = d.segments[..index].iter().map(|s| s.length).sum();
    let s = &d.segments[index];
    let mut f = ok!(
        std::fs::File::open(src).map_err(|e| tensorfs_core::err::Refusal {
            code: tensorfs_core::err::Code::IO_FAILED,
            detail: format!("{}: {e}", src.display()),
        })
    );
    std::io::Seek::seek(&mut f, std::io::SeekFrom::Start(at)).ok();
    let t = Instant::now();
    let put = ok!(store.put_stream(
        &mut std::io::Read::take(f, s.length),
        Some(s),
        &Default::default()
    ));
    let secs = t.elapsed().as_secs_f64();
    println!(
        "segment {index}/{} at +{at} ({} B, sha256:{}) {} in {secs:.2}s — {:.3}% of the {} B blob moved",
        d.segments.len(),
        s.length,
        s.sha256,
        if put.admitted { "REPAIRED" } else { "already present" },
        s.length as f64 * 100.0 / d.logical_size as f64,
        d.logical_size
    );
    println!("  peak RSS {:.1} MiB", rss_kb("VmHWM") as f64 / 1024.0);
    ExitCode::SUCCESS
}

// ------------------------------------------------------------------ fuzz

/// Bounded mutation fuzz over real canonical snapshot bytes. Not a search for crashes we
/// hope do not exist — a RESOURCE claim: every outcome is a decode or a typed refusal,
/// within a fixed time and memory envelope, and never a panic or an allocation blow-up.
pub fn cmd_fuzz(root: &Path, hex: &str, iters: u64) -> ExitCode {
    let (store, sref) = got!(opened(root, hex));
    let base = ok!(ck::read_object(&store, &sref, 1 << 26));

    let mut s = u64::from_le_bytes(sha256::digest(&base)[0..8].try_into().unwrap());
    let mut next = move || {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };

    let mut codes: Vec<(String, u64)> = Vec::new();
    let mut accepted = 0u64;
    let mut identical = 0u64;
    let t = Instant::now();
    for _ in 0..iters {
        let mut m = base.clone();
        // one of: flip a byte, truncate, splice a run, duplicate a run
        match next() % 4 {
            0 => {
                let i = (next() as usize) % m.len();
                m[i] ^= 1 << (next() % 8);
            }
            1 => {
                let i = (next() as usize) % m.len();
                m.truncate(i);
            }
            2 => {
                let i = (next() as usize) % m.len();
                let n = ((next() as usize) % 64).min(m.len() - i);
                m.splice(i..i + n, std::iter::empty());
            }
            _ => {
                let i = (next() as usize) % m.len();
                let n = ((next() as usize) % 64).min(m.len() - i);
                let run: Vec<u8> = m[i..i + n].to_vec();
                m.splice(i..i, run);
            }
        }
        if m == base {
            identical += 1;
            continue;
        }
        match SnapshotManifest::parse(&m) {
            Ok(d) => {
                accepted += 1;
                // A mutant that DECODES must still be a different document: the identity
                // law, not a refusal code, is what fuzzing may not break.
                if d.snapshot_id() == snap_id(&base) {
                    eprintln!("IDENTITY BREAK: a mutant decoded to the base snapshot id");
                    return ExitCode::FAILURE;
                }
            }
            Err(e) => {
                let k = e.code.as_str().to_string();
                match codes.iter_mut().find(|(c, _)| *c == k) {
                    Some((_, n)) => *n += 1,
                    None => codes.push((k, 1)),
                }
            }
        }
    }
    let secs = t.elapsed().as_secs_f64();
    codes.sort_by(|a, b| b.1.cmp(&a.1));
    println!(
        "fuzz: {iters} mutants of a {} B canonical snapshot in {secs:.2}s ({:.0} µs each)",
        base.len(),
        secs * 1e6 / iters as f64
    );
    println!(
        "  {accepted} decoded as a DIFFERENT document, {identical} no-op mutations, {} refused",
        iters - accepted - identical
    );
    for (c, n) in &codes {
        println!("    {n:6}  {c}");
    }
    println!(
        "  peak RSS {:.1} MiB · zero panics, zero hangs, zero identity collisions",
        rss_kb("VmHWM") as f64 / 1024.0
    );
    ExitCode::SUCCESS
}

/// Swap in a different header object: both the TYPED attachment and the one path that
/// projects it move together, because they are the same object seen two ways.
fn reheader(snap: &SnapshotManifest, href: &ObjectRef) -> SnapshotManifest {
    let mut m = snap.clone();
    let old = m.header.clone();
    for (_, e) in m.entries.iter_mut() {
        if matches!(e, Entry::File(o) if Some(&*o) == old.as_ref()) {
            *e = Entry::File(href.clone());
        }
    }
    m.header = Some(href.clone());
    m
}

fn snap_id(bytes: &[u8]) -> String {
    tensorfs_core::ids::object_id(bytes)
}

// ------------------------------------------------------------------ identity distinctions

/// The identity matrix the snapshot plane owns: what moves `checkpoint_id`, what moves
/// `header_digest`, and what moves neither. Every row is computed from live documents.
pub fn cmd_flip(root: &Path, hex: &str, f: &Flags) -> ExitCode {
    let _ = flag_num::<u64>(f, "blocks", 12);
    let (store, sref) = got!(opened(root, hex));
    let snap = ok!(ck::load_manifest(&store, &sref));
    let href = snap.header.clone().unwrap();
    let header = ok!(ck::load_header(&store, &href));

    let base_ck = ok!(snap.checkpoint_id());
    let base_h = href.id();
    let base_topo = header.topology_digest();

    let row = |what: &str, h: &str, c: &str, t: &str| {
        println!(
            "  {:<38} header {}  checkpoint {}  topology {}",
            what,
            if h == base_h { "same " } else { "MOVED" },
            if c == base_ck { "same " } else { "MOVED" },
            if t == base_topo { "same " } else { "MOVED" },
        );
    };

    println!("identity matrix for {}", sref.id());
    println!("  base header {base_h}");
    println!("  base checkpoint {base_ck}");

    // a sibling asset appears: the tree changed, the header did not
    let mut m = snap.clone();
    m.entries.push((
        "NOTICE.txt".to_string(),
        Entry::File(ObjectRef::of(b"a sibling asset with a named reason")),
    ));
    m.entries.sort_by(|a, b| a.0.cmp(&b.0));
    ok!(m.validate());
    row(
        "sibling asset added",
        &base_h,
        &ok!(m.checkpoint_id()),
        &base_topo,
    );

    // a config object changes: the header COMMITS the ConfigRef, so both ids move
    let mut h2 = header.clone();
    if let Some((_, c)) = h2.configs.first_mut() {
        c.sha256 = sha256::hex_digest(b"a different config object");
    }
    let h2ref = h2.object_ref();
    let m2 = reheader(&snap, &h2ref);
    ok!(m2.validate());
    row(
        "config object changed",
        &h2ref.id(),
        &ok!(m2.checkpoint_id()),
        &h2.topology_digest(),
    );

    // a referenced EncodingSpec object changes: same route, same result
    let mut h3 = header.clone();
    if let Some(e) = h3.encodings.first_mut() {
        e.sha256 = sha256::hex_digest(b"a different EncodingSpec object");
    }
    let h3ref = h3.object_ref();
    let m3 = reheader(&snap, &h3ref);
    ok!(m3.validate());
    row(
        "EncodingSpec object changed",
        &h3ref.id(),
        &ok!(m3.checkpoint_id()),
        &h3.topology_digest(),
    );

    // a logical dtype changes: topology moves too — the projection is not a digest of bytes
    let mut h4 = header.clone();
    if let Some((_, ts)) = h4.components.first_mut() {
        if let Some((_, t)) = ts.first_mut() {
            t.dtype = tensorfs_core::dtype::Dtype::F16;
        }
    }
    let h4ref = h4.object_ref();
    let m4 = reheader(&snap, &h4ref);
    ok!(m4.validate());
    row(
        "logical dtype changed",
        &h4ref.id(),
        &ok!(m4.checkpoint_id()),
        &h4.topology_digest(),
    );

    // the SAME header under a different sibling set — the distinguishing case the design
    // names explicitly: identical header_digest, distinct checkpoint_id
    println!(
        "  identical header_digest, distinct checkpoint_id: {}",
        ok!(m.checkpoint_id()) != base_ck
    );

    // and the pretty projection is NOT an identity: rendering changes nothing
    let pretty = tensorfs_core::canon::pretty(&snap.to_value(), 0);
    println!(
        "  pretty projection is {} B and hashes to sha256:{} — never consulted for identity",
        pretty.len(),
        &sha256::hex_digest(pretty.as_bytes())[..12]
    );
    ExitCode::SUCCESS
}
