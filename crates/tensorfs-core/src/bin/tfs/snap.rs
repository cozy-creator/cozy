//! `tfs snapshot` — the live snapshot plane (tfs-004).
//!
//! The transitive root is walked against a REAL store: SnapshotManifest → CozyTensorsHeader →
//! contained documents → tensor rules → storage-part object runs. "Every and only" is proved
//! by set-differencing the walk against the store's own object census, not asserted.

use std::collections::BTreeMap;
use std::io::{Cursor, Write};
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use tensorfs_core::canon::{self, as_arr, as_str};
use tensorfs_core::checkpoint as ck;
use tensorfs_core::err::{refuse, Code, Refusal, Result as TfsResult};
use tensorfs_core::header::{Body, Header};
use tensorfs_core::ids::{Doc, ObjectRef};
use tensorfs_core::manifest::{BlobDescriptor, Draft, Entry, SnapshotManifest};
use tensorfs_core::sha256;
use tensorfs_core::store::{Fault, Store};

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

// ------------------------------------------------------------------ metadata-only reproduction

fn read_input(path: &Path, what: &str) -> TfsResult<Vec<u8>> {
    std::fs::read(path).map_err(|error| Refusal {
        code: Code::IO_FAILED,
        detail: format!("read {what} {}: {error}", path.display()),
    })
}

fn parse_entries(value: &canon::Value, header: &ObjectRef) -> TfsResult<Draft> {
    let rows = as_arr("checkpoint reproduction", "entries", value)?;
    let mut draft = Draft {
        header: Some(header.clone()),
        entries: Vec::with_capacity(rows.len()),
    };
    let mut found_header = false;
    for (index, row) in rows.iter().enumerate() {
        let values = as_arr("checkpoint reproduction", "entry", row)?;
        if values.len() < 2 {
            return refuse(
                Code::WRONG_TYPE,
                format!("checkpoint reproduction entry {index} has fewer than 2 fields"),
            );
        }
        let path = as_str("checkpoint reproduction", "path", &values[0])?.to_string();
        let kind = as_str("checkpoint reproduction", "kind", &values[1])?;
        let entry = match (kind, values.len()) {
            ("dir", 2) => Entry::Dir,
            ("header", 2) => {
                if found_header {
                    return refuse(
                        Code::ATTACHMENT_CARDINALITY,
                        "checkpoint reproduction lists more than one header entry",
                    );
                }
                found_header = true;
                Entry::File(header.clone())
            }
            ("file", 3) => Entry::File(ObjectRef::from_value(
                "checkpoint reproduction file",
                &values[2],
            )?),
            _ => {
                return refuse(
                    Code::WRONG_TYPE,
                    format!(
                        "checkpoint reproduction entry {index} kind {kind:?} has arity {}; expected dir/header=2 or file=3",
                        values.len()
                    ),
                )
            }
        };
        draft.entries.push((path, entry));
    }
    if !found_header {
        return refuse(
            Code::ATTACHMENT_CARDINALITY,
            "checkpoint reproduction lists no header entry",
        );
    }
    Ok(draft)
}

fn add_ref(refs: &mut BTreeMap<String, u64>, object: &ObjectRef, what: &str) -> TfsResult<()> {
    if let Some(length) = refs.insert(object.sha256.clone(), object.length) {
        if length != object.length {
            return refuse(
                Code::LENGTH_MISMATCH,
                format!(
                    "{what}: sha256:{} appears at both {length} and {} bytes",
                    object.sha256, object.length
                ),
            );
        }
    }
    Ok(())
}

fn referenced_objects(header: &Header, draft: &Draft) -> TfsResult<BTreeMap<String, u64>> {
    let mut refs = BTreeMap::new();
    for (_, config) in &header.configs {
        add_ref(
            &mut refs,
            &ObjectRef {
                sha256: config.sha256.clone(),
                length: config.length,
            },
            "config",
        )?;
    }
    for (_, asset) in &header.assets {
        for segment in &asset.segments {
            add_ref(&mut refs, segment, "asset segment")?;
        }
    }
    for encoding in &header.encodings {
        add_ref(&mut refs, encoding, "encoding")?;
    }
    for (_, tensors) in &header.components {
        for (_, tensor) in tensors {
            for (_, part) in &tensor.parts {
                if let Body::Segments(segments) = &part.body {
                    for segment in segments {
                        add_ref(&mut refs, segment, "tensor segment")?;
                    }
                }
            }
        }
    }
    for (_, entry) in &draft.entries {
        if let Entry::File(object) = entry {
            if object != draft.header.as_ref().expect("header was set") {
                add_ref(&mut refs, object, "manifest file")?;
            }
        }
    }
    Ok(refs)
}

fn require_records(store: &Store, refs: &BTreeMap<String, u64>) -> TfsResult<()> {
    for (sha256, length) in refs {
        match store.record_valid(sha256) {
            Ok(record) if record.length == *length => {}
            Ok(record) => {
                return refuse(
                    Code::LENGTH_MISMATCH,
                    format!(
                        "sha256:{sha256} has a current {}-byte verification record, reproduction declares {length}",
                        record.length
                    ),
                )
            }
            Err(why) => {
                return refuse(
                    Code::OBJECT_CORRUPT,
                    format!(
                        "sha256:{sha256} has no current exact verification record ({why}); metadata reproduction never rehashes inherited payload",
                    ),
                )
            }
        }
    }
    Ok(())
}

fn put_document(store: &Store, bytes: &[u8]) -> TfsResult<ObjectRef> {
    let want = ObjectRef::of(bytes);
    let mut input = Cursor::new(bytes);
    let put = store.put_stream(&mut input, Some(&want), &Fault::default())?;
    if !put.admitted && store.record_valid(&want.sha256).is_err() {
        store.verify(&want.sha256)?;
    }
    match store.record_valid(&want.sha256) {
        Ok(record) if record.length == want.length => Ok(want),
        Ok(record) => refuse(
            Code::LENGTH_MISMATCH,
            format!(
                "new document {} has a {}-byte record, expected {}",
                want.id(),
                record.length,
                want.length
            ),
        ),
        Err(why) => refuse(
            Code::OBJECT_CORRUPT,
            format!(
                "new document {} has no current verification record ({why})",
                want.id()
            ),
        ),
    }
}

/// Reproduce canonical metadata over exact already-verified payload references. The diagnostic
/// inputs are never stored; only Header/SnapshotManifest canonical bytes can leave this operation.
pub fn cmd_reproduce(
    root: &Path,
    header_path: &Path,
    entries_path: &Path,
    flags: &Flags,
) -> ExitCode {
    let store = ok!(Store::open(root));
    let header_value = ok!(canon::parse_with_depth(
        &ok!(read_input(header_path, "header diagnostic")),
        tensorfs_core::limits::DOC_MAX_BYTES,
        16,
    ));
    let header = ok!(Header::from_diagnostic_value(&header_value));
    ok!(header.validate_structure());

    let expected_order = match crate::ingest::construction_order(flags) {
        Ok(order) => order,
        Err(code) => return code,
    };
    let actual_order: Vec<(String, String)> = header
        .components
        .iter()
        .flat_map(|(component, tensors)| {
            tensors
                .iter()
                .map(move |(key, _)| (component.clone(), key.clone()))
        })
        .collect();
    if actual_order != expected_order {
        let mismatch = actual_order
            .iter()
            .zip(&expected_order)
            .position(|(actual, expected)| actual != expected)
            .unwrap_or_else(|| actual_order.len().min(expected_order.len()));
        return bail(Refusal {
            code: Code::TRAVERSAL_ORDER_MISMATCH,
            detail: format!(
                "header traversal and construction order first disagree at {mismatch}; header has {} rows, order has {}",
                actual_order.len(),
                expected_order.len()
            ),
        });
    }

    let header_bytes = ok!(header.canonical_bytes());
    let header_ref = ObjectRef::of(&header_bytes);
    let entries_value = ok!(canon::parse(
        &ok!(read_input(entries_path, "manifest entries")),
        tensorfs_core::limits::DOC_MAX_BYTES,
    ));
    let draft = ok!(parse_entries(&entries_value, &header_ref));
    let refs = ok!(referenced_objects(&header, &draft));
    ok!(require_records(&store, &refs));
    // Semantic closure reads happen only after every inherited object proves a current record;
    // this operation may read bounded specs/configs, but never repairs trust by rehashing them.
    let closure = ok!(ck::load_closure(&store, &header));
    ok!(header.validate(&closure));
    let manifest = ok!(draft.seal());

    let stored_header = ok!(put_document(&store, &header_bytes));
    if stored_header != header_ref {
        return bail(Refusal {
            code: Code::OBJECT_ID_MISMATCH,
            detail: "stored header identity changed after canonical encoding".into(),
        });
    }
    let manifest_bytes = manifest.canonical_bytes();
    let snapshot = ok!(put_document(&store, &manifest_bytes));
    let strict = ok!(ck::load_manifest(&store, &snapshot));
    let walk = ok!(ck::walk(&store, &strict));
    ok!(walk.require_resident(&store));
    let all: BTreeMap<String, u64> = walk
        .distinct()
        .into_iter()
        .map(|object| (object.sha256.clone(), object.length))
        .collect();
    ok!(require_records(&store, &all));

    let inherited_bytes = refs.values().try_fold(0u64, |total, length| {
        total.checked_add(*length).ok_or_else(|| Refusal {
            code: Code::ARITH_OVERFLOW,
            detail: "inherited byte count overflow".into(),
        })
    });
    println!(
        "header       {} length={}",
        header_ref.id(),
        header_ref.length
    );
    println!("snapshot     {} length={}", snapshot.id(), snapshot.length);
    println!("tensors      {}", walk.tensors);
    println!(
        "inherited    {} objects {} bytes",
        refs.len(),
        ok!(inherited_bytes)
    );
    println!("tensor_payload_read 0");
    println!("tensor_payload_hash 0");
    ExitCode::SUCCESS
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
    if let Some(h) = snap.header() {
        println!("  attachment      {} (TYPED, not by pathname)", h.id());
    }
    println!("  {} entries", snap.entries().len());
    for (p, e) in snap.entries() {
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
        "  manifest → header → {} spec + {} vectors + {} config + {} asset segments → {} tensors / {} parts → {} part runs",
        w.count("spec"),
        w.count("vectors"),
        w.count("config"),
        w.count("model_asset"),
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
    let header = ok!(ck::load_header(&store, snap.header().unwrap()));

    let t = Instant::now();
    let (mut bytes, mut parts, mut blobs, mut assets) = (0u64, 0usize, 0usize, 0usize);
    for (comp, key, tensor) in header.tensors() {
        for (role, part) in &tensor.parts {
            let what = format!("{comp}/{key}#{role}");
            let mut sink = Sink(0);
            ok!(ck::materialize(&store, &what, part, &mut sink));
            bytes += sink.0;
            parts += 1;
        }
    }
    for (name, asset) in &header.assets {
        let mut sink = Sink(0);
        ok!(ck::read_asset(&store, name, asset, &mut sink));
        bytes += sink.0;
        assets += 1;
    }
    for (path, e) in snap.entries() {
        match e {
            Entry::Blob(b) => {
                let mut sink = Sink(0);
                ok!(ck::read_blob(&store, path, b, &mut sink));
                bytes += sink.0;
                blobs += 1;
            }
            Entry::File(o) => {
                if path.starts_with(&format!("{}/", tensorfs_core::manifest::ASSET_DIR)) {
                    continue;
                }
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
        "  {} tensors, {parts} parts, {assets} assets, {blobs} blobs, {} distinct objects",
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
    codes.sort_by_key(|c| std::cmp::Reverse(c.1));
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
fn reheader(snap: &SnapshotManifest, href: &ObjectRef) -> TfsResult<SnapshotManifest> {
    let mut m = Draft::of(snap);
    let old = m.header.clone();
    for (_, e) in m.entries.iter_mut() {
        if matches!(e, Entry::File(o) if Some(&*o) == old.as_ref()) {
            *e = Entry::File(href.clone());
        }
    }
    m.header = Some(href.clone());
    // Sealing IS the validation: an identity is only ever asked of a legal tree.
    m.seal()
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
    let href = snap.header().unwrap().clone();
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
    let mut m = Draft::of(&snap);
    m.entries.push((
        "NOTICE.txt".to_string(),
        Entry::File(ObjectRef::of(b"a sibling asset with a named reason")),
    ));
    m.entries.sort_by(|a, b| a.0.cmp(&b.0));
    let m = ok!(m.seal());
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
    let h2ref = ok!(h2.object_ref());
    let m2 = ok!(reheader(&snap, &h2ref));
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
    let h3ref = ok!(h3.object_ref());
    let m3 = ok!(reheader(&snap, &h3ref));
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
    let h4ref = ok!(h4.object_ref());
    let m4 = ok!(reheader(&snap, &h4ref));
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

// ------------------------------------------------------------------ audit arms

/// The two structural tightenings an external audit found, armed against the LIVE store.
/// Every arm plants the exact defect the fix names and prints the refusal it produced.
///
/// 1. A manifest could carry public mutable fields AND derive an id without validating, so
///    a programmatically-built illegal tree could be stored under a real-looking id. The
///    identity methods now exist only on the sealed type; a `Draft` has none, which the
///    compiler enforces — these arms show the sealing refusals that replace them.
/// 2. `walk` classified every unaccounted file as a sibling asset (so an orphan
///    `closure/<digest>.json` rode along as an "asset") and compared accounted objects by
///    DIGEST ALONE, ignoring the length a fetch plan is sized from.
pub fn cmd_arms(root: &Path, hex: &str) -> ExitCode {
    let (store, sref) = got!(opened(root, hex));
    let snap = ok!(ck::load_manifest(&store, &sref));
    let header = ok!(ck::load_header(&store, snap.header().unwrap()));
    let mut red = 0usize;

    let mut arm = |what: &str, d: Draft, walk_it: bool| match d.seal() {
        Err(e) => {
            red += 1;
            println!("  RED  {what:<44} seal refused {}", e.code.as_str());
        }
        Ok(m) => {
            if !walk_it {
                println!(
                    "  ??   {what:<44} SEALED as {} — arm did not fire",
                    m.snapshot_id()
                );
                return;
            }
            match ck::walk(&store, &m) {
                Err(e) => {
                    red += 1;
                    println!("  RED  {what:<44} walk refused {}", e.code.as_str());
                }
                Ok(w) => println!(
                    "  ??   {what:<44} walked {} objects, {} siblings — arm did not fire",
                    w.objects.len(),
                    w.siblings.len()
                ),
            }
        }
    };

    println!("audit arms against {}", sref.id());

    // (1) illegal trees cannot become an identity-bearing manifest at all.
    let mut d = Draft::of(&snap);
    let model = d
        .entries
        .iter()
        .position(|(path, _)| path == "model.cozytensors");
    let sibling = d
        .entries
        .iter()
        .position(|(path, _)| path == "text_encoder");
    match (model, sibling) {
        (Some(model), Some(sibling)) => d.entries.swap(model, sibling),
        _ => {
            eprintln!("REFUSED MISSING_FIELD: audit checkpoint lacks expected root files");
            return ExitCode::FAILURE;
        }
    }
    arm("draft: entries out of ascending order", d, false);

    let mut d = Draft::of(&snap);
    d.entries.clear();
    arm("draft: no entries at all", d, false);

    let mut d = Draft::of(&snap);
    d.entries.retain(|(p, _)| p != "model.cozytensors");
    arm("draft: attachment listed at zero paths", d, false);

    let mut d = Draft::of(&snap);
    d.entries.push((
        "model2.cozytensors".to_string(),
        Entry::File(snap.header().unwrap().clone()),
    ));
    d.entries.sort_by(|a, b| a.0.cmp(&b.0));
    arm("draft: attachment listed at two paths", d, false);

    // (2a) an orphan contained document: a legal PATH carrying a document the header does
    // not reach. The path law alone cannot see this — only the walk can.
    let orphan = ObjectRef::of(br#"{"format":"tensorfs.encoding_spec/1"}"#);
    let mut d = Draft::of(&snap);
    d.entries.push((
        tensorfs_core::manifest::closure_path(&orphan.sha256),
        Entry::File(orphan),
    ));
    d.entries.sort_by(|a, b| a.0.cmp(&b.0));
    arm("walk: orphan closure/<digest>.json", d, true);

    // (2b) the SAME digest at the WRONG length. Digest-only accounting called this
    // accounted and moved on; whole-ObjectRef accounting sees an object the header never
    // committed, sitting in the contained-document namespace.
    let real = header.encodings[0].clone();
    let mut d = Draft::of(&snap);
    for (p, e) in d.entries.iter_mut() {
        if p == &tensorfs_core::manifest::closure_path(&real.sha256) {
            *e = Entry::File(ObjectRef {
                sha256: real.sha256.clone(),
                length: real.length + 1,
            });
        }
    }
    arm("walk: right digest, wrong length", d, true);

    let orphan_asset = ObjectRef::of(b"uncommitted model asset");
    let mut d = Draft::of(&snap);
    d.entries.push((
        tensorfs_core::manifest::asset_path(&orphan_asset.sha256),
        Entry::File(orphan_asset),
    ));
    d.entries.sort_by(|a, b| a.0.cmp(&b.0));
    arm("walk: orphan assets/<digest>.bin", d, true);

    let mut conflicting = header.clone();
    if let Some((_, original)) = conflicting.assets.first().cloned() {
        let mut changed = original;
        changed.logical_length += 1;
        changed.segments[0].length += 1;
        conflicting
            .assets
            .push(("tokenizer/vocab2.txt".to_string(), changed));
        conflicting.assets.sort_by(|a, b| a.0.cmp(&b.0));
        let closure = ok!(ck::load_closure(&store, &header));
        match ck::build_snapshot(&conflicting, snap.header().unwrap(), &closure) {
            Err(error) if error.code == Code::LENGTH_MISMATCH => {
                red += 1;
                println!(
                    "  RED  {:<44} build refused {}",
                    "asset: same digest, conflicting length",
                    error.code.as_str()
                );
            }
            Err(error) => println!(
                "  ??   {:<44} wrong refusal {}",
                "asset: same digest, conflicting length",
                error.code.as_str()
            ),
            Ok(_) => println!(
                "  ??   {:<44} build succeeded — arm did not fire",
                "asset: same digest, conflicting length"
            ),
        }
    }

    println!("  {red}/8 arms fired");
    if red == 8 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
