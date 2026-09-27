//! Manifest inspection over the real local store.

use std::collections::BTreeMap;
use std::io::Cursor;
use std::path::Path;
use std::process::ExitCode;

use tensorfs_core::canon::{self, as_arr, as_str};
use tensorfs_core::checkpoint;
use tensorfs_core::err::{refuse, Code, Refusal, Result};
use tensorfs_core::header::{Body, Header};
use tensorfs_core::ids::{hex64, Doc, ObjectRef};
use tensorfs_core::manifest::{Draft, Entry};
use tensorfs_core::store::{Fault, Store};

use crate::{bail, flag, Flags};

fn read_input(path: &Path, what: &str) -> Result<Vec<u8>> {
    std::fs::read(path).map_err(|error| Refusal {
        code: Code::IO_FAILED,
        detail: format!("read {what} {}: {error}", path.display()),
    })
}

fn parse_entries(value: &canon::Value, header: &ObjectRef) -> Result<Draft> {
    let rows = as_arr("manifest reproduction", "entries", value)?;
    let mut entries = Vec::with_capacity(rows.len());
    let mut found_header = false;
    for (index, row) in rows.iter().enumerate() {
        let values = as_arr("manifest reproduction", "entry", row)?;
        if values.len() < 2 {
            return refuse(
                Code::WRONG_TYPE,
                format!("manifest reproduction entry {index} has fewer than 2 fields"),
            );
        }
        let path = as_str("manifest reproduction", "path", &values[0])?.to_string();
        let kind = as_str("manifest reproduction", "kind", &values[1])?;
        let entry = match (kind, values.len()) {
            ("cozytensors", 2) => {
                if found_header {
                    return refuse(
                        Code::ATTACHMENT_CARDINALITY,
                        "manifest reproduction lists more than one CozyTensors header",
                    );
                }
                found_header = true;
                Entry::CozyTensors(header.clone())
            }
            ("file", 3) => Entry::File(ObjectRef::from_value(
                "manifest reproduction file",
                &values[2],
            )?),
            _ => {
                return refuse(
                    Code::WRONG_TYPE,
                    format!(
                        "manifest reproduction entry {index} kind {kind:?} has arity {}; expected cozytensors=2 or file=3",
                        values.len()
                    ),
                )
            }
        };
        entries.push((path, entry));
    }
    if !found_header {
        return refuse(
            Code::ATTACHMENT_CARDINALITY,
            "manifest reproduction lists no CozyTensors header",
        );
    }
    Ok(Draft { entries })
}

fn add_ref(refs: &mut BTreeMap<String, u64>, object: &ObjectRef, what: &str) -> Result<()> {
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

fn referenced_blobs(header: &Header, draft: &Draft) -> Result<BTreeMap<String, u64>> {
    let mut refs = BTreeMap::new();
    for (_, asset) in &header.assets {
        for segment in &asset.segments {
            add_ref(&mut refs, segment, "asset segment")?;
        }
    }
    for (_, _, tensor) in header.tensors() {
        for (_, part) in &tensor.parts {
            if let Body::Segments(segments) = &part.body {
                for segment in segments {
                    add_ref(&mut refs, segment, "tensor segment")?;
                }
            }
        }
    }
    for (_, entry) in &draft.entries {
        if let Some(object) = entry.content() {
            add_ref(&mut refs, object, "manifest file")?;
        }
    }
    Ok(refs)
}

fn require_records(store: &Store, refs: &BTreeMap<String, u64>) -> Result<()> {
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
                        "sha256:{sha256} has no current exact verification record ({why}); metadata reproduction never rehashes inherited payload"
                    ),
                )
            }
        }
    }
    Ok(())
}

fn put_blob(store: &Store, bytes: &[u8]) -> Result<ObjectRef> {
    let want = ObjectRef::of(bytes);
    let put = store.put_stream(&mut Cursor::new(bytes), Some(&want), &Fault::default())?;
    if !put.admitted && store.record_valid(&want.sha256).is_err() {
        store.verify(&want.sha256)?;
    }
    Ok(want)
}

/// Reproduce one Header and Manifest over exact already-verified payload references.
/// Diagnostic JSON inputs never become stored formats.
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
    let closure = ok!(checkpoint::load_closure(&store, &header));
    ok!(header.validate(&closure));

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
        return bail(Refusal {
            code: Code::TRAVERSAL_ORDER_MISMATCH,
            detail: "header traversal differs from the supplied construction order".into(),
        });
    }

    let header_bytes = ok!(header.canonical_bytes());
    let header_ref = ObjectRef::of(&header_bytes);
    let entries_value = ok!(canon::parse(
        &ok!(read_input(entries_path, "manifest entries")),
        tensorfs_core::limits::DOC_MAX_BYTES,
    ));
    let draft = ok!(parse_entries(&entries_value, &header_ref));
    let refs = ok!(referenced_blobs(&header, &draft));
    ok!(require_records(&store, &refs));
    let manifest = ok!(draft.seal());

    let stored_header = ok!(put_blob(&store, &header_bytes));
    if stored_header != header_ref {
        return bail(Refusal {
            code: Code::OBJECT_ID_MISMATCH,
            detail: "stored header identity changed after canonical encoding".into(),
        });
    }
    let manifest_ref = ok!(store.put_manifest(&manifest)).obj;
    let strict = ok!(store.read_manifest(&manifest_ref));
    let walk = ok!(checkpoint::walk(&store, &strict));
    ok!(walk.require_resident(&store));
    let all = walk
        .distinct()
        .into_iter()
        .map(|object| (object.sha256.clone(), object.length))
        .collect();
    ok!(require_records(&store, &all));

    println!(
        "header       {} length={}",
        header_ref.id(),
        header_ref.length
    );
    println!(
        "manifest     {} length={}",
        manifest_ref.id(),
        manifest_ref.length
    );
    println!("tensors      {}", walk.tensors);
    println!("inherited    {} blobs", refs.len());
    println!("tensor_payload_read 0");
    println!("tensor_payload_hash 0");
    ExitCode::SUCCESS
}

fn open(root: &Path, hex: &str) -> tensorfs_core::err::Result<(Store, ObjectRef)> {
    let store = Store::open(root)?;
    let sha256 = hex64("manifest", hex.trim_start_matches("sha256:"))?;
    let length = std::fs::metadata(store.manifest_path(&sha256))
        .map_err(|error| tensorfs_core::err::Refusal {
            code: tensorfs_core::err::Code::OBJECT_ABSENT,
            detail: format!("manifest sha256:{sha256} is absent: {error}"),
        })?
        .len();
    Ok((store, ObjectRef { sha256, length }))
}

pub fn cmd_show(root: &Path, hex: &str) -> ExitCode {
    let (store, reference) = ok!(open(root, hex));
    let manifest = ok!(store.read_manifest(&reference));
    println!("manifest {} ({} B)", reference.id(), reference.length);
    println!("  {} entries", manifest.entries().len());
    for (path, entry) in manifest.entries() {
        match entry {
            Entry::File(blob) => {
                println!("    file         {path}  {} ({} B)", blob.id(), blob.length)
            }
            Entry::CozyTensors(blob) => println!(
                "    cozytensors  {path}  {} ({} B header)",
                blob.id(),
                blob.length
            ),
            Entry::Other { kind, blob } => {
                println!("    {kind:<12} {path}  {} ({} B)", blob.id(), blob.length)
            }
        }
    }
    ExitCode::SUCCESS
}

pub fn cmd_get(root: &Path, hex: &str, flags: &Flags) -> ExitCode {
    let Some(output) = flag(flags, "out") else {
        eprintln!("--out <path> is required");
        return ExitCode::from(2);
    };
    let (store, reference) = ok!(open(root, hex));
    let manifest = ok!(store.read_manifest(&reference));
    if let Err(error) = std::fs::write(output, manifest.canonical_bytes()) {
        return bail(tensorfs_core::err::Refusal {
            code: tensorfs_core::err::Code::IO_FAILED,
            detail: format!("write {output}: {error}"),
        });
    }
    println!("manifest {} ({} B)", reference.id(), reference.length);
    ExitCode::SUCCESS
}

pub fn cmd_admit(root: &Path, path: &Path, flags: &Flags) -> ExitCode {
    let Some(expected) = flag(flags, "expect") else {
        eprintln!("--expect <sha256> is required");
        return ExitCode::from(2);
    };
    let store = ok!(Store::open(root));
    let bytes = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) => {
            return bail(tensorfs_core::err::Refusal {
                code: tensorfs_core::err::Code::IO_FAILED,
                detail: format!("read {}: {error}", path.display()),
            })
        }
    };
    let manifest = ok!(tensorfs_core::manifest::Manifest::parse(&bytes));
    let observed = ObjectRef::of(&bytes);
    let expected = ok!(hex64(
        "expected manifest",
        expected.trim_start_matches("sha256:")
    ));
    if observed.sha256 != expected
        || flag(flags, "expect-length")
            .and_then(|length| length.parse::<u64>().ok())
            .is_some_and(|length| length != observed.length)
    {
        return bail(tensorfs_core::err::Refusal {
            code: tensorfs_core::err::Code::OBJECT_ID_MISMATCH,
            detail: format!(
                "manifest bytes are {}, expected sha256:{expected}",
                observed.id()
            ),
        });
    }
    let put = ok!(store.put_manifest(&manifest));
    println!(
        "manifest {} ({} B) admitted={}",
        put.obj.id(),
        put.obj.length,
        put.admitted
    );
    ExitCode::SUCCESS
}

pub fn cmd_walk(root: &Path, hex: &str, flags: &Flags) -> ExitCode {
    let (store, reference) = ok!(open(root, hex));
    let manifest = ok!(store.read_manifest(&reference));
    let walk = ok!(checkpoint::walk(&store, &manifest));
    if let Some(path) = flag(flags, "refs") {
        let mut references = std::collections::BTreeMap::new();
        for reached in &walk.objects {
            match references.insert(reached.obj.sha256.clone(), reached.obj.length) {
                Some(length) if length != reached.obj.length => {
                    return bail(tensorfs_core::err::Refusal {
                        code: tensorfs_core::err::Code::LENGTH_MISMATCH,
                        detail: format!(
                            "sha256:{} is reached at conflicting lengths",
                            reached.obj.sha256
                        ),
                    })
                }
                _ => {}
            }
        }
        let lines: Vec<Vec<u8>> = references
            .into_iter()
            .map(|(sha256, length)| {
                tensorfs_core::canon::write(&tensorfs_core::canon::Value::obj(vec![
                    ("length", tensorfs_core::canon::Value::uint(length)),
                    ("sha256", tensorfs_core::canon::Value::str(sha256)),
                ]))
            })
            .collect();
        if let Err(error) = tensorfs_core::storage::write_json_lines(Path::new(path), &lines) {
            return bail(error);
        }
    }
    println!(
        "manifest {} reaches {} references / {} distinct blobs",
        reference.id(),
        walk.objects.len(),
        walk.distinct().len()
    );
    ExitCode::SUCCESS
}

pub fn cmd_verify(root: &Path, hex: &str) -> ExitCode {
    let (store, reference) = ok!(open(root, hex));
    let manifest = ok!(store.read_manifest(&reference));
    let walk = ok!(checkpoint::walk(&store, &manifest));
    for reached in walk.distinct() {
        let file = ok!(store.open_verified(&reached.sha256));
        if file.len() != reached.length {
            eprintln!(
                "REFUSED LENGTH_MISMATCH: {} is {} bytes, manifest graph promises {}",
                reached.id(),
                file.len(),
                reached.length
            );
            return ExitCode::FAILURE;
        }
    }
    println!(
        "verified manifest {} and its complete blob closure",
        reference.id()
    );
    ExitCode::SUCCESS
}
