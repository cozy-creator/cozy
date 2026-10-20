//! One-off exact H3 staging client. Install as a Cargo example in Product25 only.
//! The default resolves and presigns metadata; only --download permits object GETs.
use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::Deserialize;
use serde_json::{json, Value};
use tensorfs::fetch::{
    self, Client, Fetcher, Location, Origin, SourcePolicy, TensorhubOrigin, Want,
};
use tensorfs::format::{Header, ObjectRef};
use tensorfs::store2::{Pin, Store};

const PRODUCT: &str = "25a1d2f6bfbf07e84f06a110b30083b9fb5a45f2";
const HUB: &str = "https://cozy-e2e-6.ngrok.app";
const HUB_HOST: &str = "cozy-e2e-6.ngrok.app";
const OBJECT_HOST: &str = "327b4294d98bb480834acd2660b423d4.r2.cloudflarestorage.com";
const MODEL: &str = "paul/minimax-h3";
const RELEASE: &str = "1.0.0-rc.3";
const LANE: &str = "int8";
const LABEL: &str = "paul/minimax-h3@1.0.0-rc.3/int8";
const MANIFEST: &str = "bebd6c6966f33bf53d176bbc3520ed2021854f5bcd8446b2ad2819ff3e6d4cff";
const HEADER: &str = "0f03804af2280bac6c81c633cd058f4dfbb5fd71f5db0a3c5829907e4ef21f2e";
const PIN_ID: &str = "xs056-h3-chunk-stage-0f038";
const OWNER: &str = "xs056-h3-chunk-stage/1";
const CHUNKS: usize = 1676;
const CHUNK_BYTES: u64 = 75_535_069_920;
const PART_BYTES: u64 = 75_528_893_476;
const HEADER_BYTES: u64 = 433_449;

#[derive(Debug)]
struct Problem {
    stage: &'static str,
    code: &'static str,
}
type Result<T> = std::result::Result<T, Problem>;

fn problem(stage: &'static str, code: &'static str) -> Problem {
    // Do not copy transport details: they can contain signed URLs.
    Problem { stage, code }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    source_label: String,
    legacy_manifest_sha256: String,
    header: PathBuf,
    header_sha256: String,
    cache_dir: PathBuf,
    tensor_count: usize,
    stored_bytes: u64,
}

struct Plan {
    input: Input,
    header_bytes: Vec<u8>,
    header: Header,
    metadata_root: PathBuf,
    cache: PathBuf,
}

fn bounded(path: &Path, limit: u64) -> Result<Vec<u8>> {
    let file = File::open(path).map_err(|_| problem("local_metadata", "OPEN_FAILED"))?;
    let mut bytes = Vec::new();
    file.take(limit + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| problem("local_metadata", "READ_FAILED"))?;
    if bytes.len() as u64 > limit {
        return Err(problem("local_metadata", "SIZE_LIMIT"));
    }
    Ok(bytes)
}

fn chunks(objects: &[ObjectRef], count: usize, length: u64) -> Result<()> {
    if objects.len() != count {
        return Err(problem("geometry", "CHUNK_COUNT"));
    }
    let mut seen = HashSet::new();
    let mut bytes = 0u64;
    for object in objects {
        if object.length == 0 || !seen.insert(&object.sha256) {
            return Err(problem("geometry", "DUPLICATE_OR_EMPTY_CHUNK"));
        }
        bytes = bytes
            .checked_add(object.length)
            .ok_or_else(|| problem("geometry", "BYTE_OVERFLOW"))?;
    }
    if bytes != length {
        return Err(problem("geometry", "CHUNK_BYTES"));
    }
    Ok(())
}

fn plan(path: &Path) -> Result<Plan> {
    let input: Input = serde_json::from_slice(&bounded(path, 16 * 1024)?)
        .map_err(|_| problem("input", "INVALID_INPUT"))?;
    if input.source_label != LABEL
        || input.legacy_manifest_sha256 != MANIFEST
        || input.header_sha256 != HEADER
        || input.tensor_count != 3858
        || input.stored_bytes != PART_BYTES
        || !input.header.is_absolute()
        || !input.cache_dir.is_absolute()
    {
        return Err(problem("input", "WRONG_SOURCE_PIN"));
    }
    let header_path =
        fs::canonicalize(&input.header).map_err(|_| problem("local_metadata", "HEADER_PATH"))?;
    let header_bytes = bounded(&header_path, HEADER_BYTES)?;
    if ObjectRef::of(&header_bytes) != header_ref() {
        return Err(problem("local_metadata", "HEADER_IDENTITY"));
    }
    // The accepted A11 bundle has this physical layout. Do not open or mutate its Store.
    let suffix = Path::new("blobs").join("0f").join("03").join(HEADER);
    let metadata_root = header_path
        .ancestors()
        .nth(4)
        .filter(|root| root.join(&suffix) == header_path)
        .ok_or_else(|| problem("local_metadata", "EXPECTED_A11_LAYOUT"))?
        .to_path_buf();
    let cache = if input.cache_dir.exists() {
        fs::canonicalize(&input.cache_dir).map_err(|_| problem("input", "CACHE_PATH"))?
    } else {
        let parent = input
            .cache_dir
            .parent()
            .ok_or_else(|| problem("input", "CACHE_PATH"))?;
        let name = input
            .cache_dir
            .file_name()
            .ok_or_else(|| problem("input", "CACHE_PATH"))?;
        fs::canonicalize(parent)
            .map_err(|_| problem("input", "CACHE_PARENT"))?
            .join(name)
    };
    if cache.starts_with(&metadata_root) || metadata_root.starts_with(&cache) {
        return Err(problem("input", "CACHE_OVERLAPS_METADATA"));
    }
    let header =
        Header::from_cbor(&header_bytes).map_err(|_| problem("local_metadata", "HEADER_FORMAT"))?;
    chunks(&header.chunks, CHUNKS, CHUNK_BYTES)?;
    let mut names: Vec<_> = header
        .components()
        .iter()
        .map(|c| c.name.as_str())
        .collect();
    names.sort_unstable();
    if names
        != [
            "audio_vae",
            "text_encoder",
            "transformer",
            "transformer_ref",
            "vae",
        ]
        || header.tensors().count() != 3858
    {
        return Err(problem("geometry", "COMPONENTS_OR_TENSORS"));
    }
    let parts: Vec<_> = header
        .tensors()
        .flat_map(|(_, tensor)| tensor.locations())
        .collect();
    if parts.len() != 4834 || parts.iter().map(|(_, at)| at.length).sum::<u64>() != PART_BYTES {
        return Err(problem("geometry", "STORED_PARTS"));
    }
    Ok(Plan {
        input,
        header_bytes,
        header,
        metadata_root,
        cache,
    })
}

fn header_ref() -> ObjectRef {
    ObjectRef {
        sha256: HEADER.into(),
        length: HEADER_BYTES,
    }
}

struct Identity<'a> {
    model: &'a str,
    release: &'a str,
    lane: &'a str,
    manifest: ObjectRef,
}

async fn preflight(
    client: &Client,
    hub: &str,
    identity: &Identity<'_>,
    chunk: &ObjectRef,
) -> Result<Arc<dyn Origin>> {
    let origin =
        TensorhubOrigin::new(client, hub).map_err(|e| problem("origin", e.code.as_str()))?;
    let found = origin
        .named(identity.model, Some(identity.release), Some(identity.lane))
        .await
        .map_err(|e| problem("resolve", e.code.as_str()))?;
    if found.tree != identity.manifest
        || found.model != identity.model
        || found.release != identity.release
        || found.variant != identity.lane
        || !found.public
    {
        return Err(problem("resolve", "WRONG_OR_PRIVATE_MANIFEST"));
    }
    let locations = origin
        .locate(std::slice::from_ref(chunk))
        .await
        .map_err(|e| problem("presign", e.code.as_str()))?;
    match locations.as_slice() {
        [Location::Url {
            url,
            headers,
            expires,
        }] if headers.is_empty() && expires.is_some() => {
            client
                .policy()
                .admit(url)
                .map_err(|e| problem("presign", e.code.as_str()))?;
        }
        _ => return Err(problem("presign", "EXPECTED_SIGNED_LOCATION")),
    }
    Ok(Arc::new(origin))
}

fn ownership_marker(header: &ObjectRef) -> String {
    // Product25 allows .format* beside an uninitialized Store's standard directories.
    // An empty exclusive marker encodes every identity in its name, so there is no
    // partially written JSON state to repair after a crash.
    format!(
        ".format-{}-{PRODUCT}-{}-{}",
        OWNER.replace('/', "-"),
        header.sha256,
        header.length
    )
}

fn claim_owned_target(root: &Path, header: &ObjectRef) -> Result<()> {
    match fs::create_dir(root) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(_) => return Err(problem("ownership", "CREATE_DIRECTORY_FAILED")),
    }
    if !fs::symlink_metadata(root)
        .map_err(|_| problem("ownership", "STAT_FAILED"))?
        .is_dir()
    {
        return Err(problem("ownership", "NOT_A_DIRECTORY"));
    }
    let marker = root.join(ownership_marker(header));
    match fs::symlink_metadata(&marker) {
        Ok(metadata) if metadata.is_file() && metadata.len() == 0 => return Ok(()),
        Ok(_) => return Err(problem("ownership", "INVALID_MARKER")),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(_) => return Err(problem("ownership", "MARKER_READ_FAILED")),
    }
    // Refuse before creating even the marker when this is somebody else's directory.
    let mut entries = fs::read_dir(root).map_err(|_| problem("ownership", "LIST_FAILED"))?;
    if entries.next().is_some() {
        return Err(problem("ownership", "UNOWNED_NONEMPTY_DIRECTORY"));
    }
    let marker = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(marker)
        .map_err(|_| problem("ownership", "EXCLUSIVE_MARKER_FAILED"))?;
    marker
        .sync_all()
        .map_err(|_| problem("ownership", "MARKER_SYNC_FAILED"))?;
    File::open(root)
        .and_then(|dir| dir.sync_all())
        .map_err(|_| problem("ownership", "DIRECTORY_SYNC_FAILED"))?;
    Ok(())
}

fn open_owned_store(root: &Path, header: &ObjectRef) -> Result<Store> {
    // A completed marker survives a crash before Store initialization or pin creation.
    claim_owned_target(root, header)?;
    Store::open(root).map_err(|e| problem("store", e.code()))
}

fn pin_store(plan: &Plan) -> Result<Arc<Store>> {
    let store = open_owned_store(&plan.cache, &header_ref())?;
    let old = store
        .get_pin(PIN_ID)
        .map_err(|e| problem("store", e.code()))?;
    let mut pin = Pin::new(vec![header_ref()]);
    pin.owner = Some(OWNER.into());
    if old
        .as_ref()
        .is_some_and(|old| old.owner != pin.owner || old.roots != pin.roots)
    {
        return Err(problem("store", "WRONG_EXISTING_PIN"));
    }
    store
        .put(plan.header_bytes.as_slice(), Some(&header_ref()))
        .map_err(|e| problem("store_header", e.code()))?;
    // Only small local component assets; README/LICENSE are repo files, not this closure.
    for (_, _, asset) in plan.header.assets() {
        let path = plan
            .metadata_root
            .join("blobs")
            .join(&asset.sha256[..2])
            .join(&asset.sha256[2..4])
            .join(&asset.sha256);
        let bytes = bounded(&path, 16 * 1024 * 1024)?;
        store
            .put(bytes.as_slice(), Some(asset))
            .map_err(|e| problem("store_asset", e.code()))?;
    }
    // A header pin may name absent chunks: it protects each object as it lands.
    store
        .pin(PIN_ID, &pin)
        .map_err(|e| problem("pin", e.code()))?;
    Ok(Arc::new(store))
}

async fn execute(plan: &Plan, download: bool) -> Result<(Value, bool)> {
    let client = Client::new(SourcePolicy {
        allowed_hosts: vec![HUB_HOST.into(), OBJECT_HOST.into()],
        any_host: false,
        allow_local: false,
        ..SourcePolicy::default()
    });
    let identity = Identity {
        model: MODEL,
        release: RELEASE,
        lane: LANE,
        manifest: ObjectRef {
            sha256: MANIFEST.into(),
            length: 743,
        },
    };
    let origin = preflight(&client, HUB, &identity, &plan.header.chunks[0]).await?;
    let mut report = json!({
        "format": "cozy.h3.chunk-stage/1", "product_source": PRODUCT,
        "client_source": option_env!("H3_STAGE_SOURCE_COMMIT"),
        "source_hub": HUB, "source_label": plan.input.source_label,
        "allowed_hosts": [HUB_HOST, OBJECT_HOST],
        "old_manifest": identity.manifest, "header": header_ref(),
        "cache_dir": plan.cache, "chunk_count": CHUNKS, "physical_chunk_bytes": CHUNK_BYTES,
        "tensor_count": 3858, "stored_parts": 4834, "stored_part_bytes": PART_BYTES,
        "metadata_preflight": true, "download_requested": download,
        "payload_transfer_called": false, "closure_verified": false,
        "benchmark_run": false, "gpu_used": false,
    });
    if !download {
        report["status"] = json!("metadata_only");
        return Ok((report, true));
    }
    let store = pin_store(plan)?;
    let fetcher = Fetcher::new(store.clone(), client, None, fetch::Config::default());
    let transfer = fetcher
        .objects(&origin, &plan.header.chunks, &Want::default())
        .await;
    fetcher.settled().await;
    let sync = store.sync();
    let checkpoint = store
        .checkpoint(&header_ref())
        .map_err(|e| problem("verify_open", e.code()))?;
    let verification = checkpoint
        .verify_selected(None)
        .map_err(|_| problem("verify", "AUDIT_IO_ERROR"))?;
    let stats = fetcher.stats();
    let unavailable: HashSet<_> = verification
        .missing
        .iter()
        .chain(&verification.corrupt)
        .map(|object| object.sha256.as_str())
        .collect();
    let verified: Vec<_> = plan
        .header
        .chunks
        .iter()
        .filter(|object| !unavailable.contains(object.sha256.as_str()))
        .collect();
    let complete = transfer.is_ok() && sync.is_ok() && verification.is_ok();
    report["status"] = json!(if complete {
        "staged_verified"
    } else {
        "incomplete"
    });
    report["payload_transfer_called"] = json!(true);
    report["closure_verified"] = json!(verification.is_ok());
    report["pin_id"] = json!(PIN_ID);
    report["verified_chunk_count"] = json!(verified.len());
    report["verified_chunk_bytes"] =
        json!(verified.iter().map(|object| object.length).sum::<u64>());
    report["missing"] = json!(verification.missing);
    report["corrupt"] = json!(verification.corrupt);
    report["transfer_error"] = json!(transfer.as_ref().err().map(|e| e.code.as_str()));
    report["sync_error"] = json!(sync.as_ref().err().map(|e| e.code()));
    report["landed"] = match transfer {
        Ok(landed) => {
            json!({ "local": landed.local, "origin": landed.origin, "volume": landed.volume })
        }
        Err(_) => Value::Null,
    };
    report["stats"] = json!({
        "requests": stats.requests, "restarts": stats.restarts, "streams": stats.streams,
        "volume_written": stats.volume_written, "volume_skipped": stats.volume_skipped,
    });
    Ok((report, complete))
}

fn run() -> Result<(Value, bool)> {
    if !option_env!("H3_STAGE_SOURCE_COMMIT").is_some_and(|source| {
        source.len() == 40
            && source
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    }) {
        return Err(problem("provenance", "CLIENT_SOURCE_COMMIT_REQUIRED"));
    }
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.is_empty() || args.len() > 2 || (args.len() == 2 && args[1] != "--download") {
        return Err(problem("arguments", "USAGE_SOURCE_JSON_OPTIONAL_DOWNLOAD"));
    }
    let plan = plan(Path::new(&args[0]))?;
    fetch::runtime()
        .map_err(|e| problem("runtime", e.code.as_str()))?
        .block_on(execute(&plan, args.len() == 2))
}

fn main() {
    let (report, success) = match run() {
        Ok(result) => result,
        Err(error) => (
            json!({
                "format": "cozy.h3.chunk-stage/1", "status": "failed",
                "product_source": PRODUCT, "stage": error.stage, "error_code": error.code,
                "benchmark_run": false, "gpu_used": false,
            }),
            false,
        ),
    };
    println!("{report}");
    if !success {
        std::process::exit(1);
    }
}

#[cfg(test)]
#[path = "../tests/common/mod.rs"]
mod common;

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn metadata_preflight_presigns_old_manifest_without_object_get() {
        let bytes = tensorfs::limits::INLINE_MAX_BYTES as usize + 4;
        let checkpoint = Arc::new(common::components(
            "acme/old-checkpoint",
            &[1],
            bytes,
            3,
            &[],
        ));
        let hub = common::hub(
            vec![checkpoint.clone()],
            None,
            Duration::ZERO,
            common::serve_all(),
        );
        let client = Client::new(SourcePolicy {
            allowed_hosts: vec!["127.0.0.1".into()],
            allow_local: true,
            ..SourcePolicy::default()
        });
        let identity = Identity {
            model: &checkpoint.model,
            release: &checkpoint.release,
            lane: &checkpoint.lane,
            manifest: ObjectRef {
                sha256: checkpoint.manifest.sha256.clone(),
                length: checkpoint.manifest.length,
            },
        };
        let chunk = ObjectRef {
            sha256: checkpoint.blobs[0].sha256.clone(),
            length: checkpoint.blobs[0].length,
        };
        fetch::runtime()
            .unwrap()
            .block_on(preflight(&client, &hub.base, &identity, &chunk))
            .unwrap();
        assert_eq!(hub.gets("closure"), 1);
        assert_eq!(hub.gets("presign"), 1);
        assert_eq!(hub.total_gets(), 0);
    }

    #[test]
    fn changed_manifest_stops_before_presign_or_object_get() {
        let bytes = tensorfs::limits::INLINE_MAX_BYTES as usize + 4;
        let checkpoint = Arc::new(common::components(
            "acme/old-checkpoint",
            &[1],
            bytes,
            4,
            &[],
        ));
        let hub = common::hub(
            vec![checkpoint.clone()],
            None,
            Duration::ZERO,
            common::serve_all(),
        );
        let client = Client::new(SourcePolicy {
            allowed_hosts: vec!["127.0.0.1".into()],
            allow_local: true,
            ..SourcePolicy::default()
        });
        let identity = Identity {
            model: &checkpoint.model,
            release: &checkpoint.release,
            lane: &checkpoint.lane,
            manifest: ObjectRef::of(b"wrong manifest"),
        };
        let chunk = ObjectRef::of(b"not requested");
        let error = fetch::runtime()
            .unwrap()
            .block_on(preflight(&client, &hub.base, &identity, &chunk))
            .err()
            .unwrap();
        assert_eq!(error.code, "WRONG_OR_PRIVATE_MANIFEST");
        assert_eq!(hub.gets("closure"), 1);
        assert_eq!(hub.gets("presign"), 0);
        assert_eq!(hub.total_gets(), 0);
    }

    #[test]
    fn chunk_set_rejects_duplicates_conflicts_counts_and_totals() {
        let first = ObjectRef::of(b"one");
        let second = ObjectRef::of(b"second");
        chunks(&[first.clone(), second.clone()], 2, 9).unwrap();
        assert_eq!(
            chunks(&[first.clone(), first.clone()], 2, 6)
                .unwrap_err()
                .code,
            "DUPLICATE_OR_EMPTY_CHUNK"
        );
        let mut conflict = first.clone();
        conflict.length += 1;
        assert_eq!(
            chunks(&[first.clone(), conflict], 2, 7).unwrap_err().code,
            "DUPLICATE_OR_EMPTY_CHUNK"
        );
        assert_eq!(
            chunks(std::slice::from_ref(&first), 2, 3).unwrap_err().code,
            "CHUNK_COUNT"
        );
        assert_eq!(
            chunks(&[first, second], 2, 10).unwrap_err().code,
            "CHUNK_BYTES"
        );
    }

    #[test]
    fn header_pin_names_absent_chunks_without_reading_them() {
        let root = common::temporary("h3-header-pin");
        let store = Store::open(root.join("stage")).unwrap();
        let chunk = ObjectRef::of(b"not downloaded");
        let header = Header::new(vec![chunk.clone()], None, vec![], Default::default()).unwrap();
        let bytes = header.to_cbor().unwrap();
        let expected = ObjectRef::of(&bytes);
        let admitted = store.put(bytes.as_slice(), Some(&expected)).unwrap();
        let mut pin = Pin::new(vec![admitted.clone()]);
        pin.owner = Some(OWNER.into());
        store.pin(PIN_ID, &pin).unwrap();
        assert_eq!(store.read(&admitted).unwrap(), bytes);
        assert_eq!(store.get_pin(PIN_ID).unwrap(), Some(pin));
        assert!(store.closure(&admitted).unwrap().contains(&chunk));
        assert!(!store.path(&chunk).unwrap().exists());
    }

    #[test]
    fn foreign_nonempty_directory_is_refused_without_any_new_file() {
        let root = common::temporary("h3-foreign-target");
        fs::create_dir(root.join("blobs")).unwrap();
        let sentinel = root.join("blobs/foreign");
        fs::write(&sentinel, b"preserve this existing data").unwrap();
        let error = open_owned_store(&root, &header_ref()).unwrap_err();
        assert_eq!(error.code, "UNOWNED_NONEMPTY_DIRECTORY");
        assert_eq!(fs::read(&sentinel).unwrap(), b"preserve this existing data");
        let entries: Vec<_> = fs::read_dir(&root)
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(entries, [std::ffi::OsString::from("blobs")]);
        assert!(!root.join("format").exists());
        assert!(!root.join(ownership_marker(&header_ref())).exists());
    }

    #[test]
    fn exact_owned_target_resumes_after_marker_store_and_pin_steps() {
        let root = common::temporary("h3-owned-resume");
        let bytes = Header::default().to_cbor().unwrap();
        let header = ObjectRef::of(&bytes);
        // Crash after the exclusive marker, before Store initialization.
        claim_owned_target(&root, &header).unwrap();
        assert!(!root.join("format").exists());
        let store = open_owned_store(&root, &header).unwrap();
        assert!(store.get_pin(PIN_ID).unwrap().is_none());
        drop(store);
        // Crash after Store initialization, before header or pin creation.
        let store = open_owned_store(&root, &header).unwrap();
        store.put(bytes.as_slice(), Some(&header)).unwrap();
        drop(store);
        let store = open_owned_store(&root, &header).unwrap();
        let mut pin = Pin::new(vec![header.clone()]);
        pin.owner = Some(OWNER.into());
        store.pin(PIN_ID, &pin).unwrap();
        drop(store);
        // A completed run remains the same owned Store and pin on resume.
        let store = open_owned_store(&root, &header).unwrap();
        assert_eq!(store.read(&header).unwrap(), bytes);
        assert_eq!(store.get_pin(PIN_ID).unwrap(), Some(pin));
    }

    #[test]
    fn marker_for_different_owner_product_or_header_is_refused_without_initialization() {
        let other = ObjectRef::of(b"other header");
        for name in [
            ownership_marker(&other),
            ownership_marker(&header_ref()).replace(PRODUCT, &"0".repeat(40)),
            ownership_marker(&header_ref()).replace(&OWNER.replace('/', "-"), "another-owner"),
        ] {
            let root = common::temporary("h3-wrong-marker");
            File::create(root.join(&name)).unwrap();
            let error = open_owned_store(&root, &header_ref()).unwrap_err();
            assert_eq!(error.code, "UNOWNED_NONEMPTY_DIRECTORY");
            assert!(!root.join("format").exists());
            let entries: Vec<_> = fs::read_dir(&root)
                .unwrap()
                .map(|e| e.unwrap().file_name())
                .collect();
            assert_eq!(entries, [std::ffi::OsString::from(name)]);
        }
    }
}
