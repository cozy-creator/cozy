//! A sharded model, read the way a rented pod reads one: from a flat content-addressed
//! staging area, with the member as the only name.
//!
//! Built from `vectors/h3-headers` — the REAL 48 members of `MiniMaxAI/MiniMax-H3`, 689 KB
//! of genuine metadata. The `*.index.json` members are complete files; the `*.safetensors`
//! members are the real 8-byte length prefix plus the real JSON header, and the tensor
//! payload each one declares is restored here as a sparse hole, so `read_header`'s geometry
//! arithmetic runs against the real declared runs at the real file lengths for zero bytes of
//! disk.
//!
//! **This is the shape the failure had.** H3 run 309 downloaded 48/48 members and 210.3 GB
//! and then refused on the first EIGHT BYTES of one file: a `*.safetensors.index.json`
//! staged at `staging/<hex>` has a hex digest for a file name, so it matched no extension
//! test, was read as a safetensors carrier, and its opening `{\n  "met` was taken for a u64
//! header length of 8,387,229,874,382,703,227 and refused against the 16 MiB cap. Every byte
//! that refusal needed was already in these 689 KB.
//!
//! One departure from production, stated because it makes the fixture HARDER rather than
//! easier: staged names here are digests of the header bytes, and the FL2VA and Ref2VA
//! shards have byte-identical headers (only their payloads differ, and the payloads are the
//! part that was not downloaded). So the fixture's 48 members present 34 paths where
//! production presents 47, and the two transformer components are backed by the SAME
//! fourteen files. Nothing but the member distinguishes them, which is exactly the property
//! under test.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use tensorfs_core::canon::{self, Value};
use tensorfs_core::err::Code;
use tensorfs_core::ingest::source::{read_carrier, read_carrier_at, CarrierInput, CarrierSet};
use tensorfs_core::limits;

const FL2VA: &str = "FL2VA/transformer/model.safetensors.index.json";
const REF2VA: &str = "Ref2VA/transformer/model.safetensors.index.json";
const TEXT_ENCODER: &str = "text_encoder/model.safetensors.index.json";
const VIDEO_VAE: &str = "vae/diffusion_pytorch_model.safetensors.index.json";
const AUDIO_VAE: &str = "audio_vae/diffusion_pytorch_model.safetensors";

fn temporary(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "tensorfs-sharded-cas-{name}-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}

fn headers_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../vectors/h3-headers")
        .canonicalize()
        .expect("vectors/h3-headers is checked in beside the crate")
}

/// `member`, `full_length` for all 48, read with the repo's own bounded JSON reader.
fn manifest() -> Vec<(String, u64)> {
    let bytes = fs::read(headers_dir().join("MANIFEST.json")).unwrap();
    let Value::Arr(rows) = canon::parse(&bytes, limits::DOC_MAX_BYTES).unwrap() else {
        panic!("MANIFEST.json is an array");
    };
    rows.iter()
        .map(|row| {
            let Value::Obj(fields) = row else {
                panic!("manifest row is an object")
            };
            let mut member = None;
            let mut length = None;
            for (key, value) in fields {
                match (key.as_str(), value) {
                    ("member", Value::Str(v)) => member = Some(v.clone()),
                    ("full_length", Value::Int(v)) => length = Some(*v as u64),
                    _ => {}
                }
            }
            (member.unwrap(), length.unwrap())
        })
        .collect()
}

/// The pod's staging area, reproduced: `<root>/staging/<hex>`, FLAT, every name a digest,
/// not one extension anywhere, and two members carrying one object sharing one file.
struct Staged {
    root: PathBuf,
    by_member: BTreeMap<String, PathBuf>,
}

impl Staged {
    fn build(root: &Path) -> Self {
        let source = headers_dir();
        let area = root.join("staging");
        fs::create_dir_all(&area).unwrap();
        let mut by_member = BTreeMap::new();
        for (member, full_length) in manifest() {
            let head = fs::read(source.join(member.replace('/', "__"))).unwrap();
            // The staged name stands in for the object id. Digesting the header bytes and
            // the declared length is what makes two members carrying one object share one
            // file here, exactly as content addressing does on the pod.
            let mut naming = head.clone();
            naming.extend_from_slice(full_length.to_string().as_bytes());
            let path = area.join(tensorfs_core::sha256::hex_digest(&naming));
            if !path.exists() {
                fs::write(&path, &head).unwrap();
                // The declared payload, as a hole. `read_header` measures the file and
                // checks every run against it; nothing in a header-only path reads it.
                let file = fs::OpenOptions::new().write(true).open(&path).unwrap();
                file.set_len(full_length).unwrap();
            }
            by_member.insert(member, path);
        }
        Self {
            root: root.to_path_buf(),
            by_member,
        }
    }

    fn set(&self) -> CarrierSet {
        CarrierSet::of(&self.carriers())
    }

    fn carriers(&self) -> Vec<CarrierInput> {
        self.by_member
            .iter()
            .map(|(member, path)| CarrierInput {
                member: Some(member.clone()),
                path: path.clone(),
            })
            .collect()
    }

    fn carrier(&self, member: &str) -> CarrierInput {
        CarrierInput {
            member: Some(member.to_string()),
            path: self.by_member[member].clone(),
        }
    }

    fn path(&self, member: &str) -> &Path {
        &self.by_member[member]
    }

    fn shard_members(&self, index: &str, count: usize) -> Vec<String> {
        let directory = index.rsplit_once('/').unwrap().0;
        let stem = if index.starts_with("vae/") {
            "diffusion_pytorch_model"
        } else {
            "model"
        };
        (1..=count)
            .map(|n| format!("{directory}/{stem}-{n:05}-of-{count:05}.safetensors"))
            .collect()
    }
}

impl Drop for Staged {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

/// THE REGRESSION, reproduced from the real bytes. Dispatching on the PATH is what run 309
/// did, and against a staged carrier it refuses on eight bytes of JSON read as a length.
#[test]
fn a_staged_index_read_by_path_refuses_exactly_as_run_309_did() {
    let staged = Staged::build(&temporary("run-309"));
    let path = staged.path(FL2VA);

    // The staged name really is what the dispatch had to work with.
    let name = path.file_name().unwrap().to_str().unwrap();
    assert_eq!(name.len(), 64, "a staged carrier is named by its digest");
    assert!(
        !name.contains('.'),
        "a staged carrier has no extension for a path test to find: {name}"
    );

    let refusal = read_carrier_at(path).expect_err("by path, an index is read as safetensors");
    assert_eq!(refusal.code, Code::CARRIER_HEADER_CAP);
    assert!(
        refusal.detail.contains("8387229874382703227"),
        "the refusal is run 309's, byte for byte: {}",
        refusal.detail
    );

    // The same file, named by its member, is a sharded carrier over 535 real tensors.
    let (header, _) = read_carrier(&staged.carrier(FL2VA), &staged.set())
        .expect("by member, the same bytes are an index");
    assert_eq!(header.tensors.len(), 535);
    assert_eq!(header.shards.len(), 13);
}

/// ONE object, TWO members, TWO shard sets. The decisive property, and the one no
/// path-keyed reader can have: H3 ships the FL2VA and Ref2VA transformer indexes as a single
/// 38,323-byte object, so a content-addressed store holds ONE file for both. Which shards it
/// names depends entirely on which member is asking.
#[test]
fn one_shared_index_object_resolves_two_member_namespaces() {
    let staged = Staged::build(&temporary("shared-index"));
    let set = staged.set();

    assert_eq!(
        staged.path(FL2VA),
        staged.path(REF2VA),
        "the two indexes are one object at one path"
    );
    assert_eq!(fs::metadata(staged.path(FL2VA)).unwrap().len(), 38_323);

    for index in [FL2VA, REF2VA] {
        let (header, _) = read_carrier(&staged.carrier(index), &set).unwrap();
        let wanted: Vec<PathBuf> = staged
            .shard_members(index, 13)
            .iter()
            .map(|member| staged.path(member).to_path_buf())
            .collect();
        let got: Vec<PathBuf> = header.shards.iter().map(|s| s.path.clone()).collect();
        assert_eq!(
            got, wanted,
            "{index} resolved shards outside its own member namespace"
        );
        assert_eq!(header.tensors.len(), 535);
        // Virtual geometry is laid end to end over the real declared regions.
        assert_eq!(header.file_len, 66_280_430_144);
        assert_eq!(header.data_start, 0);
    }
}

/// Every sharded component of the real selection, through the real reader. 48 members go in
/// as ONE carrier set; each index reaches only the shards its own `weight_map` names, and the
/// 34 carriers no index names are simply not its business.
#[test]
fn every_h3_index_resolves_through_one_carrier_set() {
    let staged = Staged::build(&temporary("all-indexes"));
    let set = staged.set();
    assert_eq!(staged.by_member.len(), 48, "48 verified members");
    assert_eq!(
        staged
            .by_member
            .values()
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        34,
        "sharing collapses them onto fewer files, and resolution must not care"
    );

    for (index, shards, tensors) in [
        (FL2VA, 13, 535),
        (REF2VA, 13, 535),
        (TEXT_ENCODER, 14, 1058),
        (VIDEO_VAE, 3, 703),
    ] {
        let (header, raw) = read_carrier(&staged.carrier(index), &set)
            .unwrap_or_else(|e| panic!("{index}: {} {}", e.code.as_str(), e.detail));
        assert_eq!(header.shards.len(), shards, "{index} shard count");
        assert_eq!(header.tensors.len(), tensors, "{index} tensor count");
        assert!(!raw.is_empty(), "{index} contributes identity bytes");
    }

    // A carrier the index does not name is not an error. The selection spans a whole model:
    // H3's audio VAE is a standalone carrier of another component and always will be.
    let (audio, _) = read_carrier(&staged.carrier(AUDIO_VAE), &set).unwrap();
    assert!(
        audio.shards.is_empty(),
        "a standalone carrier has no shards"
    );
}

/// A selection short of what its own index requires is refused BY MEMBER, before a byte of
/// any shard is opened. The path it would have had does not exist to be named, and the
/// member says which of the two faults it is: a narrowing that did not expand this index, or
/// an index that does not belong to the revision fetched.
#[test]
fn a_shard_the_carrier_set_lacks_is_refused_by_member() {
    let staged = Staged::build(&temporary("missing-shard"));
    let absent = "FL2VA/transformer/model-00007-of-00013.safetensors";
    let carriers: Vec<CarrierInput> = staged
        .carriers()
        .into_iter()
        .filter(|c| c.member.as_deref() != Some(absent))
        .collect();

    let refusal = read_carrier(&staged.carrier(FL2VA), &CarrierSet::of(&carriers))
        .expect_err("an index whose shard is not carried must refuse");
    assert_eq!(refusal.code, Code::MISSING_FIELD);
    assert!(
        refusal.detail.contains(absent),
        "the refusal names the missing MEMBER: {}",
        refusal.detail
    );
}

/// The other layout still works, unchanged: a carrier with no member is a location, its file
/// name is its name, and its shards are its siblings. `tfs ingest` over a checkout.
#[test]
fn an_index_on_disk_still_resolves_its_shards_as_siblings() {
    let root = temporary("siblings");
    let tree = root.join("vae");
    fs::create_dir_all(&tree).unwrap();
    let source = headers_dir();
    let mut index = None;
    for (member, full_length) in manifest() {
        let Some(name) = member.strip_prefix("vae/") else {
            continue;
        };
        let path = tree.join(name);
        fs::write(
            &path,
            fs::read(source.join(member.replace('/', "__"))).unwrap(),
        )
        .unwrap();
        fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(full_length)
            .unwrap();
        if name.ends_with(".index.json") {
            index = Some(path);
        }
    }
    let index = index.unwrap();

    // No member: the file name carries the extension and the directory is the scope.
    let (by_name, _) = read_carrier_at(&index).expect("a directory tree still reads");
    assert_eq!(by_name.shards.len(), 3);
    assert_eq!(by_name.tensors.len(), 703);
    for shard in &by_name.shards {
        assert_eq!(shard.path.parent().unwrap(), tree);
    }

    // Naming the member puts the same file in member space, where the selection — not the
    // directory — is the authority, and this one-carrier selection does not carry the shards.
    let named = CarrierInput {
        member: Some(VIDEO_VAE.to_string()),
        path: index.clone(),
    };
    let set = CarrierSet::of(std::slice::from_ref(&named));
    let refusal = read_carrier(&named, &set).expect_err("member space needs the shards named");
    assert_eq!(refusal.code, Code::MISSING_FIELD);

    // ...and supplying them makes it read, from the very same files.
    let mut carriers = vec![named.clone()];
    for n in 1..=3 {
        let name = format!("diffusion_pytorch_model-{n:05}-of-00003.safetensors");
        carriers.push(CarrierInput {
            member: Some(format!("vae/{name}")),
            path: tree.join(name),
        });
    }
    let (by_member, _) = read_carrier(&named, &CarrierSet::of(&carriers)).unwrap();
    assert_eq!(by_member, by_name, "both layouts reach one joined header");

    let _ = fs::remove_dir_all(&root);
}
