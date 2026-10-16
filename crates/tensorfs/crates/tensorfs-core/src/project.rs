//! Disposable local checkouts.

use std::collections::HashSet;
use std::fs;
use std::path::{Component, Path, PathBuf};

use crate::err::{refuse, Code, Refusal, Result};
use crate::manifest::{Entry, Manifest};
use crate::store::Store;

fn io(what: impl AsRef<str>, error: std::io::Error) -> Refusal {
    crate::store::classify_io(what, error)
}

#[derive(Debug, Default, Clone)]
pub struct Checkout {
    pub dirs: usize,
    pub links: usize,
    pub copies: usize,
    pub cozytensors_headers: usize,
    pub bytes: u64,
}

fn relative(from_dir: &Path, to: &Path) -> PathBuf {
    let left: Vec<Component> = from_dir.components().collect();
    let right: Vec<Component> = to.components().collect();
    let common = left
        .iter()
        .zip(right.iter())
        .take_while(|(a, b)| a == b)
        .count();
    let mut path = PathBuf::new();
    for _ in common..left.len() {
        path.push("..");
    }
    for component in &right[common..] {
        path.push(component.as_os_str());
    }
    path
}

/// Create a faithful manifest tree. Ordinary files and the one typed CozyTensors header
/// are relative symlinks where possible; `force_copy` writes small read-only copies.
/// Tensor payload segments are transitive CAS objects, not manifest entries, and never
/// appear in this disposable view.
/// Windows device stems. A manifest may legitimately name `aux.json` or `nul.safetensors`
/// — real HuggingFace trees do — and a virtual tree is not entitled to a filesystem's
/// opinions. A CHECKOUT is, because it writes real files, so the law lives here.
const RESERVED: [&str; 22] = [
    "con", "prn", "aux", "nul", "com1", "com2", "com3", "com4", "com5", "com6", "com7", "com8",
    "com9", "lpt1", "lpt2", "lpt3", "lpt4", "lpt5", "lpt6", "lpt7", "lpt8", "lpt9",
];

/// The two rules a real filesystem imposes that a manifest key does not: portable paths
/// cannot use a Windows device stem, and two entries cannot case-fold onto one file. Both
/// are checked before a single byte is written, so a refused checkout leaves nothing.
fn portable_on_disk(manifest: &Manifest) -> Result<()> {
    let mut folded: Vec<String> = Vec::with_capacity(manifest.entries().len());
    for (path, _) in manifest.entries() {
        for component in path.split('/') {
            let stem = component
                .split('.')
                .next()
                .unwrap_or(component)
                .to_ascii_lowercase();
            if RESERVED.contains(&stem.as_str()) {
                return refuse(
                    Code::PATH_RESERVED,
                    format!("{path:?}: Windows reserved stem {stem:?} cannot be checked out"),
                );
            }
        }
        folded.push(path.to_ascii_lowercase());
    }
    folded.sort();
    if let Some(pair) = folded.windows(2).find(|pair| pair[0] == pair[1]) {
        return refuse(
            Code::PATH_CASE_COLLISION,
            format!(
                "two entries case-fold onto {:?}; one checkout directory cannot hold both",
                pair[0]
            ),
        );
    }
    Ok(())
}

pub fn checkout(
    store: &Store,
    manifest: &Manifest,
    destination: &Path,
    force_copy: bool,
) -> Result<Checkout> {
    portable_on_disk(manifest)?;
    fs::create_dir_all(destination).map_err(|error| io("mkdir checkout", error))?;
    let destination = destination
        .canonicalize()
        .map_err(|error| io("canonicalize checkout", error))?;
    let mut result = Checkout::default();
    let mut directories = HashSet::new();
    for (virtual_path, entry) in manifest.entries() {
        let (blob, cozytensors) = match entry {
            Entry::CozyTensors(blob) => {
                crate::checkpoint::load_header(store, blob)?;
                (blob, true)
            }
            other => (other.materializable(virtual_path)?, false),
        };
        let source = store.blob_path(&blob.sha256);
        let metadata = fs::metadata(&source).map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                Refusal {
                    code: Code::OBJECT_ABSENT,
                    detail: format!("{virtual_path:?}: {} is absent", blob.id()),
                }
            } else {
                io(format!("stat {}", source.display()), error)
            }
        })?;
        if metadata.len() != blob.length {
            return refuse(
                Code::LENGTH_MISMATCH,
                format!(
                    "{virtual_path:?}: {} is {} bytes, manifest promises {}",
                    blob.id(),
                    metadata.len(),
                    blob.length
                ),
            );
        }
        let target = destination.join(virtual_path);
        let parent = target.parent().unwrap();
        fs::create_dir_all(parent).map_err(|error| io("mkdir checkout parent", error))?;
        let mut cursor = parent;
        while cursor.starts_with(&destination) && cursor != destination {
            if directories.insert(cursor.to_path_buf()) {
                result.dirs += 1;
            }
            cursor = cursor.parent().unwrap_or(&destination);
        }
        let _ = fs::remove_file(&target);
        let linked =
            !force_copy && std::os::unix::fs::symlink(relative(parent, &source), &target).is_ok();
        if linked {
            result.links += 1;
        } else {
            fs::copy(&source, &target).map_err(|error| io("copy checkout file", error))?;
            let mut permissions = fs::metadata(&target)
                .map_err(|error| io("stat checkout copy", error))?
                .permissions();
            std::os::unix::fs::PermissionsExt::set_mode(&mut permissions, 0o444);
            fs::set_permissions(&target, permissions)
                .map_err(|error| io("chmod checkout copy", error))?;
            result.copies += 1;
        }
        if cozytensors {
            result.cozytensors_headers += 1;
        }
        result.bytes = result
            .bytes
            .checked_add(blob.length)
            .ok_or_else(|| Refusal {
                code: Code::ARITH_OVERFLOW,
                detail: "checkout byte count overflow".into(),
            })?;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::Dtype;
    use crate::header::{Closure, Header, Part, Tensor};
    use crate::ids::ObjectRef;
    use crate::manifest::Draft;
    use crate::store::{Fault, Store};
    use std::os::unix::fs::PermissionsExt;

    fn temporary(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "tensorfs-project-{name}-{}-{}",
            std::process::id(),
            crate::meta::now_nanos_unique()
        ))
    }

    fn put(store: &Store, bytes: &[u8]) -> ObjectRef {
        let mut source = bytes;
        store
            .put_stream(&mut source, Some(&ObjectRef::of(bytes)), &Fault::default())
            .unwrap()
            .obj
    }

    struct ProjectionFixture {
        root: PathBuf,
        store: Store,
        manifest: Manifest,
        header: ObjectRef,
        ordinary: ObjectRef,
        payload: ObjectRef,
        header_bytes: Vec<u8>,
        ordinary_bytes: Vec<u8>,
    }

    fn projection_fixture(name: &str) -> ProjectionFixture {
        let root = temporary(name);
        let store = Store::init(&root).unwrap();
        let spec = crate::registry::seeds()
            .into_iter()
            .find(|seed| seed.alias == "plain/1")
            .unwrap()
            .spec;
        let payload_bytes = vec![7u8; 512];
        let payload = put(&store, &payload_bytes);
        let tensor = Tensor {
            dtype: Dtype::F32,
            shape: vec![128],
            encoding: spec.object_id(),
            parts: vec![(
                "value".into(),
                Part::plan(Dtype::F32, vec![128], &payload_bytes),
            )],
        };
        assert_eq!(tensor.parts[0].1.segments(), std::slice::from_ref(&payload));
        let header = Header {
            configs: Vec::new(),
            assets: Vec::new(),
            encodings: vec![spec],
            components: vec![("model".into(), vec![("weight".into(), tensor)])],
        };
        header.validate(&Closure::default()).unwrap();
        let header_bytes = header.canonical_bytes().unwrap();
        let header = put(&store, &header_bytes);
        let ordinary_bytes = b"ordinary file\n".to_vec();
        let ordinary = put(&store, &ordinary_bytes);
        let manifest = Draft {
            entries: vec![
                ("config.txt".into(), Entry::File(ordinary.clone())),
                (
                    "model.cozytensors".into(),
                    Entry::CozyTensors(header.clone()),
                ),
            ],
        }
        .seal()
        .unwrap();
        ProjectionFixture {
            root,
            store,
            manifest,
            header,
            ordinary,
            payload,
            header_bytes,
            ordinary_bytes,
        }
    }

    #[test]
    fn checkout_links_header_and_ordinary_file_without_payload_duplication() {
        let fixture = projection_fixture("header-link");
        let destination = fixture.root.join("checkout");
        let result = checkout(&fixture.store, &fixture.manifest, &destination, false).unwrap();
        assert_eq!(result.cozytensors_headers, 1);
        assert_eq!(result.links, 2);
        assert_eq!(result.copies, 0);
        assert_eq!(
            result.bytes,
            fixture.header.length + fixture.ordinary.length
        );
        for (path, bytes) in [
            ("model.cozytensors", fixture.header_bytes.as_slice()),
            ("config.txt", fixture.ordinary_bytes.as_slice()),
        ] {
            let projected = destination.join(path);
            assert!(fs::symlink_metadata(&projected)
                .unwrap()
                .file_type()
                .is_symlink());
            assert!(fs::read_link(&projected).unwrap().is_relative());
            assert_eq!(fs::read(projected).unwrap(), bytes);
        }
        assert!(fixture.store.blob_path(&fixture.payload.sha256).is_file());
        assert_eq!(fs::read_dir(&destination).unwrap().count(), 2);
        fs::remove_dir_all(fixture.root).unwrap();
    }

    #[test]
    fn checkout_no_symlink_copies_small_header_and_ordinary_file_read_only() {
        let fixture = projection_fixture("header-copy");
        let destination = fixture.root.join("checkout");
        let result = checkout(&fixture.store, &fixture.manifest, &destination, true).unwrap();
        assert_eq!(result.cozytensors_headers, 1);
        assert_eq!(result.links, 0);
        assert_eq!(result.copies, 2);
        for (path, bytes) in [
            ("model.cozytensors", fixture.header_bytes.as_slice()),
            ("config.txt", fixture.ordinary_bytes.as_slice()),
        ] {
            let projected = destination.join(path);
            let metadata = fs::symlink_metadata(&projected).unwrap();
            assert!(metadata.file_type().is_file());
            assert_eq!(metadata.permissions().mode() & 0o777, 0o444);
            assert_eq!(fs::read(projected).unwrap(), bytes);
        }
        assert_eq!(fs::read_dir(&destination).unwrap().count(), 2);
        fs::remove_dir_all(fixture.root).unwrap();
    }

    /// The manifest itself admits both — a virtual tree has no device files and no case
    /// folding — and the checkout is where a real filesystem gets its say.
    #[test]
    fn manifest_admits_windows_hostile_paths_that_checkout_refuses() {
        let root = temporary("windows-hostile");
        let store = Store::init(&root).unwrap();
        let blob = put(&store, b"payload");
        for (entries, code) in [
            (
                vec![
                    ("aux.json".to_string(), Entry::File(blob.clone())),
                    ("nul.safetensors".to_string(), Entry::File(blob.clone())),
                ],
                Code::PATH_RESERVED,
            ),
            (
                vec![
                    ("Config.json".to_string(), Entry::File(blob.clone())),
                    ("config.json".to_string(), Entry::File(blob.clone())),
                ],
                Code::PATH_CASE_COLLISION,
            ),
        ] {
            let manifest = Draft { entries }.seal().unwrap();
            let destination = root.join(format!("checkout-{}", code.as_str()));
            let error = checkout(&store, &manifest, &destination, false).unwrap_err();
            assert_eq!(error.code, code);
            assert!(!destination.exists(), "a refused checkout wrote nothing");
        }
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn checkout_refuses_missing_cozytensors_header() {
        let root = temporary("header-missing");
        let store = Store::init(&root).unwrap();
        let header = ObjectRef::of(b"absent header");
        let manifest = Draft {
            entries: vec![("model.cozytensors".into(), Entry::CozyTensors(header))],
        }
        .seal()
        .unwrap();
        let error = checkout(&store, &manifest, &root.join("checkout"), false).unwrap_err();
        assert_eq!(error.code, Code::OBJECT_ABSENT);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn checkout_refuses_wrong_length_cozytensors_header() {
        let fixture = projection_fixture("header-wrong-length");
        let manifest = Draft {
            entries: vec![(
                "model.cozytensors".into(),
                Entry::CozyTensors(ObjectRef {
                    sha256: fixture.header.sha256.clone(),
                    length: fixture.header.length + 1,
                }),
            )],
        }
        .seal()
        .unwrap();
        let error =
            checkout(&fixture.store, &manifest, &fixture.root.join("bad"), false).unwrap_err();
        assert_eq!(error.code, Code::LENGTH_MISMATCH);
        fs::remove_dir_all(fixture.root).unwrap();
    }

}
