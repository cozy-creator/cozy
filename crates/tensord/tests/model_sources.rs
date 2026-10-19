//! A session's model selection: its authority, its headers and its encoded size, and the
//! header and assets it serves an executor that reads no store.
use tensord::model_sources::{ModelSources, SelectedManifest};
use sha2::{Digest, Sha256};
use platform::durable::FileAt as _;
use std::{fs, io};
#[cfg(unix)]
use std::os::fd::AsRawFd;
use tensorfs_core::{
    format::{Dtype, ObjectRef},
    writer::{Data, Source},
};

fn fixture() -> (std::path::PathBuf, String, Vec<u8>, ObjectRef, ObjectRef) {
    let root = std::env::temp_dir().join(format!("machine-source-{}", uuid::Uuid::new_v4()));
    let store = tensord::storage::open(&root).unwrap();
    let data: Vec<u8> = (0..65540).map(|n| (n % 251) as u8).collect();
    let other = vec![9; 65540];
    let mut writer = store.writer(None, Some("fixture")).unwrap();
    for (component, bytes) in [("allowed", &data), ("denied", &other)] {
        let shape = [bytes.len() as u64];
        writer.put(component, "weight", &Dtype::U8, &shape, Data::Plain(Source::Read(&mut bytes.as_slice()))).unwrap();
    }
    writer.file("tokenizer/vocab.json", b"static tokenizer asset").unwrap();
    let manifest = writer.commit().unwrap();
    (root, manifest.id(), data.clone(), ObjectRef::of(&data), ObjectRef::of(&other))
}

#[test]
fn a_selection_grants_its_headers_and_sizes_and_nothing_else() {
    let (root, manifest, data, _, _) = fixture();
    let selected = SelectedManifest {
        manifest: manifest.clone(),
        components: vec!["allowed".into()],
    };
    let sources = ModelSources::open(&root, std::slice::from_ref(&selected)).unwrap();
    let (components, encoded, _) = sources.selected_facts(&manifest).unwrap();
    assert_eq!(components, vec!["allowed".to_string()]);
    assert!(encoded >= data.len() as u64, "{encoded}");
    assert!(sources.authorized(&manifest).unwrap().header().component("allowed").is_some());
    // Store presence is no authority: an unselected manifest, an absent component.
    let none = ModelSources::open(&root, &[]).unwrap();
    assert_eq!(
        none.authorized(&manifest).err().unwrap().kind(),
        io::ErrorKind::PermissionDenied
    );
    let absent = SelectedManifest {
        manifest,
        components: vec!["absent".into()],
    };
    assert!(ModelSources::open(&root, &[absent]).is_err());
    drop((sources, none));
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn a_read_grant_gets_the_selected_components_chunk_files_and_no_others() {
    let (root, manifest, _, _, _) = fixture();
    let selected = SelectedManifest { manifest: manifest.clone(), components: vec!["allowed".into()] };
    let sources = ModelSources::open(&root, &[selected]).unwrap();
    let chunks = sources.chunks(&manifest).unwrap();
    assert!(chunks.windows(2).all(|w| w[0].sha256 < w[1].sha256), "sorted by digest, each once");
    let checkpoint = sources.authorized(&manifest).unwrap();
    let header = checkpoint.header();
    let wanted = |component: &str| -> Vec<ObjectRef> {
        let tensors = &header.component(component).unwrap().tensors;
        let at = tensors.iter().flat_map(|t| t.locations()).map(|(_, at)| at);
        at.flat_map(|at| header.spans(&at)).map(|span| header.chunks[span.chunk].clone()).collect()
    };
    assert!(wanted("allowed").iter().all(|object| chunks.contains(object)));
    let denied_only = wanted("denied").into_iter().filter(|o| !wanted("allowed").contains(o));
    assert!(denied_only.into_iter().all(|object| !chunks.contains(&object)));
    for (file, object) in sources.files(&chunks).unwrap().iter().zip(&chunks) {
        assert_eq!(file.metadata().unwrap().len(), object.length);
    }
    let none = ModelSources::open(&root, &[]).unwrap();
    assert_eq!(none.chunks(&manifest).err().unwrap().kind(), io::ErrorKind::PermissionDenied);
    drop((sources, none));
    fs::remove_dir_all(root).unwrap();
}

fn frame(manifest: &str, name: &str) -> tensord::device_executor::Frame {
    serde_json::from_value(serde_json::json!({
        "event": "request", "seq": 7, "kind": "model_source", "manifest": manifest, "name": name,
    }))
    .unwrap()
}

/// What `model_source` hands an executor: the header, the manifest (the file list) and each
/// asset, verified, in a sealed read-only memfd with its digest; nothing outside the selection.
#[test]
fn the_selection_serves_its_header_and_assets_sealed_and_nothing_else() {
    let (root, manifest, _, _, _) = fixture();
    let selected = SelectedManifest {
        manifest: manifest.clone(),
        components: vec!["allowed".into()],
    };
    let sources = ModelSources::open(&root, std::slice::from_ref(&selected)).unwrap();
    let listing = tensorfs_core::store2::Store::open(&root).unwrap().read(&tensorfs_core::format::ObjectRef {
        sha256: manifest.trim_start_matches("sha256:").into(),
        length: sources.authorized(&manifest).unwrap().id().length,
    });
    let listing = listing.unwrap();
    assert!(tensorfs_core::format::Manifest::from_json(&listing).unwrap().files.contains_key("tokenizer/vocab.json"));
    for (name, want) in [
        ("", None),
        (tensord::model_sources::LISTING, Some(listing.as_slice())),
        ("tokenizer/vocab.json", Some(&b"static tokenizer asset"[..])),
    ] {
        let (answer, file) = sources.serve(&frame(&manifest, name)).unwrap();
        let mut got = vec![0; answer.length as usize];
        file.read_exact_at(&mut got, 0).unwrap();
        assert!(answer.ok && answer.held);
        assert_eq!(answer.sha256, format!("{:x}", Sha256::digest(&got)));
        match want {
            Some(bytes) => assert_eq!(got, bytes),
            None => assert!(tensorfs_core::format::Header::from_cbor(&got).is_ok(), "the header"),
        }
        // SAFETY: scalar fcntl queries on a descriptor we hold.
        #[cfg(unix)]
        unsafe {
            assert_eq!(
                libc::fcntl(file.as_raw_fd(), libc::F_GETFL) & libc::O_ACCMODE,
                libc::O_RDONLY
            );
            #[cfg(target_os = "linux")]
            assert_eq!(
                libc::fcntl(file.as_raw_fd(), libc::F_GET_SEALS) & 0x8,
                0x8,
                "write-sealed"
            );
        }
        // Windows: the handle is read-only, so a write through it is refused.
        #[cfg(windows)]
        assert!(std::io::Write::write_all(&mut &file, b"x").is_err(), "read-only");
    }
    assert!(
        sources.serve(&frame(&manifest, "../tfs.sqlite")).is_err(),
        "only declared assets"
    );
    let other = "sha256:".to_string() + &"0".repeat(64);
    assert!(
        sources.serve(&frame(&other, "")).is_err(),
        "only selected manifests"
    );
    drop(sources);
    fs::remove_dir_all(root).unwrap();
}
