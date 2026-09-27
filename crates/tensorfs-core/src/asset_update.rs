//! Metadata-only model-asset updates.
//!
//! An asset update rewrites only the CozyTensors header and manifest.  Tensor part
//! ObjectRefs are copied verbatim; asset source objects are selected from an existing
//! manifest and are never opened or rehashed by this operation.  Payload verification
//! remains the normal `read_asset` responsibility (or the repository admission gate).

use std::collections::BTreeMap;

use crate::checkpoint;
use crate::err::{refuse, Code, Refusal, Result};
use crate::header::Asset;
use crate::ids::ObjectRef;
use crate::manifest::Manifest;
use crate::store::Store;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Edit {
    /// Add or replace one named asset from an exact ordinary-file entry in the source
    /// manifest.  The source ObjectRef becomes the sole segment; no payload is read.
    Copy {
        name: String,
        source_path: String,
        media_type: String,
    },
    /// Inherit one already-committed asset from the source checkpoint header.
    /// The asset declaration and segment ObjectRefs are copied verbatim; no payload is read.
    Inherit { name: String, source_name: String },
    /// Remove an asset by its exact header name.  Missing names are harmless, making
    /// retries deterministic.
    Drop { name: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResultSnapshot {
    pub header: ObjectRef,
    pub manifest: ObjectRef,
}

fn source_object(manifest: &Manifest, path: &str) -> Result<ObjectRef> {
    manifest
        .entries()
        .iter()
        .find_map(|(candidate, entry)| {
            (candidate == path).then(|| entry.materializable(path).cloned())
        })
        .unwrap_or_else(|| {
            Err(Refusal {
                code: Code::OBJECT_ABSENT,
                detail: format!("asset source {path:?} is absent from the source manifest"),
            })
        })
}

/// Apply add/replace/drop edits to one existing checkpoint without reading tensor payloads.
///
/// The source manifest and header are read, and each selected source object is checked only
/// through its existing verification record and length.  The new header and manifest are the
/// only bytes written.  Callers that need a payload guarantee should subsequently use
/// [`crate::checkpoint::read_asset`], which hashes the reassembled asset once.
pub fn update(
    store: &Store,
    source_manifest: &ObjectRef,
    edits: &[Edit],
) -> Result<ResultSnapshot> {
    let source = checkpoint::load_manifest(store, source_manifest)?;
    let header_ref = source.header().ok_or_else(|| Refusal {
        code: Code::ATTACHMENT_CARDINALITY,
        detail: "asset update source has no CozyTensors header".into(),
    })?;
    let mut header = checkpoint::load_header(store, header_ref)?;
    let source_assets: BTreeMap<String, Asset> = header.assets.iter().cloned().collect();
    let closure = checkpoint::load_closure(store, &header)?;

    let mut assets: BTreeMap<String, Asset> = header.assets.drain(..).collect();
    for edit in edits {
        match edit {
            Edit::Drop { name } => {
                assets.remove(name);
            }
            Edit::Inherit { name, source_name } => {
                let asset = source_assets.get(source_name).ok_or_else(|| Refusal {
                    code: Code::OBJECT_ABSENT,
                    detail: format!("source asset {source_name:?} is absent"),
                })?;
                assets.insert(name.clone(), asset.clone());
            }
            Edit::Copy {
                name,
                source_path,
                media_type,
            } => {
                if name.is_empty() || media_type.is_empty() {
                    return refuse(
                        Code::MISSING_FIELD,
                        "asset name and media_type are required",
                    );
                }
                let object = source_object(&source, source_path)?;
                // This consults only the durable record's inode/length facts.  It does not
                // open or hash the payload, which is the defining property of this operation.
                let record = store
                    .record_valid(&object.sha256)
                    .map_err(|detail| Refusal {
                        code: Code::OBJECT_CORRUPT,
                        detail: format!("asset source {} is not verified: {detail}", object.id()),
                    })?;
                if record.length != object.length {
                    return refuse(
                        Code::LENGTH_MISMATCH,
                        format!("asset source {} record length differs", object.id()),
                    );
                }
                assets.insert(
                    name.clone(),
                    Asset {
                        logical_sha256: object.sha256.clone(),
                        logical_length: object.length,
                        media_type: media_type.clone(),
                        segments: vec![object],
                    },
                );
            }
        }
    }
    header.assets = assets.into_iter().collect();
    header.validate(&closure)?;
    let header_object = checkpoint::put_doc(store, &header)?;
    let manifest = checkpoint::build_manifest(&header, &header_object, &closure)?;
    let manifest_object = store.put_manifest(&manifest)?.obj;
    Ok(ResultSnapshot {
        header: header_object,
        manifest: manifest_object,
    })
}

/// Inherit selected runtime assets from one exact source checkpoint into another
/// metadata-only derived checkpoint. Tensor and asset payloads are never opened.
pub fn inherit(
    store: &Store,
    target_manifest: &ObjectRef,
    source_manifest: &ObjectRef,
    names: &[String],
) -> Result<ResultSnapshot> {
    let target = checkpoint::load_manifest(store, target_manifest)?;
    let target_header_ref = target.header().ok_or_else(|| Refusal {
        code: Code::ATTACHMENT_CARDINALITY,
        detail: "asset inheritance target has no CozyTensors header".into(),
    })?;
    let source = checkpoint::load_manifest(store, source_manifest)?;
    let source_header_ref = source.header().ok_or_else(|| Refusal {
        code: Code::ATTACHMENT_CARDINALITY,
        detail: "asset inheritance source has no CozyTensors header".into(),
    })?;
    let mut header = checkpoint::load_header(store, target_header_ref)?;
    let source_header = checkpoint::load_header(store, source_header_ref)?;
    let target_closure = checkpoint::load_closure(store, &header)?;
    let source_assets: BTreeMap<String, Asset> = source_header.assets.into_iter().collect();
    let mut assets: BTreeMap<String, Asset> = header.assets.drain(..).collect();
    for name in names {
        let asset = source_assets.get(name).ok_or_else(|| Refusal {
            code: Code::OBJECT_ABSENT,
            detail: format!("source asset {name:?} is absent"),
        })?;
        assets.insert(name.clone(), asset.clone());
    }
    header.assets = assets.into_iter().collect();
    header.validate(&target_closure)?;
    let header_object = checkpoint::put_doc(store, &header)?;
    let manifest = checkpoint::build_manifest(&header, &header_object, &target_closure)?;
    let manifest_object = store.put_manifest(&manifest)?.obj;
    Ok(ResultSnapshot {
        header: header_object,
        manifest: manifest_object,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint;
    use crate::header::{Body, Header, Part, Tensor};
    use crate::manifest::Draft;
    use crate::manifest::Entry;
    use crate::store::Fault;
    use crate::{dtype::Dtype, registry};

    fn temp() -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "tensorfs-asset-update-{}",
            crate::meta::now_nanos_unique()
        ))
    }

    fn put(store: &Store, bytes: &[u8]) -> ObjectRef {
        let expected = ObjectRef::of(bytes);
        let mut input = bytes;
        store
            .put_stream(&mut input, Some(&expected), &Fault::default())
            .unwrap()
            .obj
    }

    #[test]
    fn copies_asset_refs_without_changing_tensor_refs_and_read_asset_verifies_bytes() {
        let root = temp();
        let store = Store::init(&root).unwrap();
        let spec = registry::seeds()
            .into_iter()
            .find(|seed| seed.alias == "plain/1")
            .unwrap()
            .spec;
        let header = Header {
            configs: vec![],
            assets: vec![],
            encodings: vec![spec.clone()],
            components: vec![(
                "model".into(),
                vec![(
                    "weight".into(),
                    Tensor {
                        dtype: Dtype::U8,
                        shape: vec![3],
                        encoding: spec.object_id(),
                        parts: vec![(
                            "value".into(),
                            Part {
                                dtype: Dtype::U8,
                                shape: vec![3],
                                body: Body::Inline(vec![1, 2, 3]),
                            },
                        )],
                    },
                )],
            )],
        };
        let header_ref = put(&store, &header.canonical_bytes().unwrap());
        let asset_bytes = b"tokenizer bytes";
        let asset_ref = put(&store, asset_bytes);
        let source = Draft {
            entries: vec![
                ("model.cozytensors".into(), Entry::CozyTensors(header_ref)),
                ("tokenizer/vocab.txt".into(), Entry::File(asset_ref.clone())),
            ],
        }
        .seal()
        .unwrap();
        let source_ref = store.put_manifest(&source).unwrap().obj;

        let result = update(
            &store,
            &source_ref,
            &[Edit::Copy {
                name: "tokenizer/vocab.txt".into(),
                source_path: "tokenizer/vocab.txt".into(),
                media_type: "text/plain".into(),
            }],
        )
        .unwrap();
        let output = checkpoint::load_header(&store, &result.header).unwrap();
        assert_eq!(output.tensors().count(), 1);
        assert_eq!(
            output.tensors().next().unwrap().2.parts[0].1.body,
            Body::Inline(vec![1, 2, 3])
        );
        let asset = &output.assets[0].1;
        assert_eq!(asset.segments, vec![asset_ref]);
        assert_eq!(asset.logical_length, asset_bytes.len() as u64);
        let mut verified = Vec::new();
        checkpoint::read_asset(&store, "tokenizer/vocab.txt", asset, &mut verified).unwrap();
        assert_eq!(verified, asset_bytes);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn drops_are_deterministic_and_do_not_touch_payloads() {
        let root = temp();
        let store = Store::init(&root).unwrap();
        let spec = registry::seeds().into_iter().next().unwrap().spec;
        let payload = put(&store, b"asset");
        let header = Header {
            configs: vec![],
            assets: vec![(
                "old".into(),
                Asset {
                    logical_sha256: payload.sha256.clone(),
                    logical_length: payload.length,
                    media_type: "text/plain".into(),
                    segments: vec![payload.clone()],
                },
            )],
            encodings: vec![spec.clone()],
            components: vec![(
                "model".into(),
                vec![(
                    "w".into(),
                    Tensor {
                        dtype: Dtype::U8,
                        shape: vec![1],
                        encoding: spec.object_id(),
                        parts: vec![(
                            "value".into(),
                            Part {
                                dtype: Dtype::U8,
                                shape: vec![1],
                                body: Body::Inline(vec![0]),
                            },
                        )],
                    },
                )],
            )],
        };
        let href = put(&store, &header.canonical_bytes().unwrap());
        let source = Draft {
            entries: vec![("model.cozytensors".into(), Entry::CozyTensors(href))],
        }
        .seal()
        .unwrap();
        let source_ref = store.put_manifest(&source).unwrap().obj;
        let one = update(
            &store,
            &source_ref,
            &[Edit::Drop {
                name: "missing".into(),
            }],
        )
        .unwrap();
        let two = update(
            &store,
            &source_ref,
            &[Edit::Drop {
                name: "missing".into(),
            }],
        )
        .unwrap();
        assert_eq!(one.header, two.header);
        assert_eq!(one.manifest, two.manifest);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn inherits_existing_asset_refs_without_payload_reads() {
        let root = temp();
        let store = Store::init(&root).unwrap();
        let spec = registry::seeds().into_iter().next().unwrap().spec;
        let payload = put(&store, b"tokenizer");
        let header = Header {
            configs: vec![],
            assets: vec![(
                "tokenizer/vocab.json".into(),
                Asset {
                    logical_sha256: payload.sha256.clone(),
                    logical_length: payload.length,
                    media_type: "application/json".into(),
                    segments: vec![payload.clone()],
                },
            )],
            encodings: vec![spec.clone()],
            components: vec![(
                "model".into(),
                vec![(
                    "w".into(),
                    Tensor {
                        dtype: Dtype::U8,
                        shape: vec![1],
                        encoding: spec.object_id(),
                        parts: vec![(
                            "value".into(),
                            Part {
                                dtype: Dtype::U8,
                                shape: vec![1],
                                body: Body::Inline(vec![0]),
                            },
                        )],
                    },
                )],
            )],
        };
        let href = put(&store, &header.canonical_bytes().unwrap());
        let source = Draft {
            entries: vec![("model.cozytensors".into(), Entry::CozyTensors(href))],
        }
        .seal()
        .unwrap();
        let source_ref = store.put_manifest(&source).unwrap().obj;
        let result = update(
            &store,
            &source_ref,
            &[Edit::Inherit {
                name: "tokenizer/vocab.json".into(),
                source_name: "tokenizer/vocab.json".into(),
            }],
        )
        .unwrap();
        let output = checkpoint::load_header(&store, &result.header).unwrap();
        assert_eq!(output.assets[0].1.segments, vec![payload]);
        std::fs::remove_dir_all(root).unwrap();
    }
}
