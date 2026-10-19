//! The run output log's products: what a run has made so far, as it makes it (wire 65).
//! `Outputs.publish` commits the asset's bytes into native custody, binds them to the run's
//! actor and journals one `product` on the run's log, in that order, so every product a
//! client reads can be fetched with ReadByteTreeObject. A single output's product is replaced
//! by each publish (SET); a list output grows (APPEND). The returned result publishes only
//! what the log does not already show. Runtime checkpoint outputs carry immutable TensorFS
//! references only; TensorD records them without adopting or uploading their bytes.
use crate::{
    api::domain,
    device_executor::{Answer, Frame},
    execution::Engine,
    journal::StoredProduct,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{self, Read},
    path::{Component, Path, PathBuf},
};
use crate::storage::{self, Store};
use tensorfs_core::{format::ObjectRef, sha256};

const MAX_PARTS: usize = 128;

fn storage(error: impl std::fmt::Display) -> io::Error {
    io::Error::other(error.to_string())
}

/// Retain one byte string as a one-file tree named by `owner`, bound to `actor` so a reader
/// with that actor's capability can fetch it. Idempotent per owner.
pub fn retain(
    store: &Store,
    engine: &Engine,
    actor: &str,
    owner: &str,
    source: &mut File,
    object: &ObjectRef,
) -> io::Result<domain::NativeByteRetentionRequest> {
    // Bytes the store already holds (a product published while the run made it, a memoized
    // or child result) are not read again.
    if !store.reuse(object).map_err(storage::io_error)? {
        store.put(source, Some(object)).map_err(storage::io_error)?;
    }
    let tree = crate::trees::put(store, owner, [("payload".to_string(), object.clone())].into())?;
    let native = domain::NativeByteRetentionRequest {
        source: Some(domain::NativeByteTreeRef {
            producer_root_id: tree.pin,
            receipt_digest: unhex(&tree.manifest.sha256)?,
            manifest: Some(domain::Ref {
                digest: unhex(&tree.manifest.sha256)?,
                length: tree.manifest.length,
            }),
            content_bytes: object.length,
        }),
        retention_id: owner.into(),
    };
    engine.bind_native_output(actor, owner, &crate::archive::encode_native_byte_retention_request(&native))?;
    Ok(native)
}

fn unhex(text: &str) -> io::Result<Vec<u8>> {
    if text.len() != 64 {
        return Err(storage("digest length differs"));
    }
    (0..64)
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).map_err(storage))
        .collect()
}

/// A published asset's bytes: a regular file of the attempt's spool, at most one directory
/// down (a join's parts). Never a symlink or a path outside the spool.
fn spool_file(spool: &Path, name: &str) -> io::Result<PathBuf> {
    let refuse = || {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "the published asset names no file in this attempt's spool",
        )
    };
    let relative = Path::new(name);
    let steps: Vec<_> = relative.components().collect();
    if steps.is_empty()
        || steps.len() > 2
        || steps.iter().any(|step| {
            !matches!(step, Component::Normal(part) if !part.to_string_lossy().starts_with('.'))
        })
    {
        return Err(refuse());
    }
    let path = spool.join(relative);
    if steps.len() == 2
        && spool
            .join(steps[0])
            .symlink_metadata()?
            .file_type()
            .is_symlink()
    {
        return Err(refuse());
    }
    let metadata = path.symlink_metadata().map_err(|_| refuse())?;
    if !metadata.file_type().is_file() {
        return Err(refuse());
    }
    Ok(path)
}

/// A saved asset's spool file, named as the SDK's post phase names it:
/// `attempt:<request>/<kind>/<n>` lives at `<kind>-<n>`. A file a child returned is named by
/// its digest, `sha256:<hex>`, and linked into its parent's spool as `sha256-<hex>`
/// (`calls::receive`), so a caller can publish it as it is.
fn spool_name(asset_ref: &str) -> String {
    if let Some(hex) = asset_ref
        .strip_prefix("sha256:")
        .filter(|hex| hex.len() == 64 && hex.bytes().all(|b| b.is_ascii_hexdigit()))
    {
        return format!("sha256-{hex}");
    }
    let tail: Vec<_> = asset_ref.rsplitn(3, '/').collect();
    if tail.len() == 3 {
        format!("{}-{}", tail[1], tail[0])
    } else {
        String::new()
    }
}

fn hash_file(path: &Path) -> io::Result<ObjectRef> {
    let mut file = File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0; 1 << 20];
    let mut length = 0u64;
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
        length += count as u64;
    }
    Ok(ObjectRef {
        sha256: sha256::hex(&hash.finalize()),
        length,
    })
}

/// How a declared result field takes products: a single asset is SET, a list of assets grows
/// by APPEND. Anything else is not a publishable output.
fn output_op(result: &Value, output: &str) -> Option<domain::RunProductOp> {
    let mut schema = result;
    for name in output.split('.') {
        schema = schema
            .get("fields")?
            .as_array()?
            .iter()
            .find(|field| field.get("name").and_then(Value::as_str) == Some(name))?
            .get("type")?;
    }
    if schema.get("asset").is_some() {
        Some(domain::RunProductOp::Set)
    } else if schema.get("list")?.get("asset").is_some() {
        Some(domain::RunProductOp::Append)
    } else {
        None
    }
}

pub fn decode(stored: &StoredProduct) -> io::Result<domain::RunProduct> {
    crate::archive::decode_run_product(stored.product.as_slice())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "journaled product is corrupt"))
}

/// The SDK's `publish` request during a running attempt: answers `Published`.
pub fn publish(store: &Store, engine: &Engine, id: &str, spool: &Path, frame: &Frame) -> Answer {
    let mut answer = Answer::unavailable(frame.seq);
    match commit(store, engine, id, spool, frame) {
        Ok((content, sequence)) => {
            answer.ok = true;
            answer.code.clear();
            answer.detail.clear();
            answer.digest = format!("sha256:{}", sha256::hex(&content.digest));
            answer.length = content.length;
            answer.sequence = sequence;
        }
        Err(error) => {
            answer.code = match error.kind() {
                io::ErrorKind::InvalidInput => "output_undeclared",
                _ => "publish_refused",
            }
            .into();
            answer.detail = error.to_string();
        }
    }
    answer
}

/// The running execution's submission and its entrypoint's declaration.
fn declared(engine: &Engine, id: &str) -> io::Result<(crate::journal::SubmissionContext, Value)> {
    let record = engine.get(id)?;
    let context = record
        .submission
        .ok_or_else(|| io::Error::other("only a public execution has an output log"))?;
    let installed = engine
        .installation_for_generation(&context.actor, &record.invocation.generation)?
        .ok_or_else(|| io::Error::other("held installation interface absent"))?;
    let interface: Value = serde_json::from_slice(&installed.interface).map_err(storage)?;
    let entry = interface
        .get("entrypoints")
        .and_then(Value::as_array)
        .and_then(|rows| rows.iter().find(|row| row.get("name").and_then(Value::as_str) == Some(&record.invocation.entrypoint)))
        .cloned()
        .ok_or_else(|| io::Error::other("declared entrypoint absent"))?;
    Ok((context, entry))
}

/// Record Runtime's checkpoint reference. A later upload replaces the local reference;
/// a published remote tree need not be present in TensorD's cache.
pub fn checkpoint(engine: &Engine, id: &str, frame: &Frame) -> Answer {
    let record = || -> io::Result<u64> {
        let (_, entry) = declared(engine, id)?;
        if !entry["weights_outputs"].as_array().into_iter().flatten()
            .any(|row| row["output_id"].as_str() == Some(frame.output.as_str())) {
            return Err(io::Error::new(io::ErrorKind::InvalidInput,
                format!("{:?} is not a declared weights output", frame.output)));
        }
        if frame.repo_id.is_empty() {
            return Err(storage("a checkpoint reference needs its repository"));
        }
        storage::check_digest("checkpoint revision", &frame.revision, true)?;
        let reference = protocol::machine::CheckpointRef {
            repo_id: frame.repo_id.clone(), revision: frame.revision.clone(), variant: frame.variant.clone(),
        };
        let product = domain::RunProduct {
            output: frame.output.clone(), checkpoint: Some(reference.clone()),
            op: domain::RunProductOp::Set as i32, ..Default::default()
        };
        engine.append_product(id, |prior| {
            let prior = prior.iter().map(decode).collect::<io::Result<Vec<_>>>()?;
            let held = prior.iter().rev().find(|p| p.output == product.output)
                .is_some_and(|p| p.checkpoint.as_ref() == Some(&reference));
            Ok((!held).then(|| encode(&product)))
        })
    };
    match record() {
        Ok(sequence) => { let mut answer = Answer::ok(frame.seq); answer.sequence = sequence; answer }
        Err(error) => Answer::refused(frame.seq,
            if error.kind() == io::ErrorKind::InvalidInput { "output_undeclared" } else { "checkpoint_refused" },
            error.to_string()),
    }
}

fn commit(
    store: &Store,
    engine: &Engine,
    id: &str,
    spool: &Path,
    frame: &Frame,
) -> io::Result<(domain::Ref, u64)> {
    let (context, entry) = declared(engine, id)?;
    let result = entry.get("result").ok_or_else(|| io::Error::other("declared result schema absent"))?;
    let op = output_op(result, &frame.output).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "{:?} is not an asset field of this function's result",
                frame.output
            ),
        )
    })?;
    if frame.parts.len() > MAX_PARTS {
        return Err(storage(format!("a product has at most {MAX_PARTS} parts")));
    }
    let retain_file = |path: &Path| -> io::Result<(domain::Ref, domain::NativeByteRetentionRequest)> {
        let object = hash_file(path)?;
        // One hold per distinct byte string of the run; TensorFS names owners by digest.
        let named = format!(
            "cozy.product/1\0{}\0{}\0{}",
            engine.workspace_id(),
            context.request_id,
            object.sha256
        );
        let owner = format!("sha256:{}", sha256::hex(&sha256::digest(named.as_bytes())));
        let source = retain(
            store,
            engine,
            &context.actor,
            &owner,
            &mut File::open(path)?,
            &object,
        )?;
        Ok((
            domain::Ref {
                digest: unhex(&object.sha256)?,
                length: object.length,
            },
            source,
        ))
    };
    let mut product = domain::RunProduct {
        output: frame.output.clone(),
        op: op as i32,
        media_type: frame.media_type.clone(),
        label: frame.label.clone(),
        ..Default::default()
    };
    if frame.parts.is_empty() {
        let path = spool_file(spool, &spool_name(&frame.asset_ref))?;
        let (content, source) = retain_file(&path)?;
        product.content = Some(content);
        product.source = Some(source);
    } else {
        let mut whole = Sha256::new();
        let mut length = 0;
        for part in &frame.parts {
            let path = spool_file(spool, &part.local)?;
            let mut file = File::open(&path)?;
            length += io::copy(&mut file, &mut whole)?;
            let (content, source) = retain_file(&path)?;
            product.parts.push(domain::RunProductPart {
                content: Some(content),
                source: Some(source),
                duration_us: part.duration_us,
            });
        }
        product.content = Some(domain::Ref {
            digest: whole.finalize().to_vec(),
            length,
        });
    }
    let content = product.content.clone().unwrap_or_default();
    let sequence = engine.append_product(id, |prior| {
        let prior = prior.iter().map(decode).collect::<io::Result<Vec<_>>>()?;
        if op == domain::RunProductOp::Append {
            let items = prior
                .iter()
                .filter(|p| p.output == product.output && p.op == op as i32)
                .count();
            product.index = items as u32;
        } else if prior
            .iter()
            .rev()
            .find(|p| p.output == product.output && p.op == op as i32)
            .is_some_and(|current| current.content == product.content && current.checkpoint == product.checkpoint)
        {
            return Ok(None); // the log already shows these bytes
        }
        Ok(Some(encode(&product)))
    })?;
    Ok((content, sequence))
}

/// The canonical body of one product event.
pub fn document(product: &domain::RunProduct) -> io::Result<Value> {
    let missing = || io::Error::new(io::ErrorKind::InvalidData, "product reference absent");
    let reference = |content: &domain::Ref| json!({"digest":format!("sha256:{}",sha256::hex(&content.digest)),"length":content.length});
    let source = |source: &domain::NativeByteRetentionRequest| -> io::Result<Value> {
        let tree = source.source.as_ref().ok_or_else(missing)?;
        let manifest = tree.manifest.as_ref().ok_or_else(missing)?;
        Ok(
            json!({"retention_id":source.retention_id,"source":{"producer_root_id":tree.producer_root_id,"receipt_digest":format!("sha256:{}",sha256::hex(&tree.receipt_digest)),"manifest":reference(manifest),"content_bytes":tree.content_bytes}}),
        )
    };
    if let Some(checkpoint) = &product.checkpoint {
        return Ok(json!({"format":"cozy.machine.product/1", "output":product.output, "op":product.op,
            "checkpoint":checkpoint}));
    }
    let mut document = json!({"format":"cozy.machine.product/1","output":product.output,"op":product.op,
        "content":reference(product.content.as_ref().ok_or_else(missing)?),"media_type":product.media_type});
    if let Some(held) = &product.source {
        document["source"] = source(held)?;
    }
    if product.index != 0 {
        document["index"] = json!(product.index);
    }
    if !product.label.is_empty() {
        document["label"] = json!(product.label);
    }
    if !product.parts.is_empty() {
        document["parts"] = product
            .parts
            .iter()
            .map(|part| {
                let mut value =
                    json!({"content":reference(part.content.as_ref().ok_or_else(missing)?)});
                if let Some(held) = &part.source {
                    value["source"] = source(held)?;
                }
                if part.duration_us != 0 {
                    value["duration_us"] = json!(part.duration_us);
                }
                Ok(value)
            })
            .collect::<io::Result<Vec<_>>>()?
            .into();
    }
    Ok(document)
}

/// Whether the log already shows a returned product: the same bytes as the last SET of that
/// output, or an APPEND at an index the log already holds.
pub fn shown(log: &[domain::RunProduct], product: &domain::RunProduct) -> bool {
    if product.op == domain::RunProductOp::Append as i32 {
        log.iter()
            .any(|p| p.output == product.output && p.op == product.op && p.index == product.index)
    } else {
        log.iter()
            .rev()
            .find(|p| p.output == product.output && p.op == product.op)
            .is_some_and(|current| current.content == product.content && current.checkpoint == product.checkpoint)
    }
}

/// The product bytes held by the execution journal.
pub fn encode(product: &domain::RunProduct) -> Vec<u8> { crate::archive::encode_run_product(product) }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint_references_replace_and_survive_restart_without_byte_custody() {
        use crate::journal::{Installation, Invocation, SubmissionContext};
        let root = std::env::temp_dir().join(format!("checkpoint-reference-{}", uuid::Uuid::new_v4()));
        let engine = Engine::open(&root).unwrap();
        engine.bind_installation(Installation {
            actor: "owner".into(), alias: "package".into(), generation: "generation".into(),
            interface: serde_json::to_vec(&json!({"entrypoints":[{"name":"convert",
                "weights_outputs":[{"output_id":"weights","max_bytes":1024}]}]})).unwrap(),
            ..Default::default()
        }).unwrap();
        let record = engine.submit_public(SubmissionContext {
            actor: "owner".into(), request_id: "request".into(), submission_id: "submission".into(),
            expected_workspace_id: engine.workspace_id(), ..Default::default()
        }, Invocation { generation: "generation".into(), entrypoint: "convert".into(), ..Default::default() }).unwrap();
        engine.with_journal(|journal| {
            assert!(journal.claim(&record.id)?);
            journal.register_process(&record.id, platform::process::Birth::own()?)?;
            journal.running(&record.id, None).map(drop)
        }).unwrap();
        let mut frame = Frame {
            seq: 1, output: "weights".into(), repo_id: "local/derived".into(),
            revision: format!("sha256:{}", "a".repeat(64)), variant: "bf16".into(), ..Default::default()
        };
        assert!(checkpoint(&engine, &record.id, &frame).ok);
        assert_eq!(checkpoint(&engine, &record.id, &frame).sequence, 0);
        frame.repo_id = "owner/model".into();
        assert!(checkpoint(&engine, &record.id, &frame).ok);
        frame.output = "undeclared".into();
        assert_eq!(checkpoint(&engine, &record.id, &frame).code, "output_undeclared");
        drop(engine);
        let reopened = Engine::open(&root).unwrap();
        let rows = reopened.products(&record.id).unwrap();
        assert_eq!(rows.len(), 2);
        let local = decode(&rows[0]).unwrap();
        let remote = decode(&rows[1]).unwrap();
        assert_eq!(remote.checkpoint.as_ref().unwrap().repo_id, "owner/model");
        assert!(remote.content.is_none() && remote.source.is_none() && remote.parts.is_empty());
        let document = document(&remote).unwrap();
        assert!(document.get("content").is_none());
        assert_eq!(document["checkpoint"]["revision"], frame.revision);
        assert!(!shown(&[local], &remote));
        drop(reopened);
        crate::scratch::remove(&root);
    }
}
