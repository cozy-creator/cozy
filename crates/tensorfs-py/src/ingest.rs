//! Bounded-disk ingest: an exact provider selection converted while its bodies arrive and
//! leave, with output held by the caller's publication custodian instead of this disk.

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use tensorfs_core::err::{refuse, Code};
use tensorfs_core::ids::{hex64, ObjectRef};
use tensorfs_core::ingest::custody;
use tensorfs_core::ingest::source::{
    self as model_source, ModelSourceProfile, SelectedSourceMember,
};
use tensorfs_core::{providers, transport};

use crate::errors::IntoPy;
use crate::store::PyStore;

fn object(py: Python<'_>, id: &str, length: u64) -> PyResult<ObjectRef> {
    let sha256 = id.strip_prefix("sha256:").unwrap_or(id).to_string();
    hex64("ObjectRef", &sha256).or_refuse(py)?;
    Ok(ObjectRef { sha256, length })
}

/// Ranged header reads for exact pinned members: index documents whole, tensor carriers up
/// to their declared header. No body moves. Every member must answer.
#[pyfunction]
#[pyo3(signature = (source_uri, members, *, credential=String::new(), huggingface=None, civitai=None, allow_local=false))]
pub fn read_source_heads<'p>(
    py: Python<'p>,
    source_uri: &str,
    members: Vec<(String, String, u64, String)>,
    credential: String,
    huggingface: Option<String>,
    civitai: Option<String>,
    allow_local: bool,
) -> PyResult<Bound<'p, PyList>> {
    let uri = providers::SourceUri::parse(source_uri).or_refuse(py)?;
    let mut endpoints = providers::Endpoints::default();
    if let Some(value) = huggingface {
        endpoints.huggingface = value;
    }
    if let Some(value) = civitai {
        endpoints.civitai = value;
    }
    endpoints.allow_local = allow_local;
    let host = transport::base_host(match &uri {
        providers::SourceUri::HuggingFace { .. } => &endpoints.huggingface,
        providers::SourceUri::Civitai { .. } => &endpoints.civitai,
    })
    .or_refuse(py)?;
    let credentials = transport::credential_from_spec(&credential, vec![host]).or_refuse(py)?;
    let resolution = providers::Resolution {
        canonical: String::new(),
        selection_sha256: String::new(),
        members: members
            .into_iter()
            .map(|(member, id, length, url)| {
                Ok(providers::ResolvedMember {
                    member,
                    object: object(py, &id, length)?,
                    url,
                    provenance: providers::Provenance::Declared,
                    carrier: true,
                    requires: vec![],
                    companion: false,
                })
            })
            .collect::<PyResult<Vec<_>>>()?,
    };
    let read = py
        .detach(|| {
            let read = providers::read_heads(
                &resolution,
                &uri,
                &endpoints,
                &credentials,
                transport::Deadline::none(),
            )?;
            if let Some(unread) = read.unread.first() {
                return refuse(
                    Code::DURABILITY_UNPROVEN,
                    format!("{}: header unreadable: {}", unread.member, unread.why),
                );
            }
            Ok(read)
        })
        .or_refuse(py)?;
    let out = PyList::empty(py);
    for head in read.heads {
        out.append((head.member, PyBytes::new(py, &head.head)))?;
    }
    Ok(out)
}

/// The one reviewed profile these `(member, length, header)` heads plan under. A caller
/// that names no profile converts with this one.
#[pyfunction]
#[pyo3(signature = (store, heads, *, registry=None))]
pub fn select_source_profile(
    py: Python<'_>,
    store: &PyStore,
    heads: Vec<(String, u64, Vec<u8>)>,
    registry: Option<Vec<u8>>,
) -> PyResult<String> {
    let heads: Vec<providers::MemberHead> = heads
        .into_iter()
        .map(|(member, length, head)| providers::MemberHead {
            member,
            length,
            head,
        })
        .collect();
    py.detach(|| {
        let (locator, bytes) = model_source::effective_registry(&store.store, registry.as_deref())?;
        let scratch = store.store.root().join("tmp").join(format!(
            "select-profile-{}",
            tensorfs_core::meta::now_nanos_unique()
        ));
        tensorfs_core::ingest::preflight::select_profile(&locator, &bytes, &heads, &scratch)
    })
    .or_refuse(py)
}

/// `(converter, reference)` for every converter the named profiles bind, sorted and unique.
/// A reference is the pinned repository whose configs and tokenizers the output needs.
#[pyfunction]
#[pyo3(signature = (store, profiles, *, registry=None))]
pub fn source_profile_converters(
    py: Python<'_>,
    store: &PyStore,
    profiles: Vec<String>,
    registry: Option<Vec<u8>>,
) -> PyResult<Vec<(String, Option<String>)>> {
    py.detach(|| {
        let (_, bytes) = model_source::effective_registry(&store.store, registry.as_deref())?;
        let mut out = Vec::new();
        for profile in &profiles {
            for conv in model_source::profile_converters(&bytes, profile)? {
                let row = (conv.name.to_string(), conv.reference.map(str::to_string));
                if !out.contains(&row) {
                    out.push(row);
                }
            }
        }
        out.sort();
        Ok(out)
    })
    .or_refuse(py)
}

/// Advance one exact selection against whatever bodies the Store holds now. `members` are
/// `(member, object id, length, header)`; the result is `prepare_model_source`'s plus the
/// members counted as landed.
#[pyfunction]
#[pyo3(signature = (store, operation_id, source_selection_digest, profiles, members, *, registry=None, write_budget_bytes=None))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn prepare_selected_source<'p>(
    py: Python<'p>,
    store: &PyStore,
    operation_id: &str,
    source_selection_digest: &str,
    profiles: Vec<(String, String)>,
    members: Vec<(String, String, u64, Vec<u8>)>,
    registry: Option<Vec<u8>>,
    write_budget_bytes: Option<u64>,
) -> PyResult<Bound<'p, PyDict>> {
    let members = members
        .into_iter()
        .map(|(member, id, length, header)| {
            Ok(SelectedSourceMember {
                member,
                object: object(py, &id, length)?,
                header,
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    let profiles = profiles
        .into_iter()
        .map(|(slot, profile)| ModelSourceProfile { slot, profile })
        .collect();
    let (prepared, landed) = py
        .detach(|| {
            let (locator, bytes) =
                model_source::effective_registry(&store.store, registry.as_deref())?;
            model_source::prepare_selected_source(
                &store.store,
                operation_id,
                source_selection_digest,
                profiles,
                &members,
                &locator,
                &bytes,
                write_budget_bytes,
            )
        })
        .or_refuse(py)?;
    let out = PyDict::new(py);
    out.set_item("replayed", prepared.replayed)?;
    out.set_item("complete", prepared.complete)?;
    out.set_item("spent", prepared.spent)?;
    out.set_item("landed", landed)?;
    out.set_item("converted_roles", prepared.converted_roles)?;
    out.set_item("resumed_roles", prepared.resumed_roles)?;
    out.set_item("deferred_ops", prepared.deferred_ops)?;
    out.set_item("converted_bytes", prepared.converted_bytes)?;
    out.set_item("largest_op_bytes", prepared.largest_op_bytes)?;
    out.set_item("required_write_bytes", prepared.required_write_bytes)?;
    let rows = PyList::empty(py);
    for source in prepared.sources {
        let row = PyDict::new(py);
        row.set_item("slot", source.slot)?;
        row.set_item("profile", source.profile)?;
        row.set_item("manifest_digest", source.manifest_digest)?;
        row.set_item("manifest_length", source.manifest_length)?;
        rows.append(row)?;
    }
    out.set_item("sources", rows)?;
    Ok(out)
}

/// Record that the custodian named by `evidence` holds these journalled objects. GC may
/// drop their local copies from then on; conversion and derivation still count them.
#[pyfunction]
pub fn record_model_source_custody(
    py: Python<'_>,
    store: &PyStore,
    operation_id: &str,
    objects: Vec<(String, u64)>,
    evidence: &str,
) -> PyResult<()> {
    let objects = objects
        .into_iter()
        .map(|(id, length)| object(py, &id, length))
        .collect::<PyResult<Vec<_>>>()?;
    py.detach(|| custody::record(&store.store, operation_id, &objects, evidence))
        .or_refuse(py)
}

/// Every object this operation's conversion journal names: whether its body is resident
/// and, once recorded, the custody evidence.
#[pyfunction]
pub fn model_source_objects<'p>(
    py: Python<'p>,
    store: &PyStore,
    operation_id: &str,
) -> PyResult<Bound<'p, PyList>> {
    let (objects, held) = py
        .detach(|| {
            let objects = custody::journalled(store.store.root(), operation_id)?;
            let held = custody::read(store.store.root(), operation_id)?;
            Ok(objects
                .into_iter()
                .map(|object| {
                    let resident = store.store.contains(&object.sha256);
                    (object, resident)
                })
                .collect::<Vec<_>>())
            .map(|objects| (objects, held))
        })
        .or_refuse(py)?;
    let evidence: std::collections::BTreeMap<String, String> = held
        .rows()
        .into_iter()
        .map(|(object, evidence)| (object.sha256, evidence))
        .collect();
    let out = PyList::empty(py);
    for (object, resident) in objects {
        let row = PyDict::new(py);
        row.set_item("object_id", object.id())?;
        row.set_item("length", object.length)?;
        row.set_item("resident", resident)?;
        row.set_item("custody", evidence.get(&object.sha256))?;
        out.append(row)?;
    }
    Ok(out)
}
