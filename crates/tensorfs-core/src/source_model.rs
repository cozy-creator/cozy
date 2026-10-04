//! A provider source made into a local model in one call: resolve the immutable selection,
//! choose its reviewed profile from the headers, land its members resumably on the standard
//! downloader, prepare the selected source, compose the model with the configs and tokenizers
//! it is built from, and commit it as the local repository `local/<name>`.
//!
//! A source is `hf://<org>/<repo>@<40-hex>[/<member>]`, `civitai://<version>[/<file>]`, or
//! `object://sha256:<hex>/<file>`: a file this Store already holds (a client wrote it), made
//! like a Civitai single file with no provider at all.
//! A Hugging Face repository brings its Diffusers index, each component's config and its
//! tokenizers; a source with no index (a Civitai single file) takes them from its converters'
//! pinned reference repository. Members and the conversion journal stay retained until the
//! model is committed, so an interrupted call resumes where it stopped.

use std::collections::BTreeMap;
use std::io::Read;

use crate::canon::{self, Value};
use crate::derived::{self, ComponentDeclaration, ConfigDeclaration, Declaration, Lookup, Source};
use crate::err::{refuse, Code, Refusal, Result};
use crate::ids::{Doc, ObjectRef};
use crate::ingest::preflight;
use crate::ingest::source::{
    self as model_source, ModelSourceProfile, PreparedModelSource, SelectedSourceMember,
};
use crate::jcs::{self, Json};
use crate::meta::Meta;
use crate::providers::{self, Endpoints, MemberHead, Resolution, SourceUri};
use crate::repository::{Mutation, RepositoryName};
use crate::store::{Fault, Store};
use crate::transport::{self, Deadline, PullCancellation, SourcePolicy};

const FORMAT: &str = "cozy.source-model/1";
/// The documents a Hugging Face Diffusers repository names its pipeline components in.
const DIFFUSERS_INDEXES: [&str; 2] = ["model_index.json", "modular_model_index.json"];
const ASSET_KINDS: [&str; 3] = ["Tokenizer", "Processor", "FeatureExtractor"];
/// Config keys a saved config carries that name where it was saved from; a construction
/// config never names a source, so these are dropped.
const PATH_SUFFIXES: [&str; 5] = ["_path", "_dir", "_file", "_url", "_repo"];

/// How the provider is reached: its credential spec (`bearer <token>` or empty) and
/// endpoints (stand-ins in tests).
#[derive(Debug, Clone, Default)]
pub struct Access {
    pub credential: String,
    pub endpoints: Endpoints,
}

pub struct Request<'a> {
    /// `hf://…@<revision>[/member]`, `civitai://<version>[/file]` or `object://sha256:<hex>/<file>`.
    pub source: &'a str,
    /// Reviewed profiles to convert under; none: the one the headers select.
    pub profiles: &'a [String],
    /// The local repository's name.
    pub name: &'a str,
    pub access: &'a Access,
    /// Extends the built-in and Store-bound registry.
    pub registry: Option<&'a [u8]>,
    /// Stage, bytes done, bytes total (zero when unknown).
    pub progress: &'a (dyn Fn(&str, u64, u64) + Sync),
    pub cancellation: Option<PullCancellation>,
}

/// The committed model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Made {
    pub repository: String,
    pub manifest: ObjectRef,
    /// The canonical source the provider resolved.
    pub resolved: String,
    pub profiles: Vec<String>,
    /// `sha256:<hex>` of the exact selection.
    pub selection: String,
}

/// The source's provider URI and the carrier it names, if any.
fn parse(source: &str) -> Result<(SourceUri, Option<String>)> {
    if let Some(rest) = source.strip_prefix("hf://") {
        let at = rest.find('@').ok_or(Refusal {
            code: Code::SOURCE_NOT_ALLOWED,
            detail: "hf source is hf://<org>/<repo>@<40-hex revision>[/member]".into(),
        })?;
        let (pinned, member) = match rest[at..].find('/') {
            Some(slash) => (&rest[..at + slash], Some(&rest[at + slash + 1..])),
            None => (rest, None),
        };
        let member = member.filter(|m| !m.is_empty()).map(unquote);
        return Ok((SourceUri::parse(&format!("hf://{pinned}"))?, member));
    }
    if let Some(rest) = source.strip_prefix("civitai://") {
        let (version, file) = match rest.split_once('/') {
            Some((version, file)) => (version, Some(file.to_string())),
            None => (rest, None),
        };
        let file = file.filter(|f| !f.is_empty());
        return Ok((SourceUri::parse(&format!("civitai://{version}"))?, file));
    }
    refuse(
        Code::SOURCE_NOT_ALLOWED,
        "a model source is hf://<org>/<repo>@<revision>[/member] or civitai://<version>[/file]",
    )
}

fn unquote(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Ok(byte) = u8::from_str_radix(&value[i + 1..i + 3], 16) {
                out.push(byte);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8(out).unwrap_or_else(|_| value.to_string())
}

/// One provider reached with its credential, presented only to its own host.
struct Provider {
    uri: SourceUri,
    endpoints: Endpoints,
    credentials: transport::ScopedHeaders,
    policy: SourcePolicy,
}

impl Provider {
    fn new(uri: SourceUri, access: &Access, credential: &str) -> Result<Provider> {
        let base = match &uri {
            SourceUri::HuggingFace { .. } => &access.endpoints.huggingface,
            SourceUri::Civitai { .. } => &access.endpoints.civitai,
        };
        let host = transport::base_host(base)?;
        let credentials = transport::credential_from_spec(credential, vec![host])?;
        let policy = access.endpoints.policy(&uri);
        Ok(Provider {
            uri,
            endpoints: access.endpoints.clone(),
            credentials,
            policy,
        })
    }

    fn heads(&self, selection: &Resolution) -> Result<Vec<MemberHead>> {
        let read = providers::read_heads(
            selection,
            &self.uri,
            &self.endpoints,
            &self.credentials,
            Deadline::none(),
        )?;
        if let Some(unread) = read.unread.first() {
            return refuse(
                Code::DURABILITY_UNPROVEN,
                format!("{}: header unreadable: {}", unread.member, unread.why),
            );
        }
        Ok(read.heads)
    }

    /// The named ordinary files the pinned revision holds; one it lacks is not selected.
    fn files(&self, names: &[String]) -> Result<Option<Resolution>> {
        match providers::resolve_files(
            &self.uri,
            names,
            &self.endpoints,
            &self.credentials,
            Deadline::none(),
        ) {
            Ok(found) => Ok(Some(found)),
            Err(refusal) if refusal.code == Code::MISSING_FIELD => {
                let mut found: Option<Resolution> = None;
                for name in names.iter().filter(|_| names.len() > 1) {
                    if let Some(one) = self.files(std::slice::from_ref(name))? {
                        found = Some(match found {
                            Some(all) => merge(all, one),
                            None => one,
                        });
                    }
                }
                Ok(found)
            }
            Err(refusal) => Err(refusal),
        }
    }

    /// A Diffusers repository's index, each component's construction config and every
    /// tokenizer or processor file.
    fn diffusers(&self) -> Result<Option<Resolution>> {
        for index in DIFFUSERS_INDEXES {
            let Some(pinned) = self.files(&[index.to_string()])? else {
                continue;
            };
            let heads = self.heads(&pinned)?;
            let Some(body) = heads.first().map(|h| h.head.clone()) else {
                continue;
            };
            let components = components(&body)?;
            let mut names = vec![index.to_string()];
            let mut configs: Vec<_> = constructors(&components).into_values().collect();
            configs.sort();
            configs.dedup();
            names.extend(configs);
            names.extend(assets(&components));
            return self.files(&names);
        }
        Ok(None)
    }
}

/// `{component: (class, folder)}` from a classic or modular Diffusers index.
fn components(index: &[u8]) -> Result<BTreeMap<String, (String, String)>> {
    let field = |pairs: &[(String, Json)], key: &str| -> Option<Json> {
        pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v.clone())
    };
    let text = |value: Option<Json>| match value {
        Some(Json::Str(s)) => Some(s),
        _ => None,
    };
    let Json::Obj(document) = jcs::parse(index, crate::limits::DOC_MAX_BYTES)? else {
        return refuse(Code::WRONG_TYPE, "the Diffusers index is not an object");
    };
    let mut found = BTreeMap::new();
    for (name, entry) in &document {
        let Json::Arr(entry) = entry else { continue };
        if entry.len() < 2 || name.starts_with('_') {
            continue;
        }
        let spec = match entry.get(2) {
            Some(Json::Obj(spec)) => spec.clone(),
            _ => vec![],
        };
        let hint = match field(&spec, "type_hint") {
            Some(Json::Arr(hint)) if hint.len() == 2 => text(Some(hint[1].clone())),
            _ => None,
        };
        let kind = text(Some(entry[1].clone())).or(hint);
        let folder = text(field(&spec, "subfolder")).unwrap_or_else(|| name.clone());
        if let Some(kind) = kind {
            found.insert(name.clone(), (kind, folder));
        }
    }
    Ok(found)
}

fn asset(kind: &str) -> bool {
    ASSET_KINDS.iter().any(|word| kind.contains(word))
}

/// Each component's construction config file, by component: a model's `config.json`, a
/// scheduler's `scheduler_config.json`; tokenizers and processors are assets.
fn constructors(components: &BTreeMap<String, (String, String)>) -> BTreeMap<String, String> {
    components
        .iter()
        .filter(|(_, (kind, _))| !asset(kind))
        .map(|(name, (kind, folder))| {
            let file = if kind.ends_with("Scheduler") {
                "scheduler_config.json"
            } else {
                "config.json"
            };
            (name.clone(), format!("{folder}/{file}"))
        })
        .collect()
}

/// The folders of the index's tokenizers, processors and feature extractors.
fn assets(components: &BTreeMap<String, (String, String)>) -> Vec<String> {
    let mut folders: Vec<_> = components
        .values()
        .filter(|(kind, _)| asset(kind))
        .map(|(_, folder)| format!("{folder}/"))
        .collect();
    folders.sort();
    folders.dedup();
    folders
}

/// Members of two resolutions as one selection (metadata riding beside the carriers).
fn merge(mut into: Resolution, other: Resolution) -> Resolution {
    for row in other.members {
        if !into.members.iter().any(|m| m.member == row.member) {
            into.members.push(row);
        }
    }
    into.members.sort_by(|a, b| a.member.cmp(&b.member));
    into
}

/// Members whose header plans the conversion; ordinary files never are.
fn carrier(member: &str) -> bool {
    member.ends_with(".safetensors")
        || member.ends_with(".safetensors.index.json")
        || member
            .strip_prefix("civitai/files/")
            .is_some_and(|id| !id.is_empty() && id.bytes().all(|b| b.is_ascii_digit()))
}

/// One reviewed profile converts into `model`; several compose `part<i>` slots.
fn slots(profiles: &[String]) -> Result<Vec<ModelSourceProfile>> {
    let mut seen = profiles.to_vec();
    seen.sort();
    seen.dedup();
    if seen.len() != profiles.len() {
        return refuse(Code::DUPLICATE_KEY, "name distinct reviewed profiles");
    }
    Ok(match profiles {
        [one] => vec![ModelSourceProfile {
            slot: "model".into(),
            profile: one.clone(),
        }],
        _ => profiles
            .iter()
            .enumerate()
            .map(|(i, profile)| ModelSourceProfile {
                slot: format!("part{i}"),
                profile: profile.clone(),
            })
            .collect(),
    })
}

/// One identity over strings, as canonical bytes.
fn digest(parts: &[&str]) -> String {
    let parts = parts.iter().map(|part| Value::str(*part)).collect();
    crate::sha256::hex_digest(&canon::write(&Value::arr(parts)))
}

/// A landed carrier's header, read from the Store: an index whole, a SafeTensors file its
/// length prefix and header.
fn local_head(store: &Store, member: &str, object: &ObjectRef) -> Result<Vec<u8>> {
    let mut file = store.open_verified(&object.sha256)?.into_file();
    let io = |e: std::io::Error| Refusal {
        code: Code::IO_FAILED,
        detail: format!("read {member}: {e}"),
    };
    if member.ends_with(".json") {
        let mut body = Vec::new();
        file.take(crate::limits::DOC_MAX_BYTES as u64 + 1)
            .read_to_end(&mut body)
            .map_err(io)?;
        return Ok(body);
    }
    let mut prefix = [0u8; 8];
    file.read_exact(&mut prefix).map_err(io)?;
    let length = u64::from_le_bytes(prefix);
    if length > object.length.saturating_sub(8) || length > (256 << 20) {
        return refuse(
            Code::CARRIER_GEOMETRY,
            format!("{member}: SafeTensors header length {length} is outside the file"),
        );
    }
    let mut head = prefix.to_vec();
    head.resize(8 + length as usize, 0);
    file.read_exact(&mut head[8..]).map_err(io)?;
    Ok(head)
}

/// `object://sha256:<hex>/<file>`: one SafeTensors file the Store already holds, as a
/// selection of that one carrier.
fn held_object(store: &Store, source: &str, file: &str) -> Result<(Resolution, String)> {
    let (id, member) = file.split_once('/').ok_or(Refusal {
        code: Code::SOURCE_NOT_ALLOWED,
        detail: "a held source is object://sha256:<hex>/<file>".into(),
    })?;
    let sha256 = id.strip_prefix("sha256:").unwrap_or(id).to_string();
    crate::ids::hex64("held source", &sha256)?;
    let member = unquote(member);
    if !member.ends_with(".safetensors") || member.contains(['/', '\\']) {
        return refuse(
            Code::SOURCE_NOT_ALLOWED,
            format!("held source {member:?} is not one .safetensors file"),
        );
    }
    let length = store.open_verified(&sha256)?.len();
    let resolution = Resolution {
        canonical: source.to_string(),
        members: vec![providers::ResolvedMember {
            member: member.clone(),
            object: ObjectRef { sha256, length },
            url: String::new(),
            provenance: providers::Provenance::Computed,
            carrier: true,
            requires: vec![],
            companion: false,
        }],
        selection_sha256: String::new(),
    };
    Ok((resolution.select(std::slice::from_ref(&member))?, member))
}

/// The carriers' headers, read from the Store.
fn local_heads(store: &Store, selection: &Resolution) -> Result<Vec<MemberHead>> {
    selection
        .members
        .iter()
        .filter(|row| carrier(&row.member))
        .map(|row| {
            Ok(MemberHead {
                member: row.member.clone(),
                length: row.object.length,
                head: local_head(store, &row.member, &row.object)?,
            })
        })
        .collect()
}

/// The model `local/<name>` holds, when every byte of it is still held. Sources are
/// immutable spellings, so a held model is that source's model and needs no provider.
pub fn held(store: &Store, name: &str) -> Result<Option<(ObjectRef, String)>> {
    let repo = RepositoryName::new("local", name)?;
    let Ok(bytes) = std::fs::read(store.repository_path(&repo)) else {
        return Ok(None);
    };
    let repository = crate::repository::Repository::parse(&bytes)?;
    let checkpoint = repository.local_checkpoint()?;
    let selection = checkpoint.source_selection.clone().unwrap_or_default();
    if crate::checkpoint_root::check_source(store, &format!("local/{name}"), &checkpoint.manifest)
        .is_err()
    {
        return Ok(None);
    }
    Ok(Some((
        checkpoint.manifest.clone(),
        format!("sha256:{selection}"),
    )))
}

/// Make `request.source` the local repository `local/<name>`.
pub fn make(store: &Store, request: &Request) -> Result<Made> {
    let progress = request.progress;
    let held_file = request.source.strip_prefix("object://");
    let parsed = match held_file {
        Some(_) => None,
        None => Some(parse(request.source)?),
    };
    if let Some((manifest, selection)) = held(store, request.name)? {
        return Ok(Made {
            repository: format!("local/{}", request.name),
            manifest,
            resolved: request.source.to_string(),
            profiles: request.profiles.to_vec(),
            selection,
        });
    }
    let (locator, registry) = model_source::effective_registry(store, request.registry)?;

    // The exact carriers: the one named, else the one the source has, else the members of
    // the reviewed profiles the headers select.
    progress("resolving the model source", 0, 0);
    let none = Deadline::none();
    let (provider, resolution, named) = match (held_file, parsed) {
        (Some(file), _) => {
            let (resolution, member) = held_object(store, request.source, file)?;
            (None, resolution, Some(member))
        }
        (None, Some((uri, named))) => {
            let provider = Provider::new(uri, request.access, &request.access.credential)?;
            let resolution = match &named {
                Some(_) => {
                    providers::resolve_named(
                        &provider.uri,
                        &provider.endpoints,
                        &provider.credentials,
                        none,
                    )?
                    .0
                }
                None => providers::resolve(
                    &provider.uri,
                    &provider.endpoints,
                    &provider.credentials,
                    none,
                )?,
            };
            (Some(provider), resolution, named)
        }
        (None, None) => unreachable!("a provider source is parsed"),
    };
    let heads = |selection: &Resolution| match &provider {
        Some(provider) => provider.heads(selection),
        None => local_heads(store, selection),
    };
    let candidates: Vec<String> = resolution
        .members
        .iter()
        .filter(|row| row.carrier)
        .map(|row| row.member.clone())
        .collect();
    let carriers = match named {
        Some(member) => vec![member],
        None if request.profiles.is_empty() && candidates.len() == 1 => candidates,
        None => preflight::select_members(
            &locator,
            &registry,
            &resolution,
            request.profiles,
            heads,
            &scratch(store, "select"),
        )?,
    };
    let selected = resolution.select(&carriers)?;
    let manifest = providers::content_manifest(&selected)?;
    let selection_id = manifest.object_id();

    // The profiles: those named, else the one the carriers' headers select.
    let profiles = if request.profiles.is_empty() {
        let tensors = Resolution {
            members: selected
                .members
                .iter()
                .filter(|row| carrier(&row.member))
                .cloned()
                .collect(),
            ..selected.clone()
        };
        vec![preflight::select_profile(
            &locator,
            &registry,
            &heads(&tensors)?,
            &scratch(store, "profile"),
        )?]
    } else {
        request.profiles.to_vec()
    };
    let slots = slots(&profiles)?;

    // What the model is built from beside its tensors: the repository's own Diffusers index,
    // configs and tokenizers, or those of its converters' pinned reference repository.
    let mut metadata: Vec<(Provider, Resolution)> = vec![];
    if let Some(provider) = provider
        .as_ref()
        .filter(|p| matches!(p.uri, SourceUri::HuggingFace { .. }))
    {
        if let Some(found) = provider.diffusers()? {
            metadata.push((
                Provider::new(
                    provider.uri.clone(),
                    request.access,
                    &request.access.credential,
                )?,
                found,
            ));
        }
    }
    if metadata.is_empty() {
        let mut references = vec![];
        for profile in &profiles {
            for converter in model_source::profile_converters(&registry, profile)? {
                if let Some(reference) = converter.reference {
                    if !references.contains(&reference) {
                        references.push(reference);
                    }
                }
            }
        }
        for reference in references {
            let (uri, _) = parse(reference)?;
            let same = provider
                .as_ref()
                .is_some_and(|p| std::mem::discriminant(&uri) == std::mem::discriminant(&p.uri));
            let credential = if same {
                request.access.credential.as_str()
            } else {
                ""
            };
            let reference_provider = Provider::new(uri, request.access, credential)?;
            let found = reference_provider.diffusers()?.ok_or(Refusal {
                code: Code::MISSING_FIELD,
                detail: format!("converter reference {reference} has no pipeline index"),
            })?;
            metadata.push((reference_provider, found));
        }
    }

    // Land everything, resumably: the members stay retained under this model's owner.
    let owner = format!(
        "sha256:{}",
        digest(&[FORMAT, request.name, &selected.selection_sha256])
    );
    // A held file is already landed: nothing is fetched for it, so nothing is presented.
    let anonymous = (SourcePolicy::default(), transport::Anonymous);
    let (policy, credentials): (&SourcePolicy, &dyn transport::CredentialProvider) = match &provider
    {
        Some(provider) => (&provider.policy, &provider.credentials),
        None => (&anonymous.0, &anonymous.1),
    };
    let mut landing = vec![(policy, credentials, selected.clone())];
    landing.extend(metadata.iter().map(|(p, r)| {
        (
            &p.policy,
            &p.credentials as &dyn transport::CredentialProvider,
            r.clone(),
        )
    }));
    let total: u64 = landing.iter().map(|(_, _, r)| r.object_bytes()).sum();
    let mut done = 0;
    let mut owners = vec![];
    for (index, (policy, credentials, resolution)) in landing.iter().enumerate() {
        let report =
            |present: u64, _: u64| progress("downloading the model source", done + present, total);
        let options = transport::SourceDownload {
            cancellation: request.cancellation.clone(),
            ..Default::default()
        };
        let land = |owner: &str| {
            providers::materialize_selected_with_options(
                store,
                owner,
                resolution,
                policy,
                *credentials,
                Deadline::none(),
                &report,
                &options,
            )
        };
        let mut owner = landing_owner(&owner, index);
        if let Err(refusal) = land(&owner) {
            // An earlier make of this selection committed and released its members (its
            // model has since gone): land under a fresh owner.
            if refusal.code != Code::TRANSACTION_CLOSED {
                return Err(refusal);
            }
            owner = format!("{owner}-{}", crate::meta::now_nanos_unique());
            land(&owner)?;
        }
        owners.push(owner);
        done += resolution.object_bytes();
    }

    // Convert: every tensor member is landed, so one pass prepares the whole selection.
    progress("converting the model source", 0, 0);
    let members = selected
        .members
        .iter()
        .filter(|row| carrier(&row.member))
        .map(|row| {
            Ok(SelectedSourceMember {
                member: row.member.clone(),
                object: row.object.clone(),
                header: local_head(store, &row.member, &row.object)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let registry_id = match request.registry {
        Some(bytes) => format!("sha256:{}", crate::sha256::hex_digest(bytes)),
        None => "builtin".into(),
    };
    let slot_rows: Vec<&str> = slots
        .iter()
        .flat_map(|s| [s.slot.as_str(), s.profile.as_str()])
        .collect();
    let mut identity = vec![FORMAT, selection_id.as_str(), registry_id.as_str()];
    identity.extend(&slot_rows);
    let conversion = format!("source-{}", &digest(&identity)[..56]);
    let (prepared, _) = model_source::prepare_selected_source(
        store,
        &conversion,
        &selection_id,
        slots.clone(),
        &members,
        &locator,
        &registry,
        None,
    )?;
    if !prepared.complete {
        return refuse(
            Code::DURABILITY_UNPROVEN,
            "the conversion did not complete with every member landed",
        );
    }

    // Compose and commit.
    progress("composing the model", 0, 0);
    let metadata_rows: Vec<_> = metadata
        .iter()
        .flat_map(|(_, r)| r.members.iter().cloned())
        .collect();
    let transaction = owner.clone();
    let computation = format!("sha256:{}", digest(&identity));
    let model = compose(
        store,
        &transaction,
        &computation,
        &prepared.sources,
        &metadata_rows,
    )?;
    let repo = RepositoryName::new("local", request.name)?;
    let current = std::fs::read(store.repository_path(&repo)).ok();
    store.apply_repository(
        current.as_deref(),
        &Mutation::ReplaceLocal {
            repo,
            version: selected.selection_sha256.clone(),
            manifest: model.clone(),
        },
        &Fault::default(),
    )?;

    // The local repository holds the model: the source, its conversion and the composition
    // record go.
    let meta = Meta::open(store)?;
    let _ = derived::dispose(store, &meta, &transaction);
    let _ = model_source::release_model_source(store, &conversion);
    for owner in &owners {
        let _ = crate::source_artifact::release(store, owner);
    }
    Ok(Made {
        repository: format!("local/{}", request.name),
        manifest: model,
        resolved: selected.canonical.clone(),
        profiles,
        selection: format!("sha256:{}", selected.selection_sha256),
    })
}

/// Where one group of landed members is retained until the model is committed.
fn landing_owner(model: &str, index: usize) -> String {
    format!("sha256:{}", digest(&[FORMAT, model, &index.to_string()]))
}

fn scratch(store: &Store, what: &str) -> std::path::PathBuf {
    store
        .root()
        .join("tmp")
        .join(format!("source-{what}-{}", crate::meta::now_nanos_unique()))
}

/// The model: the prepared profiles' union, with the pipeline index, each component's
/// construction config and the tokenizers landed beside it.
fn compose(
    store: &Store,
    transaction: &str,
    computation: &str,
    prepared: &[PreparedModelSource],
    metadata: &[providers::ResolvedMember],
) -> Result<ObjectRef> {
    let meta = Meta::open(store)?;
    match derived::lookup(store, &meta, transaction)? {
        Lookup::Committed(done) => return Ok(done.receipt().manifest.clone()),
        Lookup::Open { writer_session } => {
            if let Some(session) = writer_session {
                derived::fence(&meta, transaction, session)?;
            }
            derived::abandon(store, &meta, transaction)?;
        }
        _ => {}
    }
    let mut sources = vec![];
    let mut components = vec![];
    let mut configs: Vec<ConfigDeclaration> = vec![];
    let mut bodies: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    let mut order = vec![];
    let mut rows: Vec<_> = prepared.iter().collect();
    rows.sort_by(|a, b| a.slot.cmp(&b.slot));
    for row in rows {
        let manifest = ObjectRef {
            sha256: row
                .manifest_digest
                .trim_start_matches("sha256:")
                .to_string(),
            length: row.manifest_length,
        };
        sources.push(Source {
            alias: row.slot.clone(),
            manifest: manifest.clone(),
        });
        let header = header(store, &manifest)?;
        for (name, tensors) in &header.components {
            if components
                .iter()
                .any(|c: &ComponentDeclaration| &c.target == name)
            {
                return refuse(
                    Code::CARRIER_OVERLAP,
                    format!("model_source_components_overlap: profiles both produce {name}"),
                );
            }
            components.push(ComponentDeclaration {
                target: name.clone(),
                source: Some(row.slot.clone()),
                source_component: Some(name.clone()),
                drop: vec![],
                add: vec![],
            });
            order.extend(tensors.iter().map(|(key, _)| (name.clone(), key.clone())));
        }
        for (name, body) in &header.configs {
            match bodies.get(name) {
                Some(seen) if seen != body => {
                    return refuse(
                        Code::CARRIER_OVERLAP,
                        format!("model_source_configs_overlap: profiles both carry {name}"),
                    )
                }
                Some(_) => {}
                None => {
                    bodies.insert(name.clone(), body.clone());
                    configs.push(ConfigDeclaration::Copy {
                        target: name.clone(),
                        source: row.slot.clone(),
                        source_config: name.clone(),
                    });
                }
            }
        }
    }
    let added = added_configs(store, metadata)?;
    for name in added.keys() {
        if bodies.contains_key(name) {
            return refuse(
                Code::CARRIER_OVERLAP,
                format!("model_source_configs_overlap: config {name} is carried twice"),
            );
        }
        configs.push(ConfigDeclaration::Add {
            target: name.clone(),
        });
    }
    let objects = asset_files(store, metadata)?;
    let declaration = Declaration {
        work_fingerprint: Some(computation.to_string()),
        sources,
        components,
        configs,
        files: vec![],
        objects,
        order,
        max_new_bytes: added.values().map(|b| b.len() as u64).sum(),
    };
    let session = (crate::meta::now_nanos_unique() / 1_000_000) as u64 & ((1 << 53) - 1);
    let begun = derived::begin(store, &meta, transaction, session, declaration, None)?;
    let written = added
        .iter()
        .try_for_each(|(name, body)| {
            derived::add_config(&meta, transaction, session, name, &mut body.as_slice())
        })
        .and_then(|()| derived::commit(store, &meta, transaction, session));
    for (_, lease) in begun.source_leases {
        let _ = lease.release(&meta);
    }
    let _ = begun.writer_hold.release(&meta);
    match written {
        Ok(receipt) => Ok(receipt.manifest),
        Err(refusal) => {
            let _ = derived::fence(&meta, transaction, session);
            let _ = derived::abandon(store, &meta, transaction);
            Err(refusal)
        }
    }
}

fn header(store: &Store, manifest: &ObjectRef) -> Result<crate::header::Header> {
    let manifest = store.read_manifest(manifest)?;
    let reference = manifest.header().ok_or(Refusal {
        code: Code::MISSING_FIELD,
        detail: "a prepared source carries no header".into(),
    })?;
    crate::checkpoint::load_header(store, reference)
}

fn document(store: &Store, object: &ObjectRef) -> Result<Vec<u8>> {
    let mut body = Vec::new();
    store
        .open_verified(&object.sha256)?
        .into_file()
        .take(crate::limits::DOC_MAX_BYTES as u64 + 1)
        .read_to_end(&mut body)
        .map_err(|e| Refusal {
            code: Code::IO_FAILED,
            detail: format!("read {}: {e}", object.id()),
        })?;
    Ok(body)
}

fn index(metadata: &[providers::ResolvedMember]) -> Option<&providers::ResolvedMember> {
    metadata
        .iter()
        .find(|row| DIFFUSERS_INDEXES.contains(&row.member.as_str()))
}

/// The pipeline index (by its stem) and each component's construction config (by its
/// component), as landed beside the carriers.
fn added_configs(
    store: &Store,
    metadata: &[providers::ResolvedMember],
) -> Result<BTreeMap<String, Vec<u8>>> {
    let Some(found) = index(metadata) else {
        return Ok(BTreeMap::new());
    };
    let body = document(store, &found.object)?;
    let mut added = BTreeMap::new();
    for (name, path) in constructors(&components(&body)?) {
        if let Some(row) = metadata.iter().find(|row| row.member == path) {
            added.insert(name, constructible(document(store, &row.object)?)?);
        }
    }
    added.insert(found.member.trim_end_matches(".json").to_string(), body);
    Ok(added)
}

/// A saved config minus its private save-time keys (`_name_or_path`).
fn constructible(config: Vec<u8>) -> Result<Vec<u8>> {
    let Json::Obj(mut document) = jcs::parse(&config, crate::limits::DOC_MAX_BYTES)? else {
        return refuse(Code::WRONG_TYPE, "a component config is not an object");
    };
    let before = document.len();
    document.retain(|(key, _)| {
        let lower = key.to_lowercase();
        !(key.starts_with('_') && PATH_SUFFIXES.iter().any(|s| lower.ends_with(s)))
    });
    if document.len() == before {
        return Ok(config);
    }
    Ok(jcs::write(&Json::Obj(document)))
}

/// Every tokenizer and processor file landed beside the carriers, by reference.
fn asset_files(
    store: &Store,
    metadata: &[providers::ResolvedMember],
) -> Result<Vec<(String, ObjectRef)>> {
    let Some(found) = index(metadata) else {
        return Ok(vec![]);
    };
    let folders = assets(&components(&document(store, &found.object)?)?);
    Ok(metadata
        .iter()
        .filter(|row| folders.iter().any(|f| row.member.starts_with(f.as_str())))
        .map(|row| (row.member.clone(), row.object.clone()))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_source_names_its_provider_and_optional_carrier() {
        let revision = "ab".repeat(20);
        let (uri, member) =
            parse(&format!("hf://org/repo@{revision}/unet/a%20b.safetensors")).unwrap();
        assert!(matches!(uri, SourceUri::HuggingFace { .. }));
        assert_eq!(member.as_deref(), Some("unet/a b.safetensors"));
        assert_eq!(parse(&format!("hf://org/repo@{revision}")).unwrap().1, None);
        let (uri, file) = parse("civitai://128078/123").unwrap();
        assert_eq!(uri, SourceUri::Civitai { version: 128078 });
        assert_eq!(file.as_deref(), Some("123"));
        for bad in ["hf://org/repo", "civitai://0", "s3://x"] {
            assert!(parse(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn a_diffusers_index_names_constructors_and_asset_folders() {
        let index = br#"{"_class_name":"StableDiffusionXLPipeline","unet":["diffusers","UNet2DConditionModel"],
            "scheduler":["diffusers","EulerDiscreteScheduler"],"tokenizer":["transformers","CLIPTokenizer"],
            "text_encoder":["transformers","CLIPTextModel"]}"#;
        let found = components(index).unwrap();
        let built = constructors(&found);
        assert_eq!(built["unet"], "unet/config.json");
        assert_eq!(built["scheduler"], "scheduler/scheduler_config.json");
        assert!(!built.contains_key("tokenizer"));
        assert_eq!(assets(&found), ["tokenizer/"]);
        let config = br#"{"_name_or_path":"/x","_class_name":"UNet","sample_size":128}"#.to_vec();
        let kept = constructible(config).unwrap();
        assert!(!String::from_utf8(kept.clone())
            .unwrap()
            .contains("_name_or_path"));
        assert!(String::from_utf8(kept).unwrap().contains("_class_name"));
    }
}
