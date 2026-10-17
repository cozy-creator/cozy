//! The machine API's JSON shape (xs-044), ahead of the cohort that sends it on the wire: Calcifer
//! keeps runs' specs and outcomes in it, so v0.10, which reads only JSON, finds them as they are.
//! These are the structs v0.10's `cozy_proto::machine` defines; [`pb`] converts the protobuf ones.
use serde::{Deserialize, Serialize};

/// Raw JSON, kept as its exact text: canonical bytes and 64-bit integers survive.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Json(pub Box<serde_json::value::RawValue>);

impl Json {
    /// `text` if it is JSON.
    pub fn parse(text: impl Into<String>) -> Result<Self, serde_json::Error> {
        serde_json::value::RawValue::from_string(text.into()).map(Self)
    }
    pub fn get(&self) -> &str {
        self.0.get()
    }
}

impl PartialEq for Json {
    fn eq(&self, other: &Self) -> bool {
        self.get() == other.get()
    }
}

// ---- Run ------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunKind {
    /// Every function is a call.
    #[default]
    Call,
    /// Prepare only: install and download, then succeed.
    Warm,
    /// Stage, verify and activate a software cohort at idle.
    Update,
    /// `remove/1`: the caller's installations of the source leave the machine: an installation by
    /// its name, or a release's package (its release, or every release when empty) from the run's
    /// Hub (local/ packages from none). Their environments are deleted once no process and no
    /// other installation holds them.
    Remove,
}

/// How far a function is kept ready; each level includes the ones before it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WarmLevel {
    /// Not kept: a warm set's item at this level leaves it.
    #[default]
    Off,
    /// The package environment.
    Installed,
    /// The function's bound weights in the store.
    Downloaded,
    /// A process with its imports done: no device, no VRAM.
    Imported,
    /// An executor with the model constructed, weights in the host tier.
    Host,
    /// Its weights on the GPU.
    Gpu,
}

/// What a run runs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Source {
    Release(Release),
    /// An installation this machine already holds for the signer.
    Installation(String),
    Local(LocalSource),
}

/// What a warm set's member keeps ready.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WarmSource {
    Release(Release),
    Installation(String),
}

/// One member of a caller's warm set: a function the machine keeps ready up to `level`. It is
/// the last idle thing to give its memory to a request, and comes back when room returns.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct WarmItem {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<WarmSource>,
    pub entrypoint: String,
    /// None: the slots' defaults, as for a run.
    pub models: Vec<ModelChoice>,
    pub level: WarmLevel,
    /// Status only: the level it holds now (installed, downloaded, imported, host, gpu).
    pub holds: String,
    /// Status only: why that is lower than `level`.
    pub held_back: String,
    /// As [`RunSpec::bits`].
    pub bits: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct WarmSet {
    pub items: Vec<WarmItem>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RunSpec {
    pub kind: RunKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<Source>,
    pub entrypoint: String,
    /// The function's parameters, a JSON object.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payload: Option<Json>,
    /// Objects uploaded with `PUT /v1/objects`, bound to payload fields.
    pub inputs: Vec<InputFile>,
    pub models: Vec<ModelChoice>,
    /// The catalog revision the caller made its choices under (th-245): a held resolution is
    /// reused only under the same one, so a publish, yank, bind or retarget the caller knows of
    /// resolves once more at the Hub.
    pub catalog_revision: String,
    pub attention_kernel: String,
    /// org/name: the requester's repository for the call's weights outputs, or a warm run's
    /// source model.
    pub weights_destination: String,
    /// Where the run reads, and its capability there.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hub: Option<HubAccess>,
    /// The account that org-relative references of local code name.
    pub owner: String,
    /// Provider tokens for source models, memory only like `hub`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub providers: Option<ProviderAccess>,
    /// `warm/2`: the caller's whole warm set, replacing its previous one; no items clears it.
    /// The run installs and downloads every member, and its result lists each one's `level` and
    /// `held_back`. Absent: the set is left as it is.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub set: Option<WarmSet>,
    /// 0: widest group every model slot supports on this machine. Positive: at most this many
    /// GPUs, for the run and its children.
    pub gpus: u32,
    /// The run's weight precision: 16, 8 or 4. 0: the machine's rules choose each slot's lane.
    /// Children inherit it.
    pub bits: u32,
}

/// The caller's provider tokens, for models a run makes from a provider source.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ProviderAccess {
    pub huggingface: String,
    pub civitai: String,
}

/// A published release by name (th-245): a machine that holds no installation of it reads its
/// card and locked requirements at the Hub by name.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Release {
    /// org/name
    pub package: String,
    /// Empty: the Hub's newest.
    pub release: String,
}

/// Local code: `manifest` names a JSON object written with `PUT /v1/objects`:
/// `{package: "local/<dist>", release, python_requires, python_version, callees: {distribution:
/// package}}` and its code. A live project is installed editable from `files: [{name, digest,
/// length, executable}]`: the project and the path dependencies its lock names, each file under
/// their common directory, with `project` the project's directory there. The machine keeps one
/// such tree and one environment per signer and package. `path` names that common directory on
/// this computer's machine, which reads it in place. `wheels: [{name, digest, length}]` are locked
/// dependencies the caller carries because the machine cannot fetch them. Members are objects the
/// same signer wrote.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LocalSource {
    /// `sha256:<hex>` of the manifest object.
    pub manifest: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct InputFile {
    /// The payload field path, list index included.
    pub field: String,
    /// `sha256:<hex>`
    pub digest: String,
    pub length: u64,
    pub media_type: String,
    pub order: u32,
}

/// One model slot's choice, by name (th-245: the machine resolves it at the Hub). A Hub model is
/// `repository` at `release` (empty: the newest) in `lane`, else in the lane the machine's rules
/// pick (xs-042). `manifest` names a checkpoint by hash only: the owner's private one at the Hub
/// (with `repository`; the run's capability names it) or one this machine holds (without). A slot
/// with no choice takes its declared default release, a callee's included (`<package>/<path>`).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelChoice {
    /// A root slot parameter or full slot path.
    pub parameter: String,
    pub repository: String,
    pub release: String,
    pub lane: String,
    /// `sha256:<hex>`
    pub manifest: String,
    pub manifest_length: u64,
    /// Each by name, a hash of the owner's, or a provider source.
    pub adapters: Vec<Adapter>,
    /// A provider source (`hf://<org>/<repo>@<rev>[/member]`, `civitai://<version>[/file]`) the
    /// machine makes the slot's model from, under these reviewed `profiles` (none: the one its
    /// headers select); never with repository or manifest.
    pub source: String,
    pub profiles: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Adapter {
    pub component: String,
    pub model: String,
    pub release: String,
    pub lane: String,
    /// `sha256:<hex>`, the owner's checkpoint by hash; else `model` at `release`, `lane`.
    pub manifest: String,
    /// Canonical decimal; empty means 1.
    pub scale: String,
    /// A provider source (`civitai://<version>`, `hf://…`) made here and normalized at ingest,
    /// under these reviewed `profiles` (none: the one its headers select); never with `model`.
    pub source: String,
    pub profiles: Vec<String>,
}

/// A Hub a run reads. Public content is read anonymously.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct HubAccess {
    pub origin: String,
    /// An origin's private CA (DER, base64), when it has one.
    #[serde(with = "base64_bytes", skip_serializing_if = "Vec::is_empty")]
    pub ca_der: Vec<u8>,
    /// The Hub's object-storage hosts downloads may follow.
    pub object_hosts: Vec<String>,
    /// The owner's capability for this run's private operations (th-241): a compact JWS its
    /// device key signed for this machine's leaf (`cnf.jkt`) and the Hub's resource (`aud`),
    /// naming each operation in `authorization_details`. The machine trades it once, at the run's
    /// first private operation, at `token_endpoint`. Empty: a public-only run.
    pub capability: String,
    /// The Hub's authorization server's, as its metadata names it.
    pub token_endpoint: String,
}

/// One revision of an output: a single output is replaced by each, a list output grows. Its bytes
/// are `GET /v1/runs/{id}/outputs/{output}`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Product {
    pub output: String,
    /// A list item's 1-based index; 0 for a single output.
    pub index: u32,
    pub rev: u64,
    pub length: u64,
    /// `sha256:<hex>` of these bytes.
    pub digest: String,
    pub media_type: String,
    pub label: String,
    /// Media time, for a growing video.
    pub duration_us: u64,
    /// These bytes extend the previous revision's, from here.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub appended_from: Option<u64>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Outcome {
    /// succeeded, failed, canceled
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<Reason>,
    /// The result, canonical JSON, assets as digests.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Json>,
    pub outputs: Vec<Product>,
    /// `GET /v1/runs/{id}/triage` answers its bundle.
    pub triage: bool,
    /// What its executor measured, canonical JSON: `attribution` (stage and step tracks),
    /// `execution` (degree, ranks with the GPU and attention each served, executor boot,
    /// construction load and warm) and `observations` (attention rows). None for a run that ran
    /// on no device.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub measurements: Option<Json>,
    /// Its own time running its callable, every attempt summed; 0: unknown.
    pub execution_ms: u64,
    /// A memoizable callable's computation (`sha256:<hex>`), whether it ran or was answered.
    pub computation_digest: String,
    /// Answered from a result this machine held: nothing ran.
    pub memoized: bool,
    /// Why it was neither looked up nor held (its author's condition unmet, e.g. "seed not given").
    pub unmemoized: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Reason {
    pub code: String,
    pub message: String,
    /// request, runtime, author, machine
    pub origin: String,
}

/// Bytes as base64 text.
pub(crate) mod base64_bytes {
    use base64::{Engine as _, engine::general_purpose::STANDARD};
    use serde::{Deserialize, Deserializer, Serializer, de::Error as _};

    pub fn serialize<S: Serializer>(bytes: &[u8], serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&STANDARD.encode(bytes))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<u8>, D::Error> {
        STANDARD
            .decode(String::deserialize(deserializer)?)
            .map_err(D::Error::custom)
    }
}

/// The protobuf (`cozy.machine.v1`) structs to JSON and back, at Calcifer's store while the wire is
/// still protobuf. It goes with protobuf in v0.10.
pub mod pb {
    use super::*;
    use crate::machine as v1;

    #[derive(Debug, thiserror::Error)]
    #[error("{0}")]
    pub struct Unreadable(String);

    /// Structs whose fields match one for one.
    macro_rules! flat {
        ($($name:ident { $($field:ident),* })*) => {$(
            impl From<&v1::$name> for $name {
                fn from(v: &v1::$name) -> Self {
                    Self { $($field: v.$field.clone()),* }
                }
            }
            impl From<&$name> for v1::$name {
                fn from(v: &$name) -> Self {
                    Self { $($field: v.$field.clone()),* }
                }
            }
        )*};
    }

    flat! {
        Release { package, release }
        LocalSource { manifest }
        InputFile { field, digest, length, media_type, order }
        Adapter { component, model, release, lane, manifest, scale, source, profiles }
        HubAccess { origin, ca_der, object_hosts, capability, token_endpoint }
        ProviderAccess { huggingface, civitai }
        Product { output, index, rev, length, digest, media_type, label, duration_us, appended_from }
        Reason { code, message, origin }
    }

    fn json(bytes: &[u8], what: &str) -> Result<Option<Json>, Unreadable> {
        if bytes.is_empty() {
            return Ok(None);
        }
        let text =
            std::str::from_utf8(bytes).map_err(|_| Unreadable(format!("{what} is not UTF-8")))?;
        Json::parse(text)
            .map(Some)
            .map_err(|e| Unreadable(format!("{what}: {e}")))
    }

    fn bytes(json: &Option<Json>) -> Vec<u8> {
        json.as_ref()
            .map(|j| j.get().as_bytes().to_vec())
            .unwrap_or_default()
    }

    fn models(models: &[v1::ModelChoice]) -> Vec<ModelChoice> {
        let model = |m: &v1::ModelChoice| ModelChoice {
            parameter: m.parameter.clone(),
            repository: m.repository.clone(),
            release: m.release.clone(),
            lane: m.lane.clone(),
            manifest: m.manifest.clone(),
            manifest_length: m.manifest_length,
            adapters: m.adapters.iter().map(Into::into).collect(),
            source: m.source.clone(),
            profiles: m.profiles.clone(),
        };
        models.iter().map(model).collect()
    }

    fn models_pb(models: &[ModelChoice]) -> Vec<v1::ModelChoice> {
        let model = |m: &ModelChoice| v1::ModelChoice {
            parameter: m.parameter.clone(),
            repository: m.repository.clone(),
            release: m.release.clone(),
            lane: m.lane.clone(),
            manifest: m.manifest.clone(),
            manifest_length: m.manifest_length,
            adapters: m.adapters.iter().map(Into::into).collect(),
            source: m.source.clone(),
            profiles: m.profiles.clone(),
        };
        models.iter().map(model).collect()
    }

    fn level(level: i32) -> Result<WarmLevel, Unreadable> {
        Ok(match v1::WarmLevel::try_from(level) {
            Ok(v1::WarmLevel::Unspecified) => WarmLevel::Off,
            Ok(v1::WarmLevel::Installed) => WarmLevel::Installed,
            Ok(v1::WarmLevel::Downloaded) => WarmLevel::Downloaded,
            Ok(v1::WarmLevel::Imported) => WarmLevel::Imported,
            Ok(v1::WarmLevel::Host) => WarmLevel::Host,
            Ok(v1::WarmLevel::Gpu) => WarmLevel::Gpu,
            Err(_) => return Err(Unreadable(format!("warm level {level}"))),
        })
    }

    fn level_pb(level: WarmLevel) -> i32 {
        (match level {
            WarmLevel::Off => v1::WarmLevel::Unspecified,
            WarmLevel::Installed => v1::WarmLevel::Installed,
            WarmLevel::Downloaded => v1::WarmLevel::Downloaded,
            WarmLevel::Imported => v1::WarmLevel::Imported,
            WarmLevel::Host => v1::WarmLevel::Host,
            WarmLevel::Gpu => v1::WarmLevel::Gpu,
        }) as i32
    }

    impl TryFrom<&v1::RunSpec> for RunSpec {
        type Error = Unreadable;
        fn try_from(s: &v1::RunSpec) -> Result<Self, Unreadable> {
            use v1::run_spec::Source as S;
            use v1::warm_item::Source as W;
            let items = s.set.as_ref().map(|set| {
                let item = |i: &v1::WarmItem| -> Result<WarmItem, Unreadable> {
                    Ok(WarmItem {
                        source: i.source.as_ref().map(|source| match source {
                            W::Release(r) => WarmSource::Release(r.into()),
                            W::Installation(i) => WarmSource::Installation(i.clone()),
                        }),
                        entrypoint: i.entrypoint.clone(),
                        models: models(&i.models),
                        level: level(i.level)?,
                        holds: i.holds.clone(),
                        held_back: i.held_back.clone(),
                        bits: i.bits,
                    })
                };
                set.items.iter().map(item).collect::<Result<Vec<_>, _>>()
            });
            Ok(Self {
                kind: match v1::RunKind::try_from(s.kind) {
                    Ok(v1::RunKind::Call) => RunKind::Call,
                    Ok(v1::RunKind::Warm) => RunKind::Warm,
                    Ok(v1::RunKind::Update) => RunKind::Update,
                    Ok(v1::RunKind::Remove) => RunKind::Remove,
                    Err(_) => return Err(Unreadable(format!("run kind {}", s.kind))),
                },
                source: s.source.as_ref().map(|source| match source {
                    S::Release(r) => Source::Release(r.into()),
                    S::Installation(i) => Source::Installation(i.clone()),
                    S::Local(l) => Source::Local(l.into()),
                }),
                entrypoint: s.entrypoint.clone(),
                payload: json(&s.payload, "the payload")?,
                inputs: s.inputs.iter().map(Into::into).collect(),
                models: models(&s.models),
                catalog_revision: s.catalog_revision.clone(),
                attention_kernel: s.attention_kernel.clone(),
                weights_destination: s.weights_destination.clone(),
                hub: s.hub.as_ref().map(Into::into),
                owner: s.owner.clone(),
                providers: s.providers.as_ref().map(Into::into),
                set: items.transpose()?.map(|items| WarmSet { items }),
                gpus: s.gpus,
                bits: s.bits,
            })
        }
    }

    impl From<&RunSpec> for v1::RunSpec {
        fn from(s: &RunSpec) -> Self {
            use v1::run_spec::Source as S;
            use v1::warm_item::Source as W;
            let item = |i: &WarmItem| v1::WarmItem {
                source: i.source.as_ref().map(|source| match source {
                    WarmSource::Release(r) => W::Release(r.into()),
                    WarmSource::Installation(i) => W::Installation(i.clone()),
                }),
                entrypoint: i.entrypoint.clone(),
                models: models_pb(&i.models),
                level: level_pb(i.level),
                holds: i.holds.clone(),
                held_back: i.held_back.clone(),
                bits: i.bits,
            };
            Self {
                kind: (match s.kind {
                    RunKind::Call => v1::RunKind::Call,
                    RunKind::Warm => v1::RunKind::Warm,
                    RunKind::Update => v1::RunKind::Update,
                    RunKind::Remove => v1::RunKind::Remove,
                }) as i32,
                source: s.source.as_ref().map(|source| match source {
                    Source::Release(r) => S::Release(r.into()),
                    Source::Installation(i) => S::Installation(i.clone()),
                    Source::Local(l) => S::Local(l.into()),
                }),
                entrypoint: s.entrypoint.clone(),
                payload: bytes(&s.payload),
                inputs: s.inputs.iter().map(Into::into).collect(),
                models: models_pb(&s.models),
                catalog_revision: s.catalog_revision.clone(),
                attention_kernel: s.attention_kernel.clone(),
                weights_destination: s.weights_destination.clone(),
                hub: s.hub.as_ref().map(Into::into),
                owner: s.owner.clone(),
                providers: s.providers.as_ref().map(Into::into),
                set: s.set.as_ref().map(|set| v1::WarmSet {
                    items: set.items.iter().map(item).collect(),
                }),
                gpus: s.gpus,
                bits: s.bits,
            }
        }
    }

    /// An outcome's result and measurements that are not JSON are left out: no reader of them
    /// ever showed bytes it could not parse.
    impl TryFrom<&v1::Outcome> for Outcome {
        type Error = Unreadable;
        fn try_from(o: &v1::Outcome) -> Result<Self, Unreadable> {
            Ok(Self {
                status: o.status.clone(),
                reason: o.reason.as_ref().map(Into::into),
                result: json(&o.result, "the result").unwrap_or_default(),
                outputs: o.outputs.iter().map(Into::into).collect(),
                triage: o.triage,
                measurements: json(&o.measurements, "the measurements").unwrap_or_default(),
                execution_ms: o.execution_ms,
                computation_digest: o.computation_digest.clone(),
                memoized: o.memoized,
                unmemoized: o.unmemoized.clone(),
            })
        }
    }

    impl From<&Outcome> for v1::Outcome {
        fn from(o: &Outcome) -> Self {
            Self {
                status: o.status.clone(),
                reason: o.reason.as_ref().map(Into::into),
                result: bytes(&o.result),
                outputs: o.outputs.iter().map(Into::into).collect(),
                triage: o.triage,
                measurements: bytes(&o.measurements),
                execution_ms: o.execution_ms,
                computation_digest: o.computation_digest.clone(),
                memoized: o.memoized,
                unmemoized: o.unmemoized.clone(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::machine as v1;

    /// Every field of a spec and an outcome survives protobuf → JSON text → protobuf, raw JSON byte
    /// for byte, and the JSON is the shape v0.10 reads.
    #[test]
    fn specs_and_outcomes_survive_protobuf_to_json_and_back() {
        let model = v1::ModelChoice {
            parameter: "unet".into(),
            repository: "a/m".into(),
            release: "1".into(),
            lane: "fp8".into(),
            manifest: "sha256:ab".into(),
            manifest_length: 9,
            adapters: vec![v1::Adapter {
                component: "unet".into(),
                model: "a/l".into(),
                scale: "0.5".into(),
                profiles: vec!["p".into()],
                ..Default::default()
            }],
            source: String::new(),
            profiles: vec![],
        };
        let spec = v1::RunSpec {
            kind: v1::RunKind::Warm as i32,
            source: Some(v1::run_spec::Source::Local(v1::LocalSource {
                manifest: "sha256:cd".into(),
            })),
            entrypoint: "gen".into(),
            payload: br#"{"seed":9007199254740993,"nested":{"b":2,"a":1}}"#.to_vec(),
            inputs: vec![v1::InputFile {
                field: "image".into(),
                digest: "sha256:ef".into(),
                length: 3,
                media_type: "image/png".into(),
                order: 1,
            }],
            models: vec![model.clone()],
            catalog_revision: "r7".into(),
            attention_kernel: "sage2".into(),
            weights_destination: "a/out".into(),
            hub: Some(v1::HubAccess {
                origin: "https://hub".into(),
                ca_der: vec![1, 2, 3],
                object_hosts: vec!["r2".into()],
                capability: "cap".into(),
                token_endpoint: "https://hub/token".into(),
            }),
            owner: "fidika".into(),
            providers: Some(v1::ProviderAccess {
                huggingface: "hf".into(),
                civitai: String::new(),
            }),
            set: Some(v1::WarmSet {
                items: vec![v1::WarmItem {
                    source: Some(v1::warm_item::Source::Installation("i1".into())),
                    entrypoint: "gen".into(),
                    models: vec![model],
                    level: v1::WarmLevel::Gpu as i32,
                    holds: "host".into(),
                    held_back: "room".into(),
                    bits: 8,
                }],
            }),
            gpus: 2,
            bits: 8,
        };
        let json = serde_json::to_string(&RunSpec::try_from(&spec).unwrap()).unwrap();
        assert!(
            json.contains(r#""payload":{"seed":9007199254740993,"nested":{"b":2,"a":1}}"#),
            "{json}"
        );
        assert!(
            json.contains(r#""kind":"warm""#) && json.contains(r#""level":"gpu""#),
            "{json}"
        );
        assert!(
            json.contains(r#""source":{"local":{"manifest":"sha256:cd"}}"#),
            "{json}"
        );
        assert!(json.contains(r#""ca_der":"AQID""#), "{json}");
        let back = v1::RunSpec::from(&serde_json::from_str::<RunSpec>(&json).unwrap());
        assert_eq!(back, spec);

        let outcome = v1::Outcome {
            status: "failed".into(),
            reason: Some(v1::Reason {
                code: "author_error".into(),
                message: "boom".into(),
                origin: "author".into(),
            }),
            result: br#"{"file":{"asset_ref":"sha256:aa"}}"#.to_vec(),
            outputs: vec![v1::Product {
                output: "file".into(),
                index: 2,
                rev: 3,
                length: 5,
                digest: "sha256:aa".into(),
                media_type: "text/plain".into(),
                label: "L".into(),
                duration_us: 7,
                appended_from: Some(4),
            }],
            triage: true,
            measurements: br#"{"execution":{"degree":1}}"#.to_vec(),
            execution_ms: 12,
            computation_digest: "sha256:bb".into(),
            memoized: true,
            unmemoized: "seed not given".into(),
        };
        let json = serde_json::to_string(&Outcome::try_from(&outcome).unwrap()).unwrap();
        assert!(
            json.contains(r#""measurements":{"execution":{"degree":1}}"#),
            "{json}"
        );
        let back = v1::Outcome::from(&serde_json::from_str::<Outcome>(&json).unwrap());
        assert_eq!(back, outcome);

        // Empty bytes are absent JSON; an outcome's unreadable result is left out, a spec's payload
        // that is not JSON is refused.
        let empty = Outcome::try_from(&v1::Outcome::default()).unwrap();
        assert_eq!((empty.result, empty.measurements), (None, None));
        let garbled = v1::Outcome {
            result: b"{".to_vec(),
            ..Default::default()
        };
        assert_eq!(Outcome::try_from(&garbled).unwrap().result, None);
        let garbled = v1::RunSpec {
            payload: b"{".to_vec(),
            ..Default::default()
        };
        assert!(RunSpec::try_from(&garbled).is_err());
        let unknown = v1::RunSpec {
            kind: 9,
            ..Default::default()
        };
        assert!(RunSpec::try_from(&unknown).is_err());
    }
}
