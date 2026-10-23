//! What a run sends: the machine's `RunSpec`, frozen at submit, inside Calcifer's `SubmitRun`.
use std::path::PathBuf;

use protocol::{
    calcifer::{Kind as RunKind, On, StagedInput, SubmitRun, Warning},
    error::{Error, Kind},
    machine::{self as v1, ModelChoice},
};

use super::{
    Warm,
    bind::{Bound, Choice},
    interface::Callable,
};
use crate::model::refs::ModelRef;

/// A function as a run names it: a published release's, or local code's.
pub struct Call<'a> {
    /// `org/pkg`, or local code's `local/<dist>`.
    pub package: &'a str,
    pub source: Source<'a>,
    pub callable: &'a Callable,
    /// 0: every GPU the run's slots support on its machine.
    pub gpus: u32,
    /// 0: the machine's rules choose each slot's precision.
    pub bits: u32,
    /// `--attention-kernel`; empty: the machine's own selection.
    pub attention_kernel: &'a str,
}

/// Where the machine finds the function's code.
pub enum Source<'a> {
    /// A published release; none: the one the machine holds, else the newest.
    Release(Option<&'a str>),
    /// Local code the machine installed under `installation`, from `source`, for `owner`.
    Local {
        source: &'a v1::LocalSource,
        installation: Option<&'a str>,
    },
}

impl Call<'_> {
    /// The run's name for people: `org/pkg/fn[@release]`, `local/<dist>/fn`.
    pub fn target(&self) -> String {
        let target = format!("{}/{}", self.package, self.callable.name);
        match self.source {
            Source::Release(Some(release)) => format!("{target}@{release}"),
            _ => target,
        }
    }

    /// A CALL of the function with `bound`.
    pub fn spec(&self, bound: &Bound) -> Result<v1::RunSpec, Error> {
        Ok(v1::RunSpec {
            kind: v1::RunKind::Call,
            source: Some(match self.source {
                Source::Release(_) => v1::Source::Release(self.release()),
                // The same source as the install: unchanged, the machine installs nothing again.
                Source::Local { source, .. } => v1::Source::Local(source.clone()),
            }),
            entrypoint: self.callable.name.clone(),
            payload: Some(bound.canonical_payload()),
            inputs: bound
                .files
                .iter()
                .map(|f| v1::InputFile {
                    field: f.field.clone(),
                    digest: f.digest.to_string(),
                    length: f.length,
                    media_type: f.media_type.clone(),
                    order: f.order,
                })
                .collect(),
            models: choices(&bound.models)?,
            weights_destination: bound.destination.clone().unwrap_or_default(),
            attention_kernel: self.attention_kernel.to_owned(),
            gpus: self.gpus,
            bits: self.bits,
            ..Default::default()
        })
    }

    /// A WARM run that sets the function's place in the machine's warm set: Calcifer merges
    /// this one item into the set the machine last reported, and level 0 takes it out.
    pub fn warm(&self, level: Warm, models: &[Choice]) -> Result<v1::RunSpec, Error> {
        let level = match level {
            Warm::Installed => v1::WarmLevel::Installed,
            Warm::Downloaded => v1::WarmLevel::Downloaded,
            Warm::Imported => v1::WarmLevel::Imported,
            Warm::Host => v1::WarmLevel::Host,
            Warm::Gpu => v1::WarmLevel::Gpu,
            Warm::Off => v1::WarmLevel::Off,
        };
        let source = match self.source {
            Source::Release(_) => v1::WarmSource::Release(self.release()),
            Source::Local {
                installation: Some(installation),
                ..
            } => v1::WarmSource::Installation(installation.to_owned()),
            // A kept interface names no installation; a warm set always installs first.
            Source::Local {
                installation: None, ..
            } => {
                return Err(Error::new(
                    Kind::Internal,
                    "local_warm_uninstalled",
                    "a warm set of local code needs the installation its install named",
                ));
            }
        };
        let item = v1::WarmItem {
            source: Some(source),
            entrypoint: self.callable.name.clone(),
            models: choices(models)?,
            level,
            bits: self.bits,
            ..Default::default()
        };
        Ok(v1::RunSpec {
            kind: v1::RunKind::Warm,
            set: Some(v1::WarmSet { items: vec![item] }),
            ..Default::default()
        })
    }

    /// The submission Calcifer records: the spec, the files a call reads, and its warnings.
    pub fn submission(
        &self,
        on: On,
        hub: &str,
        spec: &v1::RunSpec,
        bound: Option<&Bound>,
        out: Option<PathBuf>,
    ) -> SubmitRun {
        let target = self.target();
        let kind = match spec.kind {
            v1::RunKind::Warm => RunKind::Warm,
            _ => RunKind::Call,
        };
        let inputs = bound.map_or_else(Vec::new, |b| {
            let files = b.files.iter().map(|f| StagedInput {
                path: f.path.clone(),
                digest: f.digest,
                length: f.length,
            });
            files.chain(b.tree_files.iter().cloned()).collect()
        });
        let warnings = bound
            .and_then(|b| b.warning(&target))
            .map(|message| Warning {
                code: "request_fields_ignored".into(),
                message,
            })
            .into_iter()
            .collect();
        SubmitRun {
            agreed: None,
            on,
            hub: hub.to_owned(),
            kind,
            target,
            spec: spec.clone(),
            inputs,
            out,
            warnings,
        }
    }

    fn release(&self) -> v1::Release {
        let release = match self.source {
            Source::Release(release) => release.unwrap_or_default(),
            Source::Local { .. } => "",
        };
        v1::Release {
            package: self.package.to_owned(),
            release: release.to_owned(),
        }
    }
}

/// Each slot's choice as the machine takes it, its adapters by the same reference grammar.
fn choices(models: &[Choice]) -> Result<Vec<ModelChoice>, Error> {
    models
        .iter()
        .map(|choice| {
            let mut out = match &choice.reference {
                Some(reference) => ModelRef::parse(reference)?.choice(&choice.slot),
                None => ModelChoice {
                    parameter: choice.slot.clone(),
                    ..Default::default()
                },
            };
            for adapter in &choice.adapters {
                let parsed = ModelRef::parse(&adapter.reference)?.choice("");
                out.adapters.push(v1::Adapter {
                    component: adapter.component.clone().unwrap_or_default(),
                    model: parsed.repository,
                    release: parsed.release,
                    lane: parsed.lane,
                    manifest: parsed.manifest,
                    scale: adapter.scale.clone(),
                    source: parsed.source,
                    profiles: vec![],
                });
            }
            Ok(out)
        })
        .collect()
}
