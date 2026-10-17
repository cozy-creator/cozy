//! What a run sends: the machine's `RunSpec`, frozen at submit, inside Calcifer's `SubmitRun`.
use std::path::PathBuf;

use cozy_proto::{
    calcifer::{Kind as RunKind, On, StagedInput, SubmitRun, Warning},
    error::{Error, Kind},
    machine::{self as v1, ModelChoice},
};
use prost::Message as _;

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
            kind: v1::RunKind::Call.into(),
            source: Some(match self.source {
                Source::Release(_) => v1::run_spec::Source::Release(self.release()),
                // The same source as the install: unchanged, the machine installs nothing again.
                Source::Local { source, .. } => v1::run_spec::Source::Local(source.clone()),
            }),
            entrypoint: self.callable.name.clone(),
            payload: bound.canonical_payload(),
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
            Warm::Off => v1::WarmLevel::Unspecified,
        };
        let source = match self.source {
            Source::Release(_) => v1::warm_item::Source::Release(self.release()),
            Source::Local {
                installation: Some(installation),
                ..
            } => v1::warm_item::Source::Installation(installation.to_owned()),
            // An older machine's install names no installation, which a warm set needs.
            Source::Local {
                installation: None, ..
            } => {
                return Err(Error::new(
                    Kind::Unavailable,
                    "local_warm_unsupported",
                    "this machine names no installation for local code, so it cannot keep it ready",
                )
                .remedy("update the machine: `cozy machine update <name>`"));
            }
        };
        let item = v1::WarmItem {
            source: Some(source),
            entrypoint: self.callable.name.clone(),
            models: choices(models)?,
            level: level.into(),
            bits: self.bits,
            ..Default::default()
        };
        Ok(v1::RunSpec {
            kind: v1::RunKind::Warm.into(),
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
        let kind = match v1::RunKind::try_from(spec.kind) {
            Ok(v1::RunKind::Warm) => RunKind::Warm,
            _ => RunKind::Call,
        };
        let inputs = bound.map_or_else(Vec::new, |b| {
            b.files
                .iter()
                .map(|f| StagedInput {
                    path: f.path.clone(),
                    digest: f.digest,
                    length: f.length,
                })
                .collect()
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
            on,
            hub: hub.to_owned(),
            kind,
            target,
            spec: spec.encode_to_vec(),
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

#[cfg(test)]
mod tests {
    use cozy_proto::calcifer::Kind as RunKind;

    use super::super::{
        bind::{Terms, bind, warm_choices},
        interface::tests::interface,
    };
    use super::*;

    fn terms(inputs: &[&str]) -> Terms {
        Terms {
            inputs: inputs.iter().map(|s| (*s).to_owned()).collect(),
            key: "idem-spec".into(),
            cwd: std::env::temp_dir(),
            user_home: std::env::temp_dir(),
            ..Terms::default()
        }
    }

    #[test]
    fn a_call_names_its_release_entrypoint_payload_and_choices() {
        let sdxl = interface("sdxl-2.10.0");
        let generate = sdxl
            .callables
            .iter()
            .find(|c| c.name == "generate")
            .unwrap();
        let mut with = terms(&["a cat", "steps=8", "model.model=Fidika/SDXL-Base@1.0/int8"]);
        with.loras = vec!["model:unet=fidika/detail#sha256:".to_owned() + &"a".repeat(64) + ",0.5"];
        let bound = bind("fidika/sdxl/generate", None, generate, &with).unwrap();
        let call = Call {
            package: "fidika/sdxl",
            source: Source::Release(None),
            callable: generate,
            gpus: 2,
            bits: 8,
        };
        let spec = call.spec(&bound).unwrap();
        assert_eq!(spec.kind, v1::RunKind::Call as i32);
        assert_eq!(
            spec.source,
            Some(v1::run_spec::Source::Release(v1::Release {
                package: "fidika/sdxl".into(),
                release: String::new()
            }))
        );
        assert_eq!(spec.entrypoint, "generate");
        assert_eq!(
            String::from_utf8(spec.payload.clone()).unwrap(),
            r#"{"prompt":"a cat","steps":8}"#
        );
        assert_eq!((spec.gpus, spec.bits), (2, 8));
        let model = &spec.models[0];
        assert_eq!(
            (
                model.parameter.as_str(),
                model.repository.as_str(),
                model.release.as_str(),
                model.lane.as_str()
            ),
            ("model", "fidika/sdxl-base", "1.0", "int8")
        );
        let adapter = &model.adapters[0];
        assert_eq!(
            (
                adapter.component.as_str(),
                adapter.model.as_str(),
                adapter.scale.as_str()
            ),
            ("unet", "fidika/detail", "0.5")
        );
        assert_eq!(adapter.manifest, format!("sha256:{}", "a".repeat(64)));

        let submit = call.submission(On::Any, "https://tensorhub.com", &spec, Some(&bound), None);
        assert_eq!(
            (submit.kind, submit.target.as_str()),
            (RunKind::Call, "fidika/sdxl/generate")
        );
        assert_eq!(v1::RunSpec::decode(submit.spec.as_slice()).unwrap(), spec);
        assert!(submit.inputs.is_empty() && submit.warnings.is_empty());
    }

    #[test]
    fn a_warm_run_sets_one_warm_item() {
        let sdxl = interface("sdxl-2.10.0");
        let generate = sdxl
            .callables
            .iter()
            .find(|c| c.name == "generate")
            .unwrap();
        let models = warm_choices(generate, &terms(&["model.model=fidika/base@2"])).unwrap();
        let call = Call {
            package: "fidika/sdxl",
            source: Source::Release(Some("2.10.0")),
            callable: generate,
            gpus: 0,
            bits: 16,
        };
        let spec = call.warm(Warm::Gpu, &models).unwrap();
        let item = &spec.set.as_ref().unwrap().items[0];
        assert_eq!(
            (spec.kind, item.level, item.bits),
            (v1::RunKind::Warm as i32, v1::WarmLevel::Gpu as i32, 16)
        );
        assert_eq!(item.entrypoint, "generate");
        assert_eq!(item.models[0].repository, "fidika/base");
        assert_eq!(
            call.warm(Warm::Off, &[]).unwrap().set.unwrap().items[0].level,
            0
        );
        assert_eq!(
            call.submission(On::Machine("local".into()), "h", &spec, None, None)
                .kind,
            RunKind::Warm
        );
        assert_eq!(call.target(), "fidika/sdxl/generate@2.10.0");
    }
}
