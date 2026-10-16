//! The one model-reference grammar, shared by `cozy model …` and `cozy run model.<slot>=…`:
//!
//! ```text
//! org/name[@release[/lane]]     a Hub model by name; the machine resolves the release and picks the lane
//! org/name#sha256:<64 hex>      the owner's checkpoint by hash
//! local/<alias>                 a model a machine holds under that name
//! hf://… civitai://… https://…  a provider source, passed through: the machine pins and makes it
//! ```
use std::fmt;

use cozy_proto::error::{Error, Kind};
use cozy_proto::machine::ModelChoice;

/// A Hub repository, `org/name`, lowercased (names are case-insensitive).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Repo {
    pub org: String,
    pub name: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ModelRef {
    Hub {
        repo: Repo,
        release: Option<String>,
        lane: Option<String>,
    },
    Checkpoint {
        repo: Repo,
        manifest: String,
    },
    Local(String),
    Provider(String),
}

const GRAMMAR: &str = "org/name[@release[/lane]], org/name#sha256:<hex>, local/<alias>, hf://…, civitai://… or https://…";

fn usage(code: &str, message: String) -> Error {
    Error::new(Kind::Usage, code, message)
}

/// A Hub name segment: 1–64 of `a-z0-9._-`, alphanumeric at both ends.
fn slug(part: &str) -> bool {
    let b = part.as_bytes();
    (1..=64).contains(&b.len())
        && b[0].is_ascii_alphanumeric()
        && b[b.len() - 1].is_ascii_alphanumeric()
        && b.iter()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'.' | b'_' | b'-'))
}

/// A release or lane label: alphanumeric first, then up to 63 of `A-Za-z0-9._+!-`.
pub(crate) fn label(flag: &str, value: &str) -> Result<String, Error> {
    let b = value.as_bytes();
    let ok = (1..=64).contains(&b.len())
        && b[0].is_ascii_alphanumeric()
        && b.iter()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'.' | b'_' | b'+' | b'!' | b'-'));
    if ok {
        Ok(value.to_owned())
    } else {
        Err(usage(
            "model_label_invalid",
            format!("{flag} {value:?}: start alphanumeric, then at most 63 of letters, digits, . _ + ! -"),
        ))
    }
}

/// A checkpoint id: `sha256:` and 64 lowercase hex digits.
pub(crate) fn checkpoint(value: &str) -> Result<String, Error> {
    match value.strip_prefix("sha256:") {
        Some(hex)
            if hex.len() == 64
                && hex.bytes().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c)) =>
        {
            Ok(value.to_owned())
        }
        _ => Err(usage(
            "checkpoint_invalid",
            format!("{value:?} is not a checkpoint id"),
        )
        .remedy("write sha256:<64 lowercase hex>")),
    }
}

impl Repo {
    pub(crate) fn parse(raw: &str) -> Result<Self, Error> {
        let lower = raw.trim().to_ascii_lowercase();
        match lower.split_once('/') {
            Some((org, name)) if slug(org) && slug(name) => Ok(Repo {
                org: org.to_owned(),
                name: name.to_owned(),
            }),
            _ => Err(usage(
                "model_name_invalid",
                format!("{raw:?} is not org/name"),
            )
            .remedy("org and name use letters, digits, dots, underscores or hyphens")),
        }
    }

    /// A destination on the Hub: never `local/`.
    pub(crate) fn hub(raw: &str) -> Result<Self, Error> {
        let repo = Self::parse(raw)?;
        if repo.org == "local" {
            return Err(usage(
                "model_local_destination",
                format!("{raw} is a local alias; local/ names stay on a machine"),
            )
            .remedy(format!("name a repository of your account, e.g. <you>/{}", repo.name)));
        }
        Ok(repo)
    }
}

impl fmt::Display for Repo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.org, self.name)
    }
}

impl ModelRef {
    pub(crate) fn parse(raw: &str) -> Result<Self, Error> {
        let raw = raw.trim();
        if let Some((scheme, _)) = raw.split_once("://") {
            return match scheme {
                "hf" | "civitai" | "https" => Ok(ModelRef::Provider(raw.to_owned())),
                _ => Err(usage(
                    "model_source_scheme_unsupported",
                    format!("{raw:?} uses neither hf://, civitai:// nor https://"),
                )),
            };
        }
        if let Some((name, hash)) = raw.split_once('#') {
            let repo = Repo::parse(name).map_err(|_| grammar(raw))?;
            let manifest = checkpoint(hash).map_err(|e| {
                e.remedy("`#` names a checkpoint, org/name#sha256:<hex>; a lane is org/name@release/lane")
            })?;
            return Ok(ModelRef::Checkpoint { repo, manifest });
        }
        let (name, pin) = match raw.split_once('@') {
            Some((name, pin)) => (name, Some(pin)),
            None => (raw, None),
        };
        let repo = Repo::parse(name).map_err(|_| grammar(raw))?;
        if repo.org == "local" {
            return match pin {
                None => Ok(ModelRef::Local(repo.name)),
                Some(_) => Err(usage(
                    "model_ref_invalid",
                    format!("{raw:?}: a local alias has no releases"),
                )),
            };
        }
        let (release, lane) = match pin.map(|p| p.split_once('/').ok_or(p)) {
            None => (None, None),
            Some(Err(release)) => (Some(label("release", release)?), None),
            Some(Ok((release, lane))) => (Some(label("release", release)?), Some(label("lane", lane)?)),
        };
        Ok(ModelRef::Hub {
            repo,
            release,
            lane,
        })
    }

    /// The machine's choice for `parameter`.
    pub(crate) fn choice(&self, parameter: &str) -> ModelChoice {
        let mut choice = ModelChoice {
            parameter: parameter.to_owned(),
            ..Default::default()
        };
        match self {
            ModelRef::Hub {
                repo,
                release,
                lane,
            } => {
                choice.repository = repo.to_string();
                choice.release = release.clone().unwrap_or_default();
                choice.lane = lane.clone().unwrap_or_default();
            }
            ModelRef::Checkpoint { repo, manifest } => {
                choice.repository = repo.to_string();
                choice.manifest = manifest.clone();
            }
            ModelRef::Local(alias) => choice.repository = format!("local/{alias}"),
            ModelRef::Provider(source) => choice.source = source.clone(),
        }
        choice
    }

    /// The Hub repository it names, if any.
    pub(crate) fn repo(&self) -> Option<&Repo> {
        match self {
            ModelRef::Hub { repo, .. } | ModelRef::Checkpoint { repo, .. } => Some(repo),
            _ => None,
        }
    }
}

impl fmt::Display for ModelRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelRef::Hub {
                repo,
                release,
                lane,
            } => {
                write!(f, "{repo}")?;
                if let Some(release) = release {
                    write!(f, "@{release}")?;
                }
                if let Some(lane) = lane {
                    write!(f, "/{lane}")?;
                }
                Ok(())
            }
            ModelRef::Checkpoint { repo, manifest } => write!(f, "{repo}#{manifest}"),
            ModelRef::Local(alias) => write!(f, "local/{alias}"),
            ModelRef::Provider(source) => f.write_str(source),
        }
    }
}

fn grammar(raw: &str) -> Error {
    usage(
        "model_ref_invalid",
        format!("{raw:?} is not a model: use {GRAMMAR}"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const HASH: &str = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    #[test]
    fn one_grammar_names_each_kind_of_model() {
        let cases = [
            ("Fidika/SDXL", "fidika/sdxl", ("fidika/sdxl", "", "", "", "")),
            ("fidika/sdxl@1.0.0", "fidika/sdxl@1.0.0", ("fidika/sdxl", "1.0.0", "", "", "")),
            ("fidika/sdxl@1.0.0/int8", "fidika/sdxl@1.0.0/int8", ("fidika/sdxl", "1.0.0", "int8", "", "")),
            (&format!("fidika/sdxl#{HASH}"), &format!("fidika/sdxl#{HASH}"), ("fidika/sdxl", "", "", HASH, "")),
            ("local/Style", "local/style", ("local/style", "", "", "", "")),
            ("hf://Org/Repo@main/a.safetensors", "hf://Org/Repo@main/a.safetensors", ("", "", "", "", "hf://Org/Repo@main/a.safetensors")),
            ("civitai://12345", "civitai://12345", ("", "", "", "", "civitai://12345")),
        ];
        for (raw, shown, (repository, release, lane, manifest, source)) in cases {
            let parsed = ModelRef::parse(raw).unwrap();
            assert_eq!(parsed.to_string(), shown);
            let c = parsed.choice("model");
            assert_eq!(
                (c.parameter.as_str(), c.repository.as_str(), c.release.as_str(), c.lane.as_str(), c.manifest.as_str(), c.source.as_str()),
                ("model", repository, release, lane, manifest, source),
                "{raw}"
            );
        }
    }

    #[test]
    fn malformed_spellings_are_usage_refusals() {
        for raw in [
            "sdxl",
            "a/b/c",
            "fidika/sdxl#fp8",
            "fidika/sdxl#sha256:nope",
            "fidika/sdxl#",
            "fidika/sdxl@",
            "fidika/sdxl@1.0/",
            "fidika/sdxl@1.0/fp8#sha256:aa",
            "local/style@1.0",
            "s3://bucket/key",
            "-org/x",
        ] {
            let error = ModelRef::parse(raw).unwrap_err();
            assert_eq!(error.kind, Kind::Usage, "{raw}");
        }
        assert_eq!(Repo::hub("local/x").unwrap_err().code, "model_local_destination");
    }
}
