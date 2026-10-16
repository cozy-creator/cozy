//! `~/.cozy/config.toml`, loaded once per process through figment and shared by the CLI and
//! Calcifer. Sources, each over the last: the file, the environment (two credentials and where
//! releases come from, nothing else), the command line. The home is always `<user home>/.cozy`. A key this build does not read is named once
//! and ignored, so a file another version wrote never disables a command.
use std::{collections::BTreeMap, fmt, path::PathBuf};

use cozy_platform::paths::Home;
use figment::{
    Figment,
    providers::{Env, Format, Serialized, Toml},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use url::Url;

use crate::error::{Error, Kind};

/// The Hub every install knows by name.
pub const DEFAULT_HUB: &str = "tensorhub";
pub const DEFAULT_HUB_URL: &str = "https://tensorhub.com";

/// Every environment variable the load reads: credentials for model sources, by the names their
/// own tools use, and where releases are downloaded from (dist's installers read the same one).
const ENV: [(&str, &str); 3] = [
    ("HF_TOKEN", "huggingface_token"),
    ("CIVITAI_TOKEN", "civitai_token"),
    ("COZY_DOWNLOAD_URL", "download_url"),
];

#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(default)]
pub struct Config {
    /// The current Hub: a name in `hubs`, or a URL. Absent: `tensorhub`.
    pub hub: Option<String>,
    /// Named Hubs, like kubectl contexts: name (lowercase) → canonical URL.
    pub hubs: BTreeMap<String, String>,
    pub calcifer: CalciferConfig,
    pub huggingface_token: Option<Secret>,
    pub civitai_token: Option<Secret>,
    /// The directory holding one release's files (dist-manifest.json, the archives); absent: GitHub
    /// Releases.
    pub download_url: Option<Url>,
    /// `--tensorhub`: this command's Hub. From the command line only.
    pub tensorhub: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(default)]
pub struct CalciferConfig {
    /// Calcifer's loopback port; 0 takes any free one (its lock record says which).
    pub port: u16,
}

/// A credential: never printed.
#[derive(Clone, PartialEq, Eq, Deserialize)]
#[serde(transparent)]
pub struct Secret(String);

impl Secret {
    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for Secret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Secret(…)")
    }
}

/// One Tensorhub, named or by URL.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
pub struct Hub {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Canonical base URL: lowercase host, no default port, no trailing slash. Every record and
    /// credential is keyed by it.
    #[serde(rename = "url")]
    pub origin: String,
}

/// What the command line contributes to the load.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Flags {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tensorhub: Option<String>,
}

/// One process's configuration and where it came from.
#[derive(Clone, Debug)]
pub struct Loaded {
    pub config: Config,
    pub home: Home,
    /// The user's home directory, which services and `~` are relative to.
    pub user_home: PathBuf,
    /// What the load ignored, for the caller to show once.
    pub warnings: Vec<String>,
}

/// The one read of the environment and the config file.
pub fn load(flags: Flags) -> Result<Loaded, Error> {
    let user_home = std::env::home_dir()
        .ok_or_else(|| Error::new(Kind::Internal, "no_home", "this user has no home directory"))?;
    let home = Home::under(&user_home);
    let path = home.config();
    let mut figment = Figment::new();
    if path.exists() {
        figment = figment.merge(Toml::file_exact(&path));
    }
    for (var, key) in ENV {
        figment = figment.merge(Env::raw().only(&[var]).map(move |_| key.into()));
    }
    let figment = figment.merge(Serialized::defaults(flags));
    let (config, warnings) = extract(figment, &path.display().to_string())?;
    Ok(Loaded {
        config,
        home,
        user_home,
        warnings,
    })
}

impl Config {
    /// A config file's text as the load would read it: how an edit proves it still loads.
    pub fn parse(text: &str, name: &str) -> Result<(Config, Vec<String>), Error> {
        extract(Figment::from(Toml::string(text)), name)
    }

    /// Every named Hub, `tensorhub` included, by name.
    pub fn hubs(&self) -> Vec<Hub> {
        let mut hubs: BTreeMap<&str, &str> = [(DEFAULT_HUB, DEFAULT_HUB_URL)].into();
        hubs.extend(self.hubs.iter().map(|(n, u)| (n.as_str(), u.as_str())));
        hubs.into_iter()
            .map(|(name, origin)| Hub {
                name: Some(name.to_owned()),
                origin: origin.to_owned(),
            })
            .collect()
    }

    /// A Hub's name in any case, or its URL; a URL takes the name it has here.
    pub fn resolve(&self, text: &str) -> Result<Hub, Error> {
        let text = text.trim();
        let hubs = self.hubs();
        let lower = text.to_ascii_lowercase();
        if let Some(hub) = hubs.iter().find(|h| h.name.as_deref() == Some(&lower)) {
            return Ok(hub.clone());
        }
        if !text.contains("://") {
            return Err(Error::new(
                Kind::NotFound,
                "hub_not_found",
                format!("no Hub is named {text:?}"),
            )
            .next("cozy hub list")
            .next("cozy hub add <name> <url>"));
        }
        let origin = origin(text)?;
        let name = hubs
            .into_iter()
            .find(|h| h.origin == origin)
            .and_then(|h| h.name);
        Ok(Hub { name, origin })
    }

    /// The current Hub (`hub`, set by `cozy hub use`).
    pub fn current(&self) -> Result<Hub, Error> {
        self.resolve(self.hub.as_deref().unwrap_or(DEFAULT_HUB))
            .map_err(|e| e.remedy("set `hub` with `cozy hub use <name>`"))
    }

    /// Where new work goes: `--tensorhub`, else the current Hub.
    pub fn selected(&self) -> Result<Hub, Error> {
        match self.narrowed()? {
            Some(hub) => Ok(hub),
            None => self.current(),
        }
    }

    /// `--tensorhub`'s Hub: list verbs show every Hub without it.
    pub fn narrowed(&self) -> Result<Option<Hub>, Error> {
        self.tensorhub
            .as_deref()
            .map(|t| self.resolve(t))
            .transpose()
    }

    /// Lowercases names and canonicalizes URLs, dropping (and naming) entries that are neither.
    fn normalize(&mut self, file: &str, warnings: &mut Vec<String>) {
        let mut hubs = BTreeMap::new();
        for (name, url) in std::mem::take(&mut self.hubs) {
            let lower = name.to_ascii_lowercase();
            match (valid_name(&lower), origin(&url)) {
                (true, Ok(origin)) => {
                    hubs.insert(lower, origin);
                }
                (false, _) => warnings.push(format!(
                    "{file}: hub name {name:?} is not letters, digits, '.', '_' or '-'; ignored"
                )),
                (true, Err(e)) => {
                    warnings.push(format!("{file}: hub {name:?}: {}; ignored", e.message))
                }
            }
        }
        self.hubs = hubs;
        for secret in [&mut self.huggingface_token, &mut self.civitai_token] {
            if secret.as_ref().is_some_and(|s| s.0.trim().is_empty()) {
                *secret = None;
            }
        }
    }
}

impl Hub {
    /// How a person reads it: its name, else its URL.
    pub fn label(&self) -> &str {
        self.name.as_deref().unwrap_or(&self.origin)
    }

    /// This computer's device key at this Hub (hub-client's `DeviceKey`), mode 0600.
    pub fn device_key(&self, home: &Home) -> PathBuf {
        home.keys().join(format!("device-{}.json", self.key_name()))
    }

    /// The short Bearer that device key last minted, kept for the next process.
    pub fn bearer(&self, home: &Home) -> PathBuf {
        home.keys()
            .join(format!("device-{}.bearer.json", self.key_name()))
    }

    fn key_name(&self) -> String {
        hex::encode(Sha256::digest(self.origin.as_bytes()))
    }
}

impl fmt::Display for Hub {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.name {
            Some(name) => write!(f, "{name} ({})", self.origin),
            None => f.write_str(&self.origin),
        }
    }
}

/// Whether `name` (already lowercase) can name a Hub. A name is never a URL.
pub fn valid_name(name: &str) -> bool {
    let mut chars = name.chars();
    name.len() <= 63
        && chars
            .next()
            .is_some_and(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
        && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || "._-".contains(c))
}

/// A Tensorhub base URL's canonical spelling.
pub fn origin(text: &str) -> Result<String, Error> {
    let text = text.trim();
    let url = Url::parse(text).ok().filter(|u| {
        matches!(u.scheme(), "http" | "https")
            && u.host_str().is_some()
            && u.username().is_empty()
            && u.password().is_none()
            && u.query().is_none()
            && u.fragment().is_none()
    });
    match url {
        Some(url) => Ok(url.as_str().trim_end_matches('/').to_owned()),
        None => Err(Error::new(
            Kind::Usage,
            "hub_invalid",
            format!("{text:?} is not an http or https URL without credentials, query or fragment"),
        )),
    }
}

fn extract(figment: Figment, file: &str) -> Result<(Config, Vec<String>), Error> {
    let invalid = |e: figment::Error| {
        let key = e.path.join(".");
        let at = if key.is_empty() {
            String::new()
        } else {
            format!(" `{key}`")
        };
        Error::new(
            Kind::Usage,
            "config_invalid",
            format!("{file}{at}: {}", e.kind),
        )
        .remedy(format!("correct or remove it in {file}"))
    };
    let mut config: Config = figment.extract().map_err(invalid)?;
    let mut warnings = vec![];
    let value: figment::value::Value = figment.extract().map_err(invalid)?;
    let _: Result<Config, _> = serde_ignored::deserialize(&value, |path| {
        warnings.push(format!(
            "{file}: `{path}` is not a setting this cozy reads; ignored"
        ))
    });
    config.normalize(file, &mut warnings);
    Ok((config, warnings))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(text: &str) -> (Config, Vec<String>) {
        Config::parse(text, "config.toml").unwrap()
    }

    #[test]
    fn urls_have_one_spelling() {
        for (text, canonical) in [
            ("HTTPS://Example.TEST:443/", "https://example.test"),
            ("http://127.0.0.1:8819/", "http://127.0.0.1:8819"),
            ("https://hub.test/api/", "https://hub.test/api"),
        ] {
            assert_eq!(origin(text).unwrap(), canonical);
        }
        for bad in [
            "ftp://x",
            "https://u:p@x",
            "https://x/?q=1",
            "x",
            "https://x#f",
        ] {
            assert_eq!(origin(bad).unwrap_err().code, "hub_invalid", "{bad}");
        }
    }

    #[test]
    fn hubs_resolve_by_name_in_any_case_or_by_url() {
        let (mut config, warnings) = parse(
            "hub = \"LOCAL\"\nfuture = 1\n[hubs]\nLocal = \"http://127.0.0.1:8819/\"\n\"Bad Name\" = \"http://x\"\n[calcifer]\nport = 7\nidle = 3\n",
        );
        assert_eq!(config.calcifer.port, 7);
        assert!(
            warnings.iter().any(|w| w.contains("`future`")),
            "{warnings:?}"
        );
        assert!(
            warnings.iter().any(|w| w.contains("`calcifer.idle`")),
            "{warnings:?}"
        );
        assert!(
            warnings.iter().any(|w| w.contains("Bad Name")),
            "{warnings:?}"
        );
        let local = config.current().unwrap();
        assert_eq!(local.name.as_deref(), Some("local"));
        assert_eq!(local.origin, "http://127.0.0.1:8819");
        assert_eq!(config.resolve("HTTP://127.0.0.1:8819").unwrap(), local);
        assert_eq!(config.resolve("tensorhub").unwrap().origin, DEFAULT_HUB_URL);
        assert_eq!(config.resolve("https://other.test").unwrap().name, None);
        assert_eq!(config.resolve("nosuch").unwrap_err().code, "hub_not_found");
        config.tensorhub = Some("tensorhub".into());
        assert_eq!(config.selected().unwrap().origin, DEFAULT_HUB_URL);
        assert_ne!(
            local.device_key(&Home::at("/h")),
            config.selected().unwrap().device_key(&Home::at("/h"))
        );
    }

    #[test]
    fn a_wrong_type_names_its_key() {
        let error = Config::parse("[calcifer]\nport = \"soon\"\n", "config.toml").unwrap_err();
        assert_eq!(error.code, "config_invalid");
        assert!(
            error.message.contains("`calcifer.port`"),
            "{}",
            error.message
        );
    }
}
