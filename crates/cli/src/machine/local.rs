//! This computer's machine, `local`: Calcifer starts it when something uses it and it exits
//! after 15 minutes idle, so nothing enables it. `cozy machine authorize` lets another
//! computer's controller use it, which keeps it up.
use std::{
    io,
    path::{Path, PathBuf},
    process::ExitCode,
};

use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use platform::lock::{Kind as LockKind, LockFile};
use protocol::{
    calcifer::Machine,
    error::{Error, Kind},
};
use serde::Serialize;

use crate::{Context, output};

/// The machine's root as TensorD reads it: the config's `[machine] root` as written now.
fn machine_root(ctx: &Context) -> Result<PathBuf, Error> {
    Ok(machine_config(ctx)?.root(&ctx.home))
}

fn machine_config(ctx: &Context) -> Result<protocol::config::MachineConfig, Error> {
    let path = ctx.home.config();
    let text = std::fs::read_to_string(&path).unwrap_or_default();
    let (config, _) = protocol::config::Config::parse(&text, &path.display().to_string())?;
    Ok(config.machine)
}

/// For `cozy uninstall`: the machine's root, and the model store it keeps there (none when the
/// config names a store elsewhere, which may be shared).
pub(crate) fn root_and_store(ctx: &Context) -> Result<(PathBuf, Option<PathBuf>), Error> {
    let config = machine_config(ctx)?;
    let root = config.root(&ctx.home);
    let store = config
        .store
        .is_none()
        .then(|| root.join("var").join("lib").join("tensorfs"));
    Ok((root, store))
}

/// For `cozy uninstall`, once Calcifer is down: ends the machine now, with whatever it runs, and
/// waits until every process it was has exited.
pub(crate) fn end(root: &Path) -> Result<(), Error> {
    platform::process::end_holder(&lock(root))
        .map_err(|e| Error::new(Kind::Unavailable, "machine_stop_failed", e.to_string()))
}

fn lock(root: &Path) -> PathBuf {
    root.join("var/lib/cozy/machine/agent.lock")
}

/// What this computer's machine does with the binary `cozy update` installed: none runs, so the
/// next use starts it; or the one running moves to it once idle.
#[derive(Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub(crate) enum Adopted {
    NotRunning,
    MovesWhenIdle,
}

pub(crate) fn adopt(ctx: &Context) -> Result<Adopted, Error> {
    let lock = lock(&machine_root(ctx)?);
    let free = !lock.exists() || matches!(LockFile::new(lock).try_lock(LockKind::Exclusive), Ok(Some(_)));
    Ok(match free {
        true => Adopted::NotRunning,
        false => Adopted::MovesWhenIdle,
    })
}

/// `cozy machine authorize <public-key>`: lets the controller with this Ed25519 key (as `cozy
/// machine add` printed it on that computer) use this computer's machine. TensorD reads the file
/// again when a signer is unknown: no restart. While any controller is authorized the machine
/// stays up, and Calcifer starts it with itself at login; it starts now, saying where it serves.
pub async fn authorize(ctx: &Context, key: &str) -> Result<ExitCode, Error> {
    let key = key.trim();
    let changed = authorized(ctx, key)?;
    let path = ctx.home.keys().join("authorized");
    let local: Machine = ctx.calcifer().await?.get("/v1/machines/local").await?;
    if ctx.json {
        output::json(&serde_json::json!({"authorized": key, "changed": changed, "machine": local}));
        return Ok(ExitCode::SUCCESS);
    }
    let mut record = output::Record::new()
        .field("authorized", key)
        .field("in", path.display().to_string());
    if let Some(address) = &local.address {
        record = record.field("address", address);
    }
    if let Some(fingerprint) = &local.fingerprint {
        record = record.field("fingerprint", fingerprint.to_string());
    }
    record
        .note(match changed {
            false => "it was already authorized",
            true => "that controller can now use this machine, which stays up for it",
        })
        .print(ctx);
    Ok(ExitCode::SUCCESS)
}

/// Adds `key` to `keys/authorized` unless it is there: true when it was added.
fn authorized(ctx: &Context, key: &str) -> Result<bool, Error> {
    let valid = URL_SAFE_NO_PAD
        .decode(key)
        .ok()
        .and_then(|raw| <[u8; 32]>::try_from(raw).ok())
        .is_some_and(|raw| ed25519_dalek::VerifyingKey::from_bytes(&raw).is_ok());
    if !valid {
        return Err(Error::new(
            Kind::Validation,
            "public_key_invalid",
            format!("{key:?} is not a controller's public key (32 bytes of unpadded base64url)"),
        )
        .next("on that computer: install cozy (`curl -LsSf https://tensorhub.com/install.sh | sh`), then cozy machine add <name> <address> --fingerprint <sha256:…>"));
    }
    let path = ctx.home.keys().join("authorized");
    let unwritable = |e: io::Error| {
        Error::new(
            Kind::Internal,
            "keys_unwritable",
            format!("{}: {e}", path.display()),
        )
    };
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(e) if e.kind() == io::ErrorKind::NotFound => String::new(),
        Err(e) => return Err(unwritable(e)),
    };
    let held = text
        .lines()
        .any(|line| line.split('#').next().unwrap_or_default().trim() == key);
    if !held {
        let mut text = text;
        if !text.is_empty() && !text.ends_with('\n') {
            text.push('\n');
        }
        text.push_str(key);
        text.push('\n');
        write_private(&path, &text).map_err(unwritable)?;
    }
    Ok(!held)
}

/// Writes `text` to `path` whole at 0600, its directory 0700.
fn write_private(path: &Path, text: &str) -> io::Result<()> {
    let dir = path.parent().expect("a keys file has a directory");
    platform::durable::private_dir(dir).map_err(io::Error::other)?;
    platform::durable::replace(path, text.as_bytes(), 0o600).map_err(io::Error::other)
}
