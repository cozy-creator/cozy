//! `cozy completion <shell>`: the script that wires a shell's <TAB> to clap's dynamic completion
//! (`COMPLETE=<shell> cozy -- <words>`), answered from the grammar itself. Completers read only
//! local files: no Calcifer, no network, no writes.
use std::{ffi::OsStr, process::ExitCode};

use clap::ValueEnum;
use clap_complete::{
    CompletionCandidate,
    env::{Bash, EnvCompleter, Fish, Powershell, Zsh},
};
use cozy_proto::{
    config::{self, Flags},
    error::{Error, Kind},
};

use crate::Context;

/// The environment variable the registration scripts set: clap's completion protocol.
pub const VAR: &str = "COMPLETE";

#[derive(clap::Args, Debug)]
pub struct Args {
    shell: Shell,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Shell {
    Bash,
    Zsh,
    Fish,
    Powershell,
}

pub async fn exec(_ctx: &Context, args: Args) -> Result<ExitCode, Error> {
    let shell: &dyn EnvCompleter = match args.shell {
        Shell::Bash => &Bash,
        Shell::Zsh => &Zsh,
        Shell::Fish => &Fish,
        Shell::Powershell => &Powershell,
    };
    shell
        .write_registration(VAR, "cozy", "cozy", "cozy", &mut std::io::stdout())
        .map_err(|e| Error::new(Kind::Internal, "stdout", e.to_string()))?;
    Ok(ExitCode::SUCCESS)
}

/// Hub names, with each one's URL as its help.
pub fn hubs(current: &OsStr) -> Vec<CompletionCandidate> {
    let Ok(loaded) = config::load(Flags::default()) else {
        return vec![];
    };
    let prefix = current.to_string_lossy().to_ascii_lowercase();
    loaded
        .config
        .hubs()
        .into_iter()
        .filter_map(|hub| {
            let name = hub.name?;
            name.starts_with(&prefix)
                .then(|| CompletionCandidate::new(name).help(Some(hub.origin.into())))
        })
        .collect()
}
