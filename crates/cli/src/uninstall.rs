//! `cozy uninstall`: removes what cozy put on this computer: its machine and Calcifer with their
//! services, `~/.cozy` (this binary with it), the installer's `~/.config/cozy` and shell lines, and
//! on Windows its PATH entry. The machine's model store is TensorFS's cache, which `tfs` and Python
//! TensorFS share: it stays, and `--all` says how to remove it.
use std::{
    fs,
    io::{self, IsTerminal as _, Write as _},
    path::{Path, PathBuf},
    process::ExitCode,
};

use platform::{process, service::Services};
use protocol::{
    calcifer::{MachineKind, MachineState},
    error::{Error, Kind},
};
use serde::Serialize;

use crate::{Context, machine, output};

#[derive(clap::Args, Debug)]
pub struct Args {
    /// Name the model store (TensorFS's shared cache) with the command that removes it; it is
    /// left in place.
    #[arg(long)]
    all: bool,
}

/// The rc files the installer writes a `. "$HOME/.config/cozy/env.sh"` line to, under the home.
const RC_FILES: [&str; 7] = [
    ".profile",
    ".bashrc",
    ".bash_profile",
    ".bash_login",
    ".zshrc",
    ".zshenv",
    ".config/fish/conf.d/cozy.env.fish",
];

#[derive(Serialize)]
struct Item {
    path: PathBuf,
    bytes: u64,
    /// For a kept store under `--all`: the command that removes it.
    #[serde(skip_serializing_if = "Option::is_none")]
    remove: Option<String>,
}

pub async fn exec(ctx: &Context, args: Args) -> Result<ExitCode, Error> {
    // A Calcifer that cannot start holds nothing to stop or ask about.
    let calcifer = match ctx.calcifer().await {
        Ok(calcifer) => calcifer.machines().await.ok().map(|list| (calcifer, list)),
        Err(_) => None,
    };
    if let Some((_, list)) = &calcifer {
        let live: Vec<String> = (list.machines.iter())
            .filter(|m| m.kind == MachineKind::Rental && m.state != MachineState::Ended)
            .map(|m| m.name.clone())
            .collect();
        if let Some(first) = live.first() {
            return Err(Error::new(
                Kind::Conflict,
                "rentals_live",
                format!("rentals still bill: {}", live.join(", ")),
            )
            .remedy("end them first: uninstalling forgets their keys")
            .next(format!("cozy machine end {first}")));
        }
    }
    let (root, store) = machine::root_and_store(ctx)?;
    // TensorFS's cache, shared with `tfs` and Python TensorFS: kept, `--all` too (owner ruling).
    let keep: Vec<PathBuf> = store.into_iter().filter(|store| store.exists()).collect();
    let mut dirs = vec![ctx.home.root().to_path_buf(), root.clone(), config_dir(ctx)];
    if cfg!(windows) && platform::paths::is_login_home(&ctx.user_home) {
        dirs.extend(env("LOCALAPPDATA").map(|dir| dir.join("cozy")));
    }
    let dirs: Vec<PathBuf> = (dirs.iter())
        .filter(|dir| dir.exists() && !dirs.iter().any(|d| d != *dir && dir.starts_with(d)))
        .cloned()
        .collect();
    let item = |path: &PathBuf, keep: &[PathBuf]| Item {
        path: path.clone(),
        bytes: size(path, keep),
        remove: None,
    };
    let removed: Vec<Item> = dirs.iter().map(|dir| item(dir, &keep)).collect();
    let kept: Vec<Item> = (keep.iter())
        .map(|dir| Item {
            remove: args.all.then(|| removal(dir)),
            ..item(dir, &[])
        })
        .collect();
    let rc: Vec<PathBuf> = (rc_files(ctx).into_iter())
        .filter(|file| unsourced(file).is_some())
        .collect();
    if !ctx.yes {
        if !ctx.json {
            plan(ctx, &removed, &kept, &rc);
        }
        confirm()?;
    }

    machine::end(ctx, &root)?;
    if let Some((calcifer, _)) = calcifer {
        let tree = calcifer.holder().map_or(vec![], |h| process::tree(h.pid));
        calcifer.stop().await?;
        process::end_tree(&tree)
            .map_err(|e| Error::new(Kind::Unavailable, "calcifer_stop_failed", e.to_string()))?;
    }
    let services = Services::new(&ctx.home, &ctx.user_home);
    if services.available() {
        services
            .uninstall("calcifer")
            .map_err(|e| Error::new(Kind::Unavailable, "calcifer_stop_failed", e.to_string()))?;
    }
    for file in &rc {
        if let Some(text) = unsourced(file) {
            match file.ends_with("cozy.env.fish") && text.trim().is_empty() {
                true => fs::remove_file(file),
                false => fs::write(file, text),
            }
            .map_err(|e| failed(file, e))?;
        }
    }
    #[cfg(windows)]
    platform::paths::remove_from_user_path(&ctx.home.bin())
        .map_err(|e| Error::new(Kind::Internal, "path_unwritable", e.to_string()))?;
    // Windows cannot delete a running program: it is moved outside the tree, deleted at exit.
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = dirs.iter().find(|dir| exe.starts_with(dir))
    {
        self_replace::self_delete_outside_path(dir).map_err(|e| failed(&exe, e))?;
    }
    for dir in &dirs {
        remove(dir, &keep)?;
    }

    if ctx.json {
        output::json(&serde_json::json!({"removed": removed, "kept": kept, "rc_files": rc}));
        return Ok(ExitCode::SUCCESS);
    }
    let total = removed.iter().map(|item| item.bytes).sum();
    let mut record = output::Record::new().field("removed", output::bytes(total));
    for item in &kept {
        record = record.note(kept_store(item));
    }
    record.note("cozy is uninstalled").print(ctx);
    Ok(ExitCode::SUCCESS)
}

/// What it removes and keeps, and their sizes, before it asks.
fn plan(ctx: &Context, removed: &[Item], kept: &[Item], rc: &[PathBuf]) {
    let mut table = output::Table::new(ctx, ["REMOVES", "SIZE"]);
    for item in removed {
        table = table.row([item.path.display().to_string(), output::bytes(item.bytes)]);
    }
    for file in rc {
        table = table.row([format!("its line in {}", file.display()), String::new()]);
    }
    for item in kept {
        table = table.note(kept_store(item));
    }
    table.print();
}

/// The kept store, its size, and (`--all`) how to remove it.
fn kept_store(item: &Item) -> String {
    let (path, size) = (item.path.display(), output::bytes(item.bytes));
    match &item.remove {
        Some(command) => format!(
            "keeps TensorFS's model cache at {path} ({size}), which `tfs` and Python TensorFS share; remove it with `{command}`"
        ),
        None => format!(
            "keeps TensorFS's model cache at {path} ({size}); `--all` says how to remove it"
        ),
    }
}

/// The shell command that removes `dir`.
fn removal(dir: &Path) -> String {
    let path = dir.display().to_string();
    match cfg!(windows) {
        true => format!("Remove-Item -Recurse -Force '{}'", path.replace('\'', "''")),
        false => format!("rm -rf '{}'", path.replace('\'', "'\\''")),
    }
}

/// A person's yes; without a terminal, `--yes` is the only one.
fn confirm() -> Result<(), Error> {
    let refused = |message: &str| {
        Error::new(Kind::Usage, "uninstall_unconfirmed", message).next("cozy uninstall --yes")
    };
    if !io::stdin().is_terminal() {
        return Err(refused("nothing asked: no terminal to answer"));
    }
    anstream::eprint!("Remove these? [y/N] ");
    let _ = io::stderr().flush();
    let mut line = String::new();
    io::stdin()
        .read_line(&mut line)
        .map_err(|e| refused(&e.to_string()))?;
    match line.trim().eq_ignore_ascii_case("y") || line.trim().eq_ignore_ascii_case("yes") {
        true => Ok(()),
        false => Err(refused("nothing was removed")),
    }
}

/// The installer's receipt and env scripts: `$XDG_CONFIG_HOME/cozy`, else `~/.config/cozy`.
fn config_dir(ctx: &Context) -> PathBuf {
    env("XDG_CONFIG_HOME")
        .unwrap_or_else(|| ctx.user_home.join(".config"))
        .join("cozy")
}

fn rc_files(ctx: &Context) -> Vec<PathBuf> {
    let zsh = env("ZDOTDIR").unwrap_or_else(|| ctx.user_home.clone());
    (RC_FILES.iter())
        .map(|file| match file.starts_with(".zsh") {
            true => zsh.join(file),
            false => ctx.user_home.join(file),
        })
        .collect()
}

/// `file` without the lines that source cozy's env script (and the blank line the installer put
/// before each); none when it has none.
fn unsourced(file: &Path) -> Option<String> {
    let text = fs::read_to_string(file).ok()?;
    let ours = |line: &str| {
        let line = line.trim();
        let sourced = (line
            .strip_prefix(". ")
            .or_else(|| line.strip_prefix("source ")))
        .map(|path| path.trim_matches('"'));
        sourced.is_some_and(|path| {
            [
                "/cozy/env.sh",
                "/cozy/env.fish",
                "/.cozy/bin/env",
                "/.cozy/bin/env.fish",
            ]
            .iter()
            .any(|end| path.ends_with(end))
        })
    };
    let lines: Vec<&str> = text.lines().collect();
    if !lines.iter().any(|line| ours(line)) {
        return None;
    }
    let mut kept: Vec<&str> = vec![];
    for line in lines {
        if ours(line) {
            if kept.last().is_some_and(|last| last.trim().is_empty()) {
                kept.pop();
            }
        } else {
            kept.push(line);
        }
    }
    let end = if kept.is_empty() { "" } else { "\n" };
    Some(kept.join("\n") + end)
}

/// Bytes under `path`, less what `keep` names.
fn size(path: &Path, keep: &[PathBuf]) -> u64 {
    if keep.iter().any(|k| k == path) {
        return 0;
    }
    match fs::symlink_metadata(path) {
        Ok(meta) if meta.is_dir() => fs::read_dir(path)
            .into_iter()
            .flatten()
            .flatten()
            .map(|entry| size(&entry.path(), keep))
            .sum(),
        Ok(meta) => meta.len(),
        Err(_) => 0,
    }
}

/// Removes `path`, all but what `keep` names under it; a failure names the file it hit.
fn remove(path: &Path, keep: &[PathBuf]) -> Result<(), Error> {
    let at = |e| failed(path, e);
    let meta = match fs::symlink_metadata(path) {
        _ if keep.iter().any(|k| k == path) => return Ok(()),
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(()),
        meta => meta.map_err(at)?,
    };
    if !meta.is_dir() {
        return fs::remove_file(path).map_err(at);
    }
    for entry in fs::read_dir(path).map_err(at)? {
        remove(&entry.map_err(at)?.path(), keep)?;
    }
    match keep.iter().any(|k| k.starts_with(path)) {
        true => Ok(()),
        false => fs::remove_dir(path).map_err(at),
    }
}

fn env(name: &str) -> Option<PathBuf> {
    (protocol::config::environment().get(name))
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn failed(path: &Path, e: io::Error) -> Error {
    Error::new(
        Kind::Internal,
        "uninstall_failed",
        format!("{}: {e}", path.display()),
    )
}
