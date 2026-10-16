//! `cozy migrate`: the one-shot import of the Go CLI's `~/.cozy` into this layout (design §7),
//! hidden, and deleted one release after the cutover.
//!
//! It records the Go home's Hubs, keys, live rentals, owed runs and history here, then archives the
//! old state once under `backups/`. User content (`outputs`, `backups`, `scripts`) and the local
//! machine's `machine/` stay where they are, which is where this layout keeps them. Regenerable
//! caches are listed with their size and deleted only by a later `--delete-caches`.
mod go;
mod plan;
#[cfg(feature = "testing")]
pub mod testing;

use std::{
    io::Write as _,
    path::{Path, PathBuf},
    process::ExitCode,
};

use calcifer::store::{Store, StoreError};
use clap::Parser;
use cozy_platform::{durable, paths::Home};
use cozy_proto::error::{Envelope, Error, Kind};
use serde::Serialize;

pub use plan::{Counts, Skip};

#[derive(Parser, Debug)]
#[command(
    name = "cozy migrate",
    about = "Import the Go CLI's home once (hidden; removed after the cutover)"
)]
pub struct Args {
    /// The Go CLI's home.
    #[arg(long, value_name = "DIR")]
    pub from: PathBuf,
    /// This cozy's home; absent: `~/.cozy`.
    #[arg(long, value_name = "DIR")]
    pub into: Option<PathBuf>,
    /// Print what would be imported, archived and left, and change nothing.
    #[arg(long)]
    pub dry_run: bool,
    /// Once the import is proven: delete the regenerable caches, each listed with its size first.
    #[arg(long)]
    pub delete_caches: bool,
    /// Print the report as JSON.
    #[arg(long)]
    pub json: bool,
}

/// User content: never deleted; it stays where this layout keeps it.
const KEPT: [&str; 3] = ["outputs", "backups", "scripts"];
/// What machines and the Hub hand out again.
const CACHES: [&str; 6] = [
    "installs",
    "python",
    "releases",
    "hub-wheels",
    "tmp",
    "publications",
];
/// This layout's own entries, when the import is in place.
const LAYOUT: [&str; 5] = ["bin", "config.toml", "keys", "services", "calcifer"];

/// What a top-level entry of the Go home becomes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Fate {
    Kept,
    /// The local machine's files: the TensorD the Go CLI started still runs from `machine/root`.
    Machine,
    Layout,
    Cache,
    Archived,
}

#[derive(Debug, Serialize)]
pub struct Entry {
    pub name: String,
    pub fate: Fate,
    pub files: u64,
    pub bytes: u64,
}

#[derive(Serialize)]
pub struct Report {
    pub dry_run: bool,
    pub from: PathBuf,
    pub into: PathBuf,
    #[serde(flatten)]
    pub counts: Counts,
    pub entries: Vec<Entry>,
    pub archive: Option<PathBuf>,
    pub deleted: Vec<String>,
    pub skipped: Vec<Skip>,
}

/// `cozy migrate …`, dispatched by the binary before the CLI's own verbs.
pub fn main() -> ExitCode {
    let args = match Args::try_parse_from(std::env::args_os().skip(1)) {
        Ok(args) => args,
        Err(e) => e.exit(),
    };
    match run(&args) {
        Ok(report) => {
            print(&report, args.json);
            ExitCode::SUCCESS
        }
        Err(error) => {
            if args.json {
                println!(
                    "{}",
                    serde_json::to_string(&Envelope {
                        error: error.clone()
                    })
                    .expect("serializes")
                );
            } else {
                println!("error: {} ({})", error.message, error.code);
                error.remedy.iter().for_each(|r| println!("  {r}"));
            }
            ExitCode::from(error.exit_code() as u8)
        }
    }
}

pub fn run(args: &Args) -> Result<Report, Error> {
    let absolute =
        |p: &Path| std::path::absolute(p).map_err(|e| usage(format!("{}: {e}", p.display())));
    let from = absolute(&args.from)?;
    let into = match &args.into {
        Some(dir) => Home::at(absolute(dir)?),
        None => Home::under(
            &std::env::home_dir().ok_or_else(|| usage("this user has no home directory"))?,
        ),
    };
    let same = same_dir(&from, into.root());
    let entries = survey(&from, same)?;
    let mut report = Report {
        dry_run: args.dry_run,
        from: from.clone(),
        into: into.root().to_owned(),
        counts: Counts::default(),
        entries,
        archive: None,
        deleted: vec![],
        skipped: vec![],
    };
    if args.delete_caches {
        delete_caches(&from, &into, &mut report)?;
        return Ok(report);
    }
    if !args.dry_run && imported(&into)? {
        return Err(Error::new(
            Kind::Conflict,
            "already_migrated",
            format!(
                "{} already holds runs or machines",
                into.calcifer_db().display()
            ),
        )
        .remedy("the import runs once"));
    }
    if !from.join("creator.sqlite").is_file() {
        return Err(Error::new(
            Kind::NotFound,
            "no_go_home",
            format!(
                "{} holds no creator.sqlite: it is not the Go CLI's home",
                from.display()
            ),
        ));
    }
    let records = go::records(&from).map_err(|e| invalid("go_home_unreadable", e))?;
    let plan = plan::plan(&from, &into, records).map_err(|e| invalid("go_home_unreadable", e))?;
    report.counts = plan.counts;
    report.skipped = plan.skipped;
    if args.dry_run {
        return Ok(report);
    }
    let failed = |e: String| Error::new(Kind::Internal, "migrate_failed", e);
    let archive = archive(&from, &into, &report.entries).map_err(failed)?;
    let calcifer = from.join("calcifer");
    if same && calcifer.symlink_metadata().is_ok_and(|m| !m.is_dir()) {
        // Go's link to its binary, archived; this layout's Calcifer directory takes its place.
        std::fs::remove_file(&calcifer)
            .map_err(|e| failed(format!("{}: {e}", calcifer.display())))?;
    }
    write_files(&into, &plan.files).map_err(failed)?;
    let store = Store::open(&into.calcifer_db()).map_err(|e| failed(e.to_string()))?;
    store
        .import(&plan.machines, &plan.runs)
        .map_err(|e| failed(e.to_string()))?;
    for entry in report.entries.iter().filter(|e| e.fate == Fate::Archived) {
        remove(&from.join(&entry.name)).map_err(failed)?;
    }
    report.archive = Some(archive);
    Ok(report)
}

fn usage(message: impl Into<String>) -> Error {
    Error::new(Kind::Usage, "usage", message)
}

fn invalid(code: &str, message: String) -> Error {
    Error::new(Kind::Validation, code, message)
}

fn same_dir(a: &Path, b: &Path) -> bool {
    match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(a), Ok(b)) => a == b,
        _ => a == b,
    }
}

/// Every top-level entry of the Go home, what it becomes, and its files and bytes.
fn survey(from: &Path, same: bool) -> Result<Vec<Entry>, Error> {
    let dir = std::fs::read_dir(from)
        .map_err(|e| invalid("go_home_unreadable", format!("{}: {e}", from.display())))?;
    let mut entries: Vec<Entry> = dir
        .flatten()
        .map(|entry| {
            let name = entry.file_name().to_string_lossy().into_owned();
            let real_dir = entry.file_type().is_ok_and(|t| t.is_dir());
            let fate = match name.as_str() {
                n if KEPT.contains(&n) => Fate::Kept,
                "machine" => Fate::Machine,
                n if CACHES.contains(&n) => Fate::Cache,
                "calcifer" if same && real_dir => Fate::Layout,
                n if same && n != "calcifer" && LAYOUT.contains(&n) => Fate::Layout,
                _ => Fate::Archived,
            };
            let (files, bytes) = tally(&entry.path());
            Entry {
                name,
                fate,
                files,
                bytes,
            }
        })
        .collect();
    entries.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(entries)
}

/// Files (anything not a directory) and the bytes of regular files under `path`.
fn tally(path: &Path) -> (u64, u64) {
    let Ok(meta) = std::fs::symlink_metadata(path) else {
        return (0, 0);
    };
    if !meta.is_dir() {
        return (1, if meta.is_file() { meta.len() } else { 0 });
    }
    std::fs::read_dir(path)
        .into_iter()
        .flatten()
        .flatten()
        .fold((0, 0), |(f, b), entry| {
            let (files, bytes) = tally(&entry.path());
            (f + files, b + bytes)
        })
}

/// Whether this layout's store already holds runs or machines.
fn imported(into: &Home) -> Result<bool, Error> {
    if !into.calcifer().is_dir() || !into.calcifer_db().is_file() {
        return Ok(false);
    }
    let store = Store::open(&into.calcifer_db())
        .map_err(|e| Error::new(Kind::Internal, "store", e.to_string()))?;
    match store.import(&[], &[]) {
        Ok(()) => Ok(false),
        Err(StoreError::NotEmpty) => Ok(true),
        Err(e) => Err(Error::new(Kind::Internal, "store", e.to_string())),
    }
}

/// The old state, once, as a private tar under `backups/`, read back before anything is removed.
fn archive(from: &Path, into: &Home, entries: &[Entry]) -> Result<PathBuf, String> {
    let fail = |e: std::io::Error| e.to_string();
    let backups = into.backups();
    std::fs::create_dir_all(&backups).map_err(fail)?;
    let stamp = jiff::Timestamp::now().strftime("%Y%m%dT%H%M%SZ");
    let path = backups.join(format!("go-cli-state-{stamp}.tar"));
    let staged = backups.join(format!(".go-cli-state-{stamp}.tar.partial"));
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
    let mut tar = tar::Builder::new(options.open(&staged).map_err(fail)?);
    tar.follow_symlinks(false);
    let archived: Vec<&Entry> = entries
        .iter()
        .filter(|e| e.fate == Fate::Archived)
        .collect();
    for entry in &archived {
        let source = from.join(&entry.name);
        if source.is_dir() && !source.is_symlink() {
            tar.append_dir_all(&entry.name, &source).map_err(fail)?;
        } else {
            tar.append_path_with_name(&source, &entry.name)
                .map_err(fail)?;
        }
    }
    tar.into_inner()
        .and_then(|file| file.sync_all())
        .map_err(fail)?;
    let mut read = tar::Archive::new(std::fs::File::open(&staged).map_err(fail)?);
    let mut held = (0, 0);
    for item in read.entries().map_err(fail)? {
        let header = item.map_err(fail)?.header().clone();
        if !header.entry_type().is_dir() {
            held.0 += 1;
            held.1 += if header.entry_type().is_file() {
                header.size().map_err(fail)?
            } else {
                0
            };
        }
    }
    let want = archived
        .iter()
        .fold((0, 0), |(f, b), e| (f + e.files, b + e.bytes));
    if held != want {
        let _ = std::fs::remove_file(&staged);
        return Err(format!(
            "the archive holds {held:?} (files, bytes) of {want:?}; nothing was removed"
        ));
    }
    std::fs::rename(&staged, &path).map_err(fail)?;
    durable::sync_dir(&backups).map_err(|e| e.to_string())?;
    Ok(path)
}

/// Keys, sessions and the config, owner-only.
fn write_files(into: &Home, files: &[plan::File]) -> Result<(), String> {
    into.open().map_err(|e| e.to_string())?;
    durable::private_dir(&into.keys()).map_err(|e| e.to_string())?;
    for file in files {
        durable::replace(&file.path, &file.bytes, 0o600)
            .map_err(|e| format!("{}: {e}", file.path.display()))?;
    }
    Ok(())
}

fn remove(path: &Path) -> Result<(), String> {
    let removed = if path.is_dir() && !path.is_symlink() {
        std::fs::remove_dir_all(path)
    } else {
        std::fs::remove_file(path)
    };
    removed.map_err(|e| format!("{}: {e}", path.display()))
}

/// Deletes the caches of an imported home, each listed with its size before it goes.
fn delete_caches(from: &Path, into: &Home, report: &mut Report) -> Result<(), Error> {
    if !imported(into)? {
        return Err(Error::new(
            Kind::Conflict,
            "not_migrated",
            "the Go home is not imported yet",
        )
        .remedy("run `cozy migrate --from <dir>` first, and prove the new binary"));
    }
    let caches: Vec<&Entry> = report
        .entries
        .iter()
        .filter(|e| e.fate == Fate::Cache)
        .collect();
    for entry in &caches {
        println!(
            "{} {} ({} files): {}",
            if report.dry_run {
                "would delete"
            } else {
                "deleting"
            },
            entry.name,
            entry.files,
            size(entry.bytes)
        );
    }
    let _ = std::io::stdout().flush();
    if report.dry_run {
        return Ok(());
    }
    for entry in caches {
        remove(&from.join(&entry.name))
            .map_err(|e| Error::new(Kind::Internal, "delete_failed", e))?;
        report.deleted.push(entry.name.clone());
    }
    Ok(())
}

fn print(report: &Report, json: bool) {
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(report).expect("serializes")
        );
        return;
    }
    let c = &report.counts;
    let verb = if report.dry_run {
        "dry run, nothing changed"
    } else {
        "imported"
    };
    println!(
        "cozy migrate ({verb}): {} -> {}",
        report.from.display(),
        report.into.display()
    );
    println!(
        "  hubs      {} named, current {}",
        c.hubs,
        c.current_hub.as_deref().unwrap_or("tensorhub (default)")
    );
    println!(
        "  keys      {} device, {} sessions, owner {}, {} rental",
        c.device_keys,
        c.sessions,
        u8::from(c.owner_key),
        c.rental_keys
    );
    println!(
        "  machines  local {}, rentals {} ready and {} booting, {} ended from history",
        u8::from(c.local_machine),
        c.rentals_ready,
        c.rentals_booting,
        c.history_machines
    );
    println!(
        "  runs      {}: history {} (completed {}, failed {}, canceled {}), owed {} (accepted {}, sent {})",
        c.runs,
        c.history,
        c.completed,
        c.failed,
        c.canceled,
        c.unsettled_accepted + c.unsettled_sent,
        c.unsettled_accepted,
        c.unsettled_sent
    );
    println!(
        "  outputs   {} items, {} without a file",
        c.outputs, c.outputs_without_path
    );
    for entry in &report.entries {
        let fate = match entry.fate {
            Fate::Kept => "kept (user content)",
            Fate::Machine => "kept (the local machine)",
            Fate::Layout => "this layout",
            Fate::Cache => "cache, left until --delete-caches",
            Fate::Archived => "archived, then removed",
        };
        println!(
            "  {:<28} {:>9} files {:>10}  {fate}",
            entry.name,
            entry.files,
            size(entry.bytes)
        );
    }
    if let Some(archive) = &report.archive {
        println!("  archive   {}", archive.display());
    }
    for name in &report.deleted {
        println!("  deleted   {name}");
    }
    println!("  skipped   {}", report.skipped.len());
    for skip in &report.skipped {
        println!("    {}: {}", skip.what, skip.reason);
    }
}

fn size(bytes: u64) -> String {
    let units = ["B", "KB", "MB", "GB", "TB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1000.0 && unit < units.len() - 1 {
        value /= 1000.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.1} {}", units[unit])
    }
}
