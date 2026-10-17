//! Rendering every verb shares. With `--json` stdout carries one JSON document (NDJSON while
//! following); otherwise tables and `key  value` records for a person. Warnings and prompts go to
//! stderr. Nothing is truncated when stdout is not a terminal; colors go through anstream, which
//! drops them for pipes and `NO_COLOR`.
use std::{
    fmt::Display,
    io::{IsTerminal, Write as _},
    path::Path,
    process::ExitCode,
};

use anstyle::{AnsiColor, Style};
use comfy_table::{ContentArrangement, presets};
use cozy_proto::error::{Envelope, Error};
use serde::Serialize;

use crate::Context;

const KEY: Style = Style::new().bold();
const DIM: Style = Style::new().dimmed();
const WARN: Style = AnsiColor::Yellow.on_default().bold();
const FAIL: Style = AnsiColor::Red.on_default().bold();

/// What the terminal can show, read once at the entry.
#[derive(Clone, Copy, Debug)]
pub struct Term {
    /// stdout is a terminal: tables fit its width and paths become links.
    pub stdout: bool,
    /// stderr is a terminal: live progress may redraw there.
    pub stderr: bool,
    hyperlinks: bool,
}

impl Term {
    pub fn detect() -> Self {
        let stdout = std::io::stdout().is_terminal();
        Self {
            stdout,
            stderr: std::io::stderr().is_terminal(),
            hyperlinks: stdout && supports_hyperlinks::on(supports_hyperlinks::Stream::Stdout),
        }
    }
}

/// Prints one JSON document on stdout.
pub fn json<T: Serialize + ?Sized>(document: &T) {
    let text = serde_json::to_string_pretty(document).expect("a document serializes");
    anstream::println!("{text}");
}

/// Prints one NDJSON line on stdout, flushed: a follower's reader sees it at once.
pub fn ndjson<T: Serialize + ?Sized>(event: &T) {
    let mut out = anstream::stdout().lock();
    let line = serde_json::to_string(event).expect("an event serializes");
    let _ = writeln!(out, "{line}").and_then(|()| out.flush());
}

/// Prints a refusal on stdout (the envelope with `--json`) and answers its exit code.
pub fn refusal(ctx: &Context, error: &Error) -> ExitCode {
    refuse(ctx.json, error)
}

pub(crate) fn refuse(json: bool, error: &Error) -> ExitCode {
    if json {
        self::json(&Envelope {
            error: error.clone(),
        });
    } else {
        anstream::println!("{FAIL}error{FAIL:#}: {} ({})", error.message, error.code);
        if let Some(remedy) = &error.remedy {
            anstream::println!("  {remedy}");
        }
        for next in &error.next {
            anstream::println!("  {DIM}next:{DIM:#} {next}");
        }
    }
    ExitCode::from(error.exit_code() as u8)
}

/// What the command is doing, for a person on stderr; silent with `--json`.
pub fn say(ctx: &Context, message: &str) {
    if !ctx.json {
        anstream::eprintln!("{message}");
    }
}

/// A warning on stderr: the command goes on.
pub fn warn(_ctx: &Context, message: &str) {
    anstream::eprintln!("{WARN}warning{WARN:#}: {message}");
}

/// At most two commands to try next, for a person; a JSON document carries its own.
pub fn next(ctx: &Context, commands: &[&str]) {
    if ctx.json {
        return;
    }
    for command in commands.iter().take(2) {
        anstream::println!("{DIM}next:{DIM:#} {command}");
    }
}

/// A local path, as an OSC 8 link on a terminal that opens them; the bare path anywhere else.
pub fn link(ctx: &Context, path: &Path) -> String {
    let shown = path.display().to_string();
    if !ctx.term.hyperlinks || !path.is_absolute() {
        return shown;
    }
    match url::Url::from_file_path(path) {
        Ok(url) => format!("\x1b]8;;{url}\x1b\\{shown}\x1b]8;;\x1b\\"),
        Err(()) => shown,
    }
}

/// `key  value` lines, keys aligned, then notes.
#[derive(Default)]
pub struct Record {
    fields: Vec<(String, String)>,
    notes: Vec<String>,
}

impl Record {
    pub fn new() -> Self {
        Self::default()
    }

    /// One line per top-level key of `value`'s JSON: the same view `--json` prints whole.
    pub fn of<T: Serialize + ?Sized>(value: &T) -> Self {
        let mut record = Self::new();
        if let Ok(serde_json::Value::Object(map)) = serde_json::to_value(value) {
            for (key, value) in map {
                record = record.field(&key, plain(&value));
            }
        }
        record
    }

    pub fn field(mut self, key: &str, value: impl Display) -> Self {
        self.fields.push((key.to_owned(), value.to_string()));
        self
    }

    pub fn note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }

    pub fn print(self, _ctx: &Context) {
        for line in self.styled(KEY, DIM) {
            anstream::println!("{line}");
        }
    }

    /// The lines `print` writes, unstyled: keys aligned, then the notes.
    pub fn lines(&self) -> Vec<String> {
        self.styled(Style::new(), Style::new())
    }

    fn styled(&self, key: Style, note: Style) -> Vec<String> {
        let width = self.fields.iter().map(|(k, _)| k.len()).max().unwrap_or(0);
        let fields = (self.fields.iter()).map(|(k, v)| format!("{key}{k:<width$}{key:#}  {v}"));
        let notes = self.notes.iter().map(|n| format!("{note}{n}{note:#}"));
        fields.chain(notes).collect()
    }
}

/// A JSON value as a person reads it: strings bare, lists comma-joined, absent as `-`.
fn plain(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Null => "-".into(),
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(items) => items.iter().map(plain).collect::<Vec<_>>().join(", "),
        other => other.to_string(),
    }
}

/// A byte count in binary units: `512 B`, `6.9 GiB`.
pub fn bytes(n: u64) -> String {
    const UNITS: [&str; 6] = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"];
    let mut value = n as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    match unit {
        0 => format!("{n} B"),
        _ if value < 10.0 => format!("{value:.1} {}", UNITS[unit]),
        _ => format!("{value:.0} {}", UNITS[unit]),
    }
}

/// How long ago a Unix-milliseconds instant was, in its two largest units: `42s`, `3m12s`,
/// `5h3m`, `2d4h`.
pub fn age(at_ms: i64) -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as i64);
    let s = (now - at_ms).max(0) / 1000;
    match s {
        0..60 => format!("{s}s"),
        60..3600 => format!("{}m{}s", s / 60, s % 60),
        3600..86400 => format!("{}h{}m", s / 3600, s % 3600 / 60),
        _ => format!("{}d{}h", s / 86400, s % 86400 / 3600),
    }
}

/// Columns without borders. On a terminal rows wrap to its width; elsewhere each row is one
/// whole line, for grep and awk.
pub struct Table {
    table: comfy_table::Table,
    notes: Vec<String>,
}

impl Table {
    pub fn new<S: Display>(ctx: &Context, header: impl IntoIterator<Item = S>) -> Self {
        let mut table = comfy_table::Table::new();
        table.load_style(presets::NOTHING);
        if ctx.term.stdout {
            table.set_content_arrangement(ContentArrangement::Dynamic);
        } else {
            table.force_no_tty();
        }
        table.set_header(header.into_iter().map(|h| {
            let cell = comfy_table::Cell::new(h);
            if ctx.term.stdout {
                cell.add_attribute(comfy_table::Attribute::Bold)
            } else {
                cell
            }
        }));
        Self {
            table,
            notes: vec![],
        }
    }

    pub fn row<S: Display>(mut self, cells: impl IntoIterator<Item = S>) -> Self {
        self.table.add_row(cells.into_iter().map(|c| c.to_string()));
        self
    }

    /// A line under the table, such as "12 more: cozy run list --before 40".
    pub fn note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }

    pub fn print(mut self) {
        for column in self.table.column_iter_mut() {
            column.set_padding((0, 2));
        }
        anstream::println!("{}", self.table.trim_fmt());
        for note in &self.notes {
            anstream::println!("{DIM}{note}{DIM:#}");
        }
    }
}
