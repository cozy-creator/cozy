//! `cozy hub`: named Hubs like kubectl contexts. `use` picks where new work goes, `--tensorhub`
//! picks it for one command, and existing runs and machines keep their own Hub.
use std::{io, process::ExitCode};

use clap::Subcommand;
use clap_complete::ArgValueCompleter;
use cozy_proto::{
    config::{self, Config, DEFAULT_HUB, Hub},
    error::{Error, Kind},
};
use serde::Serialize;
use toml_edit::{DocumentMut, Item, Table, value};

use crate::{Context, completion::hubs, output};

#[derive(clap::Args, Debug)]
pub struct Args {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// List the Hubs and which one is current (the default).
    List,
    /// Make a Hub current for later commands.
    Use {
        /// A Hub name or URL.
        #[arg(add = ArgValueCompleter::new(hubs))]
        hub: String,
    },
    /// Name a Hub's URL.
    Add {
        /// Letters, digits, '.', '_' or '-', in any case.
        name: String,
        /// Its base URL, such as http://127.0.0.1:8819.
        url: String,
    },
    /// Forget a Hub name.
    Remove {
        #[arg(add = ArgValueCompleter::new(hubs))]
        name: String,
    },
}

pub async fn exec(ctx: &Context, args: Args) -> Result<ExitCode, Error> {
    match args.command.unwrap_or(Command::List) {
        Command::List => list(ctx),
        Command::Use { hub } => select(ctx, &hub),
        Command::Add { name, url } => add(ctx, &name, &url),
        Command::Remove { name } => remove(ctx, &name),
    }
}

#[derive(Serialize)]
struct Row {
    #[serde(flatten)]
    hub: Hub,
    current: bool,
    logged_in: bool,
}

fn list(ctx: &Context) -> Result<ExitCode, Error> {
    let current = ctx.config.current().ok();
    let mut hubs = ctx.config.hubs();
    // A current Hub named only by URL is listed too.
    if let Some(hub) = current.as_ref().filter(|h| !hubs.contains(h)) {
        hubs.push(hub.clone());
    }
    if let Some(narrowed) = ctx.narrowed()? {
        hubs = vec![narrowed];
    }
    let rows: Vec<Row> = hubs
        .into_iter()
        .map(|hub| Row {
            current: current.as_ref() == Some(&hub),
            logged_in: ctx.signed_in(&hub),
            hub,
        })
        .collect();
    if ctx.json {
        output::json(&serde_json::json!({ "hubs": rows }));
        return Ok(ExitCode::SUCCESS);
    }
    let mut table = output::Table::new(ctx, ["CURRENT", "NAME", "URL", "LOGIN"]);
    for row in &rows {
        table = table.row([
            if row.current { "*" } else { "" },
            row.hub.name.as_deref().unwrap_or(""),
            row.hub.origin.as_str(),
            if row.logged_in { "yes" } else { "no" },
        ]);
    }
    table.print();
    if rows.iter().any(|r| r.current && !r.logged_in) {
        output::next(ctx, &["cozy login <email>"]);
    }
    Ok(ExitCode::SUCCESS)
}

fn select(ctx: &Context, text: &str) -> Result<ExitCode, Error> {
    let hub = ctx.config.resolve(text)?;
    let spelled = hub.name.clone().unwrap_or_else(|| hub.origin.to_string());
    edit(ctx, |doc| {
        doc["hub"] = value(spelled);
        Ok(())
    })?;
    let logged_in = ctx.signed_in(&hub);
    if ctx.json {
        output::json(&Row {
            hub,
            current: true,
            logged_in,
        });
    } else {
        output::Record::new()
            .field("hub", hub.label())
            .field("url", &hub.origin)
            .note("Existing runs and machines keep their own Hub.")
            .print(ctx);
        if !logged_in {
            output::next(ctx, &["cozy login <email>"]);
        }
    }
    Ok(ExitCode::SUCCESS)
}

fn add(ctx: &Context, name: &str, url: &str) -> Result<ExitCode, Error> {
    let name = name.to_ascii_lowercase();
    if !config::valid_name(&name) {
        return Err(Error::new(
            Kind::Usage,
            "hub_name_invalid",
            format!("{name:?} is not letters, digits, '.', '_' or '-'"),
        ));
    }
    let origin = config::Origin::parse(url)?;
    edit(ctx, |doc| {
        let hubs = hubs_table(doc);
        // The same name in another case is the same Hub.
        let others: Vec<String> = hubs
            .iter()
            .map(|(key, _)| key.to_owned())
            .filter(|key| key.eq_ignore_ascii_case(&name) && *key != name)
            .collect();
        for key in others {
            hubs.remove(&key);
        }
        hubs[&name] = value(origin.as_str());
        Ok(())
    })?;
    let hub = Hub {
        name: Some(name.clone()),
        origin,
    };
    if ctx.json {
        output::json(&hub);
    } else {
        output::Record::new()
            .field("hub", &name)
            .field("url", &hub.origin)
            .print(ctx);
        output::next(
            ctx,
            &[
                &format!("cozy hub use {name}"),
                &format!("cozy login <email> --tensorhub {name}"),
            ],
        );
    }
    Ok(ExitCode::SUCCESS)
}

fn remove(ctx: &Context, name: &str) -> Result<ExitCode, Error> {
    let name = name.to_ascii_lowercase();
    let current = ctx.config.hub.as_deref().unwrap_or(DEFAULT_HUB);
    if current.eq_ignore_ascii_case(&name) {
        return Err(Error::new(
            Kind::Conflict,
            "hub_current",
            format!("{name} is the current Hub"),
        )
        .remedy("switch first with `cozy hub use <other>`")
        .next("cozy hub list"));
    }
    let mut found = false;
    edit(ctx, |doc| {
        let hubs = hubs_table(doc);
        let keys: Vec<String> = hubs
            .iter()
            .map(|(key, _)| key.to_owned())
            .filter(|key| key.eq_ignore_ascii_case(&name))
            .collect();
        for key in &keys {
            hubs.remove(key);
        }
        found = !keys.is_empty();
        if hubs.is_empty() {
            doc.remove("hubs");
        }
        Ok(())
    })?;
    if !found {
        let message = if name == DEFAULT_HUB {
            format!("{DEFAULT_HUB} is built in")
        } else {
            format!("no Hub is named {name:?}")
        };
        return Err(Error::new(Kind::NotFound, "hub_not_found", message).next("cozy hub list"));
    }
    if ctx.json {
        output::json(&serde_json::json!({ "name": name, "removed": true }));
    } else {
        output::Record::new().field("removed", &name).print(ctx);
    }
    Ok(ExitCode::SUCCESS)
}

fn hubs_table(doc: &mut DocumentMut) -> &mut Table {
    if !doc.get("hubs").is_some_and(Item::is_table) {
        doc["hubs"] = Item::Table(Table::new());
    }
    doc["hubs"].as_table_mut().expect("hubs is a table")
}

/// Applies one edit to config.toml, keeping everything else as written. A file the load would
/// refuse is never written; the new one replaces the old whole, at 0600.
fn edit(
    ctx: &Context,
    change: impl FnOnce(&mut DocumentMut) -> Result<(), Error>,
) -> Result<(), Error> {
    let path = ctx.home.config();
    let shown = path.display().to_string();
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(e) if e.kind() == io::ErrorKind::NotFound => String::new(),
        Err(e) => return Err(internal(&shown, e)),
    };
    let mut doc: DocumentMut = text.parse().map_err(|e| {
        Error::new(Kind::Usage, "config_invalid", format!("{shown}: {e}"))
            .remedy(format!("correct {shown}"))
    })?;
    change(&mut doc)?;
    let text = doc.to_string();
    Config::parse(&text, &shown)?;
    ctx.home.open().map_err(|e| internal(&shown, e))?;
    cozy_platform::durable::replace(&path, text.as_bytes(), 0o600).map_err(|e| internal(&shown, e))
}

fn internal(path: &str, error: impl std::fmt::Display) -> Error {
    Error::new(
        Kind::Internal,
        "config_unwritable",
        format!("{path}: {error}"),
    )
}
