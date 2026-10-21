//! The `cozy` command line. One module per verb group; each owns its clap arguments and handler.
use std::process::ExitCode;

use clap::{CommandFactory, Parser, Subcommand};
use clap_complete::{ArgValueCompleter, CompleteEnv};
use protocol::{
    config::{self, Flags},
    error::{Error, Kind},
};

mod account;
pub mod calcifer_client;
mod completion;
pub mod context;
mod credits;
mod daemon;
mod doctor;
mod hub;
mod machine;
mod model;
pub mod output;
mod package;
mod run;
pub mod tools;
pub mod tui;
mod uninstall;
mod update;

pub use context::Context;

#[derive(Parser, Debug)]
#[command(
    name = "cozy",
    version,
    about = "Run models on your machines and rented GPUs"
)]
struct Cli {
    /// Print JSON (NDJSON when following) instead of tables.
    #[arg(long, global = true)]
    json: bool,
    /// The Hub to use instead of the current one (`cozy hub use`).
    #[arg(long, global = true, value_name = "HUB", add = ArgValueCompleter::new(completion::hubs))]
    tensorhub: Option<String>,
    /// Agree without being asked: to a rental's quote, an uninstall, a deletion.
    #[arg(long, short = 'y', global = true)]
    yes: bool,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Run a function: `org/pkg/fn`, `./dir[/fn]` or `./script.py`.
    Run(run::Args),
    /// Local, rented and added machines.
    Machine(machine::Args),
    /// Find, install and publish packages.
    Package(package::Args),
    /// Find, download, upload and publish models.
    Model(model::Args),
    /// Sign in to the Hub with this computer's device key.
    Login(account::Login),
    /// Sign out of the Hub.
    Logout(account::Logout),
    /// Show the signed-in account.
    Whoami(account::Whoami),
    /// The Hubs this computer knows, and the current one.
    Hub(hub::Args),
    /// Rental credit: balance, history, buy.
    Credits(credits::Args),
    /// Start Calcifer.
    Up(daemon::Up),
    /// Stop Calcifer. Runs on machines continue.
    Down(daemon::Down),
    /// Calcifer, machines and runs at a glance.
    Status(daemon::Status),
    /// Calcifer's or a machine's log.
    Logs(daemon::Logs),
    /// Replace this binary with the newest release.
    Update(update::Args),
    /// Remove cozy from this computer: its services, ~/.cozy and its PATH lines.
    Uninstall(uninstall::Args),
    /// Print a shell completion script.
    Completion(completion::Args),
    /// Check a machine: driver, GPUs, toolchain, kernels.
    Doctor(doctor::Args),
    /// The tools cozy uses on this computer, and installing the ones it fetches itself.
    Tools(tools::Args),
}

/// Parses the command line, loads the config once, runs the verb and renders its refusal. Exit
/// codes: 0 success, 1 a refusal or a run that did not succeed, 2 a usage error.
pub fn main() -> ExitCode {
    CompleteEnv::with_factory(Cli::command)
        .var(completion::VAR)
        .complete();
    let cli = match Cli::try_parse() {
        Ok(cli) => cli,
        Err(error) => return usage(error),
    };
    let loaded = match config::load(Flags {
        tensorhub: cli.tensorhub,
    }) {
        Ok(loaded) => loaded,
        Err(error) => return output::refuse(cli.json, &error),
    };
    let warnings = loaded.warnings.clone();
    let ctx = Context::new(cli.json, cli.yes, loaded, output::Term::detect());
    for warning in &warnings {
        output::warn(warning);
    }
    let runtime = match tokio::runtime::Runtime::new() {
        Ok(runtime) => runtime,
        Err(error) => {
            return output::refusal(
                &ctx,
                &Error::new(Kind::Internal, "runtime", error.to_string()),
            );
        }
    };
    match runtime.block_on(dispatch(&ctx, cli.command)) {
        Ok(code) => code,
        Err(error) => output::refusal(&ctx, &error),
    }
}

/// clap's own rendering for a person; with `--json` the typed envelope, exit 2.
fn usage(error: clap::Error) -> ExitCode {
    use clap::error::ErrorKind;
    let json = std::env::args_os().any(|arg| arg == "--json");
    if !json
        || matches!(
            error.kind(),
            ErrorKind::DisplayHelp
                | ErrorKind::DisplayVersion
                | ErrorKind::DisplayHelpOnMissingArgumentOrSubcommand
        )
    {
        error.exit();
    }
    let message = error.render().to_string();
    let message = message.lines().next().unwrap_or_default();
    let message = message.strip_prefix("error: ").unwrap_or(message);
    output::refuse(true, &Error::new(Kind::Usage, "usage", message))
}

async fn dispatch(ctx: &Context, command: Command) -> Result<ExitCode, Error> {
    match command {
        Command::Run(args) => run::exec(ctx, args).await,
        Command::Machine(args) => machine::exec(ctx, args).await,
        Command::Package(args) => package::exec(ctx, args).await,
        Command::Model(args) => model::exec(ctx, args).await,
        Command::Login(args) => account::login(ctx, args).await,
        Command::Logout(args) => account::logout(ctx, args).await,
        Command::Whoami(args) => account::whoami(ctx, args).await,
        Command::Hub(args) => hub::exec(ctx, args).await,
        Command::Credits(args) => credits::exec(ctx, args).await,
        Command::Up(args) => daemon::up(ctx, args).await,
        Command::Down(args) => daemon::down(ctx, args).await,
        Command::Status(args) => daemon::status(ctx, args).await,
        Command::Logs(args) => daemon::logs(ctx, args).await,
        Command::Update(args) => update::exec(ctx, args).await,
        Command::Uninstall(args) => uninstall::exec(ctx, args).await,
        Command::Completion(args) => completion::exec(ctx, args).await,
        Command::Doctor(args) => doctor::exec(ctx, args).await,
        Command::Tools(args) => tools::exec(ctx, args).await,
    }
}

#[cfg(test)]
mod tests {
    use clap::CommandFactory as _;

    #[test]
    fn the_grammar_is_consistent() {
        super::Cli::command().debug_assert();
    }
}
