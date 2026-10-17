//! `cozy run`: run a function on a machine, and the verbs over runs.
use std::{path::PathBuf, process::ExitCode};

use cozy_proto::{
    calcifer::{On, SubmitRun},
    config::Hub,
    error::{Error, Kind},
};

use crate::{Context, output};
use bind::Terms;
pub(crate) use catalog::Card;
pub(crate) use follow::{exit_code, follow, submit};
use interface::{Callable, Interface};
use target::Target;

mod bind;
pub(crate) mod board;
mod catalog;
mod describe;
mod follow;
pub(crate) mod interface;
mod list;
mod play;
mod show;
mod spec;
mod target;
mod upload;

#[derive(clap::Args, Debug)]
#[command(args_conflicts_with_subcommands = true, subcommand_negates_reqs = true)]
pub struct Args {
    #[command(subcommand)]
    verb: Option<Verb>,
    #[command(flatten)]
    run: Box<RunArgs>,
}

#[derive(clap::Args, Debug)]
struct RunArgs {
    /// `org/pkg[/fn][@release]`, `./dir[/fn]` or `./script.py`; without a function, its functions.
    #[arg(required = true, value_name = "TARGET")]
    target: Option<String>,
    /// `key=value`, `key:=json`, `model.<slot>=<ref>`, one positional value, input files.
    #[arg(value_name = "INPUT")]
    inputs: Vec<String>,
    #[command(flatten)]
    placement: Placement,
    /// Use at most N GPUs for the run and its calls.
    #[arg(long, value_name = "N", value_parser = clap::value_parser!(u32).range(1..))]
    gpus: Option<u32>,
    /// Weight precision for every model; absent, the machine's rules choose per GPU.
    #[arg(long, value_parser = ["16", "8", "4"])]
    bits: Option<String>,
    /// The whole payload from a JSON or YAML file; inline inputs override it.
    #[arg(long = "input", value_name = "FILE")]
    input_file: Option<PathBuf>,
    /// Attach a file: `field.path=FILE`, `label=FILE`, or `FILE` into the declared assets.
    #[arg(long = "asset", value_name = "[NAME=]FILE")]
    assets: Vec<String>,
    /// Apply a LoRA: `slot:component=ref[,strength]`, in order.
    #[arg(long = "lora", value_name = "SLOT:COMPONENT=REF")]
    loras: Vec<String>,
    /// Where the outputs land (default ~/.cozy/outputs).
    #[arg(long, value_name = "DIR")]
    out: Option<PathBuf>,
    /// Print the run's record at once instead of following it.
    #[arg(long)]
    detach: bool,
    /// Keep the function ready instead of running it.
    #[arg(long, value_enum, value_name = "LEVEL")]
    warm: Option<Warm>,
    /// Print the function's request contract instead of running it.
    #[arg(long)]
    describe: bool,
    /// Names one submission forever: a retry with it never runs twice.
    #[arg(long, value_name = "KEY")]
    idempotency_key: Option<String>,
}

/// Where a run goes; shared by every verb that submits one.
#[derive(clap::Args, Debug, Clone)]
pub struct Placement {
    /// `local`, a machine's name, `any`, or `new[:sku]` to rent one.
    #[arg(long = "on", value_name = "MACHINE", default_value = "local")]
    on: String,
}

impl Placement {
    pub fn on(&self) -> On {
        On::from(self.on.clone())
    }
}

#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
enum Warm {
    Installed,
    Downloaded,
    Imported,
    Host,
    Gpu,
    /// Take it out of the machine's warm set.
    Off,
}

#[derive(clap::Subcommand, Debug)]
enum Verb {
    /// Runs on every Hub, newest first.
    List {
        /// Only runs in this state.
        #[arg(long, value_parser = ["queued", "in_progress", "completed", "failed", "canceled"])]
        state: Option<String>,
        /// Only this package's runs.
        #[arg(long)]
        package: Option<String>,
        /// At most N runs (default 50; 0 shows all).
        #[arg(long, value_name = "N")]
        limit: Option<u64>,
        /// A live board, redrawn as runs change.
        #[arg(long)]
        watch: bool,
    },
    /// One run: its result, outputs, calls, logs and how it ran.
    Show {
        /// The run's number or id.
        run: String,
    },
    /// Follow a run until it ends; Ctrl-C stops watching, never the run.
    Watch {
        /// The run's number or id.
        run: String,
    },
    /// Cancel a queued or running run.
    Cancel {
        /// The run's number or id.
        run: String,
    },
    /// Print a link that plays a run's output in a browser, straight from its machine.
    Play {
        /// The run's number or id.
        run: String,
        /// The output to play: its name, or name/index for one item of a list.
        #[arg(long, default_value = "video")]
        output: String,
        /// How long the link works: anyone holding it may watch until then.
        #[arg(long, default_value = "24h")]
        expires: String,
    },
    /// Upload a run's weights output as a private checkpoint in org/model, from its machine.
    Upload {
        /// The run's number or id, and `#output` when it wrote more than one.
        #[arg(value_name = "RUN[#OUTPUT]")]
        run: String,
        /// The destination repository.
        #[arg(value_name = "ORG/MODEL")]
        model: String,
        /// Print the upload's run record at once instead of following it.
        #[arg(long)]
        detach: bool,
    },
}

pub async fn exec(ctx: &Context, args: Args) -> Result<ExitCode, Error> {
    match args.verb {
        Some(Verb::List {
            state,
            package,
            limit,
            watch,
        }) => {
            let query = list::Query {
                status: state,
                package,
                before: None,
                limit,
            };
            list::exec(ctx, query, watch).await
        }
        Some(Verb::Show { run }) => show::exec(ctx, &run).await,
        Some(Verb::Watch { run }) => {
            let run = ctx.calcifer().await?.run(&run).await?;
            Ok(exit_code(&follow(ctx, run, false).await?))
        }
        Some(Verb::Cancel { run }) => {
            let calcifer = ctx.calcifer().await?;
            let mut run = calcifer.run(&run).await?;
            if !follow::settled(&run) {
                run = calcifer.cancel(run.id.as_str()).await?;
            }
            follow::record(ctx, &run);
            Ok(ExitCode::SUCCESS)
        }
        Some(Verb::Play {
            run,
            output,
            expires,
        }) => play::exec(ctx, &run, &output, &expires).await,
        Some(Verb::Upload { run, model, detach }) => upload::exec(ctx, &run, &model, detach).await,
        None => run(ctx, *args.run).await,
    }
}

async fn run(ctx: &Context, args: RunArgs) -> Result<ExitCode, Error> {
    let cwd = std::env::current_dir()
        .map_err(|e| Error::new(Kind::Internal, "cwd", format!("the working directory: {e}")))?;
    let raw = args.target.as_deref().unwrap_or_default();
    let (package, function, release) = match Target::parse(raw, &cwd)? {
        Target::Published {
            package,
            function,
            release,
        } => (package, function, release),
        Target::Directory { .. } | Target::Script(_) => {
            return Err(crate::not_ported("run ./local-code")
                .remedy("local code runs once machines describe it (xs-043 unit T); publish it, or use the Go cozy"));
        }
    };
    let hub = ctx.hub()?;
    let (card, read) = catalog::card(ctx, &hub, &package, release.as_deref(), false).await?;
    let Some(function) = function else {
        functions(ctx, &package, &card);
        return Ok(ExitCode::SUCCESS);
    };
    if args.describe {
        let callable = callable(&package, &card.interface, &function)?;
        if ctx.json {
            output::json(&card.request(&callable.name));
        } else {
            anstream::print!(
                "{}",
                describe::contract(&format!("{package}/{}", callable.name), callable)
            );
        }
        return Ok(ExitCode::SUCCESS);
    }
    invoke(
        ctx,
        &package,
        release.as_deref(),
        &function,
        (card, read),
        args,
        cwd,
    )
    .await
}

/// `cozy run <package>/<function>[@release] <inputs…> --on …` for another verb: `model
/// quantize` runs a package's conversion this way.
pub(crate) async fn call(
    ctx: &Context,
    package: &str,
    release: Option<&str>,
    function: &str,
    inputs: Vec<String>,
    placement: &Placement,
    detach: bool,
) -> Result<ExitCode, Error> {
    let cwd = std::env::current_dir()
        .map_err(|e| Error::new(Kind::Internal, "cwd", format!("the working directory: {e}")))?;
    let card = catalog::card(ctx, &ctx.hub()?, package, release, false).await?;
    let args = RunArgs {
        target: None,
        inputs,
        placement: placement.clone(),
        gpus: None,
        bits: None,
        input_file: None,
        assets: vec![],
        loras: vec![],
        out: None,
        detach,
        warm: None,
        describe: false,
        idempotency_key: None,
    };
    invoke(ctx, package, release, function, card, args, cwd).await
}

/// A release's card: the kept one, else the Hub's, kept from then on.
pub(crate) async fn card(
    ctx: &Context,
    hub: &Hub,
    package: &str,
    release: Option<&str>,
) -> Result<Card, Error> {
    Ok(catalog::card(ctx, hub, package, release, false).await?.0)
}

/// Binds the run's inputs, submits it and follows it.
async fn invoke(
    ctx: &Context,
    package: &str,
    release: Option<&str>,
    function: &str,
    (mut card, read): (Card, bool),
    args: RunArgs,
    cwd: PathBuf,
) -> Result<ExitCode, Error> {
    let hub = ctx.hub()?;
    let terms = Terms {
        inputs: args.inputs,
        input_file: args.input_file,
        assets: args.assets,
        loras: args.loras,
        key: args.idempotency_key.unwrap_or_else(follow::mint_key),
        cwd,
        user_home: ctx.user_home.clone(),
    };
    let bits = args
        .bits
        .map_or(0, |b| b.parse().expect("clap admits 16, 8 or 4"));
    let on = args.placement.on();
    if let Some(level) = args.warm {
        if args.gpus.is_some() {
            return Err(Error::new(
                Kind::Usage,
                "warm_gpus",
                "--gpus applies to a run, not to --warm",
            ));
        }
        let callable = callable(package, &card.interface, function)?;
        let models = bind::warm_choices(callable, &terms)?;
        let call = spec::Call {
            package,
            release,
            callable,
            gpus: 0,
            bits,
        };
        let submission = call.submission(on, &hub.origin, &call.warm(level, &models)?, None, None);
        return send(ctx, submission, &terms.key, args.detach).await;
    }
    let mut bound = bind(ctx, package, &card, function, &terms)?;
    // A kept card that lacks a field the run names may be older than the release the machine
    // runs: read the newest once, and bind again.
    if release.is_none()
        && !read
        && !bound.1.ignored.is_empty()
        && let Ok((newer, _)) = catalog::card(ctx, &hub, package, None, true).await
        && newer.release != card.release
    {
        card = newer;
        bound = bind(ctx, package, &card, function, &terms)?;
    }
    let (target, bound) = bound;
    if let Some(warning) = bound.warning(&target) {
        output::warn(ctx, &warning);
    }
    let callable = callable(package, &card.interface, function)?;
    let gpus = args.gpus.unwrap_or(0);
    let call = spec::Call {
        package,
        release,
        callable,
        gpus,
        bits,
    };
    let out = args.out.map(|dir| terms.cwd.join(dir));
    let submission = call.submission(on, &hub.origin, &call.spec(&bound)?, Some(&bound), out);
    send(ctx, submission, &terms.key, args.detach).await
}

/// Submits, then follows the run (or prints it at once with `detach`).
async fn send(
    ctx: &Context,
    submission: SubmitRun,
    key: &str,
    detach: bool,
) -> Result<ExitCode, Error> {
    let submitted = submit(ctx, submission, Some(key)).await?;
    Ok(exit_code(&follow(ctx, submitted.run, detach).await?))
}

/// Binds `terms` to `function` of `card`; a person's refusal lists the arguments.
fn bind(
    ctx: &Context,
    package: &str,
    card: &Card,
    function: &str,
    terms: &Terms,
) -> Result<(String, bind::Bound), Error> {
    let callable = callable(package, &card.interface, function)?;
    let target = format!("{package}/{}", callable.name);
    let bound = bind::bind(&target, callable, terms).map_err(|mut error| {
        if !ctx.json && error.code == "request_payload_invalid" {
            error.message = format!(
                "{}\n\n{}",
                error.message,
                describe::arguments(callable).trim_end()
            );
        }
        error
    })?;
    Ok((target, bound))
}

/// The function `asked` names: exact, else the one name equal with `-` as `_`.
fn callable<'a>(
    package: &str,
    interface: &'a Interface,
    asked: &str,
) -> Result<&'a Callable, Error> {
    let fold = |name: &str| name.replace('-', "_");
    let found = interface
        .callables
        .iter()
        .find(|c| c.name == asked)
        .or_else(|| {
            let mut matches = interface
                .callables
                .iter()
                .filter(|c| fold(&c.name) == fold(asked));
            matches.next().filter(|_| matches.next().is_none())
        });
    match found {
        Some(callable) if callable.internal => Err(Error::new(
            Kind::NotFound,
            "callable_internal",
            format!("{package}/{} is an internal function", callable.name),
        )
        .remedy("call a public function of the package")),
        Some(callable) => Ok(callable),
        None => {
            if let Some((name, why)) = interface
                .unavailable
                .iter()
                .find(|(name, _)| fold(name) == fold(asked))
            {
                return Err(Error::new(
                    Kind::Unavailable,
                    "callable_unreadable",
                    format!("{package}/{name} is not readable by this cozy: {why}"),
                )
                .remedy("update cozy, or call another function of the package"));
            }
            let public = interface.public();
            let mut error = Error::new(
                Kind::NotFound,
                "callable_unknown",
                format!("{package} has no function {asked:?}"),
            )
            .remedy(format!("its functions: {}", public.join(", ")));
            for name in public.iter().take(2) {
                error = error.next(format!("cozy run {package}/{name}"));
            }
            Err(error)
        }
    }
}

/// `cozy run org/pkg`: the package's public functions.
fn functions(ctx: &Context, package: &str, card: &Card) {
    let public = card.interface.public();
    let runs_on = |name: &str| {
        let cpu = card
            .interface
            .callables
            .iter()
            .any(|c| c.name == name && c.accelerator == Some(false));
        if cpu { "CPU" } else { "GPU" }
    };
    if ctx.json {
        let rows: Vec<_> = public
            .iter()
            .map(|name| serde_json::json!({"function": name, "runs_on": runs_on(name)}))
            .collect();
        output::json(
            &serde_json::json!({"package": package, "release": card.release, "functions": rows}),
        );
        return;
    }
    let mut table = output::Table::new(ctx, ["FUNCTION", "RUNS ON"]);
    for name in &public {
        table = table.row([*name, runs_on(name)]);
    }
    table.note(format!("{package} {}", card.release)).print();
    let next: Vec<String> = public
        .iter()
        .take(2)
        .map(|name| format!("cozy run {package}/{name} --describe"))
        .collect();
    output::next(ctx, &next.iter().map(String::as_str).collect::<Vec<_>>());
}
