//! `cozy run`: run a function on a machine, and the verbs over runs.
use std::{path::PathBuf, process::ExitCode};

use protocol::{
    calcifer::{On, Run, StagedInput, SubmitRun},
    config::Hub,
    error::{Error, Kind},
};

use crate::{Context, output};
use bind::Terms;
pub(crate) use catalog::{Card, latest};
pub(crate) use follow::{exit_code, follow, submit};
use interface::{Callable, Interface};
use target::Target;

mod bind;
mod catalog;
mod describe;
mod digests;
mod follow;
pub(crate) mod interface;
mod list;
pub(crate) mod live;
pub(crate) mod local;
mod play;
mod show;
mod spec;
mod target;
pub(crate) mod tree;
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
    #[arg(long, value_name = "N", value_parser = gpu_count)]
    gpus: Option<u32>,
    /// Weight precision for every model; absent, the machine's rules choose per GPU.
    #[arg(long, value_parser = ["16", "8", "4"])]
    bits: Option<String>,
    /// Pin an attention kernel for this run: everywhere, or at one `[slot/]component`.
    #[arg(long, value_name = "[[SLOT/]COMPONENT=]KERNEL", value_parser = attention_pin, conflicts_with = "warm")]
    attention_kernel: Option<String>,
    /// The whole payload from a JSON or YAML file; inline inputs override it.
    #[arg(long = "input", value_name = "FILE")]
    input_file: Option<PathBuf>,
    /// Attach a file: `field.path=FILE`, `label=FILE`, or `FILE` into the declared assets.
    #[arg(long = "asset", value_name = "[NAME=]FILE")]
    assets: Vec<String>,
    /// A hint for one attached asset: `label-or-index=auto|low|medium|high`.
    #[arg(long = "asset-fidelity", value_name = "ASSET=FIDELITY", value_parser = fidelity)]
    fidelities: Vec<String>,
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
        /// One snapshot even on a terminal, instead of the live view.
        #[arg(long)]
        once: bool,
    },
    /// One run: its result, outputs, calls, logs and how it ran.
    Show {
        /// The run's number or id.
        run: String,
        /// One snapshot even on a terminal, instead of the live view of a run not yet settled.
        #[arg(long)]
        once: bool,
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
        /// Where it goes: org/model[@tag], the repository and the private tag it moves (main).
        #[arg(value_name = "ORG/MODEL[@TAG]")]
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
            once,
        }) => {
            let query = list::Query {
                status: state,
                package,
                before: None,
                limit,
            };
            list::exec(ctx, query, once).await
        }
        Some(Verb::Show { run, once }) => show::exec(ctx, &run, once).await,
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
        Target::Directory { path, function } => {
            return local::run(ctx, &cwd.join(path), function, args).await;
        }
        Target::Script(path) => return local::run(ctx, &cwd.join(path), None, args).await,
    };
    let hub = ctx.hub()?;
    let resolved = resolve(ctx, &hub, &package, release.as_deref()).await?;
    let Some(function) = function else {
        unnamed(
            &package,
            release.as_deref(),
            &resolved.card.interface,
            &args,
        )?;
        functions(ctx, &package, &resolved);
        return Ok(ExitCode::SUCCESS);
    };
    if args.describe {
        let card = &resolved.card;
        let callable = callable(&package, &card.interface, &function)?;
        if ctx.json {
            let mut request = card.request(&callable.name);
            request["release"] = card.release.clone().into();
            request["release_from"] = resolved.from.to_string().into();
            output::json(&request);
        } else {
            let target = format!("{package}/{}", callable.name);
            let text = describe::contract(&target, &card.interface, callable);
            let release = format!("\n  release: {} ({})\n", card.release, resolved.from);
            anstream::print!("{}", text.replacen('\n', &release, 1));
        }
        return Ok(ExitCode::SUCCESS);
    }
    let run = invoke(ctx, &package, &function, &resolved.card, args).await?;
    Ok(exit_code(&run))
}

/// Which release a run binds against and runs, and why that one.
pub(crate) struct Resolved {
    pub card: Card,
    pub from: From,
}

pub(crate) enum From {
    /// The run named it.
    Asked,
    /// The Hub's newest published release.
    Latest,
}

impl std::fmt::Display for From {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            From::Asked => f.write_str("asked"),
            From::Latest => f.write_str("the Hub's latest"),
        }
    }
}

/// Resolves a run's release once, before binding: the one asked, else the Hub's newest published
/// release, wherever the run goes (a machine holding an older one installs it first). The run is
/// pinned to it, so the machine runs exactly the release its inputs were bound against.
pub(crate) async fn resolve(
    ctx: &Context,
    hub: &Hub,
    package: &str,
    release: Option<&str>,
) -> Result<Resolved, Error> {
    let (release, from) = match release {
        Some(release) => (release.to_owned(), From::Asked),
        None => (catalog::latest(ctx, hub, package).await?, From::Latest),
    };
    let card = catalog::card(ctx, hub, package, &release).await?;
    Ok(Resolved { card, from })
}

/// `cozy run <package>/<function>[@release] <inputs…> --on …` for another verb: `model
/// quantize` runs a package's conversion this way. The run as it settled, or as sent with
/// `detach`; its record is printed either way.
pub(crate) async fn call(
    ctx: &Context,
    package: &str,
    release: Option<&str>,
    function: &str,
    inputs: Vec<String>,
    placement: &Placement,
    detach: bool,
) -> Result<Run, Error> {
    let hub = ctx.hub()?;
    let resolved = resolve(ctx, &hub, package, release).await?;
    let args = RunArgs {
        target: None,
        inputs,
        placement: placement.clone(),
        gpus: None,
        bits: None,
        attention_kernel: None,
        input_file: None,
        assets: vec![],
        fidelities: vec![],
        loras: vec![],
        out: None,
        detach,
        warm: None,
        describe: false,
        idempotency_key: None,
    };
    invoke(ctx, package, function, &resolved.card, args).await
}

/// Binds the run's inputs to `function` of `card`, submits it pinned to `card`'s release, and
/// follows it.
async fn invoke(
    ctx: &Context,
    package: &str,
    function: &str,
    card: &Card,
    args: RunArgs,
) -> Result<Run, Error> {
    let on = args.placement.on();
    let source = spec::Source::Release(Some(&card.release));
    let release = Some(card.release.as_str());
    let code = Code {
        package,
        interface: &card.interface,
        release,
        source,
        files: &[],
    };
    call_code(ctx, code, function, on, args).await
}

/// What a run calls: a published release, or local code its machine installed.
struct Code<'a> {
    package: &'a str,
    interface: &'a Interface,
    /// The release its inputs bind against; none for local code.
    release: Option<&'a str>,
    source: spec::Source<'a>,
    /// Files sent with the call (local code's).
    files: &'a [StagedInput],
}

/// Binds the run's inputs to `function` of `code` (a warm run, its model choices only), submits
/// it `on` its machine, and follows it.
async fn call_code(
    ctx: &Context,
    code: Code<'_>,
    function: &str,
    on: On,
    args: RunArgs,
) -> Result<Run, Error> {
    let hub = ctx.hub()?;
    let terms = terms(ctx, &args)?;
    let callable = callable(code.package, code.interface, function)?;
    let call = spec::Call {
        package: code.package,
        source: code.source,
        callable,
        gpus: args.gpus.unwrap_or(0),
        bits: bits(&args),
        attention_kernel: args.attention_kernel.as_deref().unwrap_or_default(),
    };
    let (spec, bound, out) = match args.warm {
        Some(_) if args.gpus.is_some() => {
            let why = "--gpus applies to a run, not to --warm";
            return Err(Error::new(Kind::Usage, "warm_gpus", why));
        }
        Some(level) => (
            call.warm(level, &bind::warm_choices(callable, &terms)?)?,
            None,
            None,
        ),
        None => {
            let target = format!("{}/{}", code.package, callable.name);
            let bound = bind::bind(&target, code.release, callable, &terms)?;
            if let Some(warning) = bound.warning(&target) {
                output::warn(&warning);
            }
            gpuless(ctx, &on, code.interface, callable, &target, call.bits).await;
            let out = args.out.as_ref().map(|dir| terms.cwd.join(dir));
            (call.spec(&bound)?, Some(bound), out)
        }
    };
    let mut submission = call.submission(on, &hub.origin, &spec, bound.as_ref(), out);
    submission.inputs.extend(code.files.iter().cloned());
    send(ctx, submission, &terms.key, args.detach).await
}

/// The terms a run was given; an omitted key is minted, as the seed derives from it.
fn terms(ctx: &Context, args: &RunArgs) -> Result<Terms, Error> {
    let cwd = std::env::current_dir()
        .map_err(|e| Error::new(Kind::Internal, "cwd", format!("the working directory: {e}")))?;
    Ok(Terms {
        inputs: args.inputs.clone(),
        input_file: args.input_file.clone(),
        assets: args.assets.clone(),
        fidelities: args.fidelities.clone(),
        loras: args.loras.clone(),
        key: args
            .idempotency_key
            .clone()
            .unwrap_or_else(follow::mint_key),
        cwd,
        user_home: ctx.user_home.clone(),
        digests: Some(ctx.home.root().join("cache").join("digests")),
        manifests: Some(ctx.home.root().join("cache").join("trees")),
    })
}

/// `--gpus`: 1 or more. Where a run uses the CPU is its function's and its machine's to say.
fn gpu_count(text: &str) -> Result<u32, String> {
    match text.parse::<u32>() {
        Ok(0) => Err(
            "a function that needs no GPU runs on the CPU by itself, as does every run on a \
             machine without one: `--on new:cpu` rents a machine with none"
                .into(),
        ),
        Ok(n) => Ok(n),
        Err(_) => Err(format!("{text:?} is not a number of GPUs")),
    }
}

/// `--bits`; 0 lets the machine's rules choose.
fn bits(args: &RunArgs) -> u32 {
    args.bits
        .as_deref()
        .map_or(0, |b| b.parse().expect("clap admits 16, 8 or 4"))
}

/// `--attention-kernel`, in the grammar the Runtime reads (`KERNEL` or `[SLOT/]COMPONENT=KERNEL`,
/// no whitespace); the Runtime judges the names.
fn attention_pin(pin: &str) -> Result<String, String> {
    let formed = match pin.split_once('=') {
        None => !pin.is_empty(),
        Some((scope, kernel)) => !scope.is_empty() && !kernel.is_empty() && !kernel.contains('='),
    };
    match formed && !pin.contains(char::is_whitespace) {
        true => Ok(pin.to_owned()),
        false => Err("use KERNEL or [SLOT/]COMPONENT=KERNEL, with no whitespace".into()),
    }
}

/// `--asset-fidelity`'s form; which asset it names is checked as the run binds.
fn fidelity(term: &str) -> Result<String, String> {
    match term.rsplit_once('=') {
        Some((asset, "auto" | "low" | "medium" | "high")) if !asset.is_empty() => Ok(term.into()),
        _ => Err("use LABEL-OR-INDEX=auto|low|medium|high".into()),
    }
}

/// Submits, then follows the run (or prints it at once with `detach`).
async fn send(ctx: &Context, submission: SubmitRun, key: &str, detach: bool) -> Result<Run, Error> {
    let submitted = submit(ctx, submission, Some(key)).await?;
    follow(ctx, submitted.run, detach).await
}

/// Warns that a run sent to a ready machine with no supported GPU runs on its CPU (a GPU
/// function slowly) and that its models take their 16-bit lanes whatever `--bits` asked (a CPU
/// runs no 8- or 4-bit format), and how to rent one instead.
pub(crate) async fn gpuless(
    ctx: &Context,
    on: &On,
    interface: &Interface,
    callable: &Callable,
    target: &str,
    bits: u32,
) {
    let On::Machine(name) = on else { return };
    let (slow, narrow) = (interface.gpu(callable), matches!(bits, 4 | 8));
    if !slow && !narrow {
        return;
    }
    let Ok(calcifer) = ctx.calcifer().await else {
        return;
    };
    let Ok(listed) = calcifer.machines().await else {
        return;
    };
    let none = listed.machines.iter().any(|m| {
        (m.name == *name || m.alias.as_deref() == Some(name))
            && (m.status.as_ref()).is_some_and(|s| s.phase == "ready" && s.gpus.is_empty())
    });
    if none {
        let said: Vec<_> = [
            slow.then(|| format!("{target} runs on its CPU, slowly")),
            narrow
                .then(|| format!("its models take their 16-bit lanes (--bits {bits} needs a GPU)")),
        ]
        .into_iter()
        .flatten()
        .collect();
        output::warn(&format!(
            "{name} has no supported GPU, so {}; `--on new` rents one: cozy run {target} … --on new",
            said.join(", and ")
        ));
    }
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

/// Refuses inputs given to a package instead of one of its functions, naming the command that
/// runs each of its first two functions with them.
fn unnamed(
    package: &str,
    release: Option<&str>,
    interface: &Interface,
    args: &RunArgs,
) -> Result<(), Error> {
    if args.inputs.is_empty() && args.input_file.is_none() {
        return Ok(());
    }
    let inputs: String = args
        .inputs
        .iter()
        .map(|i| format!(" {}", quoted(i)))
        .collect();
    let release = release.map(|r| format!("@{r}")).unwrap_or_default();
    let public = interface.public();
    let mut error = Error::new(
        Kind::Usage,
        "function_unnamed",
        format!("name the function of {package} to run"),
    )
    .remedy(format!("its functions: {}", public.join(", ")));
    for name in public.iter().take(2) {
        error = error.next(format!("cozy run {package}/{name}{release}{inputs}"));
    }
    Err(error)
}

/// `input` as a shell word: a value with spaces or quotes is single-quoted after its `key=`.
fn quoted(input: &str) -> String {
    let plain = |c: char| c.is_ascii_alphanumeric() || "_-./:=@,+%".contains(c);
    if input.chars().all(plain) {
        return input.to_owned();
    }
    let (key, value) = match input.split_once('=') {
        Some((key, value)) if key.chars().all(plain) => (format!("{key}="), value),
        _ => (String::new(), input),
    };
    format!("{key}'{}'", value.replace('\'', r"'\''"))
}

/// `cozy run org/pkg`: the package's public functions.
fn functions(ctx: &Context, package: &str, resolved: &Resolved) {
    let from = resolved.from.to_string();
    let card = &resolved.card;
    functions_of(ctx, package, (&card.release, card.summary()), Some(&from), &card.interface);
}

/// A package's public functions, and where each runs.
fn functions_of(
    ctx: &Context,
    package: &str,
    (release, summary): (&str, &str),
    from: Option<&str>,
    interface: &Interface,
) {
    let public = interface.public();
    let callable = |name: &str| interface.callables.iter().find(|c| c.name == name);
    let runs_on = |name: &str| match callable(name) {
        Some(c) if interface.orchestrates(c) => "GPU, through its calls",
        Some(c) if c.accelerator == Some(false) => "CPU",
        _ => "GPU",
    };
    let conversion = |name: &&str| callable(name).is_some_and(Callable::is_conversion);
    if ctx.json {
        let rows: Vec<_> = (public.iter())
            .map(|name| serde_json::json!({"function": name, "runs_on": runs_on(name), "conversion": conversion(name)}))
            .collect();
        output::json(&serde_json::json!({
            "package": package, "release": release, "release_from": from, "summary": summary,
            "functions": rows,
        }));
        return;
    }
    // Conversions write model lanes: model authors' work, named below the functions a run uses.
    let (conversions, public): (Vec<&str>, Vec<&str>) = public.iter().copied().partition(conversion);
    let public = match public.is_empty() {
        true => conversions.clone(),
        false => public,
    };
    let mut table = output::Table::new(ctx, ["FUNCTION", "RUNS ON"]);
    for name in &public {
        table = table.row([*name, runs_on(name)]);
    }
    if public != conversions && !conversions.is_empty() {
        table = table.note(format!("for model authors: {} (conversions)", conversions.join(", ")));
    }
    let from = from.map(|from| format!(" ({from})")).unwrap_or_default();
    let summary = match summary.is_empty() {
        true => String::new(),
        false => format!(": {summary}"),
    };
    table.note(format!("{package} {release}{from}{summary}")).print();
    let next: Vec<String> = public
        .iter()
        .take(2)
        .map(|name| format!("cozy run {package}/{name} --describe"))
        .collect();
    output::next(ctx, &next.iter().map(String::as_str).collect::<Vec<_>>());
}
