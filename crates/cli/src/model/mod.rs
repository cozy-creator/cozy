//! `cozy model`: the Hub's model catalog, and models on machines.
use std::process::ExitCode;

use clap::{ArgGroup, Subcommand};
use protocol::error::Error;

use crate::Context;

mod catalog;
mod from_run;
mod machine;
#[cfg(feature = "migrate")]
mod migrate_headers;
mod quantize;
pub(crate) mod refs;
mod releases;

#[derive(clap::Args, Debug)]
pub struct Args {
    #[command(subcommand)]
    verb: Verb,
}

#[derive(Subcommand, Debug)]
enum Verb {
    /// Search the Hub's models: one row per model with its newest available release.
    Search(Search),
    /// A model's releases, their lanes, sizes and commits.
    Info(Info),
    /// Set or clear a model's discovery family.
    Family(Family),
    /// The models machines hold.
    List(List),
    /// Make a model present in a machine's store. Follows it to its end; Ctrl-C stops following, never the work.
    Download(Download),
    /// Make a model on a machine, or take a run's weights output (run:N), and upload it as a private
    /// checkpoint. Follows it to its end; Ctrl-C stops following, never the work.
    Upload(Upload),
    /// Write a model's lanes (--format int8,fp8,…), or any lane, by the recipe a named package
    /// declares. Follows it to its end; Ctrl-C stops following, never the work.
    Quantize(Quantize),
    /// Release a commit: a public tag named by a version, created, moved, or a private one made public.
    Publish(Publish),
    /// Yank a release: only its exact version resolves it; --undo takes it back.
    Yank(Yank),
    /// Delete one tag, or the whole model.
    Delete(Delete),
    /// Dev builds only: move models from before D10 onto it, header-only.
    #[cfg(feature = "migrate")]
    #[command(hide = true)]
    MigrateHeaders(migrate_headers::Args),
}

#[derive(clap::Args, Debug)]
pub(crate) struct Search {
    /// Search text, or an exact org/name.
    query: Vec<String>,
    /// Only models of this discovery family.
    #[arg(long)]
    family: Option<String>,
    /// At most this many models (1–100).
    #[arg(long, default_value_t = 20, value_parser = clap::value_parser!(u32).range(1..=100))]
    limit: u32,
}

#[derive(clap::Args, Debug)]
pub(crate) struct Info {
    /// org/name, or org/name@release for one release.
    model: String,
}

#[derive(clap::Args, Debug)]
#[command(group(ArgGroup::new("change").required(true).args(["family", "clear"])))]
pub(crate) struct Family {
    /// org/name.
    model: String,
    /// The family to set.
    family: Option<String>,
    /// Clear the family.
    #[arg(long)]
    clear: bool,
}

#[derive(clap::Args, Debug)]
pub(crate) struct List {
    /// Only this machine; default: every machine.
    #[arg(long, value_name = "MACHINE")]
    on: Option<String>,
}

#[derive(clap::Args, Debug)]
pub(crate) struct Download {
    /// precision lane: 16, 8 or 4 bits. Absent, 8 is preferred, in the encoding the machine's GPUs
    /// run best, else whatever works.
    #[arg(long, value_parser = ["16", "8", "4"], conflicts_with = "variant")]
    lane: Option<String>,
    /// This variant exactly, by its path in the release (`cozy model info` lists them).
    #[arg(long, value_name = "PATH")]
    variant: Option<String>,
    /// org/name[@release], org/name#sha256:…, hf://…, civitai://… or https://….
    model: String,
    /// Keep it on the machine under this name.
    #[arg(value_name = "local/ALIAS")]
    alias: Option<String>,
    #[command(flatten)]
    placement: crate::run::Placement,
}

#[derive(clap::Args, Debug)]
pub(crate) struct Upload {
    /// A provider source, local/<alias>, a Hub model or checkpoint, ./file.safetensors, a
    /// CozyTensors folder (all variants and repo files; its chunks may already be at the Hub), or
    /// run:N[#output], a run's weights output, uploaded from the machine that holds it.
    #[arg(add = clap_complete::ArgValueCompleter::new(crate::completion::local))]
    source: String,
    /// Where it goes: org/name[@tag], the Hub repository and the private tag the upload moves (main).
    destination: String,
    /// A reviewed TensorFS profile to convert with; repeat to compose one model.
    #[arg(long = "source-profile", value_name = "PROFILE")]
    profiles: Vec<String>,
    #[command(flatten)]
    placement: crate::run::Placement,
}

#[derive(clap::Args, Debug)]
#[command(group(ArgGroup::new("lane").required(true).args(["format", "recipe"])))]
pub(crate) struct Quantize {
    /// org/name[@release-or-tag] (main after an upload; none: its default) or org/name#sha256:….
    model: String,
    /// Where it goes: org/name[@tag], the tag whose tree takes the variant (main); default: the
    /// model's own.
    #[arg(value_name = "DESTINATION")]
    destination: Option<String>,
    /// The lanes to write, comma-separated, one run each: int8 (int8-convrot), fp8 (row-scaled
    /// FP8), nvfp4 (block-scaled FP4), w4a8 (4-bit ConvRot codes on int8 tensor cores: Ampere,
    /// Ada, Hopper), svdq-nvfp4 (SVDQuant: an nvfp4 residual and a 16-bit low-rank branch,
    /// Blackwell).
    #[arg(long, value_name = "FORMAT,...", value_delimiter = ',', conflicts_with = "recipe",
        value_parser = crate::run::interface::LANE_FLAGS)]
    format: Vec<String>,
    /// Run the package's recipe of this name: any lane's (`drafter_head` writes a drafter).
    #[arg(long, value_name = "NAME")]
    recipe: Option<String>,
    /// The package whose recipe writes the lane: org/name[@release], or ./dir for one not yet
    /// published. Required: packages that serve the same model can each declare one.
    #[arg(long)]
    package: Option<String>,
    #[command(flatten)]
    placement: crate::run::Placement,
}

#[derive(clap::Args, Debug)]
pub(crate) struct Publish {
    /// org/name.
    model: String,
    /// The release: any tag name; only a PEP 440 version is a candidate for the unversioned default.
    #[arg(long)]
    release: String,
    /// The commit it names: a tag, or sha256:<hex> of a commit or a variant header; default: the
    /// release's own tag.
    #[arg(long, value_name = "COMMIT")]
    commit: Option<String>,
}

#[derive(clap::Args, Debug)]
pub(crate) struct Yank {
    /// org/name.
    model: String,
    /// The release to yank.
    #[arg(long)]
    release: String,
    /// Un-yank it instead.
    #[arg(long)]
    undo: bool,
}

#[derive(clap::Args, Debug)]
pub(crate) struct Delete {
    /// org/name, or org/name@tag for one tag.
    model: String,
}

pub async fn exec(ctx: &Context, args: Args) -> Result<ExitCode, Error> {
    match args.verb {
        Verb::Search(args) => catalog::search(ctx, args).await,
        Verb::Info(args) => catalog::info(ctx, args).await,
        Verb::Family(args) => catalog::family(ctx, args).await,
        Verb::Publish(args) => releases::publish(ctx, args).await,
        Verb::Yank(args) => releases::yank(ctx, args).await,
        Verb::Delete(args) => releases::delete(ctx, args).await,
        Verb::List(args) => machine::list(ctx, args).await,
        Verb::Download(args) => machine::download(ctx, args).await,
        Verb::Upload(args) => machine::upload(ctx, args).await,
        Verb::Quantize(args) => quantize::quantize(ctx, args).await,
        #[cfg(feature = "migrate")]
        Verb::MigrateHeaders(args) => migrate_headers::exec(ctx, args).await,
    }
}
