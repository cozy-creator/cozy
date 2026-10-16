//! `cozy up|down|status|logs`.
use std::{path::Path, process::ExitCode};

use cozy_proto::error::Error;

use crate::Context;

#[derive(clap::Args, Debug)]
pub struct Up {}

pub async fn up(_ctx: &Context, _args: Up) -> Result<ExitCode, Error> {
    Err(crate::not_ported("up"))
}

#[derive(clap::Args, Debug)]
pub struct Down {}

pub async fn down(_ctx: &Context, _args: Down) -> Result<ExitCode, Error> {
    Err(crate::not_ported("down"))
}

#[derive(clap::Args, Debug)]
pub struct Status {}

pub async fn status(_ctx: &Context, _args: Status) -> Result<ExitCode, Error> {
    Err(crate::not_ported("status"))
}

#[derive(clap::Args, Debug)]
pub struct Logs {}

pub async fn logs(_ctx: &Context, _args: Logs) -> Result<ExitCode, Error> {
    Err(crate::not_ported("logs"))
}

/// Stops a running Calcifer and starts `program calcifer` in its place; false when none ran, and then
/// none is started. `cozy update` calls it with the binary it just put in place.
pub(crate) async fn restart(_ctx: &Context, _program: &Path) -> Result<bool, Error> {
    Err(crate::not_ported("up"))
}
