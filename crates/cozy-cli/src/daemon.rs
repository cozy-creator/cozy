//! `cozy up|down|status|logs`.
use std::process::ExitCode;

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
