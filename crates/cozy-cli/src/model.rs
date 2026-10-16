//! `cozy model`.
use std::process::ExitCode;

use cozy_proto::error::Error;

use crate::Context;

#[derive(clap::Args, Debug)]
pub struct Args {}

pub async fn exec(_ctx: &Context, _args: Args) -> Result<ExitCode, Error> {
    Err(crate::not_ported("model"))
}
