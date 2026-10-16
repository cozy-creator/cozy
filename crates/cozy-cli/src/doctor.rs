//! `cozy doctor`: one machine's driver, GPUs, toolchain and kernels, as its Status reports them.
use std::process::ExitCode;

use cozy_proto::error::Error;

use crate::Context;

// Read once Calcifer's machine routes land.
#[allow(dead_code)]
#[derive(clap::Args, Debug)]
pub struct Args {
    /// The machine to check.
    #[arg(long, default_value = "local")]
    on: String,
}

pub async fn exec(_ctx: &Context, _args: Args) -> Result<ExitCode, Error> {
    Err(crate::not_ported("doctor"))
}
