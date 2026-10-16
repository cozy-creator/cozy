//! Rendering shared by every verb.
use std::process::ExitCode;

use cozy_proto::error::{Envelope, Error};

use crate::Context;

/// Prints a refusal on stdout (the envelope with `--json`) and answers its exit code.
pub fn refusal(ctx: &Context, error: &Error) -> ExitCode {
    if ctx.json {
        let envelope = Envelope {
            error: error.clone(),
        };
        println!(
            "{}",
            serde_json::to_string(&envelope).expect("an envelope serializes")
        );
    } else {
        println!("error({}): {}", error.code, error.message);
        if let Some(remedy) = &error.remedy {
            println!("  {remedy}");
        }
        for next in &error.next {
            println!("  next: {next}");
        }
    }
    ExitCode::from(error.exit_code() as u8)
}
