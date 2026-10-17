//! The `cozy` binary dispatches to its roles: CLI verbs, `calcifer` and `machine serve`, and the
//! hidden one-shot `migrate` from the Go CLI's home.
fn main() -> std::process::ExitCode {
    let args: Vec<std::ffi::OsString> = std::env::args_os().skip(1).collect();
    let words: Vec<&std::ffi::OsString> = args.iter().filter(|arg| *arg != "--json").collect();
    match words.as_slice() {
        [role, ..] if *role == "calcifer" => calcifer::serve::main(),
        [verb, ..] if *verb == "migrate" => migrate::main(),
        [noun, role, rest @ ..] if *noun == "machine" && *role == "serve" => {
            machine(std::iter::once(*role).chain(rest.iter().copied()).cloned())
        }
        _ => cozy_cli::main(),
    }
}

/// `machine serve`'s arguments, from `serve` on.
#[cfg(target_os = "linux")]
fn machine(args: impl Iterator<Item = std::ffi::OsString>) -> std::process::ExitCode {
    tensord::serve::main(args)
}

#[cfg(not(target_os = "linux"))]
fn machine(_: impl Iterator<Item = std::ffi::OsString>) -> std::process::ExitCode {
    cozy_cli::output::unsupported("this computer's machine runs on Linux")
}
