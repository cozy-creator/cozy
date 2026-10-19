//! The `cozy` binary dispatches to its roles: CLI verbs, `calcifer` and `machine serve`.
fn main() -> std::process::ExitCode {
    let args: Vec<std::ffi::OsString> = std::env::args_os().skip(1).collect();
    let words: Vec<&std::ffi::OsString> = args.iter().filter(|arg| *arg != "--json").collect();
    match words.as_slice() {
        [role, ..] if *role == "calcifer" => calcifer::serve::main(),
        [noun, role, rest @ ..] if *noun == "machine" && *role == "serve" => {
            machine(std::iter::once(*role).chain(rest.iter().copied()).cloned())
        }
        _ => cli::main(),
    }
}

/// `machine serve`'s arguments, from `serve` on.
fn machine(args: impl Iterator<Item = std::ffi::OsString>) -> std::process::ExitCode {
    tensord::serve::main(args)
}
