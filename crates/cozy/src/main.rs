//! The `cozy` binary dispatches to its roles: CLI verbs, `calcifer` and `machine serve`, and the
//! hidden one-shot `migrate` from the Go CLI's home.
fn main() -> std::process::ExitCode {
    match std::env::args_os().nth(1) {
        Some(role) if role == "calcifer" => calcifer::serve::main(),
        Some(verb) if verb == "migrate" => migrate::main(),
        _ => cozy_cli::main(),
    }
}
