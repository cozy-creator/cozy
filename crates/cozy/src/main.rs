//! The `cozy` binary dispatches to its roles: CLI verbs, `calcifer` and `machine serve`.
fn main() -> std::process::ExitCode {
    cozy_cli::main()
}
