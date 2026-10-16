//! The `cozy` binary dispatches to its roles: CLI verbs, `calcifer` and `machine serve`.
fn main() -> std::process::ExitCode {
    if std::env::args_os()
        .nth(1)
        .is_some_and(|role| role == "calcifer")
    {
        return calcifer::serve::main();
    }
    cozy_cli::main()
}
