//! The `cozy` binary dispatches to its roles: CLI verbs, `calcifer` and `machine serve`.
fn main() -> std::process::ExitCode {
    let mut args = std::env::args_os().skip(1);
    match (args.next(), args.next()) {
        (Some(role), _) if role == "calcifer" => calcifer::serve::main(),
        (Some(noun), Some(role)) if noun == "machine" && role == "serve" => {
            tensord::serve::main(std::env::args_os().skip(2))
        }
        _ => cozy_cli::main(),
    }
}
