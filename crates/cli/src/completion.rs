//! `cozy completion <shell>`: the script that wires a shell's <TAB> to the binary
//! (`COMPLETE=<shell> cozy -- …`), answered from the grammar itself. Where a path goes, the shell
//! completes it with its own file completion, as for any command. Completers read only local
//! files: no Calcifer, no network, no writes.
use std::{
    ffi::{OsStr, OsString},
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
    process::ExitCode,
};

use clap::{ValueEnum, ValueHint};
use clap_complete::{
    ArgValueCompleter, CompleteEnv, CompletionCandidate,
    env::{EnvCompleter, Powershell, Shells},
};
use protocol::{
    config::{self, Flags},
    error::{Error, Kind},
};

use crate::Context;

/// The environment variable the registration scripts set: clap's completion protocol.
const VAR: &str = "COMPLETE";

/// A completer's answer that a path goes here: `<prefix>PATH<files|dirs>`. The shell completes
/// the rest of the word after `<prefix>` itself.
const PATH: char = '\u{1}';

#[derive(clap::Args, Debug)]
pub struct Args {
    shell: Shell,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Shell {
    Bash,
    Zsh,
    Fish,
    Powershell,
}

pub async fn exec(_ctx: &Context, args: Args) -> Result<ExitCode, Error> {
    args.shell
        .write_registration(VAR, "cozy", "cozy", "cozy", &mut io::stdout())
        .map_err(|e| Error::new(Kind::Internal, "stdout", e.to_string()))?;
    Ok(ExitCode::SUCCESS)
}

/// Answers a shell's completion request and exits; returns at once on any other run.
pub fn answer(command: fn() -> clap::Command) {
    CompleteEnv::with_factory(move || paths(command()))
        .var(VAR)
        .shells(Shells(&[
            &Shell::Bash,
            &Shell::Zsh,
            &Shell::Fish,
            &Shell::Powershell,
        ]))
        .complete();
}

/// Every argument whose value hint is a path (a `PathBuf`'s is) completes as one.
fn paths(command: clap::Command) -> clap::Command {
    command
        .mut_args(|arg| match arg.get_value_hint() {
            ValueHint::AnyPath | ValueHint::FilePath => arg.add(ArgValueCompleter::new(files)),
            ValueHint::DirPath => arg.add(ArgValueCompleter::new(dirs)),
            _ => arg,
        })
        .mut_subcommands(paths)
}

fn files(current: &OsStr) -> Vec<CompletionCandidate> {
    path(current, "files")
}

fn dirs(current: &OsStr) -> Vec<CompletionCandidate> {
    path(current, "dirs")
}

/// A path, or the path in `name=PATH`; nothing for a flag or a `key:=<json>` term.
fn path(current: &OsStr, kind: &str) -> Vec<CompletionCandidate> {
    let current = current.to_string_lossy();
    let prefix = match current.split_once('=') {
        _ if current.starts_with('-') => return vec![],
        Some((key, _)) if key.ends_with(':') => return vec![],
        Some((key, _)) if !explicit(key) => &current[..=key.len()],
        _ => "",
    };
    vec![CompletionCandidate::new(format!("{prefix}{PATH}{kind}"))]
}

/// A reference or local code: a path only when written as one (`./`, `/`, `~`).
pub fn local(current: &OsStr) -> Vec<CompletionCandidate> {
    match explicit(&current.to_string_lossy()) {
        true => path(current, "files"),
        false => vec![],
    }
}

fn explicit(text: &str) -> bool {
    text.starts_with(['.', '/', '~']) || Path::new(text).is_absolute()
}

/// Hub names, with each one's URL as its help.
pub fn hubs(current: &OsStr) -> Vec<CompletionCandidate> {
    let Ok(loaded) = config::load(Flags::default()) else {
        return vec![];
    };
    let prefix = current.to_string_lossy().to_ascii_lowercase();
    loaded
        .config
        .hubs()
        .into_iter()
        .filter_map(|hub| {
            let name = hub.name?;
            name.starts_with(&prefix)
                .then(|| CompletionCandidate::new(name).help(Some(hub.origin.to_string().into())))
        })
        .collect()
}

/// Asks with COMP_WORDBREAKS, COMP_LINE and COMP_POINT, so a word is the shell's, `=` and `:`
/// inside it too. The first line answers `files`, `dirs` (readline completes the path) or
/// `values`, then the values.
const BASH: &str = r#"_cozy() {
    local IFS=$'\n' reply
    read -r -d '' -a reply < <(COMPLETE=bash cozy -- "$COMP_WORDBREAKS" "$COMP_LINE" "$COMP_POINT" 2>/dev/null)
    COMPREPLY=("${reply[@]:1}")
    case ${reply[0]} in
    values) ;;
    dirs\ *) compopt -o dirnames ;;
    *) compopt -o default ;;
    esac
}
complete -F _cozy cozy
"#;

/// `files <prefix>` and `dirs <prefix>` complete the word after `<prefix>` with `_files`.
const ZSH: &str = r#"#compdef cozy
_cozy() {
    local -a reply
    reply=("${(@f)$(COMPLETE=zsh cozy -- "${(@Q)words[1,CURRENT-1]}" "${(Q)PREFIX}" 2>/dev/null)}")
    case ${reply[1]%% *} in
    values) reply=("${(@)reply[2,-1]}"); _describe -V values reply ;;
    dirs) compset -P "${(b)reply[1]#* }"; _files -/ ;;
    *) compset -P "${(b)reply[1]#* }"; _files ;;
    esac
}
if [ "$funcstack[1]" = _cozy ]; then _cozy "$@"; else compdef _cozy cozy; fi
"#;

/// `files <prefix>` and `dirs <prefix>` complete the token after `<prefix>` as fish does a path.
const FISH: &str = r#"function __cozy_complete
    set -l reply (COMPLETE=fish cozy -- (commandline -opc) (commandline -ct) 2>/dev/null)
    set -l answer (string split -m1 ' ' -- "$reply[1]") ''
    set -e reply[1]
    set -l rest (string sub -s (math (string length -- "$answer[2]") + 1) -- (commandline -ct))
    switch $answer[1]
        case values
            printf '%s\n' $reply
        case dirs
            printf '%s\n' "$answer[2]"(__fish_complete_directories "$rest")
        case '*'
            printf '%s\n' "$answer[2]"(__fish_complete_path "$rest")
    end
end
complete --keep-order --exclusive --command cozy --arguments '(__cozy_complete)'
"#;

impl EnvCompleter for Shell {
    fn name(&self) -> &'static str {
        match self {
            Shell::Bash => "bash",
            Shell::Zsh => "zsh",
            Shell::Fish => "fish",
            Shell::Powershell => "powershell",
        }
    }

    fn is(&self, name: &str) -> bool {
        match self {
            Shell::Powershell => Powershell.is(name),
            _ => name == self.name(),
        }
    }

    fn write_registration(
        &self,
        var: &str,
        name: &str,
        bin: &str,
        completer: &str,
        buf: &mut dyn Write,
    ) -> io::Result<()> {
        let script = match self {
            Shell::Bash => BASH,
            Shell::Zsh => ZSH,
            Shell::Fish => FISH,
            Shell::Powershell => {
                return Powershell.write_registration(var, name, bin, completer, buf);
            }
        };
        buf.write_all(script.as_bytes())
    }

    fn write_complete(
        &self,
        command: &mut clap::Command,
        args: Vec<OsString>,
        cwd: Option<&Path>,
        buf: &mut dyn Write,
    ) -> io::Result<()> {
        let (words, kept) = match self {
            Shell::Bash => bash_words(&args)?,
            _ => (args, String::new()),
        };
        let index = words.len().saturating_sub(1);
        let found = clap_complete::engine::complete(command, words, index, cwd)?;
        let path = found.iter().find_map(|c| {
            let value = c.get_value().to_string_lossy();
            let (prefix, kind) = value.split_once(PATH)?;
            Some(format!("{kind} {prefix}"))
        });
        match (self, path) {
            // PowerShell completes a path itself when a completer answers nothing.
            (Shell::Powershell, Some(_)) => return Ok(()),
            (_, Some(path)) => return writeln!(buf, "{path}"),
            (Shell::Powershell, None) => {}
            (_, None) => writeln!(buf, "values")?,
        }
        for candidate in found {
            let value = candidate.get_value().to_string_lossy();
            let help = (candidate.get_help())
                .map(|h| h.to_string().lines().next().unwrap_or_default().to_owned());
            match (self, help) {
                // bash replaces only the part of the word after `kept`.
                (Shell::Bash, _) => {
                    if let Some(rest) = value.strip_prefix(&kept) {
                        writeln!(buf, "{rest}")?;
                    }
                }
                (Shell::Zsh, help) => {
                    let value = value.replace('\\', "\\\\").replace(':', "\\:");
                    match help {
                        Some(help) => writeln!(buf, "{value}:{}", help.replace('\\', "\\\\"))?,
                        None => writeln!(buf, "{value}")?,
                    }
                }
                (_, Some(help)) => writeln!(buf, "{value}\t{help}")?,
                (_, None) => writeln!(buf, "{value}")?,
            }
        }
        Ok(())
    }
}

/// bash's request as words, quoting removed, the last the word at the cursor; and the part of
/// that word bash keeps, up to its last COMP_WORDBREAKS character (`name=` of `name=value`).
fn bash_words(args: &[OsString]) -> io::Result<(Vec<OsString>, String)> {
    let [breaks, line, point] = args else {
        return Err(io::Error::other(
            "bash passes COMP_WORDBREAKS, COMP_LINE and COMP_POINT",
        ));
    };
    let (breaks, line) = (breaks.to_string_lossy(), line.to_string_lossy());
    // COMP_POINT counts bytes.
    let mut point = (point.to_string_lossy().parse().unwrap_or(line.len())).min(line.len());
    while !line.is_char_boundary(point) {
        point -= 1;
    }
    let (words, raw) = shell_words(&line[..point]);
    let kept = raw
        .rfind(|c: char| breaks.contains(c) && !" \t\n\"'".contains(c))
        .map_or(0, |i| i + 1);
    Ok((words, raw[..kept].to_owned()))
}

/// A POSIX shell line's words, quotes and escapes removed, and the raw text of the last, which
/// is empty after a space.
fn shell_words(line: &str) -> (Vec<OsString>, &str) {
    let (mut words, mut word, mut start) = (vec![], String::new(), None);
    let mut quote = None;
    let mut chars = line.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        if quote.is_none() && c.is_whitespace() {
            if start.take().is_some() {
                words.push(std::mem::take(&mut word).into());
            }
            continue;
        }
        start.get_or_insert(i);
        match (quote, c) {
            (None, '\'' | '"') => quote = Some(c),
            (Some(q), _) if c == q => quote = None,
            (None, '\\') => word.extend(chars.next().map(|(_, c)| c)),
            (Some('"'), '\\') if chars.peek().is_some_and(|(_, c)| "\"\\$`".contains(*c)) => {
                word.extend(chars.next().map(|(_, c)| c));
            }
            _ => word.push(c),
        }
    }
    words.push(word.into());
    (words, start.map_or("", |i| &line[i..]))
}

/// Where bash-completion and fish load cozy's completion: stubs asking the `cozy` on PATH for
/// its script, so they never go stale.
fn stubs(user_home: &Path) -> [(PathBuf, &'static str); 2] {
    let env = |name: &str| {
        (config::environment().get(name))
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
    };
    let data = env("XDG_DATA_HOME").unwrap_or_else(|| user_home.join(".local/share"));
    let settings = env("XDG_CONFIG_HOME").unwrap_or_else(|| user_home.join(".config"));
    [
        (
            data.join("bash-completion/completions/cozy"),
            "eval \"$(cozy completion bash 2>/dev/null)\"\n",
        ),
        (
            settings.join("fish/completions/cozy.fish"),
            "cozy completion fish 2>/dev/null | source\n",
        ),
    ]
}

/// Puts the stubs in place of any older cozy completion there: bash's, and fish's where fish
/// keeps a config.
pub fn install(user_home: &Path) -> io::Result<()> {
    if cfg!(windows) {
        return Ok(());
    }
    for (file, stub) in stubs(user_home) {
        let dir = file.parent().expect("a stub has a directory");
        let fish = file.extension().is_some_and(|e| e == "fish");
        if fish && !dir.parent().is_some_and(Path::is_dir)
            || fs::read_to_string(&file).is_ok_and(|text| text == stub)
        {
            continue;
        }
        fs::create_dir_all(dir)?;
        fs::write(&file, stub)?;
    }
    Ok(())
}

/// The completion files `cozy uninstall` removes.
pub fn installed(user_home: &Path) -> Vec<PathBuf> {
    (stubs(user_home).into_iter())
        .map(|(file, _)| file)
        .filter(|file| file.is_file())
        .collect()
}
