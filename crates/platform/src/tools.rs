//! The tools cozy uses when present and never bundles: where each is found, whether a run or a
//! command needs it, what its absence costs, and how it is installed. `cozy doctor` lists them; a
//! missing optional tool is one warning (`warn_once`) and the work goes on without it; a missing
//! essential one is a refusal naming its install. Probes read no environment: the caller hands in
//! PATH.
use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
    process::Command,
    sync::{Mutex, PoisonError},
};

use crate::paths::exe;

/// Where a tool matters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    /// This computer: the CLI's own features.
    Computer,
    /// A machine: its runs.
    Machine,
}

/// Whether the work can go on without it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    /// Its feature cannot work without it: guaranteed where cozy can fetch it, else refused with
    /// its install named.
    Essential,
    /// A slower or smaller path serves without it, after one warning.
    Optional,
}

/// How a missing tool is installed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Install {
    /// A cozy verb: `cozy tools install uv`.
    Cozy(&'static str),
    /// The OS package that carries it (`system_install` spells the command).
    System(&'static str),
    /// No package installs it alone: what to install, in words.
    Note(&'static str),
}

pub struct Tool {
    pub name: &'static str,
    pub side: Side,
    pub class: Class,
    /// What its absence costs, as doctor and a warning say it.
    pub cost: &'static str,
    pub install: Install,
    /// Finds it in `tools` (cozy's own install), else on `path`.
    pub probe: fn(tools: &Path, path: &[PathBuf]) -> Option<Found>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Found {
    pub path: PathBuf,
    /// The first line of its own version output; empty for a tool that is only looked for.
    pub version: String,
}

macro_rules! looked_for {
    ($name:literal) => {
        |_, path| on_path($name, path)
    };
}

pub const TOOLS: &[Tool] = &[
    Tool {
        name: "uv",
        side: Side::Computer,
        class: Class::Essential,
        cost: "packages can't be authored, locked or published here",
        install: Install::Cozy("cozy tools install uv"),
        probe: uv,
    },
    Tool {
        name: "git",
        side: Side::Computer,
        class: Class::Essential,
        cost: "a package's git dependencies can't be built",
        install: Install::System("git"),
        probe: looked_for!("git"),
    },
    Tool {
        name: "ffmpeg",
        side: Side::Computer,
        class: Class::Optional,
        cost: "uploads go untrimmed (larger, slower)",
        install: Install::Cozy("cozy tools install ffmpeg"),
        probe: |tools, path| ffmpeg(tools, path).map(|(ffmpeg, _)| ffmpeg),
    },
    Tool {
        name: "xdg-open",
        side: Side::Computer,
        class: Class::Optional,
        cost: "the browser isn't opened; the URL is printed instead",
        install: Install::System("xdg-utils"),
        probe: looked_for!("xdg-open"),
    },
    Tool {
        name: "systemctl",
        side: Side::Computer,
        class: Class::Optional,
        cost: "Calcifer doesn't start at login; the next command starts it",
        install: Install::Note("a systemd user session"),
        probe: looked_for!("systemctl"),
    },
    Tool {
        name: "sshd",
        side: Side::Machine,
        class: Class::Optional,
        cost: "no SSH into this dev machine",
        install: Install::System("openssh-server"),
        probe: looked_for!("sshd"),
    },
    Tool {
        name: "unshare",
        side: Side::Machine,
        class: Class::Optional,
        cost: "ingest runs without network isolation",
        install: Install::System("util-linux"),
        probe: looked_for!("unshare"),
    },
    Tool {
        name: "nvcc",
        side: Side::Machine,
        class: Class::Optional,
        cost: "cards without a prebuilt kernel use the slower SDPA/kitchen path",
        install: Install::Note("the CUDA toolkit"),
        probe: looked_for!("nvcc"),
    },
    Tool {
        name: "cc",
        side: Side::Machine,
        class: Class::Optional,
        cost: "Triton kernels and package extensions can't build; those ops use slower fallbacks",
        install: Install::System("gcc"),
        probe: looked_for!("cc"),
    },
    Tool {
        name: "ninja",
        side: Side::Machine,
        class: Class::Optional,
        cost: "kernel builds compile one file at a time",
        install: Install::System("ninja"),
        probe: looked_for!("ninja"),
    },
];

/// Whether this OS uses `tool`: macOS and Windows open URLs and start Calcifer at login with their
/// own built-ins, so xdg-open and systemctl are Linux's alone.
pub fn used_here(tool: &Tool) -> bool {
    cfg!(target_os = "linux") || !matches!(tool.name, "xdg-open" | "systemctl")
}

/// The table's entry for `name`.
pub fn tool(name: &str) -> &'static Tool {
    TOOLS
        .iter()
        .find(|t| t.name == name)
        .unwrap_or_else(|| panic!("{name} is not in the tool table"))
}

/// The one warning a missing optional tool gets in this process: its name, what its absence
/// costs, and `cozy doctor`. None once it was given, so a caller prints only what it gets, the way
/// it prints (a terminal line, a log record, a run's warning).
pub fn warn_once(name: &str) -> Option<String> {
    static WARNED: Mutex<BTreeSet<&'static str>> = Mutex::new(BTreeSet::new());
    let tool = tool(name);
    let mut warned = WARNED.lock().unwrap_or_else(PoisonError::into_inner);
    warned.insert(tool.name).then(|| warning(tool))
}

/// What a missing tool's warning says.
pub fn warning(tool: &Tool) -> String {
    format!("{} not found: {}; see `cozy doctor`", tool.name, tool.cost)
}

/// `name` on `path`, the first match in order; only looked for, never run.
pub fn on_path(name: &str, path: &[PathBuf]) -> Option<Found> {
    path.iter()
        .map(|dir| dir.join(exe(name)))
        .find(|file| executable(file))
        .map(|path| Found {
            path,
            version: String::new(),
        })
}

/// A pinned archive, held to its sha256: nothing else is trusted at download.
pub struct Archive {
    pub url: &'static str,
    pub sha256: &'static str,
    /// Archive member → the file it becomes in the tool's directory.
    pub members: &'static [(&'static str, &'static str)],
}

/// The uv cozy fetches where this computer or machine brings none: one version everywhere.
pub const UV_VERSION: &str = "0.12.24";

/// This OS/CPU's uv archive, or none where no build is pinned.
pub fn uv_build() -> Option<Archive> {
    macro_rules! astral {
        ($triple:literal, $sha256:literal) => {
            Some(Archive {
                url: concat!(
                    "https://github.com/astral-sh/uv/releases/download/0.12.24/uv-",
                    $triple,
                    ".tar.gz"
                ),
                sha256: $sha256,
                members: &[(concat!("uv-", $triple, "/uv"), "uv")],
            })
        };
    }
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("linux", "x86_64") => astral!(
            "x86_64-unknown-linux-gnu",
            "b4dfaef47d491a7296981f8374a4595f55dbf84e8937c8ecd2983574d8bb3da6"
        ),
        ("linux", "aarch64") => astral!(
            "aarch64-unknown-linux-gnu",
            "5231be65f496304623895dacdbf1de8504fec90303684bdf05805aa34414dd21"
        ),
        ("macos", "aarch64") => astral!(
            "aarch64-apple-darwin",
            "0c4346de7abdb49495b393b9ec809fe387aa43e586be20fecb972216c1e71732"
        ),
        ("macos", "x86_64") => astral!(
            "x86_64-apple-darwin",
            "4fa82e37cb94767661f532b001e470b67a186c7260e305bd84ddb78fd545c0b6"
        ),
        ("windows", "x86_64") => Some(Archive {
            url: "https://github.com/astral-sh/uv/releases/download/0.12.24/uv-x86_64-pc-windows-msvc.zip",
            sha256: "7c38608c8a18ee137d748a1773053b07ec8f3a30fab49aebaa6f4e4efeceb019",
            members: &[("uv.exe", "uv.exe")],
        }),
        ("windows", "aarch64") => Some(Archive {
            url: "https://github.com/astral-sh/uv/releases/download/0.12.24/uv-aarch64-pc-windows-msvc.zip",
            sha256: "4b783bda5cc44bbae0651a837223873a7acee31152381aef5fc86ef6bef25997",
            members: &[("uv.exe", "uv.exe")],
        }),
        _ => None,
    }
}

/// The pinned uv's directory under `tools`.
pub fn uv_dir(tools: &Path) -> PathBuf {
    tools.join("uv").join(UV_VERSION)
}

/// The pinned uv in `tools`, else uv on `path`.
pub fn uv(tools: &Path, path: &[PathBuf]) -> Option<Found> {
    std::iter::once(uv_dir(tools))
        .chain(path.iter().cloned())
        .map(|dir| dir.join(exe("uv")))
        .filter(|uv| executable(uv))
        .find_map(|uv| {
            let version = first_line(&uv, &["--version"])?;
            Some(Found { path: uv, version })
        })
}

/// ffmpeg's pinned static build, one per OS/CPU: each archive held to its sha256, nothing else
/// trusted at download. Every one has libx264 and the native aac encoder, and ships ffprobe.
pub const FFMPEG_VERSION: &str = "9.0.2";

/// This OS/CPU's archives, or none where no build is pinned.
pub fn ffmpeg_build() -> &'static [Archive] {
    macro_rules! riedl {
        ($dir:literal, $ffmpeg:literal, $ffprobe:literal) => {
            &[
                Archive {
                    url: concat!(
                        "https://ffmpeg.martin-riedl.de/download/",
                        $dir,
                        "/ffmpeg.zip"
                    ),
                    sha256: $ffmpeg,
                    members: &[("ffmpeg", "ffmpeg")],
                },
                Archive {
                    url: concat!(
                        "https://ffmpeg.martin-riedl.de/download/",
                        $dir,
                        "/ffprobe.zip"
                    ),
                    sha256: $ffprobe,
                    members: &[("ffprobe", "ffprobe")],
                },
            ]
        };
    }
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("linux", "x86_64") => riedl!(
            "linux/amd64/1789931100_9.0.2",
            "fa8ecf4abbd290d98f7d188b8649cc6b391ae209a98452be955a15aab1909d7f",
            "3f428c49070be3d24ec338602b76d412e401ffcb8a5641ef0e729181a232fc32"
        ),
        ("linux", "aarch64") => riedl!(
            "linux/arm64/1789931697_9.0.2",
            "93a76ae90db5474eecdf951a729857c64f3de23567228d6a7d5e6e8e3cd1021b",
            "bcbe80fb741c180083327afaf5434812e006b33cacde2016b9aeaf6936128330"
        ),
        ("macos", "aarch64") => riedl!(
            "macos/arm64/1789931890_9.0.2",
            "c8ed4c4e6978a03c485edbfe4e0a5dc2380f8a30bba5150531b31b094492d924",
            "fcbe839537485eaee7a7a8bc5cbc0f90d53617e80943e8a5b2e31cb851197ea6"
        ),
        ("macos", "x86_64") => riedl!(
            "macos/amd64/1789931006_9.0.2",
            "7c6b4125b191cbf773832dc51f424cf2b6bb7da43007d1e066f95909e47cacd4",
            "2322438ed2f6319a691291b247d09c69dcaa3a982460d1f269a7e1af335cfdfd"
        ),
        ("windows", "x86_64") => &[Archive {
            url: "https://github.com/GyanD/codexffmpeg/releases/download/9.0.2/ffmpeg-9.0.2-essentials_build.zip",
            sha256: "60f467265b1e312373dbcd92200c2618a74850f98d3d078e94296bb3fa2047ba",
            members: &[
                ("ffmpeg-9.0.2-essentials_build/bin/ffmpeg.exe", "ffmpeg.exe"),
                (
                    "ffmpeg-9.0.2-essentials_build/bin/ffprobe.exe",
                    "ffprobe.exe",
                ),
            ],
        }],
        _ => &[],
    }
}

/// The pinned build's directory under `tools`.
pub fn ffmpeg_dir(tools: &Path) -> PathBuf {
    tools.join("ffmpeg").join(FFMPEG_VERSION)
}

/// ffmpeg and ffprobe together: the pinned build in `tools`, else the pair on `path`.
pub fn ffmpeg(tools: &Path, path: &[PathBuf]) -> Option<(Found, PathBuf)> {
    let pinned = ffmpeg_dir(tools);
    std::iter::once(pinned.as_path())
        .chain(path.iter().map(PathBuf::as_path))
        .find_map(|dir| {
            let (ffmpeg, ffprobe) = (dir.join(exe("ffmpeg")), dir.join(exe("ffprobe")));
            if !executable(&ffmpeg) || !executable(&ffprobe) {
                return None;
            }
            let version = first_line(&ffmpeg, &["-hide_banner", "-version"])?;
            Some((
                Found {
                    path: ffmpeg,
                    version,
                },
                ffprobe,
            ))
        })
}

/// Whether the ffmpeg at `ffmpeg` has an encoder named `encoder`.
pub fn ffmpeg_encodes(ffmpeg: &Path, encoder: &str) -> bool {
    let out = Command::new(ffmpeg)
        .args(["-hide_banner", "-encoders"])
        .output();
    out.is_ok_and(|out| {
        String::from_utf8_lossy(&out.stdout)
            .lines()
            .any(|line| line.split_whitespace().nth(1) == Some(encoder))
    })
}

/// The OS package manager's command for `package`, when this computer has one: `sudo` named where
/// it needs it, for the person to run.
pub fn system_install(package: &str, path: &[PathBuf]) -> Option<String> {
    let managers: &[(&str, &str)] = match std::env::consts::OS {
        "linux" => &[
            ("apt-get", "sudo apt-get install -y"),
            ("dnf", "sudo dnf install -y"),
            ("pacman", "sudo pacman -S --noconfirm"),
        ],
        "macos" => &[("brew", "brew install")],
        "windows" => &[("winget", "winget install --exact --id")],
        _ => &[],
    };
    let (manager, command) = managers
        .iter()
        .find(|(program, _)| on_path(program, path).is_some())?;
    let name = match (*manager, package) {
        ("winget", "ffmpeg") => "Gyan.FFmpeg",
        ("winget", "git") => "Git.Git",
        ("apt-get" | "dnf", "ninja") => "ninja-build",
        _ => package,
    };
    Some(format!("{command} {name}"))
}

/// A file this process may run.
fn executable(file: &Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::metadata(file).is_ok_and(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
    }
    #[cfg(not(unix))]
    {
        file.is_file()
    }
}

fn first_line(program: &Path, args: &[&str]) -> Option<String> {
    let out = Command::new(program).args(args).output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    text.lines()
        .next()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(str::to_owned)
}
