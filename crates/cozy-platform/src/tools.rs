//! The tools cozy uses when present and never bundles: where each is found, what its absence
//! costs, and how it is installed. `cozy doctor` lists them, and a warning about a missing one says
//! its cost and its install from here. Probes read no environment: the caller hands in PATH.
use std::{
    path::{Path, PathBuf},
    process::Command,
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

pub struct Tool {
    pub name: &'static str,
    pub side: Side,
    /// What its absence costs, as doctor and a warning say it.
    pub cost: &'static str,
    /// The cozy verb that installs it.
    pub install: &'static str,
    /// Finds it in `tools` (cozy's own install), else on `path`.
    pub probe: fn(tools: &Path, path: &[PathBuf]) -> Option<Found>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Found {
    pub path: PathBuf,
    /// The first line of its own version output.
    pub version: String,
}

pub const TOOLS: &[Tool] = &[Tool {
    name: "ffmpeg",
    side: Side::Computer,
    cost: "uploads go untrimmed (larger, slower)",
    install: "cozy tools install ffmpeg",
    probe: |tools, path| ffmpeg(tools, path).map(|(ffmpeg, _)| ffmpeg),
}];

/// The table's entry for `name`.
pub fn tool(name: &str) -> &'static Tool {
    TOOLS
        .iter()
        .find(|t| t.name == name)
        .unwrap_or_else(|| panic!("{name} is not in the tool table"))
}

/// ffmpeg's pinned static build, one per OS/CPU: each archive held to its sha256, nothing else
/// trusted at download. Every one has libx264 and the native aac encoder, and ships ffprobe.
pub const FFMPEG_VERSION: &str = "9.0.2";

pub struct Archive {
    pub url: &'static str,
    pub sha256: &'static str,
    /// Archive member → the file it becomes in the build's directory.
    pub members: &'static [(&'static str, &'static str)],
}

/// This OS/CPU's archives, or none where no build is pinned.
pub fn ffmpeg_build() -> &'static [Archive] {
    macro_rules! riedl {
        ($dir:literal, $ffmpeg:literal, $ffprobe:literal) => {
            &[
                Archive {
                    url: concat!("https://ffmpeg.martin-riedl.de/download/", $dir, "/ffmpeg.zip"),
                    sha256: $ffmpeg,
                    members: &[("ffmpeg", "ffmpeg")],
                },
                Archive {
                    url: concat!("https://ffmpeg.martin-riedl.de/download/", $dir, "/ffprobe.zip"),
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
            sha256: "4705843ccaaf54257c16ad90f3e952ece33c17df964ecf7bfdbb0f49c7171077",
            members: &[
                ("ffmpeg-9.0.2-essentials_build/bin/ffmpeg.exe", "ffmpeg.exe"),
                ("ffmpeg-9.0.2-essentials_build/bin/ffprobe.exe", "ffprobe.exe"),
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
            if !ffmpeg.is_file() || !ffprobe.is_file() {
                return None;
            }
            let version = first_line(&ffmpeg, &["-hide_banner", "-version"])?;
            Some((Found { path: ffmpeg, version }, ffprobe))
        })
}

/// Whether the ffmpeg at `ffmpeg` has an encoder named `encoder`.
pub fn ffmpeg_encodes(ffmpeg: &Path, encoder: &str) -> bool {
    let out = Command::new(ffmpeg).args(["-hide_banner", "-encoders"]).output();
    out.is_ok_and(|out| {
        String::from_utf8_lossy(&out.stdout)
            .lines()
            .any(|line| line.split_whitespace().nth(1) == Some(encoder))
    })
}

/// The OS package manager's command for `package`, when this computer has one: `sudo` named where
/// it needs it, for the person to run.
pub fn system_install(package: &str, path: &[PathBuf]) -> Option<String> {
    let has = |program: &str| path.iter().any(|dir| dir.join(exe(program)).is_file());
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
    let name = match (std::env::consts::OS, package) {
        ("windows", "ffmpeg") => "Gyan.FFmpeg",
        _ => package,
    };
    managers
        .iter()
        .find(|(program, _)| has(program))
        .map(|(_, command)| format!("{command} {name}"))
}

fn first_line(program: &Path, args: &[&str]) -> Option<String> {
    let out = Command::new(program).args(args).output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    text.lines().next().map(str::trim).filter(|l| !l.is_empty()).map(str::to_owned)
}
