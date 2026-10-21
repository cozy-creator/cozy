//! Which wheels this machine installs, by the platform tag in a wheel's file name.
use std::env::consts::{ARCH, OS};

/// How a wheel tagged `platform` (`manylinux_2_28_x86_64`, `win_amd64`, `macosx_11_0_arm64`, a
/// compressed `a.b`, or `any`) suits this OS and CPU: 0 built for them, 1 pure Python, `None` it
/// does not run here.
pub fn rank(platform: &str) -> Option<u8> {
    if platform == "any" {
        return Some(1);
    }
    let (os, arch) = match (OS, ARCH) {
        ("linux", arch) => ("manylinux", arch),
        ("windows", "x86_64") => ("win", "amd64"),
        ("windows", "aarch64") => ("win", "arm64"),
        ("macos", "aarch64") => ("macosx", "arm64"),
        ("macos", arch) => ("macosx", arch),
        _ => return None,
    };
    let fits = |tag: &str| {
        tag.starts_with(os) && (tag.ends_with(arch) || (os == "macosx" && tag.ends_with("universal2")))
    };
    platform.split('.').any(fits).then_some(0)
}
