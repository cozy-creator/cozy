//! `COZY_COMMIT`: the commit this build is from, which `machine show` reports. A build input
//! `COZY_COMMIT` wins (a container build without the repository), then git's HEAD, then CI's
//! `GITHUB_SHA`.
use std::process::Command;

fn main() {
    // Any owner: a container build reads a checkout mounted from another user.
    let git = |args: &[&str]| {
        let out = Command::new("git")
            .args(["-c", "safe.directory=*"])
            .args(args)
            .output()
            .ok()?;
        out.status
            .success()
            .then(|| String::from_utf8_lossy(&out.stdout).trim().to_owned())
    };
    // Rebuilt when HEAD moves: HEAD, the branch it names, and packed refs.
    let branch = git(&["symbolic-ref", "-q", "HEAD"]);
    for name in ["HEAD", "packed-refs"].into_iter().chain(branch.as_deref()) {
        if let Some(path) = git(&["rev-parse", "--git-path", name]) {
            println!("cargo:rerun-if-changed={path}");
        }
    }
    println!("cargo:rerun-if-env-changed=COZY_COMMIT");
    println!("cargo:rerun-if-env-changed=GITHUB_SHA");
    let commit = std::env::var("COZY_COMMIT")
        .ok()
        .filter(|c| !c.is_empty())
        .or_else(|| git(&["rev-parse", "HEAD"]))
        .or_else(|| std::env::var("GITHUB_SHA").ok())
        .unwrap_or_else(|| "unknown".into());
    println!("cargo:rustc-env=COZY_COMMIT={commit}");
}
