//! A launchd LaunchAgent in the user's GUI domain. `KeepAlive{SuccessfulExit=false}` restarts it
//! only after a failure, so a stop is a `bootout` (a signalled exit counts as a failure).
use super::{ServiceSpec, run};
use crate::{Error, Result, durable, paths::Home};
use std::{
    fs, io,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

const MANAGER: &str = "launchd";

pub(super) fn dir(_: &Home, user_home: &Path) -> PathBuf {
    user_home.join("Library").join("LaunchAgents")
}

pub(super) fn available() -> bool {
    launchctl(&["print", &domain()])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

pub(super) fn install(dir: &Path, spec: &ServiceSpec) -> Result<()> {
    fs::create_dir_all(dir).map_err(Error::io("create directory", dir))?;
    durable::replace(&plist(dir, &spec.name), render(spec).as_bytes(), 0o644)
}

pub(super) fn start(dir: &Path, spec: &ServiceSpec) -> Result<()> {
    // Already bootstrapped is the one expected refusal; kickstart then says whether it runs.
    let _ = launchctl(&["bootstrap", &domain()])
        .arg(plist(dir, &spec.name))
        .output();
    run(MANAGER, &mut launchctl(&["kickstart", &target(&spec.name)]))
}

pub(super) fn announce(_: &Path, _: &str) -> Result<()> {
    Ok(())
}

pub(super) fn stop(_: &Path, name: &str) -> Result<()> {
    let stopped = run(MANAGER, &mut launchctl(&["bootout", &target(name)]));
    let loaded = launchctl(&["print", &target(name)])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success());
    if loaded { stopped } else { Ok(()) }
}

pub(super) fn uninstall(dir: &Path, name: &str) -> Result<()> {
    let path = plist(dir, name);
    if let Err(error) = fs::remove_file(&path)
        && error.kind() != io::ErrorKind::NotFound
    {
        return Err(Error::io("remove", path)(error));
    }
    Ok(())
}

fn launchctl(args: &[&str]) -> Command {
    let mut command = Command::new("launchctl");
    command.args(args);
    command
}

fn domain() -> String {
    // SAFETY: getuid cannot fail.
    format!("gui/{}", unsafe { libc::getuid() })
}

fn label(name: &str) -> String {
    format!("com.cozy-creator.{name}")
}

fn target(name: &str) -> String {
    format!("{}/{}", domain(), label(name))
}

fn plist(dir: &Path, name: &str) -> PathBuf {
    dir.join(format!("{}.plist", label(name)))
}

fn render(spec: &ServiceSpec) -> String {
    let string = |text: &str| format!("<string>{}</string>", escape(text));
    let arguments: String = std::iter::once(spec.program.as_os_str())
        .chain(spec.args.iter().map(|arg| arg.as_os_str()))
        .map(|word| string(&word.to_string_lossy()))
        .collect();
    let log = spec.log.as_ref().map_or(String::new(), |log| {
        let log = string(&log.to_string_lossy());
        format!("<key>StandardOutPath</key>{log}<key>StandardErrorPath</key>{log}")
    });
    format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
         <!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n\
         <plist version=\"1.0\"><dict><key>Label</key>{}<key>ProgramArguments</key><array>{arguments}</array>\
         <key>RunAtLoad</key><true/><key>KeepAlive</key><dict><key>SuccessfulExit</key><false/></dict>{log}\
         </dict></plist>\n",
        string(&label(&spec.name))
    )
}

fn escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

#[cfg(test)]
mod tests {
    #[test]
    fn an_agent_keeps_every_argument_whole() {
        let spec = super::ServiceSpec {
            name: "calcifer".into(),
            program: "/Users/a b/.cozy/bin/cozy".into(),
            args: vec!["calcifer".into(), "<a & b>".into()],
            log: None,
        };
        let plist = super::render(&spec);
        assert!(plist.contains(
            "<array><string>/Users/a b/.cozy/bin/cozy</string><string>calcifer</string>\
             <string>&lt;a &amp; b&gt;</string></array>"
        ));
        assert!(plist.contains("<string>com.cozy-creator.calcifer</string>"));
    }
}
