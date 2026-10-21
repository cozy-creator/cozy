//! `cozy uninstall`, through the installed binary in a home set up as the installer leaves one: it
//! asks first, then ends Calcifer and the machine, removes `~/.cozy` (itself with it),
//! `~/.config/cozy` and the shell lines, and keeps only the model store unless `--all`. Every OS.
mod common;

use std::{fs, path::Path, time::Duration};

use common::Home;
use platform::process::Birth;
use serde_json::Value;

const SOURCE: &str = ". \"$HOME/.config/cozy/env.sh\"";

fn write(path: &Path, text: &str) {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, text).unwrap();
}

fn ends(birth: &Birth) {
    for _ in 0..100 {
        if birth.ended().unwrap() {
            return;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    panic!("{birth:?} still runs");
}

/// This user's PATH as the registry keeps it.
#[cfg(windows)]
fn user_path(set: Option<&str>) -> String {
    let script = match set {
        Some(path) => format!("[Environment]::SetEnvironmentVariable('Path', '{path}', 'User')"),
        None => "[Environment]::GetEnvironmentVariable('Path', 'User')".into(),
    };
    let out = std::process::Command::new("powershell")
        .args(["-NoProfile", "-Command", &script])
        .output()
        .unwrap();
    String::from_utf8_lossy(&out.stdout).trim().to_owned()
}

#[test]
fn uninstall_removes_what_cozy_made_and_keeps_the_model_store_unless_all() {
    let Some(home) = Home::new() else { return };
    let user = home.path();
    let bin = home.root().join("bin");
    let installed = home.installed();
    let install = || {
        let built = std::env::var_os("COZY_TEST_BINARY").map_or_else(
            || env!("CARGO_BIN_EXE_cozy").into(),
            std::path::PathBuf::from,
        );
        fs::create_dir_all(&bin).unwrap();
        // The harness's may be a link to the build, which a copy onto it would overwrite.
        let _ = fs::remove_file(&installed);
        fs::copy(built, &installed).unwrap();
        write(
            &user.join(".config/cozy/env.sh"),
            "export PATH=\"$HOME/.cozy/bin:$PATH\"\n",
        );
    };
    install();
    let mine = "alias ll='ls -l'\n";
    write(
        &user.join(".bashrc"),
        &format!("{mine}\n{SOURCE}\nexport EDITOR=vi\n"),
    );
    write(&user.join(".zshrc"), &format!("\n{SOURCE}\n"));
    let fish = user.join(".config/fish/conf.d/cozy.env.fish");
    write(&fish, "\nsource \"$HOME/.config/cozy/env.fish\"\n");
    #[cfg(windows)]
    let before = user_path(None);
    #[cfg(windows)]
    user_path(Some(&format!("{};{before}", bin.display())));

    let calcifer = home.json(["up", "--json"]);
    let calcifer = Birth::of(calcifer["calcifer"]["pid"].as_u64().unwrap() as u32).unwrap();
    home.json(["machine", "enable", "--json"]);
    let holder = home
        .root()
        .join("machine/var/lib/cozy/machine/agent.lock.holder");
    let machine: Birth = serde_json::from_slice(&fs::read(holder).unwrap()).unwrap();
    let store = home.root().join("machine/var/lib/tensorfs");
    write(&store.join("objects/model"), "weights");

    // No terminal to answer and no --yes: refused, nothing touched.
    let asked = home.cozy(["uninstall", "--json"]);
    assert_eq!(
        (asked.code, asked.refusal()["code"].as_str()),
        (2, Some("uninstall_unconfirmed")),
        "{asked:?}"
    );
    assert!(installed.exists() && !calcifer.ended().unwrap());

    let done = home.json(["uninstall", "--yes", "--json"]);
    assert_eq!(
        done["kept"][0]["path"].as_str().map(Path::new),
        Some(&*store),
        "{done}"
    );
    assert!(done["kept"][0]["bytes"].as_u64() >= Some(7), "{done}");
    ends(&calcifer);
    ends(&machine);
    for _ in 0..100 {
        if !installed.exists() {
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    assert!(!installed.exists(), "the binary is still there");
    assert!(!user.join(".config/cozy").exists());
    assert!(!fish.exists());
    let left: Vec<_> = fs::read_dir(home.root())
        .unwrap()
        .flatten()
        .map(|e| e.file_name())
        .collect();
    assert_eq!(left, ["machine"], "only the model store stays");
    assert_eq!(
        fs::read_to_string(store.join("objects/model")).unwrap(),
        "weights"
    );
    assert_eq!(
        fs::read_to_string(user.join(".bashrc")).unwrap(),
        format!("{mine}export EDITOR=vi\n")
    );
    assert_eq!(fs::read_to_string(user.join(".zshrc")).unwrap(), "");
    #[cfg(windows)]
    assert_eq!(user_path(None), before);

    install();
    let all: Value = home.json(["uninstall", "--yes", "--all", "--json"]);
    assert_eq!(all["kept"], serde_json::json!([]), "{all}");
    for _ in 0..100 {
        if !home.root().exists() {
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    assert!(!home.root().exists(), "~/.cozy is still there");
}
