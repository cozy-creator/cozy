//! `tfs --help`, `-h`, `help` and a verb's `--help` print usage and succeed; a bad flag, a bad
//! verb or wrong arguments fail with 2 and say why.

use std::process::{Command, Output};

fn tfs(args: &[&str]) -> (Option<i32>, String, String) {
    // Its own store: a cached target directory can hold one an older build wrote.
    let home = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("tfs-help-{}", std::process::id()));
    let Output {
        status,
        stdout,
        stderr,
    } = Command::new(env!("CARGO_BIN_EXE_tfs"))
        .args(args)
        .env("TENSORFS_HOME", home)
        .output()
        .unwrap();
    (
        status.code(),
        String::from_utf8(stdout).unwrap(),
        String::from_utf8(stderr).unwrap(),
    )
}

#[test]
fn help_succeeds() {
    for args in [&["--help"][..], &["-h"], &["help"], &["--json", "--help"]] {
        let (code, out, _) = tfs(args);
        assert_eq!(code, Some(0), "{args:?}");
        assert!(
            out.starts_with("usage: tfs")
                && out.contains("\n  import ")
                && out.contains("\n  version"),
            "{args:?}: {out}"
        );
    }
    for args in [
        &["import", "--help"][..],
        &["import", "-h"],
        &["help", "import"],
    ] {
        let (code, out, _) = tfs(args);
        assert_eq!(code, Some(0), "{args:?}");
        assert!(
            out.contains("\n  import <source>") && !out.contains("\n  version"),
            "{args:?}: {out}"
        );
    }
}

#[test]
fn misuse_fails_with_2() {
    let cases: [(&[&str], &str); 6] = [
        (&["--bogus"], "tfs: unknown flag --bogus"),
        (&["-x"], "tfs: unknown flag -x"),
        (&["bogus"], "tfs: unknown verb \"bogus\""),
        (&["convert", "model.safetensors"], "tfs: unknown verb \"convert\""),
        (&["inspect"], "tfs: inspect: wrong arguments"),
        (&[], "tfs: no verb"),
    ];
    for (args, says) in cases {
        let (code, out, err) = tfs(args);
        assert_eq!(code, Some(2), "{args:?}");
        assert!(
            out.is_empty() && err.contains(says) && err.contains("usage: tfs"),
            "{args:?}: {err}"
        );
    }
}
