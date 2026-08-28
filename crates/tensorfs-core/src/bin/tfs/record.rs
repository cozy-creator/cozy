//! Focused executable arms for local verification-record bytes.

use std::path::Path;
use std::process::ExitCode;

use tensorfs_core::err::Code;
use tensorfs_core::ids::Doc;
use tensorfs_core::store::VRecord;

use crate::bail;

fn record(identity: u64) -> VRecord {
    VRecord {
        sha256: "0".repeat(64),
        length: 1,
        dev: identity,
        ino: identity,
        mtime_s: 1,
        mtime_ns: 2,
        generation: 3,
    }
}

pub fn cmd_arms() -> ExitCode {
    let safe = tensorfs_core::limits::INT_MAX as u64;
    for value in [safe, safe + 1, u64::MAX] {
        let want = record(value);
        let bytes = want.canonical_bytes();
        if VRecord::parse(&bytes).ok().as_ref() != Some(&want) {
            eprintln!("record round-trip moved file identity {value}");
            return ExitCode::FAILURE;
        }
        let quoted = format!("\"ino\":\"{value}\"");
        if bytes
            .windows(quoted.len())
            .any(|window| window == quoted.as_bytes())
            != (value > safe)
        {
            eprintln!("record chose the wrong canonical spelling for file identity {value}");
            return ExitCode::FAILURE;
        }
    }

    let large = String::from_utf8(record(safe + 1).canonical_bytes()).unwrap();
    let needle = format!("\"ino\":\"{}\"", safe + 1);
    let malformed = [
        (
            large.replace(&needle, "\"ino\":\"09007199254740992\""),
            Code::NONCANONICAL_ENCODING,
        ),
        (
            large.replace(&needle, &format!("\"ino\":\"{safe}\"")),
            Code::NONCANONICAL_ENCODING,
        ),
        (
            large.replace(&needle, "\"ino\":\"18446744073709551616\""),
            Code::NUMBER_RANGE,
        ),
    ];
    for (bytes, code) in malformed {
        if !matches!(VRecord::parse(bytes.as_bytes()), Err(error) if error.code == code) {
            eprintln!("malformed file identity did not refuse {code:?}");
            return ExitCode::FAILURE;
        }
    }
    println!("verification record file identities: 6/6 arms green");
    ExitCode::SUCCESS
}

pub fn cmd_migrate(root: &Path) -> ExitCode {
    let store = match tensorfs_core::store::Store::open(root) {
        Ok(store) => store,
        Err(error) => return bail(error),
    };
    let result = match store.migrate_file_identity_records() {
        Ok(result) => result,
        Err(error) => return bail(error),
    };
    println!("records_scanned {}", result.scanned);
    println!("records_current {}", result.current);
    println!("records_migrated {}", result.migrated);
    println!("payload_bytes_read 0");
    println!("payload_bytes_hashed 0");
    ExitCode::SUCCESS
}
