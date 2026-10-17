//! What every process opening one Store must agree on: the `cozy` binary's TensorD and the
//! `tensorfs` wheel's extension share each machine's Store, built from one tree but released
//! apart. A Store records the [`STORE_FORMAT`] and [`CLAIM_PROTOCOL`] it is used under in
//! `<root>/format`, and every open compares them with its own:
//!
//! - no record (a Store made before it existed), or one a crash cut short: the opener writes its
//!   own. The record is never synced, since any opener can write it again;
//! - another store format: refused, since this build would misread the bytes on disk;
//! - another claim protocol: refused, except by the Store's owner (TensorD), which records its
//!   own. The owner decides how the Store is coordinated, and its restart ends what it started.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::err::{refuse, Code, Result};
use crate::{CLAIM_PROTOCOL, STORE_FORMAT};

/// The record's name under the Store root.
pub const RECORD: &str = "format";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Contract {
    pub store_format: u32,
    pub claim_protocol: u32,
}

impl Contract {
    /// This build's.
    pub const OURS: Contract = Contract {
        store_format: STORE_FORMAT,
        claim_protocol: CLAIM_PROTOCOL,
    };

    fn render(self) -> String {
        format!(
            "store-format {}\nclaim-protocol {}\n",
            self.store_format, self.claim_protocol
        )
    }

    fn parse(text: &str) -> Option<Contract> {
        let mut store_format = None;
        let mut claim_protocol = None;
        for line in text.lines() {
            match line.split_once(' ') {
                Some(("store-format", n)) => store_format = n.parse().ok(),
                Some(("claim-protocol", n)) => claim_protocol = n.parse().ok(),
                _ => {}
            }
        }
        Some(Contract {
            store_format: store_format?,
            claim_protocol: claim_protocol?,
        })
    }
}

/// The contract recorded at `root`: `None` when there is none, or only a crash's remains.
pub fn recorded(root: &Path) -> Result<Option<Contract>> {
    let path = root.join(RECORD);
    match fs::read_to_string(&path) {
        Ok(text) => Ok(Contract::parse(&text)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(crate::store::classify_io(
            format!("read {}", path.display()),
            error,
        )),
    }
}

/// Judges the Store at `root` against this build; `owner` when this process holds its owner
/// claim. Once per process per root: a root that agreed once stays agreed.
pub(crate) fn judge(root: &Path, owner: bool) -> Result<()> {
    static AGREED: Mutex<Vec<PathBuf>> = Mutex::new(Vec::new());
    let key = fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf());
    if AGREED.lock().unwrap().contains(&key) {
        return Ok(());
    }
    loop {
        match recorded(root)? {
            Some(theirs) if theirs == Contract::OURS => break,
            None => {
                // Only over a crash's remains does a write replace: two first openers race.
                if write(root, root.join(RECORD).exists())? {
                    break;
                }
            }
            Some(theirs) if theirs.store_format == STORE_FORMAT && owner => {
                write(root, true)?;
                break;
            }
            Some(theirs) => {
                return refuse(
                    Code::STORE_ERA,
                    format!(
                        "the Store at {} is store format {}, claim protocol {}; this TensorFS {} \
                         speaks store format {STORE_FORMAT}, claim protocol {CLAIM_PROTOCOL}. The \
                         cozy binary and the tensorfs wheel on one machine must agree.",
                        root.display(),
                        theirs.store_format,
                        theirs.claim_protocol,
                        crate::VERSION,
                    ),
                )
            }
        }
    }
    AGREED.lock().unwrap().push(key);
    Ok(())
}

/// Records this build's contract: in place of any other when `replace`, else only where none is
/// (false: another process recorded first).
fn write(root: &Path, replace: bool) -> Result<bool> {
    let io = |what: &str, error| crate::store::classify_io(what, error);
    let temp = root.join("tmp").join(format!(
        "format-{}-{}",
        std::process::id(),
        crate::meta::now_nanos_unique()
    ));
    let mut file = fs::File::create(&temp).map_err(|e| io("create store contract record", e))?;
    file.write_all(Contract::OURS.render().as_bytes())
        .map_err(|e| io("write store contract record", e))?;
    let target = root.join(RECORD);
    let placed = if replace {
        fs::rename(&temp, &target).map(|()| true)
    } else {
        let linked = fs::hard_link(&temp, &target);
        let _ = fs::remove_file(&temp);
        match linked {
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => return Ok(false),
            other => other.map(|()| true),
        }
    };
    placed.map_err(|e| io("place store contract record", e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_record_round_trips_and_tolerates_more_lines() {
        let text = format!("{}future 7\n", Contract::OURS.render());
        assert_eq!(Contract::parse(&text), Some(Contract::OURS));
        assert_eq!(Contract::parse("store-format 1\n"), None);
    }
}
