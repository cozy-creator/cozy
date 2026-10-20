//! Read-only integrity inspection of the selected local object closure.

use std::collections::HashSet;
use std::io::{self, Read};

use crate::failure::{Failure, Result};
use crate::format::ObjectRef;
use crate::sha256::{hex, Sha256};
use crate::store::{Checkpoint, Error};

/// Local objects absent from disk or present with the wrong length/content.
/// Verification never fetches, repairs or removes objects.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VerificationReport {
    pub missing: Vec<ObjectRef>,
    pub corrupt: Vec<ObjectRef>,
}

impl VerificationReport {
    pub fn is_ok(&self) -> bool {
        self.missing.is_empty() && self.corrupt.is_empty()
    }
}

pub(crate) fn verify(
    checkpoint: &Checkpoint,
    objects: impl IntoIterator<Item = ObjectRef>,
) -> Result<VerificationReport> {
    let mut report = VerificationReport::default();
    let mut seen = HashSet::new();
    let mut buffer = vec![0; 1 << 20];
    // A selected payload is meaningful only with the header that names it. Read the
    // actual opened header file, even for an empty selection, not a cached duplicate.
    for object in std::iter::once(checkpoint.id().clone()).chain(objects) {
        if !seen.insert((object.sha256.clone(), object.length)) {
            continue;
        }
        inspect(checkpoint, &object, &mut report, &mut buffer)?;
    }
    Ok(report)
}

fn inspect(
    checkpoint: &Checkpoint,
    object: &ObjectRef,
    report: &mut VerificationReport,
    buffer: &mut [u8],
) -> Result<()> {
    let file = match checkpoint.verification_file(object) {
        Ok(file) => file,
        Err(Error::Absent(_)) => {
            report.missing.push(object.clone());
            return Ok(());
        }
        Err(Error::Io { source, .. }) if source.kind() == io::ErrorKind::NotFound => {
            report.missing.push(object.clone());
            return Ok(());
        }
        Err(Error::Mismatch { .. }) => {
            report.corrupt.push(object.clone());
            return Ok(());
        }
        Err(error) => return Err(error.into()),
    };
    if file.metadata()?.len() != object.length {
        report.corrupt.push(object.clone());
        return Ok(());
    }
    let mut file = file.take(object.length.saturating_add(1));
    let mut hash = Sha256::new();
    let mut length = 0;
    loop {
        match file.read(buffer) {
            Ok(0) => break,
            Ok(n) => {
                length += n as u64;
                hash.update(&buffer[..n]);
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => {
                let mut failure = Failure::from(error);
                failure.detail = format!("verify {}: {}", object.id(), failure.detail);
                return Err(failure);
            }
        }
    }
    if length != object.length || hex(&hash.finish()) != object.sha256 {
        report.corrupt.push(object.clone());
    }
    Ok(())
}
