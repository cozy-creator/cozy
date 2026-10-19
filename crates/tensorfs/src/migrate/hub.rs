//! A released lane's v1 manifest and header from a Hub, through its closure and presign calls
//! (anonymous: released lanes need no credential), into a directory as `<sha256>` files. This
//! is the header-only fetch a migration plan needs; no part moves.

use std::fs;
use std::path::Path;

use crate::failure::{fail, Kind, Result};
use crate::format::ObjectRef;
use crate::transport::{self, Anonymous, Client, Deadline, Ledger, SourcePolicy};

pub struct Fetched {
    pub release: String,
    pub manifest: ObjectRef,
    /// Manifest and header bytes fetched.
    pub bytes: u64,
}

/// `refspec` (`org/name[@release]`) at `lane`, from the Hub at `origin`, into `dir`.
pub fn headers(origin: &str, refspec: &str, lane: &str, dir: &Path) -> Result<Fetched> {
    let api = transport::hub_policy(origin, &SourcePolicy::default());
    // Presigned objects are held to their digests: no host vouches for them.
    let objects = SourcePolicy {
        any_host: true,
        ..SourcePolicy::default()
    };
    let (client, ledger) = (Client::new(), Ledger::new());
    let closure = transport::closure(
        &client,
        origin,
        refspec,
        lane,
        "",
        &Anonymous,
        &api,
        Deadline::none(),
        &ledger,
    )?;
    let get = |o: &ObjectRef| -> Result<Vec<u8>> {
        let digests = [o.sha256.clone()];
        let urls = transport::presign(
            &client,
            origin,
            &digests,
            &Anonymous,
            &api,
            Deadline::none(),
            &ledger,
        )?;
        let mut bytes = Vec::new();
        let url = &urls.urls[0].1;
        transport::get_into(
            url,
            &objects,
            &Anonymous,
            0,
            None,
            &mut bytes,
            Deadline::none(),
            &ledger,
        )?;
        if ObjectRef::of(&bytes) != *o {
            return fail(
                Kind::Integrity,
                format!("{refspec} {lane}: {} is not its digest", o.sha256),
            );
        }
        let path = dir.join(&o.sha256);
        fs::write(&path, &bytes).or_else(|e| fail(Kind::Io, format!("{}: {e}", path.display())))?;
        Ok(bytes)
    };
    let manifest = ObjectRef {
        sha256: closure.manifest.sha256.clone(),
        length: closure.manifest.length,
    };
    let Some(header) = super::v1::manifest(&get(&manifest)?)?.header else {
        return fail(
            Kind::Unsupported,
            format!("{refspec} {lane}: the manifest has no header"),
        );
    };
    get(&header)?;
    Ok(Fetched {
        release: closure.release,
        bytes: manifest.length + header.length,
        manifest,
    })
}
