//! The source channel: one granted checkpoint's files and bytes (module doc of `derive`).
use platform::ipc::LocalStream;
use std::io;

use serde::Deserialize;
use serde_json::json;
use tensorfs::format::ObjectRef;
use tensorfs::store2::Store;

use super::{answer, ensure, open, parse, refuse, refused, send, serve_read, Answer, Hooks};
use crate::frame::{Channel, Frame};

#[derive(Deserialize)]
struct FileOut {
    path: String,
}

/// Serves `source` until the executor closes the channel or its attempt ends.
pub fn serve_source(store: &Store, source: &ObjectRef, stream: LocalStream, hooks: &Hooks) -> io::Result<()> {
    let mut channel = Channel(stream);
    let checkpoint = match open(store, source, hooks) {
        Ok(checkpoint) => checkpoint,
        Err(refusal) => {
            let _ = refuse(&mut channel, &refusal);
            return Err(refusal.into());
        }
    };
    loop {
        let request = match channel.receive()? {
            None => return Ok(()),
            Some(Frame::Json(request)) => request,
            Some(Frame::Bytes(_)) => {
                let refusal = refused("invalid_request", "expected an operation");
                let _ = refuse(&mut channel, &refusal);
                return Err(refusal.into());
            }
        };
        if !(hooks.current)() {
            let refusal = refused("source_closed", "the attempt is no longer current");
            let _ = refuse(&mut channel, &refusal);
            return Err(refusal.into());
        }
        match request["op"].as_str().unwrap_or_default() {
            "inspect" => answer(&mut channel, json!({ "manifest": source, "files": checkpoint.manifest().files }))?,
            "read" => serve_read(&mut channel, store, |_| Ok(&checkpoint), &request, hooks)?,
            "file" => {
                let found = (|| -> Answer<ObjectRef> {
                    let r: FileOut = parse(&request)?;
                    let object = checkpoint.manifest().files.get(&r.path).cloned();
                    let object = object.ok_or_else(|| refused("absent", format!("no file {:?}", r.path)))?;
                    ensure(store, [&object], hooks)?;
                    Ok(object)
                })();
                match found {
                    Ok(object) => {
                        answer(&mut channel, json!({ "length": object.length }))?;
                        send(&mut channel, store, &[(object.clone(), 0, object.length)])?;
                    }
                    Err(refusal) => refuse(&mut channel, &refusal)?,
                }
            }
            other => refuse(&mut channel, &refused("invalid_request", format!("{other:?} is not a source operation")))?,
        }
    }
}
