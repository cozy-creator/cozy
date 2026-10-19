//! The machine's half of an executor's fetch channel (the executor's is `tensorfs.fetch`).
//!
//! One connected stream per attempt, framed as every executor channel is (`crate::frame`). The
//! peer names checkpoints by manifest, only those its [`Grant`] selects, and objects by digest
//! and length; it never sees a Hub, a credential or a path. Each request is a JSON frame
//! `{"id","op","manifest","objects":[{"sha256","length"}]}` (`schedule` adds `rank` and
//! `behind`), answered `{"id","result"}` or `{"id","error","detail"}`:
//! - `schedule` wants the objects next (this run, `rank`), or behind, until the channel ends;
//! - `wait` wants them now and answers once they are here;
//! - `present` answers, for each, whether it is here.
//!
//! Waits answer out of order. Everything the channel wants ends when its peer closes it.
use platform::ipc::LocalStream;
use std::{
    sync::{mpsc, Arc},
};

use serde_json::{json, Map, Value};
use tensorfs::{
    err::{refuse, Code, Refusal, Result},
    format::ObjectRef,
};

use super::{priority, Machine, Remote};
use crate::{
    frame::{Channel, Frame},
    stop::Stop,
};

/// What one attempt's channel may fetch.
pub struct Grant {
    /// The run it serves: its schedules order by this run's acceptance.
    pub run: u64,
    /// The checkpoints the attempt selected, by manifest, and where each comes from.
    pub checkpoints: Vec<(ObjectRef, Remote)>,
    /// Ends the channel's waits. The channel cancels it when its peer closes.
    pub stop: Stop,
}

/// Serves `stream` until its peer closes it.
pub fn serve(machine: &Arc<Machine>, grant: Grant, stream: LocalStream) -> Result<()> {
    let io = |e: std::io::Error| Refusal { code: Code::TRANSFER_FAILED, detail: format!("fetch channel: {e}") };
    let writer = stream.try_clone().map_err(io)?;
    let (replies, outbox) = mpsc::channel::<Value>();
    let sender = std::thread::Builder::new()
        .name("fetch-replies".into())
        .spawn(move || {
            let mut channel = Channel(writer);
            for reply in outbox {
                if channel.send(&reply).is_err() {
                    break;
                }
            }
        })
        .map_err(io)?;
    let mut receiver = Channel(stream);
    let mut held = vec![];
    let served = loop {
        let request = match receiver.receive() {
            Ok(Some(Frame::Json(request))) => request,
            Ok(None) => break Ok(()),
            Ok(Some(Frame::Bytes(_))) => break refuse(Code::WRONG_TYPE, "a fetch request is JSON"),
            Err(error) => break Err(io(error)),
        };
        let id = request.get("id").and_then(Value::as_u64).unwrap_or(0);
        let answered = parse(&request, &grant).and_then(|parsed| match parsed.op.as_str() {
            "schedule" => {
                let at = match parsed.behind {
                    true => priority::BEHIND,
                    false => priority::next(grant.run, parsed.rank),
                };
                held.push(machine.want(parsed.remote, parsed.objects, at)?);
                Ok(Some(Value::Null))
            }
            "present" => {
                let store = machine.store();
                Ok(Some(parsed.objects.iter().map(|o| Value::Bool(store.reuse(o).unwrap_or(false))).collect()))
            }
            _ => {
                let (machine, remote, replies, stop) = (machine.clone(), parsed.remote.clone(), replies.clone(), grant.stop.clone());
                let objects = parsed.objects;
                std::thread::Builder::new()
                    .name("fetch-wait".into())
                    .spawn(move || {
                        let _ = replies.send(reply(id, machine.objects(&remote, &objects, priority::NOW, &stop).map(|_| Value::Null)));
                    })
                    .map_err(io)?;
                Ok(None)
            }
        });
        match answered {
            Ok(Some(result)) => {
                let _ = replies.send(reply(id, Ok(result)));
            }
            Ok(None) => {}
            Err(refusal) => {
                let _ = replies.send(reply(id, Err(refusal)));
            }
        }
    };
    grant.stop.cancel();
    for want in held {
        want.abort();
    }
    drop(replies);
    let _ = sender.join();
    served
}

fn reply(id: u64, answer: Result<Value>) -> Value {
    match answer {
        Ok(result) => json!({"id": id, "result": result}),
        Err(refusal) => {
            let detail: String = refusal.detail.chars().take(1024).collect();
            json!({"id": id, "error": refusal.code.as_str(), "detail": detail})
        }
    }
}

/// One request, checked against the grant.
struct Parsed<'g> {
    op: String,
    remote: &'g Remote,
    objects: Vec<ObjectRef>,
    rank: u32,
    behind: bool,
}

fn parse<'g>(request: &Value, grant: &'g Grant) -> Result<Parsed<'g>> {
    let Value::Object(fields) = request else {
        return refuse(Code::WRONG_TYPE, "a fetch request is an object");
    };
    let text = |key: &str| -> Result<&str> {
        match fields.get(key) {
            Some(Value::String(text)) => Ok(text),
            Some(_) => refuse(Code::WRONG_TYPE, format!("{key} is a string")),
            None => refuse(Code::MISSING_FIELD, format!("a fetch request has no {key:?}")),
        }
    };
    let op = text("op")?.to_string();
    let allowed: &[&str] = match op.as_str() {
        "schedule" => &["id", "op", "manifest", "objects", "rank", "behind"],
        "wait" | "present" => &["id", "op", "manifest", "objects"],
        _ => return refuse(Code::SOURCE_NOT_ALLOWED, format!("{op:?} is not a fetch operation")),
    };
    if let Some(key) = fields.keys().find(|k| !allowed.contains(&k.as_str())) {
        return refuse(Code::UNKNOWN_FIELD, format!("fetch {op}: unknown field {key:?}"));
    }
    let manifest = text("manifest")?;
    let remote = grant
        .checkpoints
        .iter()
        .find(|(m, _)| m.sha256 == manifest.trim_start_matches("sha256:"))
        .map(|(_, remote)| remote)
        .ok_or_else(|| Refusal { code: Code::SOURCE_NOT_ALLOWED, detail: format!("{manifest} is not a checkpoint this attempt selected") })?;
    let Some(Value::Array(rows)) = fields.get("objects") else {
        return refuse(Code::MISSING_FIELD, "a fetch request has an \"objects\" array");
    };
    let objects = rows
        .iter()
        .map(|row| match row {
            Value::Object(row) => serde_json::from_value::<ObjectRef>(Value::Object(only(row))).map_err(|e| Refusal {
                code: Code::WRONG_TYPE,
                detail: format!("an object is {{\"sha256\",\"length\"}}: {e}"),
            }),
            _ => refuse(Code::WRONG_TYPE, "an object is {\"sha256\",\"length\"}"),
        })
        .collect::<Result<_>>()?;
    let rank = match fields.get("rank") {
        Some(rank) => rank
            .as_u64()
            .and_then(|rank| u32::try_from(rank).ok())
            .ok_or_else(|| Refusal { code: Code::NUMBER_RANGE, detail: "rank is an integer below 2^32".into() })?,
        None => 0,
    };
    let behind = matches!(fields.get("behind"), Some(Value::Bool(true)));
    Ok(Parsed { op, remote, objects, rank, behind })
}

fn only(row: &Map<String, Value>) -> Map<String, Value> {
    row.iter().filter(|(k, _)| matches!(k.as_str(), "sha256" | "length")).map(|(k, v)| (k.clone(), v.clone())).collect()
}
