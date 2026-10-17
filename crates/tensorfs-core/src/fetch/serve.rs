//! The owner's half of an executor's fetch channel; the executor's is `tensorfs.fetch`.
//!
//! One connected stream per attempt, framed as the derived channels are. The peer names
//! checkpoints by manifest, only those its [`Grant`] selects, and objects by digest and
//! length; it never sees a hub, a credential or a path. Each request is a JSON frame
//! `{"id","op","manifest","objects":[{"sha256","length"}]}` (`schedule` adds `rank` and
//! `behind`), answered `{"id","result"}` or `{"id","error","detail"}`:
//! - `schedule` wants the objects at Next (this run, `rank`), or Behind;
//! - `wait` raises them to Now and answers once they are present;
//! - `present` answers, for each, whether it is here.
//!
//! Waits answer out of order. The channel's tickets end when its peer closes it.

use std::collections::HashMap;
use crate::platform::ipc::LocalStream;
use std::sync::{Arc, Mutex, PoisonError};

use crate::derived::channel::{array, num, obj, reference, text, uint, Channel, Frame};
use crate::err::{refuse, Code, Refusal, Result};
use crate::ids::ObjectRef;
use crate::jcs::Json;
use crate::transport::PullCancellation;

use super::{Access, Fetcher, Origin, Priority, Ticket};

/// What one attempt's channel may fetch.
pub struct Grant {
    /// The run it serves: its schedules order by this run's acceptance.
    pub run: u64,
    /// The checkpoints the attempt selected, and the access each is fetched under.
    pub origins: Vec<(Origin, Arc<dyn Access>)>,
    /// Ends the channel's waits. The channel stops it when its peer closes.
    pub stop: PullCancellation,
}

/// Serve `stream` until its peer closes it.
pub fn serve(fetcher: &Fetcher, grant: Grant, stream: LocalStream) -> Result<()> {
    let io = |e: std::io::Error| Refusal {
        code: Code::TRANSFER_FAILED,
        detail: format!("fetch channel: {e}"),
    };
    let sender = Mutex::new(Channel(stream.try_clone().map_err(io)?));
    let mut receiver = Channel(stream);
    let reply = |id: u64, answer: Result<Json>| -> Result<()> {
        let frame = match answer {
            Ok(result) => obj(vec![("id", num(id)), ("result", result)])?,
            Err(refusal) => {
                let detail: String = refusal.detail.chars().take(1024).collect();
                obj(vec![
                    ("id", num(id)),
                    ("error", Json::Str(refusal.code.as_str().into())),
                    ("detail", Json::Str(detail)),
                ])?
            }
        };
        sender.lock().unwrap_or_else(PoisonError::into_inner).send(&frame)
    };
    let mut scheduled: Vec<Ticket> = vec![];
    let readers: Mutex<HashMap<String, Arc<Ticket>>> = Mutex::default();
    let served = std::thread::scope(|scope| {
        // The channel's waits end with it, by return or by unwind alike: the scope never
        // waits on a wait whose objects may never come.
        let _stop = Stop(&grant.stop);
        loop {
            let request = match receiver.receive() {
                Ok(Some(Frame::Json(request))) => request,
                Ok(None) => break Ok(()),
                Ok(Some(Frame::Bytes(_))) => break refuse(Code::WRONG_TYPE, "a fetch request is JSON"),
                Err(refusal) => break Err(refusal),
            };
            let parsed = Parsed::of(&request, &grant);
            let id = parsed.as_ref().map_or_else(|(id, _)| *id, |p| p.id);
            let answered =
                parsed
                    .map_err(|(_, refusal)| refusal)
                    .and_then(|parsed| match parsed.op.as_str() {
                        "schedule" => {
                            let at = match parsed.behind {
                                true => Priority::Behind,
                                false => Priority::Next {
                                    run: grant.run,
                                    rank: parsed.rank,
                                },
                            };
                            scheduled.push(fetcher.want(
                                parsed.origin,
                                Arc::clone(parsed.access),
                                &parsed.objects,
                                at,
                            ));
                            Ok(Some(Json::Null))
                        }
                        "present" => {
                            let store = fetcher.store();
                            let present = (parsed.objects.iter())
                                .map(|o| {
                                    Json::Bool(
                                        store.contains(&o.sha256) || store.manifest_path(&o.sha256).is_file(),
                                    )
                                })
                                .collect();
                            Ok(Some(Json::Arr(present)))
                        }
                        _ => {
                            let ticket = Arc::clone(
                                readers
                                    .lock()
                                    .unwrap_or_else(PoisonError::into_inner)
                                    .entry(parsed.origin.manifest.sha256.clone())
                                    .or_insert_with(|| {
                                        Arc::new(fetcher.want(
                                            parsed.origin,
                                            Arc::clone(parsed.access),
                                            &[],
                                            Priority::Now,
                                        ))
                                    }),
                            );
                            let (reply, stop) = (&reply, &grant.stop);
                            let objects = parsed.objects;
                            let wait = move || reply(id, ticket.wait(&objects, stop).map(|()| Json::Null));
                            // A thread the container refuses: this wait is answered here, first.
                            match std::thread::Builder::new().spawn_scoped(scope, wait.clone()) {
                                Ok(_) => Ok(None),
                                Err(_) => wait().map(|()| None),
                            }
                        }
                    });
            let sent = match answered {
                Ok(Some(result)) => reply(id, Ok(result)),
                Ok(None) => Ok(()),
                Err(refusal) => reply(id, Err(refusal)),
            };
            if let Err(refusal) = sent {
                break Err(refusal);
            }
        }
    });
    drop(scheduled);
    served
}

/// Cancels a channel's waits as it ends.
struct Stop<'a>(&'a PullCancellation);

impl Drop for Stop<'_> {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

/// One request, checked against the grant.
struct Parsed<'g> {
    id: u64,
    op: String,
    origin: &'g Origin,
    access: &'g Arc<dyn Access>,
    objects: Vec<ObjectRef>,
    rank: u32,
    behind: bool,
}

impl<'g> Parsed<'g> {
    fn of(request: &Json, grant: &'g Grant) -> std::result::Result<Parsed<'g>, (u64, Refusal)> {
        let Json::Obj(pairs) = request else {
            return Err((
                0,
                Refusal {
                    code: Code::WRONG_TYPE,
                    detail: "a fetch request is an object".into(),
                },
            ));
        };
        let field = |key: &str| pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v);
        let id = field("id")
            .map_or(Ok(0), |id| uint("id", id))
            .map_err(|r| (0, r))?;
        let parsed = (|| {
            let op = text("op", field("op").ok_or_else(|| missing("op"))?)?.to_string();
            let allowed: &[&str] = match op.as_str() {
                "schedule" => &["id", "op", "manifest", "objects", "rank", "behind"],
                "wait" | "present" => &["id", "op", "manifest", "objects"],
                _ => {
                    return refuse(
                        Code::SOURCE_NOT_ALLOWED,
                        format!("{op:?} is not a fetch operation"),
                    )
                }
            };
            if let Some((key, _)) = pairs.iter().find(|(k, _)| !allowed.contains(&k.as_str())) {
                return refuse(Code::UNKNOWN_FIELD, format!("fetch {op}: unknown field {key:?}"));
            }
            let manifest = text("manifest", field("manifest").ok_or_else(|| missing("manifest"))?)?;
            let (origin, access) = (grant.origins.iter())
                .find(|(origin, _)| origin.manifest.id() == manifest || origin.manifest.sha256 == manifest)
                .map(|(origin, access)| (origin, access))
                .ok_or_else(|| Refusal {
                    code: Code::SOURCE_NOT_ALLOWED,
                    detail: format!("{manifest} is not a checkpoint this attempt selected"),
                })?;
            let mut objects = vec![];
            for row in array("objects", field("objects").ok_or_else(|| missing("objects"))?)? {
                let Json::Obj(row) = row else {
                    return refuse(Code::WRONG_TYPE, "an object is {\"sha256\",\"length\"}");
                };
                let get = |key: &str| {
                    row.iter()
                        .find(|(k, _)| k == key)
                        .map(|(_, v)| v)
                        .ok_or_else(|| missing(key))
                };
                objects.push(reference(
                    text("sha256", get("sha256")?)?,
                    uint("length", get("length")?)?,
                )?);
            }
            let rank = match field("rank") {
                Some(rank) => u32::try_from(uint("rank", rank)?).map_err(|_| Refusal {
                    code: Code::NUMBER_RANGE,
                    detail: "rank exceeds 2^32".into(),
                })?,
                None => 0,
            };
            let behind = matches!(field("behind"), Some(Json::Bool(true)));
            Ok(Parsed {
                id,
                op,
                origin,
                access,
                objects,
                rank,
                behind,
            })
        })();
        parsed.map_err(|refusal| (id, refusal))
    }
}

fn missing(key: &str) -> Refusal {
    Refusal {
        code: Code::MISSING_FIELD,
        detail: format!("a fetch request has no {key:?}"),
    }
}
