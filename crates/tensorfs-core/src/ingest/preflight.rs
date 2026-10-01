//! The conversion plan, decided from HEADERS, before a byte of payload moves (tfs-076).
//!
//! > *"why does conversion have to download all 200GBs of files and then refuse? shouldn't
//! > it refuse right away?"*
//!
//! It should. Three MiniMax-H3 runs on 2026-09-03/04 each moved **210,297,121,677 B** and
//! then refused on facts that fit in the 698,985 B of header those members carry:
//!
//! | run | moved | refused on | what decided it |
//! |---|---|---|---|
//! | 290 | 210.3 GB | `model_source_file_identity_invalid` | the member/object-id list |
//! | 294 | 210.3 GB | `DUPLICATE_KEY` | the member/object-id list |
//! | 309 | 210.3 GB | `carrier_header_cap` | the first EIGHT BYTES of one file |
//!
//! Run 309's refusal reads `header claims 8387229874382703227 B`, which is `b"{\n  \"met"` —
//! the opening of a JSON document taken for a little-endian u64. Eight bytes, discovered
//! after 210.3 GB and a rented pod.
//!
//! **Nothing here is a second opinion about anything.** The whole of this module is a
//! different way to obtain the SAME `CarrierInput` slice that `prepare_model_source` builds
//! from downloaded files, handed to the SAME [`source::plan_source_prepared`], once per
//! reviewed profile, exactly as [`source::prepare_model_source`] does it. A guard cannot be
//! present here and absent there, or drift out of agreement, because there is one guard.
//!
//! **A header is enough because a plan is a function of headers.** `durability.rs` states it
//! for the conversion journal's key — *"computed from the source headers with zero tensor
//! bytes read"* — and the session id this module reports is `session_of`, byte-identical to
//! the one the pod computes, because its inputs are the carrier header digest, `data_start`,
//! `file_len` and the tensor count. So a preflight does not merely resemble the pod's plan;
//! it names it.
//!
//! **How a header stands in for a 5 GiB file.** Every reader here checks a carrier's
//! declared tensor regions against the file's real length, so a header divorced from its
//! length is checkable against nothing. [`stage`] therefore writes the header bytes and then
//! extends the file to the member's full pinned length, leaving the payload a SPARSE HOLE:
//! the geometry arithmetic runs against the real declared runs at the real file length, for
//! zero bytes of disk and zero bytes of transfer. `vectors/h3-headers` is the same
//! construction, and `tests/sharded_cas.rs` has been reproducing run 309's exact refusal
//! from it in 0.16 s.
//!
//! **What this cannot decide, and must not pretend to.** A header says what the payload
//! CLAIMS to be. Only the payload says what it is. Every digest verification, every
//! `put_stream` admission, and the executed tensor schema are bytes-required and stay behind
//! the transfer — see [`Preflight`] and the issue's guard table. This module moves the
//! metadata-decidable refusals in front of the transfer and moves nothing else.

use std::fs;
use std::path::{Path, PathBuf};

use crate::canon::Value;
use crate::err::{refuse, Code, Refusal, Result};
use crate::providers::{MemberHead, Resolution};

use super::source::{self, CarrierInput, SourcePlan};

/// One component of a profile, as the plan assigns it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedComponent {
    pub component: String,
    pub member: Option<String>,
    /// True where the component is a PROJECTION of a shared single-file carrier rather than
    /// the whole of one.
    pub projected: bool,
}

/// One reviewed profile's conversion plan, as decided from headers alone.
///
/// Deliberately NOT a [`SourcePlan`]. A `SourcePlan` names files and can be executed; this
/// names members and cannot, because there are no files — only headers over sparse holes
/// that [`plan`] deletes before it returns. The two must not be confusable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfilePlan {
    pub profile: String,
    pub converter: String,
    pub target: String,
    /// The conversion journal's key, and IDENTICAL to the one the pod computes for this
    /// selection: `session_of` reads the carrier header digest, `data_start`, `file_len` and
    /// the tensor count, and nothing else. A preflight can therefore say in advance which
    /// journal a rented run will resume.
    pub session: String,
    pub components: Vec<PlannedComponent>,
    /// How many tensors the plan constructs.
    pub constructs: usize,
}

impl ProfilePlan {
    pub fn value(&self) -> Value {
        Value::obj(vec![
            (
                "components",
                Value::arr(
                    self.components
                        .iter()
                        .map(|component| {
                            let mut fields = vec![
                                ("component", Value::str(component.component.clone())),
                                ("projected", Value::Bool(component.projected)),
                            ];
                            if let Some(member) = &component.member {
                                fields.insert(1, ("member", Value::str(member.clone())));
                            }
                            Value::obj(fields)
                        })
                        .collect(),
                ),
            ),
            ("constructs", Value::uint(self.constructs as u64)),
            ("converter", Value::str(self.converter.clone())),
            ("profile", Value::str(self.profile.clone())),
            ("session", Value::str(self.session.clone())),
            ("target", Value::str(self.target.clone())),
        ])
    }
}

/// What a whole preflight decided, and what it cost.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Preflight {
    pub registry: String,
    pub registry_sha256: String,
    pub plans: Vec<ProfilePlan>,
    /// Members whose headers were read.
    pub members: usize,
    /// Bytes of header the decision was taken from.
    pub header_bytes: u64,
    /// Bytes of payload the transfer this preflight stands in front of would move.
    pub member_bytes: u64,
}

impl Preflight {
    pub fn value(&self) -> Value {
        Value::obj(vec![
            ("header_bytes", Value::uint(self.header_bytes)),
            ("member_bytes", Value::uint(self.member_bytes)),
            ("members", Value::uint(self.members as u64)),
            (
                "plans",
                Value::arr(self.plans.iter().map(ProfilePlan::value).collect()),
            ),
            ("registry", Value::str(self.registry.clone())),
            ("registry_sha256", Value::str(self.registry_sha256.clone())),
        ])
    }
}

/// The headers, on disk, as the carriers a plan reads.
///
/// Each member becomes one file: its real header bytes, extended to the member's real
/// pinned length so the region arithmetic has something true to check against, and the whole
/// payload a hole. An H3 staging that measures as 210,297,121,677 B holds 698,985 B.
pub struct Staged {
    root: PathBuf,
    carriers: Vec<CarrierInput>,
}

impl Staged {
    pub fn carriers(&self) -> &[CarrierInput] {
        &self.carriers
    }

    pub fn root(&self) -> &Path {
        &self.root
    }
}

impl Drop for Staged {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn io(what: impl AsRef<str>, error: std::io::Error) -> Refusal {
    Refusal {
        code: Code::IO_FAILED,
        detail: format!("{}: {error}", what.as_ref()),
    }
}

/// Write the headers into `root` as sparse carriers.
///
/// The staged NAME is a digest of the member, which matters for nothing: a member is the
/// only name a carrier has here, exactly as it is on a content-addressed pod, and the
/// `weight_map` of a sharded index is resolved through the carrier SET rather than through
/// any path. That is the property tfs-075 established and the one run 309 lacked.
pub fn stage(root: &Path, heads: &[MemberHead]) -> Result<Staged> {
    fs::create_dir_all(root).map_err(|error| io(format!("mkdir {}", root.display()), error))?;
    let mut carriers = Vec::with_capacity(heads.len());
    for head in heads {
        if head.head.len() as u64 > head.length {
            return refuse(
                Code::LENGTH_MISMATCH,
                format!(
                    "{}: {} header bytes over a {} B member",
                    head.member,
                    head.head.len(),
                    head.length
                ),
            );
        }
        let path = root.join(crate::sha256::hex_digest(head.member.as_bytes()));
        fs::write(&path, &head.head)
            .map_err(|error| io(format!("write {}", path.display()), error))?;
        if head.length > head.head.len() as u64 {
            // The payload, as a hole. Nothing on a header-only path reads it; every reader
            // on that path MEASURES it, which is the whole reason it is here.
            fs::OpenOptions::new()
                .write(true)
                .open(&path)
                .and_then(|file| file.set_len(head.length))
                .map_err(|error| io(format!("size {}", path.display()), error))?;
        }
        carriers.push(CarrierInput {
            member: Some(head.member.clone()),
            path,
        });
    }
    Ok(Staged {
        root: root.to_path_buf(),
        carriers,
    })
}

fn profile_plan(plan: &SourcePlan) -> ProfilePlan {
    ProfilePlan {
        profile: plan.profile.clone(),
        converter: plan.converter.clone(),
        target: plan.target.clone(),
        session: plan.session.clone(),
        components: plan
            .sources
            .iter()
            .map(|source| PlannedComponent {
                component: source.component.clone(),
                member: source.source_member.clone(),
                projected: source.projected,
            })
            .collect(),
        constructs: plan.construction_order.len(),
    }
}

/// The one profile these member heads plan under, chosen from their headers alone: several
/// matching reviewed profiles that plan identically are one answer; different plans refuse
/// `AMBIGUOUS_CLASSIFICATION` naming them, and none selects [`source::AS_IS`].
pub fn select_profile(
    registry_locator: &str,
    registry_bytes: &[u8],
    heads: &[MemberHead],
    scratch: &Path,
) -> Result<String> {
    let staged = stage(scratch, heads)?;
    source::plan_source(
        registry_locator,
        registry_bytes,
        staged.carriers(),
        None,
        None,
    )
    .map(|plan| plan.profile)
}

/// Decide the conversion plan for every requested profile, from headers alone.
///
/// One [`source::plan_source`] call per profile, over one carrier set — which is precisely
/// what [`source::prepare_model_source`] does on the pod, so this refuses on exactly what
/// that would refuse on and never on anything of its own invention.
///
/// `scratch` is created, used and removed; nothing survives the call.
pub fn plan(
    registry_locator: &str,
    registry_bytes: &[u8],
    heads: &[MemberHead],
    profiles: &[String],
    scratch: &Path,
) -> Result<Preflight> {
    if profiles.is_empty() {
        return refuse(
            Code::MISSING_FIELD,
            "a preflight names at least one reviewed source profile — a repository is not \
             a model, and an unnarrowed selection plans as nothing",
        );
    }
    let mut sorted: Vec<&str> = profiles.iter().map(String::as_str).collect();
    sorted.sort();
    sorted.dedup();
    let mut unique: Vec<MemberHead> = Vec::with_capacity(heads.len());
    for head in heads {
        match unique.iter().find(|kept| kept.member == head.member) {
            Some(kept) if kept == head => {}
            Some(_) => {
                return refuse(
                    Code::DUPLICATE_KEY,
                    format!("{} is listed twice with different heads", head.member),
                )
            }
            None => unique.push(head.clone()),
        }
    }
    let heads = unique.as_slice();
    let staged = stage(scratch, heads)?;
    let mut plans = Vec::with_capacity(profiles.len());
    for profile in &sorted {
        let plan = source::plan_source(
            registry_locator,
            registry_bytes,
            staged.carriers(),
            Some(profile),
            None,
        )?;
        plans.push(profile_plan(&plan));
    }
    Ok(Preflight {
        registry: registry_locator.to_string(),
        registry_sha256: format!("sha256:{}", crate::sha256::hex_digest(registry_bytes)),
        plans,
        members: heads.len(),
        header_bytes: heads.iter().map(|head| head.head.len() as u64).sum(),
        member_bytes: heads.iter().map(|head| head.length).sum(),
    })
}

/// The carriers one model takes from a resolved source, decided from headers: those the
/// reviewed `profiles` plan from or, none named, those of the profile the headers select.
/// Civitai companions join only when the version's primary alone plans as nothing reviewed
/// and they compose a reviewed model with it, so an unrecognized version stays its primary.
pub fn select_members(
    registry_locator: &str,
    registry_bytes: &[u8],
    resolution: &Resolution,
    profiles: &[String],
    read_heads: impl Fn(&Resolution) -> Result<Vec<MemberHead>>,
    scratch: &Path,
) -> Result<Vec<String>> {
    let carriers = |companions: bool| -> Vec<String> {
        resolution
            .members
            .iter()
            .filter(|row| row.carrier && (companions || !row.companion))
            .map(|row| row.member.clone())
            .collect()
    };
    let (own, all) = (carriers(false), carriers(true));
    let heads = read_heads(&resolution.select(&all)?)?;
    let plan_over = |carriers: &[String]| -> Result<(Vec<String>, bool)> {
        let members = resolution.select(carriers)?.members;
        let heads: Vec<MemberHead> = heads
            .iter()
            .filter(|head| members.iter().any(|row| row.member == head.member))
            .cloned()
            .collect();
        let profiles = match profiles {
            [] => vec![select_profile(
                registry_locator,
                registry_bytes,
                &heads,
                scratch,
            )?],
            named => named.to_vec(),
        };
        let planned = plan(registry_locator, registry_bytes, &heads, &profiles, scratch)?;
        let mut selected: Vec<String> = planned
            .plans
            .into_iter()
            .flat_map(|plan| plan.components.into_iter().filter_map(|c| c.member))
            .collect();
        selected.sort();
        selected.dedup();
        Ok((selected, !profiles.iter().any(|p| p == source::AS_IS)))
    };
    let first = plan_over(&own);
    if own.len() == all.len() || matches!(first, Ok((_, true))) {
        return first.map(|(selected, _)| selected);
    }
    match plan_over(&all) {
        Ok((selected, true)) if selected.iter().any(|member| own.contains(member)) => Ok(selected),
        _ => first.map(|(selected, _)| selected),
    }
}
