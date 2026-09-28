//! The conversion plan, decided from headers, before a byte of payload moves (tfs-076).
//!
//! > *"why does conversion have to download all 200GBs of files and then refuse? shouldn't
//! > it refuse right away?"*
//!
//! Every assertion here is against the REAL 48 members of `MiniMaxAI/MiniMax-H3` —
//! `vectors/h3-headers`, 689 KB of genuine metadata standing in for 210,297,121,677 B of
//! payload — served over a real loopback origin that answers real byte ranges. Three H3
//! runs on 2026-09-03/04 each moved the 210.3 GB and then refused; each refusal is taken
//! here for the header read instead, and the tests say how long that took.

use std::collections::BTreeMap;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use tensorfs_core::canon::{self, Value};
use tensorfs_core::err::Code;
use tensorfs_core::ids::ObjectRef;
use tensorfs_core::ingest::preflight;
use tensorfs_core::ingest::source::{self, CarrierInput, BUILTIN_REGISTRY, BUILTIN_REGISTRY_BYTES};
use tensorfs_core::limits;
use tensorfs_core::providers::{
    Endpoints, MemberHead, Provenance, Resolution, ResolvedMember, SourceUri,
};
use tensorfs_core::transport::{Anonymous, Deadline};

/// The 48 members' full lengths, summed. This is what a run moved before it refused.
const H3_MEMBER_BYTES: u64 = 210_297_121_677;
/// The same 48 members' headers, summed. This is what deciding costs instead.
const H3_HEADER_BYTES: u64 = 698_985;

const FL2VA: &str = "FL2VA/transformer/model.safetensors.index.json";
const REF2VA: &str = "Ref2VA/transformer/model.safetensors.index.json";
const DUAL: &str = "hf/minimax-h3/native-dual-bf16/1";
const SHARED: &str = "hf/minimax-h3/shared-bf16/1";

fn temporary(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "tensorfs-source-preflight-{name}-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}

fn headers_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../vectors/h3-headers")
        .canonicalize()
        .expect("vectors/h3-headers is checked in beside the crate")
}

/// `member`, `full_length` for all 48, read with the repo's own bounded JSON reader.
fn manifest() -> Vec<(String, u64)> {
    let bytes = fs::read(headers_dir().join("MANIFEST.json")).unwrap();
    let Value::Arr(rows) = canon::parse(&bytes, limits::DOC_MAX_BYTES).unwrap() else {
        panic!("MANIFEST.json is an array");
    };
    rows.iter()
        .map(|row| {
            let Value::Obj(fields) = row else {
                panic!("manifest row is an object")
            };
            let mut member = None;
            let mut length = None;
            for (key, value) in fields {
                match (key.as_str(), value) {
                    ("member", Value::Str(v)) => member = Some(v.clone()),
                    ("full_length", Value::Int(v)) => length = Some(*v as u64),
                    _ => {}
                }
            }
            (member.unwrap(), length.unwrap())
        })
        .collect()
}

/// The real header bytes of every H3 member, by member.
fn h3_bodies() -> BTreeMap<String, (Vec<u8>, u64)> {
    let source = headers_dir();
    manifest()
        .into_iter()
        .map(|(member, length)| {
            let head = fs::read(source.join(member.replace('/', "__"))).unwrap();
            (member, (head, length))
        })
        .collect()
}

fn h3_heads() -> Vec<MemberHead> {
    h3_bodies()
        .into_iter()
        .map(|(member, (head, length))| MemberHead {
            member,
            length,
            head,
        })
        .collect()
}

// ------------------------------------------------------------------ the origin

/// What the origin does with one member. The point of a fixture origin here is to be able
/// to lie in exactly the ways a real one does.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Serving {
    /// Honest 206 with `Content-Range`.
    Ranged,
    /// Answers 200 with the whole body and ignores the range, as some origins do.
    Unranged,
    /// Serves a `Content-Range` total that disagrees with what the selection pinned.
    LyingLength,
}

struct Origin {
    base: String,
    /// Bytes this origin actually put on the wire, for every request.
    served: Arc<AtomicUsize>,
}

/// A loopback origin over the real H3 header vectors, answering `Range` for real.
///
/// The declared full length of a member is its MANIFEST length — up to 5.2 GB — while the
/// bytes this holds are only the header. A range beyond the header is answered with zeros,
/// which is a fixture affordance nothing under test may use: every assertion below is that
/// the tested code never asked for one.
fn origin(bodies: BTreeMap<String, (Vec<u8>, u64)>, serving: Serving) -> Origin {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let served = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&served);
    std::thread::spawn(move || {
        for socket in listener.incoming() {
            let Ok(socket) = socket else { break };
            let bodies = bodies.clone();
            let counter = Arc::clone(&counter);
            std::thread::spawn(move || answer(socket, &bodies, serving, &counter));
        }
    });
    Origin { base, served }
}

fn answer(
    mut socket: TcpStream,
    bodies: &BTreeMap<String, (Vec<u8>, u64)>,
    serving: Serving,
    counter: &AtomicUsize,
) {
    let mut reader = BufReader::new(socket.try_clone().unwrap());
    let mut line = String::new();
    if reader.read_line(&mut line).is_err() || line.is_empty() {
        return;
    }
    let target = line.split_whitespace().nth(1).unwrap_or("/").to_string();
    let member = urldecode(target.trim_start_matches('/'));
    let mut range = None;
    loop {
        let mut header = String::new();
        if reader.read_line(&mut header).unwrap_or(0) == 0 {
            return;
        }
        let header = header.trim_end();
        if header.is_empty() {
            break;
        }
        if let Some(value) = header
            .to_ascii_lowercase()
            .strip_prefix("range:")
            .map(str::trim)
            .and_then(|value| value.strip_prefix("bytes=").map(str::to_string))
        {
            let (first, last) = value.split_once('-').unwrap();
            range = Some((
                first.trim().parse::<u64>().unwrap(),
                last.trim().parse::<u64>().unwrap(),
            ));
        }
    }
    let Some((head, length)) = bodies.get(&member) else {
        let _ = socket.write_all(b"HTTP/1.1 404 Not Found\r\ncontent-length: 0\r\n\r\n");
        return;
    };
    let total = match serving {
        Serving::LyingLength => length + 1,
        _ => *length,
    };
    let (status, body, extra) = match (serving, range) {
        // A real origin ignoring a range answers 200 with the WHOLE object's length. The
        // fixture states that length and sends only the header, which is safe precisely
        // because the code under test must drop a body it did not ask for.
        (Serving::Unranged, _) | (_, None) => {
            (200u16, head.clone(), format!("__length: {total}\r\n"))
        }
        (_, Some((first, last))) => {
            let last = last.min(total.saturating_sub(1));
            let want = (last + 1 - first) as usize;
            let mut body = vec![0u8; want];
            for (offset, byte) in body.iter_mut().enumerate() {
                let at = first as usize + offset;
                if at < head.len() {
                    *byte = head[at];
                }
            }
            (
                206u16,
                body,
                format!("content-range: bytes {first}-{last}/{total}\r\n"),
            )
        }
    };
    counter.fetch_add(body.len(), Ordering::Relaxed);
    let declared = extra
        .strip_prefix("__length: ")
        .and_then(|rest| rest.trim_end().parse::<u64>().ok());
    let head_line = match declared {
        Some(length) => {
            format!("HTTP/1.1 {status} X\r\ncontent-length: {length}\r\nconnection: close\r\n\r\n")
        }
        None => format!(
            "HTTP/1.1 {status} X\r\ncontent-length: {}\r\n{extra}connection: close\r\n\r\n",
            body.len()
        ),
    };
    let _ = socket.write_all(head_line.as_bytes());
    let _ = socket.write_all(&body);
    let _ = socket.flush();
}

fn urldecode(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' && index + 2 < bytes.len() {
            if let Ok(byte) = u8::from_str_radix(&value[index + 1..index + 3], 16) {
                out.push(byte);
                index += 3;
                continue;
            }
        }
        out.push(bytes[index]);
        index += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn hf() -> SourceUri {
    SourceUri::parse("hf://MiniMaxAI/MiniMax-H3@42ed227e42ed227e42ed227e42ed227e42ed227e").unwrap()
}

fn endpoints(base: &str) -> Endpoints {
    Endpoints {
        huggingface: base.to_string(),
        civitai: base.to_string(),
        allow_local: true,
    }
}

/// The 48-member resolution `tfs source resolve --source-profile` produces, with this
/// origin's URLs on it. The digests are of the header bytes, which is enough: nothing on a
/// header-only path verifies one, and that is itself part of the answer.
fn resolution(base: &str, bodies: &BTreeMap<String, (Vec<u8>, u64)>) -> Resolution {
    Resolution {
        canonical: "hf://MiniMaxAI/MiniMax-H3@42ed227e42ed227e42ed227e42ed227e42ed227e".into(),
        members: bodies
            .iter()
            .map(|(member, (head, length))| ResolvedMember {
                member: member.clone(),
                object: ObjectRef {
                    sha256: tensorfs_core::sha256::hex_digest(head),
                    length: *length,
                },
                url: format!("{base}/{member}"),
                provenance: Provenance::Declared,
                carrier: member.ends_with(".index.json"),
                requires: Vec::new(),
                companion: false,
            })
            .collect(),
        selection_sha256: "0".repeat(64),
    }
}

fn profiles() -> Vec<String> {
    vec![DUAL.to_string(), SHARED.to_string()]
}

// ------------------------------------------------------------------ the tests

/// THE ANSWER TO THE QUESTION. Every fact three H3 runs needed was in 689 KB of header,
/// and here those headers are read off a live origin by range, and the whole selection
/// reaches its verdict, with the 210,297,121,677 B of payload untouched.
#[test]
fn the_whole_h3_selection_reaches_its_verdict_from_headers_over_a_real_origin() {
    let bodies = h3_bodies();
    let origin = origin(bodies.clone(), Serving::Ranged);
    let selection = resolution(&origin.base, &bodies);
    let started = std::time::Instant::now();

    let read = tensorfs_core::providers::read_heads(
        &selection,
        &hf(),
        &endpoints(&origin.base),
        &Anonymous,
        Deadline::none(),
    )
    .expect("48 headers read by range");

    assert_eq!(read.heads.len(), 48, "48 verified members");
    assert!(
        read.unread.is_empty(),
        "every header read: {:?}",
        read.unread
    );
    assert_eq!(
        read.heads
            .iter()
            .map(|head| head.head.len() as u64)
            .sum::<u64>(),
        H3_HEADER_BYTES,
        "the plan is built from exactly the headers"
    );
    // One ask per member, plus one for the single member whose header does not fit the
    // speculative 64 KiB first ask. That is the whole network cost of the guard.
    assert_eq!(read.requests, 49, "48 members, one second ask");
    assert!(
        read.bytes < 4 << 20,
        "the speculative ask overshoots by design, but not by much: {} B",
        read.bytes
    );

    let scratch = temporary("h3-plan");
    let preflight = preflight::plan(
        BUILTIN_REGISTRY,
        BUILTIN_REGISTRY_BYTES,
        &read.heads,
        &profiles(),
        &scratch,
    )
    .expect("the reviewed H3 headers plan without payload transfer");
    assert_eq!(preflight.plans.len(), 2, "two reviewed profiles");
    assert_eq!(preflight.member_bytes, H3_MEMBER_BYTES);
    assert_eq!(preflight.header_bytes, H3_HEADER_BYTES);
    let dual = &preflight.plans[0];
    assert_eq!(dual.profile, DUAL);
    assert_eq!(dual.session.len(), 16, "the conversion journal's key");
    let members: Vec<&str> = dual
        .components
        .iter()
        .map(|component| component.member.as_deref().unwrap())
        .collect();
    assert_eq!(members, vec![FL2VA, REF2VA], "one object, two members");

    // The origin never sent a payload byte. Not "few" — none.
    assert_eq!(
        origin.served.load(Ordering::Relaxed) as u64,
        read.bytes,
        "the origin served bytes nobody asked for"
    );
    assert!(
        started.elapsed().as_secs() < 30,
        "a header-only verdict is seconds, not a rented pod: {:?}",
        started.elapsed()
    );
    assert!(!scratch.exists(), "the header staging is removed");
}

/// The banked H3 fingerprints must match the exact reviewed carrier headers. Changing
/// the tensor-schema projection without re-banking used to defer this refusal until
/// after 210 GB of transfer; the corpus now checks both identity and authorization.
#[test]
fn the_product_registry_matches_the_real_h3_carriers_it_names() {
    use tensorfs_core::ingest::fingerprint::{self, FingerprintRegistry};
    let root = temporary("reviewed-registry");
    let staged = preflight::stage(&root, &h3_heads()).unwrap();
    let set = source::CarrierSet::of(staged.carriers());
    let registry = FingerprintRegistry::parse(BUILTIN_REGISTRY_BYTES).unwrap();

    for (member, component) in [
        (FL2VA, "fl2va_dit"),
        (REF2VA, "ref2va_dit"),
        ("text_encoder/model.safetensors.index.json", "text_encoder"),
        ("audio_vae/diffusion_pytorch_model.safetensors", "audio_vae"),
        (
            "vae/diffusion_pytorch_model.safetensors.index.json",
            "video_vae",
        ),
    ] {
        let carrier = staged
            .carriers()
            .iter()
            .find(|carrier| carrier.member.as_deref() == Some(member))
            .unwrap();
        let (header, raw) = source::read_carrier(carrier, &set).unwrap();
        let observed = fingerprint::fingerprint(component, &header).unwrap();
        let banked: Vec<_> = registry
            .entries
            .iter()
            .filter(|entry| {
                entry.component == component && entry.keyset_digest == observed.keyset_digest
            })
            .collect();
        assert_eq!(banked.len(), 1, "{component} is banked once");
        let banked = banked[0];

        // Same carrier: the registry's own provenance says so, twice.
        assert_eq!(
            banked.provenance.source_sha256.as_deref(),
            Some(source::source_digest(&header, &raw).trim_start_matches("sha256:")),
            "{component}: the banked carrier identity is these very bytes"
        );
        assert!(
            banked
                .provenance
                .note
                .contains(&format!("header {} B", header.header_bytes)),
            "{component}: the banked header length is this header's"
        );
        // Same keys, and the same number of them.
        assert_eq!(banked.keyset_digest, observed.keyset_digest);
        assert_eq!(banked.logical_keys, observed.logical_keys as u64);

        assert_eq!(
            banked.tensor_schema_digest, observed.tensor_schema_digest,
            "{component}: the reviewed tensor-schema digest must match its carrier"
        );
    }
    drop(staged);
}

/// RUN 309, for 689 KB. `header claims 8387229874382703227 B` is `b"{\n  \"met"` — the
/// opening of a JSON document taken for a little-endian u64 header length. Eight bytes.
/// Run 309 found them after 48/48 members and 210.3 GB; here the same eight bytes refuse
/// off a header read, before anything is rented.
#[test]
fn run_309s_eight_bytes_refuse_from_the_header_read() {
    let mut bodies = h3_bodies();
    // A member NAMED as a tensor carrier whose bytes are the index document. Whatever
    // produces it — a mislabelled member, a provider serving the wrong object, run 309's
    // own staging that lost the name — the eight bytes decide it and they are free.
    let index = bodies[FL2VA].clone();
    bodies.insert(
        "FL2VA/transformer/model-00001-of-00013.safetensors".into(),
        index,
    );
    let origin = origin(bodies.clone(), Serving::Ranged);
    let selection = resolution(&origin.base, &bodies);

    let read = tensorfs_core::providers::read_heads(
        &selection,
        &hf(),
        &endpoints(&origin.base),
        &Anonymous,
        Deadline::none(),
    )
    .expect("headers still read");

    let scratch = temporary("run-309");
    let refusal = preflight::plan(
        BUILTIN_REGISTRY,
        BUILTIN_REGISTRY_BYTES,
        &read.heads,
        &profiles(),
        &scratch,
    )
    .expect_err("a JSON body under a safetensors member must refuse");

    assert_eq!(refusal.code, Code::CARRIER_HEADER_CAP);
    assert!(
        refusal.detail.contains("8387229874382703227"),
        "the refusal is run 309's, byte for byte: {}",
        refusal.detail
    );
    assert!(
        (origin.served.load(Ordering::Relaxed) as u64) < 4 << 20,
        "the refusal cost {} B; it must stay in the megabytes, against 210.3 GB",
        origin.served.load(Ordering::Relaxed)
    );
}

/// RUN 294: a repeated member decided a 210.3 GB run. The same member listed twice is one
/// member; only two DIFFERENT heads under one name is a fact about the list, and it is
/// decided from headers, before the transfer.
#[test]
fn a_repeated_member_merges_and_a_conflicting_one_refuses_before_the_transfer() {
    let mut heads = h3_heads();
    let repeat = heads[0].clone();
    heads.push(repeat);
    let merged = preflight::plan(
        BUILTIN_REGISTRY,
        BUILTIN_REGISTRY_BYTES,
        &heads,
        &profiles(),
        &temporary("duplicate"),
    )
    .expect("an identical repeated member is one member");
    assert_eq!(merged.members, h3_heads().len());

    let mut conflict = h3_heads();
    let mut other = conflict[0].clone();
    other.length += 1;
    conflict.push(other);
    let refusal = preflight::plan(
        BUILTIN_REGISTRY,
        BUILTIN_REGISTRY_BYTES,
        &conflict,
        &profiles(),
        &temporary("conflict"),
    )
    .expect_err("one member with two heads must refuse");
    assert_eq!(refusal.code, Code::DUPLICATE_KEY);
}

/// A selection short of a shard its own index requires, refused BY MEMBER, from headers.
/// The narrowing that dropped it is named, not the path it would have had.
#[test]
fn a_missing_shard_is_named_from_the_headers_alone() {
    let absent = "FL2VA/transformer/model-00007-of-00013.safetensors";
    let heads: Vec<MemberHead> = h3_heads()
        .into_iter()
        .filter(|head| head.member != absent)
        .collect();
    let scratch = temporary("missing-shard");
    let refusal = preflight::plan(
        BUILTIN_REGISTRY,
        BUILTIN_REGISTRY_BYTES,
        &heads,
        &[DUAL.to_string()],
        &scratch,
    )
    .expect_err("an index whose shard is not carried must refuse");
    assert_eq!(refusal.code, Code::MISSING_FIELD);
    assert!(
        refusal.detail.contains(absent),
        "the refusal names the missing MEMBER: {}",
        refusal.detail
    );
}

/// An unnarrowed preflight is refused rather than answered. A repository is not a model,
/// and a plan without a reviewed profile is a question with no subject.
#[test]
fn a_preflight_without_a_profile_refuses() {
    let scratch = temporary("no-profile");
    let refusal = preflight::plan(
        BUILTIN_REGISTRY,
        BUILTIN_REGISTRY_BYTES,
        &h3_heads(),
        &[],
        &scratch,
    )
    .expect_err("a preflight names at least one reviewed profile");
    assert_eq!(refusal.code, Code::MISSING_FIELD);
}

/// THE PROPERTY THE WHOLE THING RESTS ON: a header over a SPARSE HOLE and the same header
/// over the real payload are the same carrier, to the exact value the session id is built
/// from. So the session a preflight predicts is the session the pod computes, and a
/// preflight is not an approximation of the plan — it is the plan.
#[test]
fn a_sparse_header_and_a_whole_file_have_one_source_digest() {
    let root = temporary("sparse-parity");
    fs::create_dir_all(&root).unwrap();

    // A real, whole, tiny carrier: header plus every byte of the payload it declares.
    let header = br#"{"a":{"data_offsets":[0,16],"dtype":"F32","shape":[4]},"b":{"data_offsets":[16,48],"dtype":"F32","shape":[8]}}"#;
    let mut whole = (header.len() as u64).to_le_bytes().to_vec();
    whole.extend_from_slice(header);
    whole.extend_from_slice(&[7u8; 48]);
    let full = root.join("whole.safetensors");
    fs::write(&full, &whole).unwrap();

    // The same carrier as a preflight sees it: the header, and the payload as a hole.
    let staged = preflight::stage(
        &root.join("staging"),
        &[MemberHead {
            member: "component/whole.safetensors".into(),
            length: whole.len() as u64,
            head: whole[..whole.len() - 48].to_vec(),
        }],
    )
    .unwrap();

    let (from_file, file_raw) = source::read_carrier(
        &CarrierInput {
            member: Some("component/whole.safetensors".into()),
            path: full.clone(),
        },
        &source::CarrierSet::default(),
    )
    .unwrap();
    let (from_head, head_raw) =
        source::read_carrier(&staged.carriers()[0], &source::CarrierSet::default()).unwrap();

    assert_eq!(from_file, from_head, "one carrier, two ways of holding it");
    assert_eq!(
        source::source_digest(&from_file, &file_raw),
        source::source_digest(&from_head, &head_raw),
        "the session's own input must not know whether the payload is present"
    );
    assert!(
        fs::metadata(&staged.carriers()[0].path).unwrap().len() == whole.len() as u64,
        "the staged carrier measures as the whole member"
    );
    drop(staged);
    let _ = fs::remove_dir_all(&root);
}

/// An origin that will not serve a range is a real origin, and the honest verdict there is
/// UNDECIDED, never a refusal — and crucially, never a whole-object download either. The
/// prefix read drops a body it did not ask for rather than becoming the transfer it exists
/// to stand in front of.
#[test]
fn an_origin_that_refuses_ranges_leaves_the_plan_undecided_and_moves_nothing() {
    let bodies = h3_bodies();
    let origin = origin(bodies.clone(), Serving::Unranged);
    let selection = resolution(&origin.base, &bodies);

    let read = tensorfs_core::providers::read_heads(
        &selection,
        &hf(),
        &endpoints(&origin.base),
        &Anonymous,
        Deadline::none(),
    )
    .expect("an unrangeable origin is not an error");

    // The four index members are smaller than the ask, so a 200 delivers them whole and
    // they are read. The 44 tensor carriers are 5 GB objects and are left alone.
    assert_eq!(read.heads.len(), 4, "only the members that fit the ask");
    assert_eq!(read.unread.len(), 44, "every shard is undecided");
    assert!(
        read.bytes < 1 << 20,
        "an undecided read moved {} B",
        read.bytes
    );
}

/// A selection pinned to one length and an origin serving another is a fault in the
/// SELECTION. It is decidable in ONE request and no amount of transfer improves it, so it
/// refuses here rather than at the admission door 210 GB later.
#[test]
fn an_origin_that_contradicts_the_pinned_length_refuses_on_the_first_ask() {
    let bodies = h3_bodies();
    let origin = origin(bodies.clone(), Serving::LyingLength);
    let selection = resolution(&origin.base, &bodies);

    let refusal = tensorfs_core::providers::read_heads(
        &selection,
        &hf(),
        &endpoints(&origin.base),
        &Anonymous,
        Deadline::none(),
    )
    .expect_err("two opinions about one identity must not both stand");
    assert_eq!(refusal.code, Code::LENGTH_MISMATCH);
}

/// The staged carriers are named by MEMBER and nothing else — no extension, no directory
/// to be a sibling of — which is the property run 309 lacked and tfs-075 established. A
/// header-only plan resolves a sharded index's `weight_map` through the carrier SET.
#[test]
fn staged_headers_carry_no_name_but_the_member() {
    let root = temporary("naming");
    let staged = preflight::stage(&root, &h3_heads()).unwrap();
    for carrier in staged.carriers() {
        let name = carrier.path.file_name().unwrap().to_str().unwrap();
        assert_eq!(name.len(), 64, "a staged header is named by a digest");
        assert!(!name.contains('.'), "no extension for a path test to find");
        assert!(carrier.member.is_some(), "the member is the only name");
    }
    // ...and the two byte-identical H3 indexes are still two members.
    let members: Vec<&str> = staged
        .carriers()
        .iter()
        .filter_map(|carrier| carrier.member.as_deref())
        .filter(|member| member.ends_with(".index.json"))
        .collect();
    assert_eq!(members.len(), 4);
    drop(staged);
    assert!(!root.exists(), "staging is removed with the handle");
}

/// A `--json` shape a caller can gate on. `session` is the field that matters: it is the
/// conversion journal's key and the pod will compute the same one.
#[test]
fn the_preflight_document_names_the_session_and_the_accounting() {
    let document = preflight::Preflight {
        registry: BUILTIN_REGISTRY.into(),
        registry_sha256: format!("sha256:{}", "0".repeat(64)),
        plans: vec![preflight::ProfilePlan {
            profile: DUAL.into(),
            converter: "h3.native/2".into(),
            target: "fl2va_dit=plain/1,ref2va_dit=plain/1".into(),
            session: "0123456789abcdef".into(),
            components: vec![preflight::PlannedComponent {
                component: "fl2va_dit".into(),
                member: Some(FL2VA.into()),
                projected: false,
            }],
            constructs: 535,
        }],
        members: 48,
        header_bytes: H3_HEADER_BYTES,
        member_bytes: H3_MEMBER_BYTES,
    };
    let written = String::from_utf8(canon::write(&document.value())).unwrap();
    for field in [
        "\"header_bytes\"",
        "\"member_bytes\"",
        "\"session\"",
        "\"profile\"",
        "\"converter\"",
        "\"registry_sha256\"",
        "\"member\"",
    ] {
        assert!(written.contains(field), "{field} absent from {written}");
    }
    assert!(written.contains(&H3_MEMBER_BYTES.to_string()));
    assert!(written.contains(FL2VA));
}

/// A prefix read reads a prefix. Trivially stated, and worth a test because the failure
/// mode is silent: a `read_capped` that refused, or a 200 arm that streamed, would turn the
/// guard into the transfer.
#[test]
fn a_prefix_read_asks_for_a_prefix_and_takes_no_more() {
    let bodies = h3_bodies();
    let origin = origin(bodies.clone(), Serving::Ranged);
    let member = "FL2VA/transformer/model-00001-of-00013.safetensors";
    let (head, length) = &bodies[member];

    let prefix = tensorfs_core::transport::fetch_prefix(
        &format!("{}/{member}", origin.base),
        &endpoints(&origin.base).policy(&hf()),
        &Anonymous,
        8,
        Deadline::none(),
        &tensorfs_core::transport::Ledger::new(),
    )
    .unwrap();

    assert_eq!(prefix.bytes.as_deref(), Some(&head[..8]));
    assert_eq!(prefix.total, Some(*length));
    assert_eq!(origin.served.load(Ordering::Relaxed), 8);
}

/// A member whose declared header exceeds the 16 MiB cap is NOT asked for again. Run 309's
/// eight bytes claim 8.4 quintillion; a second ask for it would be the whole object, which
/// is the transfer this stands in front of.
#[test]
fn a_header_over_the_cap_costs_exactly_one_request() {
    let mut bodies = h3_bodies();
    let member = "FL2VA/transformer/model-00001-of-00013.safetensors".to_string();
    let length = bodies[&member].1;
    bodies.insert(member.clone(), (b"{\n  \"met".to_vec(), length));
    let origin = origin(bodies.clone(), Serving::Ranged);
    let selection = Resolution {
        members: resolution(&origin.base, &bodies)
            .members
            .into_iter()
            .filter(|row| row.member == member)
            .collect(),
        ..resolution(&origin.base, &bodies)
    };

    let read = tensorfs_core::providers::read_heads(
        &selection,
        &hf(),
        &endpoints(&origin.base),
        &Anonymous,
        Deadline::none(),
    )
    .unwrap();
    assert_eq!(read.requests, 1, "a capped header is not re-asked");
    assert!(read.bytes <= 64 << 10);
    assert!(
        u64::from_le_bytes(read.heads[0].head[..8].try_into().unwrap())
            == 8_387_229_874_382_703_227,
        "the eight bytes reached the planner intact"
    );
}
