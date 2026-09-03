//! Hub grading — the ONE grading layer, over `check_compatibility`'s two-valued report.
//!
//! `COMPATIBLE -> CONVERTIBLE -> PRODUCIBLE -> INCOMPATIBLE`, decided from headers alone,
//! pre-download and before any pod money, with both sides named.
//!
//! Two laws this file exists to hold:
//!
//! 1. **A registration RUNS its round trip.** CONVERTIBLE names a REGISTERED lossless
//!    mapping, and registering one executes A -> B -> A over a corpus here, in this code.
//!    An equivalence predicate supplied by the party being policed is not a fence. A
//!    re-quantization structurally cannot register: the round trip is not byte-identical,
//!    so `register` refuses `ROUND_TRIP_LOSSY` and no amount of asserting changes it.
//!    Where the source dtype is 16 bits or narrower the corpus is the COMPLETE domain —
//!    all 65,536 bit patterns — so the claim is proved rather than sampled.
//!
//! 2. **The remedy derives from the TARGET, never the observation.** This is the v1
//!    fp8-offered-as-plain-cast bug class: a verdict that reads what it sees and proposes
//!    it back gets every name, dtype and shape right and every number wrong. A `Producible`
//!    here carries the required encoding/dtype from the CONTRACT side of the report.
//!
//! And the recorded v1 recovery, now hub law: a cast-only mismatch grading flat
//! INCOMPATIBLE is forbidden over-constraint. `guard_over_constraint` is that law with an
//! exit code.
//!
//! Two questions, two functions. `grade` answers PREFLIGHT — "can this run" — off the first
//! blocking fact, which is all one fact can answer. `grade_all` answers RANKING — "how bad
//! is this" — off the WHOLE mismatch set, because the worst fact is not in general the
//! first, and a catalog search built on the first would offer conversions that cannot work.

use crate::canon::{self, as_arr, as_str, Fields, Value};
use crate::dtype::Dtype;
use crate::err::{refuse, Code, Result};
use crate::fit::{CompatibilityReport, Details};
use crate::ids::{prefixed, ObjectRef};
use crate::limits;

// ---------------------------------------------------------------- lossless mappings

/// The closed mapping vocabulary. Each one is invertible BY CONSTRUCTION or it is not in
/// here; the round trip is what proves the construction was right.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Mapping {
    /// Key renaming: zero value movement, a bijection over names.
    Rename { pairs: Vec<(String, String)> },
    /// A contiguous run split into named parts and concatenated back (fused QKV <-> split
    /// q/k/v is exactly this at the byte level).
    Regroup { parts: Vec<u64> },
    /// Element-wise dtype conversion. It REGISTERS only if `to` represents every `from`
    /// value exactly — a narrowing may be proposed and will fail its round trip, which is
    /// the point: the direction is not what makes it lossless, the round trip is.
    Widen { from: Dtype, to: Dtype },
}

impl Mapping {
    pub fn kind(&self) -> &'static str {
        match self {
            Mapping::Rename { .. } => "rename",
            Mapping::Regroup { .. } => "regroup",
            Mapping::Widen { .. } => "widen",
        }
    }

    pub fn describe(&self) -> String {
        match self {
            Mapping::Rename { pairs } => format!("rename {} key(s)", pairs.len()),
            Mapping::Regroup { parts } => format!("regroup into {:?} byte parts", parts),
            Mapping::Widen { from, to } => format!("widen {} -> {}", from.name(), to.name()),
        }
    }

    fn to_value(&self) -> Value {
        match self {
            Mapping::Rename { pairs } => Value::obj(vec![
                (
                    "pairs",
                    Value::arr(
                        pairs
                            .iter()
                            .map(|(a, b)| {
                                Value::arr(vec![Value::str(a.clone()), Value::str(b.clone())])
                            })
                            .collect(),
                    ),
                ),
                ("t", Value::str("rename")),
            ]),
            Mapping::Regroup { parts } => Value::obj(vec![
                (
                    "parts",
                    Value::arr(parts.iter().map(|p| Value::uint(*p)).collect()),
                ),
                ("t", Value::str("regroup")),
            ]),
            Mapping::Widen { from, to } => Value::obj(vec![
                ("from", Value::str(from.name())),
                ("t", Value::str("widen")),
                ("to", Value::str(to.name())),
            ]),
        }
    }

    fn from_value(v: &Value) -> Result<Mapping> {
        let mut f = Fields::new("Mapping", v)?;
        let t = as_str("Mapping", "t", f.req("t")?)?.to_string();
        let m = match t.as_str() {
            "rename" => {
                let mut pairs = Vec::new();
                for p in as_arr("Mapping", "pairs", f.req("pairs")?)? {
                    let a = as_arr("Mapping", "pair", p)?;
                    if a.len() != 2 {
                        return refuse(Code::COUNT_CAP, "a rename pair is exactly two names");
                    }
                    pairs.push((
                        as_str("pair", "from", &a[0])?.to_string(),
                        as_str("pair", "to", &a[1])?.to_string(),
                    ));
                }
                Mapping::Rename { pairs }
            }
            "regroup" => {
                let mut parts = Vec::new();
                for p in as_arr("Mapping", "parts", f.req("parts")?)? {
                    parts.push(crate::canon::as_uint("Mapping", "parts", p)?);
                }
                Mapping::Regroup { parts }
            }
            "widen" => Mapping::Widen {
                from: Dtype::parse(as_str("Mapping", "from", f.req("from")?)?)?,
                to: Dtype::parse(as_str("Mapping", "to", f.req("to")?)?)?,
            },
            o => {
                return refuse(
                    Code::UNKNOWN_FIELD,
                    format!("mapping kind {o:?} is outside the closed set"),
                )
            }
        };
        f.done()?;
        Ok(m)
    }
}

// ---------------------------------------------------------------- element conversions

/// bf16 IS the top 16 bits of an f32, so widening is an exact bit placement and narrowing
/// is round-to-nearest-even over the low half. Written out rather than assumed: the whole
/// point of the round trip is that the two directions are real code that can disagree.
fn bf16_to_f32(bits: u16) -> u32 {
    (bits as u32) << 16
}

fn f32_to_bf16(bits: u32) -> u16 {
    let lsb = (bits >> 16) & 1;
    let round = 0x7fff + lsb;
    ((bits.wrapping_add(round)) >> 16) as u16
}

fn f16_to_f32(bits: u16) -> u32 {
    let sign = ((bits >> 15) as u32) << 31;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let man = (bits & 0x3ff) as u32;
    match exp {
        0 if man == 0 => sign,
        0 => {
            // subnormal f16, normal f32: shift until the implicit bit appears
            let mut e: i32 = 0;
            let mut m = man;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (((127 - 15 + e + 1) as u32) << 23) | ((m & 0x3ff) << 13)
        }
        0x1f => sign | 0x7f80_0000 | (man << 13),
        _ => sign | ((exp + 127 - 15) << 23) | (man << 13),
    }
}

fn f32_to_f16(bits: u32) -> u16 {
    let sign = ((bits >> 31) as u16) << 15;
    let exp = ((bits >> 23) & 0xff) as i32;
    let man = bits & 0x7f_ffff;
    if exp == 0xff {
        if man == 0 {
            return sign | 0x7c00;
        }
        // NaN payloads are preserved, not flushed to a canonical quiet NaN. The round trip
        // is over every representable bit pattern, and a widening that quietly rewrites a
        // NaN is exactly the kind of "almost lossless" a registration must not certify.
        let p = (man >> 13) as u16;
        return sign | 0x7c00 | if p == 0 { 1 } else { p };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let r = (m + half - 1 + ((m >> shift) & 1)) >> shift;
        return sign | r as u16;
    }
    let half = 0x1000 + ((man >> 13) & 1);
    let r = man + half;
    if r & 0x80_0000 != 0 {
        // mantissa carried into the exponent
        return sign | (((e + 1) as u16) << 10);
    }
    sign | ((e as u16) << 10) | ((r >> 13) as u16)
}

/// Element-wise conversion in ONE direction, both directions implemented for every pair the
/// vocabulary admits. Registering a narrowing is allowed to be ATTEMPTED — it runs, and the
/// round trip refuses it — because "we did not implement that direction" and "that direction
/// loses information" are different statements and only the second is a fact about bytes.
fn convert(from: Dtype, to: Dtype, src: &[u8], out: &mut Vec<u8>) -> Result<()> {
    macro_rules! narrowing {
        ($f:expr) => {{
            for c in src.chunks_exact(4) {
                out.extend_from_slice(
                    &$f(u32::from_le_bytes([c[0], c[1], c[2], c[3]])).to_le_bytes(),
                );
            }
        }};
    }
    macro_rules! widening {
        ($f:expr) => {{
            for c in src.chunks_exact(2) {
                out.extend_from_slice(&$f(u16::from_le_bytes([c[0], c[1]])).to_le_bytes());
            }
        }};
    }
    match (from, to) {
        (Dtype::Bf16, Dtype::F32) => widening!(bf16_to_f32),
        (Dtype::F16, Dtype::F32) => widening!(f16_to_f32),
        (Dtype::F32, Dtype::Bf16) => narrowing!(f32_to_bf16),
        (Dtype::F32, Dtype::F16) => narrowing!(f32_to_f16),
        _ => {
            return refuse(
                Code::UNKNOWN_FIELD,
                format!(
                    "{} -> {} is not an element-wise cast at all: a QUANTIZATION carries \
                     scales and a role set, so it is not expressible in the closed mapping \
                     vocabulary and cannot even be proposed as a lossless conversion. It is \
                     a priced job with its own header, not a registration.",
                    from.name(),
                    to.name()
                ),
            )
        }
    }
    Ok(())
}

// ---------------------------------------------------------------- the round trip

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoundTrip {
    pub cases: u64,
    pub bytes: u64,
    /// `complete-domain` when every representable input was run, else `corpus`.
    pub coverage: String,
    pub corpus: ObjectRef,
}

/// Run A -> B -> A and require byte identity. This is the registration act: a mapping that
/// cannot survive it does not become a registration, and there is no override.
pub fn round_trip(m: &Mapping, corpus: &[Vec<u8>]) -> Result<RoundTrip> {
    let mut cases = 0u64;
    let mut bytes = 0u64;
    for (i, case) in corpus.iter().enumerate() {
        if case.is_empty() {
            return refuse(
                Code::MISSING_FIELD,
                format!(
                    "corpus case {i} is empty: zero bytes round-trip trivially and prove nothing"
                ),
            );
        }
        let back = match m {
            Mapping::Rename { pairs } => {
                // The value never moves; what must round-trip is the NAME bijection.
                let names: Vec<String> = String::from_utf8_lossy(case)
                    .split('\n')
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string())
                    .collect();
                let fwd: Vec<String> = names
                    .iter()
                    .map(|n| apply_rename(pairs, n, false))
                    .collect();
                let back: Vec<String> = fwd.iter().map(|n| apply_rename(pairs, n, true)).collect();
                let mut out = back.join("\n").into_bytes();
                out.push(b'\n');
                out
            }
            Mapping::Regroup { parts } => {
                let total: u64 = parts.iter().sum();
                if total != case.len() as u64 {
                    return refuse(
                        Code::BYTE_LENGTH_MISMATCH,
                        format!(
                            "corpus case {i}: {} B does not fill the {total} B the regroup \
                             declares",
                            case.len()
                        ),
                    );
                }
                let mut split: Vec<Vec<u8>> = Vec::new();
                let mut at = 0usize;
                for p in parts {
                    split.push(case[at..at + *p as usize].to_vec());
                    at += *p as usize;
                }
                split.concat()
            }
            Mapping::Widen { from, to } => {
                if !(case.len() as u64).is_multiple_of(from.size()) {
                    return refuse(
                        Code::BYTE_LENGTH_MISMATCH,
                        format!(
                            "corpus case {i}: not a whole number of {} elements",
                            from.name()
                        ),
                    );
                }
                let mut there = Vec::with_capacity(case.len() * 2);
                convert(*from, *to, case, &mut there)?;
                let mut back = Vec::with_capacity(case.len());
                convert(*to, *from, &there, &mut back)?;
                back
            }
        };
        if &back != case {
            let at = back
                .iter()
                .zip(case.iter())
                .position(|(a, b)| a != b)
                .unwrap_or(0);
            return refuse(
                Code::ROUND_TRIP_LOSSY,
                format!(
                    "{}: A->B->A is not byte-identical (corpus case {i}, first difference at \
                     byte {at}: got {:#04x}, want {:#04x}) — a lossy step can NEVER register \
                     as a lossless conversion",
                    m.describe(),
                    back.get(at).copied().unwrap_or(0),
                    case.get(at).copied().unwrap_or(0)
                ),
            );
        }
        cases += 1;
        bytes += case.len() as u64;
    }
    if cases == 0 {
        return refuse(
            Code::MISSING_FIELD,
            "an empty corpus proves nothing: a registration runs its round trip",
        );
    }
    let coverage = match m {
        Mapping::Widen { from, .. } if from.size() <= 2 && bytes >= (1 << 17) => "complete-domain",
        _ => "corpus",
    };
    let mut flat = Vec::new();
    for c in corpus {
        flat.extend_from_slice(c);
    }
    Ok(RoundTrip {
        cases,
        bytes,
        coverage: coverage.to_string(),
        corpus: ObjectRef::of(&flat),
    })
}

fn apply_rename(pairs: &[(String, String)], name: &str, inverse: bool) -> String {
    for (a, b) in pairs {
        let (from, to) = if inverse { (b, a) } else { (a, b) };
        if let Some(rest) = name.strip_prefix(from.as_str()) {
            return format!("{to}{rest}");
        }
    }
    name.to_string()
}

/// The complete 16-bit domain, as one corpus case. Every bit pattern, in order — so a
/// `complete-domain` widening registration is a proof, not a sample.
pub fn complete_domain_16() -> Vec<Vec<u8>> {
    let mut v = Vec::with_capacity(1 << 17);
    for b in 0u32..=0xffff {
        v.extend_from_slice(&(b as u16).to_le_bytes());
    }
    vec![v]
}

// ---------------------------------------------------------------- the registration

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConversionRegistration {
    pub name: String,
    pub mapping: Mapping,
    pub round_trip: RoundTrip,
    /// The build that RAN the round trip. A registration is a record of an execution.
    pub registered_by: String,
}

impl ConversionRegistration {
    /// The only constructor: it runs the round trip and refuses on anything but identity.
    pub fn register(name: &str, mapping: Mapping, corpus: &[Vec<u8>]) -> Result<Self> {
        crate::ids::ascii_name("registration.name", name, limits::MAX_NAME_BYTES)?;
        let round_trip = round_trip(&mapping, corpus)?;
        Ok(ConversionRegistration {
            name: name.to_string(),
            mapping,
            round_trip,
            registered_by: format!("tensorfs-core/{}", env!("CARGO_PKG_VERSION")),
        })
    }

    /// Does this registration cover a dtype cast from `present` to `required`?
    pub fn covers_cast(&self, present: Dtype, required: Dtype) -> bool {
        matches!(&self.mapping, Mapping::Widen { from, to } if *from == present && *to == required)
    }
}

impl ConversionRegistration {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        Self::from_value(&canon::parse_canonical(bytes, limits::SPEC_MAX_BYTES)?)
    }
    pub fn canonical_bytes(&self) -> Vec<u8> {
        canon::write(&self.to_value())
    }

    pub fn from_value(v: &Value) -> Result<Self> {
        let mut f = Fields::new("ConversionRegistration", v)?;
        let mapping = Mapping::from_value(f.req("mapping")?)?;
        let name = f.req_str("name")?.to_string();
        let registered_by = prefixed_build(f.req_str("registered_by")?)?;
        let round_trip = {
            let mut rf = Fields::new("RoundTrip", f.req("round_trip")?)?;
            let bytes = rf.req_uint("bytes")?;
            let cases = rf.req_uint("cases")?;
            let corpus = ObjectRef::from_value("RoundTrip.corpus", rf.req("corpus")?)?;
            let coverage = rf.req_str("coverage")?.to_string();
            rf.done()?;
            if coverage != "complete-domain" && coverage != "corpus" {
                return refuse(
                    Code::UNKNOWN_FIELD,
                    format!("round_trip.coverage {coverage:?} is outside the closed set"),
                );
            }
            if cases == 0 {
                return refuse(
                    Code::MISSING_FIELD,
                    "a registration with zero round-trip cases ran nothing",
                );
            }
            RoundTrip {
                cases,
                bytes,
                coverage,
                corpus,
            }
        };
        f.done()?;
        crate::ids::ascii_name("registration.name", &name, limits::MAX_NAME_BYTES)?;
        Ok(ConversionRegistration {
            name,
            mapping,
            round_trip,
            registered_by,
        })
    }

    pub fn to_value(&self) -> Value {
        Value::obj(vec![
            ("mapping", self.mapping.to_value()),
            ("name", Value::str(self.name.clone())),
            ("registered_by", Value::str(self.registered_by.clone())),
            (
                "round_trip",
                Value::obj(vec![
                    ("bytes", Value::uint(self.round_trip.bytes)),
                    ("cases", Value::uint(self.round_trip.cases)),
                    ("corpus", self.round_trip.corpus.to_value()),
                    ("coverage", Value::str(self.round_trip.coverage.clone())),
                ]),
            ),
        ])
    }
}

fn prefixed_build(s: &str) -> Result<String> {
    if s.is_empty() || s.len() > limits::MAX_NAME_BYTES {
        return refuse(Code::KEY_GRAMMAR, "registered_by out of bounds");
    }
    Ok(s.to_string())
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Registry {
    pub registrations: Vec<ConversionRegistration>,
}

impl Registry {
    pub fn covering_cast(
        &self,
        present: Dtype,
        required: Dtype,
    ) -> Option<&ConversionRegistration> {
        self.registrations
            .iter()
            .find(|r| r.covers_cast(present, required))
    }
}

// ---------------------------------------------------------------- the four grades

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Grade {
    Compatible,
    Convertible {
        registration: String,
        mapping: String,
    },
    Producible {
        /// Derived from the CONTRACT side. Never from the observation.
        target: String,
        source: String,
    },
    Incompatible {
        reason: String,
    },
}

impl Grade {
    pub fn name(&self) -> &'static str {
        match self {
            Grade::Compatible => "COMPATIBLE",
            Grade::Convertible { .. } => "CONVERTIBLE",
            Grade::Producible { .. } => "PRODUCIBLE",
            Grade::Incompatible { .. } => "INCOMPATIBLE",
        }
    }
}

/// Precision order, coarse and explicit. Used only to answer "is a priced job even
/// possible from what this checkpoint holds" — never to infer a role geometry.
fn precision_rank(d: Dtype) -> u8 {
    match d {
        Dtype::F64 => 6,
        Dtype::F32 => 5,
        Dtype::Bf16 | Dtype::F16 => 4,
        Dtype::F8E4M3FN | Dtype::F8E5M2 => 3,
        Dtype::I64 | Dtype::I32 => 2,
        Dtype::I16 | Dtype::I8 | Dtype::U8 => 1,
        Dtype::Bool => 0,
    }
}

/// The PREFLIGHT grading: the first blocking fact, graded. "Can this run" is answered by
/// one fact, so this reads one. It is NOT an answer to "how bad is this" — for that the
/// whole mismatch set has to be walked, which is `grade_all`.
pub fn grade(report: &CompatibilityReport, reg: &Registry) -> Grade {
    match report.mismatches.first() {
        None => Grade::Compatible,
        Some(d) => grade_detail(d, reg),
    }
}

/// The ONE grading decision, over ONE fact and the registry. Header-derived on both sides,
/// so it is decidable before a byte of weight is downloaded.
///
/// Every remedy this function builds reads `d.required` — the CONTRACT side — for the
/// target and `d.present` only for the source. That is law 2, and it is held HERE, once,
/// so that no caller which selects among facts can reconstruct a remedy out of the wrong
/// side: a selector picks a whole `Grade` this function built, never a level from one fact
/// and a target from another.
fn grade_detail(d: &Details, reg: &Registry) -> Grade {
    let where_ = format!("{}/{}", d.component, d.key);
    match d.code {
        // A mapping cannot invent bytes that are not there. A registered RENAME can make
        // this disappear, but only `resolve` may say so — proving it means applying the
        // mapping and re-checking the whole topology, and one matching key name is not that
        // proof.
        Code::MISSING_TENSOR | Code::MISSING_COMPONENT => Grade::Incompatible {
            reason: format!(
                "{where_}: the contract requires a tensor this checkpoint does not carry, \
                 and no registered mapping applied to the whole topology produces it"
            ),
        },
        Code::DTYPE_MISMATCH => {
            let (present, required) = (d.present.dtype, d.required.dtype);
            match (present, required) {
                (Some(p), Some(r)) => match reg.covering_cast(p, r) {
                    // Cast-only, lossless, registered: CONVERTIBLE. The forbidden
                    // over-constraint is exactly the branch that is NOT taken here.
                    Some(reg) => Grade::Convertible {
                        registration: reg.name.clone(),
                        mapping: reg.mapping.describe(),
                    },
                    None if precision_rank(p) > precision_rank(r) => Grade::Producible {
                        target: format!("{} at {where_}", r.name()),
                        source: format!("this checkpoint's {} tensors", p.name()),
                    },
                    None => Grade::Incompatible {
                        reason: format!(
                            "{where_}: the contract wants {} and the checkpoint holds {}. No \
                             registered lossless cast, and {} is not a higher-precision source \
                             a job could produce {} from",
                            r.name(),
                            p.name(),
                            p.name(),
                            r.name()
                        ),
                    },
                },
                _ => Grade::Incompatible {
                    reason: format!("{where_}: a dtype mismatch with a side missing"),
                },
            }
        }
        Code::ENCODING_MISMATCH => {
            let target = d
                .required
                .accepts
                .clone()
                .unwrap_or_else(|| "the contract's accepted encodings".to_string());
            Grade::Producible {
                target: format!("{target} at {where_}"),
                source: format!(
                    "this checkpoint's {}",
                    d.present.encoding.clone().unwrap_or_default()
                ),
            }
        }
        Code::SHAPE_MISMATCH => Grade::Incompatible {
            reason: format!(
                "{where_}: logical shapes disagree ({:?} required, {:?} present). A different \
                 model, not a different packaging",
                d.required.shape, d.present.shape
            ),
        },
        Code::EXTRA_TENSOR | Code::EXTRA_COMPONENT => Grade::Incompatible {
            reason: format!(
                "{where_}: a surplus key on a classified lane is drift or corruption evidence"
            ),
        },
        Code::UNKNOWN_ENCODING => Grade::Incompatible {
            reason: format!("{where_}: the cited spec object is absent from the closure"),
        },
        Code::UNKNOWN_TO_LOCAL_CAPABILITY => Grade::Incompatible {
            reason: format!(
                "{where_}: the cited spec carries no alias in this pin. Upgrade the registry \
                 pin; offline, unregistered and newer-than-pin are indistinguishable"
            ),
        },
        other => Grade::Incompatible {
            reason: format!("{where_}: {}", other.as_str()),
        },
    }
}

/// Where a grade sits on the ladder. `COMPATIBLE < CONVERTIBLE < PRODUCIBLE < INCOMPATIBLE`
/// is a total order over four levels, so "the worst of a set" is well defined.
pub fn severity(g: &Grade) -> u8 {
    match g {
        Grade::Compatible => 0,
        Grade::Convertible { .. } => 1,
        Grade::Producible { .. } => 2,
        Grade::Incompatible { .. } => 3,
    }
}

/// The RANKING grading: the WORST grade across the WHOLE mismatch set.
///
/// This is the function a catalog search needs and `grade` is not. An assignment whose
/// first mismatch is a dtype cast and whose second is a shape mismatch grades PRODUCIBLE on
/// the first fact and is actually INCOMPATIBLE; a search built on the first fact would
/// systematically over-promise, offering users conversions that cannot work.
///
/// It refuses a report taken at `Depth::FirstMismatch` rather than ranking it. A stopped
/// walk and a complete one are indistinguishable once the depth is dropped, and "the worst
/// fact in a set that stopped at the first" is precisely the over-promise above with a
/// different function name on it.
///
/// Law 2 survives selection because what is selected is a whole `Grade` that `grade_detail`
/// built from ITS OWN fact's contract side. The remedy and the level always come from the
/// same fact; ties keep the earliest in canonical order, so the answer is deterministic.
pub fn grade_all(report: &CompatibilityReport, reg: &Registry) -> Result<Grade> {
    if report.depth != crate::fit::Depth::Full {
        return refuse(
            Code::PARTIAL_MISMATCH_SET,
            format!(
                "this report was taken at depth {:?} and carries {} fact(s): the WORST fact \
                 cannot be read off a walk that stopped at the first. Re-check at full depth",
                report.depth.name(),
                report.mismatches.len()
            ),
        );
    }
    let mut worst = Grade::Compatible;
    for d in &report.mismatches {
        let g = grade_detail(d, reg);
        if severity(&g) > severity(&worst) {
            worst = g;
        }
    }
    guard_over_constraint(report, reg, &worst)?;
    Ok(worst)
}

/// Grade the report AT THE DEPTH IT WAS TAKEN. The document says which question was asked;
/// the caller does not get to pretend it asked the other one.
pub fn grade_report(report: &CompatibilityReport, reg: &Registry) -> Result<Grade> {
    match report.depth {
        crate::fit::Depth::Full => grade_all(report, reg),
        crate::fit::Depth::FirstMismatch => {
            let g = grade(report, reg);
            guard_over_constraint(report, reg, &g)?;
            Ok(g)
        }
    }
}

/// Apply a registered mapping to the CODE side, producing the contract a checkpoint in the
/// mapped form would have to satisfy. Only the mappings whose effect is expressible at the
/// topology layer participate: a rename moves keys, a widening moves the required dtype
/// down to the form the mapping can widen FROM. A regroup changes key counts and shapes and
/// is deliberately not auto-applied — it is a real conversion whose plan belongs to a job,
/// and a grader that pretended otherwise would be planning.
pub fn apply(
    m: &Mapping,
    requirements: &crate::fit::TensorRequirements,
) -> Option<crate::fit::TensorRequirements> {
    let mut out = requirements.clone();
    match m {
        Mapping::Rename { pairs } => {
            for (_, keys) in out.components.iter_mut() {
                for (k, _) in keys.iter_mut() {
                    *k = apply_rename(pairs, k, false);
                }
                keys.sort_by(|a, b| a.0.cmp(&b.0));
            }
        }
        Mapping::Widen { from, to } => {
            for (_, keys) in out.components.iter_mut() {
                for (_, r) in keys.iter_mut() {
                    if r.dtype == *to {
                        r.dtype = *from;
                    }
                }
            }
        }
        Mapping::Regroup { .. } => return None,
    }
    Some(out)
}

/// The FULL grading: check the WHOLE topology, and where a registration might close the
/// gap, apply it and check AGAIN. `CONVERTIBLE` is returned only when the mapped topology
/// is Satisfied — the registration is proved to close the whole gap, not merely to touch
/// the first key that disagreed. One matching name is not a conversion.
///
/// The verdict it falls through to is `grade_all`, so the same whole-topology standard now
/// holds for PRODUCIBLE: it is the worst of every fact, not the grade of the first.
///
/// Bounded by the registration count, and every pass is header-only, so the whole thing
/// still decides pre-download.
pub fn resolve(
    requirements: &crate::fit::TensorRequirements,
    header: &crate::header::Header,
    closure: &crate::header::Closure,
    pin: &[String],
    custody: crate::fit::Custody,
    reg: &Registry,
) -> Result<(CompatibilityReport, Grade)> {
    // The FIRST walk is FULL: the report this returns is the hub's document, and the hub
    // ranks. The re-checks below ask only "does the mapped topology satisfy outright", and
    // one fact answers that, so they take the cheap path.
    let out = crate::fit::check_compatibility(
        requirements,
        header,
        closure,
        pin,
        custody,
        crate::fit::Depth::Full,
    )?;
    let report = CompatibilityReport::of(requirements, header, custody, &out)?;
    if out.is_satisfied() {
        return Ok((report, Grade::Compatible));
    }
    for r in &reg.registrations {
        let mapped = match apply(&r.mapping, requirements) {
            Some(m) => m,
            None => continue,
        };
        let re = crate::fit::check_compatibility(
            &mapped,
            header,
            closure,
            pin,
            custody,
            crate::fit::Depth::FirstMismatch,
        )?;
        if re.is_satisfied() {
            return Ok((
                report,
                Grade::Convertible {
                    registration: r.name.clone(),
                    mapping: r.mapping.describe(),
                },
            ));
        }
    }
    let g = grade_all(&report, reg)?;
    Ok((report, g))
}

/// The v1 recovery with an exit code: a mismatch set that is cast-only and COVERED can
/// NEVER be graded INCOMPATIBLE. Run after every grading; a red arm plants the shortcut and
/// watches this fire.
///
/// It reads the WHOLE set, and every fact in it must be a covered cast before it fires.
/// Reading only the first fact would make the guard refuse an honest INCOMPATIBLE the
/// moment a covered cast happened to sort ahead of a shape mismatch — the mirror image of
/// the defect `grade_all` exists to fix, and it would refuse a correct verdict rather than
/// merely over-promise.
pub fn guard_over_constraint(
    report: &CompatibilityReport,
    reg: &Registry,
    g: &Grade,
) -> Result<()> {
    if report.mismatches.is_empty() || !matches!(g, Grade::Incompatible { .. }) {
        return Ok(());
    }
    let mut covered = Vec::new();
    for d in &report.mismatches {
        if d.code != Code::DTYPE_MISMATCH {
            return Ok(());
        }
        match (d.present.dtype, d.required.dtype) {
            (Some(p), Some(r)) => match reg.covering_cast(p, r) {
                Some(hit) => covered.push((d, p, r, hit)),
                None => return Ok(()),
            },
            _ => return Ok(()),
        }
    }
    let (d, p, r, hit) = covered[0];
    refuse(
        Code::UNREGISTERED_CONVERSION,
        format!(
            "{}/{}: graded INCOMPATIBLE while registration {:?} covers the cast {} -> {} with \
             a {} round trip, and all {} mismatch(es) are covered casts — forbidden \
             over-constraint (v1 recovery, now hub law): a cast-only mismatch names the \
             registered lossless conversion",
            d.component,
            d.key,
            hit.name,
            p.name(),
            r.name(),
            hit.round_trip.coverage,
            report.mismatches.len()
        ),
    )
}

// ---------------------------------------------------------------- the decision document

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GradeDecision {
    pub report: String,
    pub grade: Grade,
}

impl GradeDecision {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        Self::from_value(&canon::parse_canonical(bytes, limits::SPEC_MAX_BYTES)?)
    }
    pub fn canonical_bytes(&self) -> Vec<u8> {
        canon::write(&self.to_value())
    }

    pub fn from_value(v: &Value) -> Result<Self> {
        let mut f = Fields::new("GradeDecision", v)?;
        let mut gf = Fields::new("grade", f.req("grade")?)?;
        let t = as_str("grade", "t", gf.req("t")?)?.to_string();
        let grade = match t.as_str() {
            "COMPATIBLE" => Grade::Compatible,
            "CONVERTIBLE" => Grade::Convertible {
                mapping: gf.req_str("mapping")?.to_string(),
                registration: gf.req_str("registration")?.to_string(),
            },
            "PRODUCIBLE" => Grade::Producible {
                source: gf.req_str("source")?.to_string(),
                target: gf.req_str("target")?.to_string(),
            },
            "INCOMPATIBLE" => Grade::Incompatible {
                reason: gf.req_str("reason")?.to_string(),
            },
            o => {
                return refuse(
                    Code::UNKNOWN_FIELD,
                    format!("grade {o:?} is outside the four-verdict ladder"),
                )
            }
        };
        gf.done()?;
        let report = prefixed("GradeDecision.report", f.req_str("report")?)?;
        f.done()?;
        Ok(GradeDecision { report, grade })
    }

    pub fn to_value(&self) -> Value {
        let g = match &self.grade {
            Grade::Compatible => Value::obj(vec![("t", Value::str("COMPATIBLE"))]),
            Grade::Convertible {
                registration,
                mapping,
            } => Value::obj(vec![
                ("mapping", Value::str(mapping.clone())),
                ("registration", Value::str(registration.clone())),
                ("t", Value::str("CONVERTIBLE")),
            ]),
            Grade::Producible { target, source } => Value::obj(vec![
                ("source", Value::str(source.clone())),
                ("t", Value::str("PRODUCIBLE")),
                ("target", Value::str(target.clone())),
            ]),
            Grade::Incompatible { reason } => Value::obj(vec![
                ("reason", Value::str(reason.clone())),
                ("t", Value::str("INCOMPATIBLE")),
            ]),
        };
        Value::obj(vec![
            ("grade", g),
            ("report", Value::str(self.report.clone())),
        ])
    }
}
