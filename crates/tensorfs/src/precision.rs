//! The lane rule (xs-042, design `precision-tiers.md` §2.2, §2.7, §2.8; spec §7 "Selection"): a
//! variant from what its header holds, its GPUs' native formats, the request's bits and a
//! slot's declared tolerances. Nothing is keyed by a card's name or a file's. The rule's table (kinds,
//! families, GPU features, lane orders) is data, `spec/selection.json`, which the Hub vendors byte for
//! byte and evaluates the same way; this file is the evaluator.
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::LazyLock,
};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::format::{Body, Dtype, Header};

/// `spec/selection.json`. Unknown fields are ignored.
#[derive(Deserialize)]
struct Table {
    rule: u32,
    features: BTreeMap<String, Feature>,
    kinds: Vec<Kind>,
    families: BTreeMap<String, Family>,
    order: Vec<Order>,
    plain: Vec<Order>,
    schemes: Vec<String>,
}

/// A feature every GPU of a set has: its capability starts with `prefix`, or is from `min` on and
/// not at an `except`.
#[derive(Deserialize)]
struct Feature {
    #[serde(default)]
    prefix: String,
    #[serde(default)]
    min: String,
    #[serde(default)]
    except: Vec<String>,
    /// Whether a GPU whose capability is not a number has it.
    #[serde(default)]
    unread: bool,
    /// Whether no GPU (the CPU) has it.
    #[serde(default)]
    cpu: bool,
}

/// A kind (`kind` exactly, or any starting with `prefix`) and its family.
#[derive(Deserialize)]
struct Kind {
    #[serde(default)]
    kind: String,
    #[serde(default)]
    prefix: String,
    family: String,
}

#[derive(Default, Deserialize)]
struct Family {
    bits: u32,
    /// The feature every GPU needs to run it; empty: any.
    #[serde(default)]
    needs: String,
    /// The width its matmuls run at: the first row whose feature the GPUs have, else 16.
    #[serde(default)]
    compute: Vec<Compute>,
    /// The lane it ranks as; `after`: just after the order.
    #[serde(default)]
    rank: String,
}

#[derive(Deserialize)]
struct Compute {
    #[serde(default)]
    when: String,
    width: u32,
}

/// A lane order: the first row whose bits, slot and feature match.
#[derive(Deserialize)]
struct Order {
    #[serde(default)]
    bits: Option<u32>,
    /// Only a decode slot's.
    #[serde(default)]
    decode: bool,
    #[serde(default)]
    when: String,
    lanes: Vec<String>,
}

static TABLE: LazyLock<Table> = LazyLock::new(|| {
    let table: Table = serde_json::from_str(include_str!("../spec/selection.json")).expect("spec/selection.json is the rule's table");
    debug_assert!(
        {
            let mut implemented: Vec<&str> = crate::format::SCHEMES.iter().map(|scheme| scheme.name).collect();
            implemented.sort_unstable();
            implemented == table.schemes.iter().map(String::as_str).collect::<Vec<_>>()
        },
        "spec/selection.json's schemes are not the ones this reader implements"
    );
    table
});

/// A compute capability `"major.minor"` as one number (`"8.9"`: 89); `None` when it is not one.
fn capability(cc: &str) -> Option<u32> {
    let (major, minor) = cc.split_once('.')?;
    if major.is_empty() || minor.is_empty() || major.len() > 4 || minor.len() > 4 || !major.bytes().chain(minor.bytes()).all(|c| c.is_ascii_digit()) {
        return None;
    }
    Some(major.parse::<u32>().ok()? * 10 + minor.parse::<u32>().ok()?)
}

impl Feature {
    fn held(&self, ccs: &[&str]) -> bool {
        let min = capability(&self.min).unwrap_or(u32::MAX);
        let except: Vec<u32> = self.except.iter().filter_map(|cc| capability(cc)).collect();
        let one = |cc: &&str| match self.prefix.is_empty() {
            false => cc.starts_with(&self.prefix),
            true => capability(cc).map_or(self.unread, |n| n >= min && !except.contains(&n)),
        };
        if ccs.is_empty() { self.cpu } else { ccs.iter().all(one) }
    }

    /// How a refusal names the GPUs that have it: `sm80+ except sm103`.
    fn named(&self) -> String {
        let sm = |cc: &String| format!("sm{}", cc.replace('.', ""));
        match self.except.is_empty() {
            true => format!("{}+", sm(&self.min)),
            false => format!("{}+ except {}", sm(&self.min), self.except.iter().map(sm).collect::<Vec<_>>().join(", ")),
        }
    }
}

/// The table's row for a family (the empty row for none).
fn row(family: &str) -> &'static Family {
    static NONE: LazyLock<Family> = LazyLock::new(Family::default);
    TABLE.families.get(family).unwrap_or(&NONE)
}

/// Whether the rule picks `lane` by name; any other is a release's only lane, served silently.
fn named_lane(lane: &str) -> bool {
    TABLE.order.iter().chain(&TABLE.plain).any(|order| order.lanes.iter().any(|l| l == lane))
}

/// The GPUs a function runs on as the rule reads them (design §2.1, §2.7): the table's features
/// every one has (`mma`, `int8`, `fp8`, `fp4`, `bf16`, `apple`) and how a warning names them (`sm90`,
/// `sm86+sm89`). An Apple GPU's capability is its Metal family (`apple9`): it runs bf16 and fp16
/// and no encoded format until Metal kernels exist.
#[derive(Clone, Debug, PartialEq)]
pub struct Gpus {
    pub features: BTreeSet<String>,
    pub name: String,
    pub capabilities: Vec<String>,
}

/// No GPU: the CPU, which runs bf16 and no encoded format.
impl Default for Gpus {
    fn default() -> Self {
        Self::of(&[])
    }
}

impl Gpus {
    /// From each GPU's compute capability (`"9.0"`, `"apple9"`).
    pub fn of(ccs: &[&str]) -> Self {
        let named = |cc: &&str| match cc.starts_with("apple") {
            true => cc.to_string(),
            false => format!("sm{}", cc.replace('.', "")),
        };
        let mut names: Vec<String> = ccs.iter().filter(|cc| !cc.is_empty()).map(named).collect();
        names.sort();
        names.dedup();
        let name = if names.is_empty() { "no GPU".into() } else { names.join("+") };
        Self {
            features: TABLE.features.iter().filter(|(_, feature)| feature.held(ccs)).map(|(name, _)| name.clone()).collect(),
            name,
            capabilities: ccs.iter().map(|cc| cc.to_string()).collect(),
        }
    }

    /// Whether every one has `feature` (the empty feature: always).
    pub fn has(&self, feature: &str) -> bool {
        feature.is_empty() || self.features.contains(feature)
    }

    /// The width a family's matmuls run at here (the table's `compute`): 4 where FP4 tensor cores
    /// run it, 8 for int8, fp8 and w4a8, else 16 (16-bit, and 4-bit weight-only).
    pub fn compute(&self, family: &str) -> u32 {
        row(family).compute.iter().find(|c| self.has(&c.when)).map_or(16, |c| c.width)
    }
}

/// A slot's declared tolerances (package interface): `min_bits` (8 when omitted), its phase,
/// whether its class takes encoded leaves, and whether it is raw weights (`Weights[M]`, read as
/// data: always 16 bits, whatever its class declares).
#[derive(Clone, Copy, Debug)]
pub struct Slot {
    pub min_bits: u64,
    pub decode: bool,
    pub encoded: bool,
    pub raw: bool,
}

impl Slot {
    pub fn of(slot: &Value) -> Self {
        Self {
            min_bits: slot["min_bits"].as_u64().unwrap_or(8),
            decode: slot["phase"] == "decode",
            encoded: slot["encoded_leaves"] == "accept",
            raw: slot["weights"] == true,
        }
    }

    /// Its bits (§2.2): the request's; else 16 at `min_bits=16`; else 4 at `min_bits=4` for a
    /// decode slot; else 8. Memory never lowers them: paging fits weights (owner, 2026-10-22).
    pub fn bits(&self, asked: u32) -> u32 {
        match asked {
            _ if self.raw => 16,
            4 | 8 | 16 => asked,
            _ if self.min_bits == 16 => 16,
            _ if self.min_bits == 4 && self.decode => 4,
            _ => 8,
        }
    }

    /// The lanes to ask for at `bits`, in order (§2.2, the table's `order`): the width's encoded
    /// lanes every GPU runs, unless the slot declines encoded lanes, then the table's `plain` order
    /// (`fp16` first only where the GPUs run no bf16, owner 2026-10-16).
    pub fn lanes(&self, bits: u32, gpus: &Gpus) -> Vec<&'static str> {
        let first = |orders: &'static [Order], bits: Option<u32>| {
            orders.iter().find(|o| o.bits == bits && (!o.decode || self.decode) && gpus.has(&o.when)).map_or(&[][..], |o| &o.lanes[..])
        };
        let encoded = first(&TABLE.order, Some(bits)).iter().filter(|lane| self.encoded && gpus.has(&row(lane).needs));
        encoded.chain(first(&TABLE.plain, None)).map(String::as_str).collect()
    }

    /// The run's warnings for `model` served in `served` (§2.8): one when it declined encoded
    /// lanes the GPUs run, one naming every lane asked before the one served, and fp16 served
    /// because the GPUs run no bf16 says so.
    pub fn warnings(&self, bits: u32, gpus: &Gpus, model: &str, served: &str) -> Vec<String> {
        if self.raw || !named_lane(served) {
            return vec![];
        }
        let declined = !self.encoded && Self { encoded: true, ..*self }.lanes(bits, gpus).len() > 2;
        let declined = declined.then(|| format!("precision {bits}: {model} declines encoded leaves; served {served}"));
        let mut missing: Vec<&str> = self.lanes(bits, gpus).into_iter().take_while(|lane| *lane != served).collect();
        let missing = missing.pop().map(|last| match missing.is_empty() {
            true => last.to_string(),
            false => format!("{} or {last}", missing.join(", ")),
        });
        let missing = missing.map(|lanes| format!("precision {bits}: {model} has no {lanes} lane for {}; served {served}", gpus.name));
        let no_bf16 = (served == "fp16" && !gpus.has("bf16")).then(|| format!("precision {bits}: {} runs no bf16; served fp16", gpus.name));
        declined.into_iter().chain(missing).chain(no_bf16).collect()
    }
}

impl Slot {
    /// What it asks of a variant at the request's bits (`asked`, 0: none): that lane only when
    /// asked (raw weights are data and ask none), else its own lane first; never narrower than
    /// `min_bits` unasked; no encoded weights unless it takes encoded leaves.
    pub fn constraints(&self, gpus: &Gpus, asked: u32) -> Constraints {
        let bits = (!self.raw && matches!(asked, 4 | 8 | 16)).then_some(asked);
        // An Apple GPU's only width is 16 bits: its lanes are named so.
        let width = if gpus.has("apple") { 16 } else { self.bits(asked) };
        Constraints {
            gpus: gpus.capabilities.clone(),
            bits,
            prefer: bits.is_none().then_some(width),
            min_bits: u32::try_from(self.min_bits).ok(),
            plain: !self.encoded,
        }
    }
}

/// A variant as the rule reads it, from its header alone (spec §7): every kind it holds (an
/// encoded tensor's scheme, a plain one's dtype), its stored bytes, and its lane: the family and
/// width of the narrowest kind of the table it holds (`""`, 0 when it holds none).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Profile {
    pub bits: u32,
    pub family: String,
    pub bytes: u64,
    pub kinds: Vec<String>,
}

impl Profile {
    pub fn of(header: &Header) -> Self {
        let mut held: BTreeMap<String, u64> = BTreeMap::new();
        for (_, tensor) in header.tensors() {
            let kind = match &tensor.body {
                Body::Encoded { encoding, .. } => encoding.scheme.clone(),
                Body::Data(_) => tensor.dtype.name().to_owned(),
            };
            *held.entry(kind).or_default() += tensor.locations().iter().map(|(_, at)| at.length).sum::<u64>();
        }
        // Its lane is the narrowest of the table's widths it holds: an nvfp4 variant that keeps
        // its sensitive layers int8 is 4 bits, however many bytes those layers hold. Of that
        // width, the kind with the most bytes; of equal ones, the first name. An encoded kind the
        // table does not name leaves it no lane: its width is not this rule's to guess (a newer
        // family's variant is never read as one of its int8 or bf16 extras).
        let untabled = held.keys().any(|kind| matches!(Dtype::parse(kind), Dtype::Other(_)) && family(kind).1 == 0);
        let lane = held.iter().filter(|(kind, _)| !untabled && family(kind).1 > 0)
            .min_by_key(|(kind, n)| (family(kind).1, std::cmp::Reverse(**n), kind.as_str()));
        let (family, bits) = lane.map_or(("", 0), |(kind, _)| family(kind));
        Self { bits, family: family.into(), bytes: held.values().sum(), kinds: held.into_keys().collect() }
    }
}

/// A kind's family and width (the table's `kinds`); `("", 0)` outside it.
fn family(kind: &str) -> (&'static str, u32) {
    let found = TABLE.kinds.iter().find(|k| !k.kind.is_empty() && k.kind == kind || !k.prefix.is_empty() && kind.starts_with(&k.prefix));
    found.map_or(("", 0), |k| (k.family.as_str(), row(&k.family).bits))
}

/// The GPUs a scheme of the table needs (its family's feature), for a refusal to name.
fn needs(kind: &str) -> String {
    let implemented = TABLE.schemes.iter().any(|scheme| scheme == kind);
    let feature = implemented.then(|| TABLE.features.get(&row(family(kind).0).needs)).flatten();
    feature.map_or_else(|| "a GPU reader".into(), Feature::named)
}

/// Whether every GPU runs `kind`: a §5 dtype always; a scheme this reader implements where its
/// family's feature holds (one outside the families decodes anywhere); an unknown one never.
fn runs(kind: &str, gpus: &Gpus) -> bool {
    if matches!(Dtype::parse(kind), Dtype::Other(_)) && !TABLE.schemes.iter().any(|scheme| scheme == kind) {
        return false;
    }
    gpus.has(&row(family(kind).0).needs)
}

/// What a reader or a slot asks of a variant (spec §7): the GPUs it runs on, by compute
/// capability (`"8.9"`; none is the CPU); `bits`, asked explicitly, which only that lane answers;
/// else `prefer` (none: 8), the lane ranked first. A slot adds `min_bits`, the narrowest it takes
/// unasked, and `plain`, that it takes no encoded weights.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Constraints {
    #[serde(default)]
    pub gpus: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bits: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefer: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_bits: Option<u32>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub plain: bool,
}

impl Constraints {
    /// Refused unless each width they name is 4, 8 or 16, and asked 16 on Apple GPUs.
    pub fn check(&self) -> crate::Result<()> {
        if let Some(bits) = [self.bits, self.prefer, self.min_bits].into_iter().flatten().find(|bits| ![4, 8, 16].contains(bits)) {
            return crate::failure::fail(crate::Kind::Invalid, format!("bits {bits} is not 4, 8 or 16"));
        }
        match self.bits {
            Some(bits @ (4 | 8)) if self.rule().0.has("apple") => crate::failure::fail(crate::Kind::Invalid, format!(
                "precision {bits}: Apple GPUs run 16-bit weights until Metal kernels exist; run without --lane, or with --lane 16")),
            _ => Ok(()),
        }
    }

    /// The variant these take among `variants` and the reasons it is not their first choice,
    /// naming `model` (`org/name@release`): of those that work, the first in the lane order,
    /// then none narrower than the width ranked first, then native over weight-only, then fewer
    /// bytes, then path; one outside the order only when none in it works. Refused when none
    /// works, naming each variant and why.
    pub fn select(&self, model: &str, variants: &[(&str, &Profile)]) -> crate::Result<(usize, Vec<String>)> {
        self.check()?;
        let (gpus, slot, bits) = self.rule();
        let order = slot.lanes(bits, &gpus);
        let place = |p: &Profile| match row(&p.family).rank.as_str() {
            "after" => Some(order.len()),
            "" => order.iter().position(|lane| *lane == p.family),
            rank => order.iter().position(|lane| *lane == rank),
        };
        // Weight-only: a 4-bit lane whose matmuls run at 16 bits here.
        let native = |p: &Profile| !(p.bits == 4 && gpus.compute(&p.family) == 16);
        let works = variants.iter().enumerate().filter(|(_, (_, p))| self.refusal(p).is_none());
        let narrower = |p: &Profile| p.bits > 0 && p.bits < bits;
        let ranked = works.map(|(at, (name, p))| ((place(p).unwrap_or(order.len() + 1), narrower(p), !native(p), p.bytes, *name), at));
        if let Some((_, at)) = ranked.min() {
            return Ok((at, slot.warnings(bits, &gpus, model, &variants[at].1.family)));
        }
        let mut each: Vec<_> = variants.iter().map(|(name, p)| {
            let label = if p.family.is_empty() { p.kinds.join("+") } else { p.family.clone() };
            (*name, format!("{name} ({label}) {}", self.refusal(p).unwrap_or_default()))
        }).collect();
        each.sort();
        let published = if each.is_empty() { "none".into() } else { each.into_iter().map(|(_, said)| said).collect::<Vec<_>>().join("; ") };
        crate::failure::fail(crate::Kind::NotFound,
            format!("precision {bits} on {}: {model} has no variant that works; it publishes {published}", gpus.name))
    }

    /// The reasons `served`, which these selected, is not their first choice.
    pub fn warnings(&self, model: &str, served: &Profile) -> Vec<String> {
        let (gpus, slot, bits) = self.rule();
        slot.warnings(bits, &gpus, model, &served.family)
    }

    /// Their GPUs, the slot the lane order reads, and the bits it ranks first. Only a decode
    /// slot prefers 4 bits unasked, so a preferred 4 reads as one.
    fn rule(&self) -> (Gpus, Slot, u32) {
        let ccs: Vec<&str> = self.gpus.iter().map(String::as_str).collect();
        let decode = self.bits.is_none() && self.prefer == Some(4);
        let slot = Slot { min_bits: 4, decode, encoded: !self.plain, raw: false };
        (Gpus::of(&ccs), slot, self.bits.or(self.prefer).unwrap_or(8))
    }

    /// Why a variant of profile `p` does not work under these (spec §7), or `None`: another lane
    /// than the one asked, encoded weights where none are taken, narrower than `min_bits`
    /// unasked, or an encoding no one of the GPUs runs (with none, the CPU reads every one).
    fn refusal(&self, p: &Profile) -> Option<String> {
        let encoded: Vec<&str> = p.kinds.iter().map(String::as_str).filter(|kind| matches!(Dtype::parse(kind), Dtype::Other(_))).collect();
        let holds = || format!("holds encoded {}", encoded.join(", "));
        match self.bits {
            Some(bits) if p.bits != bits => return Some(format!("is not {bits}-bit")),
            Some(16) if !encoded.is_empty() => return Some(holds()),
            _ if self.plain && !encoded.is_empty() => return Some(holds()),
            None if self.min_bits.is_some_and(|min| p.bits > 0 && p.bits < min) => {
                return Some(format!("is {}-bit, under {}", p.bits, self.min_bits.unwrap_or_default()));
            }
            _ => {}
        }
        let unrun: Vec<String> = encoded.iter().filter(|kind| !self.gpus.is_empty() && !self.gpus.iter().any(|cc| runs(kind, &Gpus::of(&[cc]))))
            .map(|kind| format!("{} for {kind}", needs(kind))).collect();
        (!unrun.is_empty()).then(|| format!("needs {}", unrun.join(" and ")))
    }
}

/// This rule's version (the table's `rule`). An index answer of another (a Hub whose profiles an
/// older rule derived) is not taken: the headers are read instead.
pub fn rule() -> u32 {
    TABLE.rule
}

/// The Hub's indexed answer, with the same rule as local header selection. Only what selection
/// reads is decoded: the answer's other members (its release, the variant's profile) can change
/// shape without failing a fetch.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Selection {
    #[serde(default)]
    pub rule: u32,
    pub commit: crate::format::ObjectRef,
    pub variant: Option<String>,
    pub header: Option<crate::format::ObjectRef>,
    #[serde(default)]
    pub reasons: Vec<String>,
}
