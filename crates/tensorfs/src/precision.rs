//! The lane rule (xs-042, design `precision-tiers.md` §2.2, §2.7, §2.8; spec §7 "Selection"): a
//! variant from what its header holds, its GPUs' native formats, the request's bits and a
//! slot's declared tolerances. Nothing is keyed by a card's name or a file's.
use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::format::{Body, Dtype, Header};

/// The lanes the rule picks by name; any other is a release's only lane, served silently.
const VOCABULARY: [&str; 5] = ["bf16", "fp16", "int8", "fp8", "nvfp4"];

/// The GPUs a function runs on as the rule reads them (design §2.1, §2.7): the native formats
/// every one runs (`mma`: cc 8.0 or later, where `nvfp4` serves at least weight-only; 10.3 has
/// no int8; `bf16`: unless a GPU is known below cc 8.0; `fp4`: FP4 tensor cores, cc 10.0 or later)
/// and how a warning names them (`sm90`, `sm86+sm89`).
#[derive(Clone, Debug, PartialEq)]
pub struct Gpus {
    pub mma: bool,
    pub bf16: bool,
    pub int8: bool,
    pub fp8: bool,
    pub fp4: bool,
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
    /// From each GPU's compute capability (`"9.0"`); one below 8.0 or unread runs none.
    pub fn of(ccs: &[&str]) -> Self {
        let number = |cc: &str| cc.split_once('.').and_then(|(major, minor)| Some(major.parse::<u32>().ok()? * 10 + minor.parse::<u32>().ok()?));
        let all = |runs: fn(u32) -> bool| !ccs.is_empty() && ccs.iter().all(|cc| runs(number(cc).unwrap_or(0)));
        let mut names: Vec<String> = ccs.iter().filter(|cc| !cc.is_empty()).map(|cc| format!("sm{}", cc.replace('.', ""))).collect();
        names.sort();
        names.dedup();
        let name = if names.is_empty() { "no GPU".into() } else { names.join("+") };
        Self {
            mma: all(|cc| cc >= 80),
            // fp16 first only where a GPU is known to run no bf16: an unread cc, or no GPU, runs bf16.
            bf16: !ccs.iter().any(|cc| matches!(number(cc), Some(n) if n < 80)),
            int8: all(|cc| cc >= 80 && cc != 103),
            fp8: all(|cc| cc >= 89),
            fp4: all(|cc| cc >= 100),
            name,
            capabilities: ccs.iter().map(|cc| cc.to_string()).collect(),
        }
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
    /// decode slot or a function whose 8-bit weights exceed the budget (`short`); else 8.
    pub fn bits(&self, asked: u32, short: bool) -> u32 {
        match asked {
            _ if self.raw => 16,
            4 | 8 | 16 => asked,
            _ if self.min_bits == 16 => 16,
            _ if self.min_bits == 4 && (self.decode || short) => 4,
            _ => 8,
        }
    }

    /// The lanes to ask for at `bits`, in order (§2.2): the width's native formats (4 bits: `nvfp4`
    /// first, then the 8-bit order), then 16 bits: `bf16`, and `fp16` first only where the GPUs
    /// run no bf16 (owner, 2026-10-16). Encoded lanes go when the slot declines them.
    pub fn lanes(&self, bits: u32, gpus: &Gpus) -> Vec<&'static str> {
        let eight = [("int8", gpus.int8), ("fp8", gpus.fp8)];
        let native: Vec<_> = match bits {
            16 => vec![],
            4 => [("nvfp4", gpus.mma)].into_iter().chain(eight).collect(),
            _ => eight.to_vec(),
        };
        let native = native.into_iter().filter(|(_, runs)| *runs && self.encoded).map(|(lane, _)| lane);
        native.chain(if gpus.bf16 { ["bf16", "fp16"] } else { ["fp16", "bf16"] }).collect()
    }

    /// The run's warnings for `model` served in `served` (§2.8): one when it declined encoded
    /// lanes the GPUs run, one naming every lane asked before the one served, and fp16 served
    /// because the GPUs run no bf16 says so.
    pub fn warnings(&self, bits: u32, gpus: &Gpus, model: &str, served: &str) -> Vec<String> {
        if self.raw || !VOCABULARY.contains(&served) {
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
        let no_bf16 = (served == "fp16" && !gpus.bf16).then(|| format!("precision {bits}: {} runs no bf16; served fp16", gpus.name));
        declined.into_iter().chain(missing).chain(no_bf16).collect()
    }
}

impl Slot {
    /// A reader's slot (`open`, `cozy model download`): 8 bits by default, encoded leaves taken.
    pub const ANY: Slot = Slot { min_bits: 8, decode: false, encoded: true, raw: false };

    /// The variant this slot takes at `bits` (spec §7 "Selection"): of those whose family the
    /// order of `lanes` places and whose every kind runs on `gpus`, the first by that place, then
    /// native over weight-only, then fewer bytes, then name; a commit's only variant otherwise.
    pub fn select(&self, bits: u32, gpus: &Gpus, variants: &[(&str, &Profile)]) -> Option<usize> {
        let order = self.lanes(bits, gpus);
        let place = |p: &Profile| match p.family.as_str() {
            "f32" => Some(order.len()),
            "mxfp4" | "int4" => order.iter().position(|lane| *lane == "nvfp4"),
            family => order.iter().position(|lane| *lane == family),
        };
        let native = |p: &Profile| match p.family.as_str() {
            "nvfp4" | "mxfp4" => gpus.fp4,
            "int4" => false,
            _ => true,
        };
        let ranked = variants.iter().enumerate().filter(|(_, (_, p))| p.kinds.iter().all(|kind| runs(kind, gpus)));
        let ranked = ranked.filter_map(|(at, (name, p))| Some(((place(p)?, !native(p), p.bytes, *name), at)));
        ranked.min().map(|(_, at)| at).or_else(|| {
            (variants.len() == 1 && variants[0].1.kinds.iter().all(|kind| runs(kind, gpus) && (self.encoded || !matches!(Dtype::parse(kind), Dtype::Other(_))))).then_some(0)
        })
    }
}

/// A variant as the rule reads it, from its header alone (spec §7): every kind it holds (an
/// encoded tensor's scheme, a plain one's dtype), its stored bytes, and the family and width of
/// the kind holding most of them (`""`, 0 outside the table).
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
        // The most bytes; of equal ones, the first name.
        let dominant = held.iter().fold(None, |most: Option<(&String, u64)>, (kind, &n)| match most {
            Some((_, m)) if m >= n => most,
            _ => Some((kind, n)),
        });
        let (family, bits) = dominant.map_or(("", 0), |(kind, _)| family(kind));
        Self { bits, family: family.into(), bytes: held.values().sum(), kinds: held.into_keys().collect() }
    }
}

/// A kind's family and width (spec §7's table); `("", 0)` outside it.
fn family(kind: &str) -> (&'static str, u32) {
    match kind {
        "BF16" => ("bf16", 16),
        "F16" => ("fp16", 16),
        "F32" => ("f32", 16),
        "nvfp4" => ("nvfp4", 4),
        "mxfp4" => ("mxfp4", 4),
        k if k.starts_with("int8.") => ("int8", 8),
        k if k.starts_with("fp8.") => ("fp8", 8),
        k if k.starts_with("int4.") => ("int4", 4),
        _ => ("", 0),
    }
}

/// Whether every GPU runs `kind`: a §5 dtype always; an encoding of the table where its family
/// does; any other never.
fn runs(kind: &str, gpus: &Gpus) -> bool {
    match family(kind).0 {
        "int8" => gpus.int8,
        "fp8" => gpus.fp8,
        "nvfp4" | "mxfp4" | "int4" => gpus.mma,
        _ => !matches!(Dtype::parse(kind), Dtype::Other(_)),
    }
}

/// What a reader asks of a variant (spec §7): the GPUs it runs on, by compute capability
/// (`"8.9"`; none is the CPU), and its bits (4, 8 or 16; none: 8).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Constraints {
    #[serde(default)]
    pub gpus: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bits: Option<u32>,
}

impl Constraints {
    /// The variant these take among `variants` and the reasons it is not their first choice,
    /// naming `model` (`org/name@release`); `None` when none is selectable.
    pub fn select(&self, model: &str, variants: &[(&str, &Profile)]) -> crate::Result<Option<(usize, Vec<String>)>> {
        if self.bits.is_some_and(|bits| ![4, 8, 16].contains(&bits)) {
            return crate::failure::fail(crate::Kind::Invalid, format!("bits {} is not 4, 8 or 16", self.bits.unwrap_or_default()));
        }
        let ccs: Vec<&str> = self.gpus.iter().map(String::as_str).collect();
        let (gpus, bits) = (Gpus::of(&ccs), Slot::ANY.bits(self.bits.unwrap_or(0), false));
        Ok(Slot::ANY.select(bits, &gpus, variants).map(|at| (at, Slot::ANY.warnings(bits, &gpus, model, &variants[at].1.family))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Each row's lane for a compute slot, a `min_bits=4` decode slot, `--bits=4` and
    /// `--bits=16` (design §2.2's derived table).
    #[test]
    fn one_lane_per_row() {
        let compute = Slot::of(&json!({"encoded_leaves": "accept"}));
        let decode = Slot::of(&json!({"encoded_leaves": "accept", "min_bits": 4, "phase": "decode"}));
        let first = |slot: &Slot, asked, cc: &[&str]| slot.lanes(slot.bits(asked, false), &Gpus::of(cc))[0];
        for (cc, eight) in [("8.0", "int8"), ("8.6", "int8"), ("8.9", "int8"), ("9.0", "int8"), ("10.0", "int8"),
            ("10.3", "fp8"), ("12.0", "int8"), ("12.1", "int8"), ("11.0", "int8")] {
            assert_eq!(first(&compute, 0, &[cc]), eight, "{cc}");
            assert_eq!(first(&decode, 0, &[cc]), "nvfp4", "{cc}");
            assert_eq!(first(&compute, 4, &[cc]), "nvfp4", "{cc}");
            assert_eq!(first(&compute, 16, &[cc]), "bf16", "{cc}");
        }
        // bf16 everywhere it runs (cc 8.0 or later, or no GPU); fp16 first only below 8.0.
        for (none, sixteen) in [(&["7.5"][..], "fp16"), (&[""], "bf16"), (&[], "bf16")] {
            for (slot, asked) in [(&compute, 0), (&decode, 0), (&compute, 4), (&compute, 16)] {
                assert_eq!(first(slot, asked, none), sixteen, "{none:?}");
            }
        }
        assert_eq!(compute.lanes(8, &Gpus::of(&["9.0"])), ["int8", "fp8", "bf16", "fp16"]);
        assert_eq!(compute.lanes(4, &Gpus::of(&["8.6"])), ["nvfp4", "int8", "bf16", "fp16"]);
        assert_eq!(compute.lanes(4, &Gpus::of(&["12.0"])), ["nvfp4", "int8", "fp8", "bf16", "fp16"]);
        assert_eq!(compute.lanes(4, &Gpus::of(&["12.1"])), ["nvfp4", "int8", "fp8", "bf16", "fp16"], "GB10 proved fp8");
        let exact = Slot::of(&json!({"encoded_leaves": "accept", "min_bits": 16}));
        assert_eq!((exact.bits(0, true), exact.bits(8, false)), (16, 8), "the request's bits win");
        let raw = Slot::of(&json!({"encoded_leaves": "accept", "min_bits": 4, "phase": "decode", "weights": true}));
        assert_eq!((raw.bits(4, true), raw.lanes(raw.bits(8, false), &Gpus::of(&["10.0"]))), (16, vec!["bf16", "fp16"]), "raw weights are data");
        assert!(raw.warnings(16, &Gpus::of(&["9.0"]), "x@1", "fp16").is_empty());
    }

    /// A set's row is each column ANDed: a mixed local machine runs only what every GPU runs.
    #[test]
    fn a_mixed_set_runs_what_every_gpu_runs() {
        let slot = Slot::of(&json!({"encoded_leaves": "accept"}));
        let ampere_ada = Gpus::of(&["8.6", "8.9"]);
        assert_eq!(ampere_ada.name, "sm86+sm89");
        assert_eq!(slot.lanes(8, &ampere_ada), ["int8", "bf16", "fp16"]);
        assert_eq!(slot.lanes(8, &Gpus::of(&["10.0", "10.3"])), ["fp8", "bf16", "fp16"]);
        assert_eq!(slot.lanes(8, &Gpus::of(&["9.0", "7.5"])), ["fp16", "bf16"]);
        assert_eq!(Gpus::of(&["9.0", "9.0"]).name, "sm90");
    }

    /// A fallback warns once, naming every lane it lacked; a slot declining encoded leaves warns
    /// once; a served lane outside the vocabulary and an exact slot say nothing.
    #[test]
    fn steps_and_declines_warn() {
        let (hopper, ada) = (Gpus::of(&["9.0"]), Gpus::of(&["8.9"]));
        let slot = Slot::of(&json!({"encoded_leaves": "accept"}));
        assert_eq!(slot.warnings(4, &ada, "fidika/sdxl@1.0.0", "int8"),
            ["precision 4: fidika/sdxl@1.0.0 has no nvfp4 lane for sm89; served int8"]);
        assert_eq!(slot.warnings(8, &hopper, "x@1", "bf16"), ["precision 8: x@1 has no int8 or fp8 lane for sm90; served bf16"]);
        assert_eq!(slot.warnings(4, &ada, "x@1", "fp16"), ["precision 4: x@1 has no nvfp4, int8, fp8 or bf16 lane for sm89; served fp16"]);
        assert!(slot.warnings(8, &hopper, "x@1", "int8").is_empty());
        assert!(slot.warnings(8, &hopper, "x@1", "pdd8").is_empty());
        let declines = Slot::of(&json!({}));
        assert_eq!(declines.lanes(8, &hopper), ["bf16", "fp16"]);
        assert_eq!(declines.warnings(8, &hopper, "x@1", "bf16"), ["precision 8: x@1 declines encoded leaves; served bf16"]);
        assert!(declines.warnings(16, &hopper, "x@1", "bf16").is_empty());
        assert!(declines.warnings(8, &Gpus::of(&[]), "x@1", "bf16").is_empty());
        assert_eq!(Gpus::default(), Gpus::of(&[]), "no GPU service is the CPU: bf16 first");
        // fp16 says why: the GPUs run no bf16, or the release has no bf16 lane.
        let turing = Gpus::of(&["7.5"]);
        assert_eq!(slot.warnings(8, &turing, "x@1", "fp16"), ["precision 8: sm75 runs no bf16; served fp16"]);
        assert_eq!(slot.warnings(8, &turing, "x@1", "bf16"), ["precision 8: x@1 has no fp16 lane for sm75; served bf16"]);
        assert_eq!(slot.warnings(16, &hopper, "sdxl@1", "fp16"), ["precision 16: sdxl@1 has no bf16 lane for sm90; served fp16"]);
    }
}

/// The Hub's indexed answer, with the same rule as local header selection.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Selection {
    pub release: String,
    pub commit: crate::format::ObjectRef,
    pub variant: Option<String>,
    pub header: Option<crate::format::ObjectRef>,
    pub profile: Option<Profile>,
    #[serde(default)]
    pub reasons: Vec<String>,
}
