//! Binding table tests over real release cards from tensorhub.com (`tests/fixtures/cards`).
use std::path::Path;

use serde_json::json;

use super::super::interface::{Callable, Interface, tests::interface};
use super::*;

/// Real files, recognized by their first bytes.
const PNG: &[u8] =
    b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR\0\0\0\x01\0\0\0\x01\x08\x06\0\0\0\x1f\x15\xc4\x89";
const MP4: &[u8] = b"\0\0\0\x18ftypmp42\0\0\0\0mp42isom\0\0\0\x08free";

struct Case {
    read: Interface,
    dir: tempfile::TempDir,
}

impl Case {
    fn new(card: &str) -> Self {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("cat.png"), PNG).unwrap();
        std::fs::write(dir.path().join("dog.png"), PNG).unwrap();
        std::fs::write(dir.path().join("clip.mp4"), MP4).unwrap();
        std::fs::write(dir.path().join("notes.txt"), "plain words").unwrap();
        Self {
            read: interface(card),
            dir,
        }
    }

    fn callable(&self, name: &str) -> &Callable {
        self.read.callables.iter().find(|c| c.name == name).unwrap()
    }

    fn terms(&self, inputs: &[&str]) -> Terms {
        Terms {
            inputs: inputs.iter().map(|s| (*s).to_owned()).collect(),
            key: "idem-test".into(),
            cwd: self.dir.path().to_owned(),
            user_home: self.dir.path().join("home"),
            ..Terms::default()
        }
    }

    fn bind(&self, function: &str, terms: &Terms) -> Result<Bound, Error> {
        bind(
            &format!("fidika/x/{function}"),
            self.callable(function),
            terms,
        )
    }

    fn ok(&self, function: &str, inputs: &[&str]) -> Bound {
        self.bind(function, &self.terms(inputs))
            .unwrap_or_else(|e| panic!("{inputs:?}: {e:?}"))
    }

    fn refused(&self, function: &str, inputs: &[&str]) -> Error {
        self.bind(function, &self.terms(inputs))
            .expect_err(&format!("{inputs:?} binds"))
    }

    fn path(&self, name: &str) -> String {
        self.dir.path().join(name).to_string_lossy().into_owned()
    }
}

fn digest_of(bytes: &[u8]) -> String {
    Digest::of(bytes).to_string()
}

#[test]
fn scalars_take_their_declared_types() {
    let sdxl = Case::new("sdxl-2.10.0");
    let bound = sdxl.ok(
        "generate",
        &[
            "a cat",
            "steps=20",
            "guidance=7",
            "seed=7",
            "aspect_ratio=16:9",
            "megapixels=2",
            "HIDIFFUSION=true",
        ],
    );
    assert_eq!(
        Value::Object(bound.payload.clone()),
        json!({"prompt": "a cat", "steps": 20, "guidance": 7.0, "seed": 7, "aspect_ratio": "16:9",
               "megapixels": 2, "hidiffusion": true})
    );
    assert!(bound.ignored.is_empty() && bound.files.is_empty() && bound.models.is_empty());
    assert_eq!(
        String::from_utf8(bound.canonical_payload()).unwrap(),
        r#"{"aspect_ratio":"16:9","guidance":7.0,"hidiffusion":true,"megapixels":2,"prompt":"a cat","seed":7,"steps":20}"#
    );
    assert_eq!(
        sdxl.ok("generate", &["prompt=x", "seed=null"]).payload["seed"],
        Value::Null
    );
    assert_eq!(
        sdxl.ok("generate", &["prompt=x", "steps:=12"]).payload["steps"],
        json!(12)
    );
    assert_eq!(
        sdxl.ok("generate", &["prompt=a=b"]).payload["prompt"],
        json!("a=b")
    );
    assert_eq!(
        sdxl.ok("generate", &["prompt=x", "negative_prompt=12"])
            .payload["negative_prompt"],
        json!("12")
    );
}

#[test]
fn unknown_fields_warn_at_every_depth_and_never_refuse() {
    let sdxl = Case::new("sdxl-2.10.0");
    let bound = sdxl.ok("generate", &["prompt=x", "style=noir", "Size:=3"]);
    assert_eq!(bound.ignored, ["Size", "style"]);
    assert!(!bound.payload.contains_key("style"));
    assert_eq!(
        bound.warning("fidika/sdxl/generate").unwrap(),
        "ignored unknown fields Size, style — not in fidika/sdxl/generate's interface"
    );
    let h3 = Case::new("minimax-h3-1.37.2");
    let bound = h3.ok(
        "long_form",
        &[
            r#"segments:=[{"prompt": "a", "duration_s": 15, "camera": "dolly"}]"#,
            r#"references:=[{"name": "<Ann>", "kind": "character", "mood": "x"}]"#,
            "style=noir",
        ],
    );
    assert_eq!(bound.ignored, ["references[0].mood", "segments[0].camera"]);
    assert_eq!(
        bound.payload["segments"],
        json!([{"prompt": "a", "duration_s": 15}])
    );
    assert_eq!(bound.payload["style"], json!("noir"));
    let upscale = h3.ok(
        "ref2va",
        &[
            "prompt=p",
            "duration_s=15",
            r#"upscale:={"color_correction": "lab", "gain": 2}"#,
        ],
    );
    assert_eq!(upscale.ignored, ["upscale.gain"]);
    assert_eq!(
        upscale.payload["upscale"],
        json!({"color_correction": "lab"})
    );
}

#[test]
fn types_required_fields_and_bounds_are_strict_and_named_together() {
    let sdxl = Case::new("sdxl-2.10.0");
    let missing = sdxl.refused("generate", &["steps=99", "megapixels:=3"]);
    assert_eq!(
        (missing.kind, missing.code.as_str()),
        (Kind::Validation, "request_payload_invalid")
    );
    assert!(
        missing
            .message
            .contains("provide required arguments: [prompt: str]"),
        "{}",
        missing.message
    );
    assert!(
        missing.message.contains("steps is 99; its maximum is 50"),
        "{}",
        missing.message
    );
    assert!(
        missing.message.contains("megapixels"),
        "{}",
        missing.message
    );
    assert!(
        missing
            .remedy
            .unwrap()
            .starts_with("cozy run fidika/x/generate prompt=<str>")
    );
    for (term, code) in [
        ("steps=many", "value_type"),
        ("hidiffusion=maybe", "value_type"),
        ("aspect_ratio=2:1", "value_type"),
        ("guidance=inf", "value_type"),
    ] {
        assert_eq!(
            sdxl.refused("generate", &["prompt=x", term]).code,
            code,
            "{term}"
        );
    }
    assert_eq!(
        sdxl.refused("generate", &["prompt=x", "steps:={"]).code,
        "json_malformed"
    );
    assert_eq!(
        sdxl.refused("generate", &["one", "two"]).code,
        "positional_extra"
    );
    let h3 = Case::new("minimax-h3-1.37.2");
    let short = h3.refused("ref2va", &["prompt=p", "duration_s=20"]);
    assert!(
        short
            .message
            .contains("duration_s is 20; its maximum is 15"),
        "{}",
        short.message
    );
    let shape = h3.refused(
        "long_form",
        &[
            r#"segments:=[{"prompt": 3, "duration_s": 15}]"#,
            r#"references:=[]"#,
        ],
    );
    assert!(
        shape.message.contains("segments[0].prompt is not str"),
        "{}",
        shape.message
    );
    assert!(
        shape
            .message
            .contains("references has length 0; its minimum is 1"),
        "{}",
        shape.message
    );
}

#[test]
fn model_choices_and_loras_keep_their_slots_and_order() {
    let sdxl = Case::new("sdxl-2.10.0");
    let terms = Terms {
        loras: vec![
            "model=fidika/third,.5".into(),
            "model:unet=fidika/detail-lora,0.80".into(),
            "model:text_encoder=fidika/style@1.0,0".into(),
        ],
        ..sdxl.terms(&["prompt=x", "model.model=fidika/sdxl-base@1.0/int8"])
    };
    let bound = sdxl.bind("generate", &terms).unwrap();
    assert_eq!(
        bound.models,
        [Choice {
            slot: "model".into(),
            reference: Some("fidika/sdxl-base@1.0/int8".into()),
            adapters: vec![
                Adapter {
                    component: None,
                    reference: "fidika/third".into(),
                    scale: "0.5".into()
                },
                Adapter {
                    component: Some("unet".into()),
                    reference: "fidika/detail-lora".into(),
                    scale: "0.8".into()
                },
                Adapter {
                    component: Some("text_encoder".into()),
                    reference: "fidika/style@1.0".into(),
                    scale: "0".into()
                },
            ],
        }]
    );
    let unknown = sdxl.refused("generate", &["prompt=x", "model.unet=fidika/x"]);
    assert_eq!(
        (unknown.code.as_str(), unknown.remedy.as_deref()),
        ("model_slot_unknown", Some("generate declares: model"))
    );
    assert_eq!(
        sdxl.refused(
            "generate",
            &["prompt=x", "model.model=a/b", "model.model=c/d"]
        )
        .code,
        "model_slot_repeated"
    );
    for (lora, code) in [
        ("model=a/b,nan", "lora_strength_invalid"),
        ("model:=a/b", "lora_malformed"),
        ("model", "lora_malformed"),
        ("unet:x=a/b", "model_slot_unknown"),
    ] {
        let terms = Terms {
            loras: vec![lora.into()],
            ..sdxl.terms(&["prompt=x"])
        };
        assert_eq!(
            sdxl.bind("generate", &terms).unwrap_err().code,
            code,
            "{lora}"
        );
    }
    let file = sdxl.dir.path().join("models.json");
    std::fs::write(
        &file,
        r#"{"prompt": "x", "models": {"model": {"ref": "fidika/base@2", "lora": [{"ref": "a/b", "weight": 0.25}]}}}"#,
    )
    .unwrap();
    let enveloped = sdxl
        .bind(
            "generate",
            &Terms {
                input_file: Some(file),
                ..sdxl.terms(&[])
            },
        )
        .unwrap();
    assert_eq!(
        enveloped.models[0].reference.as_deref(),
        Some("fidika/base@2")
    );
    assert_eq!(enveloped.models[0].adapters[0].scale, "0.25");
    assert!(!enveloped.payload.contains_key("models"));
    let child = sdxl.ok(
        "generate",
        &["prompt=x", "fidika/flux/edit.models.upsampler=fidika/up@2"],
    );
    assert!(
        child
            .ignored
            .contains(&"fidika/flux/edit.models.upsampler".to_owned())
    );
    let child = sdxl.ok(
        "generate",
        &[
            "prompt=x",
            "model.fidika/flux/edit.models.upsampler=fidika/up@2",
        ],
    );
    assert_eq!(child.models[0].slot, "fidika/flux/edit.models.upsampler");
}

#[test]
fn a_conversion_reads_its_input_model_and_destination() {
    let sdxl = Case::new("sdxl-2.10.0");
    let bound = sdxl.ok("int8", &["fidika/sdxl-base@1.0", "Paul/SDXL-int8"]);
    assert_eq!(bound.models[0].slot, "source");
    assert_eq!(
        bound.models[0].reference.as_deref(),
        Some("fidika/sdxl-base@1.0")
    );
    assert_eq!(bound.destination.as_deref(), Some("paul/sdxl-int8"));
    assert_eq!(
        sdxl.ok(
            "fp8",
            &["hf://stabilityai/sdxl@abc", "max_relative_frobenius=0.02"]
        )
        .payload["max_relative_frobenius"],
        json!(0.02)
    );
    assert_eq!(
        sdxl.refused("int8", &["a/b", "c/d", "e/f"]).code,
        "conversion_positionals"
    );
    assert_eq!(
        sdxl.refused("int8", &["a/b", "not-a-model"]).code,
        "destination_malformed"
    );
}

#[test]
fn files_bind_by_name_label_collection_and_media_kind() {
    let h3 = Case::new("minimax-h3-1.37.2");
    let cat = h3.path("cat.png");
    let terms = Terms {
        assets: vec![format!("ann={cat}"), "./dog.png".into()],
        ..h3.terms(&["prompt=<Ann> meets a dog", "duration_s=15", "./clip.mp4"])
    };
    let bound = h3.bind("ref2va", &terms).unwrap();
    let png = digest_of(PNG);
    assert_eq!(
        bound.payload["assets"],
        json!([{"asset": png, "label": "ann"}, {"asset": png}, {"asset": digest_of(MP4)}])
    );
    let fields: Vec<(&str, &str, u32)> = bound
        .files
        .iter()
        .map(|f| (f.field.as_str(), f.media_type.as_str(), f.order))
        .collect();
    assert_eq!(
        fields,
        [
            ("assets.0.asset", "image/png", 0),
            ("assets.1.asset", "image/png", 1),
            ("assets.2.asset", "video/mp4", 2)
        ]
    );
    assert_eq!(bound.files[1].path, Path::new(&h3.path("dog.png")));
    assert_eq!(bound.files[0].length, PNG.len() as u64);

    let birefnet = Case::new("birefnet-0.2.0");
    let image = birefnet.ok("remove_background", &["cat.png"]);
    assert_eq!(image.payload["image"], json!(png));
    assert_eq!(image.files[0].field, "image");
    let named = birefnet
        .bind(
            "remove_background",
            &Terms {
                assets: vec!["image=cat.png".into()],
                ..birefnet.terms(&[])
            },
        )
        .unwrap();
    assert_eq!(named.payload, image.payload);
    assert_eq!(
        birefnet.refused("remove_background", &["./notes.txt"]).code,
        "positional_file_unbound"
    );
    assert_eq!(
        birefnet
            .refused("remove_background", &["./missing.png"])
            .code,
        "input_file_absent"
    );
    assert_eq!(
        birefnet.refused("remove_background", &["./clip.mp4"]).code,
        "positional_file_unbound"
    );

    let flux = Case::new("flux-3-klein-0.5.0");
    let edit = flux.ok("edit", &["prompt=swap them", "cat.png", "dog.png"]);
    assert_eq!(edit.payload["reference_images"], json!([png, png]));
    assert_eq!(
        edit.files
            .iter()
            .map(|f| f.field.as_str())
            .collect::<Vec<_>>(),
        ["reference_images.0", "reference_images.1"]
    );

    let seedvr2 = Case::new("seedvr2-0.4.0");
    let upscale = seedvr2.ok("upscale", &["clip.mp4", "resolution=1440p"]);
    assert_eq!(
        (
            upscale.files[0].field.as_str(),
            upscale.files[0].media_type.as_str()
        ),
        ("video", "video/mp4")
    );
    assert_eq!(
        seedvr2.refused("upscale", &["cat.png"]).code,
        "positional_file_unbound"
    );
}

#[test]
fn file_bounds_labels_and_counts_hold_before_any_run() {
    let h3 = Case::new("minimax-h3-1.37.2");
    let twice = Terms {
        assets: vec!["a=cat.png".into(), "a=dog.png".into()],
        ..h3.terms(&["prompt=p", "duration_s=5"])
    };
    assert_eq!(
        h3.bind("ref2va", &twice).unwrap_err().code,
        "asset_label_repeated"
    );
    let fl2va = Terms {
        assets: vec!["cat.png".into(), "dog.png".into(), "cat.png".into()],
        ..h3.terms(&["prompt=p", "duration_s=5"])
    };
    assert_eq!(
        h3.bind("fl2va", &fl2va).unwrap_err().code,
        "input_asset_count"
    );
    let text = Terms {
        assets: vec!["notes.txt".into()],
        ..h3.terms(&["prompt=p", "duration_s=5"])
    };
    assert_eq!(
        h3.bind("ref2va", &text).unwrap_err().code,
        "asset_media_type"
    );
    let sdxl = Case::new("sdxl-2.10.0");
    let none = Terms {
        assets: vec!["cat.png".into()],
        ..sdxl.terms(&["prompt=p"])
    };
    assert_eq!(
        sdxl.bind("generate", &none).unwrap_err().code,
        "assets_undeclared"
    );
    let named = Terms {
        assets: vec!["prompt=cat.png".into()],
        ..sdxl.terms(&["prompt=p"])
    };
    assert_eq!(
        sdxl.bind("generate", &named).unwrap_err().code,
        "asset_field_unknown"
    );
}

#[test]
fn the_input_file_is_the_payload_and_terms_override_it() {
    let h3 = Case::new("minimax-h3-1.37.2");
    let yaml = h3.dir.path().join("request.yaml");
    std::fs::write(
        &yaml,
        "segments:\n  - prompt: on\n    duration_s: 15\n    seed: 9007199254740993\nreferences:\n  - &ann {name: <Ann>, kind: character}\nstyle: yes\nmode: standard\n",
    )
    .unwrap();
    let terms = Terms {
        input_file: Some(yaml),
        ..h3.terms(&["mode=turbo"])
    };
    let bound = h3.bind("long_form", &terms).unwrap();
    assert_eq!(
        bound.payload["segments"][0]["seed"],
        json!(9007199254740993u64)
    );
    assert_eq!(bound.payload["style"], json!("yes"));
    assert_eq!(bound.payload["mode"], json!("turbo"));
    let json_file = h3.dir.path().join("ref.json");
    std::fs::write(
        &json_file,
        r#"{"Prompt": "p", "duration_s": 5, "assets": [{"asset": "cat.png", "label": "c"}]}"#,
    )
    .unwrap();
    let bound = h3
        .bind(
            "ref2va",
            &Terms {
                input_file: Some(json_file),
                ..h3.terms(&[])
            },
        )
        .unwrap();
    assert_eq!(bound.payload["prompt"], json!("p"));
    assert_eq!(
        bound.payload["assets"],
        json!([{"asset": digest_of(PNG), "label": "c"}])
    );
    assert_eq!(bound.files[0].path, h3.dir.path().join("cat.png"));
}

#[test]
fn an_omitted_integer_seed_follows_the_key() {
    let seedvr2 = Case::new("seedvr2-0.4.0");
    let first = seedvr2.ok("upscale", &["clip.mp4"]);
    let again = seedvr2.ok("upscale", &["clip.mp4"]);
    let seed = first.payload["seed"].as_u64().unwrap();
    assert!(seed < 1 << 53);
    assert_eq!(again.payload, first.payload);
    let other = seedvr2
        .bind(
            "upscale",
            &Terms {
                key: "idem-other".into(),
                ..seedvr2.terms(&["clip.mp4"])
            },
        )
        .unwrap();
    assert_ne!(other.payload["seed"], first.payload["seed"]);
    assert_eq!(
        seedvr2.ok("upscale", &["clip.mp4", "seed=5"]).payload["seed"],
        json!(5)
    );
    let sdxl = Case::new("sdxl-2.10.0");
    assert!(
        !sdxl
            .ok("generate", &["prompt=x"])
            .payload
            .contains_key("seed")
    );
}

#[test]
fn a_warm_run_takes_only_model_choices() {
    let sdxl = Case::new("sdxl-2.10.0");
    let choices = warm_choices(
        sdxl.callable("generate"),
        &sdxl.terms(&["model.model=fidika/base@2"]),
    )
    .unwrap();
    assert_eq!(choices[0].reference.as_deref(), Some("fidika/base@2"));
    assert!(
        warm_choices(sdxl.callable("generate"), &sdxl.terms(&[]))
            .unwrap()
            .is_empty()
    );
    let refused = warm_choices(sdxl.callable("generate"), &sdxl.terms(&["prompt=x"])).unwrap_err();
    assert_eq!(refused.code, "warm_inputs");
}
