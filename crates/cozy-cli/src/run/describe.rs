//! The one contract printer: the usage line a refusal's remedy names, `--describe`, and a
//! package's function list. No other module renders a request schema.
use std::fmt::Write as _;

use serde_json::Value;

use super::interface::{Callable, Constraints, Field, Interface, Type, Wire};

/// One concrete invocation: required terms bare, optional ones bracketed.
pub fn usage_line(target: &str, callable: &Callable) -> String {
    let mut line = format!("cozy run {target}");
    if callable.is_conversion() {
        let _ = write!(line, " <{}> [<org/model>]", callable.models[0].param);
        for slot in &callable.models[1..] {
            let _ = write!(line, " model.{}=<ref>", slot.param);
        }
    }
    for field in &callable.request.fields {
        line.push(' ');
        line.push_str(&term(callable, field));
    }
    line
}

fn term(callable: &Callable, field: &Field) -> String {
    if callable.is_assets(&field.name) {
        return if field.constraints.min_length.unwrap_or(0) > 0 {
            "--asset <file> [--asset <file>]...".into()
        } else {
            "[--asset <file>]...".into()
        };
    }
    let term = match &field.ty {
        Type::Asset(_) => format!("--asset {}=<file>", field.name),
        Type::Str
        | Type::Int
        | Type::Float
        | Type::Bool
        | Type::Null
        | Type::Literal(_)
        | Type::Union { .. }
        | Type::Tree => {
            format!("{}=<{}>", field.name, phrase(&field.ty))
        }
        _ => format!("{}:=<json>", field.name),
    };
    if field.wire == Wire::Required {
        term
    } else {
        format!("[{term}]")
    }
}

/// A type in one compact phrase: `int`, `1:1|16:9`, `int|null`, `list[image-file]`.
pub fn phrase(ty: &Type) -> String {
    match ty {
        Type::Str => "str".into(),
        Type::Int => "int".into(),
        Type::Float => "float".into(),
        Type::Bool => "bool".into(),
        Type::Null => "null".into(),
        Type::Literal(members) => members
            .iter()
            .map(|m| m.as_str().map_or_else(|| m.to_string(), str::to_owned))
            .collect::<Vec<_>>()
            .join("|"),
        Type::Union { branches, .. } => branches.iter().map(phrase).collect::<Vec<_>>().join("|"),
        Type::List(item) => format!("list[{}]", phrase(item)),
        Type::Map(item) => format!("map[{}]", phrase(item)),
        Type::Struct(_) => "object".into(),
        Type::Asset(kind) => format!("{kind}-file"),
        Type::Tree => "tree-ref".into(),
        Type::Model => "model".into(),
        Type::Unknown => "json".into(),
    }
}

/// A field as people read it: name, type and bounds.
pub fn signature(field: &Field) -> String {
    let mut out = format!("{}: ", field.name);
    match &field.ty {
        Type::Asset(kind) if !field.asset_bound.media_types.is_empty() => {
            let _ = write!(
                out,
                "{kind} file ({})",
                field.asset_bound.media_types.join(", ")
            );
        }
        Type::Asset(kind) => {
            let _ = write!(out, "{kind} file");
        }
        ty => out.push_str(&phrase(ty)),
    }
    if let Some(bounds) = bounds(&field.constraints) {
        let _ = write!(out, " ({bounds})");
    }
    out
}

fn bounds(c: &Constraints) -> Option<String> {
    let mut parts = Vec::new();
    for (bound, says) in [(c.gt, ">"), (c.ge, ">="), (c.lt, "<"), (c.le, "<=")] {
        if let Some(bound) = bound {
            parts.push(format!("{says}{bound}"));
        }
    }
    if let Some(min) = c.min_length {
        parts.push(format!("min length {min}"));
    }
    if let Some(max) = c.max_length {
        parts.push(format!("max length {max}"));
    }
    (!parts.is_empty()).then(|| parts.join(", "))
}

/// The callable's arguments, one per line, as a refusal shows them.
pub fn arguments(callable: &Callable) -> String {
    let mut out = String::from("Arguments:\n");
    fields(
        &mut out,
        callable,
        &callable.request.fields,
        "  ",
        "request",
        true,
    );
    out
}

/// `--describe`: where it runs, request, model slots with their defaults, output, and the
/// usage line.
pub fn contract(target: &str, interface: &Interface, callable: &Callable) -> String {
    let mut out = target.to_owned();
    if callable.accelerator == Some(false) {
        // An internal function exists only to be called: a CPU function of a package with
        // internal GPU ones orchestrates them (H3's ref2va and long_form).
        let gpu_calls =
            (interface.callables.iter()).any(|c| c.internal && c.accelerator != Some(false));
        out.push_str(match gpu_calls {
            true => "\n  runs on: CPU; calls the package's GPU functions",
            false => "\n  runs on: CPU",
        });
    }
    out.push('\n');
    if callable.request.fields.is_empty() {
        out.push_str("  request: none\n");
    } else {
        out.push_str("  request:\n");
        fields(
            &mut out,
            callable,
            &callable.request.fields,
            "    ",
            "request",
            true,
        );
    }
    if !callable.models.is_empty() {
        out.push_str("  models:\n");
        for slot in &callable.models {
            let _ = write!(out, "    {}: {}", slot.param, slot.class);
            if slot.weights {
                out.push_str(" weights");
            }
            let mut notes = Vec::new();
            if let Some(default) = &slot.default {
                notes.push(format!("default {default}"));
            }
            if let Some(bits) = slot.min_bits {
                notes.push(format!("at least {bits} bits"));
            }
            if let Some(phase) = &slot.phase {
                notes.push(phase.clone());
            }
            if !notes.is_empty() {
                let _ = write!(out, " ({})", notes.join("; "));
            }
            out.push('\n');
        }
    }
    if !callable.weights_outputs.is_empty() {
        let written: Vec<&str> = callable
            .weights_outputs
            .iter()
            .map(|w| w.output_id.as_str())
            .collect();
        let _ = writeln!(out, "  writes weights: {}", written.join(", "));
    }
    if callable.result.input.as_deref() == Some("model") {
        out.push_str("  output: a model\n");
    } else if !callable.result.fields.is_empty() {
        out.push_str("  output:\n");
        fields(
            &mut out,
            callable,
            &callable.result.fields,
            "    ",
            "result",
            false,
        );
    }
    let _ = writeln!(out, "\nusage: {}", usage_line(target, callable));
    out
}

fn fields(
    out: &mut String,
    callable: &Callable,
    fields: &[Field],
    indent: &str,
    path: &str,
    request: bool,
) {
    for field in fields {
        let _ = write!(out, "{indent}{}", signature(field));
        let defaults = &callable.invocable.defaults;
        let default = defaults
            .get(&format!("{path}/{}", field.name))
            .or(field.default.as_ref());
        match default {
            Some(value) => {
                let _ = write!(out, " (default = {})", spell(value));
            }
            None if request
                && callable.is_assets(&field.name)
                && field.constraints.min_length.unwrap_or(0) == 0 =>
            {
                out.push_str(" (default = [])");
            }
            None if field.wire == Wire::Optional => out.push_str(" (optional)"),
            None if field.wire == Wire::Omissible => out.push_str(" (optional; model default)"),
            None => {}
        }
        out.push('\n');
        if let Type::Struct(nested) = &field.ty
            && indent.len() < 12
        {
            self::fields(
                out,
                callable,
                &nested.fields,
                &format!("{indent}  "),
                &format!("{path}/{}", field.name),
                request,
            );
        }
    }
}

fn spell(value: &Value) -> String {
    value.to_string()
}

#[cfg(test)]
mod tests {
    use super::super::interface::tests::interface;
    use super::*;

    fn callable<'a>(read: &'a super::super::interface::Interface, name: &str) -> &'a Callable {
        read.callables.iter().find(|c| c.name == name).unwrap()
    }

    #[test]
    fn usage_lines_spell_every_term_in_its_grammar() {
        let sdxl = interface("sdxl-2.10.0");
        assert_eq!(
            usage_line("fidika/sdxl/generate", callable(&sdxl, "generate")),
            "cozy run fidika/sdxl/generate prompt=<str> [negative_prompt=<str>] [aspect_ratio=<16:9|1:1|3:4|4:3|9:16>] \
             [megapixels=<1|16|2|4|8>] [hidiffusion=<bool|null>] [steps=<int>] [guidance=<float>] [seed=<int|null>]"
        );
        assert_eq!(
            usage_line("fidika/sdxl/int8", callable(&sdxl, "int8")),
            "cozy run fidika/sdxl/int8 <source> [<org/model>]"
        );
        let h3 = interface("minimax-h3-1.37.2");
        let ref2va = usage_line("fidika/minimax-h3/ref2va", callable(&h3, "ref2va"));
        assert!(ref2va.contains(" prompt=<str> "), "{ref2va}");
        assert!(ref2va.contains(" duration_s=<int> "), "{ref2va}");
        assert!(ref2va.ends_with(" [--asset <file>]..."), "{ref2va}");
        assert!(ref2va.contains(" [upscale=<null|object>]"), "{ref2va}");
        let birefnet = interface("birefnet-0.2.0");
        assert!(
            usage_line(
                "fidika/birefnet/remove_background",
                callable(&birefnet, "remove_background")
            )
            .ends_with(" --asset image=<file>")
        );
    }

    #[test]
    fn the_contract_names_defaults_bounds_slots_and_output() {
        let sdxl = interface("sdxl-2.10.0");
        let text = contract("fidika/sdxl/generate", &sdxl, callable(&sdxl, "generate"));
        for line in [
            "    prompt: str\n",
            "    steps: int (>=1, <=50) (default = 20)\n",
            "    seed: int|null (default = null)\n",
            "    model: SdxlModel\n",
            "    image: image file (image/webp)\n",
            "usage: cozy run fidika/sdxl/generate prompt=<str>",
        ] {
            assert!(text.contains(line), "{line:?} not in\n{text}");
        }
        let mut card =
            super::super::interface::tests::card("sdxl-2.10.0")["package_interface"].clone();
        let slot = &mut card["entrypoints"][1]["models"][0];
        slot["default"] = "fidika/sdxl-base@1.0".into();
        slot["min_bits"] = 8.into();
        let after = super::super::interface::Interface::parse(&card).unwrap();
        assert!(
            contract("fidika/sdxl/generate", &after, callable(&after, "generate"))
                .contains("    model: SdxlModel (default fidika/sdxl-base@1.0; at least 8 bits)\n")
        );
        let h3 = interface("minimax-h3-1.39.0");
        let ref2va = contract("fidika/minimax-h3/ref2va", &h3, callable(&h3, "ref2va"));
        assert!(
            ref2va.contains("\n  runs on: CPU; calls the package's GPU functions\n"),
            "{ref2va}"
        );
        let fp8 = contract("fidika/sdxl/fp8", &sdxl, callable(&sdxl, "fp8"));
        assert!(fp8.contains("\n  runs on: CPU\n"), "{fp8}");
        for line in [
            "runs on: CPU",
            "source: SdxlModel weights\n",
            "writes weights: fp8\n",
            "output: a model\n",
        ] {
            assert!(fp8.contains(line), "{line:?} not in\n{fp8}");
        }
    }
}
