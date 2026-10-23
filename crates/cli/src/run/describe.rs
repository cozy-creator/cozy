//! The one contract printer: the usage line a refusal's remedy names, `--describe`, and a
//! package's function list. No other module renders a request schema.
use std::fmt::Write as _;

use serde_json::Value;

use super::interface::{Callable, Constraints, Field, Interface, Type, Wire};

/// One concrete invocation: required terms bare, optional ones bracketed. A conversion reads its
/// input model and the repository it writes first.
pub fn usage_line(target: &str, callable: &Callable) -> String {
    let mut line = format!("cozy run {target}");
    if callable.is_conversion() {
        let source = &callable.models[0].param;
        let _ = write!(line, " <{source}> [<org/model>]");
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

/// `--describe`: where it runs, request, model slots with their defaults, output, and the
/// usage line.
pub fn contract(target: &str, interface: &Interface, callable: &Callable) -> String {
    let mut out = target.to_owned();
    if callable.accelerator == Some(false) {
        out.push_str(match interface.orchestrates(callable) {
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
