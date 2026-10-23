//! First-party packages and models have no owner (xs-071): `sdxl/generate`, `flux-3-klein@0.1.0`.
//! On the wire, in records and in Hub paths their org is [`ROOT`], which no account can take, so
//! an older peer reads `_/sdxl` as an ordinary `org/name`. People never type or see `_/`.

/// The root namespace's org.
pub const ROOT: &str = "_";

/// `text` as a person reads it: each name that starts a word drops its `_/`.
pub fn shown(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    let mut word_start = true;
    while let Some(c) = rest.chars().next() {
        if word_start && rest.starts_with("_/") {
            rest = &rest[2..];
            word_start = false;
            continue;
        }
        word_start = c.is_whitespace() || matches!(c, '(' | '"' | '\'' | '`' | '[' | '=' | ',');
        out.push(c);
        rest = &rest[c.len_utf8()..];
    }
    out
}
