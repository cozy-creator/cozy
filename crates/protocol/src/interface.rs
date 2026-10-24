//! A package's interface (`cozy.package.interface/1`), as the Runtime describes it: the one place
//! every Rust reader spells its paths. Each accessor is tolerant: a member absent, null or of
//! another shape reads as absent, so no reader refuses a member it does not use.
use serde_json::Value;

/// The member that lists the entrypoints, for a reader that rewrites the document.
pub const ENTRYPOINTS: &str = "entrypoints";

/// Its format, `cozy.package.interface/<n>`; empty when it names none.
pub fn format(document: &Value) -> &str {
    document["format"].as_str().unwrap_or_default()
}

/// Its App, `module:attribute`.
pub fn application(document: &Value) -> Option<&str> {
    document["application"].as_str()
}

/// Every entrypoint it declares, in its order.
pub fn entrypoints(document: &Value) -> impl Iterator<Item = &Value> {
    document[ENTRYPOINTS].as_array().into_iter().flatten()
}

/// The entrypoint named `name`.
pub fn entrypoint<'a>(document: &'a Value, name: &str) -> Option<&'a Value> {
    entrypoints(document).find(|entry| self::name(entry) == Some(name))
}

/// Its declared quantize recipes.
pub fn recipes(document: &Value) -> impl Iterator<Item = &Value> {
    document["recipes"].as_array().into_iter().flatten()
}

/// An entrypoint's name.
pub fn name(entry: &Value) -> Option<&str> {
    entry["name"].as_str()
}

/// Whether it is internal: callable by the package's own code, never offered to people.
pub fn internal(entry: &Value) -> bool {
    entry["internal"].as_bool() == Some(true)
}

/// Whether it publishes media while it runs (`Outputs.publish`).
pub fn publishes(entry: &Value) -> bool {
    entry["publishes"].as_bool() == Some(true)
}

/// Its model slots (`<entrypoint>.models.<param>`).
pub fn models(entry: &Value) -> &[Value] {
    entry["models"].as_array().map(Vec::as_slice).unwrap_or_default()
}

/// A model slot's path, `<entrypoint>.models.<param>`.
pub fn slot_path(slot: &Value) -> Option<&str> {
    slot["path"].as_str()
}

/// Its invocable declaration (module, export, defaults, …); null when it declares none.
pub fn invocable(entry: &Value) -> &Value {
    &entry["invocable"]
}

/// The module and export its invocable is called by.
pub fn callable(entry: &Value) -> Option<(&str, &str)> {
    let invocable = invocable(entry);
    Some((invocable["module"].as_str()?, invocable["export"].as_str()?))
}

/// Whether its invocable is memoized (`memoize=True`): its result is reusable for the same request.
pub fn memoized(entry: &Value) -> bool {
    invocable(entry)["memoize"].as_bool() == Some(true)
}

/// Its result schema.
pub fn result(entry: &Value) -> Option<&Value> {
    entry.get("result")
}

/// Its weights outputs: each output's id and the most new bytes it may write, when declared.
pub fn weights_outputs(entry: &Value) -> impl Iterator<Item = (&str, Option<u64>)> {
    entry["weights_outputs"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|row| Some((row["output_id"].as_str()?, row["max_bytes"].as_u64())))
}
