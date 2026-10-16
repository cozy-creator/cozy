//! What every verb receives: the global flags and, as they land, the loaded config.

#[derive(Debug)]
pub struct Context {
    pub json: bool,
    pub tensorhub: Option<String>,
}
