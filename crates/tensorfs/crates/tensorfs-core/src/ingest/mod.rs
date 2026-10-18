//! The v1 ingest path TensorD's model upload still drives; `crate::import` replaces it.
//!
//! The border's four stages, each its own file:
//!   carrier     — strict bounded foreign-carrier decode (refuse before allocation)
//!   fingerprint — classification into a reviewed registry verdict; ambiguity refuses
//!   convert     — the PLAN (pure function of headers + target) and the reviewed converters
//!   transaction — temporary candidate write through the real CAS
//!
//! And one stage that runs BEFORE all of them:
//!   preflight   — the plan, decided from headers over sparse holes, before a byte moves

pub mod carrier;
pub mod convert;
pub mod custody;
pub mod fingerprint;
mod h3_lora;
pub mod journal;
mod narrow;
pub mod preflight;
pub mod routes;
mod sdxl_lora;
pub mod source;
pub mod transaction;
