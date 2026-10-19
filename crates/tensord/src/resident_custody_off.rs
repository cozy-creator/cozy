//! Degree 2 GPU custody where the OS cannot share an executor's device memory by handle
//! (macOS, Windows): the same types, and a custody that holds nothing. The pool keeps none there
//! (`gpu_service`), so nothing reaches these; each still answers as an empty custody would.
use platform::{handles::Owned, process::Birth};
use serde::{Deserialize, Serialize};
use std::{fs::File, io};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct HoldingKey {
    pub device: String,
    pub layout: String,
    pub variant: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SharedRegion {
    pub region: u32,
    pub chunks: Vec<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    Ready,
    Revoking,
}

#[derive(Clone, Debug, Serialize)]
pub struct HoldingFacts {
    pub key: HoldingKey,
    pub name: String,
    pub generation: u64,
    pub bytes: u64,
    pub phase: Phase,
    pub readers: Vec<Birth>,
    pub idle_ms: u64,
}

pub struct Attachment {
    pub generation: u64,
    pub regions: Vec<SharedRegion>,
    pub fds: Vec<Owned>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Offered {
    Kept { generation: u64 },
    Duplicate,
}

pub struct Reader {
    pub birth: Birth,
    pub exit: File,
    pub environment: String,
}

fn off() -> io::Error {
    io::Error::new(io::ErrorKind::Unsupported, "GPU custody is a Unix machine's")
}

impl Reader {
    pub fn lease(_birth: Birth, _exit: File, _environment: &str) -> io::Result<(Reader, Owned)> {
        Err(off())
    }
}

#[derive(Default)]
pub struct ResidentCustody(());

impl ResidentCustody {
    pub fn offer(
        &mut self,
        _key: HoldingKey,
        _name: &str,
        _regions: Vec<SharedRegion>,
        _fds: Vec<Owned>,
        _reader: Reader,
    ) -> io::Result<Offered> {
        Err(off())
    }

    pub fn attach(&mut self, _key: &HoldingKey, _reader: Reader) -> io::Result<Option<Attachment>> {
        Ok(None)
    }

    pub fn begin_revoke(&mut self, _key: &HoldingKey, _generation: u64) -> io::Result<Vec<Birth>> {
        Ok(Vec::new())
    }

    pub fn trim_regions(&self, _key: &HoldingKey, _generation: u64, _bytes: u64) -> Option<Vec<u32>> {
        None
    }

    pub fn trim(&mut self, _key: &HoldingKey, _generation: u64, _regions: &[u32]) -> u64 {
        0
    }

    pub fn released(&mut self, _key: &HoldingKey, _generation: u64, _birth: &Birth) {}

    pub fn collect(&mut self) -> Vec<(HoldingKey, u64)> {
        Vec::new()
    }

    pub fn end_environment(&mut self, _environment: &str) -> Vec<(HoldingKey, u64)> {
        Vec::new()
    }

    pub fn read_by(&self, _birth: &Birth) -> Vec<(HoldingKey, u64)> {
        Vec::new()
    }

    pub fn latest_variant(&self, _device: &str, _layout: &str) -> Option<String> {
        None
    }

    pub fn charged_bytes(&self, _device: &str) -> u64 {
        0
    }

    pub fn holdings(&self) -> Vec<HoldingFacts> {
        Vec::new()
    }
}
