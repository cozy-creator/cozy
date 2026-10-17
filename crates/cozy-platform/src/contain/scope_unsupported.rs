//! Containment scopes exist only on Linux so far (cgroup v2 or a process-tree token).
use crate::{Error, Result, Seam};

#[derive(Debug)]
pub struct Scope(());

impl Scope {
    pub fn create(_: &str) -> Result<Self> {
        Err(Error::unsupported(Seam::Containment))
    }

    pub fn sweep(_: &str) -> Result<usize> {
        Err(Error::unsupported(Seam::Containment))
    }

    pub fn cgroup_relative(&self) -> Option<&str> {
        None
    }

    pub fn environment(&self) -> Option<(&'static str, &str)> {
        None
    }

    pub fn trampoline_args(&self) -> Vec<String> {
        vec!["--scope-backend".into(), "inherit".into()]
    }

    pub fn processes(&self) -> Result<usize> {
        Err(Error::unsupported(Seam::Containment))
    }

    pub fn adopt(&self, _: u32) -> Result<()> {
        Err(Error::unsupported(Seam::Containment))
    }

    pub fn kill(&self) -> Result<()> {
        Err(Error::unsupported(Seam::Containment))
    }

    pub fn end(&self) -> Result<usize> {
        Err(Error::unsupported(Seam::Containment))
    }
}
