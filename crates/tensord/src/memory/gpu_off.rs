//! Where varena has no ledger (Windows): no GPU ledger exists, so executors run without one.
use super::{MemoryLog, Sample};
use crate::process::Liveness;
use std::io;

pub type Ender = Box<dyn Fn(u32) + Send + Sync>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Admitted {
    Now,
    Short(u64),
}

pub enum GpuLedger {}

impl GpuLedger {
    pub fn start(_: &str, _: bool, _: u64, _: Option<u64>, _: Liveness, _: MemoryLog) -> io::Result<Option<Self>> {
        Ok(None)
    }
    pub fn on_stalled(&self, _: Ender) {
        match *self {}
    }
    pub fn grant(&self) -> io::Result<std::fs::File> {
        match *self {}
    }
    pub fn capacity(&self) -> u64 {
        match *self {}
    }
    pub fn context(&self) -> u64 {
        match *self {}
    }
    pub fn sample(&self) -> Option<Sample> {
        match *self {}
    }
    pub fn now(&self) {
        match *self {}
    }
    pub fn free(&self) -> io::Result<u64> {
        match *self {}
    }
    pub fn row(&self, _: u32) -> Option<(u64, u64)> {
        match *self {}
    }
    pub fn fits(&self, _: u64, _: bool) -> bool {
        match *self {}
    }
    pub fn admit(&self, _: u64, _: &mut dyn FnMut() -> bool) -> io::Result<Admitted> {
        match *self {}
    }
}
