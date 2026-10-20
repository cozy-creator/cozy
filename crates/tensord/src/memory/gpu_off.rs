//! Where the OS has no memfd ledger (macOS, Windows): no GPU ledger exists.
use super::{MemoryLog, Sample};
use crate::process::Liveness;
use crate::device_executor::Page;
use std::{fs::File, io};

pub type Ender = Box<dyn Fn(u32) + Send + Sync>;
pub type Freed = Box<dyn Fn() + Send + Sync>;

pub enum GpuLedger {}

pub struct Active<'a> {
    _ledger: &'a GpuLedger,
}

impl GpuLedger {
    pub fn start(_: &str, _: bool, _: u64, _: Option<u64>, _: Option<u64>, _: Liveness, _: MemoryLog) -> io::Result<Option<Self>> {
        Ok(None)
    }
    pub fn on_stalled(&self, _: Ender) {
        match *self {}
    }
    pub fn on_freed(&self, _: Freed) {
        match *self {}
    }
    pub fn grant(&self) -> io::Result<[File; 2]> {
        match *self {}
    }
    pub fn sample(&self) -> Option<Sample> {
        match *self {}
    }
    pub fn now(&self) {
        match *self {}
    }
    pub fn active(&self) -> Active<'_> {
        match *self {}
    }
    pub fn fits(&self, _: &str, _: Option<&str>, _: bool) -> bool {
        match *self {}
    }
    pub fn running(&self) -> bool {
        match *self {}
    }
    pub fn settle(&self) {
        match *self {}
    }
    pub fn resident(&self, _: u32) -> bool {
        match *self {}
    }
    pub fn dispatch(&self, _: &str, _: &str) -> io::Result<()> {
        match *self {}
    }
    pub fn undispatch(&self, _: &str) {
        match *self {}
    }
    pub fn keep(&self, _: &str, _: &str, _: u32, _: Vec<(Page, File)>) {
        match *self {}
    }
    pub fn take_kept(&self, _: &str) -> Option<(String, Vec<(Page, File)>)> {
        match *self {}
    }
    pub fn forget_kept(&self) {
        match *self {}
    }
    pub fn refuse_kept(&self) {
        match *self {}
    }
}
