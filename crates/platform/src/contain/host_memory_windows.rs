//! Windows: the global memory status, and a process's private commit.
#![allow(unsafe_code)]
use super::{HostMemory, ProcessMemory};
use std::{io, mem};
use windows_sys::Win32::{
    Foundation::CloseHandle,
    System::{
        ProcessStatus::{
            K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS, PROCESS_MEMORY_COUNTERS_EX,
        },
        SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX},
        Threading::{OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION},
    },
};

fn status() -> Option<MEMORYSTATUSEX> {
    // SAFETY: plain-old-data with its length set, as the call requires.
    let mut status: MEMORYSTATUSEX = unsafe { mem::zeroed() };
    status.dwLength = mem::size_of::<MEMORYSTATUSEX>() as u32;
    // SAFETY: a writable status of the length given.
    (unsafe { GlobalMemoryStatusEx(&mut status) } != 0).then_some(status)
}

pub(super) fn total() -> i64 {
    status().map_or(-1, |status| status.ullTotalPhys as i64)
}

pub(super) fn read() -> HostMemory {
    let available = status().map_or(-1, |status| status.ullAvailPhys as i64);
    HostMemory {
        available,
        shmem: 0,
        mem_available: available,
        bound: total(),
    }
}

pub(super) fn resident(_: u32) -> io::Result<super::Resident> {
    Err(io::ErrorKind::Unsupported.into())
}

pub(super) fn reset_peak(_: u32) -> io::Result<()> {
    Err(io::ErrorKind::Unsupported.into())
}

pub(super) fn process(pid: u32) -> io::Result<ProcessMemory> {
    // SAFETY: a query-only handle, closed below.
    let process = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid) };
    if process.is_null() {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: plain-old-data the call fills, its size passed.
    let mut counters: PROCESS_MEMORY_COUNTERS_EX = unsafe { mem::zeroed() };
    let size = mem::size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32;
    // SAFETY: a live handle and a counters buffer of `size` bytes.
    let read = unsafe {
        K32GetProcessMemoryInfo(
            process,
            (&mut counters as *mut PROCESS_MEMORY_COUNTERS_EX).cast::<PROCESS_MEMORY_COUNTERS>(),
            size,
        )
    };
    let error = io::Error::last_os_error();
    // SAFETY: the handle opened above.
    unsafe { CloseHandle(process) };
    if read == 0 {
        return Err(error);
    }
    Ok(ProcessMemory {
        rss: counters.WorkingSetSize as u64,
        pss: counters.PrivateUsage as u64,
        pss_shmem: 0,
    })
}
