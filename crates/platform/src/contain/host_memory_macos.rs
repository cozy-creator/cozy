//! macOS: physical memory from `hw.memsize`; available as free plus inactive pages (what
//! psutil reports), from the host's VM statistics; a process's physical footprint.
use super::{HostMemory, ProcessMemory};
use std::{io, mem};

pub(super) fn total() -> i64 {
    let mut bytes = 0u64;
    let mut len = mem::size_of::<u64>();
    // SAFETY: a NUL-terminated name and a writable u64 of the length given.
    let read = unsafe {
        libc::sysctlbyname(
            c"hw.memsize".as_ptr(),
            (&mut bytes as *mut u64).cast(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if read == 0 { bytes as i64 } else { -1 }
}

pub(super) fn read() -> HostMemory {
    let available = available();
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

fn available() -> i64 {
    // SAFETY: plain-old-data the call fills.
    let mut stats: libc::vm_statistics64 = unsafe { mem::zeroed() };
    let mut count = libc::HOST_VM_INFO64_COUNT;
    // SAFETY: this task's host port and a statistics buffer of `count` words.
    #[allow(deprecated)]
    let read = unsafe {
        libc::host_statistics64(
            libc::mach_host_self(),
            libc::HOST_VM_INFO64,
            (&mut stats as *mut libc::vm_statistics64).cast(),
            &mut count,
        )
    };
    if read != 0 {
        return -1;
    }
    // SAFETY: sysconf reads a constant.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as i64;
    let free = i64::from(stats.free_count) - i64::from(stats.speculative_count);
    (free + i64::from(stats.inactive_count)) * page
}

pub(super) fn process(pid: u32) -> io::Result<ProcessMemory> {
    // SAFETY: plain-old-data the call fills.
    let mut usage: libc::rusage_info_v2 = unsafe { mem::zeroed() };
    // SAFETY: a writable rusage_info_v2, the flavor asked for.
    let read = unsafe {
        libc::proc_pid_rusage(
            pid as libc::c_int,
            libc::RUSAGE_INFO_V2,
            (&mut usage as *mut libc::rusage_info_v2).cast(),
        )
    };
    if read != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(ProcessMemory {
        rss: usage.ri_resident_size,
        pss: usage.ri_phys_footprint,
        pss_shmem: 0,
    })
}
