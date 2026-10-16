//! Windows: births from creation times, exact handles as process HANDLEs (an open handle keeps the
//! PID from being reused), trees as Job Objects.
use super::{Birth, Lifetime, State};
use std::{
    io, mem,
    os::windows::{
        io::{AsRawHandle, FromRawHandle, OwnedHandle},
        process::CommandExt,
    },
    process::{Child, Command},
    ptr,
    time::Duration,
};
use windows_sys::Win32::{
    Foundation::{
        ERROR_ACCESS_DENIED, ERROR_INVALID_PARAMETER, FILETIME, HANDLE, INVALID_HANDLE_VALUE,
        WAIT_OBJECT_0, WAIT_TIMEOUT,
    },
    System::{
        Diagnostics::ToolHelp::{
            CreateToolhelp32Snapshot, TH32CS_SNAPTHREAD, THREADENTRY32, Thread32First, Thread32Next,
        },
        JobObjects::{
            AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
            JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JobObjectExtendedLimitInformation,
            SetInformationJobObject, TerminateJobObject,
        },
        Threading::{
            CREATE_BREAKAWAY_FROM_JOB, CREATE_NEW_PROCESS_GROUP, CREATE_SUSPENDED,
            DETACHED_PROCESS, GetProcessTimes, INFINITE, OpenProcess, OpenThread,
            PROCESS_QUERY_LIMITED_INFORMATION, PROCESS_SYNCHRONIZE, PROCESS_TERMINATE,
            ResumeThread, THREAD_SUSPEND_RESUME, TerminateProcess, WaitForSingleObject,
        },
    },
};

#[derive(Debug)]
pub(super) struct Handle(OwnedHandle);

/// The tree's job.
#[derive(Debug)]
pub(super) struct Group(OwnedHandle);

pub(super) fn birth(pid: u32) -> io::Result<Birth> {
    let process = open_process(pid, PROCESS_QUERY_LIMITED_INFORMATION)?;
    Ok(Birth {
        pid,
        boot_id: String::new(),
        start_ticks: created(&process)?,
    })
}

pub(super) fn state(birth: &Birth) -> io::Result<State> {
    let process = match open_process(
        birth.pid,
        PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_SYNCHRONIZE,
    ) {
        Ok(process) => process,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(State::Gone),
        Err(error) => return Err(error),
    };
    if created(&process)? != birth.start_ticks || signaled(&process, Some(Duration::ZERO))? {
        return Ok(State::Gone);
    }
    Ok(State::Live)
}

/// An exited process is refused: Windows has no reaping, and any process (a console host) may keep
/// a dead one's object alive with a handle.
pub(super) fn open(birth: &Birth) -> io::Result<Option<Handle>> {
    let access = PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_SYNCHRONIZE | PROCESS_TERMINATE;
    match open_process(birth.pid, access) {
        Ok(process)
            if created(&process)? == birth.start_ticks
                && !signaled(&process, Some(Duration::ZERO))? =>
        {
            Ok(Some(Handle(process)))
        }
        Ok(_) => Ok(None),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    }
}

pub(super) fn kill(handle: &Handle, _: &Birth) -> io::Result<()> {
    // SAFETY: a live process handle opened with PROCESS_TERMINATE.
    if unsafe { TerminateProcess(raw(&handle.0), 1) } == 0 {
        let error = io::Error::last_os_error();
        // Terminating a process that already exited is refused; it is done.
        if !signaled(&handle.0, Some(Duration::ZERO))? {
            return Err(error);
        }
    }
    Ok(())
}

pub(super) fn wait(handle: &Handle, period: Option<Duration>) -> io::Result<bool> {
    signaled(&handle.0, period)
}

pub(super) fn spawn(command: &mut Command, lifetime: Lifetime) -> io::Result<(Child, Group)> {
    let job = job(lifetime)?;
    let mut flags = CREATE_SUSPENDED | CREATE_NEW_PROCESS_GROUP;
    if lifetime == Lifetime::Detached {
        flags |= DETACHED_PROCESS | CREATE_BREAKAWAY_FROM_JOB;
    }
    command.creation_flags(flags);
    let mut child = match command.spawn() {
        // The spawner's job forbids breakaway: the tree stays inside it, in its own job as well.
        Err(error)
            if flags & CREATE_BREAKAWAY_FROM_JOB != 0
                && error.raw_os_error() == Some(ERROR_ACCESS_DENIED as i32) =>
        {
            command.creation_flags(flags & !CREATE_BREAKAWAY_FROM_JOB);
            command.spawn()?
        }
        spawned => spawned?,
    };
    // Fails closed: a child the job cannot take is ended before it ran one instruction.
    // SAFETY: a live job and a live, suspended process.
    let adopted = match unsafe { AssignProcessToJobObject(raw(&job), child.as_raw_handle()) } {
        0 => Err(io::Error::last_os_error()),
        _ => resume(child.id()),
    };
    if let Err(error) = adopted {
        let _ = child.kill();
        let _ = child.wait();
        return Err(error);
    }
    Ok((child, Group(job)))
}

pub(super) fn kill_group(group: &Group, _: &Child) -> io::Result<()> {
    // SAFETY: a live job handle.
    match unsafe { TerminateJobObject(raw(&group.0), 1) } {
        0 => Err(io::Error::last_os_error()),
        _ => Ok(()),
    }
}

/// An owned tree's job ends every member when its last handle closes, the spawner's death included.
fn job(lifetime: Lifetime) -> io::Result<OwnedHandle> {
    // SAFETY: default security, unnamed.
    let job = owned(unsafe { CreateJobObjectW(ptr::null(), ptr::null()) })?;
    if lifetime == Lifetime::Owned {
        let mut limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        // SAFETY: a live job and a correctly sized limit structure.
        let set = unsafe {
            SetInformationJobObject(
                raw(&job),
                JobObjectExtendedLimitInformation,
                (&limits as *const JOBOBJECT_EXTENDED_LIMIT_INFORMATION).cast(),
                mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            )
        };
        if set == 0 {
            return Err(io::Error::last_os_error());
        }
    }
    Ok(job)
}

/// Resumes the one thread of a child created suspended (std keeps no thread handle).
fn resume(pid: u32) -> io::Result<()> {
    // SAFETY: a snapshot of every thread on the system.
    let snapshot = unsafe { CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0) };
    if snapshot == INVALID_HANDLE_VALUE {
        return Err(io::Error::last_os_error());
    }
    let snapshot = owned(snapshot)?;
    let mut entry = THREADENTRY32 {
        dwSize: mem::size_of::<THREADENTRY32>() as u32,
        ..THREADENTRY32::default()
    };
    let mut resumed = 0;
    // SAFETY: a live snapshot and an entry whose size field is set.
    let mut more = unsafe { Thread32First(raw(&snapshot), &mut entry) } != 0;
    while more {
        if entry.th32OwnerProcessID == pid {
            // SAFETY: a thread id the snapshot just listed.
            let thread =
                owned(unsafe { OpenThread(THREAD_SUSPEND_RESUME, 0, entry.th32ThreadID) })?;
            // SAFETY: a live thread handle opened for resuming.
            if unsafe { ResumeThread(raw(&thread)) } == u32::MAX {
                return Err(io::Error::last_os_error());
            }
            resumed += 1;
        }
        // SAFETY: as above.
        more = unsafe { Thread32Next(raw(&snapshot), &mut entry) } != 0;
    }
    match resumed {
        0 => Err(io::Error::other(
            "the suspended child has no thread to resume",
        )),
        _ => Ok(()),
    }
}

fn open_process(pid: u32, access: u32) -> io::Result<OwnedHandle> {
    // SAFETY: plain call; a null handle is the failure.
    let process = unsafe { OpenProcess(access, 0, pid) };
    if process.is_null() {
        let error = io::Error::last_os_error();
        // No process has this id (any more).
        if error.raw_os_error() == Some(ERROR_INVALID_PARAMETER as i32) {
            return Err(io::ErrorKind::NotFound.into());
        }
        return Err(error);
    }
    // SAFETY: a new handle that is now ours.
    Ok(unsafe { OwnedHandle::from_raw_handle(process) })
}

fn created(process: &OwnedHandle) -> io::Result<u64> {
    let mut times = [FILETIME::default(); 4];
    let [creation, exit, kernel, user] = &mut times;
    // SAFETY: a live handle with query rights and four writable FILETIMEs.
    if unsafe { GetProcessTimes(raw(process), creation, exit, kernel, user) } == 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(u64::from(creation.dwHighDateTime) << 32 | u64::from(creation.dwLowDateTime))
}

fn signaled(handle: &OwnedHandle, period: Option<Duration>) -> io::Result<bool> {
    let milliseconds = period.map_or(INFINITE, |period| {
        period.as_millis().min(u128::from(INFINITE - 1)) as u32
    });
    // SAFETY: a live handle with SYNCHRONIZE.
    match unsafe { WaitForSingleObject(raw(handle), milliseconds) } {
        WAIT_OBJECT_0 => Ok(true),
        WAIT_TIMEOUT => Ok(false),
        _ => Err(io::Error::last_os_error()),
    }
}

fn owned(handle: HANDLE) -> io::Result<OwnedHandle> {
    if handle.is_null() {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: a new handle that is now ours.
    Ok(unsafe { OwnedHandle::from_raw_handle(handle) })
}

fn raw(handle: &OwnedHandle) -> HANDLE {
    handle.as_raw_handle()
}
