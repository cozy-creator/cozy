//! Windows: births from creation times, exact handles as process HANDLEs (an open handle keeps the
//! PID from being reused), trees as Job Objects.
use super::{Birth, State};
use std::{
    ffi::OsString,
    fs::File,
    io, iter, mem,
    os::windows::{
        io::{AsRawHandle, FromRawHandle, OwnedHandle},
        process::{CommandExt, ExitStatusExt},
    },
    path::Path,
    process::{Child, Command, ExitStatus},
    ptr,
    time::Duration,
};
use windows_sys::Win32::{
    Foundation::{
        CloseHandle, ERROR_ACCESS_DENIED, ERROR_INVALID_PARAMETER, FILETIME, HANDLE,
        HANDLE_FLAG_INHERIT, INVALID_HANDLE_VALUE, STILL_ACTIVE, SetHandleInformation,
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
            CREATE_BREAKAWAY_FROM_JOB, CREATE_NEW_PROCESS_GROUP, CREATE_SUSPENDED, CreateProcessW,
            DETACHED_PROCESS, DeleteProcThreadAttributeList, EXTENDED_STARTUPINFO_PRESENT,
            GetExitCodeProcess, GetProcessIoCounters, GetProcessTimes, INFINITE, IO_COUNTERS, InitializeProcThreadAttributeList,
            OpenProcess, OpenThread, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, PROCESS_INFORMATION,
            PROCESS_QUERY_LIMITED_INFORMATION, PROCESS_SYNCHRONIZE, PROCESS_TERMINATE,
            ResumeThread, STARTF_USESTDHANDLES, STARTUPINFOEXW, THREAD_SUSPEND_RESUME,
            TerminateProcess, UpdateProcThreadAttribute, WaitForSingleObject,
        },
    },
};

#[derive(Debug)]
pub(super) struct Handle(pub(super) OwnedHandle);

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
        // A process that exited, or is exiting (its job was just ended), refuses it: it is done.
        let mut code = 0;
        // SAFETY: a live process handle and a writable exit code.
        let exiting = unsafe { GetExitCodeProcess(raw(&handle.0), &mut code) } != 0
            && code != STILL_ACTIVE as u32;
        if !exiting && !signaled(&handle.0, Some(Duration::ZERO))? {
            return Err(error);
        }
    }
    Ok(())
}

pub(super) fn set_name(_: &str) -> crate::Result<()> {
    Err(crate::Error::unsupported(crate::Seam::Process))
}

/// A leader's followers: the processes it started and theirs, born after it.
pub(super) fn group_members(leader: &Birth) -> Vec<Birth> {
    super::descendants(leader.pid)
        .into_iter()
        .filter_map(|pid| birth(pid).ok())
        .filter(|member| member.start_ticks >= leader.start_ticks)
        .collect()
}

/// CPU nanoseconds used and bytes read and written. Windows keeps no time waiting for a CPU.
pub(super) fn meter(pid: u32) -> Option<(u64, u64)> {
    let process = open_process(pid, PROCESS_QUERY_LIMITED_INFORMATION).ok()?;
    let [_, _, kernel, user] = times(&process).ok()?;
    let io = io_counters(&process)?;
    let cpu = (ticks(kernel) + ticks(user)).saturating_mul(100);
    Some((cpu, io.ReadTransferCount.saturating_add(io.WriteTransferCount)))
}

pub(super) fn read_bytes(pid: u32) -> Option<u64> {
    let process = open_process(pid, PROCESS_QUERY_LIMITED_INFORMATION).ok()?;
    Some(io_counters(&process)?.ReadTransferCount)
}

fn io_counters(process: &OwnedHandle) -> Option<IO_COUNTERS> {
    // SAFETY: plain-old-data the call fills.
    let mut counters: IO_COUNTERS = unsafe { mem::zeroed() };
    // SAFETY: a live handle with query rights and a writable record.
    (unsafe { GetProcessIoCounters(raw(process), &mut counters) } != 0).then_some(counters)
}

pub(super) fn become_subreaper() -> crate::Result<()> {
    Err(crate::Error::unsupported(crate::Seam::Process))
}

pub(super) fn die_with_parent(_: u32) -> crate::Result<bool> {
    Err(crate::Error::unsupported(crate::Seam::Process))
}

pub(super) fn transient(error: &io::Error) -> bool {
    matches!(
        error.kind(),
        io::ErrorKind::OutOfMemory | io::ErrorKind::Interrupted | io::ErrorKind::WouldBlock
    )
}

/// A process id that names no process any more.
pub(super) fn gone(error: &io::Error) -> bool {
    error.raw_os_error() == Some(ERROR_INVALID_PARAMETER as i32)
}

pub(super) fn wait(handle: &Handle, period: Option<Duration>) -> io::Result<bool> {
    signaled(&handle.0, period)
}

pub(super) fn spawn(command: &mut Command) -> io::Result<(Child, Group)> {
    let job = job()?;
    command.creation_flags(CREATE_SUSPENDED | CREATE_NEW_PROCESS_GROUP);
    let mut child = command.spawn()?;
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

/// The tree's job ends every member when its last handle closes, the spawner's death included.
pub(crate) fn job() -> io::Result<OwnedHandle> {
    // SAFETY: default security, unnamed.
    let job = owned(unsafe { CreateJobObjectW(ptr::null(), ptr::null()) })?;
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
    Ok(job)
}

#[derive(Debug)]
pub(super) struct Detached {
    process: OwnedHandle,
    pid: u32,
}

impl Detached {
    pub(super) fn pid(&self) -> u32 {
        self.pid
    }

    pub(super) fn try_wait(&mut self) -> io::Result<Option<ExitStatus>> {
        if !signaled(&self.process, Some(Duration::ZERO))? {
            return Ok(None);
        }
        let mut code = 0;
        // SAFETY: a live process handle and a writable exit code.
        if unsafe { GetExitCodeProcess(raw(&self.process), &mut code) } == 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(Some(ExitStatus::from_raw(code)))
    }
}

/// std's spawn hands a child every inheritable handle this process holds, a pipe its caller reads
/// to the end among them (`out=$(cozy up)`): a child that outlives this process would hold that
/// end open forever. So CreateProcessW here, with a handle list of exactly the child's own.
pub(super) fn detach(program: &Path, args: &[OsString], log: Option<File>) -> io::Result<Detached> {
    let null = File::options().read(true).write(true).open("NUL")?;
    let out = log.as_ref().unwrap_or(&null);
    let mut handles = vec![null.as_raw_handle(), out.as_raw_handle()];
    handles.dedup();
    for &handle in &handles {
        // SAFETY: handles this function owns, closed once the child has its own.
        if unsafe { SetHandleInformation(handle, HANDLE_FLAG_INHERIT, HANDLE_FLAG_INHERIT) } == 0 {
            return Err(io::Error::last_os_error());
        }
    }
    let attributes = Attributes::handles(&handles)?;
    let mut info = STARTUPINFOEXW::default();
    info.StartupInfo.cb = mem::size_of::<STARTUPINFOEXW>() as u32;
    info.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
    info.StartupInfo.hStdInput = null.as_raw_handle();
    info.StartupInfo.hStdOutput = out.as_raw_handle();
    info.StartupInfo.hStdError = out.as_raw_handle();
    info.lpAttributeList = attributes.0.as_ptr().cast_mut().cast();
    let mut line: Vec<u16> = command_line(program, args)
        .encode_utf16()
        .chain(iter::once(0))
        .collect();
    let mut flags = CREATE_NEW_PROCESS_GROUP
        | DETACHED_PROCESS
        | CREATE_BREAKAWAY_FROM_JOB
        | EXTENDED_STARTUPINFO_PRESENT;
    loop {
        let mut process = PROCESS_INFORMATION::default();
        // SAFETY: a writable, nul-terminated command line; the startup information and its
        // attribute list outlive the call.
        let created = unsafe {
            CreateProcessW(
                ptr::null(),
                line.as_mut_ptr(),
                ptr::null(),
                ptr::null(),
                1,
                flags,
                ptr::null(),
                ptr::null(),
                &info.StartupInfo,
                &mut process,
            )
        };
        if created != 0 {
            // SAFETY: the new thread's handle, not needed; the process's is now ours.
            unsafe { CloseHandle(process.hThread) };
            return Ok(Detached {
                process: unsafe { OwnedHandle::from_raw_handle(process.hProcess) },
                pid: process.dwProcessId,
            });
        }
        let error = io::Error::last_os_error();
        // The spawner's job forbids breakaway: the child stays inside it.
        if flags & CREATE_BREAKAWAY_FROM_JOB != 0
            && error.raw_os_error() == Some(ERROR_ACCESS_DENIED as i32)
        {
            flags &= !CREATE_BREAKAWAY_FROM_JOB;
            continue;
        }
        return Err(error);
    }
}

/// A PROC_THREAD_ATTRIBUTE_LIST holding one handle list, which must outlive it.
struct Attributes(Vec<usize>);

impl Attributes {
    fn handles(handles: &[HANDLE]) -> io::Result<Self> {
        let mut size = 0;
        // SAFETY: asks only for the size one attribute needs; fails with that size.
        unsafe { InitializeProcThreadAttributeList(ptr::null_mut(), 1, 0, &mut size) };
        let mut list = Attributes(vec![0; size.div_ceil(mem::size_of::<usize>())]);
        let start = list.0.as_mut_ptr().cast();
        // SAFETY: a pointer-aligned buffer of the size asked for.
        if unsafe { InitializeProcThreadAttributeList(start, 1, 0, &mut size) } == 0 {
            list.0 = Vec::new();
            return Err(io::Error::last_os_error());
        }
        // SAFETY: an initialized list and a handle array that outlives it.
        let updated = unsafe {
            UpdateProcThreadAttribute(
                start,
                0,
                PROC_THREAD_ATTRIBUTE_HANDLE_LIST as usize,
                handles.as_ptr().cast(),
                mem::size_of_val(handles),
                ptr::null_mut(),
                ptr::null(),
            )
        };
        if updated == 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(list)
    }
}

impl Drop for Attributes {
    fn drop(&mut self) {
        if !self.0.is_empty() {
            // SAFETY: a list InitializeProcThreadAttributeList initialized.
            unsafe { DeleteProcThreadAttributeList(self.0.as_mut_ptr().cast()) };
        }
    }
}

/// The program and its arguments as one command line, which `CommandLineToArgvW` splits back.
pub(crate) fn command_line(program: &Path, args: &[OsString]) -> String {
    iter::once(program.as_os_str())
        .chain(args.iter().map(|arg| arg.as_os_str()))
        .map(|word| quote(&word.to_string_lossy()))
        .collect::<Vec<_>>()
        .join(" ")
}

/// One argument as `CommandLineToArgvW` reads it back: quoted when it must be, a quote escaped,
/// and the backslashes before a quote doubled.
fn quote(word: &str) -> String {
    if !word.is_empty() && !word.contains([' ', '\t', '"']) {
        return word.into();
    }
    let mut quoted = String::from('"');
    let mut backslashes = 0;
    for c in word.chars() {
        match c {
            '\\' => backslashes += 1,
            '"' => {
                quoted.extend(iter::repeat_n('\\', backslashes * 2 + 1));
                quoted.push('"');
                backslashes = 0;
            }
            _ => {
                quoted.extend(iter::repeat_n('\\', backslashes));
                quoted.push(c);
                backslashes = 0;
            }
        }
    }
    quoted.extend(iter::repeat_n('\\', backslashes * 2));
    quoted.push('"');
    quoted
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

pub(super) fn usage(pid: u32) -> io::Result<super::Usage> {
    let process = open_process(pid, PROCESS_QUERY_LIMITED_INFORMATION)?;
    let [_, _, kernel, user] = times(&process)?;
    Ok(super::Usage {
        cpu: ticks(kernel) + ticks(user),
        running: false,
    })
}

/// The kernel's boot time in seconds (100 ns ticks since 1601), which a clock step can move:
/// then this boot reads as an earlier one, the safe side.
pub(super) fn boot_id() -> io::Result<String> {
    use windows_sys::Wdk::System::SystemInformation::{
        NtQuerySystemInformation, SystemTimeOfDayInformation,
    };
    // SYSTEM_TIMEOFDAY_INFORMATION: BootTime first, 48 bytes in all.
    let mut info = [0i64; 6];
    // SAFETY: a writable buffer of the size passed.
    let status = unsafe {
        NtQuerySystemInformation(
            SystemTimeOfDayInformation,
            info.as_mut_ptr().cast(),
            48,
            ptr::null_mut(),
        )
    };
    if status < 0 {
        return Err(io::Error::other(format!(
            "NtQuerySystemInformation: {status:#x}"
        )));
    }
    Ok(format!("{}", info[0] / 10_000_000))
}

pub(super) fn peak_resident() -> io::Result<u64> {
    use windows_sys::Win32::System::{
        ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS},
        Threading::GetCurrentProcess,
    };
    // SAFETY: plain-old-data the call fills, its size passed.
    let mut counters: PROCESS_MEMORY_COUNTERS = unsafe { mem::zeroed() };
    let size = mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
    if unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, size) } == 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(counters.PeakWorkingSetSize as u64)
}

/// Marks `handle` inheritable: std hands a child every inheritable handle.
pub(super) fn hand_on(handle: &impl AsRawHandle) -> io::Result<u64> {
    // SAFETY: a live handle the caller owns.
    if unsafe { SetHandleInformation(handle.as_raw_handle(), HANDLE_FLAG_INHERIT, HANDLE_FLAG_INHERIT) } == 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(handle.as_raw_handle() as usize as u64)
}

/// Takes a handle this process inherited, no longer inheritable: it goes no further.
///
/// # Safety
/// `value` must name a handle this process inherited and nothing else owns.
pub(super) unsafe fn handed(value: u64) -> io::Result<OwnedHandle> {
    // SAFETY: the caller's promise.
    let handle = unsafe { OwnedHandle::from_raw_handle(value as usize as _) };
    // SAFETY: a live handle this process owns.
    if unsafe { SetHandleInformation(handle.as_raw_handle(), HANDLE_FLAG_INHERIT, 0) } == 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(handle)
}

pub(super) fn parent_of(pid: u32) -> io::Result<u32> {
    parents()
        .into_iter()
        .find(|(child, _)| *child == pid)
        .map(|(_, parent)| parent)
        .ok_or_else(|| io::ErrorKind::NotFound.into())
}

/// Every process with its parent, from one Toolhelp snapshot.
pub(super) fn parents() -> Vec<(u32, u32)> {
    snapshot().unwrap_or_default()
}

fn snapshot() -> io::Result<Vec<(u32, u32)>> {
    use windows_sys::Win32::System::Diagnostics::ToolHelp::{
        PROCESSENTRY32W, Process32FirstW, Process32NextW, TH32CS_SNAPPROCESS,
    };
    // SAFETY: a snapshot of every process; the handle is owned at once.
    let snapshot = unsafe { CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0) };
    if snapshot == INVALID_HANDLE_VALUE {
        return Err(io::Error::last_os_error());
    }
    let snapshot = owned(snapshot)?;
    // SAFETY: plain-old-data with its size set, as the call requires.
    let mut entry: PROCESSENTRY32W = unsafe { mem::zeroed() };
    entry.dwSize = mem::size_of::<PROCESSENTRY32W>() as u32;
    // SAFETY: a live snapshot and an entry sized above.
    let mut more = unsafe { Process32FirstW(raw(&snapshot), &mut entry) } != 0;
    let mut table = vec![];
    while more {
        table.push((entry.th32ProcessID, entry.th32ParentProcessID));
        // SAFETY: as above.
        more = unsafe { Process32NextW(raw(&snapshot), &mut entry) } != 0;
    }
    Ok(table)
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
    Ok(ticks(times(process)?[0]))
}

/// Creation, exit, kernel and user times.
fn times(process: &OwnedHandle) -> io::Result<[FILETIME; 4]> {
    let mut times = [FILETIME::default(); 4];
    let [creation, exit, kernel, user] = &mut times;
    // SAFETY: a live handle with query rights and four writable FILETIMEs.
    if unsafe { GetProcessTimes(raw(process), creation, exit, kernel, user) } == 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(times)
}

fn ticks(time: FILETIME) -> u64 {
    u64::from(time.dwHighDateTime) << 32 | u64::from(time.dwLowDateTime)
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

#[cfg(test)]
mod tests {
    #[test]
    fn a_command_line_reads_back_as_its_arguments() {
        assert_eq!(
            super::quote(r"C:\Users\a\.cozy\bin\cozy.exe"),
            r"C:\Users\a\.cozy\bin\cozy.exe"
        );
        assert_eq!(
            super::quote(r"C:\Users\a b\cozy.exe"),
            r#""C:\Users\a b\cozy.exe""#
        );
        assert_eq!(super::quote(r#"say "hi""#), r#""say \"hi\"""#);
        assert_eq!(
            super::quote(r"C:\dir with space\"),
            r#""C:\dir with space\\""#
        );
        assert_eq!(super::quote(""), r#""""#);
    }
}
