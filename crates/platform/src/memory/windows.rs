//! Windows: the table a file in the user's temp directory, mapped as sections; range locks with
//! `LockFileEx` far past the data (a Windows lock bars every other handle's I/O on what it
//! covers); an auto-reset event as the wake; named pipes whose first instance is the service
//! lock; host pages as pagefile-backed sections mapped into placeholders (`VirtualAlloc2`,
//! `MapViewOfFile3`), exported as read-only duplicates.
use super::{clean, Map};
use crate::{Error, Result, Seam};
use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};
use std::path::PathBuf;
use windows_sys::Wdk::System::Threading::{NtQueryInformationThread, ThreadSuspendCount};
use windows_sys::Win32::Foundation::{CloseHandle, DuplicateHandle, ERROR_ACCESS_DENIED, ERROR_ALREADY_EXISTS, ERROR_FILE_NOT_FOUND, ERROR_LOCK_VIOLATION, ERROR_PIPE_BUSY, GENERIC_READ, GENERIC_WRITE, HANDLE, INVALID_HANDLE_VALUE, STILL_ACTIVE, WAIT_OBJECT_0};
use windows_sys::Win32::Storage::FileSystem::{FileStandardInfo, GetFileInformationByHandleEx, LockFileEx, ReOpenFile, FILE_STANDARD_INFO, UnlockFileEx, FILE_SHARE_DELETE, FILE_SHARE_READ, FILE_SHARE_WRITE, LOCKFILE_EXCLUSIVE_LOCK, LOCKFILE_FAIL_IMMEDIATELY};
use windows_sys::Win32::System::Memory::{
    CreateFileMappingW, MapViewOfFile, MapViewOfFile3, UnmapViewOfFile, UnmapViewOfFile2, VirtualAlloc2, VirtualFree, VirtualProtect, FILE_MAP_READ, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MEM_PRESERVE_PLACEHOLDER, MEM_RELEASE,
    MEM_REPLACE_PLACEHOLDER, MEM_RESERVE, MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, PAGE_READONLY, PAGE_READWRITE,
};
use windows_sys::Win32::System::IO::OVERLAPPED;
use windows_sys::Win32::System::SystemInformation::{GetSystemInfo, SYSTEM_INFO};
use windows_sys::Win32::System::Threading::{CreateEventW, OpenEventW, ResetEvent, EVENT_MODIFY_STATE, GetCurrentProcess, GetCurrentThreadId, GetExitCodeProcess, OpenProcess, OpenThread, SetEvent, WaitForSingleObject, INFINITE, PROCESS_QUERY_LIMITED_INFORMATION, THREAD_QUERY_INFORMATION};

pub type Owned = OwnedHandle;
pub type Stream = crate::ipc::LocalStream;
pub type Listener = crate::ipc::LocalListener;

fn last(op: &'static str) -> Error {
    Error::os(op)(std::io::Error::last_os_error())
}

fn raw(h: &Owned) -> HANDLE {
    h.as_raw_handle() as HANDLE
}

impl Drop for Map {
    fn drop(&mut self) {
        // SAFETY: our own view; nothing borrows it past its owner.
        unsafe { UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: self.base.cast() }) };
    }
}

/// This user's directory for tables (the temp directory is under the user's profile).
fn dir() -> Result<PathBuf> {
    let dir = std::env::temp_dir().join("cozy-varena");
    std::fs::create_dir_all(&dir).map_err(Error::io("create", &dir))?;
    Ok(dir)
}

/// The table named `name` for this user, created when absent.
pub fn table(name: &str) -> Result<Owned> {
    let path = dir()?.join(format!("{}.table", clean(name)));
    let file = std::fs::OpenOptions::new().read(true).write(true).create(true).truncate(false).open(&path).map_err(Error::io("open the table", &path))?;
    Ok(file.into())
}

/// The table lives in the user's own profile, whose ACL admits only the user: its owner is
/// this user (uids are a Unix notion; here every one is 0).
pub fn own(_: &Owned) -> Result<u32> {
    Ok(euid())
}

pub fn euid() -> u32 {
    0
}

pub fn uid_of(_: u32) -> Result<u32> {
    Ok(euid())
}

/// Whether the file still has a name (not deleted, or pending deletion).
pub fn linked(f: &Owned) -> Result<bool> {
    let mut info = FILE_STANDARD_INFO { AllocationSize: 0, EndOfFile: 0, NumberOfLinks: 0, DeletePending: false, Directory: false };
    // SAFETY: a buffer of the size we pass, for a handle we hold.
    if unsafe { GetFileInformationByHandleEx(raw(f), FileStandardInfo, (&mut info as *mut FILE_STANDARD_INFO).cast(), std::mem::size_of::<FILE_STANDARD_INFO>() as u32) } == 0 {
        return Err(last("read a file's links"));
    }
    Ok(!info.DeletePending && info.NumberOfLinks > 0)
}

pub fn unlink(name: &str) {
    if let Ok(dir) = dir() {
        let _ = std::fs::remove_file(dir.join(format!("{}.table", clean(name))));
    }
}

pub fn size(f: &Owned) -> Result<u64> {
    Ok(std::fs::File::from(f.try_clone()?).metadata()?.len())
}

/// Grows the file through a section of the new size: a mapped file's end cannot be set.
pub fn resize(f: &Owned, len: u64) -> Result<()> {
    // SAFETY: a section over a file we hold; closed at once.
    let section = unsafe { CreateFileMappingW(raw(f), std::ptr::null(), PAGE_READWRITE, (len >> 32) as u32, len as u32, std::ptr::null()) };
    if section.is_null() {
        return Err(last("grow the table"));
    }
    // SAFETY: the section we just made.
    unsafe { CloseHandle(section) };
    Ok(())
}

/// `len` bytes of `f` from `offset` (a multiple of [`granularity`]).
pub fn map(f: &Owned, offset: u64, len: usize, write: bool) -> Result<Map> {
    let protect = if write { PAGE_READWRITE } else { PAGE_READONLY };
    // SAFETY: a section over a file we hold, as large as the file.
    let section = unsafe { CreateFileMappingW(raw(f), std::ptr::null(), protect, 0, 0, std::ptr::null()) };
    if section.is_null() {
        return Err(last("map the table"));
    }
    let access = if write { FILE_MAP_READ | FILE_MAP_WRITE } else { FILE_MAP_READ };
    // SAFETY: a view of the section; the view keeps the section alive after its handle closes.
    let view = unsafe { MapViewOfFile(section, access, (offset >> 32) as u32, offset as u32, len) };
    // SAFETY: the section we just made.
    unsafe { CloseHandle(section) };
    if view.Value.is_null() {
        return Err(last("map the table"));
    }
    Ok(Map { base: view.Value.cast(), len })
}

/// View offsets and placeholders are multiples of the allocation granularity.
pub fn granularity() -> u64 {
    // SAFETY: plain-old-data the call fills.
    let mut info: SYSTEM_INFO = unsafe { std::mem::zeroed() };
    // SAFETY: as above.
    unsafe { GetSystemInfo(&mut info) };
    u64::from(info.dwAllocationGranularity)
}

/// A new handle to the same file: its locks are its own.
pub fn reopen(f: &Owned) -> Result<Owned> {
    // SAFETY: reopening a handle we hold.
    let h = unsafe { ReOpenFile(raw(f), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, 0) };
    if h == INVALID_HANDLE_VALUE {
        return Err(last("reopen the table"));
    }
    // SAFETY: a handle we just got.
    Ok(unsafe { Owned::from_raw_handle(h as _) })
}

/// VirtualFree's flag that merges adjacent placeholders (memoryapi.h); windows-sys lacks it.
const MEM_COALESCE_PLACEHOLDERS: u32 = 0x1;

/// Locks sit this far into the file, past any data it will ever hold.
const FAR: u64 = 1 << 62;

fn at(start: u64) -> OVERLAPPED {
    // SAFETY: plain-old-data.
    let mut o: OVERLAPPED = unsafe { std::mem::zeroed() };
    let offset = FAR + start;
    o.Anonymous.Anonymous.Offset = offset as u32;
    o.Anonymous.Anonymous.OffsetHigh = (offset >> 32) as u32;
    o
}

fn lock_ex(f: &Owned, start: u64, len: u64, flags: u32) -> Result<bool> {
    let mut o = at(start);
    // SAFETY: a lock on a synchronous handle we hold; the OVERLAPPED outlives the call.
    if unsafe { LockFileEx(raw(f), flags, 0, len as u32, (len >> 32) as u32, &mut o) } != 0 {
        return Ok(true);
    }
    let e = std::io::Error::last_os_error();
    match e.raw_os_error() {
        Some(code) if code == ERROR_LOCK_VIOLATION as i32 && flags & LOCKFILE_FAIL_IMMEDIATELY != 0 => Ok(false),
        _ => Err(Error::os("lock")(e)),
    }
}

/// Takes (or, `wait`, waits for) a write lock on a range, held by this handle.
pub fn lock(f: &Owned, start: u64, len: u64, wait: bool) -> Result<bool> {
    lock_ex(f, start, len, LOCKFILE_EXCLUSIVE_LOCK | if wait { 0 } else { LOCKFILE_FAIL_IMMEDIATELY })
}

/// Takes (or waits for) a read lock: many readers, no writer.
pub fn share(f: &Owned, start: u64, len: u64, wait: bool) -> Result<bool> {
    lock_ex(f, start, len, if wait { 0 } else { LOCKFILE_FAIL_IMMEDIATELY })
}

pub fn unlock(f: &Owned, start: u64, len: u64) -> Result<()> {
    let mut o = at(start);
    // SAFETY: as in `lock_ex`; a range this handle did not lock answers an error we ignore.
    unsafe { UnlockFileEx(raw(f), 0, len as u32, (len >> 32) as u32, &mut o) };
    Ok(())
}

/// Whether another handle holds a lock on the range: a lock this one can take is free.
pub fn held(f: &Owned, start: u64, len: u64) -> Result<bool> {
    if lock_ex(f, start, len, LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY)? {
        unlock(f, start, len)?;
        return Ok(false);
    }
    Ok(true)
}

/// A wake event: (the waiter's handle, the signaller's), one auto-reset event.
pub fn event() -> Result<(Owned, Owned)> {
    // SAFETY: an unnamed auto-reset event, not signalled.
    let h = unsafe { CreateEventW(std::ptr::null(), 0, 0, std::ptr::null()) };
    if h.is_null() {
        return Err(last("create an event"));
    }
    // SAFETY: a handle we just made.
    let reader = unsafe { Owned::from_raw_handle(h as _) };
    let writer = reader.try_clone()?;
    Ok((reader, writer))
}

pub fn signal(writer: &Owned) {
    // SAFETY: an event we hold.
    unsafe { SetEvent(raw(writer)) };
}

/// Blocks until the event is signalled; the wait resets it.
pub fn await_event(reader: &Owned) -> Result<()> {
    // SAFETY: an event we hold.
    match unsafe { WaitForSingleObject(raw(reader), INFINITE) } {
        WAIT_OBJECT_0 => Ok(()),
        _ => Err(last("wait for a wake")),
    }
}

/// A service's pipe, named under this user's temp directory (pipe names are machine-wide).
fn pipe(name: &str) -> Result<PathBuf> {
    Ok(dir()?.join(format!("{}.pipe", clean(name))))
}

/// The wake endpoint's event name, in this session's namespace.
fn wake_name(name: &str) -> Vec<u16> {
    format!("Local\\cozy-varena-wake-{}", clean(name)).encode_utf16().chain(std::iter::once(0)).collect()
}

/// The GPU's wake endpoint: a named auto-reset event every participant signals.
pub struct Endpoint(Owned);

impl Endpoint {
    /// Creates the named event; `None` while another process holds it (its handles keep it).
    pub fn bind(_ns: u32, name: &str) -> Result<Option<Endpoint>> {
        let name = wake_name(name);
        // SAFETY: a named auto-reset event, not signalled, with the process's default security.
        let h = unsafe { CreateEventW(std::ptr::null(), 0, 0, name.as_ptr()) };
        if h.is_null() {
            return Err(last("create the wake"));
        }
        // SAFETY: a handle we just got.
        let event = unsafe { Owned::from_raw_handle(h as _) };
        // SAFETY: reads the calling thread's last error, set by CreateEventW.
        if unsafe { windows_sys::Win32::Foundation::GetLastError() } == ERROR_ALREADY_EXISTS {
            return Ok(None);
        }
        Ok(Some(Endpoint(event)))
    }

    pub fn handle(&self) -> &Owned {
        &self.0
    }

    /// Blocks until the event is signalled; the wait clears it.
    pub fn wait(&self, _accept: impl Fn(u32) -> bool) -> Result<()> {
        await_event(&self.0)
    }

    /// Clears the signal; true when it was set. The event's default security admits only its
    /// user, so every signal is accepted.
    pub fn clear(&self, _accept: impl Fn(u32) -> bool) -> bool {
        // SAFETY: an event we hold; a zero wait only reads (and, auto-reset, clears) it.
        let set = unsafe { WaitForSingleObject(raw(&self.0), 0) } == WAIT_OBJECT_0;
        // SAFETY: as above.
        unsafe { ResetEvent(raw(&self.0)) };
        set
    }
}

/// A participant's signalling side: it opens the named event when it signals.
pub struct Notifier;

impl Notifier {
    pub fn new() -> Result<Notifier> {
        Ok(Notifier)
    }

    /// Signals endpoint `name`; none open (no TensorD) is no signal needed.
    pub fn notify(&self, _ns: u32, name: &str) {
        let name = wake_name(name);
        // SAFETY: opens a named event to signal it, then closes the handle.
        unsafe {
            let h = OpenEventW(EVENT_MODIFY_STATE, 0, name.as_ptr());
            if !h.is_null() {
                SetEvent(h);
                CloseHandle(h);
            }
        }
    }
}

/// Serves every connection to `listener` until `stop` is seen, a thread per connection (named
/// pipes have no poll): each one's message answered with `answer`'s bytes. A silent or stopped
/// peer holds up only its own thread.
pub fn serve_all(listener: Listener, stop: &std::sync::atomic::AtomicBool, _accept: impl Fn(u32) -> bool, answer: impl Fn(&[u8]) -> Vec<u8> + Send + Sync + 'static) {
    use std::io::{Read, Write};
    let answer = std::sync::Arc::new(answer);
    while let Ok(stream) = accept(&listener) {
        if stop.load(std::sync::atomic::Ordering::Acquire) {
            return;
        }
        let answer = answer.clone();
        let _ = std::thread::Builder::new().name("varena-ask-conn".into()).spawn(move || {
            let mut buf = vec![];
            let mut chunk = [0u8; 512];
            loop {
                match super::frame(&buf) {
                    super::Frame::Whole(body) => {
                        let _ = (&stream).write_all(&answer(body));
                        return;
                    }
                    super::Frame::Bad => return,
                    super::Frame::Partial => {}
                }
                match (&stream).read(&mut chunk) {
                    Ok(0) | Err(_) => return,
                    Ok(n) => buf.extend_from_slice(chunk.get(..n).unwrap_or_default()),
                }
            }
        });
    }
}

/// Opens the service's first pipe instance: `None` while another process holds it. The
/// instance is the service lock; it goes with its holder.
pub fn listen(_ns: u32, name: &str) -> Result<Option<Listener>> {
    match crate::ipc::LocalListener::bind(&pipe(name)?) {
        Ok(l) => Ok(Some(l)),
        Err(e) => {
            let e = std::io::Error::from(e);
            match e.raw_os_error() {
                Some(code) if code == ERROR_ACCESS_DENIED as i32 || code == ERROR_PIPE_BUSY as i32 => Ok(None),
                _ => Err(Error::os("open a service")(e)),
            }
        }
    }
}

pub fn accept(listener: &Listener) -> Result<Stream> {
    listener.accept()
}

/// Connects to the service; `None` when nobody serves it.
pub fn connect(_ns: u32, name: &str) -> Result<Option<Stream>> {
    match crate::ipc::connect(&pipe(name)?) {
        Ok(s) => Ok(Some(s)),
        Err(e) => {
            let e = std::io::Error::from(e);
            match e.raw_os_error() {
                Some(code) if code == ERROR_FILE_NOT_FOUND as i32 => Ok(None),
                _ => Err(Error::os("connect a service")(e)),
            }
        }
    }
}

/// The peer's pid and uid; the pipe's ACL admits only this user.
pub fn peer(stream: &Stream) -> Result<(u32, u32)> {
    Ok((crate::handles::peer_credentials(stream)?.pid, euid()))
}

/// A shareable host page of `len` bytes: a section backed by the paging file.
pub fn page(len: u64) -> Result<Owned> {
    // SAFETY: a new pagefile-backed section.
    let h = unsafe { CreateFileMappingW(INVALID_HANDLE_VALUE, std::ptr::null(), PAGE_READWRITE, (len >> 32) as u32, len as u32, std::ptr::null()) };
    if h.is_null() {
        return Err(last("create a page"));
    }
    // SAFETY: a handle we just made.
    Ok(unsafe { Owned::from_raw_handle(h as _) })
}

/// What another process imports: a duplicate that maps only for reading.
pub fn export(page: &Owned) -> Result<Owned> {
    let mut out: HANDLE = std::ptr::null_mut();
    // SAFETY: duplicating a handle we hold within this process.
    if unsafe { DuplicateHandle(GetCurrentProcess(), raw(page), GetCurrentProcess(), &mut out, FILE_MAP_READ, 0, 0) } == 0 {
        return Err(last("export a page"));
    }
    // SAFETY: a handle we just got.
    Ok(unsafe { Owned::from_raw_handle(out as _) })
}

/// Address space for host pages: one placeholder, split as pages map into it.
pub fn reserve(len: u64) -> Result<u64> {
    // SAFETY: a new placeholder reservation; nothing aliases it.
    let base = unsafe { VirtualAlloc2(std::ptr::null_mut(), std::ptr::null(), len as usize, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, std::ptr::null_mut(), 0) };
    if base.is_null() {
        return Err(last("reserve"));
    }
    Ok(base as u64)
}

/// Gives back a reservation whose pages are all unmapped: its placeholders coalesce, then go.
pub fn release(base: u64, len: u64) {
    // SAFETY: a reservation [`reserve`] made, every view of it unmapped.
    unsafe {
        VirtualFree(base as *mut _, len as usize, MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS);
        VirtualFree(base as *mut _, 0, MEM_RELEASE);
    }
}

/// Maps a host page at `at`: its range splits off the placeholder, then the view replaces it.
pub fn map_page(at: u64, page: &Owned, len: u64, write: bool) -> Result<()> {
    // SAFETY: splitting our own placeholder; a range that already is one placeholder refuses,
    // which leaves it as it is.
    unsafe { VirtualFree(at as *mut _, len as usize, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) };
    let protect = if write { PAGE_READWRITE } else { PAGE_READONLY };
    // SAFETY: a view of a section we hold, replacing our own placeholder.
    let view = unsafe { MapViewOfFile3(raw(page), GetCurrentProcess(), at as *const _, 0, len as usize, MEM_REPLACE_PLACEHOLDER, protect, std::ptr::null_mut(), 0) };
    if view.Value.is_null() {
        return Err(last("map a page"));
    }
    Ok(())
}

pub fn protect(at: u64, len: u64, write: bool) -> Result<()> {
    let mut old = 0;
    // SAFETY: a view of ours.
    if unsafe { VirtualProtect(at as *const _, len as usize, if write { PAGE_READWRITE } else { PAGE_READONLY }, &mut old) } == 0 {
        return Err(last("protect a page"));
    }
    Ok(())
}

/// The view goes; its range is a placeholder again.
pub fn unmap_page(at: u64, _len: u64) -> Result<()> {
    // SAFETY: a view of ours, returned to a placeholder.
    if unsafe { UnmapViewOfFile2(GetCurrentProcess(), MEMORY_MAPPED_VIEW_ADDRESS { Value: at as *mut _ }, MEM_PRESERVE_PLACEHOLDER) } == 0 {
        return Err(last("unmap a page"));
    }
    Ok(())
}

pub fn tier_file(_: &str) -> Result<Owned> {
    Err(Error::unsupported(Seam::SharedMemory))
}

pub fn punch(_: &Owned, _: u64, _: u64) -> Result<()> {
    Err(Error::unsupported(Seam::SharedMemory))
}

pub fn allocated(_: &Owned) -> u64 {
    0
}

/// The calling thread's id.
pub fn thread_id() -> u32 {
    // SAFETY: no arguments.
    unsafe { GetCurrentThreadId() }
}

/// Thread `tid` of `pid` cannot run: it is suspended, or its process has exited.
pub fn stalled(pid: u32, tid: u32) -> bool {
    // SAFETY: query-only handles, closed below.
    unsafe {
        let process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        if !process.is_null() {
            let mut code = 0u32;
            let exited = GetExitCodeProcess(process, &mut code) != 0 && code != STILL_ACTIVE as u32;
            CloseHandle(process);
            if exited {
                return true;
            }
        }
        let thread = OpenThread(THREAD_QUERY_INFORMATION, 0, tid);
        if thread.is_null() {
            return false;
        }
        let mut suspended = 0u32;
        let status = NtQueryInformationThread(thread, ThreadSuspendCount, (&mut suspended as *mut u32).cast(), 4, std::ptr::null_mut());
        CloseHandle(thread);
        status >= 0 && suspended > 0
    }
}

/// A process's high-water is measured on Linux only (`/proc`): here a call is unmeasured,
/// and the RAM tier (Linux's) is absent anyway.
pub fn high_water() -> Option<u64> {
    None
}

pub fn reset_high_water() {}

pub fn anon(_: u32) -> Option<u64> {
    None
}
