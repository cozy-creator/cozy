//! Windows: named pipes, byte mode, opened overlapped so a read on one clone never waits behind a
//! write on another (a synchronous pipe handle serializes them). Only this user may open them,
//! and never from another computer. A path names its pipe by the hash of its text.
use crate::{Error, Result, process::Exact};
use sha2::{Digest, Sha256};
use std::{
    io, mem,
    os::windows::io::{
        AsHandle, AsRawHandle, BorrowedHandle, FromRawHandle, IntoRawHandle, OwnedHandle, RawHandle,
    },
    path::Path,
    ptr,
    sync::Mutex,
};
use windows_sys::Win32::{
    Foundation::{
        DUPLICATE_SAME_ACCESS, DuplicateHandle, ERROR_BROKEN_PIPE, ERROR_IO_PENDING,
        ERROR_PIPE_BUSY, ERROR_PIPE_CONNECTED, ERROR_PIPE_NOT_CONNECTED, GENERIC_READ,
        GENERIC_WRITE, HANDLE, HLOCAL, INVALID_HANDLE_VALUE, LocalFree, WAIT_OBJECT_0,
    },
    Security::{
        Authorization::{
            ConvertSidToStringSidW, ConvertStringSecurityDescriptorToSecurityDescriptorW,
            SDDL_REVISION_1,
        },
        GetTokenInformation, PSECURITY_DESCRIPTOR, SECURITY_ATTRIBUTES, TOKEN_QUERY, TOKEN_USER,
        TokenUser,
    },
    Storage::FileSystem::{
        CreateFileW, FILE_FLAG_FIRST_PIPE_INSTANCE, FILE_FLAG_OVERLAPPED, OPEN_EXISTING,
        PIPE_ACCESS_DUPLEX, ReadFile, SECURITY_IDENTIFICATION, SECURITY_SQOS_PRESENT, WriteFile,
    },
    System::{
        IO::{CancelIoEx, GetOverlappedResult, OVERLAPPED},
        Pipes::{
            ConnectNamedPipe, CreateNamedPipeW, GetNamedPipeClientProcessId, GetNamedPipeInfo,
            GetNamedPipeServerProcessId, NMPWAIT_WAIT_FOREVER, PIPE_READMODE_BYTE,
            PIPE_REJECT_REMOTE_CLIENTS, PIPE_SERVER_END, PIPE_TYPE_BYTE, PIPE_UNLIMITED_INSTANCES,
            PIPE_WAIT, WaitNamedPipeW,
        },
        Threading::{
            CreateEventW, GetCurrentProcess, INFINITE, OpenProcessToken, WaitForMultipleObjects,
        },
    },
};

/// Which end this is: the listener's or the dialer's. The peer's process is read by side.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Side {
    Server,
    Client,
}

/// One end of a connected pipe. Read and Write each wait on their own event, so a clone may read
/// while this writes; two reads at once on one value are not supported.
#[derive(Debug)]
pub struct LocalStream {
    pipe: OwnedHandle,
    side: Side,
    reading: OwnedHandle,
    writing: OwnedHandle,
}

impl LocalStream {
    fn new(pipe: OwnedHandle, side: Side) -> io::Result<Self> {
        Ok(Self {
            pipe,
            side,
            reading: event()?,
            writing: event()?,
        })
    }

    /// Another handle on the same pipe, for a reader thread while this one writes.
    pub fn try_clone(&self) -> io::Result<Self> {
        Self::new(self.pipe.try_clone()?, self.side)
    }

    /// The process on the other end.
    pub fn peer_pid(&self) -> io::Result<u32> {
        let mut pid = 0u32;
        // SAFETY: a live pipe handle and a writable pid.
        let read = unsafe {
            match self.side {
                Side::Server => GetNamedPipeClientProcessId(raw(&self.pipe), &mut pid),
                Side::Client => GetNamedPipeServerProcessId(raw(&self.pipe), &mut pid),
            }
        };
        match read {
            0 => Err(io::Error::last_os_error()),
            _ => Ok(pid),
        }
    }

    fn read_into(&self, buf: &mut [u8]) -> io::Result<usize> {
        let len = buf.len().min(u32::MAX as usize) as u32;
        let done = transfer(&self.pipe, &self.reading, |overlapped| {
            // SAFETY: a live pipe, a writable buffer of `len` bytes and an OVERLAPPED that
            // outlives the operation (transfer waits for it).
            unsafe {
                ReadFile(
                    raw(&self.pipe),
                    buf.as_mut_ptr(),
                    len,
                    ptr::null_mut(),
                    overlapped,
                )
            }
        });
        match done {
            Ok(n) => Ok(n as usize),
            // The other end closed: the end of the stream.
            Err(error) if ended(&error) => Ok(0),
            Err(error) => Err(error),
        }
    }

    fn write_from(&self, buf: &[u8]) -> io::Result<usize> {
        let len = buf.len().min(u32::MAX as usize) as u32;
        transfer(&self.pipe, &self.writing, |overlapped| {
            // SAFETY: as in read_into, with a readable buffer.
            unsafe {
                WriteFile(
                    raw(&self.pipe),
                    buf.as_ptr(),
                    len,
                    ptr::null_mut(),
                    overlapped,
                )
            }
        })
        .map(|n| n as usize)
    }
}

impl io::Read for LocalStream {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.read_into(buf)
    }
}

impl io::Read for &LocalStream {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.read_into(buf)
    }
}

impl io::Write for LocalStream {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.write_from(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl io::Write for &LocalStream {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.write_from(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl AsHandle for LocalStream {
    fn as_handle(&self) -> BorrowedHandle<'_> {
        self.pipe.as_handle()
    }
}

impl AsRawHandle for LocalStream {
    fn as_raw_handle(&self) -> RawHandle {
        self.pipe.as_raw_handle()
    }
}

impl IntoRawHandle for LocalStream {
    fn into_raw_handle(self) -> RawHandle {
        self.pipe.into_raw_handle()
    }
}

/// A listener at a path an executor dials: one pipe instance waits for each dialer.
pub struct LocalListener {
    name: Vec<u16>,
    security: Security,
    next: Mutex<Option<OwnedHandle>>,
}

impl LocalListener {
    pub fn bind(path: &Path) -> Result<Self> {
        let security = Security::user_only().map_err(Error::os("secure a pipe"))?;
        let name = pipe_name(&path.to_string_lossy());
        let first = instance(&name, true, &security).map_err(Error::io("listen", path))?;
        Ok(Self {
            name,
            security,
            next: Mutex::new(Some(first)),
        })
    }

    pub fn accept(&self) -> Result<LocalStream> {
        match self.connect_one(None)? {
            Some(stream) => Ok(stream),
            None => unreachable!("only a process ends a wait"),
        }
    }

    /// A dialer, or `None` once `process` has ended first: woken by the kernel, never a timer.
    pub fn accept_unless(&self, process: &Exact) -> Result<Option<LocalStream>> {
        self.connect_one(Some(process.as_handle()))
    }

    fn connect_one(&self, unless: Option<BorrowedHandle<'_>>) -> Result<Option<LocalStream>> {
        let mut next = self.next.lock().unwrap_or_else(|e| e.into_inner());
        let pipe = match next.take() {
            Some(pipe) => pipe,
            None => instance(&self.name, false, &self.security).map_err(Error::os("listen"))?,
        };
        let signal = event().map_err(Error::os("accept"))?;
        // SAFETY: plain-old-data with its event set.
        let mut overlapped: OVERLAPPED = unsafe { mem::zeroed() };
        overlapped.hEvent = raw(&signal);
        // SAFETY: a live pipe instance and an OVERLAPPED that outlives the wait below.
        let connected = unsafe { ConnectNamedPipe(raw(&pipe), &mut overlapped) } != 0;
        let error = io::Error::last_os_error();
        let ready = connected
            || match error.raw_os_error().map(|code| code as u32) {
                // The dialer came between the instance and this call.
                Some(ERROR_PIPE_CONNECTED) => true,
                Some(ERROR_IO_PENDING) => {
                    let mut handles = vec![raw(&signal)];
                    if let Some(process) = unless {
                        handles.push(process.as_raw_handle());
                    }
                    // SAFETY: live handles and their count.
                    let woke = unsafe {
                        WaitForMultipleObjects(handles.len() as u32, handles.as_ptr(), 0, INFINITE)
                    };
                    if woke != WAIT_OBJECT_0 {
                        // The process ended first (or the wait failed): stop the pending connect
                        // and wait for it to stop, so the OVERLAPPED is never written later.
                        // SAFETY: the operation started above on this pipe.
                        unsafe { CancelIoEx(raw(&pipe), &overlapped) };
                        let mut n = 0u32;
                        // SAFETY: as above; waits for the cancellation to land.
                        unsafe { GetOverlappedResult(raw(&pipe), &overlapped, &mut n, 1) };
                        *next = Some(pipe);
                        return match woke == WAIT_OBJECT_0 + 1 {
                            true => Ok(None),
                            false => {
                                Err(Error::os("wait for a dialer")(io::Error::last_os_error()))
                            }
                        };
                    }
                    let mut n = 0u32;
                    // SAFETY: as above; the connect has completed.
                    if unsafe { GetOverlappedResult(raw(&pipe), &overlapped, &mut n, 1) } == 0 {
                        return Err(Error::os("accept")(io::Error::last_os_error()));
                    }
                    true
                }
                _ => return Err(Error::os("accept")(error)),
            };
        debug_assert!(ready);
        // The next dialer finds an instance waiting.
        *next = Some(instance(&self.name, false, &self.security).map_err(Error::os("listen"))?);
        Ok(Some(
            LocalStream::new(pipe, Side::Server).map_err(Error::os("accept"))?,
        ))
    }
}

/// Dials the listener at `path`, waiting while every instance is busy.
pub fn connect(path: &Path) -> Result<LocalStream> {
    let name = pipe_name(&path.to_string_lossy());
    let pipe = dial(&name).map_err(Error::io("connect", path))?;
    LocalStream::new(pipe, Side::Client).map_err(Error::io("connect", path))
}

/// Two connected ends, on a pipe whose name nobody else knows, its dialer checked to be this
/// process.
pub fn pair() -> Result<(LocalStream, LocalStream)> {
    let mut nonce = [0u8; 16];
    getrandom::fill(&mut nonce)
        .map_err(|e| Error::os("name a pipe")(io::Error::other(e.to_string())))?;
    let name = pipe_name(&hex::encode(nonce));
    let security = Security::user_only().map_err(Error::os("secure a pipe"))?;
    let server = instance(&name, true, &security).map_err(Error::os("pair"))?;
    let client = dial(&name).map_err(Error::os("pair"))?;
    let signal = event().map_err(Error::os("pair"))?;
    // SAFETY: plain-old-data with its event set.
    let mut overlapped: OVERLAPPED = unsafe { mem::zeroed() };
    overlapped.hEvent = raw(&signal);
    // SAFETY: a live pipe instance whose dialer has already connected, and an OVERLAPPED that
    // outlives any wait for it.
    if unsafe { ConnectNamedPipe(raw(&server), &mut overlapped) } == 0 {
        let error = io::Error::last_os_error();
        match error.raw_os_error().map(|code| code as u32) {
            Some(ERROR_PIPE_CONNECTED) => {}
            Some(ERROR_IO_PENDING) => {
                let mut n = 0u32;
                // SAFETY: the connect started above; this waits for it.
                if unsafe { GetOverlappedResult(raw(&server), &overlapped, &mut n, 1) } == 0 {
                    return Err(Error::os("pair")(io::Error::last_os_error()));
                }
            }
            _ => return Err(Error::os("pair")(error)),
        }
    }
    let server = LocalStream::new(server, Side::Server).map_err(Error::os("pair"))?;
    if server.peer_pid().map_err(Error::os("pair"))? != std::process::id() {
        return Err(Error::os("pair")(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "another process dialed a private pipe",
        )));
    }
    let client = LocalStream::new(client, Side::Client).map_err(Error::os("pair"))?;
    Ok((client, server))
}

/// The pipe end whose handle the caller hands over.
///
/// # Safety
/// `raw` must be an open, connected pipe handle the caller owns and gives up.
pub unsafe fn adopt(raw: i64) -> LocalStream {
    // SAFETY: the caller's promise above.
    let pipe = unsafe { OwnedHandle::from_raw_handle(raw as RawHandle) };
    let mut flags = 0u32;
    // SAFETY: a live pipe handle and a writable flags word.
    let side = match unsafe {
        GetNamedPipeInfo(
            pipe.as_raw_handle(),
            &mut flags,
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
        )
    } {
        0 => Side::Client,
        _ if flags & PIPE_SERVER_END != 0 => Side::Server,
        _ => Side::Client,
    };
    LocalStream::new(pipe, side).expect("an event for an adopted pipe")
}

fn pipe_name(text: &str) -> Vec<u16> {
    let digest = Sha256::digest(text.as_bytes());
    format!(r"\\.\pipe\cozy-{}", hex::encode(&digest[..16]))
        .encode_utf16()
        .chain([0])
        .collect()
}

fn instance(name: &[u16], first: bool, security: &Security) -> io::Result<OwnedHandle> {
    let mode = PIPE_ACCESS_DUPLEX
        | FILE_FLAG_OVERLAPPED
        | if first {
            FILE_FLAG_FIRST_PIPE_INSTANCE
        } else {
            0
        };
    let kind = PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT | PIPE_REJECT_REMOTE_CLIENTS;
    let attributes = security.attributes();
    // SAFETY: a NUL-terminated name and security attributes that outlive the call.
    let pipe = unsafe {
        CreateNamedPipeW(
            name.as_ptr(),
            mode,
            kind,
            PIPE_UNLIMITED_INSTANCES,
            64 << 10,
            64 << 10,
            0,
            &attributes,
        )
    };
    owned(pipe)
}

fn dial(name: &[u16]) -> io::Result<OwnedHandle> {
    loop {
        // SAFETY: a NUL-terminated name; no template file.
        let pipe = unsafe {
            CreateFileW(
                name.as_ptr(),
                GENERIC_READ | GENERIC_WRITE,
                0,
                ptr::null(),
                OPEN_EXISTING,
                FILE_FLAG_OVERLAPPED | SECURITY_SQOS_PRESENT | SECURITY_IDENTIFICATION,
                ptr::null_mut(),
            )
        };
        if pipe != INVALID_HANDLE_VALUE {
            // SAFETY: a new handle that is now ours.
            return Ok(unsafe { OwnedHandle::from_raw_handle(pipe) });
        }
        let error = io::Error::last_os_error();
        if error.raw_os_error() != Some(ERROR_PIPE_BUSY as i32) {
            return Err(error);
        }
        // Every instance is taken: the kernel wakes this when one frees.
        // SAFETY: a NUL-terminated name.
        if unsafe { WaitNamedPipeW(name.as_ptr(), NMPWAIT_WAIT_FOREVER) } == 0 {
            return Err(io::Error::last_os_error());
        }
    }
}

/// Runs one overlapped read or write and waits for it to finish.
fn transfer(
    pipe: &OwnedHandle,
    event: &OwnedHandle,
    start: impl FnOnce(*mut OVERLAPPED) -> i32,
) -> io::Result<u32> {
    // SAFETY: plain-old-data with its event set.
    let mut overlapped: OVERLAPPED = unsafe { mem::zeroed() };
    overlapped.hEvent = raw(event);
    if start(&mut overlapped) == 0 {
        let error = io::Error::last_os_error();
        if error.raw_os_error() != Some(ERROR_IO_PENDING as i32) {
            return Err(error);
        }
    }
    let mut done = 0u32;
    // SAFETY: the operation started above on this pipe; this waits for it.
    match unsafe { GetOverlappedResult(raw(pipe), &overlapped, &mut done, 1) } {
        0 => Err(io::Error::last_os_error()),
        _ => Ok(done),
    }
}

fn ended(error: &io::Error) -> bool {
    matches!(
        error.raw_os_error().map(|code| code as u32),
        Some(ERROR_BROKEN_PIPE | ERROR_PIPE_NOT_CONNECTED)
    )
}

/// A manual-reset event, unsignaled.
fn event() -> io::Result<OwnedHandle> {
    // SAFETY: no attributes and no name.
    owned(unsafe { CreateEventW(ptr::null(), 1, 0, ptr::null()) })
}

fn owned(handle: HANDLE) -> io::Result<OwnedHandle> {
    if handle.is_null() || handle == INVALID_HANDLE_VALUE {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: a new handle that is now ours.
    Ok(unsafe { OwnedHandle::from_raw_handle(handle) })
}

fn raw(handle: &OwnedHandle) -> HANDLE {
    handle.as_raw_handle()
}

/// A security descriptor whose only ACE gives this user every right: no other user, and no
/// service, may open the pipe.
struct Security(PSECURITY_DESCRIPTOR);

// SAFETY: a LocalAlloc'd descriptor, only read after it is made.
unsafe impl Send for Security {}
// SAFETY: as above.
unsafe impl Sync for Security {}

impl Security {
    fn user_only() -> io::Result<Self> {
        let sid = user_sid()?;
        let sddl: Vec<u16> = format!("D:P(A;;GA;;;{sid})")
            .encode_utf16()
            .chain([0])
            .collect();
        let mut descriptor: PSECURITY_DESCRIPTOR = ptr::null_mut();
        // SAFETY: a NUL-terminated SDDL string and a writable descriptor pointer.
        if unsafe {
            ConvertStringSecurityDescriptorToSecurityDescriptorW(
                sddl.as_ptr(),
                SDDL_REVISION_1,
                &mut descriptor,
                ptr::null_mut(),
            )
        } == 0
        {
            return Err(io::Error::last_os_error());
        }
        Ok(Self(descriptor))
    }

    fn attributes(&self) -> SECURITY_ATTRIBUTES {
        SECURITY_ATTRIBUTES {
            nLength: mem::size_of::<SECURITY_ATTRIBUTES>() as u32,
            lpSecurityDescriptor: self.0,
            bInheritHandle: 0,
        }
    }
}

impl Drop for Security {
    fn drop(&mut self) {
        // SAFETY: the descriptor the conversion allocated, freed once.
        unsafe { LocalFree(self.0 as HLOCAL) };
    }
}

/// This process's user, as an SID string.
fn user_sid() -> io::Result<String> {
    let mut token: HANDLE = ptr::null_mut();
    // SAFETY: this process's pseudo-handle and a writable token handle.
    if unsafe { OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut token) } == 0 {
        return Err(io::Error::last_os_error());
    }
    let token = owned(token)?;
    let mut size = 0u32;
    // SAFETY: a size query: no buffer.
    unsafe { GetTokenInformation(raw(&token), TokenUser, ptr::null_mut(), 0, &mut size) };
    let mut buffer = vec![0u64; (size as usize).div_ceil(8)];
    // SAFETY: a buffer of at least `size` bytes, aligned for TOKEN_USER.
    if unsafe {
        GetTokenInformation(
            raw(&token),
            TokenUser,
            buffer.as_mut_ptr().cast(),
            size,
            &mut size,
        )
    } == 0
    {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: the call filled a TOKEN_USER at the buffer's start.
    let user = unsafe { &*(buffer.as_ptr() as *const TOKEN_USER) };
    let mut text: *mut u16 = ptr::null_mut();
    // SAFETY: the SID the token names and a writable string pointer.
    if unsafe { ConvertSidToStringSidW(user.User.Sid, &mut text) } == 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: a NUL-terminated string the conversion allocated.
    let sid = unsafe {
        let len = (0..).take_while(|&i| *text.add(i) != 0).count();
        String::from_utf16_lossy(std::slice::from_raw_parts(text, len))
    };
    // SAFETY: the string the conversion allocated, freed once.
    unsafe { LocalFree(text as HLOCAL) };
    Ok(sid)
}

/// Duplicates `handle` into the process at the other end of `stream`; the value it has there.
pub(crate) fn duplicate_into_peer(stream: &LocalStream, handle: RawHandle) -> io::Result<u64> {
    use windows_sys::Win32::System::Threading::{OpenProcess, PROCESS_DUP_HANDLE};
    let peer = stream.peer_pid()?;
    // SAFETY: a plain call; a null handle is the failure.
    let process = owned(unsafe { OpenProcess(PROCESS_DUP_HANDLE, 0, peer) })?;
    let mut target: HANDLE = ptr::null_mut();
    // SAFETY: live source and target processes, a live handle and a writable target.
    if unsafe {
        DuplicateHandle(
            GetCurrentProcess(),
            handle,
            raw(&process),
            &mut target,
            0,
            0,
            DUPLICATE_SAME_ACCESS,
        )
    } == 0
    {
        return Err(io::Error::last_os_error());
    }
    Ok(target as u64)
}
