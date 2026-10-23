//! Windows: a change notification on the directory for names, sizes and last writes. NTFS reports
//! a write through a handle held open only when its cache flushes: at once after a sync or a
//! close (measured, microseconds), not within 10 s otherwise.
#![allow(unsafe_code)]
use crate::{Error, Result};
use std::{io, os::windows::ffi::OsStrExt, path::Path, time::Duration};
use windows_sys::Win32::{
    Foundation::{HANDLE, INVALID_HANDLE_VALUE, WAIT_OBJECT_0, WAIT_TIMEOUT},
    Storage::FileSystem::{
        FILE_NOTIFY_CHANGE_FILE_NAME, FILE_NOTIFY_CHANGE_LAST_WRITE, FILE_NOTIFY_CHANGE_SIZE,
        FindCloseChangeNotification, FindFirstChangeNotificationW, FindNextChangeNotification,
    },
    System::Threading::{INFINITE, WaitForSingleObject},
};

pub(super) struct Watch(HANDLE);

// SAFETY: a change-notification handle may be waited on and closed from any thread.
unsafe impl Send for Watch {}

impl Watch {
    pub(super) fn new(dir: &Path) -> Result<Self> {
        let wide: Vec<u16> = dir.as_os_str().encode_wide().chain([0]).collect();
        let changes =
            FILE_NOTIFY_CHANGE_FILE_NAME | FILE_NOTIFY_CHANGE_SIZE | FILE_NOTIFY_CHANGE_LAST_WRITE;
        // SAFETY: a NUL-terminated wide path; the handle is closed on drop.
        let handle = unsafe { FindFirstChangeNotificationW(wide.as_ptr(), 0, changes) };
        if handle == INVALID_HANDLE_VALUE {
            return Err(Error::io("watch", dir)(io::Error::last_os_error()));
        }
        Ok(Self(handle))
    }

    pub(super) fn wait(&mut self, most: Duration) -> Result<bool> {
        if !self.signaled(most)? {
            return Ok(false);
        }
        // Drain what is already signaled: one wake per wait.
        while self.signaled(Duration::ZERO)? {}
        Ok(true)
    }

    /// Whether a change came within `period`; a change re-arms the notification.
    fn signaled(&self, period: Duration) -> Result<bool> {
        let milliseconds = period.as_millis().min(u128::from(INFINITE - 1)) as u32;
        // SAFETY: a live change-notification handle.
        match unsafe { WaitForSingleObject(self.0, milliseconds) } {
            WAIT_TIMEOUT => Ok(false),
            WAIT_OBJECT_0 => {
                // SAFETY: as above.
                if unsafe { FindNextChangeNotification(self.0) } == 0 {
                    return Err(Error::os("FindNextChangeNotification")(
                        io::Error::last_os_error(),
                    ));
                }
                Ok(true)
            }
            _ => Err(Error::os("WaitForSingleObject")(io::Error::last_os_error())),
        }
    }
}

impl Drop for Watch {
    fn drop(&mut self) {
        // SAFETY: the handle this watch owns, closed once.
        unsafe { FindCloseChangeNotification(self.0) };
    }
}
