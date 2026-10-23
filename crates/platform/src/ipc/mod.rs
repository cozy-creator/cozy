//! A connected local byte stream between a machine and the processes it starts, and the listener
//! they dial. Unix: Unix sockets. Windows: named pipes opened overlapped, so one clone reads while
//! another writes; a path names its pipe by hash, so both ends derive the same name from it.
#[cfg(unix)]
mod unix;
#[cfg(windows)]
mod windows;

#[cfg(unix)]
pub use unix::*;
#[cfg(windows)]
pub use windows::*;
