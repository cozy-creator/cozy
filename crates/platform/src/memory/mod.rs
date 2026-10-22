//! The seams under varena's shared memory: one named, growable table per GPU and user with
//! byte-range locks that die with their holder; a local event; a named wake endpoint every
//! participant can signal; local services whose bound name is their lock, served without
//! waiting on any one peer; host pages (shareable, mappable into reserved address space,
//! exportable read-only); a thread's id and whether it can run. Linux, macOS and Windows each
//! implement the same calls, so the code above them is one path.
//!
//! | | Linux | macOS | Windows |
//! |---|---|---|---|
//! | table | `/dev/shm` file | file in a private `/tmp` dir | file in the user's temp dir |
//! | row lock | OFD lock | `flock` on a per-range file | `LockFileEx` far past the data |
//! | event | eventfd | datagram socket pair | auto-reset event |
//! | wake endpoint | abstract datagram socket | `flock` + datagram socket file | named event |
//! | service | abstract socket | `flock` + socket file | named pipe (first instance) |
//! | serving | one poll over all | one poll over all | a thread per connection |
//! | host page | memfd | anonymous `shm_open` | pagefile section in a placeholder |
//! | stall | `/proc` thread state | `proc_pidinfo` status | suspend count, exit code |
//!
//! The RAM tier's files (punched region by region) are Linux's; elsewhere they answer
//! `Unsupported` and the tier is absent.

/// A shared mapping; unmapped on drop.
pub struct Map {
    pub base: *mut u8,
    pub len: usize,
}

// SAFETY: the mapping is reached only through atomics and bytes its users synchronise.
unsafe impl Send for Map {}
// SAFETY: as above.
unsafe impl Sync for Map {}

#[cfg(unix)]
mod unix;
#[cfg(unix)]
pub use unix::*;

#[cfg_attr(target_os = "linux", path = "linux.rs")]
#[cfg_attr(target_os = "macos", path = "macos.rs")]
#[cfg_attr(windows, path = "windows.rs")]
mod imp;

pub use imp::*;

/// A name made safe for a file, a socket or a pipe.
fn clean(name: &str) -> String {
    name.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' { c } else { '_' }).collect()
}

/// A served message: a little-endian `u32` length, a handle flag (0: none) and the body.
enum Frame<'a> {
    Whole(&'a [u8]),
    Partial,
    /// A message with a handle, which a served call never carries.
    Bad,
}

fn frame(buf: &[u8]) -> Frame<'_> {
    let Some(&[a, b, c, d, fd]) = buf.get(..5) else { return Frame::Partial };
    if fd != 0 {
        return Frame::Bad;
    }
    let len = u32::from_le_bytes([a, b, c, d]) as usize;
    buf.get(5..5 + len).map_or(Frame::Partial, Frame::Whole)
}
