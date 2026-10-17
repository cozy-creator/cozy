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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::process::{Birth, Exact};
    use std::io::{Read, Write};

    #[test]
    fn a_pair_carries_bytes_both_ways_while_a_clone_reads() {
        let (mut a, b) = pair().unwrap();
        let mut reader = b.try_clone().unwrap();
        let mut writer = b;
        let echo = std::thread::spawn(move || {
            let mut got = [0u8; 5];
            reader.read_exact(&mut got).unwrap();
            got
        });
        writer.write_all(b"pong!").unwrap();
        a.write_all(b"ping!").unwrap();
        let mut got = [0u8; 5];
        a.read_exact(&mut got).unwrap();
        assert_eq!(&got, b"pong!");
        assert_eq!(&echo.join().unwrap(), b"ping!");
        drop(writer);
        drop(a);
    }

    #[test]
    fn a_listener_accepts_a_dialer_and_reads_its_end_as_closed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("machine.sock");
        let listener = LocalListener::bind(&path).unwrap();
        let dialer = std::thread::spawn({
            let path = path.clone();
            move || {
                let mut stream = connect(&path).unwrap();
                stream.write_all(b"hello").unwrap();
            }
        });
        let mut accepted = listener.accept().unwrap();
        let mut got = Vec::new();
        accepted.read_to_end(&mut got).unwrap();
        assert_eq!(got, b"hello");
        dialer.join().unwrap();
    }

    #[test]
    fn accept_unless_answers_none_once_the_awaited_process_ends() {
        let dir = tempfile::tempdir().unwrap();
        let listener = LocalListener::bind(&dir.path().join("machine.sock")).unwrap();
        let mut child = std::process::Command::new(if cfg!(windows) { "cmd" } else { "sh" })
            .args(if cfg!(windows) {
                &["/C", "exit 0"][..]
            } else {
                &["-c", "read x"][..]
            })
            .stdin(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        let birth = Birth::of(child.id()).unwrap();
        let exact = Exact::open(&birth).unwrap();
        drop(child.stdin.take());
        child.wait().unwrap();
        // A process that ended before this asked is ended all the same.
        match exact {
            Some(exact) => assert!(listener.accept_unless(&exact).unwrap().is_none()),
            None => assert!(birth.ended().unwrap()),
        }
    }
}
