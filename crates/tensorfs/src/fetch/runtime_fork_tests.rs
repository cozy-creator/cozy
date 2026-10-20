//! Real HTTP in a forked process. Only the test's own child is killed on a bounded failure.
use super::{runtime, runtime_slot, Client, SourcePolicy};
use std::{
    io::{Read, Write},
    net::TcpListener,
    sync::{atomic::{AtomicBool, AtomicUsize, Ordering}, mpsc, Arc},
    thread,
    time::{Duration, Instant},
};

struct Server {
    url: String,
    requests: Arc<AtomicUsize>,
    stop: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
}

impl Server {
    fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/fork", listener.local_addr().unwrap());
        listener.set_nonblocking(true).unwrap();
        let requests = Arc::new(AtomicUsize::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let (seen, ended) = (requests.clone(), stop.clone());
        let thread = thread::spawn(move || {
            while !ended.load(Ordering::Acquire) {
                let (mut stream, _) = match listener.accept() {
                    Ok(stream) => stream,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(5));
                        continue;
                    }
                    Err(error) => panic!("loopback accept: {error}"),
                };
                stream.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
                let mut request = Vec::new();
                let mut buffer = [0; 1024];
                while !request.windows(4).any(|bytes| bytes == b"\r\n\r\n") {
                    let n = stream.read(&mut buffer).unwrap();
                    if n == 0 { break; }
                    request.extend_from_slice(&buffer[..n]);
                    assert!(request.len() <= 8192);
                }
                assert!(request.starts_with(b"GET /fork "));
                seen.fetch_add(1, Ordering::Release);
                stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok").unwrap();
            }
        });
        Self { url, requests, stop, thread: Some(thread) }
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        self.thread.take().unwrap().join().unwrap();
    }
}

fn request(url: &str) -> bool {
    let Ok(runtime) = runtime() else { return false; };
    runtime.block_on(async {
        let client = Client::new(SourcePolicy { any_host: true, allow_local: true, ..Default::default() });
        let Ok(mut answer) = client.send("GET", url, &[], None).await else { return false; };
        if answer.status() != 200 { return false; }
        let mut bytes = Vec::new();
        loop {
            match answer.frame().await {
                Ok(Some(part)) => bytes.extend_from_slice(&part),
                Ok(None) => return bytes == b"ok",
                Err(_) => return false,
            }
        }
    })
}

fn fork_request(url: &str) -> libc::pid_t {
    // SAFETY: This test deliberately forks a multithreaded parent to exercise runtime
    // inheritance. The child performs only the target request and exits without unwinding
    // inherited parent owners. The parent owns, bounds, and reaps this exact child.
    let pid = unsafe { libc::fork() };
    assert!(pid >= 0, "fork: {}", std::io::Error::last_os_error());
    if pid == 0 {
        let ok = std::panic::catch_unwind(|| request(url)).unwrap_or(false);
        unsafe { libc::_exit(if ok { 0 } else { 1 }) };
    }
    pid
}

fn reaped_successfully(pid: libc::pid_t) {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let mut status = 0;
        // SAFETY: pid names the test's direct child; status is writable.
        let waited = unsafe { libc::waitpid(pid, &mut status, libc::WNOHANG) };
        if waited == pid {
            assert!(libc::WIFEXITED(status) && libc::WEXITSTATUS(status) == 0, "forked HTTP request failed: {status}");
            return;
        }
        assert!(waited >= 0, "waitpid: {}", std::io::Error::last_os_error());
        if Instant::now() >= deadline {
            // SAFETY: This is still our unreaped child, never an unrelated process.
            unsafe { libc::kill(pid, libc::SIGKILL); libc::waitpid(pid, &mut status, 0); }
            panic!("forked HTTP request never completed (inherited runtime or lock)");
        }
        thread::sleep(Duration::from_millis(10));
    }
}

#[test]
fn child_network_after_initialized_parent() {
    let server = Server::start();
    assert!(request(&server.url));
    reaped_successfully(fork_request(&server.url));
    assert!(request(&server.url), "the original parent's runtime must remain usable");
    assert_eq!(server.requests.load(Ordering::Acquire), 3);
}

#[test]
fn child_never_locks_parent_runtime_mutex() {
    let server = Server::start();
    assert!(request(&server.url));
    let (ready, started) = mpsc::channel();
    let (release, released) = mpsc::channel();
    let holder = thread::spawn(move || {
        let held = runtime_slot().unwrap().lock().unwrap();
        ready.send(()).unwrap();
        released.recv().unwrap();
        drop(held);
    });
    started.recv().unwrap();
    let child = fork_request(&server.url);
    release.send(()).unwrap();
    holder.join().unwrap();
    reaped_successfully(child);
    assert!(request(&server.url), "the original parent's runtime must remain usable");
    assert_eq!(server.requests.load(Ordering::Acquire), 3);
}
