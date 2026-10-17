//! An https server on loopback under a certificate of its own: `/files/<name>` serves a directory's
//! files, ranges honored; `/to/<name>` redirects there on another host name, as GitHub's release
//! downloads do.
use std::{
    io::{BufRead as _, BufReader, Write as _},
    net::{TcpListener, TcpStream},
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use rustls::{ServerConnection, StreamOwned};

pub struct Https {
    /// `https://localhost:<port>`
    pub base: String,
    /// Its certificate, PEM: the root a client trusts.
    pub ca: String,
    /// Files served (a redirect is not counted).
    pub served: Arc<AtomicUsize>,
}

impl Https {
    pub fn serve(dir: &Path) -> Https {
        let key = rcgen::generate_simple_self_signed(vec!["localhost".into(), "127.0.0.1".into()])
            .unwrap();
        let cert = rustls::pki_types::CertificateDer::from(key.cert.der().to_vec());
        let private =
            rustls::pki_types::PrivateKeyDer::try_from(key.signing_key.serialize_der()).unwrap();
        let provider = Arc::new(rustls::crypto::aws_lc_rs::default_provider());
        let config = rustls::ServerConfig::builder_with_provider(provider)
            .with_safe_default_protocol_versions()
            .unwrap()
            .with_no_client_auth()
            .with_single_cert(vec![cert], private)
            .unwrap();
        let (config, listener) = (Arc::new(config), TcpListener::bind("127.0.0.1:0").unwrap());
        let port = listener.local_addr().unwrap().port();
        let served = Arc::new(AtomicUsize::new(0));
        let (dir, count) = (dir.to_owned(), served.clone());
        std::thread::spawn(move || {
            for socket in listener.incoming().flatten() {
                let (config, dir, count) = (config.clone(), dir.clone(), count.clone());
                std::thread::spawn(move || {
                    let connection = ServerConnection::new(config).unwrap();
                    let mut stream = StreamOwned::new(connection, socket);
                    let _ = answer(&mut stream, &dir, port, &count);
                    stream.conn.send_close_notify();
                    let _ = stream.flush();
                });
            }
        });
        Https {
            base: format!("https://localhost:{port}"),
            ca: key.cert.pem(),
            served,
        }
    }
}

type Stream = StreamOwned<ServerConnection, TcpStream>;

fn answer(stream: &mut Stream, dir: &Path, port: u16, served: &AtomicUsize) -> std::io::Result<()> {
    let mut reader = BufReader::new(&mut *stream);
    let mut line = String::new();
    reader.read_line(&mut line)?;
    let path = line.split_whitespace().nth(1).unwrap_or("/").to_owned();
    let mut range = None;
    loop {
        let mut header = String::new();
        reader.read_line(&mut header)?;
        let Some((name, value)) = header.trim_end().split_once(':') else {
            break;
        };
        if name.eq_ignore_ascii_case("range") {
            let bounds = value
                .trim()
                .strip_prefix("bytes=")
                .and_then(|r| r.split_once('-'));
            range = bounds.map(|(a, b)| (a.parse::<usize>().unwrap_or(0), b.parse::<usize>().ok()));
        }
    }
    drop(reader);
    let close = "content-length: 0\r\nconnection: close\r\n\r\n";
    if let Some(name) = path.strip_prefix("/to/") {
        let to = format!("location: https://127.0.0.1:{port}/files/{name}");
        return write!(stream, "HTTP/1.1 302 Found\r\n{to}\r\n{close}");
    }
    let file = path.strip_prefix("/files/").filter(|n| !n.contains('/'));
    let Some(body) = file.and_then(|name| std::fs::read(dir.join(name)).ok()) else {
        return write!(stream, "HTTP/1.1 404 Not Found\r\n{close}");
    };
    served.fetch_add(1, Ordering::Relaxed);
    let total = body.len();
    let (status, part, extra) = match range {
        Some((start, end)) if start < total => {
            let end = end.unwrap_or(total - 1).min(total - 1);
            let extra = format!("content-range: bytes {start}-{end}/{total}\r\n");
            ("206 Partial Content", &body[start..=end], extra)
        }
        _ => ("200 OK", &body[..], String::new()),
    };
    let length = part.len();
    write!(
        stream,
        "HTTP/1.1 {status}\r\ncontent-length: {length}\r\n{extra}connection: close\r\n\r\n"
    )?;
    stream.write_all(part)
}
