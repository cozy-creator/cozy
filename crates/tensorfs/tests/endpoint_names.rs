//! Format-valid repo names retain their identity across directory and HTTP endpoints.

use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};

use tensorfs::err::Code;
use tensorfs::fetch::{Client, Endpoint, Fresh, SourcePolicy};
use tensorfs::format::{repo_key, ObjectRef, Repo};

struct Server {
    child: Child,
    root: PathBuf,
    url: String,
}

impl Server {
    fn start() -> Self {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "tensorfs-endpoint-names-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let mut child = Command::new("python3")
            .args([
                "-u",
                "-c",
                r#"
import hashlib, http.server, pathlib, sys, urllib.parse
root = pathlib.Path(sys.argv[1])
class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def resource(self):
        with (root / 'requests').open('a') as log: log.write(self.path + '\n')
        url = urllib.parse.urlsplit(self.path)
        if url.query or url.fragment:
            self.send_error(400)
            return None
        return root / urllib.parse.unquote(url.path).lstrip('/')
    def etag(self, data): return '"sha256:' + hashlib.sha256(data).hexdigest() + '"'
    def reply(self, status, data=b'', tag=None):
        self.send_response(status)
        self.send_header('Content-Length', str(len(data)))
        if tag: self.send_header('ETag', tag)
        self.end_headers()
        self.wfile.write(data)
    def do_GET(self):
        path = self.resource()
        if path is None: return
        if not path.is_file(): return self.reply(404)
        data = path.read_bytes()
        tag = self.etag(data)
        if self.headers.get('If-None-Match') == tag: return self.reply(304, tag=tag)
        self.reply(200, data, tag)
    def do_PUT(self):
        path = self.resource()
        if path is None: return
        data = self.rfile.read(int(self.headers['Content-Length']))
        exists = path.is_file()
        if exists:
            if self.headers.get('If-Match') != self.etag(path.read_bytes()): return self.reply(412)
        elif self.headers.get('If-None-Match') != '*': return self.reply(412)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        self.reply(200 if exists else 201, tag=self.etag(data))
server = http.server.HTTPServer(('127.0.0.1', 0), Handler)
print(server.server_port, flush=True)
server.serve_forever()
"#,
            ])
            .arg(&root)
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap();
        let mut port = String::new();
        BufReader::new(child.stdout.take().unwrap())
            .read_line(&mut port)
            .unwrap();
        let port: u16 = port.trim().parse().unwrap();
        Self {
            child,
            root,
            url: format!("http://127.0.0.1:{port}"),
        }
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

#[tokio::test]
async fn repo_names_are_encoded_for_http_and_raw_on_disk() {
    let server = Server::start();
    let client = Client::new(SourcePolicy {
        any_host: true,
        allow_local: true,
        ..Default::default()
    });
    let http = Endpoint::new(&client, &server.url, None)
        .unwrap()
        .anonymous();
    let directory = Endpoint::new(&client, server.root.to_str().unwrap(), None).unwrap();

    for (name, target) in [
        (
            "AcMe Space/MoDEL%?#É",
            "/refs/acme%20space/model%25%3F%23%C3%89.json",
        ),
        ("Org%2F/.Hidden", "/refs/org%252f/.hidden.json"),
        ("組/名", "/refs/%E7%B5%84/%E5%90%8D.json"),
    ]
    // A folder endpoint keeps names raw, and NTFS holds no `?` in a file name.
    .into_iter()
    .filter(|(name, _)| !cfg!(windows) || !name.contains('?'))
    {
        let mut initial = Repo::new(name);
        initial.set_tag("first", ObjectRef::of(b"first"));
        directory
            .update_repo(name, |_| Ok(Some(initial.clone())))
            .await
            .unwrap();
        assert!(
            server.root.join(repo_key(name)).is_file(),
            "raw directory key for {name}"
        );
        let (read, tag) = http.repo(name).await.unwrap().unwrap();
        assert_eq!(read, initial);
        assert_eq!(
            http.repo_since(name, Some(&tag)).await.unwrap(),
            Fresh::Unchanged
        );
        let updated = http
            .update_repo(name, |old| {
                let mut repo = old.unwrap();
                repo.set_tag("second", ObjectRef::of(b"second"));
                Ok(Some(repo))
            })
            .await
            .unwrap();
        assert_eq!(directory.repo(name).await.unwrap().unwrap().0, updated);
        let requests = std::fs::read_to_string(server.root.join("requests")).unwrap();
        assert!(
            requests.lines().rev().take(4).all(|path| path == target),
            "{name}: {requests}"
        );
    }
}

#[tokio::test]
async fn repo_names_reject_traversal_without_an_http_request() {
    let server = Server::start();
    let client = Client::new(SourcePolicy {
        any_host: true,
        allow_local: true,
        ..Default::default()
    });
    let http = Endpoint::new(&client, &server.url, None)
        .unwrap()
        .anonymous();
    let directory = Endpoint::new(&client, server.root.to_str().unwrap(), None).unwrap();
    for name in [
        "/name",
        "org/",
        "./name",
        "../name",
        "org/.",
        "org/..",
        "org/a/b",
        "or\\g/name",
        "org/a\0b",
    ] {
        for endpoint in [&http, &directory] {
            assert_eq!(
                endpoint.repo_bytes(name, None).await.unwrap_err().code,
                Code::KEY_GRAMMAR,
                "GET {name:?}"
            );
            assert_eq!(
                endpoint
                    .put_repo_bytes(name, b"{}".to_vec(), None)
                    .await
                    .unwrap_err()
                    .code,
                Code::KEY_GRAMMAR,
                "PUT {name:?}"
            );
        }
    }
    assert!(!server.root.join("requests").exists());
}
