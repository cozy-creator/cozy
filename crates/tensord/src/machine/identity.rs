//! The machine lifetime: its boot id and TLS leaf, persisted with the root so a restarted
//! service is the same machine. Files and formats are the Go agent's (`layout.go`).
use platform::{
    lock::{Kind, LockFile},
    process::{Birth, State},
};
use super::grant::Layout;
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use std::{
    fs::{self, OpenOptions},
    io::{self, Write},
    path::Path,
};
use platform::durable::Mode;

/// The one DNS name of every machine leaf; clients pin the leaf itself.
pub const SERVER_NAME: &str = "cozy-worker";

pub struct Lifetime {
    pub boot_id: String,
    pub cert_pem: String,
    pub key_pem: String,
    pub cert_der: Vec<u8>,
}

/// One machine per root. Every machine on a root holds this lock for its life, and a launcher
/// probes it before it starts or replaces one: the Go agent's `agent.lock`, so each agent
/// excludes the other. The kernel releases it when the last machine process is gone. A
/// killed predecessor's service holds it a moment longer, exiting (perhaps in uninterruptible
/// sleep, its descriptors already closed): a live holder refuses the root, and a lock only
/// dying processes or none hold is waited out until the kernel releases it.
pub fn hold(layout: &Layout) -> io::Result<()> {
    fs::create_dir_all(&layout.state)?;
    let lock = LockFile::new(layout.state.join("agent.lock"));
    let held = match lock.try_lock(Kind::Exclusive)? {
        Some(held) => held,
        None => {
            let holder = lock.holder()?.and_then(|record| serde_json::from_slice::<Birth>(&record).ok());
            let live = holder.filter(|birth| birth.state().is_ok_and(|state| matches!(state, State::Live | State::Stopped)));
            if let Some(live) = live {
                let root = layout.root.display();
                return Err(io::Error::new(io::ErrorKind::AlreadyExists, format!("another machine owns the root {root} (pid {})", live.pid)));
            }
            lock.lock(Kind::Exclusive)?
        }
    };
    held.publish(&serde_json::to_vec(&Birth::own()?)?)?;
    // Held by this parent and the service it forks; never by an executor (close-on-exec).
    std::mem::forget(held);
    Ok(())
}

/// Mints or reopens this root's lifetime. A root booted by the Go agent keeps its boot id
/// and leaf; so does a later Go agent on a root this service booted.
pub fn open(layout: &Layout) -> io::Result<Lifetime> {
    for dir in [&layout.bootstrap, &layout.state] {
        fs::create_dir_all(dir)?;
    }
    let path = layout.state.join("boot-id");
    let boot_id = match fs::read_to_string(&path) {
        Ok(text) if !text.trim().is_empty() => text.trim().to_owned(),
        Ok(_) => return Err(invalid("the persisted boot id is empty")),
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            let inherited =
                fs::read_to_string(layout.bootstrap.join("pod-boot-id")).unwrap_or_default();
            let boot_id = match inherited.trim() {
                "" => URL_SAFE_NO_PAD.encode(random::<32>()?),
                kept => kept.to_owned(),
            };
            write_atomic(&path, boot_id.as_bytes(), 0o400)?;
            boot_id
        }
        Err(e) => return Err(e),
    };
    if URL_SAFE_NO_PAD.decode(&boot_id).map(|b| b.len()) != Ok(32) {
        return Err(invalid(
            "the persisted boot id is not unpadded base64url for 32 bytes",
        ));
    }
    write_atomic(
        &layout.bootstrap.join("pod-boot-id"),
        format!("{boot_id}\n").as_bytes(),
        0o444,
    )?;
    let (cert_path, key_path) = (
        layout.bootstrap.join("tls.crt"),
        layout.bootstrap.join("tls.key"),
    );
    if !cert_path.exists() {
        let (cert_pem, key_pem) = mint_leaf()?;
        write_atomic(&key_path, key_pem.as_bytes(), 0o400)?;
        write_atomic(&cert_path, cert_pem.as_bytes(), 0o444)?;
    }
    let cert_pem = fs::read_to_string(&cert_path)?;
    let key_pem = fs::read_to_string(&key_path)?;
    let certs: Vec<_> =
        rustls_pemfile::certs(&mut cert_pem.as_bytes()).collect::<Result<_, _>>()?;
    let [cert] = certs.as_slice() else {
        return Err(invalid("the machine leaf must be exactly one certificate"));
    };
    rustls_pemfile::private_key(&mut key_pem.as_bytes())?
        .ok_or_else(|| invalid("the machine leaf has no private key"))?;
    Ok(Lifetime {
        boot_id,
        cert_der: cert.as_ref().to_vec(),
        cert_pem,
        key_pem,
    })
}

/// The leaf the Hub admits: P-256, self-signed, exactly the DNS SAN `cozy-worker`, usable as
/// both TLS server and Tensorhub client, valid until year 9999 (clients pin the bytes).
fn mint_leaf() -> io::Result<(String, String)> {
    use rcgen::{
        CertificateParams, DistinguishedName, DnType, ExtendedKeyUsagePurpose, KeyPair,
        KeyUsagePurpose,
    };
    let key = KeyPair::generate_for(&rcgen::PKCS_ECDSA_P256_SHA256).map_err(io::Error::other)?;
    let mut params = CertificateParams::new(vec![SERVER_NAME.into()]).map_err(io::Error::other)?;
    let mut name = DistinguishedName::new();
    name.push(DnType::CommonName, "tensord");
    params.distinguished_name = name;
    params.not_before = time::OffsetDateTime::now_utc() - time::Duration::minutes(5);
    params.not_after = rcgen::date_time_ymd(9999, 12, 31);
    params.key_usages = vec![KeyUsagePurpose::DigitalSignature];
    params.extended_key_usages = vec![
        ExtendedKeyUsagePurpose::ServerAuth,
        ExtendedKeyUsagePurpose::ClientAuth,
    ];
    let cert = params.self_signed(&key).map_err(io::Error::other)?;
    Ok((cert.pem(), key.serialize_pem()))
}

pub fn random<const N: usize>() -> io::Result<[u8; N]> {
    let mut out = [0; N];
    getrandom::fill(&mut out).map_err(|error| io::Error::other(error.to_string()))?;
    Ok(out)
}

pub fn write_atomic(target: &Path, data: &[u8], mode: u32) -> io::Result<()> {
    let dir = target
        .parent()
        .ok_or_else(|| invalid("a target needs a directory"))?;
    fs::create_dir_all(dir)?;
    let temp = dir.join(format!(".write-{}", URL_SAFE_NO_PAD.encode(random::<9>()?)));
    let result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&temp)?;
        file.write_all(data)?;
        platform::durable::set_file_mode(&file, mode)?;
        file.sync_all()?;
        // Closed before it is named: an executable written here runs at once, and an open
        // writer (this one, or a fork's copy of it) makes exec fail with ETXTBSY.
        drop(file);
        fs::rename(&temp, target)?;
        Ok(platform::durable::sync_dir(dir)?)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result
}

fn invalid(detail: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, detail.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(target_os = "linux")]
    use std::io::Read;

    /// A holder just killed is waited out, never refused; a live one is refused. Its holder
    /// writes a Linux birth (/proc); platform's seams prove the lock on every OS.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_killed_holder_is_waited_out_and_a_live_one_refused() {
        const HOLD: &str = "import fcntl, json, os, sys, time; f = open(sys.argv[1], 'a'); \
            fcntl.flock(f, fcntl.LOCK_EX); \
            stat = open('/proc/self/stat').read().rsplit(') ', 1)[1].split(); \
            boot = open('/proc/sys/kernel/random/boot_id').read().strip(); \
            open(sys.argv[1] + '.holder', 'w').write(json.dumps( \
            {'pid': os.getpid(), 'boot_id': boot, 'start_ticks': int(stat[19])})); \
            print('held', flush=True); time.sleep(600)";
        let root = std::env::temp_dir().join(format!("tensord-hold-{}", URL_SAFE_NO_PAD.encode(random::<9>().unwrap())));
        let layout = Layout::new(&root, None);
        fs::create_dir_all(&layout.state).unwrap();
        let mut holder = std::process::Command::new("python3").args(["-c", HOLD]).arg(layout.state.join("agent.lock"))
            .stdout(std::process::Stdio::piped()).spawn().unwrap();
        holder.stdout.take().unwrap().read_exact(&mut [0u8; 5]).unwrap();
        let birth = Birth::of(holder.id()).unwrap();
        let dying = |birth: &Birth| matches!(birth.state().unwrap(), State::Gone | State::Dying);
        assert!(!dying(&birth));
        assert_eq!(hold(&layout).unwrap_err().kind(), io::ErrorKind::AlreadyExists);
        holder.kill().unwrap();
        assert!(dying(&birth));
        hold(&layout).unwrap();
        holder.wait().unwrap();
        assert!(dying(&birth));
        // This process holds it now: a second machine here is refused.
        assert_eq!(hold(&layout).unwrap_err().kind(), io::ErrorKind::AlreadyExists);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn lifetime_survives_restart_and_adopts_a_go_booted_root() {
        let root = std::env::temp_dir().join(format!(
            "tensord-identity-{}",
            URL_SAFE_NO_PAD.encode(random::<9>().unwrap())
        ));
        let layout = Layout::new(&root, None);
        let first = open(&layout).unwrap();
        let again = open(&layout).unwrap();
        assert_eq!(
            (first.boot_id.as_str(), &first.cert_der),
            (again.boot_id.as_str(), &again.cert_der)
        );
        let cert = x509_parser::parse_x509_certificate(&first.cert_der)
            .unwrap()
            .1;
        let names: Vec<_> = cert
            .subject_alternative_name()
            .unwrap()
            .unwrap()
            .value
            .general_names
            .iter()
            .map(|n| format!("{n}"))
            .collect();
        assert_eq!(names, ["DNSName(cozy-worker)"]);
        // A root the Go agent booted: its boot id is adopted, never replaced.
        fs::remove_file(layout.state.join("boot-id")).unwrap();
        let go = URL_SAFE_NO_PAD.encode([7u8; 32]);
        write_atomic(
            &layout.bootstrap.join("pod-boot-id"),
            format!("{go}\n").as_bytes(),
            0o444,
        )
        .unwrap();
        assert_eq!(open(&layout).unwrap().boot_id, go);
        fs::remove_dir_all(root).unwrap();
    }
}
