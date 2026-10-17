//! The machine role: `cozy machine serve --rental|--persistent` on a pod or this computer, and
//! the `tensord` binary's modes, which TensorD's and Calcifer's tests run.
use crate::{
    config::{self, Flags},
    machine::grant::Lifetime,
    owner::{Owner, Shared},
    protocol::{self, Body, Command, Reply, CAPS},
};
use std::{
    collections::HashSet,
    io::{self, Write},
    os::unix::{
        fs::PermissionsExt,
        net::{UnixListener, UnixStream},
    },
    path::PathBuf,
    process::ExitCode,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};
use tensorfs_core::fetch::{Fetcher, Volume};

/// `cozy machine serve --rental|--persistent`, from the arguments after `machine`: this
/// computer's or a pod's machine, until it stops.
pub fn main(args: impl IntoIterator<Item = std::ffi::OsString>) -> ExitCode {
    #[derive(clap::Parser)]
    #[command(name = "cozy machine serve", about = "Run this computer's machine (TensorD)")]
    #[command(group(clap::ArgGroup::new("lifetime").args(["rental", "persistent"]).required(true)))]
    struct Serve {
        #[command(flatten)]
        flags: Flags,
    }
    let flags = <Serve as clap::Parser>::parse_from(args).flags;
    machine(flags.lifetime().expect("clap requires one"), false)
}

/// The machine, as its stable parent (or, with `service`, as the service an activated binary
/// runs under that parent). Exits the process on failure.
pub fn machine(lifetime: Lifetime, service: bool) -> ExitCode {
    if let Err(error) = run(lifetime, service) {
        config::init_logs(None);
        tracing::error!("{error}");
        std::process::exit(1);
    }
    ExitCode::SUCCESS
}

fn run(lifetime: Lifetime, service: bool) -> io::Result<()> {
    use crate::machine::{identity, provider::ProviderSelf, ssh, supervise, update};
    // Linux names the calling thread: this one, before any other starts.
    if let Err(error) = cozy_platform::process::set_name("tensord") {
        if !error.is_unsupported() {
            return Err(io::Error::other(format!("cannot set TensorD process name: {error}")));
        }
    }
    // Whatever this process inherited (a parent's readiness, key or credential pipe) reaches
    // nothing it starts; descriptors it hands on are handed explicitly.
    cozy_platform::process::cloexec_from(3);
    hold_handed_descriptors();
    let mut config = config::load(lifetime)?;
    config::init_logs(Some(&config.logs));
    for warning in &config.warnings {
        tracing::warn!("{warning}");
    }
    let grant = &mut config.grant;
    // A pod's provider credential, kept from everything this machine starts: on its first start
    // this re-executes the machine without the key in its environment, before any thread.
    if lifetime == Lifetime::Rental {
        grant.provider = ProviderSelf::take(&grant.layout.root, &config.provider);
    }
    config::forget_receipt_key();
    let ready = if service {
        // The parent's key pipe is read and closed whether or not the grant has a key.
        let inherited = supervise::inherited_key();
        grant.receipt_key = grant.receipt_key.take().or(inherited);
        supervise::inherited()
    } else {
        identity::hold(&grant.layout)?;
        if let Some(key) = grant.developer_key.as_deref() {
            if let Err(error) = ssh::start(key) {
                tracing::warn!("{error}");
            }
        }
        let paths = update::Paths::new(&grant.layout.engine(), &grant.layout.root);
        supervise::supervise(&paths, lifetime, grant.receipt_key.as_deref(), grant.provider.as_ref())?
    };
    run_machine(config, ready)
}

/// Descriptors 3, 4 and 5 carry a parent's readiness, key and credential pipes. Until they are
/// read, nothing else may take those numbers: the log file there would be read as a pipe, or a
/// forked service's log written into the readiness pipe.
fn hold_handed_descriptors() {
    for fd in 3..=5 {
        // SAFETY: plain syscalls on this process's own descriptor table, before any thread.
        unsafe {
            if libc::fcntl(fd, libc::F_GETFD) < 0 {
                let null = libc::open(c"/dev/null".as_ptr(), libc::O_RDONLY | libc::O_CLOEXEC);
                if null >= 0 && null != fd {
                    libc::dup3(null, fd, libc::O_CLOEXEC);
                    libc::close(null);
                }
            }
        }
    }
}

/// The machine's one async runtime: its API, its lifecycle and TensorFS's transport, where every
/// object request in flight is a task, never a thread. Made in the service, after the stable
/// parent's fork and before anything fetches.
fn machine_runtime() -> io::Result<&'static tokio::runtime::Runtime> {
    static RUNTIME: std::sync::OnceLock<tokio::runtime::Runtime> = std::sync::OnceLock::new();
    if let Some(runtime) = RUNTIME.get() {
        return Ok(runtime);
    }
    // The API's two threads and the transport's two, together.
    let runtime = tokio::runtime::Builder::new_multi_thread().worker_threads(4).thread_name("tensord").enable_all().build()?;
    tensorfs_core::transport::use_runtime(runtime.handle().clone());
    Ok(RUNTIME.get_or_init(|| runtime))
}

/// `tensord serve`: a machine on a state directory with no grant, for tests.
#[derive(Debug, clap::Args)]
pub struct Dev {
    #[arg(long)]
    state: PathBuf,
    #[arg(long)]
    generations: Option<PathBuf>,
    #[arg(long)]
    machine_config: Option<PathBuf>,
    #[arg(long)]
    listen: Option<String>,
    #[arg(long)]
    gpu_config: Option<PathBuf>,
    #[arg(long)]
    installer_python: Option<PathBuf>,
    #[arg(long)]
    client_wheel: Option<PathBuf>,
    #[arg(long, default_value = "3.12")]
    package_python: String,
    #[arg(long, default_value = "uv")]
    uv: PathBuf,
    #[arg(long)]
    package_sdk: Vec<String>,
    #[arg(long)]
    package_find_links: Option<PathBuf>,
    #[arg(long, default_value_t = 1)]
    cpu_parallelism: usize,
    #[arg(long, default_value_t = 16 * 1024 * 1024)]
    host_bytes: u64,
    #[arg(long, default_value_t = 300)]
    cache_ttl_seconds: u64,
}

pub fn dev(args: Dev) -> io::Result<()> {
    config::init_logs(None);
    if let Err(error) = cozy_platform::process::set_name("tensord") {
        if !error.is_unsupported() {
            return Err(io::Error::other(format!("cannot set TensorD process name: {error}")));
        }
    }
    cozy_platform::process::cloexec_from(3);
    machine_runtime()?;
    let root = args.state;
    let generations = args.generations.unwrap_or_else(|| root.join("generations"));
    let owner = Owner::new(&root, &root.join("tensorfs"), args.host_bytes, Duration::from_secs(args.cache_ttl_seconds))?;
    let service = crate::service::Service::open(&root, &generations, args.cpu_parallelism)?;
    let pool = match args.gpu_config {
        Some(config) => crate::gpu_service::GpuConfig::load(&config)?,
        None => crate::gpu_service::GpuConfig::deviceless(&root.join("cpu").join("home"))?,
    };
    service.configure_pool(crate::gpu_service::Pool::new(&root.join("gpu"), pool, owner.lock().unwrap().store())?)?;
    let listener = bind_control(&owner)?;
    let sdk = crate::published::PackageSdk {
        uv: args.uv,
        requirements: args.package_sdk,
        find_links: args.package_find_links,
        python: args.package_python.clone(),
        ..Default::default()
    };
    match (args.machine_config, args.listen) {
        (Some(config), Some(listen)) => {
            let identity = crate::api::MachineIdentity::retained(&crate::api::identity::MachineConfig::load(&config)?)?;
            start_api(&root, &generations, identity, std::net::TcpListener::bind(listen)?, &owner, &service,
                args.installer_python, args.client_wheel, args.package_python, sdk, None, None)?
        }
        (None, None) => {
            crate::calls::Calls::configure(&service, owner.lock().unwrap().store(), None)?;
        }
        _ => return Err(io::Error::other("--machine-config and --listen are required together")),
    }
    serve(owner, service, listener)
}

/// `tensord version --json`: what a candidate binary declares to the machine's update.
pub fn version() -> io::Result<()> {
    #[derive(serde::Serialize)]
    struct Version {
        name: &'static str,
        version: &'static str,
        #[serde(skip_serializing_if = "Option::is_none")]
        commit: Option<&'static str>,
        api: [&'static str; 1],
        capabilities: Vec<&'static str>,
    }
    // Machine contracts beside the private socket's: clients choose by capability.
    let capabilities = crate::api::CAPABILITIES
        .iter()
        .chain(CAPS)
        .chain(&[crate::machine::provider::CREDENTIAL_CAPABILITY])
        .copied()
        .collect();
    let record = Version {
        name: "tensord",
        version: env!("CARGO_PKG_VERSION"),
        commit: option_env!("TENSORD_COMMIT"),
        api: ["cozy.machine.v1"],
        capabilities,
    };
    println!("{}", serde_json::to_string(&record)?);
    Ok(())
}

/// Runs the machine from its grant: identity and readiness under the machine root, the engine
/// under `var/lib/cozy/tensord`, the API on the granted port.
fn run_machine(config: config::Config, mut ready: crate::machine::supervise::Ready) -> io::Result<()> {
    let config::Config { mut grant, visible_devices, endpoint, logs, .. } = config;
    let runtime = machine_runtime()?;
    use crate::machine::{identity, receipt};
    // An older stable parent may have selected a partially published candidate. Restore the
    // links before opening services, then let that parent select the restored executable.
    let paths = crate::machine::update::Paths::new(&grant.layout.engine(), &grant.layout.root);
    if crate::machine::update::recover_activation(&paths)? {
        std::process::exit(crate::machine::update::REPLACE_EXIT);
    }
    let layout = grant.layout.clone();
    let lifetime = identity::open(&layout)?;
    let rental = grant.lifetime == Lifetime::Rental;
    let readiness = receipt::Readiness::open(
        Some(layout.bootstrap.join("readiness-envelope.json")),
        grant.receipt_key.take(),
        rental,
    )?;
    if let Some(attested) = readiness.attested() {
        if receipt::boot_id(&attested).as_deref() != Some(lifetime.boot_id.as_str()) {
            return Err(io::Error::other(
                "the retained readiness envelope names another boot than this machine root",
            ));
        }
    }
    let fresh = readiness.attested().is_none();
    let keys = grant.keys()?;
    let mut identity = crate::api::MachineIdentity::machine(
        grant.worker_id.clone(),
        keys.clone(),
        lifetime,
        readiness,
    )?;
    let lifecycle = crate::machine::lifecycle::Lifecycle::open(
        layout.state.join("idle.json"),
        rental,
        fresh,
        grant.idle_ms,
    )?;
    identity.lifecycle = Some(lifecycle.clone());
    let engine = layout.engine();
    identity.store = Some(layout.store.clone());
    let generations = engine.join("generations");
    let owner = Owner::new(&engine, &layout.store, 16 * 1024 * 1024, Duration::from_secs(300))?;
    // The machine's one Fetcher, held for its life: it reads the datacenter volume before the Hub
    // and fills it. A variable naming no mount point (the volume was not delivered) is no volume.
    let _fetcher = match layout.cache.as_ref().and_then(|cache| Volume::mounted(cache, crate::reclaim::TTL)) {
        Some(volume) => Some(Fetcher::of(&owner.lock().unwrap().store()).map_err(io::Error::other)?.with_volume(volume)),
        None => None,
    };
    let service = crate::service::Service::open(&engine, &generations, 1)?;
    // The jobs this root held (a preparation the restart ended counts as ended now): Status
    // reports the true deadline from the first frame.
    let (working, ended_ms) = service.engine.jobs(0)?;
    lifecycle.observe(crate::machine::lifecycle::Jobs { working, ended_ms }, lifecycle.epoch());
    let paths = crate::machine::update::Paths::new(&engine, &layout.root);
    let updates = {
        let (service, admitted) = (service.clone(), lifecycle.clone());
        let idle = move || service.idle().unwrap_or(false) && admitted.admitted() == 0;
        crate::machine::update::Updates::open(
            paths.clone(),
            Box::new(idle),
            Some(lifecycle.clone()),
            Some(owner.lock().unwrap().store()),
        )?
    };
    identity.updates = Some(updates.clone());
    let readiness = identity.readiness.clone();
    std::thread::Builder::new()
        .name("readiness-report".into())
        .spawn(move || {
            readiness.wait_proved();
            // A Runtime update this process started on is now committed.
            if let Err(error) = updates.commit() {
                tracing::warn!("Runtime update commit: {error}");
            }
            ready.report();
        })?;
    // A machine serves every package on its pool: its GPUs (the envelope, sorted by index; a
    // degree-K plan uses the first K) and deviceless executors, with the executor environment
    // the pod proofs used.
    let mut gpus = receipt::gpus(visible_devices.as_deref())?;
    // A driver below CUDA 13.0's floor serves no GPU; the machine stays for CPU work.
    if let Some(why) = gpus.first().and_then(|gpu| cozy_platform::gpu::below_floor(&gpu.driver_version)) {
        tracing::error!("{why}; this machine offers no GPU");
        gpus.clear();
    }
    identity.gpus = gpus.clone();
    {
        let devices: Vec<String> = gpus.iter().map(|g| g.device_index.to_string()).collect();
        let home = engine.join("executor-home");
        std::fs::create_dir_all(&home)?;
        let mut config = serde_json::json!({
            "devices": devices.join(","),
            "inventory": gpus,
            "authorized_device_limit_bytes": null,
            "pinned_budget_bytes": 4i64 << 30,
            "environment": {"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8", "HOME": home, "COZY_HOME": home},
            // A rented pod is the machine's alone (its grant says so); anywhere else it is a guest.
            "host": {"mode": if rental { "dedicated" } else { "shared" }},
            "image_kernels": (!gpus.is_empty()).then(|| image_kernels(&layout.root, &paths.sdk())).flatten(),
        });
        // The operator's GPU settings, when the machine root has them, over these defaults.
        if let Ok(file) = std::fs::File::open(layout.state.join("gpu-config.json")) {
            let settings: serde_json::Map<String, serde_json::Value> =
                serde_json::from_reader(file).map_err(io::Error::other)?;
            config.as_object_mut().expect("an object").extend(settings);
        }
        let config: crate::gpu_service::GpuConfig =
            serde_json::from_value(config).map_err(io::Error::other)?;
        let store = owner.lock().unwrap().store();
        service.configure_pool(crate::gpu_service::Pool::new(
            &engine.join("gpu"),
            config,
            store,
        )?)?;
    }
    if let Some(hub) = grant.hub.clone().filter(|_| rental) {
        identity.hubs = vec![(hub.origin.clone(), hub.worker_id.clone())];
        let hub = Arc::new(crate::machine::hub::Hub::new(hub)?);
        let service = service.clone();
        let provider = grant.provider.clone();
        std::thread::Builder::new()
            .name("rental-lifecycle".into())
            .spawn(move || {
                runtime.block_on(async {
                    tokio::spawn(crate::machine::lifecycle::keep_authority(
                        hub.clone(),
                        keys,
                    ));
                    let jobs = move |since| {
                        let (working, ended_ms) = service.engine.jobs(since)?;
                        Ok(crate::machine::lifecycle::Jobs { working, ended_ms })
                    };
                    crate::machine::lifecycle::release_when_idle(lifecycle, hub, provider, jobs).await;
                });
                // The Hub accepted the release, or the rental ended itself: this process ends.
                std::process::exit(0);
            })?;
    }
    let control = bind_control(&owner)?;
    let api = crate::machine::net::kept_listener(&layout.state.join("api-port"), grant.listen_host, grant.worker_port)?;
    identity.webrtc = crate::machine::player::listen(&layout.state, grant.webrtc_port, !rental)?;
    identity.player = grant.webrtc_reach.clone();
    // The client this binary embeds; the installer helper over it is made at the first install.
    let uv = crate::machine::client::uv(&layout.root);
    let wheel = crate::machine::client::wheel(&engine)?;
    start_api(
        &engine,
        &generations,
        identity,
        api,
        &owner,
        &service,
        Some(crate::machine::client::helper_python(&engine)),
        Some(wheel),
        "3.12".into(),
        crate::published::PackageSdk::selected(&paths.sdk(), &layout.root, uv),
        // A run naming no Hub reads this rental's own Hub's public content.
        grant.hub.clone().filter(|_| rental),
        Some(logs),
    )?;
    // The Hub probes the endpoint the provider put in this pod's environment at once,
    // instead of waiting for the provider to report it.
    if let (Some(hub), Some(endpoint), true) = (grant.hub.clone(), endpoint, rental) {
        std::thread::Builder::new()
            .name("endpoint-report".into())
            .spawn(move || report_endpoint(hub, endpoint))?;
    }
    serve(owner, service, control)
}

/// The image's Runtime interpreter and the executors' Runtime wheel, when both are here: the
/// machine compiles its kernels with them as it starts.
fn image_kernels(root: &std::path::Path, sdk: &std::path::Path) -> Option<serde_json::Value> {
    let python = root.join("opt/cozy/python/bin/python");
    let wheel = std::fs::read_dir(sdk).ok()?.flatten().map(|entry| entry.path()).find(|path| {
        path.file_name().is_some_and(|name| {
            let name = name.to_string_lossy();
            name.starts_with("cozy_runtime-") && name.ends_with(".whl")
        })
    })?;
    python.is_file().then(|| serde_json::json!({"python": python, "runtime_wheel": wheel}))
}


#[allow(clippy::too_many_arguments)]
fn start_api(
    root: &std::path::Path,
    generations: &std::path::Path,
    identity: crate::api::MachineIdentity,
    listener: std::net::TcpListener,
    owner: &Shared,
    service: &Arc<crate::service::Service>,
    helper: Option<PathBuf>,
    wheel: Option<PathBuf>,
    python: String,
    mut sdk: crate::published::PackageSdk,
    own_hub: Option<crate::machine::grant::HubGrant>,
    logs: Option<PathBuf>,
) -> io::Result<()> {
    use crate::{api, machine_api::NativeBackend};
    sdk.client_wheel = wheel.clone();
    let (pair, uv, seed_cache): (Vec<PathBuf>, _, _) = (
        sdk.requirements.iter().map(PathBuf::from).collect(),
        sdk.uv.clone(),
        sdk.seed_cache.clone(),
    );
    let store = owner.lock().unwrap().store();
    let mut backend = NativeBackend::new(service.clone(), identity.authority.clone(), store.clone());
    backend.logs = logs;
    let publisher =
        crate::published::Publisher::new(&root.join("published"), sdk, store.clone())?;
    service.configure_publisher(publisher.clone());
    backend.publisher = Some(publisher);
    // Local packages install only with the helper; package environments get the client either way.
    backend.installer = match (helper, wheel) {
        (Some(helper_python), Some(client_wheel)) => Some(api::install::InstallerConfig {
            // The machine's own helper is made on first use; a given one is the caller's.
            helper_engine: (helper_python == crate::machine::client::helper_python(root)).then(|| root.to_path_buf()),
            helper_python,
            client_wheel,
            python,
            generations: generations.to_path_buf(),
            staging_root: root.join("package-staging"),
            sdk: pair,
            uv: uv.clone(),
            seed_cache,
        }),
        (Some(_), None) => {
            return Err(io::Error::other(
                "--installer-python requires --client-wheel",
            ))
        }
        _ => None,
    };
    let objects = Arc::new(crate::objects::Objects::new(
        &root.join("writes"),
        store.clone(),
        service.engine.clone(),
    )?);
    service.engine.hold_memos(&objects);
    // The leaf the CLI pins is the key a run's capability names (th-241).
    let leaf = crate::hub::Leaf::from_pem(&identity.key_pem).map_err(|e| io::Error::other(e.0))?;
    backend.runs = Some(Arc::new(crate::runs::Runs {
        service: service.clone(),
        objects: objects.clone(),
        publisher: backend.publisher.clone(),
        local: backend.installer.clone().map(|installer| {
            Arc::new(crate::local_source::LocalSources::new(
                objects,
                installer,
                store.clone(),
            ))
        }),
        own_hub: own_hub
            .map(|hub| crate::hub::Source::new(&hub.origin, hub.ca_der, hub.object_hosts, None))
            .transpose()
            .map_err(|e| io::Error::other(e.0))?,
        leaf: Some(Arc::new(leaf)),
        contexts: Default::default(),
    }));
    crate::calls::Calls::configure(service, store.clone(), backend.runs.as_ref())?;
    listener.set_nonblocking(true)?;
    #[derive(serde::Serialize)]
    struct Ready<'a> {
        address: String,
        worker_id: &'a str,
        boot_id: &'a str,
        cert_pem: &'a str,
    }
    let ready = serde_json::to_vec(&Ready {
        address: listener.local_addr()?.to_string(),
        worker_id: &identity.authority.worker_id,
        boot_id: &identity.authority.boot_id,
        cert_pem: &identity.cert_pem,
    })?;
    crate::machine::identity::write_atomic(&root.join("api-ready.json"), &ready, 0o600)?;
    let runtime = machine_runtime()?;
    std::thread::Builder::new()
        .name("machine-api".into())
        .spawn(move || {
            let result = runtime.block_on(async {
                api::serve(
                    tokio::net::TcpListener::from_std(listener)?,
                    identity,
                    Arc::new(backend),
                )
                .await
            });
            if let Err(error) = result {
                tracing::error!("machine API stopped: {error}");
            }
        })?;
    Ok(())
}
fn bind_control(owner: &Shared) -> io::Result<(UnixListener, UnixListener)> {
    let (weights, admin) = {
        let owner = owner.lock().unwrap();
        (owner.socket.clone(), owner.admin.clone())
    };
    Ok((bind_private(&weights)?, bind_private(&admin)?))
}
fn bind_private(path: &std::path::Path) -> io::Result<UnixListener> {
    // Only the singleton owner can remove a stale socket; no active peer store is touched.
    match std::fs::remove_file(path) {
        Ok(()) => (),
        Err(e) if e.kind() == io::ErrorKind::NotFound => (),
        Err(e) => return Err(e),
    }
    let listener = UnixListener::bind(path)?;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
    Ok(listener)
}
fn serve(
    owner: Shared,
    service: Arc<crate::service::Service>,
    (listener, admin): (UnixListener, UnixListener),
) -> io::Result<()> {
    let paths = {
        let owner = owner.lock().unwrap();
        Arc::new([owner.socket.clone(), owner.admin.clone()])
    };
    let stopped = Arc::new(AtomicBool::new(false));
    println!("READY {}", paths[0].display());
    println!("ADMIN {}", paths[1].display());
    io::stdout().flush()?;
    {
        let (owner, stopped, paths, service) = (
            owner.clone(),
            stopped.clone(),
            paths.clone(),
            service.clone(),
        );
        std::thread::Builder::new()
            .name("machine-admin".into())
            .spawn(move || {
                for stream in admin.incoming() {
                    let Ok(mut stream) = stream else {
                        // Out of descriptors, say: the listener still stands, so pause and
                        // accept again rather than spin.
                        std::thread::sleep(Duration::from_millis(100));
                        continue;
                    };
                    if stopped.load(Ordering::Acquire) {
                        break;
                    }
                    if !matches!(os_peer_outside(&stream), Ok(true)) {
                        let _ = protocol::write(
                            &mut stream,
                            &Reply {
                                seq: 0,
                                body: Body::Error {
                                    code: "connection_unavailable",
                                    detail: "administration is refused to the machine's own \
                                             descendants"
                                        .into(),
                                },
                            },
                        );
                        continue;
                    }
                    let (owner, stopped, paths, service) = (
                        owner.clone(),
                        stopped.clone(),
                        paths.clone(),
                        service.clone(),
                    );
                    std::thread::spawn(move || {
                        if let Err(error) = client(stream, None, &owner, &stopped, &paths, &service)
                        {
                            tracing::warn!("admin peer: {error}");
                        }
                    });
                }
            })?;
    }
    for stream in listener.incoming() {
        // A failed accept (out of descriptors, say) is not the end of the weight socket.
        let Ok(stream) = stream else {
            std::thread::sleep(Duration::from_millis(100));
            continue;
        };
        if stopped.load(Ordering::Acquire) {
            break;
        }
        let peer = match owner.lock().unwrap().connect(&stream) {
            Ok(peer) => peer,
            Err(error) => {
                let mut stream = stream;
                let _ = protocol::write(
                    &mut stream,
                    &Reply {
                        seq: 0,
                        body: Body::Error {
                            code: "connection_unavailable",
                            detail: error.to_string(),
                        },
                    },
                );
                continue;
            }
        };
        let (owner, stopped, paths, service) = (
            owner.clone(),
            stopped.clone(),
            paths.clone(),
            service.clone(),
        );
        std::thread::spawn(move || {
            let result = client(stream, Some(peer), &owner, &stopped, &paths, &service);
            owner.lock().unwrap().disconnect(peer);
            if let Err(error) = result {
                tracing::warn!("peer {peer}: {error}");
            }
        });
    }
    for path in paths.iter() {
        std::fs::remove_file(path)?;
    }
    Ok(())
}
fn os_peer_outside(stream: &UnixStream) -> io::Result<bool> {
    Ok(!cozy_platform::handles::peer_descends_from_machine(stream)?)
}
/// `peer` is a registered weight peer; `None` is the owner's admin connection.
fn client(
    mut stream: UnixStream,
    peer: Option<u64>,
    owner: &Shared,
    stopped: &AtomicBool,
    paths: &[PathBuf; 2],
    service: &Arc<crate::service::Service>,
) -> io::Result<()> {
    let mut capabilities = HashSet::new();
    loop {
        let request = match protocol::read(&mut stream) {
            Ok(Some(request)) => request,
            Ok(None) => return Ok(()),
            Err(error) if error.kind() != io::ErrorKind::InvalidData => {
                // A complete malformed JSON request is operation-local; truncated transport ends this peer.
                if !error.to_string().contains("JSON") && error.kind() != io::ErrorKind::Other {
                    return Err(error);
                }
                protocol::write(
                    &mut stream,
                    &Reply {
                        seq: 0,
                        body: Body::Error {
                            code: "invalid_request",
                            detail: error.to_string(),
                        },
                    },
                )?;
                continue;
            }
            Err(error) => return Err(error),
        };
        let seq = request.seq;
        if let Command::Hello {
            runtime,
            capabilities: offered,
        } = request.command
        {
            let _ = runtime; // Version is provenance, never an admission condition.
            capabilities = offered
                .into_iter()
                .filter(|c| CAPS.contains(&c.as_str()))
                .collect();
            let mut selected: Vec<_> = capabilities.iter().cloned().collect();
            selected.sort();
            let incarnation = owner.lock().unwrap().incarnation.clone();
            protocol::write(
                &mut stream,
                &Reply {
                    seq,
                    body: Body::Hello {
                        version: env!("CARGO_PKG_VERSION"),
                        tensorfs: tensorfs_core::VERSION,
                        incarnation,
                        capabilities: selected,
                    },
                },
            )?;
            continue;
        }
        let required = match &request.command {
            Command::Import { .. } => Some("objects.put-fd/1"),
            Command::Attach { .. } => Some("weights.hosted/1"),
            Command::Submit { .. }
            | Command::Execution { .. }
            | Command::Executions
            | Command::Cancel { .. }
            | Command::ReadResult { .. } => Some("execution.cpu/1"),
            _ => None,
        };
        let admin = peer.is_none();
        let wrong = match &request.command {
            Command::Import { .. } | Command::Attach { .. } | Command::Release { .. } => {
                admin.then_some("weight operations use the machine socket")
            }
            Command::Submit { .. }
            | Command::Execution { .. }
            | Command::Executions
            | Command::Cancel { .. }
            | Command::ReadResult { .. }
            | Command::Shutdown => (!admin).then_some("administration uses the admin socket"),
            _ => None,
        };
        if let Some(detail) = wrong {
            if matches!(request.command, Command::Import { .. }) {
                drop(cozy_platform::handles::recv_fd(&stream)?);
            }
            protocol::write(
                &mut stream,
                &Reply {
                    seq,
                    body: Body::Error {
                        code: "wrong_socket",
                        detail: detail.into(),
                    },
                },
            )?;
            continue;
        }
        let peer = peer.unwrap_or(0); // only weight operations, never reached by admin, use it
        if let Some(cap) = required {
            if !capabilities.contains(cap) {
                if matches!(request.command, Command::Import { .. }) {
                    drop(cozy_platform::handles::recv_fd(&stream)?);
                }
                protocol::write(
                    &mut stream,
                    &Reply {
                        seq,
                        body: Body::Error {
                            code: "capability_missing",
                            detail: cap.into(),
                        },
                    },
                )?;
                continue;
            }
        }
        let result: io::Result<(Body, Option<std::fs::File>)> = match request.command {
            Command::Import { object } => {
                let fd = cozy_platform::handles::recv_fd(&stream)?;
                owner
                    .lock()
                    .unwrap()
                    .import(object, fd.into())
                    .map(|body| (body, None))
            }
            Command::Attach { object } => owner
                .lock()
                .unwrap()
                .attach(peer, object)
                .map(|(body, fd)| (body, Some(fd))),
            Command::Release { lease, incarnation } => owner
                .lock()
                .unwrap()
                .release(peer, lease, &incarnation)
                .map(|body| (body, None)),
            Command::Stats => Ok((owner.lock().unwrap().stats(), None)),
            Command::Submit {
                key,
                generation,
                entrypoint,
                input,
            } => service
                .submit(&key, &generation, &entrypoint, input)
                .map(|record| {
                    (
                        Body::Execution {
                            record: Box::new(record),
                        },
                        None,
                    )
                }),
            Command::Execution { id } => service.engine.get(&id).map(|record| {
                (
                    Body::Execution {
                        record: Box::new(record),
                    },
                    None,
                )
            }),
            Command::Executions => service
                .engine
                .list()
                .map(|records| (Body::Executions { records }, None)),
            Command::Cancel { id } => {
                service
                    .engine
                    .cancel(&id, "local-machine-owner")
                    .map(|record| {
                        (
                            Body::Execution {
                                record: Box::new(record),
                            },
                            None,
                        )
                    })
            }
            Command::ReadResult { id, index } => (|| {
                let record = service.engine.get(&id)?;
                let artifact = record
                    .result
                    .as_ref()
                    .and_then(|r| r.artifacts.get(index))
                    .cloned()
                    .ok_or_else(|| io::Error::other("result artifact absent"))?;
                let file = service.engine.open_result(&id, index)?;
                Ok((Body::ResultArtifact { id, artifact }, Some(file)))
            })(),
            Command::Shutdown => {
                // Hold the weight gate while quiescing dispatch: neither half may
                // stop after discovering the other still has live obligations.
                let mut weight_owner = owner.lock().unwrap();
                if !weight_owner.idle() || !service.stop()? {
                    Err(io::Error::other(
                        "accepted work or active leases prevent shutdown",
                    ))
                } else {
                    assert!(weight_owner.stop());
                    drop(weight_owner);
                    let acknowledgement = protocol::write(
                        &mut stream,
                        &Reply {
                            seq,
                            body: Body::Shutdown,
                        },
                    );
                    stopped.store(true, Ordering::Release);
                    for path in paths {
                        let _ = UnixStream::connect(path);
                    }
                    return acknowledgement;
                }
            }
            _ => Ok((
                Body::Error {
                    code: "operation_unsupported",
                    detail: "this operation is not implemented".into(),
                },
                None,
            )),
        };
        let (body, fd) = result.unwrap_or_else(|e| {
            (
                Body::Error {
                    code: "operation_failed",
                    detail: e.to_string(),
                },
                None,
            )
        });
        protocol::write(&mut stream, &Reply { seq, body })?;
        if let Some(fd) = fd {
            cozy_platform::handles::send_fd(&stream, &fd)?;
        }
    }
}

/// Tells the Hub this machine's endpoint, trying again while the Hub cannot be reached. The
/// report only saves the provider's lag: after a few misses the provider's view serves.
fn report_endpoint(grant: crate::machine::grant::HubGrant, endpoint: crate::machine::endpoint::Endpoint) {
    use crate::machine::hub::{Hub, Refusal};
    let runtime = match machine_runtime() {
        Ok(runtime) => runtime,
        Err(error) => return tracing::warn!("endpoint report: {error}"),
    };
    let hub = match Hub::new(grant) {
        Ok(hub) => hub,
        Err(error) => return tracing::warn!("endpoint report: {error}"),
    };
    runtime.block_on(async {
        for attempt in 0..5u32 {
            match hub.report_endpoint(&endpoint).await {
                Ok(true) => return tracing::info!("reported endpoint {}", endpoint.public_host),
                Ok(false) => return tracing::info!("the Hub takes no endpoint reports"),
                Err(Refusal::Denied) => return tracing::warn!("the Hub refused the endpoint report"),
                Err(Refusal::Transport(error)) => {
                    tracing::warn!("endpoint report: {error}");
                    tokio::time::sleep(std::time::Duration::from_secs(1 << attempt)).await;
                }
            }
        }
    });
}
