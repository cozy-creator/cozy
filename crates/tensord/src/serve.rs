//! The machine role: `cozy machine serve --rental|--persistent` on a pod or this computer, and
//! the `tensord` binary's modes, which TensorD's and Calcifer's tests run.
use crate::{
    config::{self, Flags},
    machine::grant::Lifetime,
};
use std::{io, path::PathBuf, process::ExitCode, sync::Arc};
use fs2::FileExt;
use crate::{origin::Machine, storage::Store};

/// `cozy machine serve --rental|--persistent`, from the arguments after `machine`: this
/// computer's or a pod's machine, until it stops.
pub fn main(args: impl IntoIterator<Item = std::ffi::OsString>) -> ExitCode {
    #[derive(clap::Parser)]
    #[command(name = "cozy machine serve", about = "Run this computer's machine (TensorD)")]
    #[command(group(clap::ArgGroup::new("lifetime").args(["rental", "persistent"]).required(true)))]
    struct Serve {
        #[command(flatten)]
        flags: Flags,
        /// Windows: the readiness pipe a stable parent handed the service it started.
        #[arg(long, hide = true)]
        service_ready: Option<u64>,
    }
    let serve = <Serve as clap::Parser>::parse_from(args);
    machine(serve.flags.lifetime().expect("clap requires one"), serve.service_ready)
}

/// The machine: its stable parent, then the service it starts (`service_ready`: this is that
/// service, on Windows). Exits the process on failure.
pub fn machine(lifetime: Lifetime, service_ready: Option<u64>) -> ExitCode {
    if let Err(error) = run(lifetime, service_ready) {
        config::init_logs(None);
        tracing::error!("{error}");
        std::process::exit(1);
    }
    ExitCode::SUCCESS
}

fn run(lifetime: Lifetime, service_ready: Option<u64>) -> io::Result<()> {
    use crate::machine::{binary, identity, provider::ProviderSelf, ssh, supervise, update};
    // Where this binary was started from, before anything can rename another over it.
    if let Err(error) = binary::record() {
        eprintln!("cannot record this binary's launch path ({error}): it never activates another");
    }
    // Linux names the calling thread: this one, before any other starts.
    if let Err(error) = cozy_platform::process::set_name("tensord") {
        if !error.is_unsupported() {
            return Err(io::Error::other(format!("cannot set TensorD process name: {error}")));
        }
    }
    // Whatever this process inherited (its own re-execution's credential pipe) reaches nothing
    // it starts; descriptors it hands on are handed explicitly. Windows hands on none.
    #[cfg(unix)]
    {
        cozy_platform::process::cloexec_from(3);
        hold_handed_descriptors();
    }
    let mut config = config::load(lifetime)?;
    config::init_logs(Some(&config.logs));
    for warning in &config.warnings {
        tracing::warn!("{warning}");
    }
    // Windows forks nothing: this process is the stable parent until the service it starts as
    // this executable again returns here.
    #[cfg(windows)]
    let ready = supervise::supervise(
        &update::Paths::new(&config.grant.layout.engine(), &config.grant.layout.root),
        None,
        service_ready,
    )?;
    #[cfg(unix)]
    let _ = service_ready;
    let grant = &mut config.grant;
    // A pod's provider credential, kept from everything this machine starts: on its first start
    // this re-executes the machine without the key in its environment, before any thread.
    if lifetime == Lifetime::Rental {
        grant.provider = ProviderSelf::take(&grant.layout.root, &config.provider);
    }
    config::forget_receipt_key();
    identity::hold(&grant.layout)?;
    if let Some(key) = grant.developer_key.as_deref() {
        if let Err(error) = ssh::start(key) {
            tracing::warn!("{error}");
        }
    }
    let paths = update::Paths::new(&grant.layout.engine(), &grant.layout.root);
    update::retire_agent(&paths);
    #[cfg(unix)]
    let ready = supervise::supervise(&paths, grant.provider.clone())?;
    config.gpu_memory_limit = config::gpu_memory_limit()?;
    run_machine(config, ready)
}

/// Descriptor 5 carries a re-executed machine's credential pipe. Until it is read, nothing else
/// may take 3 to 5: a log file at 5 would be read as the pipe.
#[cfg(unix)]
fn hold_handed_descriptors() {
    for fd in 3..=5 {
        // SAFETY: plain syscalls on this process's own descriptor table, before any thread.
        unsafe {
            if libc::fcntl(fd, libc::F_GETFD) < 0 {
                let null = libc::open(c"/dev/null".as_ptr(), libc::O_RDONLY | libc::O_CLOEXEC);
                if null >= 0 && null != fd {
                    libc::dup2(null, fd);
                    libc::fcntl(fd, libc::F_SETFD, libc::FD_CLOEXEC);
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
}

pub fn dev(args: Dev) -> io::Result<()> {
    config::init_logs(None);
    if let Err(error) = cozy_platform::process::set_name("tensord") {
        if !error.is_unsupported() {
            return Err(io::Error::other(format!("cannot set TensorD process name: {error}")));
        }
    }
    #[cfg(unix)]
    cozy_platform::process::cloexec_from(3);
    machine_runtime()?;
    let root = args.state;
    let generations = args.generations.unwrap_or_else(|| root.join("generations"));
    let (_owned, store) = own(&root, &root.join("tensorfs"))?;
    let machine = Machine::new(store.clone(), None)?;
    let service = crate::service::Service::open(&root, &generations, args.cpu_parallelism)?;
    let pool = match args.gpu_config {
        Some(config) => crate::gpu_service::GpuConfig::load(&config)?,
        None => crate::gpu_service::GpuConfig::deviceless(&root.join("cpu").join("home"))?,
    };
    service.configure_pool(crate::gpu_service::Pool::new(&root.join("gpu"), pool, store.clone())?)?;
    // A Runtime wheel named here is the operator's choice, as an update's `--runtime-wheel` is: it
    // runs every package whatever the package's bounds on the pair.
    let runtime_build = (args.package_sdk.iter())
        .filter_map(|r| std::path::Path::new(r).file_name()?.to_str()?.strip_prefix("cozy_runtime-")?.split('-').next().map(str::to_owned))
        .next()
        .unwrap_or_default();
    let sdk = crate::published::PackageSdk {
        uv: args.uv,
        requirements: args.package_sdk,
        find_links: args.package_find_links,
        python: args.package_python.clone(),
        runtime_build,
        ..Default::default()
    };
    match (args.machine_config, args.listen) {
        (Some(config), Some(listen)) => {
            let identity = crate::api::MachineIdentity::retained(&crate::api::identity::MachineConfig::load(&config)?)?;
            start_api(&root, &generations, identity, std::net::TcpListener::bind(listen)?, &store, &machine, &service,
                args.installer_python, args.client_wheel, args.package_python, sdk, None, None)?
        }
        (None, None) => {
            crate::calls::Calls::configure(&service, machine.clone(), None)?;
        }
        _ => return Err(io::Error::other("--machine-config and --listen are required together")),
    }
    drop(service);
    // The machine runs on its own threads until it is stopped.
    loop {
        std::thread::park();
    }
}

/// The state root's one owner: its lock held while the process lives, and the TensorFS store it
/// serves (which may live outside `root`), claimed for this machine alone.
fn own(root: &std::path::Path, store: &std::path::Path) -> io::Result<(std::fs::File, Arc<Store>)> {
    std::fs::create_dir_all(root)?;
    let lock = std::fs::OpenOptions::new().read(true).write(true).create(true).truncate(false)
        .open(root.join("owner.lock"))?;
    lock.try_lock_exclusive()
        .map_err(|_| io::Error::new(io::ErrorKind::AlreadyExists, "state already has a machine owner"))?;
    cozy_platform::durable::set_mode(root, 0o700)?;
    Ok((lock, crate::storage::open(store)?))
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
    // Machine contracts: clients choose by capability.
    let capabilities = crate::api::CAPABILITIES.to_vec();
    let record = Version {
        name: "tensord",
        version: env!("CARGO_PKG_VERSION"),
        commit: Some(cozy_proto::COMMIT),
        api: ["cozy.machine.v1"],
        capabilities,
    };
    println!("{}", serde_json::to_string(&record)?);
    Ok(())
}

/// Runs the machine from its grant: identity and readiness under the machine root, the engine
/// under `var/lib/cozy/tensord`, the API on the granted port.
fn run_machine(config: config::Config, mut ready: crate::machine::supervise::Ready) -> io::Result<()> {
    let config::Config { mut grant, visible_devices, gpu_memory_limit, endpoint, logs, .. } = config;
    let runtime = machine_runtime()?;
    use crate::machine::{identity, receipt};
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
    // A rental's own store only: never the datacenter volume or an operator-named root.
    if rental && layout.own_store() {
        crate::storage::drop_pre_cut(&layout.store)?;
    }
    let (_owned, store) = own(&engine, &layout.store)?;
    // The machine's one downloader, held for its life: it reads the datacenter volume before the
    // Hub and fills it. A variable naming no mount point (the volume was not delivered) is none.
    let machine = Machine::new(store.clone(), layout.cache.as_deref())?;
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
            Some(machine.clone()),
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
    identity.gpu_memory_limit = gpu_memory_limit;
    {
        let devices: Vec<String> = gpus.iter().map(|g| g.device_index.to_string()).collect();
        let home = engine.join("executor-home");
        std::fs::create_dir_all(&home)?;
        let mut config = serde_json::json!({
            "devices": devices.join(","),
            "inventory": gpus,
            "authorized_device_limit_bytes": null,
            "gpu_memory_limit": gpu_memory_limit,
            "environment": {"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8", "HOME": home, "COZY_HOME": home},
            // A rented pod is the machine's alone (its grant says so); anywhere else it is a guest.
            "host": {"mode": if rental { "dedicated" } else { "shared" }},
            "image_kernels": (!gpus.is_empty()).then(|| image_kernels(&layout.root, &paths.sdk())).flatten(),
            "toolchain": crate::machine::toolchain::dir(&engine),
        });
        // The operator's GPU settings, when the machine root has them, over these defaults.
        if let Ok(file) = std::fs::File::open(layout.state.join("gpu-config.json")) {
            let settings: serde_json::Map<String, serde_json::Value> =
                serde_json::from_reader(file).map_err(io::Error::other)?;
            config.as_object_mut().expect("an object").extend(settings);
        }
        let config: crate::gpu_service::GpuConfig =
            serde_json::from_value(config).map_err(io::Error::other)?;
        let store = store.clone();
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
    let api = crate::machine::net::kept_listener(&layout.state.join("api-port"), grant.listen_host, grant.worker_port)?;
    identity.webrtc = crate::machine::player::listen(&layout.state, grant.webrtc_port, !rental)?;
    identity.player = grant.webrtc_reach.clone();
    // The client this binary embeds; the installer helper over it is made at the first install.
    let uv = crate::machine::client::uv(&layout.root);
    let wheel = crate::machine::client::wheel(&engine)?;
    let mut sdk = crate::published::PackageSdk::selected(&paths.sdk(), &layout.root, uv);
    // A local machine installs releases through its owner's uv cache, as its local packages do;
    // a cache of its own from before is no longer read (environments keep their linked files).
    if !rental {
        sdk.owner_uv = crate::published::OwnerUv::of(cozy_proto::config::environment());
        if sdk.owner_uv.is_some() {
            let _ = std::fs::remove_dir_all(engine.join("published/uv-cache"));
        }
    }
    start_api(
        &engine,
        &generations,
        identity,
        api,
        &store,
        &machine,
        &service,
        Some(crate::machine::client::helper_python(&engine)),
        Some(wheel),
        "3.12".into(),
        sdk,
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
    drop(service);
    // The machine runs on its own threads until it is stopped.
    loop {
        std::thread::park();
    }
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
    store: &Arc<Store>,
    machine: &Arc<Machine>,
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
    let (pair, uv, seed_cache, runtime_build): (Vec<PathBuf>, _, _, _) = (
        sdk.requirements.iter().map(PathBuf::from).collect(),
        sdk.uv.clone(),
        sdk.seed_cache.clone(),
        sdk.runtime_build.clone(),
    );
    let store = store.clone();
    let mut backend = NativeBackend::new(service.clone(), identity.authority.clone(), store.clone());
    backend.logs = logs;
    let publisher =
        crate::published::Publisher::new(&root.join("published"), sdk, machine.clone())?;
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
            runtime_build,
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
    crate::calls::Calls::configure(service, machine.clone(), backend.runs.as_ref())?;
    if let Some(runs) = &backend.runs {
        crate::api::machine_runs::resume(runs)?;
    }
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
