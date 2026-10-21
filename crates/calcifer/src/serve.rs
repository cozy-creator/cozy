//! `cozy calcifer`, the role entry. One Calcifer per home, held by `calcifer.lock`. It serves the
//! loopback API as soon as its store is open and publishes its `Holder` record, the readiness
//! signal; only then does it take up what an earlier Calcifer left (runs, their outputs), while its
//! machines reattach in their own tasks. It runs until `POST /v1/down`, an update or a crash: the
//! login service keeps it up. Machines keep running what they accepted either way. Each startup
//! step is logged with its time, so a slow start names itself.
use std::{
    net::{Ipv4Addr, SocketAddr},
    process::ExitCode,
    time::Instant,
};

use platform::{
    durable,
    lock::{Kind as LockKind, LockFile},
    paths::Home,
    process,
    service::Services,
};
use protocol::{
    calcifer::{Holder, REFUSED},
    config::{self, Flags, Loaded},
    error::{Envelope, Error, Kind},
};
use tokio::net::TcpListener;
use tracing::{error, info, warn};

use crate::{
    Calcifer,
    api::{self, Api},
    machines::Machines,
};

/// Why `serve` returned.
#[derive(Debug, PartialEq, Eq)]
pub enum Stopped {
    /// Another Calcifer holds this home.
    AnotherHolds,
    /// `POST /v1/down`.
    Down,
}

/// `cozy calcifer`: exits 0 unless Calcifer could not start, which `refused.json` then says.
/// Its log is its stdout (the service's `calcifer.log`).
pub fn main() -> ExitCode {
    tracing_subscriber::fmt().init();
    // Before the runtime's workers start: Linux names the calling thread.
    if let Err(e) = process::set_name("calcifer")
        && !e.is_unsupported()
    {
        warn!("{e}");
    }
    let served = match config::load(Flags::default()) {
        Ok(loaded) => match tokio::runtime::Runtime::new() {
            Ok(runtime) => runtime.block_on(serve(&loaded)),
            Err(e) => Err(refuse(
                &loaded.home,
                Error::new(Kind::Internal, "calcifer_runtime", e.to_string()),
            )),
        },
        Err(error) => match std::env::home_dir() {
            Some(user_home) => Err(refuse(&Home::under(&user_home), error)),
            None => Err(error),
        },
    };
    match served {
        Ok(_) => ExitCode::SUCCESS,
        Err(_) => ExitCode::FAILURE,
    }
}

/// Logs why Calcifer cannot start and leaves it in `<home>/calcifer/refused.json` for the CLI.
fn refuse(home: &Home, error: Error) -> Error {
    error!("Calcifer cannot start: {error}");
    let envelope = serde_json::to_vec(&Envelope {
        error: error.clone(),
    })
    .expect("serializes");
    let written = home
        .open()
        .and_then(|()| durable::private_dir(&home.calcifer()))
        .and_then(|()| durable::replace(&home.calcifer().join(REFUSED), &envelope, 0o600));
    if let Err(e) = written {
        error!("{e}");
    }
    error
}

/// Serves `loaded.home` on `127.0.0.1:<calcifer.port>` (0: any free port) until `POST /v1/down`.
pub async fn serve(loaded: &Loaded) -> Result<Stopped, Error> {
    let (home, port) = (&loaded.home, loaded.config.calcifer.port);
    let (began, mut steps) = (Instant::now(), Steps(Instant::now()));
    let failed = |error| refuse(home, error);
    let lock = LockFile::new(home.calcifer_lock());
    let held = home
        .open()
        .and_then(|()| durable::private_dir(&home.calcifer()))
        .and_then(|()| lock.try_lock(LockKind::Exclusive))
        .map_err(|e| failed(Error::new(Kind::Internal, "calcifer_home", e.to_string())))?;
    let Some(held) = held else {
        info!("another Calcifer holds this home");
        return Ok(Stopped::AnotherHolds);
    };
    // Only the holder announces: its birth is what `Services::stop` must end.
    Services::new(home, &loaded.user_home)
        .announce("calcifer")
        .map_err(|e| {
            failed(Error::new(
                Kind::Internal,
                "calcifer_service",
                e.to_string(),
            ))
        })?;
    for warning in &loaded.warnings {
        warn!("{warning}");
    }
    steps.done("home");

    let calcifer = Calcifer::open(&home.calcifer(), &home.outputs())
        .map_err(|e| failed(Error::new(Kind::Internal, "calcifer_store", e.to_string())))?;
    steps.done("store");
    let listener = platform::net::listen((Ipv4Addr::LOCALHOST, port).into()).and_then(|listener| {
        listener.set_nonblocking(true)?;
        TcpListener::from_std(listener)
    });
    let listener = match listener {
        Ok(listener) => listener,
        Err(e) => {
            calcifer.close();
            return Err(failed(
                Error::new(
                    Kind::Unavailable,
                    "calcifer_port",
                    format!("Calcifer cannot listen on 127.0.0.1:{port}: {e}"),
                )
                .remedy("free the port, or set calcifer.port in config.toml (0 takes any)"),
            ));
        }
    };
    let address = listener
        .local_addr()
        .map_err(|e| failed(Error::new(Kind::Internal, "calcifer_port", e.to_string())))?;
    steps.done("listen");
    let mut token = [0; 32];
    getrandom::fill(&mut token).expect("the OS provides randomness");
    let token = hex::encode(token);
    let machines = Machines::start(calcifer.clone(), home.clone(), &loaded.config);
    steps.done("machines");
    let api = Api::new(calcifer.clone(), machines.clone(), address.port(), &token);
    let app = api::router(api.clone()).into_make_service_with_connect_info::<SocketAddr>();
    let server = tokio::spawn(
        axum::serve(listener, app)
            .with_graceful_shutdown(api.stopped())
            .into_future(),
    );

    let holder = Holder {
        address,
        token,
        pid: std::process::id(),
        version: env!("CARGO_PKG_VERSION").into(),
        started_at_ms: crate::store::now_ms(),
    };
    let mut record = serde_json::to_vec(&holder).expect("serializes");
    record.push(b'\n');
    if let Err(e) = held.publish(&record) {
        api.stop();
        let _ = server.await;
        machines.close();
        calcifer.close();
        return Err(failed(Error::new(
            Kind::Internal,
            "calcifer_home",
            e.to_string(),
        )));
    }
    let _ = std::fs::remove_file(home.calcifer().join(REFUSED));
    steps.done("record");
    info!(%address, ms = began.elapsed().as_millis() as u64, "Calcifer serves");
    match calcifer.resume() {
        Ok(runs) => steps.done(&format!("resume ({runs} runs)")),
        Err(e) => warn!("resuming what an earlier Calcifer left: {e}"),
    }

    api.stopped().await;
    if let Ok(Err(e)) = server.await {
        warn!("the API server ended: {e}");
    }
    machines.close();
    calcifer.close();
    drop(held);
    info!("Calcifer stopped");
    Ok(Stopped::Down)
}

/// Logs each startup step with the time it took.
struct Steps(Instant);

impl Steps {
    fn done(&mut self, step: &str) {
        info!(step, ms = self.0.elapsed().as_millis() as u64, "started");
        self.0 = Instant::now();
    }
}
