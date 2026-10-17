//! `cozy calcifer`, the role entry. One Calcifer per home, held by `calcifer.lock`: it opens its
//! store (resuming every run it owes), serves the loopback API, and only then publishes its
//! `Holder` record, which is the readiness signal. It exits once nothing is owed, no client is
//! connected and no client is starting it (`wanted.lock`), or on `POST /v1/down`; machines keep
//! running what they accepted either way.
use std::{
    io,
    net::{Ipv4Addr, SocketAddr},
    process::ExitCode,
    sync::Arc,
};

use axum::{
    extract::connect_info::Connected,
    serve::{IncomingStream, Listener},
};
use cozy_platform::{
    durable,
    lock::{Kind as LockKind, LockFile},
    paths::Home,
    process,
    service::Services,
};
use cozy_proto::{
    calcifer::{Holder, REFUSED, WANTED},
    config::{self, Flags, Loaded},
    error::{Envelope, Error, Kind},
};
use tokio::{
    net::{TcpListener, TcpStream},
    sync::watch,
};
use tracing::{error, info, warn};

use crate::{
    Calcifer,
    api::{self, Api},
    machines::Machines,
    store::StoreError,
};

/// Why `serve` returned.
#[derive(Debug, PartialEq, Eq)]
pub enum Stopped {
    /// Another Calcifer holds this home.
    AnotherHolds,
    /// `POST /v1/down`.
    Down,
    /// Nothing was owed and no client was connected.
    Idle,
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

/// Serves `loaded.home` on `127.0.0.1:<calcifer.port>` (0: any free port) until Calcifer stops.
pub async fn serve(loaded: &Loaded) -> Result<Stopped, Error> {
    let (home, port) = (&loaded.home, loaded.config.calcifer.port);
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

    let calcifer = Calcifer::open(&home.calcifer(), &home.outputs()).map_err(|e| {
        failed(match e {
            crate::Error::Store(e @ StoreError::Conversion { .. }) => {
                Error::new(Kind::Internal, "calcifer_store_conversion", e.to_string())
                    .remedy("nothing was changed: the cozy you had before still reads this store")
                    .next("cozy logs calcifer")
            }
            e => Error::new(Kind::Internal, "calcifer_store", e.to_string()),
        })
    })?;
    let listener = match TcpListener::bind((Ipv4Addr::LOCALHOST, port)).await {
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
    let mut token = [0; 32];
    getrandom::fill(&mut token).expect("the OS provides randomness");
    let token = hex::encode(token);
    let machines = Machines::start(calcifer.clone(), home.clone(), &loaded.config);
    let api = Api::new(calcifer.clone(), machines.clone(), address.port(), &token);
    // Kept here too, so `open` always has a sender while Calcifer serves.
    let (count, mut open) = watch::channel(0);
    let count = Arc::new(count);
    let listener = Counted {
        listener,
        open: count.clone(),
    };
    let app = api::router(api.clone()).into_make_service_with_connect_info::<Peer>();
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
    info!(%address, "Calcifer serves");

    let mut owed = calcifer.owed_changed();
    let stopped = api.stopped();
    tokio::pin!(stopped);
    // Checked at once (a login start owing nothing stops here) and on every change after.
    let wanted = LockFile::new(home.calcifer().join(WANTED));
    let why = loop {
        if api.stop_unless_owed(&open, &wanted).await {
            break Stopped::Idle;
        }
        tokio::select! {
            () = &mut stopped => break Stopped::Down,
            _ = owed.changed() => {}
            _ = open.changed() => {}
        }
    };
    api.stop();
    if let Ok(Err(e)) = server.await {
        warn!("the API server ended: {e}");
    }
    machines.close();
    calcifer.close();
    drop((held, count));
    info!(?why, "Calcifer stopped");
    Ok(why)
}

/// A client's connection: its address, and its share in the count of open connections.
#[derive(Clone, Debug)]
pub struct Peer {
    pub addr: SocketAddr,
    _open: Option<Arc<Open>>,
}

/// Counted in for as long as its connection, or a request on it, lives.
#[derive(Debug)]
struct Open(Arc<watch::Sender<usize>>);

impl Drop for Open {
    fn drop(&mut self) {
        self.0.send_modify(|n| *n -= 1);
    }
}

/// The loopback listener, counting the connections open on it.
struct Counted {
    listener: TcpListener,
    open: Arc<watch::Sender<usize>>,
}

impl Listener for Counted {
    type Io = TcpStream;
    type Addr = Peer;

    async fn accept(&mut self) -> (TcpStream, Peer) {
        let (io, addr) = Listener::accept(&mut self.listener).await;
        self.open.send_modify(|n| *n += 1);
        let open = Open(self.open.clone());
        (
            io,
            Peer {
                addr,
                _open: Some(Arc::new(open)),
            },
        )
    }

    fn local_addr(&self) -> io::Result<Peer> {
        Ok(Peer {
            addr: self.listener.local_addr()?,
            _open: None,
        })
    }
}

impl Connected<IncomingStream<'_, Counted>> for Peer {
    fn connect_info(stream: IncomingStream<'_, Counted>) -> Self {
        stream.remote_addr().clone()
    }
}
