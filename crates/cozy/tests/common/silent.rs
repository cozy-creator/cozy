//! A Hub that takes connections and never answers: a run on a machine that reads it stays
//! preparing, accepted by its machine, for as long as the test needs.
use std::net::TcpListener;

use super::Home;

const CARDS: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../cozy-cli/tests/fixtures/cards"
);

pub struct SilentHub {
    _listener: TcpListener,
    pub origin: String,
}

impl SilentHub {
    /// Makes it this home's Hub, with sdxl's card kept so the CLI reads no Hub either.
    pub fn for_home(home: &Home) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let origin = format!("http://{address}");
        home.json(["hub", "add", "silent", &origin, "--json"]);
        home.json(["hub", "use", "silent", "--json"]);
        let dir = home
            .root()
            .join(format!("packages/127.0.0.1_{}", address.port()))
            .join("fidika/sdxl");
        std::fs::create_dir_all(&dir).unwrap();
        let card = std::fs::read_to_string(format!("{CARDS}/sdxl-2.10.0.json")).unwrap();
        std::fs::write(dir.join("2.10.0.json"), card).unwrap();
        Self {
            _listener: listener,
            origin,
        }
    }
}
