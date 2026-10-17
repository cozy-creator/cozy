//! `cozy` on a pseudo-terminal, as a person runs it: keys typed in, the screen as a terminal
//! shows it.
use std::{
    io::{Read as _, Write},
    process::Command,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use portable_pty::{Child, CommandBuilder, MasterPty, PtySize, native_pty_system};

const ROWS: u16 = 40;
const COLUMNS: u16 = 160;

pub struct Pty {
    screen: Arc<Mutex<vt100::Parser>>,
    keys: Box<dyn Write + Send>,
    child: Box<dyn Child + Send + Sync>,
    _master: Box<dyn MasterPty + Send>,
}

impl Pty {
    /// Runs `command` (a `Home::command()`, its environment kept) on a terminal of its own.
    pub fn spawn(command: &Command) -> Self {
        let size = PtySize {
            rows: ROWS,
            cols: COLUMNS,
            pixel_width: 0,
            pixel_height: 0,
        };
        let pair = native_pty_system().openpty(size).unwrap();
        let mut built = CommandBuilder::new(command.get_program());
        built.args(command.get_args());
        for (key, value) in command.get_envs() {
            match value {
                Some(value) => built.env(key, value),
                None => built.env_remove(key),
            }
        }
        built.env("TERM", "xterm-256color");
        if let Some(dir) = command.get_current_dir() {
            built.cwd(dir);
        }
        let child = pair.slave.spawn_command(built).unwrap();
        drop(pair.slave);
        let mut output = pair.master.try_clone_reader().unwrap();
        let keys = pair.master.take_writer().unwrap();
        let screen = Arc::new(Mutex::new(vt100::Parser::new(ROWS, COLUMNS, 0)));
        let drawn = screen.clone();
        std::thread::spawn(move || {
            let mut buf = [0; 8192];
            while let Ok(n @ 1..) = output.read(&mut buf) {
                drawn.lock().unwrap().process(&buf[..n]);
            }
        });
        Self {
            screen,
            keys,
            child,
            _master: pair.master,
        }
    }

    /// The screen's text now.
    pub fn screen(&self) -> String {
        self.screen.lock().unwrap().screen().contents()
    }

    /// The screen once it shows what `want` looks for.
    pub fn until(&self, what: &str, want: impl Fn(&str) -> bool) -> String {
        // A hang bound for a screen that never shows it; the product waits without one.
        let until = Instant::now() + Duration::from_secs(120);
        loop {
            let screen = self.screen();
            if want(&screen) {
                return screen;
            }
            assert!(
                Instant::now() < until,
                "the screen never showed {what}:\n{screen}"
            );
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    /// Types `keys`: `\r` is Enter, `\x1b` Esc.
    pub fn key(&mut self, keys: &str) {
        self.keys.write_all(keys.as_bytes()).unwrap();
        self.keys.flush().unwrap();
    }

    /// Its exit, once it quit.
    pub fn wait(&mut self) -> portable_pty::ExitStatus {
        self.child.wait().unwrap()
    }
}

impl Drop for Pty {
    fn drop(&mut self) {
        let _ = self.child.kill();
    }
}
