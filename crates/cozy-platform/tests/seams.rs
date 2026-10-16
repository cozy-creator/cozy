//! The seams on real processes, files and services. A test's child processes are this binary run
//! as `seams child <role> …`. `--include-ignored` adds the test that installs a real user service:
//! hosted runners only, never a developer's own computer.
use cozy_platform::{
    durable, gpu,
    lock::{Kind, LockFile},
    paths::Home,
    process::{self, Birth, Exact, Lifetime, State},
    service::{ServiceSpec, Services},
};
use std::{
    env, fs,
    io::{BufRead, BufReader, Read, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::Duration,
};

type Test = (&'static str, fn(), bool);

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.first().is_some_and(|arg| arg == "child") {
        return child(&args[1..]);
    }
    let installs = args
        .iter()
        .any(|arg| arg == "--include-ignored" || arg == "--ignored");
    let filters: Vec<&String> = args.iter().filter(|arg| !arg.starts_with('-')).collect();
    let tests: &[Test] = &[
        (
            "a_held_lock_is_refused_until_its_holder_dies",
            a_held_lock_is_refused_until_its_holder_dies,
            false,
        ),
        (
            "shared_holders_share_and_exclude_a_writer",
            shared_holders_share_and_exclude_a_writer,
            false,
        ),
        (
            "a_replaced_file_is_whole_under_a_reader",
            a_replaced_file_is_whole_under_a_reader,
            false,
        ),
        ("a_tree_dies_whole", a_tree_dies_whole, false),
        (
            "an_owned_tree_ends_with_its_handle",
            an_owned_tree_ends_with_its_handle,
            false,
        ),
        (
            "an_exact_handle_never_names_another_birth",
            an_exact_handle_never_names_another_birth,
            false,
        ),
        (
            "a_stopped_process_reads_stopped",
            a_stopped_process_reads_stopped,
            false,
        ),
        (
            "a_detached_process_outlives_its_spawner",
            a_detached_process_outlives_its_spawner,
            false,
        ),
        (
            "the_gpu_probe_answers_without_a_driver",
            the_gpu_probe_answers_without_a_driver,
            false,
        ),
        (
            "a_user_service_starts_and_stops",
            a_user_service_starts_and_stops,
            true,
        ),
    ];
    let mut failed = vec![];
    for &(name, test, installing) in tests {
        if (installing && !installs)
            || !(filters.is_empty() || filters.iter().any(|f| name.contains(f.as_str())))
        {
            continue;
        }
        println!("test {name} ...");
        match std::panic::catch_unwind(test) {
            Ok(()) => println!("test {name} ... ok"),
            Err(_) => failed.push(name),
        }
    }
    if !failed.is_empty() {
        eprintln!("failed: {failed:?}");
        std::process::exit(1);
    }
}

/// The roles a test's child processes play.
fn child(args: &[String]) {
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    match args[..] {
        ["sleep"] => ready_and_wait(),
        ["exit"] => {}
        ["hold", kind, path, record] => {
            let kind = if kind == "shared" {
                Kind::Shared
            } else {
                Kind::Exclusive
            };
            let _held = {
                let held = LockFile::new(path).lock(kind).unwrap();
                held.publish(record.as_bytes()).unwrap();
                held
            };
            ready_and_wait();
        }
        ["tree", file] => {
            // Never waited for: it dies with its tree.
            #[allow(clippy::zombie_processes)]
            let mut grandchild = child_command(&["sleep"]).spawn().unwrap();
            ready(grandchild.stdout.take().unwrap());
            durable::replace(
                Path::new(file),
                grandchild.id().to_string().as_bytes(),
                0o600,
            )
            .unwrap();
            ready_and_wait();
        }
        ["detach", file] => {
            let mut command = child_command(&["sleep"]);
            command.stdout(Stdio::null());
            let tree = process::spawn(command, Lifetime::Detached).unwrap();
            let birth = serde_json::to_vec(&Birth::of(tree.pid()).unwrap()).unwrap();
            durable::replace(Path::new(file), &birth, 0o600).unwrap();
        }
        ["service", file] => {
            let birth = serde_json::to_vec(&Birth::own().unwrap()).unwrap();
            durable::replace(Path::new(file), &birth, 0o600).unwrap();
            loop {
                thread::park();
            }
        }
        _ => panic!("unknown child role {args:?}"),
    }
}

fn ready_and_wait() -> ! {
    println!("ready");
    std::io::stdout().flush().unwrap();
    loop {
        thread::park();
    }
}

fn child_command(args: &[&str]) -> Command {
    let mut command = Command::new(env::current_exe().unwrap());
    command
        .arg("child")
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped());
    command
}

fn ready(output: impl Read) {
    let mut line = String::new();
    BufReader::new(output).read_line(&mut line).unwrap();
    assert_eq!(line.trim(), "ready");
}

fn scratch(name: &str) -> PathBuf {
    let dir = env::temp_dir().join(format!("cozy-platform-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn a_held_lock_is_refused_until_its_holder_dies() {
    let path = scratch("lock").join("held.lock");
    let lock = LockFile::new(&path);
    let mut holder = child_command(&["hold", "exclusive", path.to_str().unwrap(), "holder 1"])
        .spawn()
        .unwrap();
    ready(holder.stdout.take().unwrap());
    assert!(lock.try_lock(Kind::Exclusive).unwrap().is_none());
    assert!(lock.try_lock(Kind::Shared).unwrap().is_none());
    assert_eq!(lock.holder().unwrap().as_deref(), Some(&b"holder 1"[..]));
    // A waiter is woken by the kernel when the holder dies: a kill releases its claim.
    let (taken, waited) = mpsc::channel();
    let waiter = {
        let path = path.clone();
        thread::spawn(move || {
            let held = LockFile::new(path).lock(Kind::Exclusive).unwrap();
            taken.send(()).unwrap();
            held
        })
    };
    holder.kill().unwrap();
    holder.wait().unwrap();
    waited.recv().unwrap();
    let held = waiter.join().unwrap();
    held.publish(b"holder 2").unwrap();
    assert!(lock.try_lock(Kind::Exclusive).unwrap().is_none());
    assert_eq!(lock.holder().unwrap().as_deref(), Some(&b"holder 2"[..]));
    drop(held);
    assert_eq!(
        lock.holder().unwrap(),
        None,
        "a released holder's record is erased"
    );
    assert!(lock.try_lock(Kind::Exclusive).unwrap().is_some());
}

fn shared_holders_share_and_exclude_a_writer() {
    let path = scratch("shared").join("shared.lock");
    let lock = LockFile::new(&path);
    let mut holder = child_command(&["hold", "shared", path.to_str().unwrap(), "reader"])
        .spawn()
        .unwrap();
    ready(holder.stdout.take().unwrap());
    assert!(lock.try_lock(Kind::Shared).unwrap().is_some());
    assert!(lock.try_lock(Kind::Exclusive).unwrap().is_none());
    holder.kill().unwrap();
    holder.wait().unwrap();
    assert!(lock.try_lock(Kind::Exclusive).unwrap().is_some());
}

fn a_replaced_file_is_whole_under_a_reader() {
    let dir = scratch("replace");
    let path = dir.join("record.json");
    let (small, large) = (vec![b'a'; 1000], vec![b'b'; 70_000]);
    durable::replace(&path, &small, 0o600).unwrap();
    let done = Arc::new(AtomicBool::new(false));
    let reader = {
        let (path, done, small, large) = (path.clone(), done.clone(), small.clone(), large.clone());
        thread::spawn(move || {
            let mut reads = 0u64;
            while !done.load(Ordering::Relaxed) {
                let bytes = fs::read(&path).unwrap();
                assert!(
                    bytes == small || bytes == large,
                    "a torn read of {} bytes",
                    bytes.len()
                );
                reads += 1;
            }
            reads
        })
    };
    for round in 0..300 {
        durable::replace(&path, if round % 2 == 0 { &large } else { &small }, 0o600).unwrap();
    }
    done.store(true, Ordering::Relaxed);
    assert!(reader.join().unwrap() > 0);
    assert_eq!(fs::read(&path).unwrap(), small);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = |path: &Path| fs::metadata(path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode(&path), 0o600);
        durable::replace(&path, &large, 0o644).unwrap();
        assert_eq!(mode(&path), 0o644);
    }
    assert_eq!(
        fs::read_dir(&dir).unwrap().count(),
        1,
        "no temporary file is left behind"
    );
    durable::sync_dir(&dir).unwrap();
}

fn a_tree_dies_whole() {
    let file = scratch("tree").join("grandchild");
    let mut tree = process::spawn(
        child_command(&["tree", file.to_str().unwrap()]),
        Lifetime::Owned,
    )
    .unwrap();
    ready(tree.stdout().unwrap());
    let pid: u32 = fs::read_to_string(&file).unwrap().parse().unwrap();
    let grandchild = Exact::open(&Birth::of(pid).unwrap())
        .unwrap()
        .expect("the grandchild runs");
    assert_eq!(grandchild.birth().state().unwrap(), State::Live);
    assert!(!grandchild.ended());
    tree.kill().unwrap();
    assert!(!tree.wait().unwrap().success());
    // Its exit, seen through the handle that names it and nothing else.
    grandchild.wait().unwrap();
    assert!(grandchild.ended());
}

fn an_owned_tree_ends_with_its_handle() {
    let mut tree = process::spawn(child_command(&["sleep"]), Lifetime::Owned).unwrap();
    ready(tree.stdout().unwrap());
    let leader = Exact::open(&Birth::of(tree.pid()).unwrap())
        .unwrap()
        .unwrap();
    drop(tree);
    leader.wait().unwrap();
}

fn an_exact_handle_never_names_another_birth() {
    let own = Birth::own().unwrap();
    let alive = Exact::open(&own).unwrap().expect("this process");
    assert!(!alive.wait_for(Duration::from_millis(10)).unwrap());
    let reborn = Birth {
        start_ticks: own.start_ticks + 1,
        ..own
    };
    assert!(Exact::open(&reborn).unwrap().is_none());
    assert_eq!(reborn.state().unwrap(), State::Gone);
    let mut exited = child_command(&["exit"]).spawn().unwrap();
    // Unreaped, its PID cannot be reused; reaped and released, its birth names nothing.
    let birth = Birth::of(exited.id()).unwrap();
    exited.wait().unwrap();
    drop(exited);
    assert!(Exact::open(&birth).unwrap().is_none());
    assert_eq!(birth.state().unwrap(), State::Gone);
}

fn a_stopped_process_reads_stopped() {
    if cfg!(windows) {
        return println!("Windows has no stop signal");
    }
    let mut child = child_command(&["sleep"]).spawn().unwrap();
    ready(child.stdout.take().unwrap());
    let birth = Birth::of(child.id()).unwrap();
    assert_eq!(birth.state().unwrap(), State::Live);
    let signal = |name: &str| {
        let status = Command::new("kill")
            .args([name, &child.id().to_string()])
            .status()
            .unwrap();
        assert!(status.success());
    };
    signal("-STOP");
    // The kernel applies a stop at once; this samples until it shows.
    while birth.state().unwrap() != State::Stopped {
        thread::sleep(Duration::from_millis(5));
    }
    signal("-CONT");
    child.kill().unwrap();
    child.wait().unwrap();
}

fn a_detached_process_outlives_its_spawner() {
    let file = scratch("detach").join("birth.json");
    let mut spawner = child_command(&["detach", file.to_str().unwrap()]);
    assert!(spawner.stdout(Stdio::null()).status().unwrap().success());
    let birth: Birth = serde_json::from_slice(&fs::read(&file).unwrap()).unwrap();
    let detached = Exact::open(&birth)
        .unwrap()
        .expect("the detached child outlived its spawner");
    assert_eq!(birth.state().unwrap(), State::Live);
    detached.kill().unwrap();
    detached.wait().unwrap();
}

fn the_gpu_probe_answers_without_a_driver() {
    assert!(gpu::probe(Some("")).unwrap().is_empty());
    let gpus = gpu::probe(None).unwrap();
    let driver = if cfg!(target_os = "linux") {
        Path::new("/dev/nvidiactl").exists()
    } else {
        cfg!(windows) && Path::new(r"C:\Windows\System32\nvml.dll").exists()
    };
    if !driver {
        assert!(gpus.is_empty(), "no NVIDIA driver, yet {gpus:?}");
    }
    for gpu in &gpus {
        assert!(
            gpu.uuid.starts_with("GPU-") && !gpu.name.is_empty() && !gpu.driver.is_empty(),
            "{gpu:?}"
        );
        assert!(
            gpu.memory_bytes > 0 && !gpu.pci_bus_id.is_empty(),
            "{gpu:?}"
        );
        assert_eq!(gpu::probe(Some(&gpu.uuid)).unwrap(), vec![gpu.clone()]);
    }
}

fn a_user_service_starts_and_stops() {
    let dir = scratch("service");
    // Every word of the definition passes through a space and a percent sign.
    let odd = dir.join("a dir 100%");
    fs::create_dir_all(&odd).unwrap();
    let home = Home::at(dir.join("home"));
    home.open().unwrap();
    #[allow(deprecated)]
    let services = Services::new(&home, &env::home_dir().unwrap());
    if !services.available() {
        println!("no user service manager in this session: nothing installed");
        return;
    }
    let file = odd.join("birth.json");
    let spec = ServiceSpec {
        name: format!("platform-test-{}", std::process::id()),
        program: env::current_exe().unwrap(),
        args: vec!["child".into(), "service".into(), file.clone().into()],
        log: Some(odd.join("service.log")),
    };
    services.start(&spec).unwrap();
    // The service writes its own birth once it runs; this samples for it.
    let birth: Birth = loop {
        if let Ok(bytes) = fs::read(&file) {
            break serde_json::from_slice(&bytes).unwrap();
        }
        thread::sleep(Duration::from_millis(50));
    };
    let running = Exact::open(&birth).unwrap().expect("the service runs");
    services.start(&spec).unwrap();
    assert!(
        !running.ended(),
        "starting a running service leaves it as it is"
    );
    services.stop(&spec.name).unwrap();
    running.wait().unwrap();
    services.uninstall(&spec.name).unwrap();
}
