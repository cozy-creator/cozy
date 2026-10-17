//! The seams on real processes, files and services. A test's child processes are this binary run
//! as `seams child <role> …`. `--include-ignored` adds the test that installs a real user service:
//! hosted runners only, never a developer's own computer.
use cozy_platform::{
    durable, gpu,
    lock::{Kind, LockFile},
    paths::Home,
    process::{self, Birth, Exact, State},
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
        (
            "a_hard_link_counts_and_changes_its_file",
            a_hard_link_counts_and_changes_its_file,
            false,
        ),
        ("a_tree_dies_whole", a_tree_dies_whole, false),
        #[cfg(unix)]
        (
            "a_child_watching_its_parent_dies_with_it",
            a_child_watching_its_parent_dies_with_it,
            false,
        ),
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
            "a_tree_whose_leader_exited_is_killed_quietly",
            a_tree_whose_leader_exited_is_killed_quietly,
            false,
        ),
        (
            "a_process_takes_its_role_name",
            a_process_takes_its_role_name,
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
        ["named", name] => {
            process::set_name(name).unwrap();
            ready_and_wait();
        }
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
        #[cfg(unix)]
        ["watching-parent", file] => {
            // Never waited for: it dies with this process.
            #[allow(clippy::zombie_processes)]
            let mut watcher = child_command(&["watch-parent"]).spawn().unwrap();
            ready(watcher.stdout.take().unwrap());
            let birth = serde_json::to_vec(&Birth::of(watcher.id()).unwrap()).unwrap();
            durable::replace(Path::new(file), &birth, 0o600).unwrap();
            ready_and_wait();
        }
        #[cfg(unix)]
        ["watch-parent"] => {
            assert!(process::die_with_parent(std::os::unix::process::parent_id()).unwrap());
            ready_and_wait();
        }
        ["detach", file] => {
            let args = ["child".into(), "sleep".into()];
            let detached = process::detach(&env::current_exe().unwrap(), &args, None).unwrap();
            let birth = serde_json::to_vec(&Birth::of(detached.pid()).unwrap()).unwrap();
            durable::replace(Path::new(file), &birth, 0o600).unwrap();
        }
        ["service", file, home, name] => {
            Services::new(&Home::at(home), Path::new(home))
                .announce(name)
                .unwrap();
            let report = serde_json::json!({
                "birth": Birth::own().unwrap(),
                "console_visible": console_visible(),
            });
            durable::replace(Path::new(file), report.to_string().as_bytes(), 0o600).unwrap();
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
    assert_eq!(
        lock.holder().unwrap(),
        None,
        "the next exclusive taker erases a killed holder's record"
    );
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

fn a_hard_link_counts_and_changes_its_file() {
    let dir = scratch("links");
    let file = dir.join("file");
    fs::write(&file, b"bytes").unwrap();
    assert_eq!(durable::links(&file).unwrap(), 1);
    let before = durable::changed(&file).unwrap();
    let now = std::time::SystemTime::now();
    // The system's own clock: neither ahead of it nor from another epoch.
    assert!(before <= now + Duration::from_secs(5), "{before:?} is after {now:?}");
    assert!(now.duration_since(before).is_ok_and(|age| age < Duration::from_secs(3600)), "{before:?}");
    let link = dir.join("link");
    fs::hard_link(&file, &link).unwrap();
    assert_eq!(durable::links(&file).unwrap(), 2);
    assert_eq!(durable::links(&link).unwrap(), 2);
    assert!(durable::changed(&file).unwrap() >= before, "a new link is a status change");
    #[cfg(unix)]
    {
        let symlink = dir.join("symlink");
        std::os::unix::fs::symlink(&file, &symlink).unwrap();
        assert_eq!(durable::links(&symlink).unwrap(), 1, "a symlink is not followed");
    }
    fs::remove_file(&link).unwrap();
    assert_eq!(durable::links(&file).unwrap(), 1);
    let _ = fs::remove_dir_all(&dir);
}

fn a_tree_dies_whole() {
    let file = scratch("tree").join("grandchild");
    let mut tree = process::spawn(child_command(&["tree", file.to_str().unwrap()])).unwrap();
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
    let mut tree = process::spawn(child_command(&["sleep"])).unwrap();
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
    let mut ended = child_command(&["sleep"]).spawn().unwrap();
    ready(ended.stdout.take().unwrap());
    let birth = Birth::of(ended.id()).unwrap();
    ended.kill().unwrap();
    ended.wait().unwrap();
    // Reaped and released, its birth names nothing.
    drop(ended);
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

/// Its spawner's stdout is a pipe read to its end, as `out=$(cozy up)` reads: the end comes when
/// the spawner exits, though the detached child lives on.
fn a_detached_process_outlives_its_spawner() {
    let file = scratch("detach").join("birth.json");
    let mut spawner = child_command(&["detach", file.to_str().unwrap()])
        .spawn()
        .unwrap();
    let mut out = spawner.stdout.take().unwrap();
    let (sent, read) = mpsc::channel();
    thread::spawn(move || sent.send(out.read_to_end(&mut vec![]).is_ok()));
    assert!(spawner.wait().unwrap().success());
    let birth: Birth = serde_json::from_slice(&fs::read(&file).unwrap()).unwrap();
    let detached = Exact::open(&birth)
        .unwrap()
        .expect("the detached child outlived its spawner");
    // Only a pipe end the detached child kept holds the read open now; it never lets go.
    let ended = read.recv_timeout(Duration::from_secs(30));
    assert_eq!(birth.state().unwrap(), State::Live);
    detached.kill().unwrap();
    detached.wait().unwrap();
    assert_eq!(
        ended,
        Ok(true),
        "the detached child kept its spawner's stdout open"
    );
}

/// macOS answers EPERM, not ESRCH, when it signals an exited, unreaped process or its group.
fn a_tree_whose_leader_exited_is_killed_quietly() {
    if cfg!(windows) {
        return println!("Windows ends a tree through its job");
    }
    let mut tree = process::spawn(child_command(&["sleep"])).unwrap();
    ready(tree.stdout().unwrap());
    let leader = Birth::of(tree.pid()).unwrap();
    let status = Command::new("kill")
        .args(["-KILL", &tree.pid().to_string()])
        .status()
        .unwrap();
    assert!(status.success());
    // Unreaped, it stays a zombie; this samples until the kernel shows it exited.
    while leader.state().unwrap() != State::Gone {
        thread::sleep(Duration::from_millis(5));
    }
    tree.kill().unwrap();
    assert!(!tree.wait().unwrap().success());
}

fn a_process_takes_its_role_name() {
    if !cfg!(target_os = "linux") {
        let refused = process::set_name("seams").unwrap_err();
        assert!(refused.is_unsupported(), "{refused}");
        return;
    }
    let mut child = child_command(&["named", "calcifer"]).spawn().unwrap();
    ready(child.stdout.take().unwrap());
    let comm = fs::read_to_string(format!("/proc/{}/comm", child.id())).unwrap();
    assert_eq!(comm.trim(), "calcifer");
    child.kill().unwrap();
    child.wait().unwrap();
}

fn the_gpu_probe_answers_without_a_driver() {
    assert!(gpu::probe(Some("")).unwrap().is_empty());
    gpu::memory_classes(&mut []).unwrap();
    let mut gpus = gpu::probe(None).unwrap();
    gpu::memory_classes(&mut gpus).unwrap();
    let driver = if cfg!(target_os = "linux") {
        Path::new("/dev/nvidiactl").exists()
    } else {
        cfg!(windows) && Path::new(r"C:\Windows\System32\nvml.dll").exists()
    };
    if !driver {
        assert!(gpus.is_empty(), "no NVIDIA driver, yet {gpus:?}");
        assert!(
            gpu::Device::open("0").is_err(),
            "a device opened with no driver"
        );
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
        assert!(
            gpu.compute_capability.is_some() && gpu.memory_class.is_some(),
            "{gpu:?}"
        );
        let probed = gpu::Gpu {
            memory_class: None,
            ..gpu.clone()
        };
        assert_eq!(gpu::probe(Some(&gpu.uuid)).unwrap(), vec![probed]);
        let device = gpu::Device::open(&gpu.uuid).unwrap();
        assert!(device.key().starts_with(&gpu.uuid));
        assert_eq!(device.read().unwrap().total, gpu.memory_bytes);
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
    let file = odd.join("report.json");
    let name = format!("platform-test-{}", std::process::id());
    let spec = ServiceSpec {
        name: name.clone(),
        program: env::current_exe().unwrap(),
        args: vec![
            "child".into(),
            "service".into(),
            file.clone().into(),
            home.root().into(),
            name.into(),
        ],
        log: Some(odd.join("service.log")),
    };
    services.start(&spec).unwrap();
    let running = started(&file, None);
    services.start(&spec).unwrap();
    assert!(
        !running.ended(),
        "starting a running service leaves it as it is"
    );
    services.stop(&spec.name).unwrap();
    running.wait().unwrap();
    #[cfg(windows)]
    {
        // Logon runs the Run value in a new console with no `start`: the service leaves that console
        // in its announce, and `stop` finds it through the birth the announce recorded.
        use std::os::windows::process::CommandExt;
        use windows_sys::Win32::System::Threading::CREATE_NEW_CONSOLE;
        fs::remove_file(&file).unwrap();
        let value = run_value(&spec.name);
        let (program, rest) = match value.strip_prefix('"') {
            Some(quoted) => quoted.split_once("\" ").unwrap(),
            None => value.split_once(' ').unwrap(),
        };
        let mut logon = Command::new(program);
        logon.raw_arg(rest).creation_flags(CREATE_NEW_CONSOLE);
        let mut service = logon.spawn().unwrap();
        let running = started(&file, Some(&mut service));
        services.stop(&spec.name).unwrap();
        running.wait().unwrap();
        service.wait().unwrap();
    }
    services.uninstall(&spec.name).unwrap();
}

/// The service's report, once it runs: it has no visible console window, and it is alive.
fn started(file: &Path, mut spawned: Option<&mut std::process::Child>) -> Exact {
    let report: serde_json::Value = loop {
        if let Ok(bytes) = fs::read(file) {
            break serde_json::from_slice(&bytes).unwrap();
        }
        if let Some(spawned) = spawned.as_mut() {
            assert_eq!(
                spawned.try_wait().unwrap(),
                None,
                "the service ended before it reported"
            );
        }
        thread::sleep(Duration::from_millis(50));
    };
    assert_eq!(
        report["console_visible"], false,
        "the service opened a console window"
    );
    let birth: Birth = serde_json::from_value(report["birth"].clone()).unwrap();
    Exact::open(&birth).unwrap().expect("the service runs")
}

fn console_visible() -> bool {
    #[cfg(windows)]
    {
        use windows_sys::Win32::{
            System::Console::GetConsoleWindow, UI::WindowsAndMessaging::IsWindowVisible,
        };
        // SAFETY: plain queries about this process's own console.
        unsafe {
            let window = GetConsoleWindow();
            !window.is_null() && IsWindowVisible(window) != 0
        }
    }
    #[cfg(unix)]
    {
        false
    }
}

/// What logon runs for `name`, read back from HKCU's Run key.
#[cfg(windows)]
fn run_value(name: &str) -> String {
    use windows_sys::Win32::System::Registry::{HKEY_CURRENT_USER, RRF_RT_REG_SZ, RegGetValueW};
    let wide = |text: &str| {
        text.encode_utf16()
            .chain(std::iter::once(0))
            .collect::<Vec<u16>>()
    };
    let mut data = vec![0u16; 4096];
    let mut size = (data.len() * 2) as u32;
    // SAFETY: NUL-terminated names and a writable buffer of `size` bytes.
    let status = unsafe {
        RegGetValueW(
            HKEY_CURRENT_USER,
            wide(r"Software\Microsoft\Windows\CurrentVersion\Run").as_ptr(),
            wide(&format!("cozy-{name}")).as_ptr(),
            RRF_RT_REG_SZ,
            std::ptr::null_mut(),
            data.as_mut_ptr().cast(),
            &mut size,
        )
    };
    assert_eq!(status, 0, "the Run value of {name}");
    data.truncate(size as usize / 2 - 1);
    String::from_utf16(&data).unwrap()
}

/// A child that asked to die with its parent is gone soon after the parent is killed.
#[cfg(unix)]
fn a_child_watching_its_parent_dies_with_it() {
    let dir = scratch("parent-watch");
    let file = dir.join("watcher.json");
    let mut parent = child_command(&["watching-parent", file.to_str().unwrap()])
        .spawn()
        .unwrap();
    ready(parent.stdout.take().unwrap());
    let watcher: Birth = serde_json::from_slice(&fs::read(&file).unwrap()).unwrap();
    let watched = process::Exact::open(&watcher)
        .unwrap()
        .expect("the watcher lives");
    parent.kill().unwrap();
    parent.wait().unwrap();
    // A hang bound for a watcher that never dies; the product waits without one.
    let died = watched
        .wait_for(std::time::Duration::from_secs(30))
        .unwrap();
    if !died {
        watched.kill().unwrap();
    }
    assert!(died, "the watcher outlived its parent");
    fs::remove_dir_all(dir).unwrap();
}
