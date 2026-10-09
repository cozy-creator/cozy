package producttest

import (
	"encoding/json"
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/records"
)

// holdSource is a package whose one call reports a step every 200 ms until its release file
// exists, then answers how many it took.
const holdSource = `import os
import time

import msgspec

from cozy_runtime.author import App, Context, Telemetry

app = App()


class Hold(msgspec.Struct):
    release: str


class Held(msgspec.Struct):
    steps: int


@app.entrypoint
def hold(payload: Hold, ctx: Context, tel: Telemetry) -> Held:
    on_step = tel.step_callback(1_000_000, stage="holding")
    steps = 0
    while not os.path.exists(payload.release):
        ctx.raise_if_cancelled()
        time.sleep(0.2)
        on_step(steps)
        steps += 1
    return Held(steps)
`

func holdProject(t *testing.T) string {
	t.Helper()
	project := filepath.Join(t.TempDir(), "observation-hold")
	must(t, os.MkdirAll(project, 0o700))
	sources := ""
	if *machineRuntimeWheel != "" {
		sources = fmt.Sprintf("[tool.uv.sources]\ncozy-runtime={path=%q}\ntensorfs={path=%q}\n", *machineRuntimeWheel, *machineTensorFSWheel)
	}
	must(t, os.WriteFile(filepath.Join(project, "pyproject.toml"), []byte(`[project]
name="observation-hold"
version="0.0.1"
requires-python=">=3.12,<3.13"
dependencies=["cozy-runtime>=0.21", "msgspec>=0.19"]
[project.entry-points."cozy.application"]
default="observation_hold:app"
`+sources+`[build-system]
requires=["hatchling"]
build-backend="hatchling.build"
[tool.hatch.build.targets.wheel]
only-include=["observation_hold.py"]
`), 0o600))
	must(t, os.WriteFile(filepath.Join(project, "package.toml"), []byte("[application]\nobject=\"observation_hold:app\"\n"), 0o600))
	must(t, os.WriteFile(filepath.Join(project, "observation_hold.py"), []byte(holdSource), 0o600))
	if out, err := exec.Command("uv", "lock", "--project", project).CombinedOutput(); err != nil {
		t.Fatalf("locking the hold package: %v\n%s", err, out)
	}
	return project
}

// blackhole fronts a machine's address. Cut, every connection through it carries nothing
// more, new ones included, while each socket stays open and acknowledged: a path that died
// silently, as a provider's TCP proxy or a lost NAT mapping leaves it. Only the connection's
// own pings can notice. Healed, new connections pass; the cut ones never recover.
type blackhole struct {
	addr  string
	mu    sync.Mutex
	dark  bool
	conns []*atomic.Bool
}

func newBlackhole(t *testing.T, target string) *blackhole {
	listener, err := net.Listen("tcp", "127.0.0.1:0") //cozy:allow test machine transport
	must(t, err)
	b := &blackhole{addr: listener.Addr().String()}
	var open []net.Conn
	t.Cleanup(func() {
		_ = listener.Close()
		b.mu.Lock()
		defer b.mu.Unlock()
		for _, conn := range open {
			_ = conn.Close()
		}
	})
	go func() {
		for {
			client, err := listener.Accept()
			if err != nil {
				return
			}
			server, err := net.Dial("tcp", target)
			if err != nil {
				_ = client.Close()
				continue
			}
			dark := new(atomic.Bool)
			b.mu.Lock()
			dark.Store(b.dark)
			b.conns = append(b.conns, dark)
			open = append(open, client, server)
			b.mu.Unlock()
			pipe := func(from, to net.Conn) {
				buf := make([]byte, 32<<10)
				for {
					n, err := from.Read(buf)
					if n > 0 && !dark.Load() {
						if _, err := to.Write(buf[:n]); err != nil {
							return
						}
					}
					if err != nil {
						if !dark.Load() {
							_ = to.Close()
						}
						return
					}
				}
			}
			go pipe(client, server)
			go pipe(server, client)
		}
	}()
	return b
}

func (b *blackhole) cut() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.dark = true
	for _, conn := range b.conns {
		conn.Store(true)
	}
}

func (b *blackhole) heal() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.dark = false
}

// listedRun is one run's row of `cozy run list --json`.
type listedRun struct {
	ID              string                   `json:"id"`
	Status          string                   `json:"status"`
	Position        *int64                   `json:"position"`
	ObservationLost *records.ObservationLoss `json:"observation_lost"`
}

func listed(t *testing.T, root, id string) listedRun {
	t.Helper()
	code, out := runCozy(t, root, "run", "list", "--json", "--full")
	var response struct {
		Invocations []listedRun `json:"invocations"`
	}
	if code != 0 || json.Unmarshal([]byte(lastJSONLine(out)), &response) != nil {
		t.Fatalf("run list [exit %d]: %s", code, out)
	}
	for _, run := range response.Invocations {
		if run.ID == id {
			return run
		}
	}
	t.Fatalf("run list has no %s: %s", id, out)
	return listedRun{}
}

func within(t *testing.T, root, what string, limit time.Duration, done func() bool) time.Duration {
	t.Helper()
	start := time.Now()
	for !done() {
		if time.Since(start) > limit {
			t.Fatalf("%s: not within %s\n%s", what, limit, tail(filepath.Join(root, "daemon.log")))
		}
		time.Sleep(500 * time.Millisecond)
	}
	return time.Since(start)
}

// holdOn submits the hold call to the rental and answers its request.
func holdOn(t *testing.T, root string, store *records.Store, key, release string) records.Request {
	t.Helper()
	if code, out := runCozy(t, root, "run", "local/observation-hold/hold", "release="+release, "--rental=tessa", "--json", "--idempotency-key", key); code != 0 {
		t.Fatalf("submit %s [exit %d]\n%s", key, code, out)
	}
	request, problem := store.RequestByIdempotencyKey(key)
	fatal(t, problem)
	return *request
}

func observations(t *testing.T, store *records.Store, id string) (lost, restored int) {
	t.Helper()
	events, problem := store.EvidenceEvents(id, 10000)
	fatal(t, problem)
	for _, event := range events {
		if event.Type == "machine.observation" {
			if event.Payload["lost"] != nil {
				lost++
			} else {
				restored++
			}
		}
	}
	return lost, restored
}

// A machine path that dies silently mid-run, the TCP socket still open and acknowledged, is
// noticed by the connection's own unanswered pings, not by any deadline on the run. While the
// machine cannot be heard, `cozy run list` says the run is reconnecting instead of showing its
// last progress as if live; once the path heals the daemon attaches again by itself, with no
// reader asking, and the run goes on to its result.
func TestASilentMachinePathIsNoticedAndTheRunAttachedAgain(t *testing.T) {
	var front *blackhole
	_, root, _, store := parityMachinesOn(t, machines.Source{Host: *machineHostBinary, RuntimeWheel: *machineRuntimeWheel, TensorFSWheel: *machineTensorFSWheel},
		func(addr string) string { front = newBlackhole(t, addr); return front.addr })
	if code, out := runCozy(t, root, "package", "install", holdProject(t), "--editable"); code != 0 {
		t.Fatalf("editable install [exit %d]\n%s", code, out)
	}

	release := filepath.Join(t.TempDir(), "release")
	run := holdOn(t, root, store, "silent-path", release)
	steps := func() int64 {
		life := listed(t, root, run.ID)
		if life.ObservationLost != nil || life.Position == nil {
			return -1
		}
		return *life.Position
	}
	within(t, root, "the run's progress", 5*time.Minute, func() bool { return steps() > 0 })

	front.cut()
	noticed := within(t, root, "the silent path noticed", 2*time.Minute, func() bool {
		life := listed(t, root, run.ID)
		return life.Status == "in_progress" && life.ObservationLost != nil
	})
	t.Logf("silent path noticed after %s", noticed)
	if _, human := runCozy(t, root, "run", "list"); !strings.Contains(human, "reconnecting to tessa") {
		t.Fatalf("run list does not say the run is reconnecting:\n%s", human)
	}
	if code, shown := runCozy(t, root, "run", "show", run.ID); code != 0 || !strings.Contains(shown, "reconnecting to tessa") {
		t.Fatalf("run show does not say the run is reconnecting [exit %d]:\n%s", code, shown)
	}

	front.heal()
	before := int64(0)
	within(t, root, "the run attached again", time.Minute, func() bool { before = steps(); return before > 0 })
	within(t, root, "live progress after attaching again", time.Minute, func() bool { return steps() > before })
	must(t, os.WriteFile(release, nil, 0o600))
	within(t, root, "the run's result", 2*time.Minute, func() bool {
		link, _ := store.MachineExecution(run.ID)
		return link != nil && link.Collected
	})
	if code, shown := runCozy(t, root, "run", "show", run.ID, "--json"); code != 0 || !strings.Contains(shown, `"steps":`) {
		t.Fatalf("the run lacks its result [exit %d]\n%s", code, shown)
	}
	if lost, restored := observations(t, store, run.ID); lost != 1 || restored != 1 {
		t.Fatalf("the run recorded %d losses and %d restorations of its observation", lost, restored)
	}
}

// A Hub the rental cannot reach for many of its authority leases is weather: the run on it
// goes on and is observed to its result, and a run submitted meanwhile is taken. A machine
// that lets the lease lapse ends the run's stream and refuses this computer's key until the
// Hub answers; the daemon attaches again and submits again, never failing either run.
func TestARentalsHubOutageNeitherLosesNorRefusesItsRuns(t *testing.T) {
	h, root, _, store := parityMachines(t)
	if code, out := runCozy(t, root, "package", "install", holdProject(t), "--editable"); code != 0 {
		t.Fatalf("editable install [exit %d]\n%s", code, out)
	}
	// One-second leases from here on, so an outage outlasts many.
	h.mu.Lock()
	h.lease, h.leased = 1, 0
	h.mu.Unlock()
	within(t, root, "a one-second lease", time.Minute, func() bool { h.mu.Lock(); defer h.mu.Unlock(); return h.leased >= 2 })

	release := filepath.Join(t.TempDir(), "release")
	long := holdOn(t, root, store, "outage-long", release)
	within(t, root, "the run's progress", 5*time.Minute, func() bool {
		life := listed(t, root, long.ID)
		return life.Position != nil && *life.Position > 0
	})
	h.mu.Lock()
	h.unreachable = true
	h.mu.Unlock()
	time.Sleep(4 * time.Second)
	done := filepath.Join(t.TempDir(), "done")
	must(t, os.WriteFile(done, nil, 0o600))
	short := holdOn(t, root, store, "outage-short", done)
	time.Sleep(4 * time.Second)
	h.mu.Lock()
	h.unreachable = false
	h.mu.Unlock()
	must(t, os.WriteFile(release, nil, 0o600))
	for _, run := range []records.Request{long, short} {
		within(t, root, run.IdemKey+"'s result", 3*time.Minute, func() bool {
			link, _ := store.MachineExecution(run.ID)
			return link != nil && link.Collected
		})
		current, problem := store.RequestRow(run.ID)
		fatal(t, problem)
		if current.State != "succeeded" {
			t.Fatalf("%s ended %s through the Hub outage", run.IdemKey, current.State)
		}
		lost, restored := observations(t, store, run.ID)
		t.Logf("%s: observation lost %d, restored %d", run.IdemKey, lost, restored)
		if lost != restored {
			t.Fatalf("%s's observation was never restored", run.IdemKey)
		}
	}
}
