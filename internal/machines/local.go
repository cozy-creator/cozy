package machines

import (
	"bufio"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"encoding/pem"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/flock"
	"github.com/cozy-creator/cozy/internal/hostruntime"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/rental"
	"github.com/cozy-creator/cozy/internal/userunit"
	"github.com/cozy-creator/cozy/internal/workertls"
	"github.com/google/uuid"
)

// Host installs and discovers the independent machine agent. Its process and durable
// execution state outlive the personal controller, using the same server as a rented pod.
type Host struct {
	dir string
	// store is the box's one TensorFS store, the location this machine's Host uses for
	// its own, so models are never held twice on one disk. Empty keeps the pod layout.
	store  string
	mu     sync.Mutex
	cached *cachedLaunch
	// asked is the machine executable servesAPI last asked, and serves its answer.
	asked  os.FileInfo
	serves bool
	// inherited are the locale, trust-store and GPU-visibility values a Host may carry from its
	// launcher.
	inherited []string
	// GPUBudget seeds a new machine config. Existing machine settings belong to its owner.
	GPUBudget any
	// WebRTCPort is explicitly configured by the machine operator, never by a Hub grant.
	WebRTCPort int
}

func NewHost(dir, store string, environ []string) *Host {
	// A machine moved elsewhere by a symlink (another disk, a shorter path) runs there: the
	// Runtime's sockets live under the root it is given, and a socket path is bounded.
	if resolved, err := filepath.EvalSymlinks(dir); err == nil {
		dir = resolved
	}
	h := &Host{dir: dir, store: store}
	for _, value := range environ {
		name, _, _ := strings.Cut(value, "=")
		switch name {
		case "LANG", "LC_ALL", "TZ", "SSL_CERT_FILE", "SSL_CERT_DIR", "CUDA_VISIBLE_DEVICES":
			h.inherited = append(h.inherited, value)
		}
	}
	return h
}

func (h *Host) Root() string            { return filepath.Join(h.dir, "root") }
func (h *Host) path(name string) string { return filepath.Join(h.dir, name) }

// The image layout the Host and Runtime share, rooted.
func (h *Host) binary() string { return filepath.Join(h.Root(), "usr/local/bin/tensord") }

// wheels holds exactly the executor SDK the machine runs package code with: the Runtime and
// TensorFS wheels, where a worker image keeps them.
func (h *Host) wheels() string { return filepath.Join(h.Root(), "opt/cozy/machine/wheels") }
func (h *Host) readinessEnvelope() string {
	return filepath.Join(h.Root(), "run/cozy/bootstrap/readiness-envelope.json")
}

// Source is the software an install names: local files, or published versions (none: the
// newest). Named files pin the machine; it then never follows the Hub's target.
type Source struct {
	Host, RuntimeWheel, TensorFSWheel string
	RuntimeVersion, TensorFSVersion   string
}

func (s Source) pinned() bool { return s.Host != "" || s.RuntimeWheel != "" || s.TensorFSWheel != "" }

type installedArtifact struct {
	Name   string `json:"name"`
	SHA256 string `json:"sha256"`
}

// Installed names the artifacts the root holds.
type Installed struct {
	Host        installedArtifact `json:"host"`
	Runtime     installedArtifact `json:"runtime"`
	TensorFS    installedArtifact `json:"tensorfs"`
	InstalledAt time.Time         `json:"installed_at"`
	// Pinned: its owner named the files it runs, so it never follows the Hub's target.
	Pinned bool `json:"pinned,omitempty"`
	// Replaced is set when this install replaced a machine that predated MachineAPI.
	Replaced *Replaced `json:"replaced,omitempty"`
}

// placeHost atomically installs the independent agent without changing machine state.
func (h *Host) placeHost(source string) (installedArtifact, error) {
	digest, err := fileDigest(source)
	if err != nil {
		return installedArtifact{}, err
	}
	artifact := installedArtifact{Name: filepath.Base(source), SHA256: digest}
	target := filepath.Join(h.Root(), "usr/local/bin/tensord")
	if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
		return artifact, err
	}
	return artifact, copyFile(source, target, 0o755)
}

func (h *Host) Installed() (*Installed, *exit.Error) {
	raw, err := os.ReadFile(h.path("installed.json"))
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, exit.Internalf("cannot read the local machine's installation: %s", err)
	}
	var out Installed
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, exit.New(exit.Conflict, "the local machine's installation record is unreadable")
	}
	return &out, nil
}

// Install makes this computer's machine: a new one, an installed one updated in place, or an
// installed one that predates MachineAPI (the Go agent) replaced.
func (h *Host) Install(ctx context.Context, source Source, uv string) (*Installed, *exit.Error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	unlock, problem := h.lock(ctx)
	if problem != nil {
		return nil, problem
	}
	defer unlock()
	if problem := h.recoverReplace(ctx); problem != nil {
		return nil, problem
	}
	previous, problem := h.Installed()
	if problem != nil {
		return nil, problem
	}
	if problem := h.refuseRenamedNative(ctx); problem != nil {
		return nil, problem
	}
	if (source.RuntimeWheel == "") != (source.TensorFSWheel == "") ||
		source.RuntimeWheel != "" && (!strings.HasSuffix(source.RuntimeWheel, ".whl") || !strings.HasSuffix(source.TensorFSWheel, ".whl")) {
		return nil, exit.New(exit.Validation, "the machine install names both wheels or neither, with an optional machine executable")
	}
	if previous != nil && h.servesAPI(ctx) {
		return h.updateLocked(ctx, source)
	}
	if previous == nil {
		// A root some machine owns is refused before anything is fetched.
		release, problem := h.guard()
		if problem != nil {
			return nil, problem
		}
		release()
	}
	// Everything the new machine needs is in hand before an installed one is touched.
	staged, problem := h.stage(ctx, source)
	if problem != nil {
		return nil, problem
	}
	defer os.RemoveAll(staged.dir)
	if previous == nil {
		return h.place(staged, uv)
	}
	return h.replaceLocked(ctx, staged, uv)
}

// staged is a machine ready to place: its executable and the executor SDK's wheel pair.
type staged struct {
	dir, agent string
	wheels     []string
	// pinned: the install named files; bundled: its machine came from the Runtime wheel.
	pinned, bundled bool
}

// stage fetches what source leaves unnamed (its published versions or the newest, the machine
// its Runtime wheel bundles) and checks the machine serves MachineAPI.
func (h *Host) stage(ctx context.Context, source Source) (*staged, *exit.Error) {
	dir, err := os.MkdirTemp(h.dir, ".install-")
	if err != nil {
		return nil, exit.Internalf("cannot stage the machine install: %s", err)
	}
	out := &staged{dir: dir, agent: source.Host, wheels: []string{source.RuntimeWheel, source.TensorFSWheel}, pinned: source.pinned(), bundled: source.Host == ""}
	problem := func() *exit.Error {
		if source.RuntimeWheel == "" {
			for i, published := range [][2]string{{hostruntime.Distribution, source.RuntimeVersion}, {"tensorfs", source.TensorFSVersion}} {
				if out.wheels[i], err = publishedWheel(ctx, published[0], published[1], dir); err != nil {
					return exit.New(exit.Unavailable, "cannot fetch the published %s: %s", published[0], err)
				}
			}
		}
		if out.agent == "" {
			if out.agent, err = bundledAgent(out.wheels[0], dir); err != nil {
				return exit.Named(exit.Structural, "machine.agent_unsupported", "%s", err).
					WithRemedy("name a tensord with --host")
			}
		}
		if !servesAPI(ctx, out.agent) {
			return exit.Named(exit.Structural, "machine.agent_unsupported", "%s is not a tensord serving %s", filepath.Base(out.agent), MachineAPI).
				WithRemedy("install a current Runtime pair, or name a current tensord with --host")
		}
		return nil
	}()
	if problem != nil {
		os.RemoveAll(dir)
		return nil, problem
	}
	return out, nil
}

// place lays the root out as a worker image does: the machine at usr/local/bin/tensord,
// the executor SDK (Runtime and TensorFS wheels) at opt/cozy/machine/wheels and uv at
// usr/local/bin/uv. The machine makes its own identity files, installer helper and package
// environments there, as on a rental.
func (h *Host) place(staged *staged, uv string) (*Installed, *exit.Error) {
	release, problem := h.guard()
	if problem != nil {
		return nil, problem
	}
	defer release()
	if _, problem := h.identity(); problem != nil {
		return nil, problem
	}
	var err error
	installed := Installed{InstalledAt: time.Now().UTC(), Pinned: staged.pinned}
	if installed.Host, err = h.placeHost(staged.agent); err != nil {
		return nil, exit.Internalf("cannot install the machine: %s", err)
	}
	if staged.bundled {
		installed.Host.Name = "tensord"
	}
	kept, err := h.keepWheels(staged.wheels)
	if err != nil {
		return nil, exit.Internalf("cannot keep the machine's wheels: %s", err)
	}
	for i, artifact := range []*installedArtifact{&installed.Runtime, &installed.TensorFS} {
		digest, err := fileDigest(kept[i])
		if err != nil {
			return nil, exit.Internalf("cannot read %s: %s", kept[i], err)
		}
		*artifact = installedArtifact{Name: filepath.Base(kept[i]), SHA256: digest}
	}
	link := filepath.Join(h.Root(), "usr/local/bin/uv")
	_ = os.Remove(link)
	if err := os.Symlink(uv, link); err != nil {
		return nil, exit.Internalf("cannot link uv: %s", err)
	}
	if err := h.recordInstalled(installed); err != nil {
		return nil, exit.Internalf("cannot record the machine installation: %s", err)
	}
	return &installed, nil
}

// Only the legacy identity is read during adoption. Worker tokens and Hub settings
// are deliberately not decoded into the new lifecycle.
type registration struct {
	Hub string `json:"hub"`
	ID  string `json:"id"`
}

type hostRecord struct {
	PID        int    `json:"pid"`
	WorkerID   string `json:"worker_id"`
	WorkerPort int    `json:"worker_port"`
	MediaPort  int    `json:"media_port"`
	ReceiptKey string `json:"receipt_key"`
	StartTicks uint64 `json:"start_ticks,omitempty"`
}

type cachedLaunch struct {
	*Launch
	record hostRecord
}

// Launch is the running Host's identity once its readiness receipt verified.
type Launch struct {
	Addr, MediaAddr  string
	WorkerID, BootID string
	AgentVersion     string
	Capabilities     []string
	Leaf             []byte
	GPUs             []ReceiptGPU
	PID              int
	// Kept says why a machine this launch started kept its software instead of the Hub's
	// target, or "".
	Kept string
}

type attachKey struct{}

// AttachOnly marks work that uses this computer's machine only while it runs: observing,
// collecting and controlling runs it already accepted. Such work never starts a stopped
// machine; it waits for the next local run or `cozy machine start`.
func AttachOnly(ctx context.Context) context.Context {
	return context.WithValue(ctx, attachKey{}, true)
}

func stopped() *exit.Error {
	return exit.Named(exit.Unavailable, "machine.stopped", "this computer's machine is stopped; work it already accepted waits for it").
		WithRemedy("`cozy machine start`, or the next local run, starts it")
}

// Ensure starts the machine independently of any Hub, or attaches to the running one.
func (h *Host) Ensure(ctx context.Context, account *hub.Client) (*Launch, *exit.Error) {
	attach, _ := ctx.Value(attachKey{}).(bool)
	if _, err := os.Stat(h.path("agent.json")); attach && errors.Is(err, os.ErrNotExist) {
		return nil, stopped() // one file read: background work asks often while the machine is stopped
	}
	if !h.mu.TryLock() {
		return nil, exit.Named(exit.Unavailable, "machine.busy", "another command is changing the local machine")
	}
	defer h.mu.Unlock()
	unlock, problem := h.acquireLock(ctx, false)
	if problem != nil {
		return nil, problem
	}
	defer unlock()
	if problem := h.recoverReplace(ctx); problem != nil {
		return nil, problem
	}
	return h.ensureLocked(ctx, account, !attach)
}

// Start starts the machine, or attaches to the running one, after any command changing it. A
// machine it starts follows the target software account's Hub names.
func (h *Host) Start(ctx context.Context, account *hub.Client) (*Launch, *exit.Error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	unlock, problem := h.lock(ctx)
	if problem != nil {
		return nil, problem
	}
	defer unlock()
	if problem := h.recoverReplace(ctx); problem != nil {
		return nil, problem
	}
	return h.ensureLocked(ctx, account, true)
}

func (h *Host) ensureLocked(ctx context.Context, account *hub.Client, start bool) (*Launch, *exit.Error) {
	if problem := h.refuseRenamedNative(ctx); problem != nil {
		return nil, problem
	}
	var launch *Launch
	if cached := h.cached; cached != nil && h.alive(cached.PID) && (cached.record.StartTicks == 0 || cached.record.StartTicks == processStartTicks(cached.PID)) {
		launch = cached.Launch
	}
	if launch == nil {
		record, problem := h.running()
		switch {
		case problem != nil:
			return nil, problem
		case record == nil && !start:
			return nil, stopped()
		case h.predatesAPI(ctx):
			// Asked only of a machine about to be awaited or launched; it never readies on the API.
			return nil, exit.Named(exit.Structural, "machine.predates_api", "this computer's machine is an older kind this cozy cannot run").
				WithRemedy("run `cozy machine install` to replace it; your downloaded models are kept")
		case record != nil:
			if launch, problem = h.await(ctx, record); problem != nil {
				return nil, problem
			}
			h.remember(launch, *record)
		default:
			if launch, problem = h.launchLocked(ctx); problem != nil {
				return nil, problem
			}
			launch.Kept = h.follow(ctx, account)
		}
	}
	return launch, nil
}

func (h *Host) remember(launch *Launch, record hostRecord) *Launch {
	if launch != nil {
		h.cached = &cachedLaunch{Launch: launch, record: record}
	}
	return launch
}

// await reads the recorded Host's readiness receipt, waiting while it boots.
func (h *Host) await(ctx context.Context, record *hostRecord) (*Launch, *exit.Error) {
	key, err := base64.RawURLEncoding.DecodeString(record.ReceiptKey)
	if err != nil {
		return nil, exit.New(exit.Conflict, "the machine Host record carries no receipt key")
	}
	for {
		receipt, leaf, err := readReceipt(ctx, record.WorkerPort, key)
		if err == nil {
			if receipt.WorkerInternalPort != record.WorkerPort {
				return nil, exit.New(exit.Conflict, "the machine Host's receipt names another worker port")
			}
			if err := writePrivate(h.path("leaf.pem"), pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: leaf})); err != nil {
				return nil, exit.Internalf("cannot record the machine's TLS leaf: %s", err)
			}
			return &Launch{Addr: "127.0.0.1:" + strconv.Itoa(record.WorkerPort), MediaAddr: "127.0.0.1:" + strconv.Itoa(record.MediaPort),
				WorkerID: record.WorkerID, BootID: receipt.PodBootID, AgentVersion: receipt.MachineVersion, Capabilities: receipt.MachineCapabilities, Leaf: leaf, GPUs: receipt.RuntimeGPUs, PID: record.PID}, nil
		}
		if refused := (*receiptRefusal)(nil); errors.As(err, &refused) {
			return nil, exit.New(exit.Credential, "the machine Host's readiness receipt did not verify: %s", err)
		}
		if !h.alive(record.PID) && !h.starting(record.PID) {
			return nil, exit.Named(exit.Structural, "machine.host_exited", "the machine Host exited before readiness")
		}
		select {
		case <-ctx.Done():
			return nil, exit.Named(exit.Unavailable, "machine.host_starting", "the machine Host is still starting")
		case <-time.After(250 * time.Millisecond):
		}
	}
}

// launchLocked starts a new agent once running found none.
func (h *Host) launchLocked(ctx context.Context) (*Launch, *exit.Error) {
	// A directly launched machine may have no client record: the root's guard, not the
	// record, says whether one runs.
	release, problem := h.guard()
	if problem != nil {
		return nil, problem
	}
	release()

	id, problem := h.identity()
	if problem != nil {
		return nil, problem
	}
	installed, problem := h.Installed()
	if problem != nil {
		return nil, problem
	}
	if installed == nil {
		return nil, exit.Named(exit.Structural, "machine.not_installed", "this computer has no machine installed").
			WithRemedy("cozy machine install")
	}
	owner, problem := rental.OwnerIdentityAt(h.path("owner.pem"))
	if problem != nil {
		return nil, problem
	}
	key := make([]byte, 32)
	if _, err := rand.Read(key); err != nil {
		return nil, exit.Internalf("cannot mint the machine receipt key: %s", err)
	}
	// A retained envelope is the previous launch's; this launch proves itself under a new key.
	if err := os.Remove(h.readinessEnvelope()); err != nil && !errors.Is(err, os.ErrNotExist) {
		return nil, exit.Internalf("cannot retire the previous readiness receipt: %s", err)
	}
	if err := writePrivate(h.path("receipt-key"), []byte(base64.RawURLEncoding.EncodeToString(key))); err != nil {
		return nil, exit.Internalf("cannot retain the machine receipt key: %s", err)
	}
	base := append([]string{
		"COZY_MACHINE_ROOT=" + h.Root(),
		"COZY_LISTEN_HOST=127.0.0.1",
		"COZY_WORKER_ID=" + id,
		"COZY_MACHINE_LIFETIME=persistent",
		"COZY_BOOTSTRAP_RECEIPT_HMAC_KEY_FILE=" + h.path("receipt-key"),
		"COZY_AUTHORIZED_KEYS=" + owner.PublicKey(),
	}, h.inherited...)
	if h.store != "" {
		base = append(base, "COZY_TENSORFS_ROOT="+h.store)
	}
	if h.WebRTCPort != 0 {
		base = append(base, "COZY_WEBRTC_INTERNAL_PORT="+strconv.Itoa(h.WebRTCPort))
	}
	// Free ports are chosen, not reserved: a port taken before the Host binds it ends
	// that Host before readiness, and the next launch chooses again.
	for attempt := 0; ; attempt++ {
		launch, record, problem := h.start(ctx, base, key, hostRecord{WorkerID: id})
		if problem == nil || problem.ErrName() != "machine.host_port_taken" || attempt == 2 {
			return h.remember(launch, record), problem
		}
	}
}

func (h *Host) start(ctx context.Context, base []string, key []byte, record hostRecord) (*Launch, hostRecord, *exit.Error) {
	workerPort := freePort(0)
	mediaPort := freePort(workerPort)
	env := append(append([]string(nil), base...),
		"COZY_WORKER_INTERNAL_PORT="+strconv.Itoa(workerPort), "COZY_MEDIA_INTERNAL_PORT="+strconv.Itoa(mediaPort))
	if problem := h.writeRuntimeConfig(); problem != nil {
		return nil, record, problem
	}
	log, err := os.OpenFile(h.path("host.log"), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		return nil, record, exit.Internalf("cannot open the machine Host log: %s", err)
	}
	defer log.Close()
	offset, _ := log.Seek(0, io.SeekEnd)
	pid, err := h.spawn(env, log)
	if err != nil {
		return nil, record, exit.Internalf("cannot start the machine Host: %s", err)
	}
	record.PID, record.WorkerPort, record.MediaPort = pid, workerPort, mediaPort
	record.StartTicks = processStartTicks(pid)
	record.ReceiptKey = base64.RawURLEncoding.EncodeToString(key)
	raw, _ := json.Marshal(record)
	if err := writePrivate(h.path("agent.json"), raw); err != nil {
		_ = terminate(pid)
		return nil, record, exit.Internalf("cannot record the machine Host: %s", err)
	}
	launch, problem := h.await(ctx, &record)
	if problem != nil && problem.ErrName() == "machine.host_exited" {
		output := logSince(h.path("host.log"), offset)
		if strings.Contains(output, "address already in use") {
			return nil, record, exit.Named(exit.Unavailable, "machine.host_port_taken", "the machine Host's port was taken")
		}
		return nil, record, exit.Named(exit.Structural, "machine.host_exited", "the machine Host exited before readiness: %s", output)
	}
	if problem != nil && problem.Code == exit.Credential {
		_ = terminate(pid)
	}
	return launch, record, problem
}

// spawn starts the Host detached from whatever started this process: as its own user unit
// where user systemd runs (it outlives the daemon's unit and any session scope), else in
// its own session. Its output appends to host.log.
func (h *Host) spawn(env []string, log *os.File) (int, error) {
	if userunit.Available() {
		unit := h.Unit()
		already, err := userunit.Start(userunit.Spec{Unit: unit, Argv: []string{h.binary()}, Env: env, Dir: h.Root(), Stdout: log.Name(), Stderr: log.Name()})
		if err != nil {
			return 0, err
		}
		if already {
			// Only a launcher outside host.lock gets here; the next use adopts its agent.
			return 0, fmt.Errorf("the machine unit started meanwhile; the next use attaches to it")
		}
		if pid := userunit.MainPID(unit); pid > 0 {
			return pid, nil
		}
		return 0, fmt.Errorf("the machine Host unit %s started no process", unit)
	}
	command := exec.Command(h.binary())
	command.Env, command.Dir = env, h.Root()
	command.Stdout, command.Stderr = log, log
	detach(command)
	if err := command.Start(); err != nil {
		return 0, err
	}
	go func() { _ = command.Wait() }()
	return command.Process.Pid, nil
}

// Stop ends the running Host. Its Store, installs and identity remain under the root.
func (h *Host) Stop(ctx context.Context) *exit.Error {
	h.mu.Lock()
	defer h.mu.Unlock()
	unlock, problem := h.lock(ctx)
	if problem != nil {
		return problem
	}
	defer unlock()
	if problem := h.refuseRenamedNative(ctx); problem != nil {
		return problem
	}
	return h.stopLocked(ctx)
}

// stopLocked ends the root's agent, then retires its record. The root's unit is stopped by
// name, record or not: systemd answers once every process it ran is gone, its executor's too
// (a PID's executable is not yet the agent's there, and a stop that trusted it left one running).
func (h *Host) stopLocked(ctx context.Context) *exit.Error {
	h.cached = nil
	record, problem := h.record()
	if problem != nil {
		return problem
	}
	if unit := h.Unit(); userunit.Available() && userunit.Running(unit) {
		if err := userunit.Stop(unit); userunit.Running(unit) {
			return exit.Internalf("cannot stop the machine unit %s: %v", unit, err)
		}
	}
	if record != nil && h.alive(record.PID) {
		if err := terminate(record.PID); err != nil {
			return exit.Internalf("cannot stop the machine Host: %s", err)
		}
		for h.alive(record.PID) {
			select {
			case <-ctx.Done():
				return exit.Named(exit.Unavailable, "machine.host_stopping", "the machine Host is still stopping")
			case <-time.After(100 * time.Millisecond):
			}
		}
	}
	if err := os.Remove(h.path("agent.json")); err != nil && !errors.Is(err, os.ErrNotExist) {
		return exit.Internalf("cannot retire the machine Host record: %s", err)
	}
	return nil
}

// Status is the local machine as `cozy machine status` shows it.
type Status struct {
	Installed *Installed
	MachineID string
	PID       int
	Running   bool
	Recorded  bool // the running agent has its launch record; the next use adopts one without
}

func (h *Host) Status() (Status, *exit.Error) {
	installed, problem := h.Installed()
	if problem != nil {
		return Status{}, problem
	}
	out := Status{Installed: installed}
	record, problem := h.record()
	if problem != nil {
		return Status{}, problem
	}
	if record != nil {
		out.PID, out.Running, out.MachineID = record.PID, h.alive(record.PID), record.WorkerID
	}
	if unit := h.Unit(); userunit.Available() && userunit.Running(unit) {
		out.PID, out.Running = userunit.MainPID(unit), true
	}
	if !out.Running {
		out.PID = 0
	}
	out.Recorded = out.Running && record != nil && record.PID == out.PID
	if out.MachineID == "" {
		if raw, err := os.ReadFile(h.path("machine-id")); err == nil {
			out.MachineID = strings.TrimSpace(string(raw))
		}
	}
	return out, nil
}

// Owner is the key this controller signs the machine's Claims with.
func (h *Host) Owner() (rental.CreatorIdentity, *exit.Error) {
	return rental.OwnerIdentityAt(h.path("owner.pem"))
}

func (h *Host) ExistingOwner() (rental.CreatorIdentity, *exit.Error) {
	return rental.ExistingOwnerIdentityAt(h.path("owner.pem"))
}

// Pin is the TLS leaf the running Host proved in its receipt.
func (h *Host) Pin() (*workertls.Pin, *exit.Error) {
	pin, err := workertls.LoadPin(h.path("leaf.pem"))
	if err != nil {
		return nil, exit.New(exit.Credential, "the machine's TLS leaf is unreadable: %s", err)
	}
	return pin, nil
}

// registrations reads legacy identities for one-time adoption. It neither reads legacy
// worker capabilities nor contacts the registry that issued them.
func (h *Host) registrations() (map[string]registration, *exit.Error) {
	out := map[string]registration{}
	raw, err := os.ReadFile(h.path("registrations.json"))
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		return nil, exit.Internalf("cannot read the machine's registrations: %s", err)
	}
	if err == nil && json.Unmarshal(raw, &out) != nil {
		return nil, exit.New(exit.Conflict, "the machine's registrations are unreadable")
	}
	var legacy registration
	if raw, err := os.ReadFile(h.path("registration.json")); err == nil && json.Unmarshal(raw, &legacy) == nil && legacy.ID != "" {
		if _, known := out[legacy.Hub]; !known {
			out[legacy.Hub] = legacy
		}
	}
	return out, nil
}

// identity is locally allocated once. Import the last active legacy identity on adoption
// so old records retain their machine, without retaining its Hub worker capability.
func (h *Host) identity() (string, *exit.Error) {
	if raw, err := os.ReadFile(h.path("machine-id")); err == nil {
		id := strings.TrimSpace(string(raw))
		if id == "" || len(id) > 128 || strings.ContainsAny(id, "/?#% \t\r\n") {
			return "", exit.New(exit.Conflict, "the machine identity is unreadable")
		}
		return id, nil
	} else if !errors.Is(err, os.ErrNotExist) {
		return "", exit.Internalf("cannot read machine identity: %s", err)
	}
	id := ""
	if record, problem := h.record(); problem != nil {
		return "", problem
	} else if record != nil {
		id = record.WorkerID
	}
	if id == "" {
		all, problem := h.registrations()
		if problem != nil {
			return "", problem
		}
		keys := make([]string, 0, len(all))
		for key := range all {
			keys = append(keys, key)
		}
		slices.Sort(keys)
		for _, key := range keys {
			if all[key].ID != "" {
				id = all[key].ID
				break
			}
		}
	}
	if id == "" {
		id = "machine-" + uuid.NewString()
	}
	if err := writePrivate(h.path("machine-id"), []byte(id+"\n")); err != nil {
		return "", exit.Internalf("cannot record machine identity: %s", err)
	}
	return id, nil
}

func (h *Host) secret(name string) (string, *exit.Error) {
	if raw, err := os.ReadFile(h.path(name)); err == nil && len(raw) == 43 {
		return string(raw), nil
	}
	random := make([]byte, 32)
	if _, err := rand.Read(random); err != nil {
		return "", exit.Internalf("cannot mint %s: %s", name, err)
	}
	value := base64.RawURLEncoding.EncodeToString(random)
	if err := writePrivate(h.path(name), []byte(value)); err != nil {
		return "", exit.Internalf("cannot record %s: %s", name, err)
	}
	return value, nil
}

func (h *Host) record() (*hostRecord, *exit.Error) {
	if userunit.Available() && userunit.MainPID(userunit.Name("tensord", h.dir, false)) > 0 {
		return nil, exit.Named(exit.Conflict, "machine.legacy_process_running", "the retired machine unit still owns this root; stop it with the original CLI after its accepted work finishes")
	}

	// A namespace cut must not start a second process over an occupied root.
	// This is a safety census only: no legacy request or credential is honored.
	if raw, err := os.ReadFile(h.path("host.json")); err == nil {
		var old hostRecord
		if json.Unmarshal(raw, &old) == nil && h.legacyAlive(old.PID) {
			return nil, exit.Named(exit.Conflict, "machine.legacy_process_running", "a retired machine process still owns this root; stop it with the original CLI after its accepted work finishes")
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		return nil, exit.Internalf("cannot inspect the previous machine record: %s", err)
	}
	raw, err := os.ReadFile(h.path("agent.json"))
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, exit.Internalf("cannot read the machine Host record: %s", err)
	}
	var record hostRecord
	if json.Unmarshal(raw, &record) != nil {
		return nil, nil
	}
	if record.StartTicks != 0 && record.StartTicks != processStartTicks(record.PID) {
		record.PID = 0
	}
	return &record, nil
}

// Unit is the root's systemd user unit. Its name derives from the root, so whatever it runs
// is this root's agent, launch record or not.
func (h *Host) Unit() string { return userunit.Name("tensord-agent", h.dir, false) }

// running is the root's live agent's launch record, nil when none runs; a record whose agent
// is gone is retired. An agent the root's unit runs without its record (an older cozy's stop
// removed it) is adopted: its record is read back from the agent's own environment.
func (h *Host) running() (*hostRecord, *exit.Error) {
	record, problem := h.record()
	if problem != nil {
		return nil, problem
	}
	unit := h.Unit()
	if !userunit.Available() || !userunit.Running(unit) {
		if record != nil && h.alive(record.PID) {
			return record, nil
		}
		if err := os.Remove(h.path("agent.json")); err != nil && !errors.Is(err, os.ErrNotExist) {
			return nil, exit.Internalf("cannot retire the machine Host record: %s", err)
		}
		return nil, nil
	}
	pid := userunit.MainPID(unit)
	if record != nil && record.PID == pid {
		return record, nil
	}
	if !h.alive(pid) { // systemd's executor has not exec'd the agent yet
		return nil, exit.Named(exit.Unavailable, "machine.host_starting", "the machine unit %s is still starting", unit)
	}
	environ, _ := os.ReadFile("/proc/" + strconv.Itoa(pid) + "/environ")
	env := map[string]string{}
	for _, pair := range strings.Split(string(environ), "\x00") {
		name, value, _ := strings.Cut(pair, "=")
		env[name] = value
	}
	key, _ := os.ReadFile(h.path("receipt-key"))
	adopted := hostRecord{PID: pid, WorkerID: env["COZY_WORKER_ID"], ReceiptKey: strings.TrimSpace(string(key)), StartTicks: processStartTicks(pid)}
	adopted.WorkerPort, _ = strconv.Atoi(env["COZY_WORKER_INTERNAL_PORT"])
	adopted.MediaPort, _ = strconv.Atoi(env["COZY_MEDIA_INTERNAL_PORT"])
	if env["COZY_MACHINE_ROOT"] != h.Root() || adopted.WorkerID == "" || adopted.WorkerPort == 0 || adopted.MediaPort == 0 || adopted.ReceiptKey == "" {
		return nil, exit.Named(exit.Conflict, "machine.process_untracked", "the machine unit %s runs an agent whose launch cannot be read back", unit).
			WithRemedy("`cozy machine stop` ends it; `cozy machine start` then starts a fresh one")
	}
	raw, _ := json.Marshal(adopted)
	if err := writePrivate(h.path("agent.json"), raw); err != nil {
		return nil, exit.Internalf("cannot record the adopted machine Host: %s", err)
	}
	h.note(fmt.Sprintf("adopted the running agent %d of unit %s, whose launch record was missing", pid, unit))
	return &adopted, nil
}

// alive is whether pid is this root's Host, not merely a live process that reused the pid.
func (h *Host) alive(pid int) bool {
	if pid <= 0 {
		return false
	}
	target, err := os.Readlink("/proc/" + strconv.Itoa(pid) + "/exe")
	if err != nil {
		return false
	}
	target = strings.TrimSuffix(target, " (deleted)")
	for _, relative := range []string{"usr/local/bin/tensord", "opt/cozy/python/bin/tensord"} {
		path := filepath.Join(h.Root(), relative)
		if target == path {
			return true
		}
		resolved, err := filepath.EvalSymlinks(path)
		if err == nil && target == resolved && strings.HasPrefix(resolved, h.Root()+string(filepath.Separator)) {
			return true
		}
	}
	return false
}

// starting is a Host its user unit runs that is not yet the agent: systemd's executor runs
// first as the unit's main process, at the unit's nice and quota, and on a loaded box it can
// outlast the first readiness probe (run 2311 failed so, 42 ms after the unit started).
func (h *Host) starting(pid int) bool {
	return userunit.Available() && userunit.MainPID(h.Unit()) == pid && userunit.Running(h.Unit())
}

// Legacy identity is only a refusal census, never permission to signal a PID.
func (h *Host) legacyAlive(pid int) bool {
	if h.alive(pid) {
		return true
	}
	if pid <= 0 {
		return false
	}
	target, err := os.Readlink("/proc/" + strconv.Itoa(pid) + "/exe")
	if err != nil {
		return false
	}
	target = strings.TrimSuffix(target, " (deleted)")
	for _, relative := range []string{"usr/local/bin/pod-supervisor", "usr/local/bin/cozy"} {
		path := filepath.Join(h.Root(), relative)
		if target == path {
			return true
		}
		resolved, err := filepath.EvalSymlinks(path)
		cwd, cwdErr := os.Readlink("/proc/" + strconv.Itoa(pid) + "/cwd")
		if err == nil && target == resolved && cwdErr == nil && cwd == h.Root() {
			return true
		}
	}
	return false
}

func processStartTicks(pid int) uint64 {
	raw, err := os.ReadFile("/proc/" + strconv.Itoa(pid) + "/stat")
	if err != nil {
		return 0
	}
	end := strings.LastIndexByte(string(raw), ')')
	if end < 0 {
		return 0
	}
	fields := strings.Fields(string(raw[end+1:]))
	if len(fields) <= 19 {
		return 0
	}
	ticks, _ := strconv.ParseUint(fields[19], 10, 64)
	return ticks
}

// guardLock is the one lock every machine on a root holds for its life: the Go agent's and
// the Rust machine's alike (tensord `machine::identity::hold`). runtimeLock is the Go
// agent's Python Runtime's, which can outlive its agent; it counts while such a Runtime can exist.
const (
	guardLock   = "var/lib/cozy/machine/agent.lock"
	runtimeLock = "run/cozy/worker/worker.lock"
)

// guard holds the root's lock while this controller changes or starts its machine: the
// kernel, not a client record, says no machine runs there.
func (h *Host) guard() (func(), *exit.Error) {
	busy := exit.Named(exit.Conflict, "machine.busy", "a live machine or Runtime owns this root").
		WithRemedy("`cozy machine stop` ends it")
	file, held, err := probe(filepath.Join(h.Root(), guardLock), true)
	if err != nil {
		return nil, exit.Internalf("cannot inspect machine ownership: %s", err)
	}
	if held {
		return nil, busy
	}
	release := func() { _ = flock.Release(file); _ = file.Close() }
	runtime, held, err := probe(filepath.Join(h.Root(), runtimeLock), false)
	if runtime != nil {
		_ = flock.Release(runtime)
		_ = runtime.Close()
	}
	if err != nil || held {
		release()
		if err != nil {
			return nil, exit.Internalf("cannot inspect machine ownership: %s", err)
		}
		return nil, busy
	}
	return release, nil
}

// probe takes path's exclusive lock, or reports that a live process holds it. An absent file
// is created only when asked; otherwise nothing holds it.
func probe(path string, create bool) (*os.File, bool, error) {
	flags := os.O_RDWR
	if create {
		if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
			return nil, false, err
		}
		flags |= os.O_CREATE
	}
	file, err := os.OpenFile(path, flags, 0o600)
	if errors.Is(err, os.ErrNotExist) {
		return nil, false, nil
	}
	if err != nil {
		return nil, false, err
	}
	if flock.Exclusive(file) != nil {
		file.Close()
		return nil, true, nil
	}
	return file, false, nil
}

func (h *Host) lock(ctx context.Context) (func(), *exit.Error) { return h.acquireLock(ctx, true) }

func (h *Host) acquireLock(ctx context.Context, wait bool) (func(), *exit.Error) {
	if err := os.MkdirAll(h.dir, 0o700); err != nil {
		return nil, exit.Internalf("cannot create the machine directory: %s", err)
	}
	file, err := os.OpenFile(h.path("host.lock"), os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return nil, exit.Internalf("cannot open the machine lock: %s", err)
	}
	err = flock.Exclusive(file)
	if err != nil && wait {
		err = flock.Wait(ctx, file)
	}
	if err != nil {
		file.Close()
		return nil, exit.Named(exit.Unavailable, "machine.busy", "another command is changing the local machine")
	}
	return func() { _ = flock.Release(file); file.Close() }, nil
}

// freePort picks a loopback port below the kernel's ephemeral range that nothing answers on.
func freePort(not int) int {
	for {
		var raw [2]byte
		_, _ = rand.Read(raw[:])
		port := 20000 + (int(raw[0])<<8|int(raw[1]))%12768
		if port == not {
			continue
		}
		connection, err := net.DialTimeout("tcp", "127.0.0.1:"+strconv.Itoa(port), time.Second)
		if err != nil {
			return port
		}
		connection.Close()
	}
}

func fileDigest(path string) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer file.Close()
	hash := sha256.New()
	if _, err := io.Copy(hash, file); err != nil {
		return "", err
	}
	return hex.EncodeToString(hash.Sum(nil)), nil
}

// keepWheels makes the wheel directory hold exactly these files, each written to a temporary
// name and renamed into place, and answers the kept paths the environment installs from.
func (h *Host) keepWheels(sources []string) ([]string, error) {
	dir := h.wheels()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	kept := make([]string, len(sources))
	for i, source := range sources {
		kept[i] = filepath.Join(dir, filepath.Base(source))
		if err := copyFile(source, kept[i], 0o644); err != nil {
			return nil, err
		}
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	for _, entry := range entries {
		if !slices.Contains(kept, filepath.Join(dir, entry.Name())) {
			if err := os.RemoveAll(filepath.Join(dir, entry.Name())); err != nil {
				return nil, err
			}
		}
	}
	return kept, nil
}

func copyFile(source, target string, mode os.FileMode) error {
	input, err := os.Open(source)
	if err != nil {
		return err
	}
	defer input.Close()
	output, err := os.CreateTemp(filepath.Dir(target), ".copy-*")
	if err != nil {
		return err
	}
	staged := output.Name()
	defer os.Remove(staged)
	defer output.Close()
	err = output.Chmod(mode)
	if err == nil {
		_, err = io.Copy(output, input)
		if err == nil {
			err = output.Sync()
		}
		if closeErr := output.Close(); err == nil {
			err = closeErr
		}
	}
	if err != nil {
		_ = os.Remove(staged)
		return err
	}
	return os.Rename(staged, target)
}

func writePrivate(path string, body []byte) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	staged := path + ".new"
	if err := os.WriteFile(staged, body, 0o600); err != nil {
		return err
	}
	return os.Rename(staged, path)
}

func tail(output []byte) string {
	text := strings.TrimSpace(string(output))
	if len(text) > 2000 {
		text = text[len(text)-2000:]
	}
	return text
}

func logSince(path string, offset int64) string {
	file, err := os.Open(path)
	if err != nil {
		return ""
	}
	defer file.Close()
	if _, err := file.Seek(offset, io.SeekStart); err != nil {
		return ""
	}
	var lines []string
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
		if len(lines) > 20 {
			lines = lines[1:]
		}
	}
	return fmt.Sprint(strings.Join(lines, "\n"))
}

// writeRuntimeConfig seeds an absent machine config. Launch never overwrites operator settings.
func (h *Host) writeRuntimeConfig() *exit.Error {
	path := filepath.Join(h.Root(), "etc/cozy/runtime.yaml")
	if _, err := os.Stat(path); err == nil || h.GPUBudget == nil {
		return nil
	} else if !errors.Is(err, os.ErrNotExist) {
		return exit.Internalf("cannot inspect the machine's Runtime config: %s", err)
	}
	body, err := config.RuntimeYAML(h.GPUBudget)
	if err == nil {
		err = os.MkdirAll(filepath.Dir(path), 0o755)
	}
	if err == nil {
		err = os.WriteFile(path, body, 0o644)
	}
	if err != nil {
		return exit.Internalf("cannot write the machine's Runtime config: %s", err)
	}
	return nil
}

// note appends one line to the machine's log, where its Host writes.
func (h *Host) note(line string) {
	log, err := os.OpenFile(h.path("host.log"), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		return
	}
	defer log.Close()
	fmt.Fprintln(log, "cozy:", line)
}
