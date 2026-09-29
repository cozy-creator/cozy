package machines

import (
	"bufio"
	"cmp"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"debug/buildinfo"
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
	// inherited are the locale and trust-store values a Host may carry from its launcher.
	inherited []string
	// GPUBudget is `machine.gpu_budget`, written for the Runtime at every launch.
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
		case "LANG", "LC_ALL", "TZ", "SSL_CERT_FILE", "SSL_CERT_DIR":
			h.inherited = append(h.inherited, value)
		}
	}
	return h
}

func (h *Host) Root() string            { return filepath.Join(h.dir, "root") }
func (h *Host) path(name string) string { return filepath.Join(h.dir, name) }

// The image layout the Host and Runtime share, rooted.
func (h *Host) binary() string {
	path := filepath.Join(h.Root(), "usr/local/bin/cozy-machine")
	if _, err := os.Stat(path); err == nil {
		return path
	}
	return filepath.Join(h.Root(), "usr/local/bin/pod-supervisor") // pre-agent installations
}
func (h *Host) python() string { return filepath.Join(h.Root(), "opt/cozy/python") }

// wheels holds exactly the Runtime and TensorFS wheels the machine installed from (none after
// a published install): the Runtime's --find-links for a package's SDK. A pod's image links
// /opt/cozy/wheels to /var/lib/cozy/dev/current, the pair dev/update.py last installed.
func (h *Host) wheels() string { return filepath.Join(h.Root(), "opt/cozy/wheels") }
func (h *Host) readinessEnvelope() string {
	return filepath.Join(h.Root(), "run/cozy/bootstrap/readiness-envelope.json")
}

// RuntimeFloor is the first Runtime with the single-agent supervisor launch contract.
const RuntimeFloor = "0.18.85"

// Source is a machine agent and optional paired development Runtime and TensorFS wheels.
// Pinned retains an explicitly selected agent instead of adopting a published one.
type Source struct {
	Host, RuntimeWheel, TensorFSWheel string
	Pinned                            bool
}

type installedArtifact struct {
	Name   string `json:"name"`
	SHA256 string `json:"sha256"`
	Module string `json:"module,omitempty"` // a Host's Go main module
}

// Installed names the artifacts the root holds.
type Installed struct {
	Host        installedArtifact `json:"host"`
	Runtime     installedArtifact `json:"runtime"`
	TensorFS    installedArtifact `json:"tensorfs"`
	InstalledAt time.Time         `json:"installed_at"`
	HostPinned  bool              `json:"host_pinned,omitempty"`
}

// HostModule is the Go main module a Host binary was built from, "" when it is not Go.
func HostModule(path string) string {
	info, err := buildinfo.ReadFile(path)
	if err != nil {
		return ""
	}
	return info.Main.Path
}

// placeHost atomically installs the independent agent without changing machine state.
func (h *Host) placeHost(source string) (installedArtifact, error) {
	digest, err := fileDigest(source)
	if err != nil {
		return installedArtifact{}, err
	}
	artifact := installedArtifact{Name: filepath.Base(source), SHA256: digest, Module: HostModule(source)}
	target := filepath.Join(h.Root(), "usr/local/bin/cozy-machine")
	if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
		return artifact, err
	}
	return artifact, copyFile(source, target, 0o755)
}

// Adopt migrates an idle legacy installation to the separately released agent. The
// identity, Runtime journal, outputs, Python environment and TensorFS store stay in place.
func (h *Host) Adopt(ctx context.Context, idle bool) (bool, *exit.Error) {
	if !h.Outdated() {
		return false, nil
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	unlock, problem := h.lock(ctx)
	if problem != nil {
		return false, problem
	}
	defer unlock()
	record, problem := h.record()
	if problem != nil {
		return false, problem
	}
	if running := record != nil && h.alive(record.PID); running && (!idle || !h.runtimeIdle()) {
		return false, nil
	}
	return h.adoptLocked(ctx)
}

// runtimeIdle requires a fresh explicit Runtime observation. Missing/stale evidence never
// authorizes replacing a process that may be serving another controller's accepted work.
func (h *Host) runtimeIdle() bool {
	path := filepath.Join(h.Root(), "run/cozy/bootstrap/worker-activity")
	info, err := os.Stat(path)
	if err != nil || time.Since(info.ModTime()) > 10*time.Second || info.ModTime().After(time.Now().Add(10*time.Second)) {
		return false
	}
	raw, err := os.ReadFile(path)
	var state struct {
		ActiveWork *bool    `json:"active_work"`
		Holding    []string `json:"holding"`
	}
	return err == nil && json.Unmarshal(raw, &state) == nil && state.ActiveWork != nil && !*state.ActiveWork && len(state.Holding) == 0
}

// Outdated only identifies the legacy embedded Host. Agent upgrades are explicit;
// updating the CLI does not replace a running machine server.
func (h *Host) Outdated() bool {
	installed, problem := h.Installed()
	return problem == nil && installed != nil && !installed.HostPinned &&
		cmp.Or(installed.Host.Module, HostModule(h.binary())) != AgentModule
}

func (h *Host) adoptLocked(ctx context.Context) (bool, *exit.Error) {
	if !h.Outdated() {
		return false, nil
	}
	installed, problem := h.Installed()
	if problem != nil {
		return false, problem
	}
	source, problem := h.PublishedAgent(ctx)
	if problem != nil {
		return false, problem
	}
	if problem := h.stopLocked(ctx); problem != nil {
		return false, problem
	}
	artifact, err := h.placeHost(source)
	if err != nil {
		return false, exit.Internalf("cannot install the machine agent: %s", err)
	}
	installed.Host, installed.HostPinned = artifact, false
	raw, _ := json.MarshalIndent(installed, "", "  ")
	if err := writePrivate(h.path("installed.json"), raw); err != nil {
		return false, exit.Internalf("cannot record the machine installation: %s", err)
	}
	h.note("adopted independent machine agent; retained machine identity and execution state")
	return true, nil
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

// Install lays the root out as a pod image does: the Host, a Python 3.12 environment
// holding the Runtime and TensorFS wheels, and the three executables the image bakes. A
// running idle agent is stopped first; the next use launches the new one.
func (h *Host) Install(ctx context.Context, source Source, uv string) (*Installed, *exit.Error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	unlock, problem := h.lock(ctx)
	if problem != nil {
		return nil, problem
	}
	defer unlock()
	if record, problem := h.record(); problem != nil {
		return nil, problem
	} else if record != nil && h.alive(record.PID) && !h.runtimeIdle() {
		return nil, exit.Named(exit.Conflict, "machine.busy", "the machine is running work or has not confirmed that it is idle").
			WithRemedy("wait for accepted work to finish before installing; use machine stop only to explicitly stop the machine")
	}
	if _, problem := h.identity(); problem != nil {
		return nil, problem
	}
	installed := Installed{InstalledAt: time.Now().UTC()}
	if source.Host == "" || (source.RuntimeWheel == "") != (source.TensorFSWheel == "") ||
		source.RuntimeWheel != "" && (!strings.HasSuffix(source.RuntimeWheel, ".whl") || !strings.HasSuffix(source.TensorFSWheel, ".whl")) {
		return nil, exit.New(exit.Validation, "the machine install names the Host binary, and both wheels or neither")
	}
	installed.HostPinned = source.Pinned
	for _, artifact := range []struct {
		path string
		out  *installedArtifact
	}{{source.RuntimeWheel, &installed.Runtime}, {source.TensorFSWheel, &installed.TensorFS}} {
		if artifact.path == "" {
			continue
		}
		digest, err := fileDigest(artifact.path)
		if err != nil {
			return nil, exit.New(exit.NotFound, "cannot read %s: %s", artifact.path, err)
		}
		*artifact.out = installedArtifact{Name: filepath.Base(artifact.path), SHA256: digest}
	}
	if problem := h.stopLocked(ctx); problem != nil {
		return nil, problem
	}
	if err := os.Remove(h.path("installed.json")); err != nil && !errors.Is(err, os.ErrNotExist) {
		return nil, exit.Internalf("cannot retire the previous machine installation: %s", err)
	}
	root := h.Root()
	for _, dir := range []string{"usr/local/bin", "opt/cozy/bin"} {
		if err := os.MkdirAll(filepath.Join(root, dir), 0o755); err != nil {
			return nil, exit.Internalf("cannot lay out the machine root: %s", err)
		}
	}
	host, err := h.placeHost(source.Host)
	if err != nil {
		return nil, exit.Internalf("cannot install the machine Host: %s", err)
	}
	installed.Host = host
	var wheels []string
	if source.RuntimeWheel != "" {
		wheels = []string{source.RuntimeWheel, source.TensorFSWheel}
	}
	wheels, err = h.keepWheels(wheels)
	if err != nil {
		return nil, exit.Internalf("cannot keep the machine's wheels: %s", err)
	}
	// A venv is not relocatable, so it is rebuilt in place.
	if err := os.RemoveAll(h.python()); err != nil {
		return nil, exit.Internalf("cannot replace the machine Python environment: %s", err)
	}
	if output, err := exec.CommandContext(ctx, uv, "venv", "--no-config", "--no-project", "--python", "3.12", h.python()).CombinedOutput(); err != nil {
		return nil, exit.New(exit.Structural, "cannot create the machine Python environment: %s", tail(output))
	}
	// Without wheels, the published Runtime and the TensorFS it depends on.
	// The worker's base is the CPU image's: the Runtime with its media extra. Every
	// package environment carries its own framework closure.
	// An explicit install means the newest release: refresh just these two from the index, not uv's cache.
	requirements := []string{"--refresh-package", hostruntime.Distribution, "--refresh-package", "tensorfs", hostruntime.Distribution + "[media]>=" + RuntimeFloor}
	if len(wheels) > 0 {
		requirements = []string{hostruntime.Distribution + "[media] @ file://" + wheels[0], wheels[1]}
	}
	python := filepath.Join(h.python(), "bin/python")
	install := exec.CommandContext(ctx, uv, append([]string{"pip", "install", "--no-config", "--python", python}, requirements...)...)
	if output, err := install.CombinedOutput(); err != nil {
		return nil, exit.New(exit.Structural, "cannot install the Runtime and TensorFS: %s", tail(output))
	}
	for distribution, artifact := range map[string]*installedArtifact{hostruntime.Distribution: &installed.Runtime, "tensorfs": &installed.TensorFS} {
		if artifact.Name != "" {
			continue
		}
		version, err := exec.CommandContext(ctx, python, "-c", "import importlib.metadata as m, sys; print(m.version(sys.argv[1]))", distribution).Output()
		if err != nil {
			return nil, exit.New(exit.Structural, "the machine environment has no %s", distribution)
		}
		artifact.Name = distribution + " " + strings.TrimSpace(string(version))
	}
	for link, target := range map[string]string{
		"opt/cozy/bin/cozy-runtime-worker": filepath.Join(h.python(), "bin/cozy-runtime-worker"),
		"usr/local/bin/tfs":                filepath.Join(h.python(), "bin/tfs"),
		"usr/local/bin/uv":                 uv,
	} {
		path := filepath.Join(root, link)
		if _, err := os.Stat(target); err != nil {
			return nil, exit.New(exit.Structural, "the installed machine has no %s: %s", filepath.Base(target), err)
		}
		_ = os.Remove(path)
		if err := os.Symlink(target, path); err != nil {
			return nil, exit.Internalf("cannot link %s: %s", link, err)
		}
	}
	raw, _ := json.MarshalIndent(installed, "", "  ")
	if err := writePrivate(h.path("installed.json"), raw); err != nil {
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
	Leaf             []byte
	GPUs             []ReceiptGPU
	PID              int
	// Reads is the origin the machine reads the asked hub at when that is not its own
	// hub: each run of it names this (wire 67). "" at its own hub.
	Reads string
}

func (l *Launch) at(reads string) *Launch {
	if l == nil {
		return nil
	}
	out := *l
	out.Reads = reads
	return &out
}

// Ensure starts the machine independently of any Hub. A named Hub receives a scoped
// execution credential only after the agent proves its TLS identity; it never owns this
// machine's identity or lifecycle. Empty hubOrigin is entirely offline.
func (h *Host) Ensure(ctx context.Context, hubOrigin string, client *hub.Client, _ bool) (*Launch, *exit.Error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	unlock, problem := h.lock(ctx)
	if problem != nil {
		return nil, problem
	}
	defer unlock()
	var launch *Launch
	if cached := h.cached; cached != nil && h.alive(cached.PID) {
		launch = cached.Launch
	}
	if launch == nil {
		record, problem := h.record()
		if problem != nil {
			return nil, problem
		}
		if record != nil && h.alive(record.PID) {
			launch, problem = h.await(ctx, record)
			if problem != nil {
				return nil, problem
			}
			h.remember(launch, *record)
		} else {
			launch, problem = h.launchLocked(ctx)
			if problem != nil {
				return nil, problem
			}
		}
	}
	if hubOrigin == "" {
		return launch.at(""), nil
	}
	reads, problem := h.attachAccess(ctx, launch, hubOrigin, client)
	return launch.at(reads), problem
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
		receipt, leaf, err := readReceipt(ctx, record.MediaPort, key)
		if err == nil {
			if receipt.WorkerInternalPort != record.WorkerPort {
				return nil, exit.New(exit.Conflict, "the machine Host's receipt names another worker port")
			}
			if err := writePrivate(h.path("leaf.pem"), pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: leaf})); err != nil {
				return nil, exit.Internalf("cannot record the machine's TLS leaf: %s", err)
			}
			return &Launch{Addr: "127.0.0.1:" + strconv.Itoa(record.WorkerPort), MediaAddr: "127.0.0.1:" + strconv.Itoa(record.MediaPort),
				WorkerID: record.WorkerID, BootID: receipt.PodBootID, AgentVersion: receipt.MachineVersion, Leaf: leaf, GPUs: receipt.RuntimeGPUs, PID: record.PID}, nil
		}
		if refused := (*receiptRefusal)(nil); errors.As(err, &refused) {
			return nil, exit.New(exit.Credential, "the machine Host's readiness receipt did not verify: %s", err)
		}
		if gone := (*runtimeGone)(nil); errors.As(err, &gone) {
			return nil, exit.Named(exit.Structural, "machine.runtime_gone", "%s", gone)
		}
		if !h.alive(record.PID) {
			return nil, exit.Named(exit.Structural, "machine.host_exited", "the machine Host exited before readiness")
		}
		select {
		case <-ctx.Done():
			return nil, exit.Named(exit.Unavailable, "machine.host_starting", "the machine Host is still starting")
		case <-time.After(250 * time.Millisecond):
		}
	}
}

func (h *Host) launchLocked(ctx context.Context) (*Launch, *exit.Error) {
	id, problem := h.identity()
	if problem != nil {
		return nil, problem
	}
	// Nothing runs: migrate the legacy embedded Host before the next launch.
	if _, problem := h.adoptLocked(ctx); problem != nil {
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
	mediaToken, problem := h.secret("media-token")
	if problem != nil {
		return nil, problem
	}
	mediaHash := sha256.Sum256([]byte(mediaToken))
	auth, _ := json.Marshal(map[string]any{
		"control_public_key_ed25519_b64url": owner.PublicKey(),
		"media_token_sha256":                []string{hex.EncodeToString(mediaHash[:])},
	})
	key := make([]byte, 32)
	if _, err := rand.Read(key); err != nil {
		return nil, exit.Internalf("cannot mint the machine receipt key: %s", err)
	}
	// A retained envelope is the previous launch's; this launch proves itself under a new key.
	if err := os.Remove(h.readinessEnvelope()); err != nil && !errors.Is(err, os.ErrNotExist) {
		return nil, exit.Internalf("cannot retire the previous readiness receipt: %s", err)
	}
	base := append([]string{
		"COZY_MACHINE_ROOT=" + h.Root(),
		"COZY_LISTEN_HOST=127.0.0.1",
		"COZY_WORKER_ID=" + id,
		"COZY_MACHINE_LIFETIME=persistent",
		"COZY_BOOTSTRAP_RECEIPT_HMAC_KEY_B64URL=" + base64.RawURLEncoding.EncodeToString(key),
		"COZY_RECORD_OWNER_AUTH_JSON=" + string(auth),
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
	record.ReceiptKey = base64.RawURLEncoding.EncodeToString(key)
	raw, _ := json.Marshal(record)
	if err := writePrivate(h.path("host.json"), raw); err != nil {
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
		unit := userunit.Name("cozy-machine", h.dir, false)
		_ = userunit.Stop(unit) // a Host this record does not name
		if _, err := userunit.Start(userunit.Spec{Unit: unit, Argv: []string{h.binary()}, Env: env, Dir: h.Root(),
			Stdout: log.Name(), Stderr: log.Name()}); err != nil {
			return 0, err
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
	return h.stopLocked(ctx)
}

func (h *Host) stopLocked(ctx context.Context) *exit.Error {
	h.cached = nil
	record, problem := h.record()
	if problem != nil || record == nil {
		return problem
	}
	if h.alive(record.PID) {
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
	if err := os.Remove(h.path("host.json")); err != nil && !errors.Is(err, os.ErrNotExist) {
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
	raw, err := os.ReadFile(h.path("host.json"))
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
	return &record, nil
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
	for _, name := range []string{"cozy-machine", "pod-supervisor", "cozy"} {
		binary, err := filepath.EvalSymlinks(filepath.Join(h.Root(), "usr/local/bin", name))
		if err == nil && strings.TrimSuffix(target, " (deleted)") == binary {
			return true
		}
	}
	return false
}

func (h *Host) lock(ctx context.Context) (func(), *exit.Error) {
	if err := os.MkdirAll(h.dir, 0o700); err != nil {
		return nil, exit.Internalf("cannot create the machine directory: %s", err)
	}
	file, err := os.OpenFile(h.path("host.lock"), os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return nil, exit.Internalf("cannot open the machine lock: %s", err)
	}
	if err := flock.Wait(ctx, file); err != nil {
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
	staged := target + ".new"
	output, err := os.OpenFile(staged, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, mode)
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

// writeRuntimeConfig gives the Runtime this computer's GPU budget: gpu.budget in the machine
// root's etc/cozy/runtime.yaml, or no file when none is configured.
func (h *Host) writeRuntimeConfig() *exit.Error {
	path := filepath.Join(h.Root(), "etc/cozy/runtime.yaml")
	if h.GPUBudget == nil {
		if err := os.Remove(path); err != nil && !errors.Is(err, os.ErrNotExist) {
			return exit.Internalf("cannot clear the machine's Runtime config: %s", err)
		}
		return nil
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
