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
func (h *Host) binary() string { return filepath.Join(h.Root(), "usr/local/bin/cozy-machine") }
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

// Source optionally pins an agent and paired development Runtime and TensorFS wheels.
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

// startupPolicy preserves an explicitly installed Host or wheel pair.
// The published agent hash alone does not pin the installation.
func (i Installed) startupPolicy() []byte {
	mode := "auto"
	agent := "bundled"
	if i.HostPinned {
		agent = "explicit"
	}
	if i.HostPinned || i.Runtime.SHA256 != "" || i.TensorFS.SHA256 != "" {
		mode = "off"
	}
	body, _ := json.Marshal(struct {
		Mode  string `json:"startup_update"`
		Agent string `json:"agent"`
	}{mode, agent})
	return body
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

// Outdated identifies legacy Hosts and unpinned copies shadowing a wheel-owned
// agent. Adoption still requires an idle machine; CLI updates do not replace it.
func (h *Host) Outdated() bool {
	installed, problem := h.Installed()
	if problem != nil || installed == nil || installed.HostPinned {
		return false
	}
	if bundle, err := os.Stat(h.bundledAgent()); err == nil && HostModule(h.bundledAgent()) == AgentModule {
		current, err := os.Stat(h.binary())
		return err != nil || !os.SameFile(bundle, current)
	}
	return cmp.Or(installed.Host.Module, HostModule(h.binary())) != AgentModule
}

func (h *Host) adoptLocked(ctx context.Context) (bool, *exit.Error) {
	if !h.Outdated() {
		return false, nil
	}
	installed, problem := h.Installed()
	if problem != nil {
		return false, problem
	}
	source, problem := h.defaultAgent(ctx)
	if problem != nil {
		return false, problem
	}

	var artifact installedArtifact
	var err error
	if source == h.bundledAgent() {
		artifact, err = h.linkAgent()
	} else {
		artifact, err = h.placeHost(source)
	}
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
// running idle agent is stopped first; the next use launches the new one. Existing
// base packages (including the machine's PyTorch/CUDA closure) remain installed.
func (h *Host) Install(ctx context.Context, source Source, uv string) (*Installed, *exit.Error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	unlock, problem := h.lock(ctx)
	if problem != nil {
		return nil, problem
	}
	defer unlock()
	record, problem := h.record()
	if problem != nil {
		return nil, problem
	}
	previous, problem := h.Installed()
	if problem != nil {
		return nil, problem
	}
	if previous == nil && record != nil && h.alive(record.PID) {
		return nil, exit.Named(exit.Conflict, "machine.installation_unreadable", "the running machine has no installation metadata; repair it through its maintenance API without replacing its process")
	}
	installed := Installed{InstalledAt: time.Now().UTC()}
	if (source.RuntimeWheel == "") != (source.TensorFSWheel == "") ||
		source.RuntimeWheel != "" && (!strings.HasSuffix(source.RuntimeWheel, ".whl") || !strings.HasSuffix(source.TensorFSWheel, ".whl")) {
		return nil, exit.New(exit.Validation, "the machine install names both wheels or neither, with an optional Host binary")
	}
	if source.Host != "" && !compatibleAgent(ctx, source.Host) {
		return nil, exit.Named(exit.Structural, "machine.agent_update_required", "the selected agent must advertise %s, %s and %s", HubAccessCapability, RuntimeUpdateCapability, BootstrapCapability)
	}
	if previous != nil {
		return h.updateLocked(ctx, source)
	}
	release, problem := h.bootstrapLease()
	if problem != nil {
		return nil, problem
	}
	defer release()
	if _, problem := h.identity(); problem != nil {
		return nil, problem
	}

	installed.HostPinned = source.Pinned
	hostName := filepath.Base(source.Host)
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
	if source.Host != "" {
		if _, err := fileDigest(source.Host); err != nil {
			return nil, exit.New(exit.NotFound, "cannot read the machine agent: %s", err)
		}
	}
	retain := source.Host
	if retain != "" {
		// Preserve an explicitly selected executable before its wheel is replaced.
		bundle, bundleErr := os.Stat(h.bundledAgent())
		current, currentErr := os.Stat(retain)
		if bundleErr == nil && currentErr == nil && os.SameFile(bundle, current) {
			file, err := os.CreateTemp(h.dir, ".agent-before-*")
			if err != nil {
				return nil, exit.Internalf("cannot retain the current agent: %s", err)
			}
			file.Close()
			defer os.Remove(file.Name())
			if err := copyFile(retain, file.Name(), 0755); err != nil {
				return nil, exit.Internalf("cannot retain the current agent: %s", err)
			}
			source.Host = file.Name()
		}
	}
	root := h.Root()
	for _, dir := range []string{"usr/local/bin", "opt/cozy/bin"} {
		if err := os.MkdirAll(filepath.Join(root, dir), 0o755); err != nil {
			return nil, exit.Internalf("cannot lay out the machine root: %s", err)
		}
	}
	var wheels []string
	if source.RuntimeWheel != "" {
		wheels = []string{source.RuntimeWheel, source.TensorFSWheel}
	}
	python := filepath.Join(h.python(), "bin/python")
	if _, err := os.Lstat(h.python()); errors.Is(err, os.ErrNotExist) {
		if output, err := exec.CommandContext(ctx, uv, "venv", "--no-config", "--no-project", "--python", "3.12", h.python()).CombinedOutput(); err != nil {
			return nil, exit.New(exit.Structural, "cannot create the machine Python environment: %s", tail(output))
		}
	} else if err != nil {
		return nil, exit.Internalf("cannot read the machine Python environment: %s", err)
	} else if output, err := exec.CommandContext(ctx, python, "-I", "-c", "import sys; assert sys.version_info[:2] == (3, 12); assert sys.prefix != sys.base_prefix").CombinedOutput(); err != nil {
		return nil, exit.New(exit.Structural, "the existing machine Python environment is unusable; it was preserved: %s", tail(output))
	}
	// Without wheels, the published Runtime and the TensorFS it depends on.
	// The worker's base is the CPU image's: the Runtime with its media extra. Every
	// package environment carries its own framework closure. Existing machine base
	// packages are retained: installing a Runtime is not an environment sync.
	// An explicit published install selects the newest pair even in an existing venv.
	requirements := []string{"--upgrade-package", hostruntime.Distribution, "--upgrade-package", "tensorfs", hostruntime.Distribution + "[media]>=" + RuntimeFloor}
	if len(wheels) > 0 {
		requirements = []string{hostruntime.Distribution + "[media] @ file://" + wheels[0], wheels[1]}
	}
	if err := installMachinePair(ctx, uv, python, requirements); err != nil {
		return nil, exit.New(exit.Structural, "%s", err)
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
	selected := source.Host
	if selected == "" {
		selected, problem = h.defaultAgent(ctx)
		if problem != nil {
			return nil, problem
		}
	}
	var host installedArtifact
	var err error
	if source.Host == "" && selected == h.bundledAgent() {
		host, err = h.linkAgent()
	} else {
		host, err = h.placeHost(selected)
		if source.Host == "" {
			host.Name = "cozy-machine"
		} else {
			host.Name = hostName
		}
	}
	if err != nil {
		return nil, exit.Internalf("cannot install the machine Host: %s", err)
	}
	installed.Host = host
	if _, err := h.keepWheels(wheels); err != nil {
		return nil, exit.Internalf("cannot keep the machine's wheels: %s", err)
	}
	if err := writePrivate(filepath.Join(h.Root(), "etc/cozy/software-policy.json"), installed.startupPolicy()); err != nil {
		return nil, exit.Internalf("cannot record the initial software policy: %s", err)
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

// Ensure starts the machine independently of any Hub, or attaches to the running one. A named
// Hub receives a scoped execution credential only after the agent proves its TLS identity; it
// never owns this machine's identity or lifecycle. Empty hubOrigin is entirely offline.
func (h *Host) Ensure(ctx context.Context, hubOrigin string, client *hub.Client, _ bool) (*Launch, *exit.Error) {
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
	return h.ensureLocked(ctx, hubOrigin, client, !attach)
}

// Start starts the machine, or attaches to the running one, after any command changing it.
func (h *Host) Start(ctx context.Context) (*Launch, *exit.Error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	unlock, problem := h.lock(ctx)
	if problem != nil {
		return nil, problem
	}
	defer unlock()
	return h.ensureLocked(ctx, "", nil, true)
}

func (h *Host) ensureLocked(ctx context.Context, hubOrigin string, client *hub.Client, start bool) (*Launch, *exit.Error) {
	var launch *Launch
	if cached := h.cached; cached != nil && h.alive(cached.PID) && (cached.record.StartTicks == 0 || cached.record.StartTicks == processStartTicks(cached.PID)) {
		launch = cached.Launch
	}
	if launch == nil {
		record, problem := h.running()
		switch {
		case problem != nil:
			return nil, problem
		case record != nil:
			if launch, problem = h.await(ctx, record); problem != nil {
				return nil, problem
			}
			h.remember(launch, *record)
		case !start:
			return nil, stopped()
		default:
			if launch, problem = h.launchLocked(ctx); problem != nil {
				return nil, problem
			}
		}
	}
	h.ResumeExecutionAccessCleanup(ctx, launch)
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
	// A directly launched agent or surviving Runtime may have no client record.
	// Observe its kernel ownership before touching receipt or launch metadata.
	release, problem := h.bootstrapLease()
	if problem != nil {
		return nil, problem
	}
	release()

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
	if err := writePrivate(h.path("receipt-key"), []byte(base64.RawURLEncoding.EncodeToString(key))); err != nil {
		return nil, exit.Internalf("cannot retain the machine receipt key: %s", err)
	}
	base := append([]string{
		"COZY_MACHINE_ROOT=" + h.Root(),
		"COZY_LISTEN_HOST=127.0.0.1",
		"COZY_WORKER_ID=" + id,
		"COZY_MACHINE_LIFETIME=persistent",
		"COZY_BOOTSTRAP_RECEIPT_HMAC_KEY_FILE=" + h.path("receipt-key"),
		"COZY_RECORD_OWNER_AUTH_JSON=" + string(auth),
	}, h.inherited...)
	if err := h.writeStartupPolicy(*installed); err != nil {
		return nil, exit.Internalf("cannot record the machine update policy: %s", err)
	}
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
	if userunit.Available() && userunit.MainPID(userunit.Name("cozy-machine", h.dir, false)) > 0 {
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
func (h *Host) Unit() string { return userunit.Name("cozy-machine-agent", h.dir, false) }

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
	for _, relative := range []string{"usr/local/bin/cozy-machine", "opt/cozy/python/bin/cozy-machine"} {
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

// A missing client record cannot override kernel-owned machine/Runtime leases.
func (h *Host) bootstrapLease() (func(), *exit.Error) {
	var files []*os.File
	release := func() {
		for _, file := range files {
			_ = flock.Release(file)
			_ = file.Close()
		}
	}
	for _, relative := range []string{"var/lib/cozy/machine/agent.lock", "run/cozy/worker/worker.lock"} {
		path := filepath.Join(h.Root(), relative)
		if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
			release()
			return nil, exit.Internalf("cannot inspect machine ownership: %s", err)
		}
		file, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0600)
		if err != nil {
			release()
			return nil, exit.Internalf("cannot inspect machine ownership: %s", err)
		}
		if err := flock.Exclusive(file); err != nil {
			file.Close()
			release()
			return nil, exit.Named(exit.Conflict, "machine.busy", "a live machine or Runtime owns this root; bootstrap cannot replace its software")
		}
		files = append(files, file)
	}
	return release, nil
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

// Only an explicit installation changes the operator's update policy. A missing
// policy is seeded from existing installation provenance once.
func (h *Host) writeStartupPolicy(installed Installed) error {
	path := filepath.Join(h.Root(), "etc/cozy/software-policy.json")
	if _, err := os.Stat(path); err == nil {
		return nil
	} else if !errors.Is(err, os.ErrNotExist) {
		return err
	}
	return writePrivate(path, installed.startupPolicy())
}
