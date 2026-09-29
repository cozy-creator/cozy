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
	"maps"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"sort"
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
)

// Host is this computer's machine: the literal pod-supervisor a rented pod runs, launched
// detached under a root laid out as the pod's `/`. It outlives the daemon as a pod outlives
// its controller, and it exits by itself after the same fixed idle period; the next use
// launches it again.
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
func (h *Host) binary() string { return filepath.Join(h.Root(), "usr/local/bin/pod-supervisor") }
func (h *Host) python() string { return filepath.Join(h.Root(), "opt/cozy/python") }

// wheels holds exactly the Runtime and TensorFS wheels the machine installed from (none after
// a published install): the Runtime's --find-links for a package's SDK. A pod's image links
// /opt/cozy/wheels to /var/lib/cozy/dev/current, the pair dev/update.py last installed.
func (h *Host) wheels() string { return filepath.Join(h.Root(), "opt/cozy/wheels") }
func (h *Host) readinessEnvelope() string {
	return filepath.Join(h.Root(), "run/cozy/bootstrap/readiness-envelope.json")
}

// RuntimeFloor is the oldest published Runtime proven as a rooted machine worker.
const RuntimeFloor = "0.18.53"

// Source is one cohort's worker artifacts: the Host binary and the Runtime and TensorFS
// wheels a pod image carries. Without wheels, the published Runtime is installed. The Host is
// the cozy binary itself; Pinned keeps a Host named for development in place of the cozy that
// runs this machine later (Adopt).
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

// CozyModule is the Go module of the cozy binary, which is also every machine's Host.
const CozyModule = "github.com/cozy-creator/cozy"

// HostModule is the Go main module a Host binary was built from, "" when it is not Go.
func HostModule(path string) string {
	info, err := buildinfo.ReadFile(path)
	if err != nil {
		return ""
	}
	return info.Main.Path
}

// Self is the cozy binary this process runs, the Host it installs and adopts.
func Self() (string, error) {
	self, err := os.Executable()
	return strings.TrimSuffix(self, " (deleted)"), err
}

var selfDigest = sync.OnceValue(func() string {
	self, err := Self()
	if err != nil {
		return ""
	}
	digest, _ := fileDigest(self)
	return digest
})

// placeHost puts a Host binary where the launcher runs it. The cozy binary is kept as
// usr/local/bin/cozy and runs as pod-supervisor, the name the launch contract starts it by;
// any other Host is pod-supervisor itself.
func (h *Host) placeHost(source string) (installedArtifact, error) {
	digest, err := fileDigest(source)
	if err != nil {
		return installedArtifact{}, err
	}
	artifact := installedArtifact{Name: filepath.Base(source), SHA256: digest, Module: HostModule(source)}
	bin := filepath.Join(h.Root(), "usr/local/bin")
	if err := os.MkdirAll(bin, 0o755); err != nil {
		return artifact, err
	}
	if artifact.Module != CozyModule {
		_ = os.Remove(filepath.Join(bin, "cozy"))
		return artifact, copyFile(source, h.binary(), 0o755)
	}
	if err := copyFile(source, filepath.Join(bin, "cozy"), 0o755); err != nil {
		return artifact, err
	}
	link := h.binary() + ".new"
	_ = os.Remove(link)
	if err := os.Symlink("cozy", link); err != nil {
		return artifact, err
	}
	return artifact, os.Rename(link, h.binary())
}

// Adopt makes this cozy binary the machine's Host when the installed Host is another program
// (the tensorhub pod-supervisor, before one binary) or another cozy build, unless a
// development install pinned it. A running Host is replaced only when the machine is idle, stopped by
// its recorded PID; the next use launches this one. Runs, installs and the TensorFS store stay.
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
	if running := record != nil && h.alive(record.PID); running && !idle {
		return false, nil
	}
	return h.adoptLocked(ctx)
}

// Outdated says whether Adopt would replace the installed Host: it is not this cozy, and no
// development install pinned it.
func (h *Host) Outdated() bool {
	installed, problem := h.Installed()
	if problem != nil || installed == nil || installed.HostPinned || selfDigest() == "" {
		return false
	}
	return installed.Host.SHA256 != selfDigest()
}

func (h *Host) adoptLocked(ctx context.Context) (bool, *exit.Error) {
	if !h.Outdated() {
		return false, nil
	}
	installed, problem := h.Installed()
	if problem != nil {
		return false, problem
	}
	module := cmp.Or(installed.Host.Module, HostModule(h.binary()))
	self, err := Self()
	if err != nil {
		return false, nil
	}
	if problem := h.stopLocked(ctx); problem != nil {
		return false, problem
	}
	was := installed.Host.Name
	if installed.Host, err = h.placeHost(self); err != nil {
		return false, exit.Internalf("cannot install cozy as the machine Host: %s", err)
	}
	installed.HostPinned = false
	raw, _ := json.MarshalIndent(installed, "", "  ")
	if err := writePrivate(h.path("installed.json"), raw); err != nil {
		return false, exit.Internalf("cannot record the machine installation: %s", err)
	}
	h.note(fmt.Sprintf("the machine's Host is now %s (was %s %s)", self, was, module))
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
// running Host is stopped first; the next use launches the new one.
func (h *Host) Install(ctx context.Context, source Source, uv string) (*Installed, *exit.Error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	unlock, problem := h.lock(ctx)
	if problem != nil {
		return nil, problem
	}
	defer unlock()
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

type registration struct {
	Hub         string            `json:"hub"`
	ID          string            `json:"id"`
	WorkerToken string            `json:"worker_token"`
	Environment map[string]string `json:"environment,omitempty"`
}

// hostRecord is one launched Host, as a rental row records its pod.
type hostRecord struct {
	PID        int    `json:"pid"`
	Hub        string `json:"hub"`
	WorkerID   string `json:"worker_id"`
	WorkerPort int    `json:"worker_port"`
	MediaPort  int    `json:"media_port"`
	ReceiptKey string `json:"receipt_key"`
	// Hubs are the hubs it holds a registration for, each with the origin its Runtime
	// reads that hub at; WireMinor is the top of its protocol range once dialed.
	Hubs      map[string]string `json:"hubs,omitempty"`
	WireMinor uint32            `json:"wire_minor,omitempty"`
}

// HubPerRunWire is the wire minor from which a machine reads each run at the hub the run
// names (ReleaseRoot.hub and the rest), instead of only at the hub it was launched for.
const HubPerRunWire = 67

// serves says whether the Host runs hub's work where it is: at its own hub, or, for work
// whose every call names its hub (named), at another it holds a registration for once it
// reads a run's hub. reads is what such a call names.
func (r *hostRecord) serves(hub string, named bool) (reads string, ok bool) {
	if r.Hub == hub {
		return "", true
	}
	reads = r.Hubs[hub]
	return reads, named && reads != "" && r.WireMinor >= HubPerRunWire
}

type cachedLaunch struct {
	*Launch
	record hostRecord
}

// Launch is the running Host's identity once its readiness receipt verified.
type Launch struct {
	Addr, MediaAddr  string
	WorkerID, BootID string
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

// Ensure answers the running Host, launching it when none runs. hubOrigin is the hub whose
// work it is asked for, client speaks to that hub as the signed-in user, and named says that
// work names its hub on every call. A Host that cannot do that work where it is moves there.
func (h *Host) Ensure(ctx context.Context, hubOrigin string, client *hub.Client, named bool) (*Launch, *exit.Error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if cached := h.cached; cached != nil && h.alive(cached.PID) {
		if reads, ok := cached.record.serves(hubOrigin, named); ok {
			return cached.at(reads), nil
		}
	}
	unlock, problem := h.lock(ctx)
	if problem != nil {
		return nil, problem
	}
	defer unlock()
	record, problem := h.record()
	if problem != nil {
		return nil, problem
	}
	if record != nil && h.alive(record.PID) {
		if reads, ok := record.serves(hubOrigin, named); !ok {
			// One root holds one Host: the machine moves to the selected hub.
			if problem := h.stopLocked(ctx); problem != nil {
				return nil, problem
			}
		} else if launch, problem := h.await(ctx, record); problem == nil || h.alive(record.PID) {
			return h.remember(launch, *record).at(reads), problem
		}
	}
	return h.launchLocked(ctx, hubOrigin, client)
}

func (h *Host) remember(launch *Launch, record hostRecord) *Launch {
	if launch != nil {
		h.cached = &cachedLaunch{Launch: launch, record: record}
	}
	return launch
}

// Dialed records the running Host's protocol range, which says whether it reads a run's
// hub where it is.
func (h *Host) Dialed(ctx context.Context, pid int, wireMinor uint32) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if cached := h.cached; cached != nil && cached.PID == pid && cached.record.WireMinor == wireMinor {
		return
	}
	unlock, problem := h.lock(ctx)
	if problem != nil {
		return
	}
	defer unlock()
	record, problem := h.record()
	if problem != nil || record == nil || record.PID != pid {
		return
	}
	if record.WireMinor != wireMinor {
		record.WireMinor = wireMinor
		raw, _ := json.Marshal(record)
		if writePrivate(h.path("host.json"), raw) != nil {
			return
		}
	}
	if h.cached != nil && h.cached.PID == pid {
		h.cached.record = *record
	}
}

// Serves says whether the running Host does hub's work without moving; false when none runs.
func (h *Host) Serves(hub string, named bool) bool {
	record, problem := h.record()
	if problem != nil || record == nil || !h.alive(record.PID) {
		return false
	}
	_, ok := record.serves(hub, named)
	return ok
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
				WorkerID: record.WorkerID, BootID: receipt.PodBootID, Leaf: leaf, GPUs: receipt.RuntimeGPUs, PID: record.PID}, nil
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

func (h *Host) launchLocked(ctx context.Context, hubOrigin string, client *hub.Client) (*Launch, *exit.Error) {
	// Nothing runs: the cozy binary takes over from any other Host first.
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
	registered, problem := h.registration(ctx, hubOrigin, client)
	if problem != nil {
		return nil, problem
	}
	all, problem := h.registrations()
	if problem != nil {
		return nil, problem
	}
	environment := registered.Environment
	if len(environment) == 0 {
		return nil, exit.Named(exit.Unavailable, "machine.environment_unavailable", "this machine's registration with %s recorded no hub environment", hubOrigin).
			WithRemedy("re-register it: remove %s and run it again", h.path("registrations.json"))
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
		"COZY_WORKER_ID=" + registered.ID,
		"COZY_WORKER_AUTH_TOKEN=" + registered.WorkerToken,
		"COZY_BOOTSTRAP_RECEIPT_HMAC_KEY_B64URL=" + base64.RawURLEncoding.EncodeToString(key),
		"COZY_RECORD_OWNER_AUTH_JSON=" + string(auth),
	}, h.inherited...)
	if h.store != "" {
		base = append(base, "COZY_TENSORFS_ROOT="+h.store)
	}
	names := make([]string, 0, len(environment))
	for name := range environment {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		base = append(base, name+"="+environment[name])
	}
	// Every other registration rides along: a machine that reads a run's hub (wire 67)
	// serves each of them without moving; an older one ignores them.
	held := map[string]string{hubOrigin: environment["TENSORHUB_ORIGIN"]}
	var others []map[string]any
	for _, origin := range slices.Sorted(maps.Keys(all)) {
		other := all[origin]
		if origin == hubOrigin || other.ID == "" || other.Environment["TENSORHUB_ORIGIN"] == "" {
			continue
		}
		held[origin] = other.Environment["TENSORHUB_ORIGIN"]
		others = append(others, map[string]any{"worker_id": other.ID, "worker_token": other.WorkerToken, "environment": other.Environment})
	}
	if len(others) > 0 {
		raw, _ := json.Marshal(others)
		base = append(base, "COZY_MACHINE_HUBS_JSON="+string(raw))
	}
	// Free ports are chosen, not reserved: a port taken before the Host binds it ends
	// that Host before readiness, and the next launch chooses again.
	for attempt := 0; ; attempt++ {
		launch, record, problem := h.start(ctx, base, key, hostRecord{Hub: hubOrigin, WorkerID: registered.ID, Hubs: held})
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
	Hub       string
	PID       int
	Running   bool
}

func (h *Host) Status() (Status, *exit.Error) {
	installed, problem := h.Installed()
	if problem != nil {
		return Status{}, problem
	}
	out := Status{Installed: installed}
	registered, problem := h.registrations()
	if problem != nil {
		return Status{}, problem
	}
	record, problem := h.record()
	if problem != nil {
		return Status{}, problem
	}
	if record != nil {
		out.PID, out.Running = record.PID, h.alive(record.PID)
		out.MachineID, out.Hub = registered[record.Hub].ID, record.Hub
	} else if len(registered) == 1 {
		for hub, one := range registered {
			out.MachineID, out.Hub = one.ID, hub
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

// registrations are this machine's registrations, one per hub, keyed by origin. A machine
// registers once with each hub it runs work of; moving between hubs never registers again.
// A registration from before this record (registration.json and environment.json) is one of them.
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
			if raw, err := os.ReadFile(h.path("environment.json")); err == nil {
				_ = json.Unmarshal(raw, &legacy.Environment)
			}
			out[legacy.Hub] = legacy
		}
	}
	return out, nil
}

func (h *Host) registration(ctx context.Context, hubOrigin string, client *hub.Client) (registration, *exit.Error) {
	all, problem := h.registrations()
	if problem != nil {
		return registration{}, problem
	}
	if registered, ok := all[hubOrigin]; ok && registered.ID != "" {
		return registered, nil
	}
	if client == nil {
		return registration{}, exit.Named(exit.Credential, "machine.registration_required", "this machine is not registered with %s", hubOrigin)
	}
	machine, problem := client.RegisterMachine(ctx)
	if problem != nil {
		return registration{}, problem.WithRemedy("sign in with `cozy auth login`; a machine is registered to the user who owns it")
	}
	if len(machine.Ignored) > 0 {
		h.note(fmt.Sprintf("the hub %s sent settings this machine does not read: %s; ignored", hubOrigin, strings.Join(machine.Ignored, ", ")))
	}
	registered := registration{Hub: hubOrigin, ID: machine.ID, WorkerToken: machine.WorkerToken, Environment: machine.Environment}
	all[hubOrigin] = registered
	raw, _ := json.Marshal(all)
	if err := writePrivate(h.path("registrations.json"), raw); err != nil {
		return registration{}, exit.Internalf("cannot record the machine registration: %s", err)
	}
	return registered, nil
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
	binary, err := filepath.EvalSymlinks(h.binary())
	return err == nil && strings.TrimSuffix(target, " (deleted)") == binary
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
