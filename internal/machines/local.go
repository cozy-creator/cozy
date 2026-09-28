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
	"sort"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/flock"
	"github.com/cozy-creator/cozy/internal/hostruntime"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/rental"
	"github.com/cozy-creator/cozy/internal/workertls"
)

// Host is this computer's machine: the literal pod-supervisor a rented pod runs, launched
// detached under a root laid out as the pod's `/`. It outlives the daemon as a pod outlives
// its controller, and it exits by itself after the same fixed idle period; the next use
// launches it again.
type Host struct {
	dir    string
	mu     sync.Mutex
	cached *cachedLaunch
	// inherited are the locale and trust-store values a Host may carry from its launcher.
	inherited []string
}

func NewHost(dir string, environ []string) *Host {
	// A machine moved elsewhere by a symlink (another disk, a shorter path) runs there: the
	// Runtime's sockets live under the root it is given, and a socket path is bounded.
	if resolved, err := filepath.EvalSymlinks(dir); err == nil {
		dir = resolved
	}
	h := &Host{dir: dir}
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
func (h *Host) readinessEnvelope() string {
	return filepath.Join(h.Root(), "run/cozy/bootstrap/readiness-envelope.json")
}

// RuntimeFloor is the oldest published Runtime proven as a rooted machine worker.
const RuntimeFloor = "0.18.53"

// Source is one cohort's worker artifacts: the Host binary and the Runtime and TensorFS
// wheels a pod image carries. Without wheels, the published Runtime is installed.
type Source struct {
	Host, RuntimeWheel, TensorFSWheel string
}

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
	for _, artifact := range []struct {
		path string
		out  *installedArtifact
	}{{source.Host, &installed.Host}, {source.RuntimeWheel, &installed.Runtime}, {source.TensorFSWheel, &installed.TensorFS}} {
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
	if err := copyExecutable(source.Host, h.binary()); err != nil {
		return nil, exit.Internalf("cannot install the machine Host: %s", err)
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
	requirements := []string{hostruntime.Distribution + "[media]>=" + RuntimeFloor}
	if source.RuntimeWheel != "" {
		requirements = []string{hostruntime.Distribution + "[media] @ file://" + source.RuntimeWheel, source.TensorFSWheel}
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
	Hub         string `json:"hub"`
	ID          string `json:"id"`
	WorkerToken string `json:"worker_token"`
}

// hostRecord is one launched Host, as a rental row records its pod.
type hostRecord struct {
	PID        int    `json:"pid"`
	Hub        string `json:"hub"`
	WorkerID   string `json:"worker_id"`
	WorkerPort int    `json:"worker_port"`
	MediaPort  int    `json:"media_port"`
	ReceiptKey string `json:"receipt_key"`
}

type cachedLaunch struct {
	*Launch
	hub string
}

// Launch is the running Host's identity once its readiness receipt verified.
type Launch struct {
	Addr, MediaAddr  string
	WorkerID, BootID string
	Leaf             []byte
	GPUs             []ReceiptGPU
	PID              int
}

// Ensure answers the running Host, launching it when none runs. hubOrigin is the hub the
// machine belongs to, and client speaks to it as the signed-in user.
func (h *Host) Ensure(ctx context.Context, hubOrigin string, client *hub.Client) (*Launch, *exit.Error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if cached := h.cached; cached != nil && cached.hub == hubOrigin && h.alive(cached.PID) {
		return cached.Launch, nil
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
		if record.Hub != hubOrigin {
			// One root holds one Host: the machine moves to the selected hub.
			if problem := h.stopLocked(ctx); problem != nil {
				return nil, problem
			}
		} else if launch, problem := h.await(ctx, record); problem == nil || h.alive(record.PID) {
			return h.remember(launch, hubOrigin), problem
		}
	}
	return h.launchLocked(ctx, hubOrigin, client)
}

func (h *Host) remember(launch *Launch, hubOrigin string) *Launch {
	if launch != nil {
		h.cached = &cachedLaunch{Launch: launch, hub: hubOrigin}
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
				WorkerID: record.WorkerID, BootID: receipt.PodBootID, Leaf: leaf, GPUs: receipt.RuntimeGPUs, PID: record.PID}, nil
		}
		if refused := (*receiptRefusal)(nil); errors.As(err, &refused) {
			return nil, exit.New(exit.Credential, "the machine Host's readiness receipt did not verify: %s", err)
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
	installed, problem := h.Installed()
	if problem != nil {
		return nil, problem
	}
	if installed == nil {
		return nil, exit.Named(exit.Structural, "machine.not_installed", "this computer has no machine installed").
			WithRemedy("cozy machine install --host <pod-supervisor>")
	}
	registered, problem := h.registration(ctx, hubOrigin, client)
	if problem != nil {
		return nil, problem
	}
	environment, problem := h.environment()
	if problem != nil {
		return nil, problem
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
	names := make([]string, 0, len(environment))
	for name := range environment {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		base = append(base, name+"="+environment[name])
	}
	// Free ports are chosen, not reserved: a port taken before the Host binds it ends
	// that Host before readiness, and the next launch chooses again.
	for attempt := 0; ; attempt++ {
		launch, problem := h.start(ctx, base, key, hubOrigin, registered.ID)
		if problem == nil || problem.ErrName() != "machine.host_port_taken" || attempt == 2 {
			return h.remember(launch, hubOrigin), problem
		}
	}
}

func (h *Host) start(ctx context.Context, base []string, key []byte, hubOrigin, workerID string) (*Launch, *exit.Error) {
	workerPort := freePort(0)
	mediaPort := freePort(workerPort)
	env := append(append([]string(nil), base...),
		"COZY_WORKER_INTERNAL_PORT="+strconv.Itoa(workerPort), "COZY_MEDIA_INTERNAL_PORT="+strconv.Itoa(mediaPort))
	log, err := os.OpenFile(h.path("host.log"), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		return nil, exit.Internalf("cannot open the machine Host log: %s", err)
	}
	defer log.Close()
	offset, _ := log.Seek(0, io.SeekEnd)
	command := exec.Command(h.binary())
	command.Env, command.Dir = env, h.Root()
	command.Stdout, command.Stderr = log, log
	detach(command)
	if err := command.Start(); err != nil {
		return nil, exit.Internalf("cannot start the machine Host: %s", err)
	}
	go func() { _ = command.Wait() }()
	record := &hostRecord{PID: command.Process.Pid, Hub: hubOrigin, WorkerID: workerID,
		WorkerPort: workerPort, MediaPort: mediaPort, ReceiptKey: base64.RawURLEncoding.EncodeToString(key)}
	raw, _ := json.Marshal(record)
	if err := writePrivate(h.path("host.json"), raw); err != nil {
		_ = command.Process.Signal(syscall.SIGTERM)
		return nil, exit.Internalf("cannot record the machine Host: %s", err)
	}
	launch, problem := h.await(ctx, record)
	if problem != nil && problem.ErrName() == "machine.host_exited" {
		output := logSince(h.path("host.log"), offset)
		if strings.Contains(output, "address already in use") {
			return nil, exit.Named(exit.Unavailable, "machine.host_port_taken", "the machine Host's port was taken")
		}
		return nil, exit.Named(exit.Structural, "machine.host_exited", "the machine Host exited before readiness: %s", output)
	}
	if problem != nil && problem.Code == exit.Credential {
		_ = command.Process.Signal(syscall.SIGTERM)
	}
	return launch, problem
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
	if raw, err := os.ReadFile(h.path("registration.json")); err == nil {
		var registered registration
		if json.Unmarshal(raw, &registered) == nil {
			out.MachineID, out.Hub = registered.ID, registered.Hub
		}
	}
	record, problem := h.record()
	if problem != nil {
		return Status{}, problem
	}
	if record != nil {
		out.PID, out.Running = record.PID, h.alive(record.PID)
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

func (h *Host) registration(ctx context.Context, hubOrigin string, client *hub.Client) (registration, *exit.Error) {
	var registered registration
	if raw, err := os.ReadFile(h.path("registration.json")); err == nil {
		if json.Unmarshal(raw, &registered) == nil && registered.Hub == hubOrigin && registered.ID != "" {
			return registered, nil
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		return registration{}, exit.Internalf("cannot read the machine registration: %s", err)
	}
	if client == nil {
		return registration{}, exit.Named(exit.Credential, "machine.registration_required", "this machine is not registered with %s", hubOrigin)
	}
	machine, problem := client.RegisterMachine(ctx)
	if problem != nil {
		return registration{}, problem.WithRemedy("sign in with `cozy auth login`; a machine is registered to the user who owns it")
	}
	registered = registration{Hub: hubOrigin, ID: machine.ID, WorkerToken: machine.WorkerToken}
	environment, _ := json.Marshal(machine.Environment)
	if err := writePrivate(h.path("environment.json"), environment); err != nil {
		return registration{}, exit.Internalf("cannot record the machine environment: %s", err)
	}
	raw, _ := json.Marshal(registered)
	if err := writePrivate(h.path("registration.json"), raw); err != nil {
		return registration{}, exit.Internalf("cannot record the machine registration: %s", err)
	}
	return registered, nil
}

// environment is the hub-authored half of the grant, as the hub answered the registration.
func (h *Host) environment() (map[string]string, *exit.Error) {
	var environment map[string]string
	raw, err := os.ReadFile(h.path("environment.json"))
	if err != nil || json.Unmarshal(raw, &environment) != nil || len(environment) == 0 {
		return nil, exit.Named(exit.Unavailable, "machine.environment_unavailable", "this machine's registration recorded no hub environment").
			WithRemedy("re-register it: remove %s and run it again", h.path("registration.json"))
	}
	return environment, nil
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

func copyExecutable(source, target string) error {
	input, err := os.Open(source)
	if err != nil {
		return err
	}
	defer input.Close()
	staged := target + ".new"
	output, err := os.OpenFile(staged, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o755)
	if err != nil {
		return err
	}
	if _, err := io.Copy(output, input); err != nil {
		output.Close()
		return err
	}
	if err := output.Close(); err != nil {
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
