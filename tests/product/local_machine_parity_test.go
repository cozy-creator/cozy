package producttest

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"maps"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hostruntime"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	"github.com/cozy-creator/cozy/internal/secret"
)

const (
	parityRental  = "pr-parity-0001"
	parityWorker  = "wrk-parity-rental"
	parityPackage = "local/machine-parity"
)

// machineHub stands in for Tensorhub's account API and, over TLS, its rental worker API.
// An owned machine receives scoped execution access without joining the rental registry.
type machineHub struct {
	*fakeRentalHub
	worker                                        *httptest.Server
	access                                        *httptest.Server // local account-delegated content API, separate from account calls
	ca                                            []byte           // the worker doors' private CA
	provider                                      string           // the rental's machine root, as its provider booted it
	mu                                            sync.Mutex
	grants                                        map[string]string // more of the grant, as a Hub adds a port for an image that serves it
	authorityWorker, authorityToken, authorityKey string            // current provider attempt only
}

func newMachineHub(t *testing.T) *machineHub {
	t.Helper()
	h := &machineHub{fakeRentalHub: newFakeRentalHub(t, 0)}
	h.worker, h.ca = hubTLSServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/worker/rental/authorized-keys":
			h.mu.Lock()
			worker, token, key := h.authorityWorker, h.authorityToken, h.authorityKey
			h.mu.Unlock()
			if r.Method != http.MethodGet || worker == "" || token == "" || key == "" ||
				r.Header.Get("X-Cozy-Worker-ID") != worker || r.Header.Get("X-Cozy-Worker-Token") != token {
				http.Error(w, "worker authority refused", http.StatusForbidden)
				return
			}
			_ = json.NewEncoder(w).Encode(map[string]any{"worker_id": worker, "authorized_keys": []string{key}, "lease_seconds": 30})
		case "/v1/worker/rental/release", "/v1/worker/rental/cache-observations":
			w.WriteHeader(http.StatusNoContent)
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(h.worker.Close)
	h.access = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		h.worker.Config.Handler.ServeHTTP(w, r)
	}))
	t.Cleanup(h.access.Close)
	served := h.server.Config.Handler
	h.server.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		authorized := r.Header.Get("Authorization") == "Bearer rental-idle-test"
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/v1/machines":
			t.Error("owned machine attempted Hub registration")
			http.Error(w, "registration is not a local machine lifecycle", http.StatusGone)
		case r.Method == http.MethodPost && r.URL.Path == "/v1/execution-access" && authorized:
			_ = json.NewEncoder(w).Encode(map[string]any{"token": executionGrantToken(h.server.URL, "fixture-account", 1), "expires_at": time.Now().Add(time.Hour),
				"environment": map[string]string{"TENSORHUB_ORIGIN": h.access.URL, "TENSORHUB_PUBLIC_ORIGIN": h.server.URL}})
		case strings.HasPrefix(r.URL.Path, "/v1/tensorfs/"):
			h.worker.Config.Handler.ServeHTTP(w, r)
		default:
			served.ServeHTTP(w, r)
		}
	})
	return h
}

// environment is the hub-known half of a Host's grant, as a rental's pod receives it.
func (h *machineHub) environment() map[string]string {
	env := map[string]string{
		"TENSORHUB_ORIGIN": h.worker.URL, "TENSORHUB_PUBLIC_ORIGIN": h.worker.URL,
		"TENSORHUB_CA_DER_B64URL": base64.RawURLEncoding.EncodeToString(h.ca),
	}
	maps.Copy(env, h.grants)
	return env
}

func randomToken(t *testing.T) string {
	raw := make([]byte, 32)
	_, err := rand.Read(raw)
	must(t, err)
	return base64.RawURLEncoding.EncodeToString(raw)
}

// providerHost boots the same Host binary as a rental's provider would: under its own root,
// with the rental's Creator key as the owner key its grant names.
func providerHost(t *testing.T, h *machineHub, layout home.Layout, source machines.Source, uv string) (*machines.Launch, rental.CreatorIdentity, string, string) {
	t.Helper()
	identity, problem := rental.PendingCreatorIdentity(layout, "parity-rental")
	fatal(t, problem)
	// Short roots: a Runtime's executor socket lives under the machine root.
	providerHome, err := os.MkdirTemp("", "czr")
	must(t, err)
	claimScratch(providerHome)
	dir := filepath.Join(providerHome, "machine")
	t.Cleanup(func() {
		if t.Failed() {
			t.Logf("rental machine log tail:\n%s", tail(filepath.Join(dir, "host.log")))
		}
		_ = removeAllForce(providerHome)
	})
	trackDaemonRoot(t, providerHome)
	host := machines.NewHost(dir, "", nil)
	host.WebRTCPort, _ = strconv.Atoi(h.grants["COZY_WEBRTC_INTERNAL_PORT"])
	_, problem = host.Install(context.Background(), source, uv)
	fatal(t, problem)
	key, err := os.ReadFile(layout.PendingRentalCreatorIdentity("parity-rental"))
	must(t, err)
	must(t, os.WriteFile(filepath.Join(dir, "owner.pem"), key, 0o600))
	// A provider passes the rental grant directly to the same executable. Local Host.Ensure
	// intentionally ignores legacy registration/environment files and always boots persistent.
	free := func() int {
		listener, err := net.Listen("tcp", "127.0.0.1:0") //cozy:allow choose fixture provider ports
		must(t, err)
		port := listener.Addr().(*net.TCPAddr).Port
		must(t, listener.Close())
		return port
	}
	workerPort, mediaPort := free(), free()
	for mediaPort == workerPort {
		mediaPort = free()
	}
	receiptKey, mediaToken := randomToken(t), randomToken(t)
	must(t, os.WriteFile(filepath.Join(dir, "media-token"), []byte(mediaToken), 0600))
	environment := h.environment()
	workerToken := randomToken(t)
	h.mu.Lock()
	h.authorityWorker, h.authorityToken, h.authorityKey = parityWorker, workerToken, identity.PublicKey()
	h.mu.Unlock()
	maps.Copy(environment, map[string]string{
		"COZY_MACHINE_ROOT": host.Root(), "COZY_MACHINE_LIFETIME": "rental", "COZY_LISTEN_HOST": "127.0.0.1",
		"COZY_WORKER_ID": parityWorker, "COZY_WORKER_AUTH_TOKEN": workerToken,
		"COZY_WORKER_INTERNAL_PORT": strconv.Itoa(workerPort), "COZY_MEDIA_INTERNAL_PORT": strconv.Itoa(mediaPort),
		"COZY_BOOTSTRAP_RECEIPT_HMAC_KEY_B64URL": receiptKey, "COZY_AUTHORIZED_KEYS": identity.PublicKey(),
	})
	command := exec.Command(filepath.Join(host.Root(), "usr/local/bin/cozy-machine"))
	command.Dir = host.Root()
	for name, value := range environment {
		command.Env = append(command.Env, name+"="+value)
	}
	// As a pod's image does: the machine's own tools (uv) are on its PATH.
	command.Env = append(command.Env, "PATH="+filepath.Join(host.Root(), "usr/local/bin")+":/usr/bin:/bin")
	// A machine finds GPUs where its launcher may: a run that hides them hides them here too.
	if visible, named := os.LookupEnv("CUDA_VISIBLE_DEVICES"); named {
		command.Env = append(command.Env, "CUDA_VISIBLE_DEVICES="+visible)
	}
	log, err := os.OpenFile(filepath.Join(dir, "host.log"), os.O_CREATE|os.O_WRONLY, 0600)
	must(t, err)
	command.Stdout, command.Stderr = log, log
	must(t, command.Start())
	done := make(chan error, 1)
	go func() { done <- command.Wait() }()
	t.Cleanup(func() { _ = command.Process.Signal(syscall.SIGTERM); <-done; _ = log.Close() })
	record, err := json.Marshal(map[string]any{"pid": command.Process.Pid, "worker_id": parityWorker,
		"worker_port": workerPort, "media_port": mediaPort, "receipt_key": receiptKey})
	must(t, err)
	must(t, os.WriteFile(filepath.Join(dir, "agent.json"), record, 0600))
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	launch, problem := host.Ensure(ctx, nil)
	if problem != nil {
		log, _ := os.ReadFile(filepath.Join(dir, "host.log"))
		t.Fatalf("the provider Host did not boot: %s\n%s", problem.Message, log)
	}
	return launch, identity, mediaToken, host.Root()
}

func parityProject(t *testing.T) string {
	t.Helper()
	return parityProjectOn(t, machines.Source{RuntimeWheel: *machineRuntimeWheel, TensorFSWheel: *machineTensorFSWheel})
}

// parityProjectOn locks the parity package against source's wheels, or the published Runtime.
func parityProjectOn(t *testing.T, source machines.Source) string {
	t.Helper()
	project := filepath.Join(t.TempDir(), "machine-parity")
	must(t, os.MkdirAll(project, 0o700))
	runtime := "cozy-runtime>=" + hostruntime.ToolFloor
	sources := ""
	if source.RuntimeWheel != "" {
		sources = fmt.Sprintf("[tool.uv.sources]\ncozy-runtime={path=%q}\ntensorfs={path=%q}\n", source.RuntimeWheel, source.TensorFSWheel)
	}
	must(t, os.WriteFile(filepath.Join(project, "pyproject.toml"), []byte(`[project]
name="machine-parity"
version="0.0.1"
requires-python=">=3.12,<3.13"
dependencies=["`+runtime+`", "msgspec>=0.19"]
[project.entry-points."cozy.application"]
default="machine_parity:app"
`+sources+`[build-system]
requires=["hatchling"]
build-backend="hatchling.build"
[tool.hatch.build.targets.wheel]
only-include=["machine_parity.py"]
`), 0o600))
	must(t, os.WriteFile(filepath.Join(project, "package.toml"), []byte("[application]\nobject=\"machine_parity:app\"\n"), 0o600))
	must(t, os.WriteFile(filepath.Join(project, "machine_parity.py"), []byte(`import msgspec
from cozy_runtime.author import App


class AddRequest(msgspec.Struct, forbid_unknown_fields=True):
    value: int


class AddResult(msgspec.Struct):
    value: int


app = App()


@app.job
def add(payload: AddRequest) -> AddResult:
    return AddResult(value=payload.value + 1)


@app.entrypoint
def echo(payload: AddRequest) -> AddResult:
    return AddResult(value=payload.value * 2)

`), 0o600))
	lock := exec.Command("uv", "lock", "--project", project)
	if out, err := lock.CombinedOutput(); err != nil {
		t.Fatalf("locking the parity package: %v\n%s", err, out)
	}
	return project
}

// journal is a request's durable evidence with every venue fact removed: event kinds in
// order, and the GPU widths the Runtime granted. Choosing a rental from the fleet is the
// one step a call on this computer does not take.
func journal(t *testing.T, store *records.Store, id string) []string {
	t.Helper()
	events, problem := store.EvidenceEvents(id, 1000)
	fatal(t, problem)
	var out []string
	for _, event := range events {
		kind := event.Type
		// Where a run chose its rental is the one step a call on this computer skips; the
		// Runtime's collection record is written after the terminal a client returns on.
		if kind == "request.rentals" || kind == "request.placement" || kind == "machine.collected" {
			continue
		}
		// Local boot may briefly park while the already-started rental connects.
		// Connection diagnostics and stdout can interleave with durable machine
		// events; parity concerns the accepted execution and its result lifecycle.
		if kind == "request.parked" || kind == "request.log" ||
			kind == "request.preparing" && event.Payload["stage"] == "connect" {
			continue
		}
		// Progress and timing are observations, not lifecycle transitions. Their
		// sample count and position vary with scheduling on each venue.
		if kind == "machine.progress" || kind == "request.progress" || kind == "machine.run.timing" {
			continue
		}
		if kind == "request.preparing" {
			kind += ":" + fmt.Sprint(event.Payload["stage"])
		}
		if ordinals, ok := event.Payload["ordinals"].([]any); ok && kind == "machine.gpu.grant" {
			kind += fmt.Sprintf(":width=%d", len(ordinals))
		}
		if len(out) > 0 && out[len(out)-1] == kind {
			continue // repeated observation of one state
		}
		out = append(out, kind)
	}
	return out
}

// parityMachines is one daemon root with both machines of the parity body: this computer's,
// installed and launched by the daemon, and the rental `tessa`, the same Host binary a
// provider booted and the daemon attached. Each measures what its launcher lets it see.
func parityMachines(t *testing.T) (*machineHub, string, home.Layout, *records.Store) {
	t.Helper()
	return parityMachinesOn(t, machines.Source{Host: *machineHostBinary, RuntimeWheel: *machineRuntimeWheel, TensorFSWheel: *machineTensorFSWheel})
}

// parityMachinesOn is parityMachines with both machines running source's Host and Runtime.
func parityMachinesOn(t *testing.T, source machines.Source) (*machineHub, string, home.Layout, *records.Store) {
	t.Helper()
	if source.Host == "" {
		t.Skip("requires -machine-host: the standalone agent both machines run")
	}
	uv, err := exec.LookPath("uv")
	if err != nil {
		t.Skip("uv lays out both machine roots")
	}
	h := newMachineHub(t)
	root, err := os.MkdirTemp("", "czp")
	must(t, err)
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+h.server.URL+
		"\ntensorhub_token: rental-idle-test\ndaemon:\n  idle_shutdown_s: 0\n"), 0o600))
	t.Cleanup(func() {
		_, _ = runCozy(t, root, "machine", "stop")
		_, _ = runCozy(t, root, "down")
		if t.Failed() {
			log, _ := os.ReadFile(filepath.Join(root, "machine", "host.log"))
			t.Logf("parity evidence retained at %s\nlocal Host log:\n%s", root, log)
		} else {
			_ = removeAllForce(root)
		}
	})
	install := []string{"machine", "install", "--host", source.Host}
	if source.RuntimeWheel != "" {
		install = append(install, "--runtime-wheel", source.RuntimeWheel, "--tensorfs-wheel", source.TensorFSWheel)
	}
	if code, out := runCozy(t, root, install...); code != 0 {
		t.Fatalf("machine install [exit %d]\n%s", code, out)
	}

	layout, problem := home.Open(root)
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	t.Cleanup(func() { store.Close() })
	launch, identity, token, provider := providerHost(t, h, layout, source, uv)
	h.provider = provider
	// The rental's paid hardware is what its worker's ClaimAck reads back: the inventory the
	// Runtime schedules on, the same four virtual devices its workspace reports.
	const model, count = "Virtual Accelerator", 4
	h.mu.Lock()
	h.rentals[parityRental] = map[string]any{"rental_id": parityRental, "name": "tessa", "state": "ready",
		"requested_accelerator_model": model, "accelerator_count": count, "hourly_rate_usd_micros": 1,
		"worker_address": launch.Addr, "media_address": launch.MediaAddr}
	h.mu.Unlock()
	fatal(t, rental.Attach(layout, store, records.Rental{ID: parityRental, MachineName: "tessa", SKU: "virtual-4", State: "ready",
		AcceleratorModel: model, AcceleratorCount: count, HourlyRateUSDMicros: 1, Hub: h.server.URL,
		Address: launch.Addr, MediaAddress: launch.MediaAddr, ExpectedWorkerID: launch.WorkerID, ExpectedWorkerBootID: launch.BootID},
		string(pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: launch.Leaf})), secret.New(token), identity))
	return h, root, layout, store
}

// One product body, run on this computer's machine and on a rental: the literal same machine
// binary. The local
// machine uses its own identity; the rental is the same binary a provider booted, pinned
// and attached. Only the machine name and address may differ.
func TestLocalAndRentedMachinesRunOneBody(t *testing.T) {
	_, root, _, store := parityMachines(t)
	if code, out := runCozy(t, root, "package", "install", parityProject(t), "--editable"); code != 0 {
		t.Fatalf("editable install [exit %d]\n%s", code, out)
	}
	journals := map[string][][]string{}
	for _, venue := range []struct {
		name string
		args []string
	}{{"local", nil}, {"rental", []string{"--rental=tessa"}}} {
		t.Run(venue.name, func(t *testing.T) {
			for _, call := range []struct{ function, want string }{{"add", `"value":42`}, {"echo", `"value":82`}} {
				key := "parity-" + venue.name + "-" + call.function
				args := append([]string{"run", parityPackage + "/" + call.function, "value=41", "--await", "--json", "--idempotency-key", key}, venue.args...)
				code, out := runCozy(t, root, args...)
				if code != 0 || !strings.Contains(out, call.want) {
					t.Fatalf("%s on %s [exit %d]\n%s", call.function, venue.name, code, out)
				}
				request, problem := store.RequestByIdempotencyKey(key)
				fatal(t, problem)
				link, problem := store.MachineExecution(request.ID)
				fatal(t, problem)
				want := map[string]string{"local": machines.Local, "rental": parityRental}[venue.name]
				if link == nil || link.MachineID != want || !link.Collected {
					t.Fatalf("%s on %s was not a collected execution on %s: %+v", call.function, venue.name, want, link)
				}
				// The run's record shows the result the client received, as JSON and as text.
				code, shown := runCozy(t, root, "run", "show", request.ID, "--json")
				var report struct{ Result json.RawMessage }
				if code != 0 || json.Unmarshal([]byte(shown), &report) != nil || !strings.Contains(string(report.Result), call.want) {
					t.Fatalf("run show --json of %s on %s lacks its result [exit %d]\n%s", call.function, venue.name, code, shown)
				}
				if code, shown = runCozy(t, root, "run", "show", request.ID); code != 0 || !strings.Contains(shown, "result {"+call.want+"}") {
					t.Fatalf("run show of %s on %s lacks its result [exit %d]\n%s", call.function, venue.name, code, shown)
				}
				journals[venue.name] = append(journals[venue.name], journal(t, store, request.ID))
			}
		})
	}
	if len(journals["local"]) != 2 || len(journals["rental"]) != 2 {
		t.Fatal("a venue did not complete the body")
	}
	for index := range journals["local"] {
		if !slices.Equal(journals["local"][index], journals["rental"][index]) {
			t.Errorf("call %d journals differ between venues:\nlocal:  %v\nrental: %v", index, journals["local"][index], journals["rental"][index])
		}
	}
	t.Logf("journals: %v", journals["local"])

	// A stopped agent launches again on its next call, keeping its machine root and boot.
	// The single supervisor needs no durable Runtime ownership history on either launch.
	bootID := filepath.Join(root, "machine", "root", "run/cozy/bootstrap/pod-boot-id")
	before, err := os.ReadFile(bootID)
	must(t, err)
	noHistory := func() {
		if _, err := os.Stat(filepath.Join(root, "machine", "root", "run/cozy/worker/ownership.json")); !os.IsNotExist(err) {
			t.Fatalf("single-agent Runtime retained ownership history: %v", err)
		}
	}
	noHistory()
	if code, out := runCozy(t, root, "machine", "stop"); code != 0 {
		t.Fatalf("machine stop [exit %d]\n%s", code, out)
	}
	if code, out := runCozy(t, root, "run", parityPackage+"/add", "value=1", "--await", "--json", "--idempotency-key", "parity-relaunch"); code != 0 || !strings.Contains(out, `"value":2`) {
		t.Fatalf("the relaunched machine did not run [exit %d]\n%s", code, out)
	}
	if after, err := os.ReadFile(bootID); err != nil || string(after) != string(before) {
		t.Fatalf("the relaunched machine is another boot (%q, was %q): %v", after, before, err)
	}
	noHistory()

	// Both machines answer the same Status: `machine show` and `rental show` read it.
	for _, show := range [][]string{{"machine", "show", "--json"}, {"rental", "show", "tessa", "--json"}} {
		code, out := runCozy(t, root, show...)
		var shown map[string]any
		if code != 0 || json.Unmarshal([]byte(lastJSONLine(out)), &shown) != nil || shown["phase"] != "ready" || shown["runtime_version"] == nil {
			t.Fatalf("cozy %s did not report its machine's status [exit %d]\n%s", strings.Join(show, " "), code, out)
		}
	}
}
