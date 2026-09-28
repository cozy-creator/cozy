package producttest

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	"github.com/cozy-creator/cozy/internal/secret"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

const (
	parityRental  = "pr-parity-0001"
	parityWorker  = "wrk-parity-rental"
	parityPackage = "local/machine-parity"
)

// machineHub stands in for Tensorhub on both sides of a machine: the account API the daemon
// calls (the rental hub plus owned-machine registration) and, over TLS, the worker doors
// every Host calls.
type machineHub struct {
	*fakeRentalHub
	worker   *httptest.Server
	mu       sync.Mutex
	machines map[string]string
}

func newMachineHub(t *testing.T) *machineHub {
	t.Helper()
	h := &machineHub{fakeRentalHub: newFakeRentalHub(t, 0), machines: map[string]string{}}
	h.worker = httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/worker/rental/release", "/v1/worker/rental/cache-observations":
			w.WriteHeader(http.StatusNoContent)
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(h.worker.Close)
	served := h.server.Config.Handler
	h.server.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		authorized := r.Header.Get("Authorization") == "Bearer rental-idle-test"
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/v1/machines" && authorized:
			id, token := "om-"+randomToken(t)[:22], randomToken(t)
			h.mu.Lock()
			h.machines[id] = token
			h.mu.Unlock()
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusCreated)
			_ = json.NewEncoder(w).Encode(map[string]string{"id": id, "worker_token": token})
		case r.Method == http.MethodGet && strings.HasPrefix(r.URL.Path, "/v1/machines/") && strings.HasSuffix(r.URL.Path, "/environment") && authorized:
			_ = json.NewEncoder(w).Encode(map[string]any{"environment": h.environment()})
		default:
			served.ServeHTTP(w, r)
		}
	})
	return h
}

// environment is the hub-known half of a Host's grant, as a rental's pod receives it.
func (h *machineHub) environment() map[string]string {
	return map[string]string{
		"TENSORHUB_ORIGIN": h.worker.URL, "TENSORHUB_PUBLIC_ORIGIN": h.worker.URL,
		"TENSORHUB_CA_DER_B64URL": base64.RawURLEncoding.EncodeToString(h.worker.Certificate().Raw),
	}
}

func randomToken(t *testing.T) string {
	raw := make([]byte, 32)
	_, err := rand.Read(raw)
	must(t, err)
	return base64.RawURLEncoding.EncodeToString(raw)
}

// virtualInventory replaces the image's worker entry with the same `main` measuring a
// virtual four-device inventory: the in-process launcher seam, never an environment switch.
func virtualInventory(t *testing.T, root string) {
	t.Helper()
	python := filepath.Join(root, "opt/cozy/python/bin/python")
	worker := filepath.Join(root, "opt/cozy/bin/cozy-runtime-worker")
	must(t, os.Remove(worker))
	must(t, os.WriteFile(worker, []byte("#!"+python+`
import sys
from cozy_runtime.cli import runtime_worker
sys.exit(runtime_worker.main([], gpus=[{"device_index": i, "device_name": "Virtual Accelerator", "device_uuid": f"GPU-virtual-{i}",
    "driver_version": "0.0", "memory_bytes": 8 << 30, "pci_bus_id": f"00000000:0{i}:00.0"} for i in range(4)]))
`), 0o755))
}

// providerHost boots the same Host binary as a rental's provider would: under its own root,
// with the rental's Creator key as the owner key its grant names.
func providerHost(t *testing.T, h *machineHub, layout home.Layout, source machines.Source, uv string) (*machines.Launch, rental.CreatorIdentity, string) {
	t.Helper()
	identity, problem := rental.PendingCreatorIdentity(layout, "parity-rental")
	fatal(t, problem)
	// Short roots: a Runtime's executor socket lives under the machine root.
	dir, err := os.MkdirTemp("", "czr")
	must(t, err)
	t.Cleanup(func() { _ = removeAllForce(dir) })
	host := machines.NewHost(dir, nil)
	_, problem = host.Install(context.Background(), source, uv)
	fatal(t, problem)
	virtualInventory(t, host.Root())
	key, err := os.ReadFile(layout.PendingRentalCreatorIdentity("parity-rental"))
	must(t, err)
	must(t, os.WriteFile(filepath.Join(dir, "owner.pem"), key, 0o600))
	registration, _ := json.Marshal(map[string]string{"hub": "provider", "id": parityWorker, "worker_token": randomToken(t)})
	must(t, os.WriteFile(filepath.Join(dir, "registration.json"), registration, 0o600))
	environment, _ := json.Marshal(h.environment())
	must(t, os.WriteFile(filepath.Join(dir, "environment.json"), environment, 0o600))
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	launch, problem := host.Ensure(ctx, "provider", nil)
	if problem != nil {
		log, _ := os.ReadFile(filepath.Join(dir, "host.log"))
		t.Fatalf("the provider Host did not boot: %s\n%s", problem.Message, log)
	}
	t.Cleanup(func() { _ = host.Stop(context.Background()) })
	token, err := os.ReadFile(filepath.Join(dir, "media-token"))
	must(t, err)
	return launch, identity, string(token)
}

func parityProject(t *testing.T) string {
	t.Helper()
	project := filepath.Join(t.TempDir(), "machine-parity")
	must(t, os.MkdirAll(project, 0o700))
	runtime := "cozy-runtime>=" + machines.RuntimeFloor
	sources := ""
	if *machineRuntimeWheel != "" {
		sources = fmt.Sprintf("[tool.uv.sources]\ncozy-runtime={path=%q}\ntensorfs={path=%q}\n", *machineRuntimeWheel, *machineTensorFSWheel)
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

// One product body, run on this computer's machine and on a rental: the literal same Host
// binary with a real Runtime worker measuring a virtual four-device inventory. The local
// machine is registered and launched by the daemon; the rental is the same binary a
// provider booted, pinned and attached. Only the machine name and address may differ.
func TestLocalAndRentedMachinesRunOneBody(t *testing.T) {
	if *machineHostBinary == "" {
		t.Skip("requires -machine-host: the pod-supervisor both machines run")
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
	source := machines.Source{Host: *machineHostBinary, RuntimeWheel: *machineRuntimeWheel, TensorFSWheel: *machineTensorFSWheel}
	install := []string{"machine", "install", "--host", source.Host}
	if source.RuntimeWheel != "" {
		install = append(install, "--runtime-wheel", source.RuntimeWheel, "--tensorfs-wheel", source.TensorFSWheel)
	}
	if code, out := runCozy(t, root, install...); code != 0 {
		t.Fatalf("machine install [exit %d]\n%s", code, out)
	}
	virtualInventory(t, filepath.Join(root, "machine", "root"))

	layout, problem := home.Open(root)
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	launch, identity, token := providerHost(t, h, layout, source, uv)
	// The rental's paid hardware is what its worker's ClaimAck reads back: the inventory the
	// Runtime schedules on, the same four virtual devices its workspace reports.
	const model, count = "Virtual Accelerator", 4
	h.mu.Lock()
	h.rentals[parityRental] = map[string]any{"rental_id": parityRental, "name": "tessa", "state": "ready",
		"requested_accelerator_model": model, "accelerator_count": count, "hourly_rate_usd_micros": 1,
		"worker_address": launch.Addr, "media_address": launch.MediaAddr}
	h.inventories = map[string]json.RawMessage{parityRental: json.RawMessage(`{"format":"tensorhub.image_inventory/1","profile":"python3.12-cpu-linux-x86","python":"3.12"}`)}
	h.mu.Unlock()
	fatal(t, rental.Attach(layout, store, records.Rental{ID: parityRental, MachineName: "tessa", SKU: "virtual-4", State: "ready",
		AcceleratorModel: model, AcceleratorCount: count, HourlyRateUSDMicros: 1, Hub: h.server.URL,
		Address: launch.Addr, MediaAddress: launch.MediaAddr, ExpectedWorkerID: launch.WorkerID, ExpectedWorkerBootID: launch.BootID},
		string(pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: launch.Leaf})), secret.New(token), identity))

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

	// A stopped machine launches again on its next call under a fresh receipt key, keeping
	// its root and boot: the Host's idle exit and relaunch. The Runtime counts every Control
	// Claim in its ownership record: a Runtime reporting claim_survives_restart runs the call
	// on the Claim the daemon already holds; an older one is Claimed again.
	bootID := filepath.Join(root, "machine", "root", "run/cozy/bootstrap/pod-boot-id")
	before, err := os.ReadFile(bootID)
	must(t, err)
	streams := func() int {
		var ownership struct {
			Epoch int `json:"control_stream_epoch"`
		}
		raw, err := os.ReadFile(filepath.Join(root, "machine", "root", "run/cozy/worker/ownership.json"))
		must(t, err)
		must(t, json.Unmarshal(raw, &ownership))
		return ownership.Epoch
	}
	claimsBefore := streams()
	if code, out := runCozy(t, root, "machine", "stop"); code != 0 {
		t.Fatalf("machine stop [exit %d]\n%s", code, out)
	}
	if code, out := runCozy(t, root, "run", parityPackage+"/add", "value=1", "--await", "--json", "--idempotency-key", "parity-relaunch"); code != 0 || !strings.Contains(out, `"value":2`) {
		t.Fatalf("the relaunched machine did not run [exit %d]\n%s", code, out)
	}
	if after, err := os.ReadFile(bootID); err != nil || string(after) != string(before) {
		t.Fatalf("the relaunched machine is another boot (%q, was %q): %v", after, before, err)
	}
	relaunchClaims := streams() - claimsBefore

	// Both machines report the Runtime's measured inventory through the same Host call.
	found := &machines.Resolver{Host: machines.NewHost(layout.Machine, nil), HubOrigin: h.server.URL,
		Rentals: rental.Resolver(layout, store), UseRental: func(string, string) (func(), *exit.Error) { return func() {}, nil },
		RentalKey: func(id string) (rental.CreatorIdentity, *exit.Error) { return rental.CreatorIdentityFor(layout, id) }}
	for _, name := range []string{machines.Local, parityRental} {
		machine, problem := found.Dial(context.Background(), name, "parity inventory")
		fatal(t, problem)
		workspace, err := machine.Host.GetMachineExecutionWorkspace(context.Background(), &pb.MachineExecutionWorkspaceQuery{Claim: machine.Claim})
		machine.Close()
		must(t, err)
		if len(workspace.Devices) != 4 || workspace.Devices[3].Name != "Virtual Accelerator" || workspace.ExecutorUidIsolation {
			t.Fatalf("%s reports %d devices (%v), executor isolation %v", name, len(workspace.Devices), workspace.Devices, workspace.ExecutorUidIsolation)
		}
		if name == machines.Local {
			// This fresh daemon-side resolver's own Control Claim read the capability back.
			want := map[bool]int{true: 0, false: 1}[machine.ClaimSurvivesRestart]
			t.Logf("claim_survives_restart=%v; the relaunch took %d Control Claim(s)", machine.ClaimSurvivesRestart, relaunchClaims)
			if relaunchClaims != want {
				t.Fatalf("the relaunch took %d Control Claims; claim_survives_restart=%v wants %d", relaunchClaims, machine.ClaimSurvivesRestart, want)
			}
		}
	}
}
