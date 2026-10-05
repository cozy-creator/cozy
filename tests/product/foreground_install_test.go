package producttest

import (
	"encoding/json"
	"encoding/pem"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	"github.com/cozy-creator/cozy/internal/secret"
)

// Under a running daemon that predates cozy.machine.v1, an installation on a rental is this
// command's own warm run on the rental's machine: a model download reaches the machine and ends
// with the machine's own answer, and an upload grants publication to the rental by its Hub
// identity. Nothing is queued for the older daemon, and this computer's machine never starts.
func TestARentalInstallUnderAnOlderDaemonIsThisCommandsWarmRun(t *testing.T) {
	if *machineHostBinary == "" || *olderCozy == "" {
		t.Skip("requires -machine-host (the rental's machine) and -older-cozy (a daemon that predates cozy.machine.v1)")
	}
	uv, err := exec.LookPath("uv")
	if err != nil {
		t.Skip("uv lays out the rental's machine root")
	}
	h := newMachineHub(t)
	manifest := "sha256:" + strings.Repeat("a", 64)
	var mu sync.Mutex
	var grants []map[string]any
	served := h.server.Config.Handler
	h.server.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.URL.Path == "/v1/models/resolve":
			_ = json.NewEncoder(w).Encode(map[string]any{"model": "proof/model", "manifest_id": manifest,
				"manifest_length": 128, "bytes": 4096, "components": []string{"transformer"}})
		case r.Method == http.MethodPost && r.URL.Path == "/v1/machine-authorizations":
			raw, _ := io.ReadAll(r.Body)
			var body map[string]any
			_ = json.Unmarshal(raw, &body)
			if strings.Contains(string(raw), "quantized") {
				_ = json.NewEncoder(w).Encode(map[string]any{"token": "machine-grant", "expires_at": time.Now().Add(time.Hour).UTC().Format(time.RFC3339)})
				return
			}
			mu.Lock()
			grants = append(grants, body)
			mu.Unlock()
			w.WriteHeader(http.StatusConflict)
			_, _ = w.Write([]byte(`{"error":{"code":"publication.grant_refused","message":"the stand-in hub grants nothing"}}`))
		default:
			served.ServeHTTP(w, r)
		}
	})
	serveOtherBytes(h, manifest)
	root, err := os.MkdirTemp("", "czf")
	must(t, err)
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+h.server.URL+
		"\ntensorhub_token: rental-idle-test\ndaemon:\n  idle_shutdown_s: 0\n"), 0o600))
	witness := filepath.Join(root, "local-machine-launched")
	stubMachine(t, root, "#!/bin/sh\necho \"$@\" >> "+witness+"\nexit 3\n")
	t.Cleanup(func() {
		_, _ = runCozy(t, root, "down")
		if !t.Failed() {
			_ = removeAllForce(root)
		}
	})
	layout, problem := home.Open(root)
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	source := machines.Source{Host: *machineHostBinary, RuntimeWheel: *machineRuntimeWheel, TensorFSWheel: *machineTensorFSWheel}
	launch, identity, token, provider := providerHost(t, h, layout, source, uv)
	h.provider = provider
	h.mu.Lock()
	h.rentals[parityRental] = map[string]any{"rental_id": parityRental, "name": "tessa", "state": "ready",
		"requested_accelerator_model": "CPU", "accelerator_count": 1, "hourly_rate_usd_micros": 1,
		"worker_address": launch.Addr, "media_address": launch.MediaAddr, "worker_id": launch.WorkerID, "worker_boot_id": launch.BootID}
	h.mu.Unlock()
	fatal(t, rental.Attach(layout, store, records.Rental{ID: parityRental, MachineName: "tessa", SKU: "cpu", State: "ready",
		AcceleratorModel: "CPU", AcceleratorCount: 1, HourlyRateUSDMicros: 1, Hub: h.server.URL,
		Address: launch.Addr, MediaAddress: launch.MediaAddr, ExpectedWorkerID: launch.WorkerID, ExpectedWorkerBootID: launch.BootID},
		string(pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: launch.Leaf})), secret.New(token), identity))
	store.Close()
	up := exec.Command(*olderCozy, "up")
	up.Env = childEnv(t, root)
	if raw, err := up.CombinedOutput(); err != nil {
		t.Fatalf("the older cozy did not start its daemon: %v\n%s", err, raw)
	}

	// The stand-in hub serves bytes that are not the model: the rental's machine refuses them
	// by its own name, and that answer is the command's.
	code, out := runCozy(t, root, "model", "download", "proof/model#"+manifest, "--rental=tessa", "--json")
	t.Logf("model download under the older daemon [exit %d]:\n%s", code, out)
	if code == 0 || !strings.Contains(out, "tessa: ") || strings.Contains(out, "rental_install") || strings.Contains(out, "v1_absent") {
		t.Fatalf("the download did not end with the rental machine's own answer [exit %d]\n%s", code, out)
	}

	// `cozy model quantize --rental` is this command's own job on the rental's machine, which
	// fetches the checkpoint itself and refuses the stand-in's bytes; nothing is asked of the
	// older daemon.
	if code, out := runCozy(t, root, "package", "install", quantizeProject(t), "--editable"); code != 0 {
		t.Fatalf("editable install [exit %d]\n%s", code, out)
	}
	code, out = runCozy(t, root, "model", "quantize", "proof/model#"+manifest, "proof/quantized", "--fp8",
		"--package", "local/quantize-proof", "--rental=tessa", "--await", "--json")
	t.Logf("model quantize under the older daemon [exit %d]:\n%s", code, out)
	if code == 0 || !strings.Contains(out, "OBJECT_ID_MISMATCH") {
		t.Fatalf("the quantize job did not reach the rental's machine [exit %d]\n%s", code, out)
	}

	// The upload is granted to the rental itself (its Hub id and pinned leaf); the stand-in hub
	// refuses the grant, so nothing is made.
	code, out = runCozy(t, root, "model", "upload", "civitai://1", "proof/model", "--rental=tessa", "--json")
	t.Logf("model upload under the older daemon [exit %d]:\n%s", code, out)
	mu.Lock()
	asked := append([]map[string]any(nil), grants...)
	mu.Unlock()
	if code == 0 || !strings.Contains(out, "publication.grant_refused") || len(asked) != 1 {
		t.Fatalf("the upload did not ask the Hub to grant the rental [exit %d, %d grants]\n%s", code, len(asked), out)
	}
	if grant, _ := asked[0]["requested_grant"].(map[string]any); grant["machine_id"] != parityRental {
		t.Fatalf("the grant does not name the rental: %v", asked[0])
	}

	store, problem = records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	if queued, problem := store.PendingRentalInstalls(); problem != nil || len(queued) != 0 {
		t.Fatalf("an installation was queued for the older daemon: %v %v", queued, problem)
	}
	if raw, err := os.ReadFile(witness); err == nil {
		t.Fatalf("an installation on a rental started this computer's machine: %s", strings.TrimSpace(string(raw)))
	}

	// This client's own daemon names the foreground quantize run's machine by the rental.
	if code, out := runCozy(t, root, "down"); code != 0 {
		t.Fatalf("down [exit %d]\n%s", code, out)
	}
	code, out = runCozy(t, root, "run", "list", "--json")
	if code != 0 || !strings.Contains(out, `"machine":"tessa"`) || strings.Contains(out, `"machine":"endpoint-`) {
		t.Fatalf("run list does not name the rental the foreground run used [exit %d]\n%s", code, out)
	}
}

// quantizeProject is local/quantize-proof: one fp8 quantizer, a job that takes a model and
// declares the lane's weights output.
func quantizeProject(t *testing.T) string {
	t.Helper()
	project := filepath.Join(t.TempDir(), "quantize-proof")
	must(t, os.MkdirAll(project, 0o700))
	sources := ""
	if *machineRuntimeWheel != "" {
		sources = fmt.Sprintf("[tool.uv.sources]\ncozy-runtime={path=%q}\ntensorfs={path=%q}\n", *machineRuntimeWheel, *machineTensorFSWheel)
	}
	must(t, os.WriteFile(filepath.Join(project, "pyproject.toml"), []byte(`[project]
name="quantize-proof"
version="0.0.1"
requires-python=">=3.12,<3.13"
dependencies=["cozy-runtime>=`+runtimeFloor+`"]
[project.entry-points."cozy.application"]
default="quantize_proof:app"
`+sources+`[build-system]
requires=["hatchling"]
build-backend="hatchling.build"
[tool.hatch.build.targets.wheel]
only-include=["quantize_proof.py"]
`), 0o600))
	must(t, os.WriteFile(filepath.Join(project, "package.toml"), []byte("[application]\nobject=\"quantize_proof:app\"\n"), 0o600))
	must(t, os.WriteFile(filepath.Join(project, "quantize_proof.py"), []byte(`from cozy_runtime.author import App, Context, Loader, Model, ModelArtifact, WeightsOutput, invocable, uses_components

app = App()


class Demo(Model):
    def load(self, loader: Loader) -> None:
        pass

    @uses_components("transformer")
    def generate(self) -> object:
        return None


@invocable
async def fp8(ctx: Context, *, source: Demo) -> ModelArtifact:
    raise RuntimeError("the stand-in hub serves no model to quantize")


app.job(fp8, name="fp8", weights=(WeightsOutput("fp8", max_new_bytes=65536),), accelerator=False)
`), 0o600))
	if out, err := exec.Command("uv", "lock", "--project", project).CombinedOutput(); err != nil {
		t.Fatalf("locking the quantize package: %v\n%s", err, out)
	}
	return project
}
