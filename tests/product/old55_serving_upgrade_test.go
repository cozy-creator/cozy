package producttest

import (
	"bytes"
	"crypto/sha256"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/records"
	rentalpkg "github.com/cozy-creator/cozy/internal/rental"
)

var old55Proof = flag.String("old55-proof", "", "explicit retained old55 CLI, seed helper, launcher and matching old/new Runtime bins")

// The old binary creates its genuine schema39 queue. The current daemon migrates
// it without rewriting the private wheel/payload and refuses before an attempt
// when the old capture lacks the new immutable Runtime floor marker.
// No user's home, rental or provider is contacted by this proof.
func TestOld55QueuedPrivateServingRefusesUnprovenFloorAfterUpgrade(t *testing.T) {
	if *old55Proof == "" {
		t.Skip("requires exact old55 artifacts and task-owned actual CPU Host")
	}
	proof, err := filepath.Abs(*old55Proof)
	must(t, err)
	root, err := os.MkdirTemp(proof, "home-")
	must(t, err)
	t.Logf("retained proof home %s", root)
	oldCLI := filepath.Join(proof, "cozy-old55")
	raw, err := os.ReadFile(oldCLI)
	must(t, err)
	if fmt.Sprintf("%x", sha256.Sum256(raw)) != "88bc5f0ad1d87d8c6646b69c9dbe32810414a6e38c972502e9bf0fc64c6476f3" {
		t.Fatal("not the exact installed old55 binary")
	}
	seed := filepath.Join(proof, "seed")
	authority, err := exec.Command(seed, "authority", root).Output()
	must(t, err)
	authorityPath := filepath.Join(root, "authority.json")
	must(t, os.WriteFile(authorityPath, authority, 0600))
	launched, err := exec.Command("python3", filepath.Join(proof, "launch.py"), authorityPath).Output()
	must(t, err)
	var host actualChildHost
	must(t, json.Unmarshal(launched, &host))
	must(t, os.WriteFile(filepath.Join(root, "host-container.json"), launched, 0600))
	paused := false
	t.Cleanup(func() {
		if paused {
			_ = exec.Command("docker", "unpause", host.Container).Run()
		}
		if pid := daemonOnRoot(root); pid != 0 {
			if p, e := os.FindProcess(pid); e == nil {
				_ = p.Signal(os.Interrupt)
			}
		}
		_ = exec.Command("docker", "rm", "-f", host.Container).Run()
	})
	cert := filepath.Join(root, "worker.pem")
	waitUntil(t, "actual CPU Host certificate", func() bool {
		return exec.Command("docker", "cp", host.Container+":"+host.Certificate, cert).Run() == nil
	})
	certBytes, err := os.ReadFile(cert)
	must(t, err)
	pool := x509.NewCertPool()
	if !pool.AppendCertsFromPEM(certBytes) {
		t.Fatal("invalid owned Host certificate")
	}
	client := &http.Client{Transport: &http.Transport{TLSClientConfig: &tls.Config{RootCAs: pool, ServerName: "cozy-worker", MinVersion: tls.VersionTLS12}}, Timeout: time.Second}
	var ready struct {
		Payload []byte `json:"payload"`
	}
	waitUntil(t, "actual CPU Host55 readiness", func() bool {
		res, e := client.Get(host.Readiness)
		if e != nil {
			return false
		}
		defer res.Body.Close()
		return res.StatusCode == 200 && json.NewDecoder(res.Body).Decode(&ready) == nil && len(ready.Payload) > 0
	})
	var body struct {
		Boot string            `json:"pod_boot_id"`
		GPUs []json.RawMessage `json:"runtime_gpus"`
	}
	must(t, json.Unmarshal(ready.Payload, &body))
	if body.Boot == "" || len(body.GPUs) != 0 {
		t.Fatal("proof must use a real CPU-only worker")
	}
	facts, err := exec.Command("docker", "exec", host.Container, "python3", "-I", "-c", `import importlib.metadata as m,json,platform,re
from cozy.worker.v1.wire_version import WIRE_MINOR
assert WIRE_MINOR==55
rows={re.sub(r"[-_.]+","-",d.metadata["Name"]).lower():m.version(d.metadata["Name"]) for d in m.distributions()}
print(json.dumps({"format":"tensorhub.image_inventory/1","profile":"python3.12-cpu-linux-x86","python":platform.python_version(),"distributions":[{"name":n,"version":v} for n,v in sorted(rows.items())]}))`).Output()
	must(t, err)
	const rental = "pr-11111111111111111111"
	hub := newFakeRentalHub(t, 0)
	var catalog struct {
		URL string `json:"url"`
	}
	publicConfig, err := os.ReadFile(filepath.Join(proof, "catalog-public.json"))
	must(t, err)
	must(t, json.Unmarshal(publicConfig, &catalog))
	rentalHandler := hub.server.Config.Handler
	hub.server.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		readMetadata := r.Method == http.MethodGet && !strings.HasPrefix(r.URL.Path, "/v1/rentals") && r.URL.Path != "/v1/rental-skus"
		packageReadURLs := r.Method == http.MethodPost && r.URL.Path == "/v1/packages/paul/minimax-h3/download"
		if readMetadata || packageReadURLs {
			request, e := http.NewRequest(r.Method, strings.TrimSuffix(catalog.URL, "/")+r.URL.RequestURI(), r.Body)
			if e != nil {
				http.Error(w, e.Error(), 500)
				return
			}
			request.Header.Set("Content-Type", "application/json")
			response, e := (&http.Client{Timeout: time.Minute}).Do(request)
			if e != nil {
				http.Error(w, e.Error(), 502)
				return
			}
			defer response.Body.Close()
			w.Header().Set("Content-Type", response.Header.Get("Content-Type"))
			w.WriteHeader(response.StatusCode)
			_, _ = io.Copy(w, response.Body)
			return
		}
		rentalHandler.ServeHTTP(w, r)
	})

	hub.skus = []map[string]any{{"name": "cpu", "accelerator_model": "CPU", "accelerator_count": 1, "price_usd_micros_per_hour": 1, "base_worker_profile": "python3.12-cpu-linux-x86"}}
	hub.inventories = map[string]json.RawMessage{rental: facts}
	hub.rentals[rental] = map[string]any{"rental_id": rental, "name": "old55-worker", "state": "ready", "worker_address": host.Control, "media_address": strings.TrimPrefix(host.Media, "https://"), "cert_pem": string(certBytes), "worker_id": "old55-worker", "worker_boot_id": body.Boot, "accelerator_count": 1, "hourly_rate_usd_micros": 1}
	meta, _ := json.Marshal(map[string]string{"certificate": cert, "control": host.Control, "media": host.Media, "boot": body.Boot, "hub": hub.server.URL})
	metaPath := filepath.Join(root, "attach.json")
	must(t, os.WriteFile(metaPath, meta, 0600))
	if output, e := exec.Command(seed, "attach", root, metaPath).CombinedOutput(); e != nil {
		t.Fatalf("old schema seed: %v %s", e, output)
	}
	cfg := "tensorhub_url: " + hub.server.URL + "\ntensorhub_token: rental-idle-test\nport: 0\ndaemon:\n  idle_shutdown_s: 0\n"
	must(t, os.WriteFile(filepath.Join(root, "config.yaml"), []byte(cfg), 0600))
	oldBin := filepath.Join(proof, "old-runtime-bin")
	newBin := filepath.Join(proof, "new-runtime-bin")
	run := func(binary, bin string, args ...string) (int, string) {
		command := exec.Command(binary, args...)
		basePath := ""
		for _, item := range childEnv(t, root) {
			if value, ok := strings.CutPrefix(item, "PATH="); ok {
				basePath = value
			}
		}
		command.Env = childEnv(t, root, "PATH="+bin+string(os.PathListSeparator)+basePath, "CUDA_VISIBLE_DEVICES=")
		data, _ := command.CombinedOutput()
		if command.ProcessState == nil {
			return -1, string(data)
		}
		return command.ProcessState.ExitCode(), string(data)
	}
	project := filepath.Join(root, "project")
	copyCode := exec.Command("python3", "-c", `import shutil,sys,pathlib
shutil.copytree(sys.argv[1],sys.argv[2],ignore=shutil.ignore_patterns('.venv','__pycache__'))
p=pathlib.Path(sys.argv[2])/'weightless.py';p.write_text(p.read_text().replace('REVISION = "edited-after-submission"','REVISION = "first"'))`, filepath.Join(proof, "project"), project)
	if output, e := copyCode.CombinedOutput(); e != nil {
		t.Fatalf("copy proof source: %v %s", e, output)
	}
	if code, out := run(oldCLI, oldBin, "package", "install", project, "--editable", "--no-model-download", "--json"); code != 0 {
		t.Fatalf("old55 install [%d]: %s", code, out)
	}
	// With an already recorded rental and immutable source, suspension creates a
	// genuine queue window without manufacturing a request/attempt or GPU result.
	must(t, exec.Command("docker", "pause", host.Container).Run())
	paused = true
	outputDir := filepath.Join(root, "saved")
	code, out := run(oldCLI, oldBin, "run", localWeightlessRef+"/tile", "size=16", "seed=47", "--rental="+rental, "--idempotency-key=old55-queued", "--json", "--out="+outputDir)
	if code != 0 {
		t.Fatalf("old55 submission [%d]: %s", code, out)
	}
	snapshot, err := exec.Command(seed, "snapshot", root, "old55-queued").Output()
	must(t, err)
	must(t, os.WriteFile(filepath.Join(root, "before.json"), snapshot, 0600))
	var before records.Request
	must(t, json.Unmarshal(snapshot, &before))
	if before.ID == "" || (before.State != "queued" && before.State != "submitted") || before.LocalInstallationID == "" {
		t.Fatalf("not an old private queued request: %+v", before)
	}
	for _, model := range before.Models {
		if len(model.Adapters) > 0 {
			t.Fatal("this proof must be unadapted")
		}
	}
	oldPID := daemonOnRoot(root)
	if oldPID == 0 {
		t.Fatal("old daemon absent")
	}
	process, err := os.FindProcess(oldPID)
	must(t, err)
	must(t, process.Signal(syscall.SIGKILL))
	waitUntil(t, "old55 daemon process exits", func() bool { return syscall.Kill(oldPID, 0) != nil })
	source := filepath.Join(project, "weightless.py")
	text, err := os.ReadFile(source)
	must(t, err)
	must(t, os.WriteFile(source, bytes.Replace(text, []byte(`REVISION = "first"`), []byte(`REVISION = "edited-after-submission"`), 1), 0600))
	if code, out := run(cozyBin, newBin, "run", "list", "--json"); code != 0 {
		t.Fatalf("new daemon migration [%d]: %s", code, out)
	}
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	after, problem := store.RequestByIdempotencyKey("old55-queued")
	fatal(t, problem)
	if after.ID != before.ID || after.BodyDigest != before.BodyDigest || after.PlanID != before.PlanID || after.InstallID != before.InstallID || after.LocalInstallationID != before.LocalInstallationID || !bytes.Equal(after.Payload, before.Payload) {
		t.Fatal("upgrade rewrote immutable request/captured-code identity")
	}
	must(t, exec.Command("docker", "unpause", host.Container).Run())
	paused = false
	code, out = run(cozyBin, newBin, "run", "watch", before.ID, "--json")
	must(t, os.WriteFile(filepath.Join(root, "watch.jsonl"), []byte(out), 0600))

	if code == 0 || !strings.Contains(out, "request.capture_runtime_floor_unproven") {
		t.Fatalf("old capture did not fail closed at its exact gate [%d]: %s", code, out)
	}
	after, problem = store.RequestByIdempotencyKey("old55-queued")
	fatal(t, problem)
	if after.ID != before.ID || after.BodyDigest != before.BodyDigest || after.LocalInstallationID != before.LocalInstallationID || !bytes.Equal(after.Payload, before.Payload) {
		t.Fatal("refusal changed old request identity")
	}
	if attempts, e := store.Attempts(before.ID); e != nil || len(attempts) != 0 {
		t.Fatalf("unproved capture reached an attempt: %v %v", attempts, e)
	}
	if _, e := os.Stat(outputDir); e == nil {
		files, e := os.ReadDir(outputDir)
		must(t, e)
		if len(files) != 0 {
			t.Fatal("refused capture produced substitute output")
		}
	}
	// The real published H3 metadata is read-only. All machine/provider routes
	// still terminate in this isolated fixture; the CPU host is paused so no
	// checkpoint weights or GPU inference can start.
	must(t, exec.Command("docker", "pause", host.Container).Run())
	paused = true
	// The published-H3 admission arm uses a separate provider stand-in and the
	// actual measured CUDA image inventory. No GPU worker is claimed or run.
	const h3Rental = "pr-22222222222222222222"
	quiet, err := net.Listen("tcp", "127.0.0.1:0") //cozy:allow owned loopback provider stand-in; accepts no requests or code
	must(t, err)
	defer quiet.Close()
	layout, e := home.Open(root)
	fatal(t, e)
	identity, e := rentalpkg.PendingCreatorIdentity(layout, "h3-metadata-proof")
	fatal(t, e)
	token, e := rentalpkg.PendingMediaToken(layout, "h3-metadata-proof")
	fatal(t, e)
	fatal(t, rentalpkg.Attach(layout, store, records.Rental{ID: h3Rental, MachineName: "metadata-only-h3", SKU: "h100", AcceleratorModel: "NVIDIA H100", AcceleratorCount: 1, HourlyRateUSDMicros: 1, State: "ready", Hub: hub.server.URL, Address: quiet.Addr().String(), MediaAddress: quiet.Addr().String(), ExpectedWorkerID: "metadata-only-h3", ExpectedWorkerBootID: "metadata-only"}, string(certBytes), token, identity))
	gpuInventory, err := os.ReadFile(filepath.Join(proof, "gpu-image-inventory.json"))
	must(t, err)
	hub.mu.Lock()
	hub.inventories[h3Rental] = gpuInventory
	hub.rentals[h3Rental] = map[string]any{"rental_id": h3Rental, "name": "metadata-only-h3", "state": "ready", "worker_address": quiet.Addr().String(), "media_address": quiet.Addr().String(), "cert_pem": string(certBytes), "worker_id": "metadata-only-h3", "worker_boot_id": "metadata-only", "accelerator_count": 1, "hourly_rate_usd_micros": 1}
	hub.skus = append(hub.skus, map[string]any{"name": "h100", "accelerator_model": "NVIDIA H100", "accelerator_count": 1, "price_usd_micros_per_hour": 1, "base_worker_profile": "torch2.13.0-cu130-cp312-linux-x86", "compute_capability": "9.0", "vram_gb": 80, "minimum_ram_per_gpu_gb": 1})
	hub.mu.Unlock()
	if status, text := run(cozyBin, newBin, "package", "install", "paul/minimax-h3", "--version=1.14.3", "--json"); status != 0 {
		t.Fatalf("exact published H3 code install [%d]: %s", status, text)
	}
	h3Input := filepath.Join(proof, "h3-input.json")
	code, out = run(cozyBin, newBin, "run", "paul/minimax-h3/fl2va", "model.model=paul/minimax-h3@1.0.0-h3-audit.1/fp8-adaln-pruned", "--in="+h3Input, "--rental="+h3Rental, "--idempotency-key=new57-published-h3", "--json")
	must(t, os.WriteFile(filepath.Join(root, "published-h3-submit.json"), []byte(out), 0600))
	h3, e := store.RequestByIdempotencyKey("new57-published-h3")
	fatal(t, e)
	if h3 == nil {
		t.Fatalf("new normal H3 invocation never recorded [%d]: %s", code, out)
	}
	frozen, err := json.MarshalIndent(h3, "", "  ")
	must(t, err)
	must(t, os.WriteFile(filepath.Join(root, "published-h3-record.json"), frozen, 0600))
	if h3.Package != "paul/minimax-h3" || h3.Release != "1.14.3" || h3.LocalInstallationID != "" {
		t.Fatalf("published H3 unexpectedly became a private capture: %+v", h3)
	}
	t.Logf("new normal H3 submission %s remains published1.14.3 with no private-capture floor path; output %s", h3.ID, out)

	t.Logf("actual old55 schema39 request %s migrated unchanged to schema40 and refused before attempts at capture floor; body %s/code %s", before.ID, before.BodyDigest, before.LocalInstallationID)
}
