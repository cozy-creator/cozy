package producttest

import (
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"flag"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	"github.com/cozy-creator/cozy/internal/secret"
)

var childHostLauncher = flag.String("child-host-launcher", "", "actual isolated PodHost container launcher")
var childHostHome = flag.String("child-host-home", "", "new retained proof home; kept for native custody inspection")
var childHostProject = flag.String("child-host-project", "", "private source/candidate artifact fixture")
var childHostRuntimeBin = flag.String("child-host-runtime-bin", "", "installed matching Runtime bin directory")

// Only the provider catalog/readback is a test peer. Both control planes, package
// preparation, signed native effects and executors run in the actual Host image.
func TestPrivateChildActualHostArtifacts(t *testing.T) {
	if *childHostLauncher == "" || *childHostHome == "" || *childHostProject == "" || *childHostRuntimeBin == "" {
		t.Skip("requires an explicit task-owned actual Host image and artifact fixture")
	}
	layout, problem := home.Open(*childHostHome)
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	identity, problem := rental.PendingCreatorIdentity(layout, "child-host-proof")
	fatal(t, problem)
	token, problem := rental.PendingMediaToken(layout, "child-host-proof")
	fatal(t, problem)
	authority, err := json.Marshal(map[string]any{"worker_id": "private-child-host", "control_public_key_ed25519_b64url": identity.PublicKey(), "media_token_sha256": []string{secret.HashHex(token)}})
	must(t, err)
	authorityPath := filepath.Join(layout.Root, "host-authority.json")
	must(t, os.WriteFile(authorityPath, authority, 0o600))
	started, err := os.ReadFile(filepath.Join(layout.Root, "host-container.json"))
	if os.IsNotExist(err) {
		command := exec.Command("python3", *childHostLauncher, authorityPath)
		started, err = command.Output()
		must(t, err)
		must(t, os.WriteFile(filepath.Join(layout.Root, "host-container.json"), started, 0o600))
	}
	must(t, err)
	var host struct {
		Container   string `json:"container"`
		Control     string `json:"control_address"`
		Media       string `json:"media_url"`
		Certificate string `json:"tls_certificate_path_in_container"`
		Readiness   string `json:"readiness_url"`
	}
	must(t, json.Unmarshal(started, &host))
	host.Media = strings.TrimPrefix(host.Media, "https://")
	t.Logf("actual Host container %s retained in %s", host.Container, layout.Root)
	copyFromHost := func(source, destination string) {
		t.Helper()
		for deadline := time.Now().Add(time.Minute); ; {
			if out, err := exec.Command("docker", "cp", host.Container+":"+source, destination).CombinedOutput(); err == nil {
				return
			} else if time.Now().After(deadline) {
				t.Fatalf("actual Host readiness unavailable: %v %s", err, out)
			}
			time.Sleep(200 * time.Millisecond)
		}
	}
	certificatePath := filepath.Join(layout.Root, "host-certificate.pem")
	readinessPath := filepath.Join(layout.Root, "host-readiness.json")
	copyFromHost(host.Certificate, certificatePath)
	certificate, err := os.ReadFile(certificatePath)
	must(t, err)
	pool := x509.NewCertPool()
	if !pool.AppendCertsFromPEM(certificate) {
		t.Fatal("Host certificate is not PEM")
	}
	client := &http.Client{Transport: &http.Transport{TLSClientConfig: &tls.Config{RootCAs: pool, ServerName: "cozy-worker", MinVersion: tls.VersionTLS12}}, Timeout: 10 * time.Second}
	var envelope struct {
		Payload []byte `json:"payload"`
	}
	for deadline := time.Now().Add(time.Minute); ; {
		response, callError := client.Get(host.Readiness)
		if callError == nil {
			raw, readError := io.ReadAll(io.LimitReader(response.Body, 32<<10))
			response.Body.Close()
			if readError == nil && response.StatusCode == http.StatusOK && json.Unmarshal(raw, &envelope) == nil && len(envelope.Payload) > 0 {
				break
			}
		}
		if time.Now().After(deadline) {
			t.Fatal("public Host readiness did not finish its probes")
		}
		time.Sleep(200 * time.Millisecond)
	}
	readiness := envelope.Payload
	must(t, os.WriteFile(readinessPath, readiness, 0o600))
	var ready struct {
		WorkerBootID string `json:"pod_boot_id"`
	}
	must(t, json.Unmarshal(readiness, &ready))
	if ready.WorkerBootID == "" {
		t.Fatalf("actual Host readiness has no worker boot: %s", readiness)
	}
	hub := newFakeRentalHub(t, 0)
	hub.skus = []map[string]any{{"name": "cpu", "accelerator_model": "CPU", "price_usd_micros_per_hour": 1, "base_worker_profile": "python3.12-cpu-linux-x86"}}
	const rentalID = "rental-private-child-host"
	hub.rentals[rentalID] = map[string]any{"rental_id": rentalID, "name": "child-host", "state": "ready", "worker_address": host.Control, "media_address": host.Media, "cert_pem": string(certificate), "worker_id": "private-child-host", "worker_boot_id": ready.WorkerBootID, "creator_public_key": identity.PublicKey(), "media_token_sha256": []string{secret.HashHex(token)}, "hourly_rate_usd_micros": 1}
	fatal(t, rental.Attach(layout, store, records.Rental{ID: rentalID, MachineName: "child-host", SKU: "cpu", AcceleratorModel: "CPU", HourlyRateUSDMicros: 1, State: "ready", Hub: hub.server.URL, Address: host.Control, MediaAddress: host.Media, ExpectedWorkerID: "private-child-host", ExpectedWorkerBootID: ready.WorkerBootID}, string(certificate), token, identity))
	must(t, os.WriteFile(filepath.Join(layout.Root, "config.yaml"), []byte("tensorhub_url: "+hub.server.URL+"\ntensorhub_token: rental-idle-test\nport: 0\nrentals:\n  max_hourly_spend_usd: 1\n  idle_release_s: 0\ndaemon:\n  idle_shutdown_s: 0\n"), 0o600))
	path := *childHostRuntimeBin
	for _, item := range childEnv(t, layout.Root) {
		if strings.HasPrefix(item, "PATH=") {
			path += string(os.PathListSeparator) + strings.TrimPrefix(item, "PATH=")
		}
	}
	script := filepath.Join(*childHostProject, "recipe.py")
	status, out := runCozyPath(t, layout.Root, path, "run", script, "--rental-only", "--await", "--json")
	if status == 0 || !strings.Contains(out, "candidate quality gate failed") {
		t.Fatalf("actual artifact source/candidate failed [%d]: %s", status, out)
	}
	first, problem := store.RequestByReference("1")
	fatal(t, problem)
	children, problem := store.Children(first.ID)
	fatal(t, problem)
	if len(children) != 2 || children[0].State != "succeeded" || children[1].State != "blocked" {
		t.Fatalf("native source was not retained before candidate failure: %+v", children)
	}
	code, err := os.ReadFile(script)
	must(t, err)
	must(t, os.WriteFile(script, []byte(strings.Replace(string(code), "factor=0", "factor=2", 1)), 0o600))
	status, out = runCozyPath(t, layout.Root, path, "run", script, "--retry", "1", "--rental-only", "--await", "--json")
	if status != 0 {
		t.Fatalf("actual artifact corrected run failed [%d]: %s", status, out)
	}
	t.Logf("actual retained artifact corrected result: %s", out)
	second, problem := store.RequestByReference("4")
	fatal(t, problem)
	nextChildren, problem := store.Children(second.ID)
	fatal(t, problem)
	if len(nextChildren) != 2 || nextChildren[0].Ordinal != 0 || nextChildren[0].ReusedFrom != children[0].ID || nextChildren[1].Ordinal != 1 {
		t.Fatalf("corrected transaction did not independently reuse A and execute B: %+v", nextChildren)
	}
	holds, problem := store.WeightsRetentions(nextChildren[0].ID)
	fatal(t, problem)
	if len(holds) == 0 {
		t.Fatal("reused A has no independent native result root")
	}
	for _, hold := range holds {
		if hold.State != "held" || hold.ProducerRequestID != children[0].ID {
			t.Fatalf("reused A changed original receipt custody: %+v", hold)
		}
	}
}
