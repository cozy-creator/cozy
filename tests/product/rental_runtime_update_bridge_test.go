package producttest

import (
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
)

var rentalUpdateBridge = flag.String("rental-update-bridge", "", "isolated real CPU Host/SSH fixture JSON")

// The external guardian proof owns this CPU container and its ephemeral keys.
// This bridge runs the ordinary CLI against its real daemon, TLS control and SSH
// update path; the local Hub stub grants no provider purchase capability.
func TestRentalRuntimeUpdateCLIBridge(t *testing.T) {
	if *rentalUpdateBridge == "" {
		t.Skip("requires the separately launched CPU guardian")
	}
	var f struct {
		Root, Address, SSHAddress, SSHKey, CertificatePath, IdentityPath, WorkerID, WorkerBootID, CLI, Output string
		Target                                                                                                hub.RuntimeUpdateTarget
	}
	raw, err := os.ReadFile(*rentalUpdateBridge)
	must(t, err)
	must(t, json.Unmarshal(raw, &f))
	if !filepath.IsAbs(f.Root) || !filepath.IsAbs(f.CLI) {
		t.Fatal("fixture must use fresh absolute paths")
	}
	layout, problem := home.Open(f.Root)
	fatal(t, problem)
	if _, err := os.Stat(layout.DB); !os.IsNotExist(err) {
		t.Fatal("fixture refuses existing Creator database")
	}
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	const id = "pr-11111111111111111111"
	must(t, os.MkdirAll(layout.Rentals, 0700))
	key, err := os.ReadFile(f.IdentityPath)
	must(t, err)
	must(t, os.WriteFile(layout.RentalCreatorIdentity(id), key, 0600))
	identity, problem := rental.CreatorIdentityFor(layout, id)
	fatal(t, problem)
	cert, err := os.ReadFile(f.CertificatePath)
	must(t, err)
	token, problem := rental.PendingMediaToken(layout, "isolated-update-cli")
	fatal(t, problem)
	remote := map[string]any{"rental_id": id, "name": "cpu-proof", "state": "ready", "development": true, "ssh_address": f.SSHAddress, "worker_address": f.Address, "media_address": "https://127.0.0.1:1", "cert_pem": string(cert), "worker_id": f.WorkerID, "worker_boot_id": f.WorkerBootID, "creator_public_key": identity.PublicKey(), "media_token_sha256": []string{strings.Repeat("a", 64)}, "requested_accelerator_model": "CPU", "accelerator_count": 1, "hourly_rate_usd_micros": 1}
	f.Target.RentalID, f.Target.WorkerBootID = id, f.WorkerBootID
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "GET" {
			http.Error(w, "fixture forbids provider writes", 405)
			return
		}
		switch r.URL.Path {
		case "/v1/rentals":
			_ = json.NewEncoder(w).Encode([]any{remote})
		case "/v1/rentals/" + id:
			_ = json.NewEncoder(w).Encode(remote)
		case "/v1/rentals/" + id + "/runtime-update-target":
			_ = json.NewEncoder(w).Encode(f.Target)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()
	fatal(t, rental.Attach(layout, store, records.Rental{AcceleratorCount: 1, ID: id, MachineName: "cpu-proof", SKU: "cpu", AcceleratorModel: "CPU", HourlyRateUSDMicros: 1, State: "ready", Hub: server.URL, Address: f.Address, ExpectedWorkerID: f.WorkerID, ExpectedWorkerBootID: f.WorkerBootID}, string(cert), token, identity))
	must(t, os.WriteFile(filepath.Join(f.Root, config.FileName), []byte(fmt.Sprintf("tensorhub_url: %s\ntensorhub_token: isolated-proof\nrentals:\n  ssh_public_key: %s.pub\n  idle_release_s: 0\ndaemon:\n  idle_shutdown_s: 0\n", server.URL, f.SSHKey)), 0600))
	command := func(args ...string) *exec.Cmd {
		cmd := exec.Command(f.CLI, args...)
		cmd.Env = childEnv(t, f.Root)
		return cmd
	}
	defer func() { _ = command("down").Run() }()
	log, err := os.Create(filepath.Join(f.Output, "first-cli.log"))
	must(t, err)
	defer log.Close()
	first := command("rental", "update", "cpu-proof", "--json", "--full")
	first.Stdout, first.Stderr = log, log
	must(t, first.Start())
	completed := make(chan error, 1)
	go func() { completed <- first.Wait() }()
	var operation string
	deadline := time.Now().Add(2 * time.Minute)
	for operation == "" && time.Now().Before(deadline) {
		row, problem := store.RuntimeUpdate(id)
		fatal(t, problem)
		if row != nil && (row.State == "updating" || row.State == "reconciling") {
			operation = row.ID
			break
		}
		select {
		case err := <-completed:
			t.Fatalf("update CLI exited before disconnect point: %v; %s", err, tail(log.Name()))
		default:
		}
		time.Sleep(20 * time.Millisecond)
	}
	if operation == "" {
		_ = first.Process.Kill()
		t.Fatal("update never reached guardian enqueue")
	}
	must(t, first.Process.Kill())
	<-completed
	second := command("rental", "update", "cpu-proof", "--json", "--full")
	result, err := second.CombinedOutput()
	must(t, os.WriteFile(filepath.Join(f.Output, "reconnected-cli.json"), result, 0600))
	if err != nil {
		t.Fatalf("reconnected update failed: %v: %s", err, result)
	}
	row, problem := store.RuntimeUpdate(id)
	fatal(t, problem)
	if row == nil || row.ID != operation || row.State != "succeeded" {
		t.Fatalf("observer loss changed operation identity or completion: %+v", row)
	}
	result, err = command("rental", "update", "cpu-proof", "--json", "--full").CombinedOutput()
	must(t, os.WriteFile(filepath.Join(f.Output, "already-current-cli.json"), result, 0600))
	if err != nil || !strings.Contains(string(result), "already current") {
		t.Fatalf("duplicate update was not unchanged: %v: %s", err, result)
	}
	fmt.Printf("RENTAL_UPDATE_CLI %s\n", operation)
}
