package producttest

import (
	"encoding/base64"
	"encoding/json"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
)

// An ingest placed with --rental-only does not land on an attached rental whose bought disk
// cannot hold it (production run 1235: a 30 GB pod for a 210 GB source failed
// insufficient_storage); a pod sized to the ingest is bought instead. A rental whose disk
// fits is used, and one whose Hub reports no disk is unknown, not excluded.
func TestIngestPlacementHonoursTheAttachedRentalsDisk(t *testing.T) {
	probe := &http.Client{Timeout: 20 * time.Second}
	response, err := probe.Get("https://civitai.com/api/v1/model-versions/128078")
	if err != nil {
		t.Skipf("provider unreachable from this runner: %v", err)
	}
	var version struct {
		Files []struct {
			Primary bool    `json:"primary"`
			SizeKB  float64 `json:"sizeKB"`
		} `json:"files"`
	}
	must(t, json.NewDecoder(response.Body).Decode(&version))
	response.Body.Close()
	var planned int64
	for _, file := range version.Files {
		if file.Primary {
			planned = int64(file.SizeKB * 1024)
		}
	}
	for _, arm := range []struct {
		name string
		disk int // 0: the Hub reports none
		buys bool
	}{{"too small", 30, true}, {"fits", 100, false}, {"unreported", 0, false}} {
		t.Run(arm.name, func(t *testing.T) {
			bought, request := ingestOnAttachedRental(t, arm.disk)
			if !arm.buys {
				if len(bought) != 0 || request.Worker != podRental {
					t.Fatalf("an attached rental with a %d GB disk was passed over: bought %v, placed on %q", arm.disk, bought, request.Worker)
				}
				return
			}
			if len(bought) == 0 || bought[0]["planned_source_bytes"] != float64(planned) || request.Worker == podRental {
				t.Fatalf("the ingest was not placed on a pod sized to it: bought %v, placed on %q", bought, request.Worker)
			}
		})
	}
}

// ingestOnAttachedRental submits a real rented ingest beside one attached rental whose Hub
// view reports diskGB, and answers the paid asks and the request once placement settles.
func ingestOnAttachedRental(t *testing.T, diskGB int) ([]map[string]any, records.Request) {
	t.Helper()
	root, err := os.MkdirTemp(scratchBase, "disk-fit-")
	must(t, err)
	t.Cleanup(func() { _ = removeAllForce(root) })
	layout, problem := home.Open(root)
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	t.Cleanup(store.Close)
	identity, problem := rental.PendingCreatorIdentity(layout, "disk-fit")
	fatal(t, problem)
	public, err := base64.RawURLEncoding.DecodeString(identity.PublicKey())
	must(t, err)
	connection, certPath := startFakePod(t, root, &fakePod{controlKey: public})
	cert, err := os.ReadFile(certPath)
	must(t, err)

	stand := newFakeRentalHub(t, 0)
	stand.publishListing()
	stand.setSKUs(map[string]any{"name": "cpu", "accelerator_model": "CPU", "accelerator_count": 1,
		"price_usd_micros_per_hour": 100_000, "base_worker_profile": "python3.12-cpu-linux-x86"})
	stand.add(podRental, "attached")
	if diskGB > 0 {
		stand.set(podRental, "container_disk_gb", diskGB)
	}
	var bought []map[string]any
	stand.mu.Lock()
	stand.inventories = map[string]json.RawMessage{podRental: json.RawMessage(`{"format":"tensorhub.image_inventory/1",` +
		`"profile":"python3.12-cpu-linux-x86","python":"3.12.12","interpreters":[{"version":"3.12.12","abi":"cp312"}],"distributions":[]}`)}
	stand.rent = func(request map[string]any) map[string]any {
		bought = append(bought, request)
		return map[string]any{"rental_id": "pr-disk-fit-bought", "name": request["name"], "state": "pending_acquisition",
			"requested_accelerator_model": "CPU", "accelerator_count": 1, "hourly_rate_usd_micros": 100_000}
	}
	stand.mu.Unlock()
	fatal(t, rental.Attach(layout, store, records.Rental{ID: podRental, MachineName: "attached", State: "ready", SKU: "cpu",
		AcceleratorModel: "CPU", AcceleratorCount: 1, HourlyRateUSDMicros: 100_000, Hub: stand.server.URL,
		Address: connection.Addr, MediaAddress: connection.Media.Addr, ExpectedWorkerID: podWorkerID,
		ExpectedWorkerBootID: podBootID}, string(cert), connection.Media.Token, identity))
	// A static Hub token publishes to its named org without an account read.
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+stand.server.URL+
		"\ntensorhub_token: rental-idle-test\ndaemon:\n  idle_shutdown_s: 0\n"), 0o600))
	startDaemonProcess(t, root)

	cmd := exec.Command("/usr/bin/nice", "-n", "19", cozyBin, "model", "upload", "civitai://128078", "proof/sdxl",
		"--source-profile", "civitai/sdxl/single-file/1", "--rental-only", "--json")
	cmd.Env = childEnv(t, root)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("the rented ingest was not submitted: %v\n%s", err, out)
	}
	var request records.Request
	waitFor(t, root, "the ingest to be placed", func() bool {
		stand.mu.Lock()
		asked := len(bought) > 0
		stand.mu.Unlock()
		rows, problem := store.Requests("", "", 10)
		if problem != nil || len(rows) == 0 {
			return false
		}
		request = rows[0]
		return asked || request.Worker != ""
	})
	stand.mu.Lock()
	defer stand.mu.Unlock()
	return append([]map[string]any(nil), bought...), request
}
