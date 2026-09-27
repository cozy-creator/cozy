package producttest

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/records"
)

// A managed purchase whose pod the Hub never readies holds only itself. While it is
// acquiring, `cozy rental list` answers and shows it with its age, a second independent
// purchase reaches the Hub, and another rental ends. Before, the purchase held the fleet
// lock through the whole readiness wait, and every one of those hung behind it.
func TestStuckRentalPurchaseBlocksOnlyItself(t *testing.T) {
	root := t.TempDir()
	port := reservePort(t)
	origin := fmt.Sprintf("http://127.0.0.1:%d", port)
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+origin+
		"\ntensorhub_token: rental-idle-test\ndaemon:\n  idle_shutdown_s: 0\n"), 0o600))
	peer := newFakeRentalHub(t, port)
	// The second package is a real published job, so `cozy run` can submit it.
	iface := []byte(`{"application":"proof:app","entrypoints":[],"format":"cozy.package.interface/1","jobs":[{"name":"convert","publishes":false,"request":{"fields":[]},"result":{"fields":[]}}]}`)
	other := rentalReleaseFacts()
	other.PackageInterface = iface
	other.Release.PackageInterfaceDigest, other.Release.PackageInterfaceLength = assessmentDigest(iface), int64(len(iface))
	peer.packageReleases = map[string]any{"proof/source-producer@1": rentalReleaseFacts(), "proof/other-producer@1": other}
	peer.server.Config.Handler.(*http.ServeMux).HandleFunc("GET /v1/packages/proof/other-producer",
		func(w http.ResponseWriter, _ *http.Request) {
			_ = json.NewEncoder(w).Encode(hub.PackageCard{Package: hub.Resource{Org: "proof", Name: "other-producer"},
				Releases: []hub.ReleaseSummary{{Release: "1"}}})
		})
	peer.setSKUs(map[string]any{"name": "cpu", "accelerator_model": "CPU", "accelerator_count": 1,
		"price_usd_micros_per_hour": 100_000, "base_worker_profile": "python3.12-cpu-linux-x86"})
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()

	peer.add("rental-other", "heron")
	fatal(t, store.RecordRental(records.Rental{ID: "rental-other", MachineName: "heron", SKU: "cpu",
		AcceleratorModel: "CPU", AcceleratorCount: 1, HourlyRateUSDMicros: 100_000, State: "ready", Hub: origin}))
	// The daemon resumes this job at start and buys it a machine of its own.
	_, _, problem = store.Submit(records.Request{ID: "job-stuck", IdemKey: "job-stuck", BodyDigest: "sha256:" + strings.Repeat("7", 64),
		Package: "proof/source-producer", Release: "1", Entrypoint: "convert", Kind: "job", Rental: true, RentNew: true,
		Payload: []byte("{}"), Outputs: "[]", WeightsOutputs: "[]"})
	fatal(t, problem)
	var mu sync.Mutex
	var bought []string
	peer.rent = func(body map[string]any) map[string]any {
		mu.Lock()
		defer mu.Unlock()
		id := fmt.Sprintf("pr-stuck-%d", len(bought)+1)
		bought = append(bought, id)
		// Accepted, and never readied: the Hub keeps saying `acquiring`.
		return map[string]any{"rental_id": id, "name": body["name"], "state": "acquiring",
			"requested_accelerator_model": "CPU", "accelerator_count": 1, "hourly_rate_usd_micros": 100_000}
	}
	purchases := func() []string {
		mu.Lock()
		defer mu.Unlock()
		return append([]string(nil), bought...)
	}
	startDaemonProcess(t, root)
	waitFor(t, root, "the first purchase to reach the Hub", func() bool { return len(purchases()) > 0 })

	first := purchases()[0]
	var listed stuckListing
	waitFor(t, root, "the stuck purchase to be listed as acquiring", func() bool {
		listed = listStuck(t, root)
		return listed.acquiring(first)
	})
	// Another package's job, submitted now, insists on a fresh machine too.
	code, out, errs := runCozyWithin(t, root, "run", "proof/other-producer/convert", "--rental-only", "--rent-new",
		"--idempotency-key", "job-second", "--json")
	if code != 0 {
		t.Fatalf("a second run could not be submitted beside the stuck purchase [exit %d]: %s%s", code, out, errs)
	}
	waitFor(t, root, "a second independent purchase to reach the Hub", func() bool { return len(purchases()) == 2 })

	code, out, errs = runCozyWithin(t, root, "rental", "end", "rental-other", "--json")
	if code != 0 || peer.releases("rental-other") != 1 {
		t.Fatalf("another rental did not end beside the stuck purchase [exit %d]: %s%s", code, out, errs)
	}
	listed = listStuck(t, root)
	for _, id := range purchases() {
		if !listed.acquiring(id) {
			t.Fatalf("purchase %s left the listing while acquiring: %+v", id, listed.Rentals)
		}
	}
}

type stuckListing struct {
	Rentals []struct {
		ID       string `json:"rental_id"`
		State    string `json:"state"`
		RentedAt string `json:"rented_at"`
	} `json:"rentals"`
}

// acquiring is whether the listing shows the rental acquiring, with the moment it began.
func (l stuckListing) acquiring(id string) bool {
	for _, row := range l.Rentals {
		if row.ID == id {
			_, err := time.Parse(time.RFC3339Nano, row.RentedAt)
			return row.State == "acquiring" && err == nil
		}
	}
	return false
}

func listStuck(t *testing.T, root string) stuckListing {
	t.Helper()
	code, out, errs := runCozyWithin(t, root, "rental", "list", "--json", "--full")
	var listed stuckListing
	if code != 0 || json.Unmarshal([]byte(out), &listed) != nil {
		t.Fatalf("rental list failed [exit %d]: %s%s", code, out, errs)
	}
	return listed
}

// runCozyWithin runs one CLI verb under the test's own deadline; a verb still running
// at it is killed and reported as hung, never waited on.
func runCozyWithin(t *testing.T, root string, args ...string) (int, string, string) {
	t.Helper()
	const within = 20 * time.Second
	ctx, cancel := context.WithTimeout(context.Background(), within)
	defer cancel()
	cmd := exec.CommandContext(ctx, "/usr/bin/nice", append([]string{"-n", "19", cozyBin}, args...)...)
	cmd.Env = childEnv(t, root)
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	_ = cmd.Run()
	if ctx.Err() != nil {
		t.Fatalf("cozy %s hung past %s\n%s", strings.Join(args, " "), within,
			tail(filepath.Join(root, "daemon.log")))
	}
	return cmd.ProcessState.ExitCode(), stdout.String(), stderr.String()
}
