package producttest

import (
	"encoding/json"
	"encoding/pem"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/media"
	"github.com/cozy-creator/cozy/internal/mediawire"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/rental"
	"github.com/cozy-creator/cozy/internal/secret"
)

func TestRentalDetachStopsLiveControlBeforeCredentialRemoval(t *testing.T) {
	owner := hostOwner(t, "rental-detach")
	revision := mediawire.ContractRev
	mediaServer := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/health" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(mediawire.Health{
			Service: mediawire.Service, ContractRev: &revision,
		})
	}))
	defer mediaServer.Close()

	rentalID := "rnt-detach-proof"
	certPath := owner.l.RentalCert(rentalID)
	certificate := mediaServer.TLS.Certificates[0].Certificate[0]
	must(t, os.MkdirAll(owner.l.Rentals, 0o700))
	must(t, os.WriteFile(certPath,
		pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: certificate}), 0o644))

	spec := orchestrator.WorkerLaunchSpec{
		Connection: &orchestrator.WorkerConnection{
			RentalID: rentalID,
			Addr:     "127.0.0.1:1",
			CACert:   certPath,
			Media: &media.Spec{
				Addr:   strings.TrimPrefix(mediaServer.URL, "https://"),
				Token:  secret.Mint(),
				CACert: certPath,
			},
		},
		Placement: orchestrator.DesiredPlacement{Package: "proof/marco"},
	}
	_, change, problem := owner.c.EnsureWorker(spec)
	fatal(t, problem)
	if change != orchestrator.ChangeWorkerStarted {
		t.Fatalf("remote worker change = %q", change)
	}
	waitControlRetries(t, owner, 2)

	if !owner.c.DetachRental(rentalID) {
		t.Fatal("live rental slot was not detached")
	}
	if _, problem := rental.Forget(owner.l, owner.store, rentalID); problem != nil {
		t.Fatal(problem)
	}
	if _, err := os.Stat(certPath); !os.IsNotExist(err) {
		t.Fatalf("pinned certificate survived completed detach: %v", err)
	}
	eventsBefore := len(owner.c.Events())
	time.Sleep(450 * time.Millisecond) // more than two former 200 ms retry periods
	for _, event := range owner.c.Events()[eventsBefore:] {
		if strings.Contains(event, "pinned worker cert") {
			t.Fatalf("control retried after detach and credential removal: %s", event)
		}
	}
	if owner.c.DetachRental(rentalID) {
		t.Fatal("repeated detach was not idempotent")
	}
}

func TestRentalDetachRouteIsProcessIdempotent(t *testing.T) {
	daemon := startDaemonProcess(t, t.TempDir())
	for i := 0; i < 2; i++ {
		response := daemon.call(t, http.MethodDelete,
			"/v1/local/rentals/rnt-already-detached/claim", nil)
		var answer struct {
			Changed bool `json:"changed"`
		}
		if response.Status != http.StatusOK || json.Unmarshal(response.Body, &answer) != nil || answer.Changed {
			t.Fatalf("idempotent process detach = %s", response.brief())
		}
	}
}

func waitControlRetries(t *testing.T, owner *owner, wanted int) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if countEvents(owner, "control stream ended") >= wanted {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("control loop did not exhibit %d live retries: %v", wanted, owner.c.Events())
}
