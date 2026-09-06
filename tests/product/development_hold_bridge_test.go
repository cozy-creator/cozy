package producttest

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/cli"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
)

var developmentHoldBridge = flag.String("development-hold-bridge", "", "private isolated Runtime wrapper fixture JSON")

// TestDevelopmentHoldBridge is armed only by the isolated SSH/Runtime proof. It
// uses the real owner/record APIs and never prepares or submits any work.
func TestDevelopmentHoldBridge(t *testing.T) {
	if *developmentHoldBridge == "" {
		t.Skip("requires the separately launched real Runtime wrapper")
	}
	var fixture struct {
		Root, Address, CertificatePath, IdentityPath, WorkerID, WorkerBootID string
	}
	raw, err := os.ReadFile(*developmentHoldBridge)
	must(t, err)
	must(t, json.Unmarshal(raw, &fixture))
	if !filepath.IsAbs(fixture.Root) || fixture.WorkerID == "" || fixture.WorkerBootID == "" {
		t.Fatal("bridge requires a fresh absolute root and exact peer identity")
	}
	layout, problem := home.Open(fixture.Root)
	fatal(t, problem)
	if _, err := os.Stat(layout.DB); !os.IsNotExist(err) {
		t.Fatal("bridge refuses an existing Creator database")
	}
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	const rentalID = "pr-11111111111111111111"
	must(t, os.MkdirAll(layout.Rentals, 0700))
	key, err := os.ReadFile(fixture.IdentityPath)
	must(t, err)
	must(t, os.WriteFile(layout.RentalCreatorIdentity(rentalID), key, 0600))
	identity, problem := rental.CreatorIdentityFor(layout, rentalID)
	fatal(t, problem)
	certificate, err := os.ReadFile(fixture.CertificatePath)
	must(t, err)
	token, problem := rental.PendingMediaToken(layout, "isolated-development-hold")
	fatal(t, problem)
	cfg := config.Config{Home: fixture.Root, HubURL: "http://127.0.0.1:1"}
	fatal(t, rental.Attach(layout, store, records.Rental{ID: rentalID, MachineName: "isolated-development-proof",
		SKU: "cpu", AcceleratorModel: "CPU", HourlyRateUSDMicros: 1, State: "ready", Hub: cfg.HubURL,
		Address: fixture.Address, ExpectedWorkerID: fixture.WorkerID, ExpectedWorkerBootID: fixture.WorkerBootID},
		string(certificate), token, identity))
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() { scanner := bufio.NewScanner(os.Stdin); scanner.Scan(); cancel() }()
	problem = cli.HoldStoredDevelopmentWorker(ctx, cfg, rentalID, fixture.WorkerBootID, os.Stderr, func(state cli.DevelopmentHoldResult) {
		body, err := json.Marshal(state)
		if err != nil {
			cancel()
			return
		}
		fmt.Printf("DEVELOPMENT %s\n", body)
	})
	fatal(t, problem)
}
