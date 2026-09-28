package producttest

import (
	"os"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
)

const (
	updateWorkerID = "dev-worker"
	updateBootID   = "dev-boot"
)

// updateFixture is one owned, ready rental recorded in a fresh home.
type updateFixture struct {
	cfg            config.Config
	layout         home.Layout
	store          *records.Store
	cert, rentalID string
}

func updateFixtureAt(t *testing.T) updateFixture {
	t.Helper()
	root := t.TempDir()
	layout, problem := home.Open(root)
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	t.Cleanup(store.Close)
	id := "pr-11111111111111111111"
	adoptCreatorIdentity(t, layout, id)
	return updateFixture{cfg: config.Config{Home: root, HubURL: "http://127.0.0.1:1"}, layout: layout, store: store,
		cert: standInCertificate(t, root), rentalID: id}
}

func (f updateFixture) attach(t *testing.T, address string) {
	t.Helper()
	identity, problem := rental.CreatorIdentityFor(f.layout, f.rentalID)
	fatal(t, problem)
	token, problem := rental.PendingMediaToken(f.layout, "runtime-update-proof")
	fatal(t, problem)
	certificate, err := os.ReadFile(f.cert)
	must(t, err)
	fatal(t, rental.Attach(f.layout, f.store, records.Rental{AcceleratorCount: 1, ID: f.rentalID, State: "ready", Hub: f.cfg.HubURL,
		MachineName: "proof", SKU: "cpu", AcceleratorModel: "CPU", HourlyRateUSDMicros: 1, Address: address,
		ExpectedWorkerID: updateWorkerID, ExpectedWorkerBootID: updateBootID}, string(certificate), token, identity))
}
