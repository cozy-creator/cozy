package producttest

import (
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/cli"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
)

// th-150 arm 2. The catalog is live provider inventory, so a name it does not
// carry means one of two opposite things, and both used to be spelled
// "Tensorhub currently offers no rental SKU". On 2026-09-04 an explicit
// `cozy rental new rtx-a4000` was refused during a 32-minute stock-out; the
// message read as "no such machine"; the conclusion drawn was that the rental
// code had substituted a dearer card. It had not.
func refusalCatalog() []hub.RentalSKU {
	return []hub.RentalSKU{
		{Name: "rtx-4090", AcceleratorModel: "NVIDIA GeForce RTX 4090",
			PriceUSDMicrosPerHour: 740_000, StorageUSDMicrosPerHour: 213_504},
		{Name: "cpu", AcceleratorModel: "CPU", PriceUSDMicrosPerHour: 70_000},
	}
}

func TestASKURefusalNamesWhichAbsenceItHit(t *testing.T) {
	// The live moment: refused at 07:12:33Z, last A4000 offer 06:48:08Z.
	now := time.Date(2026, 9, 4, 7, 12, 33, 0, time.UTC)
	lastSeen := time.Date(2026, 9, 4, 6, 48, 8, 0, time.UTC)

	stockOut := cli.SKURefusal("rtx-a4000", refusalCatalog(),
		&hub.RentalSKUStatus{Name: "rtx-a4000", Known: true, LastSeenAt: &lastSeen}, now)
	if stockOut == nil {
		t.Fatal("an unbuyable SKU was not refused")
	}
	text := stockOut.Error() + " " + stockOut.Remedy
	// The timestamp is the actionable half — it is what turns "stop" into "wait".
	if !strings.Contains(text, "2026-09-04T06:48:08Z") {
		t.Fatalf("a stock-out refusal does not say when the SKU was last offered: %s", text)
	}
	if !strings.Contains(text, "is a Tensorhub product") {
		t.Fatalf("a stock-out refusal does not say the product is real: %s", text)
	}
	if stockOut.ErrName() != "rental.sku_out_of_stock" {
		t.Fatalf("a stock-out is named %q", stockOut.ErrName())
	}

	unknown := cli.SKURefusal("rtx-9090", refusalCatalog(),
		&hub.RentalSKUStatus{Name: "rtx-9090"}, now)
	if unknown == nil {
		t.Fatal("an unknown SKU was not refused")
	}
	if unknown.ErrName() != "rental.sku_unknown" {
		t.Fatalf("an unknown name is named %q", unknown.ErrName())
	}
	if strings.Contains(unknown.Error(), "is a Tensorhub product") {
		t.Fatalf("an unsold name is described as a real product: %s", unknown.Error())
	}

	// THE ASSERTION THE ISSUE REDUCES TO: the two absences must not read alike.
	if stockOut.ErrName() == unknown.ErrName() {
		t.Fatalf("both absences are reported as %q", stockOut.ErrName())
	}

	// Both remain refusals. This is presentation, not a fallback: neither may
	// name a substitute machine as something that was rented.
	for _, refusal := range []*exit.Error{stockOut, unknown} {
		if !strings.Contains(refusal.Error(), "NOTHING was rented") {
			t.Fatalf("a refusal does not say nothing was rented: %s", refusal.Error())
		}
		for _, other := range []string{"rtx-4090", "cpu"} {
			if strings.Contains(refusal.Error(), "renting "+other) {
				t.Fatalf("a refusal reads as though %s was rented instead: %s", other, refusal.Error())
			}
		}
	}
}

// TestARefusalSurvivesAHubThatCannotAnswer: the status lookup is an explanation,
// never the refusal itself. A hub too old to serve the route must still produce
// a refusal about the SKU.
func TestARefusalSurvivesAHubThatCannotAnswer(t *testing.T) {
	refusal := cli.SKURefusal("rtx-a4000", refusalCatalog(), nil, time.Now())
	if refusal == nil {
		t.Fatal("no refusal when the hub could not be asked")
	}
	if refusal.ErrName() != "rental.sku_unavailable" {
		t.Fatalf("an unanswerable lookup is named %q", refusal.ErrName())
	}
	if !strings.Contains(refusal.Error(), "NOTHING was rented") {
		t.Fatalf("the fallback refusal drops the no-substitution statement: %s", refusal.Error())
	}
	// It must still list what IS buyable, so the person has somewhere to go.
	if !strings.Contains(refusal.Remedy, "rtx-4090") {
		t.Fatalf("the fallback refusal does not show the live catalog: %v", refusal.Remedy)
	}
}
