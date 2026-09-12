package api

import (
	"net/http"

	"github.com/cozy-creator/cozy/internal/hub"
)

// RentalInventory is the daemon's reconciled fleet read model. It contains
// public rental facts and activity, never records schema, credentials, certificate
// paths, or acquisition request bodies. Additive fields allow older clients to
// keep reading a newer daemon without opening its SQLite database.
type RentalInventory struct {
	MachinesRunning      int             `json:"machines_running"`
	HourlySpendUSDMicros int64           `json:"hourly_spend_usd_micros"`
	IdleReleaseSeconds   int64           `json:"idle_release_s"`
	Rentals              []RentalSummary `json:"rentals"`
	Unrecorded           []RentalSummary `json:"unrecorded"`
	Pending              []RentalSummary `json:"pending"`
}

type RentalSummary struct {
	ID                  string            `json:"rental_id"`
	MachineName         string            `json:"machine"`
	SKU                 string            `json:"sku,omitempty"`
	State               string            `json:"state"`
	AcceleratorModel    string            `json:"accelerator,omitempty"`
	AcceleratorCount    int               `json:"accelerator_count"`
	HourlyRateUSDMicros int64             `json:"hourly_rate_usd_micros"`
	Address             string            `json:"address,omitempty"`
	MediaAddress        string            `json:"media_address,omitempty"`
	Hub                 string            `json:"hub,omitempty"`
	RentedAt            string            `json:"rented_at,omitempty"`
	ReadyAt             string            `json:"ready_at,omitempty"`
	BoughtFor           string            `json:"bought_for,omitempty"`
	Activity            *RentalActivity   `json:"activity,omitempty"`
	Operation           string            `json:"operation,omitempty"`
	ProviderState       string            `json:"provider_state,omitempty"`
	ContainerState      string            `json:"container_state,omitempty"`
	Failure             hub.RentalFailure `json:"failure"`
}

// Activity is absent for machines known only to the Hub: this daemon cannot
// observe their queued/running work and must not report zero for them.
type RentalActivity struct {
	Running    int    `json:"running"`
	Queued     int    `json:"queued"`
	IdleSince  string `json:"idle_since_at,omitempty"`
	ReleaseDue string `json:"release_due_at,omitempty"`
}

func (s *Server) listRentals(w http.ResponseWriter, r *http.Request) {
	if s.rentalInventory == nil {
		s.refuse(w, r, http.StatusNotImplemented, "rental_inventory_unavailable",
			"this daemon cannot report its rental inventory", "update the daemon when its work permits")
		return
	}
	inventory, problem := s.rentalInventory(r.URL.Query().Get("reconcile") != "false")
	if problem != nil {
		s.refuseTyped(w, r, problem)
		return
	}
	s.ok(w, r, http.StatusOK, inventory)
}
