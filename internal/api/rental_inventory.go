package api

import (
	"net/http"
	"slices"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
)

// RentalInventory is the daemon's reconciled fleet read model. It contains
// public rental facts and activity, never records schema, credentials, certificate
// paths, or acquisition request bodies.
type RentalInventory struct {
	MachinesRunning      int             `json:"machines_running"`
	HourlySpendUSDMicros int64           `json:"hourly_spend_usd_micros"`
	IdleReleaseSeconds   int64           `json:"idle_release_s"`
	Rentals              []RentalSummary `json:"rentals"`
	Unrecorded           []RentalSummary `json:"unrecorded"`
	Pending              []RentalSummary `json:"pending"`
	// HubUnanswered is set when Tensorhub did not answer this read. Rows are then
	// this host's last records and the totals are unknown, not zero.
	HubUnanswered *exit.Error `json:"hub_unanswered,omitempty"`
	// OtherHubs counts this host's live rentals on hubs this read did not cover, so a
	// listing scoped to one hub never hides a machine billing on another.
	OtherHubs []HubRentals `json:"other_hubs,omitempty"`
	// UnreadableHubs are the hubs an every-hub read could not ask. Each one's rows are
	// this host's records, marked unverified; every other hub is still listed.
	UnreadableHubs []HubProblem `json:"unreadable_hubs,omitempty"`
}

// HubProblem is why one Tensorhub could not be read.
type HubProblem struct {
	Hub   string      `json:"hub"`
	Error *exit.Error `json:"error"`
}

// HubRentals is one Tensorhub's count of live rentals recorded on this host.
type HubRentals struct {
	Hub     string `json:"hub"`
	Rentals int    `json:"rentals"`
}

// Current removes proven-absent rentals from the fleet projection without
// changing retained history or the Hub-reconciled spend totals. A Hub-unknown host
// record is not proven absent and stays.
func (inventory RentalInventory) Current() RentalInventory {
	current := func(rows []RentalSummary) []RentalSummary {
		return slices.DeleteFunc(slices.Clone(rows), func(row RentalSummary) bool {
			return hub.RentalAbsent(row.State) && !row.HubUnknown
		})
	}
	inventory.Rentals = current(inventory.Rentals)
	inventory.Unrecorded = current(inventory.Unrecorded)
	inventory.Pending = current(inventory.Pending)
	return inventory
}

type RentalSummary struct {
	ID                  string          `json:"rental_id"`
	MachineName         string          `json:"machine"`
	SKU                 string          `json:"sku,omitempty"`
	State               string          `json:"state"`
	AcceleratorModel    string          `json:"accelerator,omitempty"`
	AcceleratorCount    int             `json:"accelerator_count"`
	HourlyRateUSDMicros int64           `json:"hourly_rate_usd_micros"`
	Address             string          `json:"address,omitempty"`
	MediaAddress        string          `json:"media_address,omitempty"`
	Hub                 string          `json:"hub,omitempty"`
	RentedAt            string          `json:"rented_at,omitempty"`
	ReadyAt             string          `json:"ready_at,omitempty"`
	BoughtFor           string          `json:"bought_for,omitempty"`
	Activity            *RentalActivity `json:"activity,omitempty"`
	Operation           string          `json:"operation,omitempty"`
	// Boot is the pod's boot while the Hub still acquires it.
	Boot *hub.RentalBoot `json:"boot,omitempty"`
	// RuntimeUpdate is the state of a Runtime update holding this rental's work.
	RuntimeUpdate string `json:"runtime_update,omitempty"`
	// The worker image the hub froze for this rental, from its live listing.
	BaseWorkerImageDigest string            `json:"base_worker_image_digest,omitempty"`
	BaseWorkerImageTag    string            `json:"base_worker_image_tag,omitempty"`
	Failure               hub.RentalFailure `json:"failure"`
	// HubUnknown is a host record the Hub answered 404 for: kept, because a missing
	// Hub record is not proof the provider pod is gone.
	HubUnknown bool `json:"hub_unknown,omitempty"`

	// Unverified marks a row whose hub could not be asked: this host's last record only.
	Unverified bool `json:"unverified,omitempty"`
	// SpendUSDMicros and SpendBasis are the Hub's accrued spend; a blank basis is unknown.
	SpendUSDMicros int64  `json:"spend_usd_micros,omitempty"`
	SpendBasis     string `json:"spend_basis,omitempty"`
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
	hub, problem := s.hubOf(r)
	if problem != nil {
		s.refuseTyped(w, r, problem)
		return
	}
	inventory, problem := s.rentalInventory(hub, r.URL.Query().Get("hubs") == "all",
		r.URL.Query().Get("reconcile") != "false")
	if problem != nil {
		s.refuseTyped(w, r, problem)
		return
	}
	s.ok(w, r, http.StatusOK, inventory)
}
