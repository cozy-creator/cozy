package producttest

import (
	"crypto/ed25519"
	"crypto/rand"
	"strings"
	"sync"
	"testing"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// The first authenticated peer holds its real preparation stream open. A request
// for another explicit rental must prepare and dispatch before that stream ends.
// Automatic requests for the same package retain the existing single-flight rule.
func TestPreparationScopesExplicitRentalsIndependently(t *testing.T) {
	for _, explicit := range []bool{true, false} {
		name := "automatic"
		if explicit {
			name = "explicit"
		}
		t.Run(name, func(t *testing.T) {
			public, private, err := ed25519.GenerateKey(rand.Reader)
			must(t, err)
			held := &fakePod{identity: "prepare-held", controlKey: public, serve: true}
			free := &fakePod{identity: "prepare-free", controlKey: public, serve: true}
			a, _ := startFakePod(t, t.TempDir(), held)
			b, _ := startFakePod(t, t.TempDir(), free)
			entered, release := make(chan struct{}, 1), make(chan struct{})
			defer close(release)
			held.prepareEvent = func(event *pb.PrepareEvent) {
				if event.Stage == pb.PrepareStage_PREPARE_STAGE_PREPARING {
					select {
					case entered <- struct{}{}:
					default:
					}
					<-release
				}
			}
			var mu sync.Mutex
			acquisitions := map[string]int{}
			o := hostOwner(t, "pinned-preparation-"+name, rentalWiring(a, private), func(opt *orchestrator.Options) {
				opt.Rentals = func(id string) (*orchestrator.RemoteTarget, *exit.Error) {
					for _, connection := range []*orchestrator.WorkerConnection{a, b} {
						if connection.RentalID == id {
							return &orchestrator.RemoteTarget{Connection: connection}, nil
						}
					}
					return nil, exit.New(exit.NotFound, "unknown fixture rental")
				}
				opt.RentalFleet = func() (string, *exit.Error) { return "two existing rentals", nil }
				opt.AcquireManagedRental = func(req records.Request) (orchestrator.PlacementDecision, string, *exit.Error) {
					id := req.RequestedRental
					if id == "" {
						id = a.RentalID
					}
					mu.Lock()
					acquisitions[id]++
					mu.Unlock()
					_, problem := opt.Store.PinRental(req.ID, id, nil)
					return orchestrator.PlacementDecision{RentalID: id}, "", problem
				}
			})
			for _, connection := range []*orchestrator.WorkerConnection{a, b} {
				fatal(t, o.store.RecordRental(records.Rental{ID: connection.RentalID,
					MachineName: strings.TrimPrefix(connection.RentalID, "pr-"), SKU: "cpu",
					AcceleratorModel: "CPU", AcceleratorCount: 1, HourlyRateUSDMicros: 1, State: "ready", Hub: "fixture"}))
			}
			submit := func(key, requested string) string {
				t.Helper()
				if !explicit {
					requested = ""
				}
				id, _, problem := o.c.Submit(orchestrator.Submission{IdemKey: key,
					Package: "proof/preparation", Release: "1.0.0", Entrypoint: "tile",
					PlanID: podPlanID("proof/preparation"), Payload: []byte(`{"size":16}`),
					Outputs: []string{"image"}, RequestedRental: requested,
					Rental: true, RentalRequired: true})
				fatal(t, problem)
				return id
			}
			first := submit("first", a.RentalID)
			waitUntil(t, "first rental holds its preparation stream", func() bool {
				select {
				case <-entered:
					return true
				default:
					return false
				}
			})
			second := submit("second", b.RentalID)
			third := submit("same-rental-follower", a.RentalID)
			mu.Lock()
			aCount, bCount := acquisitions[a.RentalID], acquisitions[b.RentalID]
			mu.Unlock()
			wantB := 0
			if explicit {
				wantB = 1
			}
			if aCount != 1 || bCount != wantB {
				t.Fatalf("preparation selections: first=%d second=%d; want1/%d while first is held", aCount, bCount, wantB)
			}
			if explicit {
				waitUntil(t, "second rental dispatches while first is still preparing", func() bool {
					free.mu.Lock()
					defer free.mu.Unlock()
					return len(free.offers) == 1
				})
				free.mu.Lock()
				offered := free.offers[0].RequestId
				free.mu.Unlock()
				if offered != second {
					t.Fatal("wrong request crossed the independent rental")
				}
			}
			held.mu.Lock()
			heldOffers := len(held.offers)
			held.mu.Unlock()
			if heldOffers != 0 || o.c.QueuePosition(first) == 0 || o.c.QueuePosition(third) == 0 {
				t.Fatal("same-rental work overtook unfinished preparation")
			}
		})
	}
}

func TestWakeQueuePreservesUnassignedRequestedRentalAffinities(t *testing.T) {
	var mu sync.Mutex
	seen := map[string]bool{}
	recording := false
	o := hostOwner(t, "pinned-preparation-wake", func(opt *orchestrator.Options) {
		opt.RentalFleet = func() (string, *exit.Error) { return "existing requested rentals", nil }
		opt.AcquireManagedRental = func(req records.Request) (orchestrator.PlacementDecision, string, *exit.Error) {
			mu.Lock()
			if recording {
				seen[req.RequestedRental] = true
			}
			mu.Unlock()
			return orchestrator.PlacementDecision{}, "", exit.Unavailablef("fixture preparation is not ready")
		}
	})
	for _, rental := range []string{"wanted-a", "wanted-b"} {
		_, _, problem := o.c.Submit(orchestrator.Submission{IdemKey: rental,
			Package: "proof/wake", Entrypoint: "tile", Release: "1.0.0", Payload: []byte(`{}`),
			Rental: true, RentalRequired: true, RequestedRental: rental})
		fatal(t, problem)
	}
	mu.Lock()
	recording = true
	mu.Unlock()
	o.c.WakeQueue()
	mu.Lock()
	defer mu.Unlock()
	if !seen["wanted-a"] || !seen["wanted-b"] {
		t.Fatalf("wake omitted an independent requested rental: %v", seen)
	}
}

func TestForeignStagedRentalDoesNotSuppressExplicitPreparation(t *testing.T) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	foreign := &fakePod{identity: "staged-foreign", controlKey: public, serve: true, noSeats: true}
	connection, _ := startFakePod(t, t.TempDir(), foreign)
	var mu sync.Mutex
	selected := ""
	o := hostOwner(t, "pinned-preparation-foreign", rentalWiring(connection, private), func(opt *orchestrator.Options) {
		opt.RentalFleet = func() (string, *exit.Error) { return "another requested rental", nil }
		opt.AcquireManagedRental = func(req records.Request) (orchestrator.PlacementDecision, string, *exit.Error) {
			mu.Lock()
			selected = req.RequestedRental
			mu.Unlock()
			return orchestrator.PlacementDecision{}, "", exit.Unavailablef("requested rental preparation pending")
		}
	})
	instance, _, _, problem := o.c.EnsureRental(connection.RentalID)
	fatal(t, problem)
	fatal(t, o.c.ConvergePackageSet(instance, []*pb.DownloadPackageRef{{Package: "proof/staged", Release: "1.0.0"}}, nil))
	waitUntil(t, "foreign rental reports the exact binding staged", func() bool {
		facts := o.c.Worker(instance)
		return facts != nil && len(facts.Dispatchable) > 0
	})
	warm, _, problem := o.c.Submit(orchestrator.Submission{IdemKey: "prepare-foreign-binding",
		Package: "proof/staged", Release: "1.0.0", Entrypoint: "tile", PlanID: podPlanID("proof/staged"),
		Payload: []byte(`{"size":16}`), Worker: connection.RentalID, RequestedRental: connection.RentalID,
		Rental: true, RentalRequired: true})
	fatal(t, problem)
	waitUntil(t, "owner records the foreign rental's selected binding", func() bool {
		_, holders := o.c.PackageHolders("proof/staged")
		for _, holder := range holders {
			if holder.RentalID == connection.RentalID {
				return true
			}
		}
		return false
	})
	fatal(t, o.c.CancelQueued(warm, "foreign preparation fixture complete"))
	_, _, problem = o.c.Submit(orchestrator.Submission{IdemKey: "wanted-independent",
		Package: "proof/staged", Release: "1.0.0", Entrypoint: "tile", PlanID: podPlanID("proof/staged"),
		Payload: []byte(`{"size":16}`), RequestedRental: "wanted-independent", Rental: true, RentalRequired: true})
	fatal(t, problem)
	mu.Lock()
	defer mu.Unlock()
	if selected != "wanted-independent" {
		t.Fatal("a staged binding on a different rental suppressed explicit preparation")
	}
}
