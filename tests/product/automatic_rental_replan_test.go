package producttest

import (
	"crypto/ed25519"
	"crypto/rand"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// Two independent authenticated peers host the same exact package. An old
// automatic assignment to the full peer can use the other peer's real free seat;
// an explicit named-rental request must keep waiting on its chosen machine.
func TestAutomaticQueuedRentalUsesAnotherReadyPeer(t *testing.T) {
	for _, explicit := range []bool{false, true} {
		name := "automatic"
		if explicit {
			name = "explicit"
		}
		t.Run(name, func(t *testing.T) {
			public, private, err := ed25519.GenerateKey(rand.Reader)
			must(t, err)
			busy := &fakePod{identity: "busy", controlKey: public, serve: true, noSeats: true}
			free := &fakePod{identity: "free", controlKey: public, serve: true, noSeats: true}
			busyConnection, _ := startFakePod(t, t.TempDir(), busy)
			freeConnection, _ := startFakePod(t, t.TempDir(), free)
			o := hostOwner(t, "automatic-ready-"+name, rentalWiring(busyConnection, private), func(options *orchestrator.Options) {
				options.Rentals = func(id string) (*orchestrator.RemoteTarget, *exit.Error) {
					switch id {
					case busyConnection.RentalID:
						return &orchestrator.RemoteTarget{Connection: busyConnection}, nil
					case freeConnection.RentalID:
						return &orchestrator.RemoteTarget{Connection: freeConnection}, nil
					}
					return nil, exit.New(exit.NotFound, "unknown rental")
				}
			})
			const pkg = "proof/video"
			for _, conn := range []*orchestrator.WorkerConnection{busyConnection, freeConnection} {
				fatal(t, o.store.RecordRental(records.Rental{ID: conn.RentalID, MachineName: strings.TrimPrefix(conn.RentalID, "pr-"), SKU: "cpu", AcceleratorModel: "CPU", AcceleratorCount: 1, HourlyRateUSDMicros: 1, State: "ready", Hub: "proof"}))
				instance, _, _, problem := o.c.EnsureRental(conn.RentalID)
				fatal(t, problem)
				fatal(t, o.c.ConvergePackageSet(instance, []*pb.DownloadPackageRef{{Package: pkg, Release: "1.0.0"}}, nil))
				warm, _, problem := o.c.Submit(orchestrator.Submission{IdemKey: "warm-" + conn.RentalID,
					Package: pkg, Release: "1.0.0", Entrypoint: "tile", PlanID: podPlanID(pkg),
					Payload: []byte(`{"size":16}`), Outputs: []string{"image"}, Worker: conn.RentalID,
					RequestedRental: conn.RentalID, Rental: true, RentalRequired: true})
				fatal(t, problem)
				waitUntil(t, "the package is ready for invocation on "+conn.RentalID, func() bool {
					_, holders := o.c.PackageHolders(pkg)
					for _, holder := range holders {
						if holder.RentalID == conn.RentalID {
							return true
						}
					}
					return false
				})
				fatal(t, o.c.CancelQueued(warm, "preparation proof complete"))
			}
			waitUntil(t, "both peers advertise prepared packages", func() bool {
				busy.mu.Lock()
				a := len(busy.desired)
				busy.mu.Unlock()
				free.mu.Lock()
				b := len(free.desired)
				free.mu.Unlock()
				return a > 0 && b > 0
			})
			free.mu.Lock()
			free.noSeats = false
			free.mu.Unlock()
			freeInstance, _, _, problem := o.c.EnsureRental(freeConnection.RentalID)
			fatal(t, problem)
			fatal(t, o.c.ConvergePackageSet(freeInstance, []*pb.DownloadPackageRef{{Package: pkg, Release: "1.0.0"}}, nil))
			waitUntil(t, "the alternative advertises a free seat", func() bool { return o.c.Worker(freeInstance).AvailableSlots == 1 })
			requested := ""
			if explicit {
				requested = busyConnection.RentalID
			}
			id, _, problem := o.c.Submit(orchestrator.Submission{IdemKey: name, Package: pkg, Release: "1.0.0", Entrypoint: "tile", PlanID: podPlanID(pkg), Payload: []byte(`{"size":16}`), Outputs: []string{"image"}, Worker: busyConnection.RentalID, RequestedRental: requested, Rental: true, RentalRequired: true})
			fatal(t, problem)
			if explicit {
				waitUntil(t, "explicit request parks on its busy peer", func() bool { return o.c.QueuePosition(id) > 0 })
				time.Sleep(100 * time.Millisecond)
			} else {
				waitUntil(t, "automatic request is offered to the free peer", func() bool { free.mu.Lock(); defer free.mu.Unlock(); return len(free.offers) > 0 })
			}
			row, problem := o.store.RequestRow(id)
			fatal(t, problem)
			want := freeConnection.RentalID
			if explicit {
				want = busyConnection.RentalID
			}
			if row.Worker != want {
				t.Fatalf("worker=%s want%s", row.Worker, want)
			}
			busy.mu.Lock()
			busyOffers := len(busy.offers)
			busy.mu.Unlock()
			free.mu.Lock()
			freeOffers := len(free.offers)
			free.mu.Unlock()
			if busyOffers != 0 || explicit && freeOffers != 0 {
				t.Fatalf("request crossed affinity/full seat: busy=%d free=%d", busyOffers, freeOffers)
			}
		})
	}
}

func TestRentalReassignmentPreservesExecutionAndPurchaseCustody(t *testing.T) {
	for _, mode := range []string{"unattempted", "explicit", "retained", "uploaded", "purchased", "attempted"} {
		t.Run(mode, func(t *testing.T) {
			st, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
			fatal(t, problem)
			defer st.Close()
			row := records.Rental{ID: "pr-busy", MachineName: "busy", SKU: "cpu", AcceleratorModel: "CPU", AcceleratorCount: 1, HourlyRateUSDMicros: 1, HourlyRateSource: "estimate", State: "ready", Hub: "proof"}
			fatal(t, st.RecordRental(row))
			request := records.Request{ID: "job-replan", IdemKey: "replan", BodyDigest: "sha256:" + strings.Repeat("a", 64), Package: "proof/video", Entrypoint: "generate", PlanID: "plan", Payload: []byte(`{}`), Outputs: "[]", Kind: "job", Worker: row.ID, Rental: true, RentalRequired: true}
			switch mode {
			case "explicit":
				request.RequestedRental = row.ID
			case "retained":
				request.RetainWork = true
			case "uploaded":
				request.LocalPackageUploadedBootID = "boot-busy"
			}
			request, _, problem = st.Submit(request)
			fatal(t, problem)
			if mode == "purchased" {
				op, _, problem := st.BeginRentalOperation(records.RentalOperation{Key: "purchase", Hub: "proof", Reason: "job", ManagedRequestID: request.ID, HourlyRateUSDMicros: 1}, 100, 0, func(string) ([]byte, string, *exit.Error) { return []byte(`{}`), "proof", nil }, nil)
				fatal(t, problem)
				fatal(t, st.AdvanceRentalOperation(op.Key, row.ID, "ready"))
			}
			if mode == "attempted" {
				fatal(t, st.SpawnWorker(records.WorkerProcess{InstanceID: "instance", Package: request.Package, WorkerID: "worker", Devices: []string{"cpu"}}))
				_, problem := st.Dispatch(records.Attempt{RequestID: request.ID, InstanceID: "instance", SessionID: "session", InvocationCanonical: []byte(`{}`), WeightsOutputs: `[]`})
				fatal(t, problem)
			}
			changed, problem := st.ReleaseUnattemptedRentalAssignment(request.ID, row.ID)
			fatal(t, problem)
			if changed != (mode == "unattempted") {
				t.Fatalf("assignment released=%t for %s", changed, mode)
			}
			got, problem := st.RequestRow(request.ID)
			fatal(t, problem)
			if !changed && got.Worker != row.ID {
				t.Fatal("retained assignment changed")
			}
		})
	}
}
