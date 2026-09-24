package producttest

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/secret"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// The independent peer withholds PREPARED after reporting a download. Enqueuing
// desired state is not sufficient evidence that the requested files are usable.
func TestExplicitRentalPreparationWaitsForWorkerReceiptWithoutActivating(t *testing.T) {
	for _, mode := range []string{"code-only", "model", "refusal"} {
		t.Run(mode, func(t *testing.T) {
			public, private, err := ed25519.GenerateKey(rand.Reader)
			must(t, err)
			entered, release := make(chan struct{}), make(chan struct{})
			defer func() {
				select {
				case <-release:
				default:
					close(release)
				}
			}()
			var claims atomic.Int32
			pod := &fakePod{controlKey: public}
			pod.onFrame = func(frame *pb.RecordOwnerFrame, _ func(*pb.WorkerFrame) error) (bool, error) {
				if frame.GetClaim() != nil {
					claims.Add(1)
				}
				return false, nil
			}
			pod.prepareEvent = func(event *pb.PrepareEvent) {
				if event.Stage == pb.PrepareStage_PREPARE_STAGE_DOWNLOADING {
					close(entered)
					<-release
					if mode == "refusal" {
						event.Stage = pb.PrepareStage_PREPARE_STAGE_REFUSED
						event.SafeCode, event.SafeDetail = "checkpoint_source_unavailable", "the selected checkpoint was removed"
					}
				}
			}
			connection, _ := startFakePod(t, t.TempDir(), pod)
			o := hostOwner(t, "explicit-preparation-"+mode, rentalWiring(connection, private))
			instance, _, _, problem := o.c.EnsureRental(podRental)
			fatal(t, problem)
			var models []*pb.DownloadModelRef
			if mode != "code-only" {
				models = []*pb.DownloadModelRef{{Package: "proof/video", Slot: "generate.models.model", Model: "proof/base", Release: "1.0.0", Lane: "fp8", Manifest: childDigest("8")}}
			}
			result := make(chan *exit.Error, 1)
			go func() {
				result <- o.c.PrepareRentalPackage(context.Background(), instance, &pb.DownloadPackageRef{Package: "proof/video", Release: "1.0.0"}, models)
			}()
			select {
			case <-entered:
			case <-time.After(5 * time.Second):
				t.Fatal("worker did not begin preparation")
			}
			select {
			case problem := <-result:
				t.Fatal("reported preparation complete before worker receipt", problem)
			case <-time.After(30 * time.Millisecond):
			}
			// Concurrent machine execution borrows authority instead of opening
			// a second control stream and canceling this preparation's connection.
			claim, problem := o.c.RentalExecutionClaim(context.Background(), podRental)
			fatal(t, problem)
			if claim.WorkerId != podWorkerID || claim.WorkerBootId != podBootID || claims.Load() != 1 {
				t.Fatalf("execution replaced the active preparation claim: %+v, %d claims", claim, claims.Load())
			}
			claim.Proof = nil
			other, problem := o.c.RentalExecutionClaim(context.Background(), podRental)
			fatal(t, problem)
			if len(other.Proof) == 0 || claims.Load() != 1 {
				t.Fatal("borrowed claim mutated the owner or opened a control stream")
			}
			close(release)
			select {
			case problem := <-result:
				if mode == "refusal" {
					if problem == nil || problem.ErrName() != "worker.prepare_refused" {
						t.Fatal("worker refusal was lost", problem)
					}
				} else {
					fatal(t, problem)
				}
			case <-time.After(5 * time.Second):
				t.Fatal("verified preparation did not return")
			}
			pod.mu.Lock()
			defer pod.mu.Unlock()
			if len(pod.prepares) != 1 {
				t.Fatalf("worker preparations=%d", len(pod.prepares))
			}
			doc, err := canonical.Read(pod.prepares[0].PackageSet.DownloadDelegation, &pb.DownloadDelegation{})
			must(t, err)
			if len(doc.List("models")) != len(models) {
				t.Fatal("preparation changed its explicit model set")
			}
			if len(models) > 0 && doc.List("models")[0].Str("manifest") != models[0].Manifest {
				t.Fatal("model preparation did not freeze the selected checkpoint")
			}
			for _, desired := range pod.desired {
				if desired.GetPlacementSet() != nil && len(desired.GetPlacementSet().PlacementSetCanonicalBytes) > 0 {
					t.Fatal("explicit preparation activated a serving placement")
				}
			}
		})
	}
}

func TestRentalImageFactsAdvertisePublicModelOriginWithoutPackagePublication(t *testing.T) {
	calls := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		if r.URL.Path != "/v1/rentals/pr-test/image-inventory" || r.Header.Get("Authorization") != "Bearer inventory-token" || r.URL.RawQuery != "" {
			t.Errorf("unexpected metadata request %s", r.URL)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"image_inventory": map[string]any{"format": "tensorhub.image_inventory/1"}, "public_origin": "https://private-hub-tunnel.example"})
	}))
	defer server.Close()
	client := hub.New(config.Config{HubURL: server.URL, HubToken: secret.New("inventory-token")}, "private-model-origin-test")
	facts, problem := client.RentalImageInventory(context.Background(), "pr-test")
	fatal(t, problem)
	if calls != 1 || facts.PublicOrigin != "https://private-hub-tunnel.example" || len(facts.ImageInventory) == 0 {
		t.Fatal("rental byte origin was lost", facts, calls)
	}
}
