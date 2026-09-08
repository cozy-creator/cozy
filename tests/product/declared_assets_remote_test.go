package producttest

import (
	"bytes"
	"crypto/ed25519"
	"crypto/rand"
	"fmt"
	"image"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// Parse the public bare/label syntax, then exercise the existing remote media
// upload and claimed TLS offer. The independent peer checks actual landed bytes;
// real author execution is covered by the composed local callable/child proofs.
func TestDeclaredAssetsUseExistingRemoteInputBindings(t *testing.T) {
	ep := assetsCallable(t)
	var encoded bytes.Buffer
	must(t, png.Encode(&encoded, image.NewRGBA(image.Rect(0, 0, 2, 2))))
	photo := filepath.Join(t.TempDir(), "photo.png")
	must(t, os.WriteFile(photo, encoded.Bytes(), 0600))
	payload, assets, problem := launch.ParseAssets(ep, []byte(`{"prompt":"unchanged"}`), []string{"alice=" + photo, photo}, nil, nil)
	fatal(t, problem)
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	checked := make(chan error, 1)
	pod := &fakePod{controlKey: public, serve: true}
	pod.answerOffer = func(offer *pb.AttemptOffer) (*pb.AttemptOutcome, error) {
		check := func() error {
			spec, err := canonical.Read(offer.InvocationSpecCanonicalBytes, &pb.InvocationSpec{})
			if err != nil {
				return err
			}
			if !bytes.Equal(canonical.Digest(offer.InvocationSpecCanonicalBytes), offer.InvocationSpecDigest) {
				return fmt.Errorf("invocation digest changed")
			}
			bindings := spec.List("inputs")
			if len(bindings) != 3 {
				return fmt.Errorf("got%d bindings,want3", len(bindings))
			}
			for index, asset := range assets {
				binding := bindings[index+1]
				if binding.Str("input_id") != asset.FieldPath || binding.Str("digest") != asset.Digest || binding.Int("length") != asset.Length || binding.Str("kind_mime") != asset.MediaType || binding.Int("order") != int64(index) {
					return fmt.Errorf("immutable remote binding changed at%d", index)
				}
			}
			if offer.Grant == nil || len(offer.Grant.Inputs) != 3 {
				return fmt.Errorf("remote grant omitted inputs")
			}
			for _, input := range offer.Grant.Inputs {
				if !strings.HasPrefix(input.Url, "file://") {
					return fmt.Errorf("remote input did not land in pod-local custody")
				}
				path := strings.TrimPrefix(input.Url, "file://")
				data, err := os.ReadFile(path)
				if err != nil {
					return err
				}
				if input.InputId == "payload" {
					if !bytes.Equal(data, payload) {
						return fmt.Errorf("labels/order changed in remote payload")
					}
				} else {
					if path == photo || !bytes.Equal(data, encoded.Bytes()) {
						return fmt.Errorf("remote input is not the uploaded original bytes")
					}
				}
			}
			return nil
		}
		problem := check()
		checked <- problem
		if problem != nil {
			return nil, problem
		}
		return privateAttemptOutcome(offer, pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED, pb.CauseCode_CAUSE_CODE_UNSPECIFIED, pb.CauseOrigin_CAUSE_ORIGIN_RUNTIME).GetAttemptOutcome(), nil
	}
	connection, _ := startFakePod(t, t.TempDir(), pod)
	o := hostOwner(t, "declared-assets-remote", rentalWiring(connection, private))
	const pkg = "proof/assets"
	id, _, problem := o.c.Submit(orchestrator.Submission{IdemKey: "assets-remote", Package: pkg, Release: "1.0.0", Entrypoint: "tile", PlanID: podPlanID(pkg), Payload: payload, Assets: assets, Worker: podRental, Rental: true, RentalRequired: true})
	fatal(t, problem)
	select {
	case err := <-checked:
		must(t, err)
	case <-time.After(10 * time.Second):
		t.Fatal("no remote Assets offer")
	}
	waitUntil(t, "remote input transfer completion", func() bool {
		row, p := o.store.RequestRow(id)
		fatal(t, p)
		return row != nil && row.State == "succeeded"
	})
}

// A queued request borrows its submitted path. Replacing or removing that file
// must abort the grant before any attempt offer can reach the remote worker.
func TestBorrowedRemoteInputChangedBeforeDispatch(t *testing.T) {
	for _, mutation := range []string{"same_length_edit", "removed"} {
		t.Run(mutation, func(t *testing.T) {
			ep := assetsCallable(t)
			var encoded bytes.Buffer
			must(t, png.Encode(&encoded, image.NewRGBA(image.Rect(0, 0, 2, 2))))
			photo := filepath.Join(t.TempDir(), "original.png")
			must(t, os.WriteFile(photo, encoded.Bytes(), 0600))
			payload, assets, problem := launch.ParseAssets(ep, []byte(`{"prompt":"unchanged"}`), []string{photo}, nil, nil)
			fatal(t, problem)
			public, private, err := ed25519.GenerateKey(rand.Reader)
			must(t, err)
			var offered atomic.Bool
			pod := &fakePod{controlKey: public, serve: true}
			pod.answerOffer = func(offer *pb.AttemptOffer) (*pb.AttemptOutcome, error) {
				offered.Store(true)
				return privateAttemptOutcome(offer, pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED, pb.CauseCode_CAUSE_CODE_UNSPECIFIED, pb.CauseOrigin_CAUSE_ORIGIN_RUNTIME).GetAttemptOutcome(), nil
			}
			connection, _ := startFakePod(t, t.TempDir(), pod)
			o := hostOwner(t, "borrowed-input-"+mutation, rentalWiring(connection, private))
			const pkg = "proof/assets"
			row, _, problem := o.c.RecordSubmission(orchestrator.Submission{IdemKey: mutation, Package: pkg, Release: "1.0.0", Entrypoint: "tile", PlanID: podPlanID(pkg), Payload: payload, Assets: assets, Worker: podRental, Rental: true, RentalRequired: true})
			fatal(t, problem)
			if mutation == "removed" {
				must(t, os.Remove(photo))
			} else {
				must(t, os.WriteFile(photo, bytes.Repeat([]byte("x"), encoded.Len()), 0600))
			}
			_, _ = o.c.ActivateRecordedRequest(row)
			waitUntil(t, "changed source aborted before offer", func() bool {
				events, problem := o.store.EventsAfter(row.ID, 0, 100)
				fatal(t, problem)
				for _, event := range events {
					if event.Type == "request.dispatch_aborted" {
						return true
					}
				}
				return false
			})
			if offered.Load() {
				t.Fatal("changed original reached worker invocation")
			}
		})
	}
}
