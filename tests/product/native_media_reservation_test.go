package producttest

import (
	"crypto/ed25519"
	"crypto/rand"
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

func TestNativeAndMixedOutputMediaReservation(t *testing.T) {
	native, _ := json.Marshal([]orchestrator.WeightsOutput{{OutputID: "model", MimeType: orchestrator.WeightsManifestMime, MaxBytes: 200 << 30}})
	for _, test := range []struct {
		outputs string
		weights string
		want    int64
	}{
		{"model", string(native), 0},
		{"image,model", string(native), 256},
		{"image,video", `[]`, 512},
	} {
		got, problem := orchestrator.MediaReservationBytes(records.Request{Outputs: test.outputs, WeightsOutputs: test.weights}, 256)
		fatal(t, problem)
		if got != test.want {
			t.Fatalf("%s reserved%d want%d", test.outputs, got, test.want)
		}
	}
	if _, problem := orchestrator.MediaReservationBytes(records.Request{Outputs: "a,b", WeightsOutputs: `[]`}, math.MaxInt64); problem == nil {
		t.Fatal("media bound overflow was accepted")
	}
}

func TestNativeWeightsDispatchDoesNotRequireUnusedMediaCapacity(t *testing.T) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	offers := make(chan *pb.AttemptOffer, 1)
	pod := &fakePod{controlKey: public, serve: true, jobReady: true}
	pod.mediaReservation = func(bytes int64, count int) error {
		// This byte plane has room for payload metadata, but not even the
		// default media ceiling. Native body storage is a distinct admission.
		if bytes != 0 || count != 1 {
			return fmt.Errorf("unexpected media reservation%d/%d", bytes, count)
		}
		return nil
	}
	pod.onFrame = func(frame *pb.RecordOwnerFrame, _ func(*pb.WorkerFrame) error) (bool, error) {
		if offer := frame.GetAttemptOffer(); offer != nil {
			offers <- offer
			return true, nil
		}
		return false, nil
	}
	connection, _ := startFakePod(t, t.TempDir(), pod)
	owner := hostOwner(t, "native-media-budget", rentalWiring(connection, private))
	_, _, problem := owner.c.Submit(orchestrator.Submission{IdemKey: "native-budget", Package: "cozy/h3-package", Entrypoint: "four-lane", PlanID: "sha256:" + strings.Repeat("35", 32), Release: "1.0.7", Kind: "job", Org: "paul", Payload: []byte(`{}`), Outputs: []string{"model"}, WeightsOutputs: []orchestrator.WeightsOutput{{OutputID: "model", MimeType: orchestrator.WeightsManifestMime, MaxBytes: 200 << 30}}, Worker: podRental, Rental: true, RentalRequired: true})
	fatal(t, problem)
	waitUntil(t, "native offer on a media-limited pod", func() bool { return len(offers) > 0 })
	if offer := <-offers; len(offer.Grant.Outputs) != 1 || offer.Grant.Outputs[0].OutputId != "model" {
		t.Fatal("native grant/file-count identity changed")
	}
}
