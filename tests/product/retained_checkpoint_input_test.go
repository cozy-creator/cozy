package producttest

import (
	"crypto/ed25519"
	"crypto/rand"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

func TestRemoteJobDownloadsRetainedCheckpointWithoutRelease(t *testing.T) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	const pkg = "cozy/h3-package"
	manifest := "sha256:" + strings.Repeat("a", 64)
	pod := &fakePod{controlKey: public, serve: true, jobReady: true}
	connection, _ := startFakePod(t, t.TempDir(), pod)
	o := hostOwner(t, "retained-checkpoint", rentalWiring(connection, private))
	model := orchestrator.ModelRef{Package: pkg, Slot: "source", BindingPath: "four-lane.models.source",
		Model: "proof/model", Manifest: manifest, ManifestLength: 164, HubCheckpoint: true}
	request, _, problem := o.c.Submit(orchestrator.Submission{IdemKey: "retained-checkpoint",
		Package: pkg, Release: "1.0.7", Entrypoint: "four-lane", PlanID: "sha256:" + strings.Repeat("35", 32),
		Kind: "job", Org: "proof", Payload: []byte("{}"), Worker: podRental, Rental: true, RentalRequired: true,
		Models: []orchestrator.ModelRef{model},
	})
	fatal(t, problem)
	waitUntil(t, "retained checkpoint job reaches the existing worker", func() bool {
		pod.mu.Lock()
		defer pod.mu.Unlock()
		return len(pod.offers) == 1
	})
	pod.mu.Lock()
	prepared := append([]byte(nil), pod.prepares[len(pod.prepares)-1].PackageSet.DownloadDelegation...)
	offer := append([]byte(nil), pod.offers[0].InvocationSpecCanonicalBytes...)
	pod.mu.Unlock()
	selection, err := canonical.Read(prepared, &pb.DownloadDelegation{})
	must(t, err)
	models := selection.List("models")
	if len(models) != 1 || models[0].Str("manifest") != manifest || models[0].Str("release") != "" || models[0].Str("lane") != "" {
		t.Fatal("retained checkpoint was omitted or given fabricated release metadata")
	}
	invocation, err := canonical.Read(offer, &pb.InvocationSpec{})
	must(t, err)
	matched := false
	for _, input := range invocation.List("inputs") {
		if input.Str("input_id") == "model:source" {
			matched = input.Str("digest") == manifest && input.Int("length") == 164
		}
	}
	if !matched {
		t.Fatal("job did not consume the exact checkpoint input")
	}
	sized, problem := o.store.DeclaredServingModels(request)
	fatal(t, problem)
	if len(sized) != 1 || sized[0].Manifest != manifest {
		t.Fatal("rental sizing omitted the retained checkpoint")
	}
	// A repository-shaped operation-local label is not a Hub resolution fact.
	local := records.ModelRef{Model: "job-local/source", Manifest: manifest, ManifestLength: 164}
	if local.Downloadable() {
		t.Fatal("operation-local source acquired public download authority")
	}
}
