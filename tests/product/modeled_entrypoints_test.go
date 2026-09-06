package producttest

import (
	"bytes"
	"crypto/ed25519"
	"crypto/rand"
	"flag"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/cli"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

var modeledPackageHub = flag.String("modeled-package-hub", "", "real Tensorhub for read-only published H3 interface validation")

// The published package interface, rather than a copied test schema, supplies
// both callable/slot declarations. This reads metadata only and claims no model custody.
func TestPublishedH3ModeledEntrypointSelection(t *testing.T) {
	if *modeledPackageHub == "" {
		t.Skip("requires -modeled-package-hub for the published H3 1.1.2 metadata")
	}
	root := t.TempDir()
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	resolver := cli.NewResolver(store, config.Config{HubURL: *modeledPackageHub, Home: root}, nil)
	for _, function := range []string{"first_last_frame_to_video", "reference_media_to_video"} {
		models := []orchestrator.ModelRef{{Package: "paul/minimax-h3", Slot: function + ".models.model",
			Model: "paul/minimax-h3", Release: "1.0.0", Lane: "bf16-full",
			Manifest: "sha256:" + strings.Repeat("1", 64), ManifestLength: 164}}
		logical, entrypoint, problem := resolver.ResolveRemoteRelease("paul/minimax-h3", "1.1.2", function, models)
		fatal(t, problem)
		if entrypoint.Name != function || logical.Function != function || logical.PlanID != "" ||
			len(logical.Models) != 1 || logical.Models[0] != models[0] {
			t.Fatal("remote metadata resolution changed the callable or its selected model")
		}
		models[0].Slot = "unselected.models.model"
		if _, _, problem := resolver.ResolveRemoteRelease("paul/minimax-h3", "1.1.2", function, models); problem == nil {
			t.Fatal("another callable's model selection was accepted")
		}
	}
}

// The claimed worker prepares two dispatchable bindings in one package. The
// owner must select by the requested function, including the non-first name.
func TestModeledRentalSelectsTheNamedPreparedBinding(t *testing.T) {
	for _, function := range []string{"first_last_frame_to_video", "reference_media_to_video"} {
		t.Run(function, func(t *testing.T) {
			public, private, err := ed25519.GenerateKey(rand.Reader)
			must(t, err)
			bindings := map[string][]byte{
				"first_last_frame_to_video": bytes.Repeat([]byte{0x31}, 32),
				"reference_media_to_video":  bytes.Repeat([]byte{0x32}, 32),
			}
			pod := &fakePod{controlKey: public, serve: true,
				preparedPlacement: func(download []byte, pkg, release string) *pb.Placement {
					placement := podPlacement(download, pkg, release, "")
					placement.Entrypoints = []*pb.Entrypoint{
						{Name: "first_last_frame_to_video", EntrypointBindingDigest: bindings["first_last_frame_to_video"]},
						{Name: "reference_media_to_video", EntrypointBindingDigest: bindings["reference_media_to_video"]},
					}
					return placement
				}}
			connection, _ := startFakePod(t, t.TempDir(), pod)
			o := hostOwner(t, "modeled-"+function, rentalWiring(connection, private))
			requestID, _, problem := o.c.Submit(orchestrator.Submission{
				IdemKey: "modeled-" + function, Package: "cozy/h3-package", Entrypoint: function,
				Release: "1.1.2", Payload: []byte(`{"prompt":"a fox"}`), Outputs: []string{"video"},
				Worker: podRental, Rental: true, RentalRequired: true,
				Models: []orchestrator.ModelRef{{Package: "cozy/h3-package", Slot: function + ".models.model",
					Model: "source/h3", Release: "1.0.0", Lane: "bf16-full",
					Manifest: "sha256:" + strings.Repeat("1", 64), ManifestLength: 164}},
			})
			fatal(t, problem)
			deadline := time.Now().Add(5 * time.Second)
			for {
				pod.mu.Lock()
				offers := append([]*pb.AttemptOffer(nil), pod.offers...)
				pod.mu.Unlock()
				if len(offers) > 0 {
					if len(offers) != 1 {
						t.Fatalf("one invocation received %d offers", len(offers))
					}
					spec, err := canonical.Read(offers[0].InvocationSpecCanonicalBytes, &pb.InvocationSpec{})
					must(t, err)
					want, err := canonical.Spell(bindings[function])
					must(t, err)
					if got := spec.Sub("serving").Str("entrypoint_binding_digest"); got != want {
						t.Fatalf("%s dispatched binding %s instead of %s", function, got, want)
					}
					row, problem := o.store.RequestRow(requestID)
					fatal(t, problem)
					if row.PlanID != want || row.Models[0].Slot != function+".models.model" {
						t.Fatal("named binding or selected model changed before durable dispatch")
					}
					return
				}
				if time.Now().After(deadline) {
					t.Fatalf("%s never received its named prepared binding: %v", function, o.c.Events())
				}
				time.Sleep(10 * time.Millisecond)
			}
		})
	}
}
