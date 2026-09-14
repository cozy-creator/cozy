package producttest

import (
	"bytes"
	"crypto/ed25519"
	"crypto/rand"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// This independent wire55 peer advertises readiness even when its preparation
// implementation ignores adapters, like a different additive schema at minor55.
// Actual TLS/Claim, streamed preparation, snapshots, routing and offers run here.
func TestLoRAWire55PreparedEchoMustMatchBeforeAnyOffer(t *testing.T) {
	for _, behavior := range []string{"omitted", "changed_scale", "changed_order", "unexpected", "matched"} {
		t.Run(behavior, func(t *testing.T) {
			public, private, err := ed25519.GenerateKey(rand.Reader)
			must(t, err)
			requested := []records.ModelAdapterRef{
				{Component: "dit", Model: "owner/style", Release: "1.0.0", Lane: "fp16", Manifest: "sha256:" + strings.Repeat("2", 64), ManifestLength: 128, SourceComponent: "adapter", Scale: "0"},
				{Component: "dit", Model: "owner/style", Release: "1.0.0", Lane: "fp16", Manifest: "sha256:" + strings.Repeat("2", 64), ManifestLength: 128, SourceComponent: "adapter", Scale: "-0.25"},
			}
			pod := &fakePod{controlKey: public, serve: true,
				preparedPlacement: func(download []byte, pkg, release string) *pb.Placement {
					placement := podPlacement(download, pkg, release, "")
					placement.Models = []*pb.Model{{Id: "base", Repo: "owner/base", Version: "1.0.0", Lane: "fp8", Manifest: &pb.Ref{Digest: bytes.Repeat([]byte{0x11}, 32), Length: 128}}}
					slot := &pb.Slot{Slot: "model", ReferenceModelId: "base", Components: []*pb.Component{{Component: "dit", ModelId: "base"}}}
					if behavior != "omitted" {
						placement.Models = append(placement.Models, &pb.Model{Id: "style", Repo: "owner/style", Version: "1.0.0", Lane: "fp16", Manifest: &pb.Ref{Digest: bytes.Repeat([]byte{0x22}, 32), Length: 128}})
						for _, row := range requested {
							slot.Adapters = append(slot.Adapters, &pb.ModelAdapter{Component: row.Component, ModelId: "style", SourceComponent: row.SourceComponent, Scale: row.Scale})
						}
						if behavior == "changed_scale" {
							slot.Adapters[0].Scale = "1"
						}
						if behavior == "changed_order" {
							slot.Adapters[0], slot.Adapters[1] = slot.Adapters[1], slot.Adapters[0]
						}
					}
					placement.Entrypoints = []*pb.Entrypoint{{Name: "tile", Slots: []*pb.Slot{slot}}}
					return placement
				}}
			requestStack := requested
			if behavior == "unexpected" {
				requestStack = nil
			}
			connection, _ := startFakePod(t, t.TempDir(), pod)
			owner := hostOwner(t, "lora-echo-20260914-"+behavior, rentalWiring(connection, private))
			id, _, problem := owner.c.Submit(orchestrator.Submission{IdemKey: behavior, Package: "cozy/h3-package", Entrypoint: "tile", Release: "1.1.2", Payload: []byte(`{"prompt":"same"}`), Worker: podRental, Rental: true, RentalRequired: true,
				Models: []orchestrator.ModelRef{{Package: "cozy/h3-package", Slot: "tile.models.model", Model: "owner/base", Release: "1.0.0", Lane: "fp8", Manifest: "sha256:" + strings.Repeat("1", 64), ManifestLength: 128, Adapters: requestStack}},
			})
			fatal(t, problem)
			waitUntil(t, "matching offer or typed preparation refusal", func() bool {
				pod.mu.Lock()
				offers := len(pod.offers)
				pod.mu.Unlock()
				row, problem := owner.store.RequestRow(id)
				fatal(t, problem)
				return offers > 0 || row.State == "failed"
			})
			pod.mu.Lock()
			offers := len(pod.offers)
			prepares := len(pod.prepares)
			pod.mu.Unlock()
			row, problem := owner.store.RequestRow(id)
			fatal(t, problem)
			if prepares == 0 {
				t.Fatal("test did not cross actual preparation")
			}
			if behavior == "matched" {
				if offers != 1 {
					t.Fatalf("matched stack did not serve: %+v", row)
				}
				return
			}
			if offers != 0 {
				t.Fatalf("wire55 peer %s silently received %d inference offer(s)", behavior, offers)
			}
			if row.State != "failed" {
				t.Fatalf("missing echo was not refused: %+v", row)
			}
			events, problem := owner.store.EventsAfter(id, 0, 100)
			fatal(t, problem)
			found := false
			for _, event := range events {
				if event.Type == "request.failed" && event.Payload["error_type"] == "model_adapters_preparation_mismatch" {
					found = true
				}
			}
			if !found {
				t.Fatalf("failure did not identify the adapter echo: %v", events)
			}
			time.Sleep(50 * time.Millisecond)
			pod.mu.Lock()
			defer pod.mu.Unlock()
			if len(pod.offers) != 0 {
				t.Fatal("refused request was later offered")
			}
		})
	}
}
