package producttest

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/client"
	"github.com/cozy-creator/cozy/internal/daemon"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// Drive canonical PrepareEvent rows through the real gRPC preparation stream,
// then attach an ordinary authenticated SSE client after the last sample. A late
// subscriber must see each checkpoint's own counters and origin/cache distinction.
func TestModelPrepareProgressReachesLateSSESubscriber(t *testing.T) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	h3 := &pb.DownloadModelRef{Package: "cozy/h3-package", Slot: "model", Model: "paul/minimax-h3", Release: "1.0.0", Lane: "fp8", Manifest: strings.Repeat("a", 64)}
	encoder := &pb.DownloadModelRef{Package: "cozy/h3-package", Slot: "encoder", Model: "paul/encoder", Release: "2.0.0", Lane: "bf16", Manifest: strings.Repeat("b", 64)}
	pod := &fakePod{controlKey: public, downloadSamples: 3}
	pod.prepareEvent = func(event *pb.PrepareEvent) {
		event.ModelProgress = []*pb.PrepareModelProgress{
			{Model: h3, TotalBytes: 10000, TransferredBytes: 10000, OriginBytes: 4000, CachedBytes: 1000},
			{Model: encoder, TotalBytes: 9000, TransferredBytes: 9000},
		}
	}
	connection, _ := startFakePod(t, t.TempDir(), pod)
	o := hostOwner(t, "model-prepare-sse", rentalWiring(connection, private))
	instance, _, _, e := o.c.EnsureRental(podRental)
	fatal(t, e)
	_, _, e = o.store.Submit(records.Request{ID: "model-download-request", IdemKey: "model-download-request",
		BodyDigest: "model-download-request", Package: "cozy/h3-package", Entrypoint: "fake",
		State: "queued", Payload: []byte("{}"), Outputs: "[]", Worker: podRental, Rental: true})
	fatal(t, e)
	fatal(t, o.c.ConvergePackageSet(instance, []*pb.DownloadPackageRef{{Package: "cozy/h3-package", Release: "1.0.7"}}, nil))
	waitUntil(t, "model preparation rows received", func() bool {
		phase, ok := o.c.PreparationPhase(instance)
		return ok && phase.Name == orchestrator.PhaseWarming && len(phase.Models) == 2
	})
	closeAPI := publicationControlAPI(t, o)
	defer closeAPI()
	cli, e := client.Open(o.cfg, daemon.Probe(o.cfg))
	fatal(t, e)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	var rows []orchestrator.ModelDownloadProgress
	var phaseName string
	_, e = cli.WatchContext(ctx, "model-download-request", 0, func(event client.Event) bool {
		if event.Type != "request.phase" {
			return true
		}
		body, err := json.Marshal(event.Payload["value"])
		must(t, err)
		var payload struct {
			Phase  string                               `json:"phase"`
			Models []orchestrator.ModelDownloadProgress `json:"models"`
		}
		must(t, json.Unmarshal(body, &payload))
		rows, phaseName = payload.Models, payload.Phase
		return false
	})
	fatal(t, e)
	if phaseName != orchestrator.PhaseWarming || len(rows) != 2 {
		t.Fatalf("SSE lost model rows: phase=%s rows=%+v", phaseName, rows)
	}
	if rows[0].Model != encoder.Model || rows[0].Moved != 9000 || rows[0].Total != 9000 || rows[0].OriginBytes != 0 || rows[0].Rate != 0 || rows[0].RemainingMS != nil {
		t.Fatalf("already-held encoder gained a transfer or estimate: %+v", rows[0])
	}
	if rows[1].Model != h3.Model || rows[1].Lane != "fp8" || rows[1].Manifest != h3.Manifest || rows[1].Moved != 10000 || rows[1].Total != 10000 || rows[1].OriginBytes != 4000 || rows[1].CachedBytes != 1000 {
		t.Fatalf("SSE changed model identity or byte sources: %+v", rows[1])
	}
}
