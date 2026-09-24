package producttest

import (
	"crypto/ed25519"
	"crypto/rand"
	"encoding/json"
	"math"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// A lossy stream may fail without the durable control stream failing. It must
// reconnect on its own, both before its first frame and after an established feed.
func TestProgressWatchReconnectsWithoutRestartingControl(t *testing.T) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	var calls atomic.Int32
	pod := &fakePod{controlKey: public}
	pod.watchProgress = func(open *pb.ProgressOpen, stream pb.WorkerControl_WatchProgressServer) error {
		call := calls.Add(1)
		if call == 1 {
			return status.Error(codes.Unavailable, "startup race")
		}
		data, err := json.Marshal(map[string]any{"type": "progress", "payload": map[string]any{
			"stage": "denoise", "position": call - 1, "total": 30, "step_ms": 2000,
		}})
		if err != nil {
			return err
		}
		frame := &pb.AttemptProgress{RecordOwnerEpoch: open.RecordOwnerEpoch,
			ControlStreamEpoch: open.ControlStreamEpoch, WorkerBootId: open.WorkerBootId,
			RequestId: "watch-reconnect", AttemptOrdinal: 1, Seq: uint64(call), Data: data}
		if err := stream.Send(frame); err != nil {
			return err
		}
		if call == 2 {
			return status.Error(codes.Unavailable, "stream interruption")
		}
		<-stream.Context().Done()
		return nil
	}
	connection, _ := startFakePod(t, t.TempDir(), pod)
	o := hostOwner(t, "progress-reconnect", rentalWiring(connection, private))
	_, _, _, e := o.c.EnsureRental(podRental)
	fatal(t, e)
	waitUntil(t, "progress after the independent watch reconnect", func() bool {
		progress, ok := o.c.LatestProgress("watch-reconnect", 1)
		return ok && progress.Position != nil && *progress.Position == 2
	})
	progress, _ := o.c.LatestProgress("watch-reconnect", 1)
	if calls.Load() != 3 || progress.Stage != "denoise" || progress.StepMS == nil || *progress.StepMS != 2000 {
		t.Fatalf("watch progress after reconnect: calls=%d progress=%+v", calls.Load(), progress)
	}
	pod.mu.Lock()
	snapshots := len(pod.acks)
	pod.mu.Unlock()
	if snapshots != 1 {
		t.Fatalf("progress-only disconnect restarted control: %d snapshots", snapshots)
	}
}

func TestRoundedCountedProgressSurvivesLossyTransport(t *testing.T) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	frames := make(chan map[string]any, 1)
	pod := &fakePod{controlKey: public}
	pod.watchProgress = func(open *pb.ProgressOpen, stream pb.WorkerControl_WatchProgressServer) error {
		for seq := uint64(1); ; seq++ {
			select {
			case fields := <-frames:
				data, err := json.Marshal(map[string]any{"type": "progress", "payload": fields})
				if err != nil {
					return err
				}
				if err := stream.Send(&pb.AttemptProgress{
					RecordOwnerEpoch: open.RecordOwnerEpoch, ControlStreamEpoch: open.ControlStreamEpoch,
					WorkerBootId: open.WorkerBootId, RequestId: "rounded-progress", AttemptOrdinal: 1,
					Seq: seq, Data: data,
				}); err != nil {
					return err
				}
			case <-stream.Context().Done():
				return nil
			}
		}
	}
	connection, _ := startFakePod(t, t.TempDir(), pod)
	o := hostOwner(t, "rounded-progress", rentalWiring(connection, private))
	_, _, _, e := o.c.EnsureRental(podRental)
	fatal(t, e)

	// These are Runtime's six-decimal fractions. The 2 -> 5 jump drops two
	// frames: its 42s interval represents one step, while three steps advanced.
	for _, position := range []int64{0, 1, 2, 5, 18} {
		fraction := math.Round(float64(position)/30*1e6) / 1e6
		frames <- map[string]any{"stage": "denoise", "position": position, "total": 30,
			"stage_fraction": fraction, "overall_fraction": fraction, "step_ms": 42000}
		waitUntil(t, "rounded work coordinate", func() bool {
			progress, ok := o.c.LatestProgress("rounded-progress", 1)
			return ok && progress.Position != nil && *progress.Position == position
		})
		progress, _ := o.c.LatestProgress("rounded-progress", 1)
		if progress.StageFraction == nil || *progress.StageFraction != float64(position)/30 {
			t.Fatalf("counted fraction was not derived from its exact coordinates: %+v", progress)
		}
		if position > 0 {
			want := (30 - position) * 42000
			if !progress.Estimated || math.Abs(float64(progress.RemainingMS-want)) > 20 {
				t.Fatalf("ETA priced coalesced steps as one interval: position=%d got=%d want=%d",
					position, progress.RemainingMS, want)
			}
		}
	}
}

func TestPreparationSnapshotCarriesRentalAndSeparateModelProgress(t *testing.T) {
	o := hostOwner(t, "progress-metadata")
	rental := &orchestrator.RentalProgress{AcceleratorModel: "H100 NVL", AcceleratorCount: 1, HourlyRateUSDMicros: 3236009}
	o.c.ObservePhase("rental-stage", orchestrator.PhaseSample{Name: "booting", Machine: "aldra", Rental: rental})
	rental.AcceleratorModel = "mutated after emission"
	boot, ok := o.c.PreparationPhase("rental-stage")
	if !ok || boot.Rental == nil || boot.Rental.AcceleratorModel != "H100 NVL" {
		t.Fatalf("rental facts not isolated: %+v", boot)
	}
	if _, estimated := boot.Remaining(); estimated {
		t.Fatal("invented a boot denominator")
	}
	first := orchestrator.ModelDownloadProgress{Model: "paul/minimax-h3", Release: "1", Lane: "fp8", Manifest: "manifest", Moved: 9000, Total: 10000}
	second := orchestrator.ModelDownloadProgress{Model: "paul/sdxl", Release: "1", Lane: "bf16", Manifest: "other", Moved: 2, Total: 2}
	o.c.ObservePhase("model-stage", orchestrator.PhaseSample{Name: "downloading", Detail: "package@1", Models: []orchestrator.ModelDownloadProgress{first, second}})
	time.Sleep(10 * time.Millisecond)
	first.Moved, first.OriginBytes = 9500, 500
	o.c.ObservePhase("model-stage", orchestrator.PhaseSample{Name: "downloading", Detail: "package@1", Models: []orchestrator.ModelDownloadProgress{first}})
	phase, _ := o.c.PreparationPhase("model-stage")
	if len(phase.Models) != 2 || phase.Models[0].Model != first.Model || phase.Models[0].Moved != 9500 || phase.Models[0].Rate <= 0 || phase.Models[0].RemainingMS == nil {
		t.Fatalf("model progress lost identity, counters, or measured rate: %+v", phase.Models)
	}
	if *phase.Models[0].RemainingMS > 500 {
		t.Fatalf("remaining estimate included already-held bytes: %+v", phase.Models[0])
	}
	body, err := json.Marshal(phase.Frame("request"))
	must(t, err)
	var frame struct {
		Value struct {
			Detail string `json:"detail"`
			Models []struct {
				Model string `json:"model"`
				Moved uint64 `json:"moved_bytes"`
			} `json:"models"`
		} `json:"value"`
	}
	must(t, json.Unmarshal(body, &frame))
	if frame.Value.Detail != "package@1" || len(frame.Value.Models) != 2 || frame.Value.Models[0].Moved != 9500 {
		t.Fatalf("phase wire lost detail or model counters: %s", body)
	}
	// A heartbeat with unchanged byte counters is not evidence that the old
	// transfer rate continues. Keep the bytes but withdraw its speed and ETA.
	o.c.ObservePhase("model-stage", orchestrator.PhaseSample{Name: "downloading", Models: []orchestrator.ModelDownloadProgress{first}})
	quiet, _ := o.c.PreparationPhase("model-stage")
	if quiet.Models[0].Moved != first.Moved || quiet.Models[0].Rate != 0 || quiet.Models[0].RemainingMS != nil {
		t.Fatalf("unchanged model bytes retained an old estimate: %+v", quiet.Models[0])
	}
	// A new native fetch may start at a smaller counter; it needs a fresh rate basis.
	first.Moved, first.OriginBytes = 0, 0
	o.c.ObservePhase("model-stage", orchestrator.PhaseSample{Name: "downloading", Models: []orchestrator.ModelDownloadProgress{first}})
	reset, _ := o.c.PreparationPhase("model-stage")
	if reset.Models[0].Rate != 0 || reset.Models[0].RemainingMS != nil {
		t.Fatalf("counter reset retained rate: %+v", reset.Models[0])
	}
}

// Reattaching after daemon recovery has durable rental facts but may have no live
// preparation sample yet. The queue fallback must preserve that existing quote.
func TestQueuePhaseFallbackRetainsRecordedRental(t *testing.T) {
	o := hostOwner(t, "queue-rental-progress")
	fatal(t, o.store.RecordRental(records.Rental{ID: "rental-progress", MachineName: "aldra",
		SKU: "h100-nvl", AcceleratorModel: "H100 NVL", AcceleratorCount: 1,
		HourlyRateUSDMicros: 3236009, State: "ready"}))
	_, _, e := o.store.Submit(records.Request{ID: "queue-progress", IdemKey: "queue-progress",
		BodyDigest: "queue-progress", Package: "paul/minimax-h3", Entrypoint: "ref2va",
		State: "queued", Payload: []byte("{}"), Outputs: "[]", Worker: "rental-progress", Rental: true})
	fatal(t, e)
	phase, ok := o.c.QueuePhase("queue-progress")
	if !ok || phase.Name != orchestrator.WaitWorkerStart || phase.Machine != "aldra" || phase.Rental == nil || phase.Rental.AcceleratorModel != "H100 NVL" || phase.Rental.HourlyRateUSDMicros != 3236009 {
		t.Fatalf("fallback lost durable rental details: %+v", phase)
	}
	if !phase.Since.IsZero() || phase.HasBytes {
		t.Fatalf("fallback invented measurements: %+v", phase)
	}
}
