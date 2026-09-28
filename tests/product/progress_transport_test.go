package producttest

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
)

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
