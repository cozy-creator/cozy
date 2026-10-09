package producttest

import (
	"math"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/output"
)

func TestMeasuredStageETARequiresValidUnitsAndSourceTiming(t *testing.T) {
	sample := func(stage string, position, total, at, step float64) orchestrator.ProgressSnapshot {
		t.Helper()
		value, ok := orchestrator.DecodeProgressSnapshot(map[string]any{"stage": stage, "position": position,
			"total": total, "sample_unix_ms": at, "step_ms": step})
		if !ok {
			t.Fatal("valid progress sample refused")
		}
		return value
	}
	previous := sample("decode_video", 170, 362, 100000, 1900)
	for _, test := range []struct {
		name              string
		current, previous orchestrator.ProgressSnapshot
		want              int64
		known             bool
	}{
		{"first_unit", sample("denoise", 1, 30, 0, 35541.02015681565), orchestrator.ProgressSnapshot{}, 1030689, true},
		{"first_chunk_unknown", sample("decode_video", 170, 362, 100000, 1879.7), orchestrator.ProgressSnapshot{}, 0, false},
		{"chunk", sample("decode_video", 187, 362, 101880, 1879.7), previous, 19352, true},
		{"coalesced", sample("decode_video", 204, 362, 103760, 1879.7), previous, 17472, true},
		{"missing_source_time", sample("decode_video", 187, 362, 0, 1879.7), previous, 0, false},
		{"same_source_time", sample("decode_video", 187, 362, 100000, 1879.7), previous, 0, false},
		{"reversed_time", sample("decode_video", 187, 362, 99999, 1879.7), previous, 0, false},
		{"repeated_count", sample("decode_video", 170, 362, 101880, 1879.7), previous, 0, false},
		{"counter_reset", sample("decode_video", 1, 362, 101880, 1879.7), previous, 0, false},
		{"total_changed", sample("decode_video", 187, 400, 101880, 1879.7), previous, 0, false},
		{"stage_changed", sample("denoise", 187, 362, 101880, 1879.7), previous, 0, false},
		{"duration_overflow", sample("denoise", 1, 30, 0, math.MaxFloat64), orchestrator.ProgressSnapshot{}, 0, false},
		{"nonfinite_interval", sample("denoise", 1, 30, 0, math.Inf(1)), orchestrator.ProgressSnapshot{}, 0, false},
		{"complete_stage", sample("denoise", 30, 30, 0, 0), orchestrator.ProgressSnapshot{}, 0, true},
	} {
		t.Run(test.name, func(t *testing.T) {
			got, known := test.current.EstimateStage(test.previous)
			if known != test.known || known && got.RemainingMS != test.want {
				t.Fatalf("estimate=%+v known=%v, want %d known=%v", got, known, test.want, test.known)
			}
		})
	}
	changedUnit := sample("decode_video", 187, 362, 101880, 1879.7)
	changedUnit.Unit = "frames"
	if _, known := changedUnit.EstimateStage(previous); known {
		t.Fatal("changed units inherited the previous sample rate")
	}
	explicitRate := 20.0
	changedUnit.Rate = &explicitRate
	if got, known := changedUnit.EstimateStage(previous); !known || got.RemainingMS != 8750 {
		t.Fatalf("explicit current rate was coupled to prior units: %+v %v", got, known)
	}
	for _, rate := range []float64{0, 20} {
		value, ok := orchestrator.DecodeProgressSnapshot(map[string]any{"stage": "frames", "position": 100., "total": 400., "rate": rate, "unit": "frames"})
		if !ok {
			t.Fatal("rate sample refused")
		}
		got, known := value.EstimateStage(orchestrator.ProgressSnapshot{})
		if known != (rate > 0) || known && got.RemainingMS != 15000 {
			t.Fatalf("producer rate not respected: %+v %v", got, known)
		}
	}
}

func TestLiveETAUsesStageUnitsAndNeverProjectsWeightedFutureStages(t *testing.T) {
	p, _ := progressSink(output.Mode{Human: true, Live: true}, false)
	t.Cleanup(p.Done)
	p.On(liveEvent("accepted", nil))
	p.On(liveEvent("progress", map[string]any{"stage": "denoise", "overall_fraction": .15, "sample_unix_ms": 1000}))
	p.On(liveEvent("progress", map[string]any{"stage": "denoise", "position": 1, "total": 30,
		"overall_fraction": .173333, "step_ms": 35541.02015681565, "sample_unix_ms": 39000}))
	got := liveFrame(p, time.Now())
	if !strings.Contains(got, "1/30") || !strings.Contains(got, "ETA ~17m11s") || strings.Count(got, "ETA") != 1 {
		t.Fatalf("live first-step ETA did not stay stage-local: %s", got)
	}
	for _, line := range strings.Split(got, "\n") {
		if strings.Contains(line, "overall") && strings.Contains(line, "ETA") {
			t.Fatalf("weighted overall progress invented future-stage time: %s", line)
		}
	}
	p.On(liveEvent("progress", map[string]any{"stage": "denoise", "position": 30, "total": 30,
		"overall_fraction": .85, "sample_unix_ms": 700000}))
	p.On(liveEvent("progress", map[string]any{"stage": "decode_video", "overall_fraction": .85, "sample_unix_ms": 700001}))
	p.On(liveEvent("progress", map[string]any{"stage": "decode_video", "position": 170, "total": 362,
		"overall_fraction": .90, "step_ms": 1879.7, "sample_unix_ms": 720000}))
	p.On(liveEvent("progress", map[string]any{"stage": "decode_video", "position": 187, "total": 362,
		"overall_fraction": .91, "step_ms": 1879.7, "sample_unix_ms": 721880}))
	if got = liveFrame(p, time.Now()); !strings.Contains(got, "187/362") || !strings.Contains(got, "ETA ~19s") {
		t.Fatalf("live decode used chunk duration as a frame duration: %s", got)
	}
}
