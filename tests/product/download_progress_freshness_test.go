package producttest

import (
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/cli"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/output"
)

func TestDownloadETAExcludesAlreadyRetainedBytes(t *testing.T) {
	o := hostOwner(t, "resumed-download-rate")
	const retained = 8 << 30
	sample := orchestrator.PhaseSample{Name: "downloading", HasBytes: true, Moved: retained, Total: retained + 2<<20}
	o.c.ObservePhase("resumed", sample)
	time.Sleep(time.Millisecond)
	sample.Moved += 1 << 20
	o.c.ObservePhase("resumed", sample)
	phase, _ := o.c.PreparationPhase("resumed")
	remaining, estimated := phase.Remaining()
	// One MiB arrived during this observed interval and one MiB remains. The
	// eight GiB retained before observation cannot make that estimate 8192x faster.
	interval := phase.At.Sub(phase.Since)
	if !estimated || remaining < interval/2 || remaining > interval*2 {
		t.Fatalf("retained bytes distorted remaining time: remaining=%s interval=%s phase=%+v", remaining, interval, phase)
	}
	o.c.ObservePhase("resumed", orchestrator.PhaseSample{Name: "downloading"})
	heartbeat, _ := o.c.PreparationPhase("resumed")
	if !heartbeat.At.Equal(phase.At) || heartbeat.Moved != phase.Moved {
		t.Fatalf("a byte-less heartbeat refreshed transfer evidence: before=%+v after=%+v", phase, heartbeat)
	}
	o.c.ObservePhase("resumed", sample)
	quiet, _ := o.c.PreparationPhase("resumed")
	if _, estimated := quiet.Remaining(); quiet.Rate != 0 || estimated {
		t.Fatalf("unchanged counters kept an optimistic speed/ETA: %+v", quiet)
	}
	sample.Moved += 1 << 18
	o.c.ObservePhase("resumed", sample)
	resumed, _ := o.c.PreparationPhase("resumed")
	if _, estimated := resumed.Remaining(); resumed.Rate <= 0 || !estimated {
		t.Fatalf("fresh progress did not restore measurement: %+v", resumed)
	}
	sample.Moved, sample.Total = 0, 50
	o.c.ObservePhase("resumed", sample)
	reset, _ := o.c.PreparationPhase("resumed")
	if reset.Moved != 0 || reset.Total != 50 || reset.Rate != 0 || reset.Average != 0 {
		t.Fatalf("a smaller fetch underflowed or reused old counters: %+v", reset)
	}
}

func TestQuietPreparationKeepsBytesButHidesOldSpeedAndETA(t *testing.T) {
	phase := orchestrator.PhaseObservation{Name: "downloading", HasBytes: true,
		Since: time.Now().Add(-time.Hour), At: time.Now().Add(-90 * time.Second),
		Moved: 45 << 30, Total: 100 << 30, Rate: 55 << 20, Average: 55 << 20}
	fields := phase.Frame("quiet-download").Value.(map[string]any)
	if fields["moved_bytes"] != phase.Moved || fields["sample_age_ms"].(int64) < 90000 {
		t.Fatalf("quiet transfer lost its last observation: %+v", fields)
	}
	if _, present := fields["rate_bytes_per_second"]; present {
		t.Fatalf("old speed advertised as current: %+v", fields)
	}
	if _, present := fields["remaining_ms"]; present {
		t.Fatalf("old ETA advertised as current: %+v", fields)
	}
	line := cli.HumanPhaseLine(fields)
	if !strings.Contains(line, "last update") || strings.Contains(line, "/s") || strings.Contains(line, "ETA") {
		t.Fatalf("attached CLI hides the lack of updates: %q", line)
	}
	age, moved, total, eta, speed := int64(90000), int64(phase.Moved), int64(phase.Total), int64(126735), phase.Rate
	cell := cli.PhaseCell(api.Lifecycle{Phase: "downloading", PhaseSampleAgeMS: &age,
		PhaseMovedBytes: &moved, PhaseTotalBytes: &total, PhaseRate: &speed, PhaseRemainingMS: &eta})
	if !strings.Contains(cell, "last update") || strings.Contains(cell, "/s") || strings.Contains(cell, "~") {
		t.Fatalf("run list keeps stale estimates: %q", cell)
	}
}

func TestAttachedDownloadAgesItsLastSample(t *testing.T) {
	p, buf := progressSink(output.Mode{Human: true, Color: true}, false)
	t.Cleanup(p.Done)
	e := liveEvent("phase", map[string]any{"phase": "downloading", "sample_age_ms": 0,
		"moved_bytes": 45 << 30, "total_bytes": 100 << 30,
		"rate_bytes_per_second": 55 << 20, "remaining_ms": 126735})
	e.At = time.Now().Add(-90 * time.Second).UTC().Format(time.RFC3339Nano)
	p.On(e)
	if got := buf.String(); !strings.Contains(got, "last update") || strings.Contains(got, "/s") || strings.Contains(got, "ETA") {
		t.Fatalf("quiet attached display kept the original frame's rate: %q", got)
	}
	before := len(buf.String())
	p.On(liveEvent("phase", map[string]any{"phase": "downloading", "sample_age_ms": 0,
		"moved_bytes": 46 << 30, "total_bytes": 100 << 30, "rate_bytes_per_second": 23 << 20}))
	if got := buf.String()[before:]; !strings.Contains(got, "/s") || strings.Contains(got, "last update") {
		t.Fatalf("fresh sample did not replace stale display: %q", got)
	}
}

func TestAttachedDownloadAgesModelsIndependently(t *testing.T) {
	p, buf := progressSink(output.Mode{Human: true, Color: true}, false)
	t.Cleanup(p.Done)
	model := map[string]any{"model": "paul/minimax-h3", "sample_age_ms": 9000,
		"moved_bytes": 45 << 30, "total_bytes": 100 << 30, "rate_bytes_per_second": 55 << 20}
	e := liveEvent("phase", map[string]any{"phase": "downloading", "sample_age_ms": 0,
		"models": []any{model}})
	e.At = time.Now().Add(-2 * time.Second).UTC().Format(time.RFC3339Nano)
	p.On(e)
	if got := buf.String(); !strings.Contains(got, "last update") || strings.Contains(got, "/s") {
		t.Fatalf("model's nine-second sample failed to age beyond eleven seconds: %q", got)
	}
	if model["sample_age_ms"] != 9000 {
		t.Fatalf("rendering mutated the original model observation: %+v", model)
	}
	before := len(buf.String())
	model["sample_age_ms"] = 0
	p.On(liveEvent("phase", map[string]any{"phase": "downloading", "sample_age_ms": 90000,
		"models": []any{model}}))
	if got := buf.String()[before:]; !strings.Contains(got, "/s") {
		t.Fatalf("old aggregate counters hid a freshly reported model rate: %q", got)
	}
}
