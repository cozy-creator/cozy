package producttest

import (
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/cli"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/orchestrator"
)

// cl-121: the preparation a queued request is waiting on is OBSERVED, not logged and
// discarded.
//
// The defect these prove absent had three layers and only the last one is this repo's:
// the pod host reported byte counters, the owner received them, and the owner wrote them
// to a log line that fires once per stage change. A 210 GB materialization therefore
// produced four log lines and nothing on any surface a person looks at, so a healthy run
// and a hung one rendered identically for two and a half hours.
//
// These tests assert the samples FIRE — that a stream which advances produces an
// observation that advances — rather than that the wiring exists. Wiring that is never
// exercised is exactly how the defect survived: the protocol's own comment says the
// counters are monotonic within a call, and nothing had ever checked that anybody read
// a second one.

// TestAPhaseWithNoDenominatorRendersBytesAndRate is the honesty arm. A producer that
// declares no total must not cause a fraction to be invented, and the cell must still be
// useful: the rate is the number that answers "is this still moving", and it needs no
// denominator at all.
func TestAPhaseWithNoDenominatorRendersBytesAndRate(t *testing.T) {
	moved, rate := int64(13_314_398_617), 67_108_864.0
	cell := cli.PhaseCell(api.Lifecycle{
		Status: "queued", Phase: orchestrator.PhaseDownloading, PhaseMachine: "uzume",
		PhaseMovedBytes: &moved, PhaseRate: &rate,
	})
	for _, want := range []string{"downloading models", "/s"} {
		if !strings.Contains(cell, want) {
			t.Fatalf("the cell %q does not carry %q", cell, want)
		}
	}
	if strings.Contains(cell, "uzume") || strings.Contains(cell, "waiting:") {
		t.Fatalf("progress contains redundant context: %q", cell)
	}
	if strings.Contains(cell, "%") || strings.Contains(cell, " / ") {
		t.Fatalf("the cell invented a fraction with no declared total: %q", cell)
	}

	// With a total the fraction becomes real, and only then.
	moved, total := int64(50<<30), int64(100<<30)
	cell = cli.PhaseCell(api.Lifecycle{
		Status: "queued", Phase: orchestrator.PhaseDownloading,
		PhaseMovedBytes: &moved, PhaseTotalBytes: &total, PhaseRate: &rate,
	})
	for _, want := range []string{"50", " / ", "100"} {
		if !strings.Contains(cell, want) {
			t.Fatalf("the cell does not show finished / total bytes: %q", cell)
		}
	}
	t.Log(cell)
}

// TestAPhaseWithNoCountersStillNamesItself is the other honesty arm, and the one that
// splits an otherwise silent wait without repeating the machine or elapsed fields.
func TestAPhaseWithNoCountersStillNamesItself(t *testing.T) {
	elapsed := int64(41_000)
	cell := cli.PhaseCell(api.Lifecycle{
		Status: "queued", Phase: orchestrator.PhaseProvisioning, PhaseMachine: "fumiya", PhaseElapsedMS: &elapsed,
	})
	if cell != "provisioning machine" {
		t.Fatalf("a counterless phase should name only the activity: %q", cell)
	}
	if strings.Contains(cell, "0 B") {
		t.Fatalf("a counterless phase rendered a zero byte count: %q", cell)
	}
	cell = cli.PhaseCell(api.Lifecycle{Status: "queued", Phase: orchestrator.PhaseResolving,
		PhaseMachine: "fumiya", PhaseElapsedMS: &elapsed})
	if cell != "preparing downloads" {
		t.Fatalf("resolution should not claim model bytes are downloading: %q", cell)
	}
}

// TestTheProviderDrawsTheProvisioningBoundary. The split between "waiting on the
// provider's host" and "the container is up and coming to life" is the difference between
// a delay we cannot fix and one we can. It is drawn where the PROVIDER draws it — a started
// container — from the boot the Hub reports, and a Hub that reports no boot yields the
// coarse phase rather than a guess.
func TestTheProviderDrawsTheProvisioningBoundary(t *testing.T) {
	booting := func(boot hub.RentalBoot) *hub.RentalBoot { return &boot }
	for _, c := range []struct {
		name   string
		rental hub.Rental
		want   string
	}{
		{"no attempt yet", hub.Rental{State: "pending_acquisition"}, orchestrator.PhaseAcquiring},
		{"creating the pod", hub.Rental{State: "acquiring", Boot: booting(hub.RentalBoot{Attempt: 1, State: "obligated"})}, orchestrator.PhaseProvisioning},
		{"no container yet", hub.Rental{State: "booting", Boot: booting(hub.RentalBoot{Attempt: 1, State: "booting"})}, orchestrator.PhaseProvisioning},
		{"created container", hub.Rental{State: "booting", Boot: booting(hub.RentalBoot{Attempt: 1, State: "booting", Container: "created"})}, orchestrator.PhaseProvisioning},
		{"image pull read from the boot log", hub.Rental{State: "booting", Boot: booting(hub.RentalBoot{Attempt: 1, State: "booting", Activity: "pulling_image"})}, orchestrator.PhasePullingImage},
		{"started container", hub.Rental{State: "booting", Boot: booting(hub.RentalBoot{Attempt: 1, State: "booting", Container: "running", RuntimeObserved: true})}, orchestrator.PhaseBooting},
		{"container phase", hub.Rental{State: "booting", Boot: booting(hub.RentalBoot{Attempt: 1, State: "booting", Phase: "container"})}, orchestrator.PhaseProvisioning},
		{"supervisor starting", hub.Rental{State: "booting", Boot: booting(hub.RentalBoot{Attempt: 1, State: "booting", Phase: "host"})}, orchestrator.PhaseBooting},
		// A rental back in acquisition after a refused attempt is buying AGAIN. Measured on
		// run 207 — 32.1s of a 227.9s wait was one datacenter declining, and the request
		// silently moved.
		{"replanning", hub.Rental{State: "pending_acquisition", Boot: booting(hub.RentalBoot{Attempt: 2, State: "replanning"})}, orchestrator.PhaseReplanning},
		{"refused, no boot", hub.Rental{State: "pending_acquisition", Failure: &hub.RentalFailure{Code: "x"}}, orchestrator.PhaseReplanning},
		{"ready", hub.Rental{State: "ready"}, ""},
		{"failed", hub.Rental{State: "failed", Failure: &hub.RentalFailure{Code: "x"}}, ""},
	} {
		if got := orchestrator.PhaseOfHubRental(c.rental); got != c.want {
			t.Errorf("%s: PhaseOfHubRental = %q, want %q", c.name, got, c.want)
		}
	}
}

func TestPackageSetupAndModelLoadingHaveDistinctLabels(t *testing.T) {
	complete := int64(100 << 30)
	if got := cli.PhaseCell(api.Lifecycle{Status: "queued", Kind: "invocation", Phase: orchestrator.PhasePreparing,
		PhaseMovedBytes: &complete, PhaseTotalBytes: &complete}); got != "setting up package" {
		t.Fatalf("completed downloads hide package setup: %q", got)
	}
	for _, tc := range []struct {
		phase, kind, want string
	}{
		{orchestrator.PhasePreparing, "invocation", "setting up package"},
		{orchestrator.PhaseWarming, "invocation", "loading models"},
		{orchestrator.PhasePreparing, "job", "preparing inputs"},
	} {
		if got := cli.PhaseCell(api.Lifecycle{Status: "queued", Phase: tc.phase, Kind: tc.kind}); got != tc.want {
			t.Errorf("%s/%s = %q, want %q", tc.kind, tc.phase, got, tc.want)
		}
	}
}
