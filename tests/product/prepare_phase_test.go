package producttest

import (
	"crypto/ed25519"
	"crypto/rand"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/cli"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
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

// TestPreparationPhaseAdvancesWithTheStream drives the real PodHost prepare lane and
// watches the phase lane move with it.
func TestPreparationPhaseAdvancesWithTheStream(t *testing.T) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	pod := &fakePod{controlKey: public, downloadSamples: 5}
	connection, _ := startFakePod(t, t.TempDir(), pod)
	o := hostOwner(t, "prepare-phase", rentalWiring(connection, private))

	instance, _, _, e := o.c.EnsureRental(podRental)
	fatal(t, e)

	// Sample the lane while the stream runs. The prepare is on its own goroutine — a
	// multi-GiB materialization must never sit on the control stream's read loop — so
	// the observation is read concurrently exactly as `run list` reads it.
	type reading struct {
		name  string
		moved uint64
		rate  float64
	}
	readings := make(chan []reading, 1)
	go func() {
		var seen []reading
		deadline := time.Now().Add(20 * time.Second)
		for time.Now().Before(deadline) {
			if phase, ok := o.c.PreparationPhase(instance); ok {
				last := reading{name: phase.Name, moved: phase.Moved, rate: phase.Rate}
				if len(seen) == 0 || seen[len(seen)-1] != last {
					seen = append(seen, last)
				}
				if phase.Name == orchestrator.PhaseWarming {
					break
				}
			}
			time.Sleep(time.Millisecond)
		}
		readings <- seen
	}()

	fatal(t, o.c.ConvergePackageSet(instance, []*pb.DownloadPackageRef{{
		Package: "cozy/h3-package", Release: "1.0.7"}}, nil))
	waitUntil(t, "the prepared placement_set on WorkerControl", func() bool {
		pod.mu.Lock()
		defer pod.mu.Unlock()
		return len(pod.desired) >= 1
	})

	seen := <-readings
	downloads, advanced, rated := 0, false, false
	var previous uint64
	for _, r := range seen {
		if r.name != orchestrator.PhaseDownloading {
			continue
		}
		downloads++
		if downloads > 1 && r.moved > previous {
			advanced = true
		}
		if r.rate > 0 {
			rated = true
		}
		previous = r.moved
	}
	// THE ASSERTION THAT MATTERS. One reading proves the lane is connected; several
	// increasing readings prove it is being fed. The old code would have passed a
	// "downloading was observed" check and failed this one, because it read the counters
	// only when the stage changed.
	if downloads < 2 {
		t.Fatalf("the lane saw %d downloading readings across %d samples; it is connected but not fed: %+v",
			downloads, pod.downloadSamples, seen)
	}
	if !advanced {
		t.Fatalf("downloading never advanced across %d readings: %+v", downloads, seen)
	}
	if !rated {
		t.Fatalf("no reading carried a measured rate: %+v", seen)
	}

	final, ok := o.c.PreparationPhase(instance)
	if !ok || final.Name != orchestrator.PhaseWarming {
		t.Fatalf("after PREPARED the phase is %q, want %q", final.Name, orchestrator.PhaseWarming)
	}
	// The denominator the pod declared is carried through unchanged, and the phase never
	// reports more moved than declared.
	if final.Total == 0 || final.Moved > final.Total {
		t.Fatalf("the terminal reading does not carry the pod's own bounds: %+v", final)
	}
}

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
	for _, want := range []string{"downloading", "uzume", "/s"} {
		if !strings.Contains(cell, want) {
			t.Fatalf("the cell %q does not carry %q", cell, want)
		}
	}
	if strings.Contains(cell, "%") || strings.Contains(cell, " of ") {
		t.Fatalf("the cell invented a fraction with no declared total: %q", cell)
	}

	// With a total the fraction becomes real, and only then.
	total := int64(226_000_000_000)
	cell = cli.PhaseCell(api.Lifecycle{
		Status: "queued", Phase: orchestrator.PhaseDownloading,
		PhaseMovedBytes: &moved, PhaseTotalBytes: &total, PhaseRate: &rate,
	})
	if !strings.Contains(cell, " of ") {
		t.Fatalf("the cell does not name the declared total: %q", cell)
	}
}

// TestAPhaseWithNoCountersStillNamesItself is the other honesty arm, and the one that
// splits an otherwise silent wait: acquiring a machine has no byte counter and is still
// worth saying, with the one quantity that was actually measured.
func TestAPhaseWithNoCountersStillNamesItself(t *testing.T) {
	elapsed := int64(41_000)
	cell := cli.PhaseCell(api.Lifecycle{
		Status: "queued", Phase: orchestrator.PhaseProvisioning, PhaseElapsedMS: &elapsed,
	})
	if !strings.Contains(cell, "provisioning") || !strings.Contains(cell, "41s") {
		t.Fatalf("a counterless phase must still name itself and its elapsed: %q", cell)
	}
	if strings.Contains(cell, "0 B") {
		t.Fatalf("a counterless phase rendered a zero byte count: %q", cell)
	}
}

// TestTheProviderDrawsTheProvisioningBoundary. The split between "waiting in the
// provider's queue" and "the container is up and coming to life" is the difference
// between a delay we cannot fix and one we can — a 9 GB image pull. It is therefore
// drawn where the PROVIDER draws it, and a hub that says nothing yields the coarse
// phase rather than a guess.
func TestTheProviderDrawsTheProvisioningBoundary(t *testing.T) {
	for _, c := range []struct {
		state, provider, container string
		retrying                   bool
		want                       string
	}{
		{"pending_acquisition", "", "", false, orchestrator.PhaseAcquiring},
		{"pending_acquisition", "CREATED", "", false, orchestrator.PhaseProvisioning},
		{"pending_acquisition", "RUNNING", "PULLING", false, orchestrator.PhaseProvisioning},
		{"pending_acquisition", "RUNNING", "RUNNING", false, orchestrator.PhaseBooting},
		{"pending_acquisition", "RUNNING", "", false, orchestrator.PhaseBooting},
		// A rental back in pending_acquisition carrying a failure is buying AGAIN. It
		// outranks every other reading: an operator must not be shown a first attempt's
		// vocabulary for a second attempt's spend. Measured on run 207 — 32.1s of a
		// 227.9s wait was one datacenter declining, and the request silently moved.
		{"pending_acquisition", "", "", true, orchestrator.PhaseReplanning},
		{"pending_acquisition", "RUNNING", "RUNNING", true, orchestrator.PhaseReplanning},
		{"ready", "RUNNING", "RUNNING", false, ""},
		{"failed", "EXITED", "EXITED", true, ""},
	} {
		if got := orchestrator.PhaseOfHubRental(c.state, c.provider, c.container, c.retrying); got != c.want {
			t.Errorf("PhaseOfHubRental(%q,%q,%q,retrying=%t) = %q, want %q",
				c.state, c.provider, c.container, c.retrying, got, c.want)
		}
	}
}
