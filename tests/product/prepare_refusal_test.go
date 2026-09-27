package producttest

import (
	"strings"
	"testing"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// A PREPARE STATUS NOBODY ENUMERATED MUST NOT BECOME A REDIAL CADENCE (proto-035).
//
// `classifyPrepareEnd` used to LIST the terminal codes and let everything else fall through
// to "no verdict; the reconnect re-issues". codes.NotFound fell through. The desire names an
// immutable identity, so a host that cannot find it now cannot find it on the next dial
// either — the owner re-issued the same revision forever on a pod at roughly a dollar an
// hour. It is the same defect that produced the Unimplemented loop (cozy #298), and the same
// one that re-issued `request_invalid` 1,198 times on SDXL run 198.
//
// The rule: a refusal is resumable only if the refusing party could answer differently to
// the IDENTICAL request later. A deny-list cannot enforce that, because the violating codes
// are precisely the ones not yet enumerated. The classifier now allow-lists the transient
// ones and treats everything else as the host's word.
//
// This runs the real orchestrator against a real second implementation of the worker
// protocol over pinned TLS. Nothing is stubbed inside the code under test.
func TestAnUnenumeratedPrepareStatusIsTheHostsVerdictNotARedial(t *testing.T) {
	for _, arm := range []struct {
		name string
		code codes.Code
	}{
		{"not found", codes.NotFound},
		{"out of range", codes.OutOfRange},
		{"data loss", codes.DataLoss},
	} {
		t.Run(arm.name, func(t *testing.T) {
			pod := &standInPod{prepareStatus: status.New(arm.code,
				"this host cannot serve that desire")}
			o, instance := attachStandInRental(t, "prepare-"+strings.ReplaceAll(arm.name, " ", "-"), pod)

			fatal(t, o.c.ConvergePackageSet(instance, delegatedPackages(), nil))

			if _, ok := waitEvent(o, "REFUSED before it was applied", 30*time.Second); !ok {
				t.Fatalf("a %s prepare status was not the host's verdict:\n%s",
					arm.code, pod.report())
			}
			// The money assertion: it was not ALSO deferred and re-issued. A deferral is
			// what put the owner back on the 200 ms redial.
			if n := countEvents(o, "DEFERRED without a verdict"); n != 0 {
				t.Fatalf("a %s prepare status was deferred %d time(s) and re-issued; "+
					"a fixed request cannot be answered differently later:\n%s",
					arm.code, n, pod.report())
			}
			// Five redial windows (200 ms each). A deferral would have re-issued the same
			// desire repeatedly; a verdict is asked for once.
			settled := pod.prepares()
			time.Sleep(time.Second)
			if grew := pod.prepares() - settled; grew > 0 {
				t.Fatalf("the owner re-issued the same desire %d more time(s) after a %s "+
					"verdict; a fixed request cannot be answered differently later",
					grew, arm.code)
			}
		})
	}
}

// The other half of the rule: a condition of the MOMENT still defers, so a network blip or a
// busy host does not strand a paid pod on a permanent verdict it never gave.
func TestATransientPrepareStatusStillDefers(t *testing.T) {
	pod := &standInPod{prepareStatus: status.New(codes.Unavailable, "restarting")}
	o, instance := attachStandInRental(t, "prepare-unavailable", pod)

	fatal(t, o.c.ConvergePackageSet(instance, delegatedPackages(), nil))

	if _, ok := waitEvent(o, "DEFERRED without a verdict", 30*time.Second); !ok {
		t.Fatalf("an Unavailable host was treated as a permanent verdict:\n%s", pod.report())
	}
}
