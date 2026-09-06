package producttest

import (
	"crypto/ed25519"
	"crypto/rand"
	"testing"

	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// An older pod can remain attached to preserve its data and settle old work,
// but new directives are authored at this client's current wire minor.
func TestRentalReuseRequiresNegotiatedCurrentProtocol(t *testing.T) {
	for _, older := range []bool{false, true} {
		name := "current"
		minor := uint32(pb.WireMinor)
		if older {
			name = "older"
			minor--
		}
		t.Run(name, func(t *testing.T) {
			public, private, err := ed25519.GenerateKey(rand.Reader)
			must(t, err)
			pod := &fakePod{controlKey: public, wireMinor: minor}
			connection, _ := startFakePod(t, t.TempDir(), pod)
			o := hostOwner(t, "protocol-"+name, rentalWiring(connection, private))
			_, _, _, problem := o.c.EnsureRental(podRental)
			fatal(t, problem)
			for _, job := range []bool{false, true} {
				kept, excluded := o.c.ModeCompatibleRentalsWithExclusions([]string{podRental}, job)
				if older {
					if len(kept) != 0 || len(excluded) != 1 || excluded[0].RentalID != podRental || excluded[0].Reason != "protocol_unsupported" {
						t.Fatalf("older claimed pod remained eligible (job=%t): kept=%v excluded=%v", job, kept, excluded)
					}
				} else if len(kept) != 1 || kept[0] != podRental || len(excluded) != 0 {
					t.Fatalf("current claimed pod excluded (job=%t): kept=%v excluded=%v", job, kept, excluded)
				}
			}
		})
	}
}
