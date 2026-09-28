package producttest

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/orchestrator"

	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

func TestIdleControlAcceptsMinorSkew(t *testing.T) {
	for _, minor := range []uint32{pb.MinCompatibleWireMinor, pb.WireMinor, pb.WireMinor + 1, pb.MinCompatibleWireMinor - 1} {
		public, private, err := ed25519.GenerateKey(rand.Reader)
		must(t, err)
		pod := &fakePod{controlKey: public, wireMinor: minor}
		connection, _ := startFakePod(t, t.TempDir(), pod)
		var options orchestrator.Options
		rentalWiring(connection, private)(&options)
		ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
		control, problem := orchestrator.DialIdleControl(ctx, connection, options.RentalClaimProof, nil)
		if control != nil {
			must(t, control.Close())
		}
		cancel()
		fatal(t, problem)
		if control == nil {
			t.Fatalf("minor %d did not open idle control", minor)
		}
		pod.mu.Lock()
		mutated := len(pod.offers) + len(pod.prepares) + len(pod.desired)
		pod.mu.Unlock()
		if mutated != 0 {
			t.Fatalf("idle control prepared or dispatched work for minor %d", minor)
		}
	}
}
