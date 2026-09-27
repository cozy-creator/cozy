package producttest

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc/status"
)

func TestProtocolProbeNeverAnswerHasDeadline(t *testing.T) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	entered := make(chan time.Duration, 1)
	var claims atomic.Int64
	pod := &fakePod{controlKey: public}
	// The probe's bound is read where it arrives: the deadline the owner propagated. The
	// peer then answers as the expired call would, so the owner's handling of a deadline
	// is exercised without waiting the bound out.
	pod.protocolInfo = func(ctx context.Context, _ *pb.ProtocolInfoRequest) (*pb.ProtocolInfoResult, error) {
		remaining := time.Duration(0)
		if deadline, ok := ctx.Deadline(); ok {
			remaining = time.Until(deadline)
		}
		entered <- remaining
		return nil, status.FromContextError(context.DeadlineExceeded).Err()
	}
	pod.onFrame = func(frame *pb.RecordOwnerFrame, _ func(*pb.WorkerFrame) error) (bool, error) {
		if frame.GetClaim() != nil {
			claims.Add(1)
		}
		return false, nil
	}
	connection, _ := startFakePod(t, t.TempDir(), pod)
	var options orchestrator.Options
	rentalWiring(connection, private)(&options)
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	result := make(chan *exit.Error, 1)
	go func() {
		control, problem := orchestrator.DialIdleControl(ctx, connection, options.RentalClaimProof, nil)
		if control != nil {
			_ = control.Close()
		}
		result <- problem
	}()
	select {
	case remaining := <-entered:
		if remaining <= 0 || remaining > hub.Timeout {
			t.Fatalf("protocol probe has no existing short-RPC bound: %s", remaining)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("protocol probe never reached the pinned peer")
	}
	select {
	case problem := <-result:
		if problem == nil || problem.Code != exit.Unavailable {
			t.Fatalf("unexpected probe outcome: %v", problem)
		}
	case <-time.After(hub.Timeout):
		t.Fatal("an expired protocol probe was not answered")
	}
	if claims.Load() != 0 {
		t.Fatal("an unanswered protocol probe mutated ownership")
	}
}

func TestOwnerCloseCancelsProtocolProbeBeforeClaim(t *testing.T) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	entered, canceled, release := make(chan struct{}, 1), make(chan struct{}, 1), make(chan struct{})
	var claims atomic.Int64
	pod := &fakePod{controlKey: public}
	pod.protocolInfo = func(ctx context.Context, _ *pb.ProtocolInfoRequest) (*pb.ProtocolInfoResult, error) {
		entered <- struct{}{}
		select {
		case <-ctx.Done():
			canceled <- struct{}{}
		case <-release:
		}
		return nil, status.FromContextError(context.Canceled).Err()
	}
	pod.onFrame = func(frame *pb.RecordOwnerFrame, _ func(*pb.WorkerFrame) error) (bool, error) {
		if frame.GetClaim() != nil {
			claims.Add(1)
		}
		return false, nil
	}
	connection, _ := startFakePod(t, t.TempDir(), pod)
	owner := hostOwner(t, "cancel-protocol-probe-"+records.NewID("proof"), rentalWiring(connection, private))
	// This test-owned release lets even the old defective implementation clean up.
	t.Cleanup(func() { close(release) })
	go func() { _, _, _, _ = owner.c.EnsureRental(podRental) }()
	select {
	case <-entered:
	case <-time.After(5 * time.Second):
		t.Fatal("owner never reached the protocol probe")
	}
	closed := make(chan struct{})
	go func() { owner.c.Close(0); close(closed) }()
	select {
	case <-canceled:
	case <-time.After(2 * time.Second):
		t.Fatal("owner shutdown did not cancel the in-flight protocol probe")
	}
	select {
	case <-closed:
	case <-time.After(2 * time.Second):
		t.Fatal("owner shutdown did not finish")
	}
	if claims.Load() != 0 {
		t.Fatal("shutdown sent a Claim after the unanswered probe")
	}
}
