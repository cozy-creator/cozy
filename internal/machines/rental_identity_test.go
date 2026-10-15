package machines

import (
	"context"
	"crypto/ed25519"
	"crypto/tls"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/machinev1"
	"github.com/cozy-creator/cozy/internal/orchestrator"
)

// A run on a recorded ready rental dials it as recorded and reads nothing at its Hub. Only once
// a connection to that record broke is the Hub asked where the machine is now.
func TestRentalDialReadsTheHubOnlyAfterItsRecordBroke(t *testing.T) {
	recorded := &orchestrator.RemoteTarget{Connection: &orchestrator.WorkerConnection{RentalID: "pr-x", Addr: "127.0.0.1:1"}}
	hubReads := 0
	r := &Resolver{
		Recorded: func(string) (*orchestrator.RemoteTarget, *exit.Error) { return recorded, nil },
		Rentals: func(string) (*orchestrator.RemoteTarget, *exit.Error) {
			hubReads++
			return recorded, nil
		},
	}
	if remote, problem := r.rentalIdentity("pr-x"); problem != nil || remote != recorded || hubReads != 0 {
		t.Fatalf("a recorded ready rental was read at its Hub: %v %d", problem, hubReads)
	}
	public, private, _ := ed25519.GenerateKey(nil)
	client, err := machinev1.Dial("127.0.0.1:1", &tls.Config{InsecureSkipVerify: true}, "worker",
		machinev1.Signer{Public: public, Sign: func(b []byte) []byte { return ed25519.Sign(private, b) }})
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()
	r.v1 = map[string]*keptV1{"pr-x": {client: client}}
	for deadline := time.Now().Add(10 * time.Second); !client.Broken(); {
		ctx, cancel := context.WithTimeout(t.Context(), 200*time.Millisecond)
		_, _ = client.Status(ctx)
		cancel()
		if time.Now().After(deadline) {
			t.Fatal("a refused connection never broke")
		}
	}
	if _, problem := r.rentalIdentity("pr-x"); problem != nil || hubReads != 1 {
		t.Fatalf("a broken connection's next dial did not ask the Hub: %v %d", problem, hubReads)
	}
	r.Recorded = func(string) (*orchestrator.RemoteTarget, *exit.Error) { return nil, exit.New(exit.NotFound, "unrecorded") }
	r.v1 = nil
	if _, problem := r.rentalIdentity("pr-x"); problem != nil || hubReads != 2 {
		t.Fatalf("an unrecorded rental was not read at its Hub: %v %d", problem, hubReads)
	}
}
