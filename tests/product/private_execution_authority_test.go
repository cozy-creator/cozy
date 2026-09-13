package producttest

import (
	"bytes"
	"crypto/ed25519"
	"crypto/rand"
	"testing"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

func TestPrivateExecutionCannotAcquireRentalAuthority(t *testing.T) {
	public, key, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	owner := &orchestrator.PrivateExecutionOwner{Authorization: &pb.SignedExecutionOwnerGrant{Grant: &pb.ExecutionOwnerGrant{RecordOwnerEpoch: 2, RecordOwnerId: "pod-coordinator", WorkerId: "worker", WorkerBootId: "boot", WorkerTlsCertificateDigest: bytes.Repeat([]byte{1}, 32), InitialCapsuleDigest: bytes.Repeat([]byte{2}, 32), ExecutionPublicKey: public}, Signature: bytes.Repeat([]byte{3}, 64)}, PrivateKey: key}
	for _, configure := range []func(*orchestrator.Options){
		func(o *orchestrator.Options) {
			o.AcquireManagedRental = func(records.Request) (orchestrator.PlacementDecision, string, *exit.Error) {
				t.Fatal("pod acquired another rental")
				return orchestrator.PlacementDecision{}, "", nil
			}
		},
		func(o *orchestrator.Options) {
			o.ReleaseManagedRental = func(string) (string, *exit.Error) { t.Fatal("pod released a rental"); return "", nil }
		},
		func(o *orchestrator.Options) {
			o.ReleaseRetainedRental = func(string) (string, *exit.Error) { t.Fatal("pod abandoned a rental"); return "", nil }
		},
	} {
		options := orchestrator.Options{PrivateExecution: owner}
		configure(&options)
		if coordinator, problem := orchestrator.Open(options); problem == nil {
			coordinator.Close(0)
			t.Fatal("pod coordinator accepted a provider mutation callback")
		}
	}
	coordinator, problem := orchestrator.Open(orchestrator.Options{PrivateExecution: owner})
	if problem != nil {
		t.Fatal(problem)
	}
	if _, _, problem := coordinator.EnsureWorker(orchestrator.WorkerLaunchSpec{}); problem == nil {
		coordinator.Close(0)
		t.Fatal("pod coordinator spawned an ungranted worker")
	}
	coordinator.Close(0)
	wrong := *owner
	wrong.PrivateKey = append(ed25519.PrivateKey(nil), key...)
	wrong.PrivateKey[0] ^= 1
	if coordinator, problem := orchestrator.Open(orchestrator.Options{PrivateExecution: &wrong}); problem == nil {
		coordinator.Close(0)
		t.Fatal("pod coordinator accepted a key outside its execution grant")
	}
}
