package producttest

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	"github.com/cozy-creator/cozy/internal/workertls"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/proto"
)

var observationHome = flag.String("worker-observation-home", "", "existing owned Creator home for a bounded Claim-only diagnostic")
var observationRental = flag.String("worker-observation-rental", "", "exact owned rental identity")
var observationOutput = flag.String("worker-observation-output", "", "new private JSONL diagnostic file")

// A Claim changes the stream epoch and owner-presence observation. This diagnostic
// sends no SnapshotAck, DesiredState, offer or control of an execution. Compare
// separate process/journal samples before and after attaching; it is not proof of
// observer-free progress. All credentials stay in the existing rental signer.
func TestOwnedWorkerObservation(t *testing.T) {
	if *observationHome == "" || *observationRental == "" || *observationOutput == "" {
		t.Skip("requires explicit owned home, rental and private output path")
	}
	_, err := os.Stat(filepath.Join(*observationHome, "creator.sqlite"))
	must(t, err)
	layout, problem := home.Open(*observationHome)
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	target, problem := rental.Resolver(layout, store)(*observationRental)
	fatal(t, problem)
	remote := target.Connection
	instance := (orchestrator.WorkerLaunchSpec{Connection: remote}).InstanceID()
	legacy, problem := store.OpenAttemptsOf(instance)
	fatal(t, problem)
	if len(legacy) != 0 {
		t.Fatal("diagnostic cannot supersede a controller with legacy attempts")
	}
	proof, problem := rental.ClaimProof(layout)(remote, 1)
	fatal(t, problem)
	pin, err := workertls.LoadPin(remote.CACert)
	must(t, err)
	connection, err := grpc.NewClient(remote.Addr,
		grpc.WithTransportCredentials(credentials.NewTLS(pin.TLSConfig())),
		grpc.WithDefaultCallOptions(grpc.MaxCallRecvMsgSize(int(pb.MaxInlineControlBytes))))
	must(t, err)
	defer connection.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	stream, err := pb.NewWorkerControlClient(connection).Control(ctx)
	must(t, err)
	defer stream.CloseSend()
	claim := &pb.Claim{RecordOwnerEpoch: 1, RecordOwnerId: "cozy-local-client", WorkerId: remote.WorkerID,
		WorkerBootId: remote.WorkerBootID, WireMinor: pb.WireMinor, Proof: proof}
	out, err := os.OpenFile(*observationOutput, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	must(t, err)
	defer out.Close()
	encoder := json.NewEncoder(out)
	must(t, encoder.Encode(map[string]any{"kind": "attachment", "at": time.Now().UTC(),
		"worker": remote.WorkerID, "boot": remote.WorkerBootID, "rental": remote.RentalID,
		"scope": "Claim-only observer changes stream epoch/presence; no execution commands"}))
	must(t, stream.Send(&pb.RecordOwnerFrame{Msg: &pb.RecordOwnerFrame_Claim{Claim: claim}}))
	var epoch uint64
	var snapshot bool
	observations := 0
	for {
		frame, err := stream.Recv()
		must(t, err)
		if proto.Size(frame) > pb.MaxInlineControlBytes {
			t.Fatal("diagnostic frame exceeded protocol bound")
		}
		if ack := frame.GetClaimAck(); ack != nil {
			if !ack.Accepted || ack.WorkerId != remote.WorkerID || ack.WorkerBootId != remote.WorkerBootID || ack.RecordOwnerEpoch != 1 || ack.ControlStreamEpoch == 0 {
				t.Fatal("diagnostic claim did not preserve the pinned owner/worker/boot")
			}
			epoch = ack.ControlStreamEpoch
			continue
		}
		if snap := frame.GetSnapshot(); snap != nil {
			if epoch == 0 || snap.RecordOwnerEpoch != 1 || snap.WorkerBootId != remote.WorkerBootID || snap.ControlStreamEpoch != epoch || !bytes.Equal(canonical.Digest(snap.SnapshotCanonicalBytes), snap.SnapshotDigest) {
				t.Fatal("diagnostic snapshot failed its identity or digest fence")
			}
			must(t, encoder.Encode(map[string]any{"kind": "snapshot", "at": time.Now().UTC(), "body": json.RawMessage(snap.SnapshotCanonicalBytes)}))
			snapshot = true
		}
		if observed := frame.GetObservedState(); observed != nil {
			if epoch == 0 || observed.WorkerBootId != remote.WorkerBootID || observed.ControlStreamEpoch != epoch || observed.RecordOwnerEpoch != 1 {
				t.Fatal("diagnostic observation changed its stream identity")
			}
			raw, err := (protojson.MarshalOptions{UseProtoNames: true}).Marshal(observed)
			must(t, err)
			must(t, encoder.Encode(map[string]any{"kind": "observed", "at": time.Now().UTC(), "body": json.RawMessage(raw)}))
			observations++
		}
		if snapshot && observations >= 2 {
			must(t, out.Sync())
			t.Logf("saved fenced snapshot and %d observations; no dispatch or execution control sent", observations)
			return
		}
	}
}
