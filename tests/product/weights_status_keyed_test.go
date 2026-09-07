package producttest

import (
	"crypto/ed25519"
	"crypto/rand"
	"database/sql"
	"fmt"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/protobuf/proto"
)

// The actual full H3 closure contains 5,424 objects. Drive its status consumer
// over the claimed TLS stream; individual status events must not decode that
// whole closure. No tensor payload, Hub, or provider is involved.
func TestWeightsStatusUsesOnlyTheNamedObject(t *testing.T) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	type peer struct {
		offer *pb.AttemptOffer
		send  func(*pb.WorkerFrame) error
	}
	connected := make(chan peer, 1)
	pod := &fakePod{controlKey: public, serve: true, jobReady: true}
	pod.onFrame = func(frame *pb.RecordOwnerFrame, send func(*pb.WorkerFrame) error) (bool, error) {
		if offer := frame.GetAttemptOffer(); offer != nil {
			connected <- peer{offer, send}
			return true, nil
		}
		return false, nil
	}
	connection, _ := startFakePod(t, t.TempDir(), pod)
	o := hostOwner(t, "weights-status-keyed", rentalWiring(connection, private))
	id, _, problem := o.c.Submit(orchestrator.Submission{IdemKey: "keyed-status", Package: "cozy/h3-package", Entrypoint: "four-lane", PlanID: "sha256:" + strings.Repeat("35", 32), Release: "1.0.7", Kind: "job", Org: "paul", Payload: []byte(`{}`), Outputs: []string{"model"}, WeightsOutputs: []orchestrator.WeightsOutput{{OutputID: "model", MimeType: orchestrator.WeightsManifestMime, MaxBytes: 1 << 30}}, Worker: podRental, Rental: true, RentalRequired: true})
	fatal(t, problem)
	var p peer
	select {
	case p = <-connected:
	case <-time.After(10 * time.Second):
		t.Fatal("claimed peer never received the job offer")
	}
	invocation, err := canonical.Spell(p.offer.InvocationSpecDigest)
	must(t, err)
	objects := make([]records.ModelTransferObject, 5424)
	for i := range objects {
		objects[i] = records.ModelTransferObject{ObjectID: fmt.Sprintf("sha256:%064x", i+1), Length: 4096, SourceRef: "native-retained-object"}
	}
	weights := records.ModelTransferWeights{RequestID: id, Attempt: int64(p.offer.AttemptOrdinal), OutputSlot: "model", ManifestID: objects[0].ObjectID, ManifestLength: 4096, InvocationDigest: invocation, TransactionID: "weights-status-proof", Objects: objects}
	fatal(t, o.store.RecordModelTransferWeights(weights))
	full, problem := o.store.ModelTransferWeights(id, weights.Attempt, "model")
	fatal(t, problem)
	if full == nil || len(full.Objects) != len(objects) {
		t.Fatal("the ordinary transfer getter lost its full object roster")
	}
	db, err := sql.Open("sqlite", o.l.DB)
	must(t, err)
	defer db.Close()
	read := func(objectID string) (state string, sequence int64) {
		t.Helper()
		must(t, db.QueryRow(`SELECT state,update_sequence FROM request_model_transfer_objects WHERE request_id=? AND attempt=? AND output_slot='model' AND object_id=?`, id, weights.Attempt, objectID).Scan(&state, &sequence))
		return
	}
	frame := &pb.WeightsTransferStatus{RecordOwnerEpoch: p.offer.RecordOwnerEpoch, ControlStreamEpoch: p.offer.ControlStreamEpoch, WorkerBootId: p.offer.WorkerBootId, RequestId: id, AttemptOrdinal: p.offer.AttemptOrdinal, InvocationSpecDigest: p.offer.InvocationSpecDigest, OutputSlot: "model", WeightsTransactionId: weights.TransactionID, ObjectId: objects[1].ObjectID, Length: 4096, OperationId: "publication-proof", GrantRevision: 1, State: pb.WeightsTransferState_WEIGHTS_TRANSFER_STATE_ACCEPTED}
	send := func(status *pb.WeightsTransferStatus) {
		t.Helper()
		must(t, p.send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_WeightsTransferStatus{WeightsTransferStatus: status}}))
	}
	wait := func(objectID string, sequence int64) {
		t.Helper()
		deadline := time.Now().Add(20 * time.Second)
		for time.Now().Before(deadline) {
			if _, got := read(objectID); got == sequence {
				return
			}
			time.Sleep(time.Millisecond)
		}
		t.Fatalf("named object did not reach sequence %d", sequence)
	}
	frame.UpdateSequence = 1
	send(frame)
	wait(frame.ObjectId, 1)
	// Warm the transport before measuring 64 same-state progress updates. The
	// 32 MiB ceiling leaves ample protocol/SQLite headroom but rejects allocating
	// 128 full 5,424-object rosters. This is allocation, not a timing assertion.
	runtime.GC()
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	start := time.Now()
	for sequence := uint64(2); sequence <= 65; sequence++ {
		next := proto.Clone(frame).(*pb.WeightsTransferStatus)
		next.UpdateSequence, next.TransferredBytes = sequence, sequence
		send(next)
	}
	wait(frame.ObjectId, 65)
	runtime.ReadMemStats(&after)
	allocated := after.TotalAlloc - before.TotalAlloc
	t.Logf("5424-object closure, 64 real TLS status events: %s, %d allocated bytes", time.Since(start), allocated)
	if allocated > 32<<20 {
		t.Errorf("individual status updates allocated %d bytes: full-roster work remains in the hot path", allocated)
	}
	// A scan trap in an unrelated row makes any accidental full roster decode
	// fail deterministically. The full-reader API must still detect the corrupt
	// row; a valid status for a different object must not read it.
	_, err = db.Exec(`UPDATE request_model_transfer_objects SET length='scan-trap' WHERE request_id=? AND object_id=?`, id, objects[len(objects)-1].ObjectID)
	must(t, err)
	if _, problem := o.store.ModelTransferWeights(id, weights.Attempt, "model"); problem == nil {
		t.Fatal("the scan trap did not reject a full roster read")
	}
	frame.UpdateSequence, frame.TransferredBytes = 66, 66
	send(frame)
	wait(frame.ObjectId, 66)
	_, err = db.Exec(`UPDATE request_model_transfer_objects SET length=4096 WHERE request_id=? AND object_id=?`, id, objects[len(objects)-1].ObjectID)
	must(t, err)
	// Invalid identities and state regressions precede a valid barrier object
	// on the same stream, so their refusal is observed without a sleep guess.
	for _, mutate := range []func(*pb.WeightsTransferStatus){
		func(s *pb.WeightsTransferStatus) { s.WeightsTransactionId = "other" },
		func(s *pb.WeightsTransferStatus) { s.InvocationSpecDigest = make([]byte, 32) },
		func(s *pb.WeightsTransferStatus) { s.AttemptOrdinal++ },
		func(s *pb.WeightsTransferStatus) { s.OutputSlot = "unknown" },
		func(s *pb.WeightsTransferStatus) { s.ObjectId = "sha256:" + strings.Repeat("f", 64) },
		func(s *pb.WeightsTransferStatus) { s.Length++ },
		func(s *pb.WeightsTransferStatus) { s.UpdateSequence = 1 },
		func(s *pb.WeightsTransferStatus) {
			s.State = pb.WeightsTransferState_WEIGHTS_TRANSFER_STATE_UNSPECIFIED
		},
		func(s *pb.WeightsTransferStatus) { s.TransferredBytes = s.Length + 1 },
		func(s *pb.WeightsTransferStatus) { s.WorkerBootId = "other-boot" },
	} {
		bad := proto.Clone(frame).(*pb.WeightsTransferStatus)
		bad.UpdateSequence = 67
		mutate(bad)
		send(bad)
	}
	barrier := proto.Clone(frame).(*pb.WeightsTransferStatus)
	barrier.ObjectId, barrier.UpdateSequence = objects[2].ObjectID, 1
	send(barrier)
	wait(barrier.ObjectId, 1)
	if state, sequence := read(frame.ObjectId); state != "accepted" || sequence != 66 {
		t.Fatalf("invalid status changed the named row: %s/%d", state, sequence)
	}
}

// Exercise the real durable transition's wake decision, including the stale
// failure rule: keep the verdict, but never roll byte/revision counters back.
func TestWeightsStatusCommittedStateChange(t *testing.T) {
	store, db, id := publicationRetryFixture(t)
	rows, problem := store.AllModelTransferWeights(id, 1)
	fatal(t, problem)
	object := rows[0].Objects[0]
	_, err := db.Exec(`UPDATE request_model_transfer_objects SET operation_id='',grant_revision=0,update_sequence=0,state='pending',transferred=0,safe_code='',safe_detail='' WHERE request_id=?`, id)
	must(t, err)
	object.OperationID, object.State = "proof", "accepted"
	object.GrantRevision, object.UpdateSequence, object.Transferred = 1, 1, 0
	cases := []struct {
		name    string
		mutate  func(*records.ModelTransferObject)
		changed bool
		code    string
	}{
		{"accepted", func(*records.ModelTransferObject) {}, true, ""},
		{"exact replay", func(*records.ModelTransferObject) {}, false, ""},
		{"same state progress", func(o *records.ModelTransferObject) { o.UpdateSequence++; o.Transferred++ }, false, ""},
		{"stale sequence", func(o *records.ModelTransferObject) { o.UpdateSequence-- }, false, "model_transfer.object_status_superseded"},
		{"new grant same state", func(o *records.ModelTransferObject) { o.GrantRevision = 4; o.UpdateSequence = 1 }, false, ""},
		{"stale failure retains verdict", func(o *records.ModelTransferObject) {
			o.State = "failed"
			o.GrantRevision = 1
			o.Transferred = 0
			o.SafeCode = "origin_refused"
		}, true, ""},
		{"same grant cannot undo failure", func(o *records.ModelTransferObject) { o.State = "accepted"; o.GrantRevision = 4; o.UpdateSequence = 2 }, false, "model_transfer.object_status_superseded"},
		{"new grant retries failure", func(o *records.ModelTransferObject) {
			o.State = "accepted"
			o.GrantRevision = 5
			o.UpdateSequence = 1
			o.SafeCode = ""
		}, true, ""},
		{"held", func(o *records.ModelTransferObject) { o.State = "held"; o.UpdateSequence = 2; o.Transferred = o.Length }, true, ""},
		{"held replay", func(*records.ModelTransferObject) {}, false, ""},
		{"held absorbs newer failure", func(o *records.ModelTransferObject) { o.State = "failed"; o.GrantRevision++ }, false, "model_transfer.object_status_superseded"},
		{"changed identity", func(o *records.ModelTransferObject) { o.Length++ }, false, "model_transfer.object_status_changed"},
		{"unknown object", func(o *records.ModelTransferObject) { o.ObjectID = "absent" }, false, "*"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			next := object
			tc.mutate(&next)
			changed, problem := store.RecordModelTransferObjectStatus(next)
			code := ""
			if problem != nil {
				code = problem.ErrName()
			}
			if changed != tc.changed || tc.code == "*" && problem == nil || tc.code != "*" && code != tc.code {
				t.Fatalf("changed=%t code=%q; want %t/%q", changed, code, tc.changed, tc.code)
			}
			if problem == nil {
				rows, problem := store.ModelTransferObjects(id, object.Attempt, object.OutputSlot)
				fatal(t, problem)
				for _, row := range rows {
					if row.ObjectID == object.ObjectID {
						object = row
					}
				}
				if tc.name == "stale failure retains verdict" && (object.GrantRevision != 4 || object.Transferred != 1 || object.SafeCode != "origin_refused") {
					t.Fatalf("stale verdict changed counters: %+v", object)
				}
			}
		})
	}
}
