package records

import (
	"bytes"
	"database/sql"
	"encoding/json"
	"errors"
	"math"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/protobuf/proto"
)

// The weights subject is a Host observation in the same cursor as source progress.
// It does not assign a transaction ID or manufacture a second declaration.
type ModelWeightsProgress struct {
	ModelCheckpointProgress
	Subject *pb.WeightsCheckpointSubject
	Attempt int64
}

func weightsCheckpointSubject(status *pb.WeightsTransactionStatus) (*pb.WeightsCheckpointSubject, *exit.Error) {
	if status == nil || status.RequestId == "" || status.OutputSlot == "" || status.WriterEpoch == 0 || status.AttemptOrdinal == 0 ||
		status.WriterEpoch > math.MaxInt64 || status.AttemptOrdinal > math.MaxInt64 || len(status.TensorfsDeclarationDigest) != 32 {
		return nil, exit.New(exit.Validation, "weights checkpoint status has invalid authority")
	}
	invocation, err := canonical.Raw(status.InvocationSpecDigest)
	if err != nil {
		return nil, exit.New(exit.Validation, "weights checkpoint status has invalid invocation")
	}
	if _, err := canonical.Raw(status.WeightsTransactionId); err != nil {
		return nil, exit.New(exit.Validation, "weights checkpoint has invalid transaction ID")
	}
	return &pb.WeightsCheckpointSubject{RequestId: status.RequestId, InvocationSpecDigest: invocation, OutputSlot: status.OutputSlot,
		WeightsTransactionId: status.WeightsTransactionId, WriterEpoch: status.WriterEpoch, TensorfsDeclarationDigest: status.TensorfsDeclarationDigest}, nil
}

func CheckpointFromRef(slot string, ref *pb.CheckpointRef) (ModelCheckpoint, *exit.Error) {
	var out ModelCheckpoint
	if ref == nil || ref.Head == nil || ref.Head.Length == 0 || ref.Head.Length > math.MaxInt64 || ref.Index > math.MaxInt64 || ref.Bytes > math.MaxInt64 {
		return out, exit.New(exit.Validation, "checkpoint reference has invalid bounds")
	}
	head, headErr := canonical.Spell(ref.Head.Digest)
	plan, planErr := canonical.Spell(ref.PlanDigest)
	if headErr != nil || planErr != nil || slot == "" {
		return out, exit.New(exit.Validation, "checkpoint reference has invalid identity")
	}
	return ModelCheckpoint{Slot: slot, HeadID: head, HeadLength: int64(ref.Head.Length), PlanDigest: plan, Index: int64(ref.Index), Bytes: int64(ref.Bytes)}, nil
}

func scanWeightsProgress(row interface{ Scan(...any) error }) (ModelWeightsProgress, error) {
	var out ModelWeightsProgress
	var subject []byte
	var observed, acknowledged string
	if err := row.Scan(&out.WorkerBootID, &subject, &out.Attempt, &observed, &acknowledged); err != nil {
		return out, err
	}
	out.Subject = new(pb.WeightsCheckpointSubject)
	if err := proto.Unmarshal(subject, out.Subject); err != nil {
		return out, err
	}
	if observed != "" {
		if err := json.Unmarshal([]byte(observed), &out.Observed); err != nil {
			return out, err
		}
	}
	if acknowledged != "" {
		out.Acknowledged = new(ModelCheckpoint)
		if err := json.Unmarshal([]byte(acknowledged), out.Acknowledged); err != nil {
			return out, err
		}
	}
	return out, nil
}

func (s *Store) ModelWeightsProgress(requestID string) ([]ModelWeightsProgress, *exit.Error) {
	rows, err := s.db.Query(`SELECT worker_boot_id,subject,attempt,observed,acknowledged FROM request_model_checkpoints
 WHERE request_id=? AND kind='weights' ORDER BY slot`, requestID)
	if err != nil {
		return nil, exit.Internalf("cannot read weights checkpoint cursors: %s", err)
	}
	defer rows.Close()
	var out []ModelWeightsProgress
	for rows.Next() {
		row, err := scanWeightsProgress(rows)
		if err != nil {
			return nil, exit.Internalf("cannot decode weights checkpoint cursor: %s", err)
		}
		out = append(out, row)
	}
	if err := rows.Err(); err != nil {
		return nil, exit.Internalf("cannot finish weights checkpoint cursors: %s", err)
	}
	return out, nil
}

func sameWeightsWork(a, b *pb.WeightsCheckpointSubject) bool {
	return a != nil && b != nil && a.RequestId == b.RequestId && a.OutputSlot == b.OutputSlot && a.WeightsTransactionId == b.WeightsTransactionId &&
		bytes.Equal(a.InvocationSpecDigest, b.InvocationSpecDigest) && bytes.Equal(a.TensorfsDeclarationDigest, b.TensorfsDeclarationDigest)
}

// Confirm current request/attempt ownership inside the same transaction as progress.
func weightsCheckpointOwner(tx *sql.Tx, subject *pb.WeightsCheckpointSubject, attempt int64, instance string) *exit.Error {
	var intent ModelTransferIntent
	var raw, transferState, attemptInstance, invocation, attemptState, requestState string
	var current int64
	err := tx.QueryRow(`SELECT t.intent,t.state,a.instance_id,a.invocation_digest,a.state,r.state,r.ordinal
 FROM request_model_transfers t JOIN requests r ON r.id=t.request_id
 JOIN attempts a ON a.request_id=r.id WHERE r.id=? AND a.attempt=?`, subject.RequestId, attempt).
		Scan(&raw, &transferState, &attemptInstance, &invocation, &attemptState, &requestState, &current)
	if err != nil || json.Unmarshal([]byte(raw), &intent) != nil {
		return exit.New(exit.Conflict, "weights checkpoint has no request owner")
	}
	spec, _ := canonical.Spell(subject.InvocationSpecDigest)
	if current != attempt || instance != "" && attemptInstance != instance || invocation != spec ||
		requestState == "canceled" || transferState == "failed" || transferState == "canceling" || transferState == "canceled" || transferState == "completed" ||
		(attemptState != "offered" && attemptState != "accepted" && attemptState != "recovered_open" && attemptState != "terminal") {
		return exit.Named(exit.Conflict, "model_transfer.checkpoint_superseded", "weights checkpoint is not owned by the current producer attempt")
	}
	for _, output := range intent.Outputs {
		if output.Name == subject.OutputSlot {
			return nil
		}
	}
	return exit.New(exit.Validation, "weights checkpoint output is not declared by this request")
}

func (s *Store) ObserveModelWeightsCheckpoint(instance, boot string, status *pb.WeightsTransactionStatus) *exit.Error {
	subject, problem := weightsCheckpointSubject(status)
	if problem != nil {
		return problem
	}
	if boot == "" {
		return exit.New(exit.Validation, "weights checkpoint has no worker boot")
	}
	var checkpoint ModelCheckpoint
	if status.Checkpoint != nil {
		checkpoint, problem = CheckpointFromRef(subject.OutputSlot, status.Checkpoint)
		if problem != nil {
			return problem
		}
	}
	tx, err := s.db.Begin()
	if err != nil {
		return exit.Internalf("cannot begin weights checkpoint observation: %s", err)
	}
	defer tx.Rollback()
	if problem := weightsCheckpointOwner(tx, subject, int64(status.AttemptOrdinal), instance); problem != nil {
		return problem
	}
	prior, err := scanWeightsProgress(tx.QueryRow(`SELECT worker_boot_id,subject,attempt,observed,acknowledged
 FROM request_model_checkpoints WHERE request_id=? AND kind='weights' AND slot=?`, subject.RequestId, subject.OutputSlot))
	if err != nil && !errors.Is(err, sql.ErrNoRows) {
		return exit.Internalf("cannot read prior weights progress: %s", err)
	}
	observed := ""
	if err == nil {
		if !sameWeightsWork(prior.Subject, subject) || int64(status.AttemptOrdinal) < prior.Attempt ||
			(int64(status.AttemptOrdinal) == prior.Attempt && boot != prior.WorkerBootID) ||
			(boot == prior.WorkerBootID && subject.WriterEpoch < prior.Subject.WriterEpoch) {
			return exit.Named(exit.Conflict, "model_transfer.checkpoint_superseded", "weights checkpoint changed its frozen work or writer")
		}
		if checkpoint.HeadID == "" && int64(status.AttemptOrdinal) == prior.Attempt && boot == prior.WorkerBootID {
			checkpoint = prior.Observed
		}
		if checkpoint.HeadID != "" {
			checkpoint, problem = advanceCheckpoint(prior.ModelCheckpointProgress, boot, checkpoint)
			if problem != nil {
				return problem
			}
		}
	}
	if checkpoint.HeadID != "" {
		raw, _ := json.Marshal(checkpoint)
		observed = string(raw)
	}
	raw, err := proto.MarshalOptions{Deterministic: true}.Marshal(subject)
	if err != nil {
		return exit.Internalf("cannot encode weights checkpoint subject: %s", err)
	}
	_, err = tx.Exec(`INSERT INTO request_model_checkpoints(request_id,kind,slot,subject,attempt,worker_boot_id,observed)
 VALUES(?,'weights',?,?,?,?,?) ON CONFLICT(request_id,kind,slot) DO UPDATE SET subject=excluded.subject,
 attempt=excluded.attempt,worker_boot_id=excluded.worker_boot_id,observed=excluded.observed`,
		subject.RequestId, subject.OutputSlot, raw, int64(status.AttemptOrdinal), boot, observed)
	if err != nil {
		return exit.Internalf("cannot record weights checkpoint: %s", err)
	}
	if err := tx.Commit(); err != nil {
		return exit.Internalf("cannot commit weights checkpoint: %s", err)
	}
	return nil
}

func (s *Store) AcknowledgeModelWeightsCheckpoint(subject *pb.WeightsCheckpointSubject, attempt int64, boot, previous string, checkpoint ModelCheckpoint) *exit.Error {
	tx, err := s.db.Begin()
	if err != nil {
		return exit.Internalf("cannot begin weights checkpoint acknowledgment: %s", err)
	}
	defer tx.Rollback()
	if problem := weightsCheckpointOwner(tx, subject, attempt, ""); problem != nil {
		return problem
	}
	row, err := scanWeightsProgress(tx.QueryRow(`SELECT worker_boot_id,subject,attempt,observed,acknowledged FROM request_model_checkpoints
 WHERE request_id=? AND kind='weights' AND slot=?`, subject.RequestId, subject.OutputSlot))
	if err != nil {
		return exit.Internalf("cannot read weights checkpoint acknowledgment: %s", err)
	}
	if !proto.Equal(subject, row.Subject) || row.Attempt != attempt || row.WorkerBootID != boot {
		return exit.Named(exit.Conflict, "model_transfer.checkpoint_superseded", "weights checkpoint writer was replaced")
	}
	if problem := checkCheckpointAcknowledgment(row.ModelCheckpointProgress, previous, checkpoint); problem != nil {
		return problem
	}
	raw, _ := json.Marshal(checkpoint)
	if _, err := tx.Exec(`UPDATE request_model_checkpoints SET acknowledged=? WHERE request_id=? AND kind='weights' AND slot=?`, string(raw), subject.RequestId, subject.OutputSlot); err != nil {
		return exit.Internalf("cannot acknowledge weights checkpoint: %s", err)
	}
	if err := tx.Commit(); err != nil {
		return exit.Internalf("cannot commit weights checkpoint acknowledgment: %s", err)
	}
	return nil
}
