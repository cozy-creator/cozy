package producttest

import (
	"bytes"
	"crypto/ed25519"
	"crypto/rand"
	"database/sql"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

func completedPublicationFixture(t *testing.T) (*records.Store, *sql.DB, string) {
	t.Helper()
	store, db, id := publicationRetryFixture(t)
	_, err := db.Exec(`UPDATE requests SET worker='rental-unavailable',rental=1`)
	must(t, err)
	changed, problem := store.RetryModelTransferPublication(id, "settlement-proof")
	fatal(t, problem)
	if !changed {
		t.Fatal("successful producer fixture did not resume publication")
	}
	return store, db, id
}

func publicationStateOwner(t *testing.T, name string, store *records.Store, releases *atomic.Int64) *owner {
	t.Helper()
	return hostOwner(t, name, func(options *orchestrator.Options) {
		options.Store = store
		options.ReleaseManagedRental = func(string) (string, *exit.Error) { releases.Add(1); return "", nil }
	})
}

func TestCompletedPublicationSucceedsWithoutWorkerSession(t *testing.T) {
	store, db, id := completedPublicationFixture(t)
	rows, problem := store.AllModelTransferWeights(id, 1)
	fatal(t, problem)
	fatal(t, store.CompleteModelTransfer(id, map[string]string{"model": rows[0].ManifestID}))
	before, problem := store.Attempts(id)
	fatal(t, problem)
	var releases atomic.Int64
	first := publicationStateOwner(t, "publication-missing-session", store, &releases)
	waited := make(chan *exit.Error, 1)
	go func() { _, problem := first.c.AwaitSettled(id, 5*time.Second); waited <- problem }()
	fatal(t, first.c.ResumeModelTransfers())
	waitUntil(t, "completed publication visible without worker ACK", func() bool {
		row, problem := store.RequestRow(id)
		fatal(t, problem)
		return row.State == "succeeded"
	})
	fatal(t, <-waited)
	first.close()
	store.Close()
	reopened, problem := records.Open(dbPathForRetry(t, db))
	fatal(t, problem)
	defer reopened.Close()
	second := publicationStateOwner(t, "publication-restarted-session", reopened, &releases)
	fatal(t, second.c.ResumeModelTransfers())
	defer second.close()
	after, problem := reopened.Attempts(id)
	fatal(t, problem)
	old, _ := json.Marshal(before)
	fresh, _ := json.Marshal(after)
	var cleaned int
	must(t, db.QueryRow(`SELECT media_cleaned FROM attempts WHERE request_id=?`, id).Scan(&cleaned))
	if !bytes.Equal(old, fresh) || len(after) != 1 || after[0].State != "terminal" || cleaned != 0 {
		t.Fatal("request settlement changed successful attempt or pretended worker cleanup occurred")
	}
	transfer, problem := reopened.ModelTransferOf(id)
	fatal(t, problem)
	if transfer.State != "completed" || transfer.Checkpoints["model"] != rows[0].ManifestID || releases.Load() != 0 {
		t.Fatalf("publication result or provider retention changed: %+v releases=%d", transfer, releases.Load())
	}
	events, problem := reopened.EventsAfter(id, 0, 100)
	fatal(t, problem)
	completions := 0
	for _, event := range events {
		if event.Type == "request.completed" {
			completions++
		}
	}
	if completions != 1 {
		t.Fatalf("got %d completion events across owner restart", completions)
	}
	pending, problem := reopened.OpenAttemptsOf(after[0].InstanceID)
	fatal(t, problem)
	if len(pending) != 1 {
		t.Fatal("settled publication lost the retained terminal cleanup obligation")
	}
}

func TestIncompleteOrFailedPublicationCannotSucceedWithoutSession(t *testing.T) {
	for _, state := range []string{"finalizing", "failed", "failed-producer"} {
		t.Run(state, func(t *testing.T) {
			store, db, id := completedPublicationFixture(t)
			if state == "failed" {
				fatal(t, store.FailModelTransfer(id, "upload_failed", "bytes are missing"))
			}
			if state == "failed-producer" {
				rows, problem := store.AllModelTransferWeights(id, 1)
				fatal(t, problem)
				fatal(t, store.CompleteModelTransfer(id, map[string]string{"model": rows[0].ManifestID}))
				_, err := db.Exec(`UPDATE attempts SET terminal_status='FAILED'`)
				must(t, err)
			}
			var releases atomic.Int64
			o := publicationStateOwner(t, "publication-incomplete-"+state, store, &releases)
			fatal(t, o.c.ResumeModelTransfers())
			o.close()
			row, problem := store.RequestRow(id)
			fatal(t, problem)
			if row.State == "succeeded" || releases.Load() != 0 {
				t.Fatalf("unproven publication succeeded: %s", row.State)
			}
		})
	}
}

func TestCompletedPublicationActualRetainedSnapshot(t *testing.T) {
	if *retainedPublicationProof == "" || *retainedPublicationRequest == "" {
		t.Skip("requires isolated post-publication producer snapshot")
	}
	raw, err := os.ReadFile(*retainedPublicationProof)
	must(t, err)
	path := filepath.Join(t.TempDir(), "creator.sqlite")
	must(t, os.WriteFile(path, raw, 0600))
	store, problem := records.Open(path)
	fatal(t, problem)
	defer store.Close()
	id := *retainedPublicationRequest
	before, problem := store.Attempts(id)
	fatal(t, problem)
	weights, problem := store.AllModelTransferWeights(id, 1)
	fatal(t, problem)
	var releases atomic.Int64
	o := publicationStateOwner(t, "publication-actual-retained", store, &releases)
	fatal(t, o.c.ResumeModelTransfers())
	waitUntil(t, "actual retained publication succeeded without worker", func() bool { row, p := store.RequestRow(id); fatal(t, p); return row.State == "succeeded" })
	o.close()
	after, problem := store.Attempts(id)
	fatal(t, problem)
	retained, problem := store.AllModelTransferWeights(id, 1)
	fatal(t, problem)
	for _, pair := range [][2]any{{before, after}, {weights, retained}} {
		left, _ := json.Marshal(pair[0])
		right, _ := json.Marshal(pair[1])
		if !bytes.Equal(left, right) {
			t.Fatal("settlement changed actual producer or checkpoint facts")
		}
	}
	if releases.Load() != 0 {
		t.Fatal("settlement attempted provider teardown without worker ACK")
	}
	t.Logf("%d exact terminal attempt(s), %d checkpoint receipts preserved; user request succeeded without ACK", len(after), len(retained))
}

func TestCompletedPublicationEventuallyAcknowledgesSameOutcome(t *testing.T) {
	store, db, id := completedPublicationFixture(t)
	rows, problem := store.AllModelTransferWeights(id, 1)
	fatal(t, problem)
	fatal(t, store.CompleteModelTransfer(id, map[string]string{"model": rows[0].ManifestID}))
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	pod := &fakePod{controlKey: public}
	connection, _ := startFakePod(t, t.TempDir(), pod)
	instance := (orchestrator.WorkerLaunchSpec{Connection: connection}).InstanceID()
	declared, err := json.Marshal([]orchestrator.WeightsOutput{{OutputID: "model", MimeType: orchestrator.WeightsManifestMime, MaxBytes: 1 << 30}})
	must(t, err)
	_, err = db.Exec(`UPDATE requests SET worker=?`, podRental)
	must(t, err)
	_, err = db.Exec(`UPDATE attempts SET instance_id=?,session_id=?,terminal_id='out-retained',weights_outputs=?`, instance, podBootID, string(declared))
	must(t, err)
	attempt, problem := store.AttemptRow(id, 1)
	fatal(t, problem)
	// Bind the schema corpus to this real RecordOwner before the fixture begins.
	receipt, err := canonical.Read(rows[0].Receipt, &pb.WeightsReceipt{})
	must(t, err)
	receipt["owner_authority_scope"] = "cozy-local-client"
	receiptBytes, err := canonical.Write(receipt)
	must(t, err)
	receiptDigest, err := canonical.Spell(canonical.Digest(receiptBytes))
	must(t, err)
	body, err := canonical.Read(attempt.TerminalBody, &pb.AttemptOutcomeBody{})
	must(t, err)
	body["weights_receipts"] = []canonical.Value{map[string]canonical.Value{
		"weights_receipt_digest":          receiptDigest,
		"weights_receipt_canonical_bytes": base64.StdEncoding.EncodeToString(receiptBytes),
	}}
	outcomeBytes, err := canonical.Write(body)
	must(t, err)
	spelledOutcome, err := canonical.Spell(canonical.Digest(outcomeBytes))
	must(t, err)
	_, err = db.Exec(`UPDATE attempts SET terminal_body=?,terminal_digest=?`, outcomeBytes, spelledOutcome)
	must(t, err)
	_, err = db.Exec(`UPDATE request_model_transfer_outputs SET receipt=?,receipt_digest=?`, receiptBytes, receiptDigest)
	must(t, err)
	attempt, problem = store.AttemptRow(id, 1)
	fatal(t, problem)
	specDigest, err := canonical.Raw(attempt.InvocationDigest)
	must(t, err)
	outcomeDigest, err := canonical.Raw(attempt.TerminalDigest)
	must(t, err)
	pod.snapshotHeld = []*pb.HeldAttempt{{RequestId: id, AttemptOrdinal: 1, Kind: pb.AttemptKind_ATTEMPT_KIND_JOB,
		State: pb.AttemptState_ATTEMPT_STATE_OUTCOME_PENDING_ACK, InvocationSpecDigest: specDigest,
		OutcomeId: attempt.TerminalID, OutcomeDigest: outcomeDigest}}
	var releases, acks atomic.Int64
	first := publicationStateOwner(t, "publication-before-reconnect", store, &releases)
	fatal(t, first.c.ResumeModelTransfers())
	waitUntil(t, "publication succeeds before peer reconnect", func() bool { row, p := store.RequestRow(id); fatal(t, p); return row.State == "succeeded" })
	first.close()
	var outcome *pb.AttemptOutcome
	pod.onFrame = func(frame *pb.RecordOwnerFrame, send func(*pb.WorkerFrame) error) (bool, error) {
		if snapshot := frame.GetSnapshotAck(); snapshot != nil {
			outcome = &pb.AttemptOutcome{RecordOwnerEpoch: snapshot.RecordOwnerEpoch, ControlStreamEpoch: snapshot.ControlStreamEpoch,
				WorkerBootId: podBootID, RequestId: id, AttemptOrdinal: 1, InvocationSpecDigest: specDigest,
				OutcomeId: attempt.TerminalID, OutcomeDigest: outcomeDigest, OutcomeCanonicalBytes: attempt.TerminalBody}
			return true, send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_AttemptOutcome{AttemptOutcome: outcome}})
		}
		if ack := frame.GetOutcomeAck(); ack != nil {
			if !bytes.Equal(ack.OutcomeDigest, outcomeDigest) || ack.OutcomeId != attempt.TerminalID || ack.RequestId != id {
				return true, fmt.Errorf("ACK substituted retained outcome")
			}
			if acks.Add(1) == 1 {
				return true, send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_AttemptOutcome{AttemptOutcome: outcome}})
			}
			return true, nil
		}
		return false, nil
	}
	second := hostOwner(t, "publication-after-reconnect", rentalWiring(connection, private), func(options *orchestrator.Options) {
		options.Store = store
		options.ReleaseManagedRental = func(string) (string, *exit.Error) { releases.Add(1); return "", nil }
	})
	_, _, _, problem = second.c.EnsureRental(podRental)
	fatal(t, problem)
	waitUntil(t, "same retained outcome ACKed on actual control stream", func() bool { return acks.Load() >= 2 })
	row, problem := store.RequestRow(id)
	fatal(t, problem)
	attempts, problem := store.Attempts(id)
	fatal(t, problem)
	events, problem := store.EventsAfter(id, 0, 100)
	fatal(t, problem)
	completions := 0
	for _, event := range events {
		if event.Type == "request.completed" {
			completions++
		}
	}
	if row.State != "succeeded" || len(attempts) != 1 || attempts[0].State != "closed" || !bytes.Equal(attempts[0].TerminalBody, attempt.TerminalBody) || completions != 1 {
		t.Fatalf("ACK replay changed completed publication: state=%s attempts=%d completions=%d", row.State, len(attempts), completions)
	}
	pod.mu.Lock()
	offers := len(pod.offers)
	pod.mu.Unlock()
	if offers != 0 {
		t.Fatal("publication cleanup reexecuted the producer")
	}
}
