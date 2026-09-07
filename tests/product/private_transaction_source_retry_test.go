package producttest

import (
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

// The edited program may reuse an acknowledged acquisition, but the descendant
// needs its own custody acknowledgment before the predecessor's hold is released.
// This tests the durable owner boundary; live tests verify Hub and native bytes.
func TestPrivateTransactionRetryAcquiresIndependentSourceCustody(t *testing.T) {
	store, prior, retry, checkpoint := sourceRetryRecords(t, "unchanged")
	parent, available, problem := store.RetriedSourceCheckpoints(retry.ID)
	fatal(t, problem)
	if parent != prior.ID || len(available) != 1 || available[0].Acknowledged == nil || *available[0].Acknowledged != checkpoint {
		t.Fatalf("exact acknowledged acquisition unavailable: parent=%s progress=%+v", parent, available)
	}
	before, problem := store.ModelSourceProgress(prior.ID)
	fatal(t, problem)
	pending, problem := store.RetriedSourceCustodyPending(prior.ID)
	fatal(t, problem)
	if !pending {
		t.Fatal("predecessor could release its source before the descendant acquired custody")
	}
	current, problem := store.ModelSourceProgress(retry.ID)
	fatal(t, problem)
	if len(current) != 0 {
		t.Fatal("admission fabricated descendant source custody")
	}
	changed := checkpoint
	changed.HeadID = "sha256:" + strings.Repeat("9", 64)
	if problem := store.AdoptRetriedSourceCheckpoint(retry.ID, prior.ID, changed); problem == nil {
		t.Fatal("adoption accepted a different checkpoint under the predecessor's custody")
	}
	if problem := store.AdoptRetriedSourceCheckpoint(retry.ID, "another-request", checkpoint); problem == nil {
		t.Fatal("adoption accepted an unrelated predecessor")
	}
	fatal(t, store.AdoptRetriedSourceCheckpoint(retry.ID, prior.ID, checkpoint))
	fatal(t, store.AdoptRetriedSourceCheckpoint(retry.ID, prior.ID, checkpoint))
	current, problem = store.ModelSourceProgress(retry.ID)
	fatal(t, problem)
	if len(current) != 1 || current[0].Acknowledged == nil || *current[0].Acknowledged != checkpoint {
		t.Fatalf("verified adoption did not independently persist exact custody: %+v", current)
	}
	pending, problem = store.RetriedSourceCustodyPending(prior.ID)
	fatal(t, problem)
	if pending {
		t.Fatal("independently acknowledged descendant still requires predecessor's hold")
	}
	after, problem := store.ModelSourceProgress(prior.ID)
	fatal(t, problem)
	if !reflect.DeepEqual(before, after) {
		t.Fatal("descendant adoption changed predecessor checkpoint history")
	}
}

func TestPrivateTransactionRetryRefusesChangedOrUnacknowledgedSource(t *testing.T) {
	for _, arm := range []string{"source", "profile", "selection", "unacknowledged"} {
		t.Run(arm, func(t *testing.T) {
			store, prior, retry, checkpoint := sourceRetryRecords(t, arm)
			_, available, problem := store.RetriedSourceCheckpoints(retry.ID)
			fatal(t, problem)
			if len(available) != 0 {
				t.Fatal("changed or merely observed source was offered as reusable custody")
			}
			if problem := store.AdoptRetriedSourceCheckpoint(retry.ID, prior.ID, checkpoint); problem == nil {
				t.Fatal("source adoption accepted changed or unacknowledged computation")
			}
			current, problem := store.ModelSourceProgress(retry.ID)
			fatal(t, problem)
			if len(current) != 0 {
				t.Fatal("refused adoption left a progress or custody row")
			}
		})
	}
}

func sourceRetryRecords(t *testing.T, arm string) (*records.Store, records.Request, records.Request, records.ModelSourceCheckpoint) {
	t.Helper()
	store, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
	fatal(t, problem)
	t.Cleanup(func() { store.Close() })
	intent := &records.ModelTransferIntent{
		Kind: "model-upload", Destination: "proof/original",
		Source:          "hf://proof/source@" + strings.Repeat("4", 40),
		SourceSelection: "sha256:" + strings.Repeat("2", 64),
		SourceProfiles:  map[string]string{"shared": "hf/minimax-h3/shared-bf16/1"},
		SourceFiles:     []records.ModelTransferSourceFile{{Member: "model.safetensors.index.json", SHA256: strings.Repeat("1", 64), Length: 2, Header: []byte("{}")}},
		Outputs:         []records.ModelTransferOutput{{Name: "model"}},
	}
	prior, _, problem := store.Submit(records.Request{
		ID: "req-source-original", IdemKey: "source-original", BodyDigest: "sha256:" + strings.Repeat("a", 64),
		Package: "local/original", Entrypoint: "produce", Kind: "job", RetainWork: true,
		Payload: []byte("{}"), Outputs: "[]", WeightsOutputs: "[]", ModelTransfer: intent,
	})
	fatal(t, problem)
	fatal(t, store.BeginModelTransferMaterialization(prior.ID))
	checkpoint := records.ModelSourceCheckpoint{Slot: "shared", HeadID: "sha256:" + strings.Repeat("3", 64),
		HeadLength: 500, PlanDigest: "sha256:" + strings.Repeat("4", 64), Index: 7, Bytes: 4096}
	fatal(t, store.ObserveModelSourceCheckpoints(prior.ID, intent.SourceSelection, "old-boot", []records.ModelSourceCheckpoint{checkpoint}))
	if arm != "unacknowledged" {
		fatal(t, store.AcknowledgeModelSourceCheckpoint(prior.ID, intent.SourceSelection, "old-boot", "", checkpoint))
	}
	_, problem = store.BlockRetainedWork(prior.ID, "author_exception", "step B failed")
	fatal(t, problem)
	newIntent := *intent
	newIntent.Destination = "proof/fixed"
	switch arm {
	case "source":
		newIntent.Source = "hf://proof/source@" + strings.Repeat("5", 40)
	case "profile":
		newIntent.SourceProfiles = map[string]string{"shared": "hf/minimax-h3/dit-bf16/1"}
	case "selection":
		newIntent.SourceSelection = "sha256:" + strings.Repeat("6", 64)
	}
	retry, _, problem := store.Submit(records.Request{
		ID: "req-source-fixed", IdemKey: "source-fixed", BodyDigest: "sha256:" + strings.Repeat("b", 64),
		Package: "local/fixed", Entrypoint: "produce-fixed", Kind: "job", RetainWork: true, RetryOf: prior.ID,
		Payload: []byte("{}"), Outputs: "[]", WeightsOutputs: "[]", ModelTransfer: &newIntent,
	})
	fatal(t, problem)
	return store, prior, retry, checkpoint
}
