package producttest

import (
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

// A private retry reuses same-machine observed state. Its independent native
// head and provenance release the ancestor's hold; no remote ACK is fabricated.
func TestPrivateTransactionRetryRecordsIndependentLocalSource(t *testing.T) {
	store, prior, retry, old := localSourceRetryRecords(t, "unchanged")
	parent, available, problem := store.RetriedSourceCheckpoints(retry.ID)
	fatal(t, problem)
	if parent != prior.ID || len(available) != 1 || available[0].Observed != old || available[0].Acknowledged != nil {
		t.Fatalf("private local source was not offered as local observation: %s %+v", parent, available)
	}
	before, problem := store.ModelSourceProgress(prior.ID)
	fatal(t, problem)
	assertPending := func(want bool) {
		t.Helper()
		pending, problem := store.RetriedSourceCustodyPending(prior.ID)
		fatal(t, problem)
		if pending != want {
			t.Fatalf("predecessor custody pending=%t, want %t", pending, want)
		}
	}
	assertPending(true)
	fresh := old
	fresh.HeadID = "sha256:" + strings.Repeat("6", 64)
	// Native adoption compacts checkpoint metadata. Cross-chain counters are not
	// a progress ordering, and may shrink while all tensor parts remain retained.
	fresh.Index, fresh.Bytes = 0, old.Bytes-100
	fatal(t, store.ObserveModelSourceCheckpoints(retry.ID, retry.ModelTransfer.SourceSelection, "retained-boot", []records.ModelCheckpoint{fresh}))
	assertPending(true) // crash between observation and provenance cannot free the old root
	changed := fresh
	changed.HeadID = "sha256:" + strings.Repeat("9", 64)
	for _, arm := range []struct {
		name, parent, boot string
		old, fresh         records.ModelCheckpoint
	}{
		{"unrelated predecessor", "unrelated", "retained-boot", old, fresh},
		{"different worker boot", prior.ID, "replacement-boot", old, fresh},
		{"unobserved new head", prior.ID, "retained-boot", old, changed},
		{"borrowed old head", prior.ID, "retained-boot", old, old},
	} {
		if problem := store.RecordRetriedSourceAdoption(retry.ID, arm.parent, arm.boot,
			[]records.ModelCheckpoint{arm.old}, []records.ModelCheckpoint{arm.fresh}); problem == nil {
			t.Fatalf("adoption accepted %s", arm.name)
		}
		assertPending(true)
	}
	fatal(t, store.RecordRetriedSourceAdoption(retry.ID, prior.ID, "retained-boot", []records.ModelCheckpoint{old}, []records.ModelCheckpoint{fresh}))
	fatal(t, store.RecordRetriedSourceAdoption(retry.ID, prior.ID, "retained-boot", []records.ModelCheckpoint{old}, []records.ModelCheckpoint{fresh}))
	assertPending(false)
	current, problem := store.ModelSourceProgress(retry.ID)
	fatal(t, problem)
	if len(current) != 1 || current[0].Observed != fresh || current[0].Acknowledged != nil {
		t.Fatalf("local adoption fabricated remote custody or changed the native result: %+v", current)
	}
	after, problem := store.ModelSourceProgress(prior.ID)
	fatal(t, problem)
	if !reflect.DeepEqual(before, after) {
		t.Fatal("descendant adoption changed predecessor checkpoint history")
	}
	// Reconnection resumes the descendant's native head, including after its
	// predecessor's cancellation and after further progress in the new chain.
	fatal(t, store.RequestRetainedCancellation(prior.ID, "new request owns its source"))
	_, problem = store.ReleaseRetainedWork(prior.ID)
	fatal(t, problem)
	_, problem = store.CompleteRetainedCancellation(prior.ID)
	fatal(t, problem)
	fresh.HeadID = "sha256:" + strings.Repeat("7", 64)
	fresh.Index++
	fresh.Bytes += 100
	fatal(t, store.ObserveModelSourceCheckpoints(retry.ID, retry.ModelTransfer.SourceSelection, "retained-boot", []records.ModelCheckpoint{fresh}))
	_, offered, problem := store.RetriedSourceCheckpoints(retry.ID)
	fatal(t, problem)
	if len(offered) != 0 {
		t.Fatal("retry tried to re-adopt released predecessor instead of resuming its own head")
	}
}

func TestPrivateTransactionRetryDoesNotOfferChangedSource(t *testing.T) {
	for _, arm := range []string{"source", "profile", "selection"} {
		t.Run(arm, func(t *testing.T) {
			store, _, retry, _ := localSourceRetryRecords(t, arm)
			_, available, problem := store.RetriedSourceCheckpoints(retry.ID)
			fatal(t, problem)
			if len(available) != 0 {
				t.Fatal("changed source computation was offered as reusable local state")
			}
			current, problem := store.ModelSourceProgress(retry.ID)
			fatal(t, problem)
			if len(current) != 0 {
				t.Fatal("admission invented progress for changed source computation")
			}
		})
	}
}

func localSourceRetryRecords(t *testing.T, arm string) (*records.Store, records.Request, records.Request, records.ModelCheckpoint) {
	t.Helper()
	store, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
	fatal(t, problem)
	t.Cleanup(func() { store.Close() })
	intent := &records.ModelTransferIntent{
		Kind: "model-upload", Destination: "proof/original",
		Source: "hf://proof/source@" + strings.Repeat("4", 40), SourceSelection: "sha256:" + strings.Repeat("2", 64),
		SourceProfiles: map[string]string{"shared": "hf/minimax-h3/shared-bf16/1"},
		SourceFiles:    []records.ModelTransferSourceFile{{Member: "model.safetensors.index.json", SHA256: strings.Repeat("1", 64), Length: 2, Header: []byte("{}")}},
		Outputs:        []records.ModelTransferOutput{{Name: "model"}},
	}
	prior, _, problem := store.Submit(records.Request{
		ID: "req-source-original", IdemKey: "source-original", BodyDigest: "sha256:" + strings.Repeat("a", 64),
		Package: "local/original", Entrypoint: "produce", Kind: "job", RetainWork: true,
		Payload: []byte("{}"), Outputs: "[]", WeightsOutputs: "[]", ModelTransfer: intent,
	})
	fatal(t, problem)
	fatal(t, store.BeginModelTransferMaterialization(prior.ID))
	old := records.ModelCheckpoint{Slot: "shared", HeadID: "sha256:" + strings.Repeat("3", 64),
		HeadLength: 500, PlanDigest: "sha256:" + strings.Repeat("4", 64), Index: 7, Bytes: 4096}
	fatal(t, store.ObserveModelSourceCheckpoints(prior.ID, intent.SourceSelection, "retained-boot", []records.ModelCheckpoint{old}))
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
	fatal(t, store.BeginModelTransferMaterialization(retry.ID))
	return store, prior, retry, old
}
