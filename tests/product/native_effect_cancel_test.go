package producttest

import (
	"bytes"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
)

func effectCancelFixture(t *testing.T) (*records.Store, records.NativeCall, string) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "creator.sqlite")
	store, problem := records.Open(path)
	fatal(t, problem)
	t.Cleanup(func() { store.Close() })
	fatal(t, store.SpawnWorker(records.WorkerProcess{InstanceID: "private-worker", Package: "local/test", WorkerID: "worker", Devices: []string{"cpu"}}))
	parent := offerChildParent(t, store, recordPrivateTransaction(t, store, "effect-cancel", ""))
	call, _, problem := store.AcceptNativeCall(records.NativeCall{ID: "effect-cancel", ParentRequestID: parent.ID, Kind: "effect", Operation: "publish_release", IntentDigest: childDigest("4"), Request: []byte(`{}`)}, 1, childDigest("1"), "private-boot")
	fatal(t, problem)
	return store, call, path
}

func cancelEffect(t *testing.T, store *records.Store, call records.NativeCall) *records.NativeCall {
	t.Helper()
	row, problem := store.RequestNativeEffectCancel(call.ParentRequestID, call.CallIndex, 1, childDigest("1"), "private-boot", call.IntentDigest)
	fatal(t, problem)
	if row == nil {
		t.Fatal("accepted effect disappeared")
	}
	parent, problem := store.RequestRow(call.ParentRequestID)
	fatal(t, problem)
	if parent.State != "dispatching" {
		t.Fatal("canceling one await stopped the parent")
	}
	return row
}

func TestAwaitedEffectCancellationFencesWritesAndPreservesPausedIntent(t *testing.T) {
	for _, phase := range []string{"accepted", "frozen", "executing", "paused"} {
		t.Run(phase, func(t *testing.T) {
			store, call, _ := effectCancelFixture(t)
			frozen := []byte(`{"baseline":1}`)
			if phase != "accepted" {
				fatal(t, store.FreezeNativeCall(call.ID, frozen))
			}
			if phase == "executing" {
				fatal(t, store.StartNativeEffectWrite(call.ID))
			}
			for _, bad := range []struct {
				attempt               int64
				spec, session, intent string
			}{
				{2, childDigest("1"), "private-boot", call.IntentDigest},
				{1, childDigest("2"), "private-boot", call.IntentDigest},
				{1, childDigest("1"), "another-boot", call.IntentDigest},
				{1, childDigest("1"), "private-boot", childDigest("5")},
				{1, strings.ToUpper(childDigest("1")), "private-boot", call.IntentDigest},
			} {
				if _, problem := store.RequestNativeEffectCancel(call.ParentRequestID, call.CallIndex, bad.attempt, bad.spec, bad.session, bad.intent); problem == nil {
					t.Fatal("unowned cancellation accepted")
				}
			}
			if phase == "paused" {
				_, problem := store.RequestPause(call.ParentRequestID, "preserve work")
				fatal(t, problem)
				row, problem := store.RequestNativeEffectCancel(call.ParentRequestID, 0, 1, childDigest("1"), "private-boot", call.IntentDigest)
				fatal(t, problem)
				if row.CancelRequested || row.State != "frozen" {
					t.Fatalf("pause canceled the effect: %+v", row)
				}
				if problem = store.StartNativeEffectWrite(call.ID); problem == nil || problem.Code != exit.Unavailable {
					t.Fatal("paused parent issued write", problem)
				}
				return
			}
			row := cancelEffect(t, store, call)
			want := "canceled"
			if phase == "executing" {
				want = "executing"
			}
			if !row.CancelRequested || row.State != want || (phase != "accepted" && !bytes.Equal(row.Frozen, frozen)) {
				t.Fatalf("cancellation erased frozen history: %+v", row)
			}
			if problem := store.StartNativeEffectWrite(call.ID); problem == nil || problem.Code != exit.Canceled {
				t.Fatal("canceled await issued write", problem)
			}
		})
	}
}
