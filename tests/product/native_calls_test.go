package producttest

import (
	"bytes"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/publication"
	"github.com/cozy-creator/cozy/internal/records"
)

// A newer Runtime's effect request carries members this cozy does not read: the effect is
// still resolved by the members it does.
func TestPublicationEffectsFromANewerRuntimeAreRead(t *testing.T) {
	for operation, raw := range map[string]string{
		"publish_release":   `{"destination":"alice/model","release":"v1","lanes":{},"visibility":"public"}`,
		"upload_checkpoint": `{"destination":"alice/model","artifact":{},"resumable":true}`,
	} {
		if ref, problem := publication.EffectDestination(operation, []byte(raw)); problem != nil || ref.String() != "alice/model" {
			t.Fatalf("%s from a newer Runtime: %v", operation, problem)
		}
	}
}

func TestAcceptedPublicationCallRefusesChangedInputsBeforeReexecution(t *testing.T) {
	for _, phase := range []string{"accepted", "frozen", "executing"} {
		t.Run(phase, func(t *testing.T) {
			store, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
			fatal(t, problem)
			defer store.Close()
			fatal(t, store.SpawnWorker(records.WorkerProcess{InstanceID: "private-worker", Package: "local/test", WorkerID: "worker", Devices: []string{"cpu"}}))
			parent := offerChildParent(t, store, recordPrivateTransaction(t, store, "effect-immutable", ""))
			body := []byte(`{"destination":"owner/model","expected_revision":1,"lanes":{"bf16":"checkpoint-a"},"release":"proof"}`)
			call := records.NativeCall{ID: "effect-immutable", ParentRequestID: parent.ID, Kind: "effect", Operation: "publish_release", IntentDigest: childDigest("4"), Request: body}
			_, _, problem = store.AcceptNativeCall(call, 1, childDigest("1"), "private-boot")
			fatal(t, problem)
			if phase != "accepted" {
				fatal(t, store.FreezeNativeCall(call.ID, body))
			}
			if phase == "executing" {
				fatal(t, store.StartNativeEffectWrite(call.ID))
			}
			before, problem := store.NativeCall(parent.ID, 0)
			fatal(t, problem)
			for _, change := range [][2]string{{"owner/model", "other/model"}, {"checkpoint-a", "checkpoint-b"}, {`"expected_revision":1`, `"expected_revision":2`}} {
				changed := call
				changed.Request = []byte(strings.Replace(string(body), change[0], change[1], 1))
				for _, digest := range []string{call.IntentDigest, childDigest("5")} {
					changed.IntentDigest = digest
					if _, _, problem := store.AcceptNativeCall(changed, 1, childDigest("1"), "private-boot"); problem == nil || problem.ErrName() != "child.intent_changed" {
						t.Fatalf("%s changed accepted publication inputs: %v", phase, problem)
					}
				}
			}
			after, problem := store.NativeCall(parent.ID, 0)
			fatal(t, problem)
			if before.State != after.State || !bytes.Equal(before.Request, after.Request) || !bytes.Equal(before.Frozen, after.Frozen) || before.IntentDigest != after.IntentDigest {
				t.Fatal("refused replay mutated the accepted publication")
			}
		})
	}
}

func TestNativeAndPackageCallsShareOneParentIndexAndFreezeBeforeEffects(t *testing.T) {
	store, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	fatal(t, store.SpawnWorker(records.WorkerProcess{InstanceID: "private-worker", Package: "local/test", WorkerID: "worker", Devices: []string{"cpu"}}))
	parent := offerChildParent(t, store, recordPrivateTransaction(t, store, "native-parent", ""))
	call := records.NativeCall{ID: "source-native", ParentRequestID: parent.ID, CallIndex: 0, Kind: "source", Operation: "download_civitai", IntentDigest: childDigest("4"), Request: []byte(`{"version":42}`)}
	first, fresh, problem := store.AcceptNativeCall(call, 1, childDigest("1"), "private-boot")
	fatal(t, problem)
	if !fresh || first.State != "accepted" {
		t.Fatal("native admission not recorded")
	}
	_, fresh, problem = store.AcceptNativeCall(call, 1, childDigest("1"), "private-boot")
	fatal(t, problem)
	if fresh {
		t.Fatal("replay created another call")
	}
	changed := call
	changed.Kind = "effect"
	if _, _, problem = store.AcceptNativeCall(changed, 1, childDigest("1"), "private-boot"); problem == nil {
		t.Fatal("source index accepted effect reinterpretation")
	}
	child := records.Request{ID: "req-collision", IdemKey: "collision", BodyDigest: childDigest("3"), Package: "local/operation", Entrypoint: "run", Kind: "job", Payload: []byte(`{}`), ParentRequestID: parent.ID, ParentCallIndex: 0, ChildIntentDigest: childDigest("4"), ChildTargetDigest: childDigest("5")}
	if _, _, problem = store.SubmitChild(child, 1, childDigest("1"), "private-boot", nil); problem == nil {
		t.Fatal("package call stole native index")
	}
	child.ParentCallIndex = 1
	_, _, problem = store.SubmitChild(child, 1, childDigest("1"), "private-boot", nil)
	fatal(t, problem)
	call.ID = "source-collision"
	call.CallIndex = 1
	if _, _, problem = store.AcceptNativeCall(call, 1, childDigest("1"), "private-boot"); problem == nil {
		t.Fatal("native call stole package index")
	}
	if store.StartNativeCall(first.ID) == nil {
		t.Fatal("execution started before frozen intent")
	}
	frozen := []byte(`{"checkpoint":"exact"}`)
	fatal(t, store.FreezeNativeCall(first.ID, frozen))
	fatal(t, store.StartNativeCall(first.ID))
	fatal(t, store.StartNativeCall(first.ID))
	if store.FreezeNativeCall(first.ID, []byte(`{"checkpoint":"changed"}`)) == nil {
		t.Fatal("executing intent changed")
	}
	result, receipt := []byte(`{"result":"done"}`), []byte(`{"native":"receipt"}`)
	fatal(t, store.CompleteNativeCallAt(first.ID, result, receipt, "private-worker", "private-boot"))
	fatal(t, store.StopNativeCall(first.ID, "canceled", "late_cancel"))
	observed, problem := store.NativeCall(parent.ID, 0)
	fatal(t, problem)
	if observed.State != "succeeded" || !bytes.Equal(observed.Result, result) || observed.InstanceID != "private-worker" || observed.WorkerBootID != "private-boot" {
		t.Fatalf("native terminal changed: %+v", observed)
	}
	if store.CompleteNativeCall(first.ID, []byte(`{"result":"other"}`), receipt) == nil {
		t.Fatal("contradictory completion accepted")
	}
}
