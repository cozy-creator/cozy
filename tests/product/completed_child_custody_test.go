package producttest

import (
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

func TestCancelledAncestorWaitsForNestedCompletedCustody(t *testing.T) {
	store := successReleaseStore(t)
	root := offerChildParent(t, store, recordPrivateTransaction(t, store, "nested-root", ""))
	middle := offerChildParent(t, store, operationHistory(t, store, "nested-middle", root))
	leaf := operationHistory(t, store, "nested-leaf", middle, true)
	closeChild(t, store, leaf, "SUCCEEDED", "succeeded")
	closeChild(t, store, middle, "SUCCEEDED", "succeeded")
	closeChild(t, store, root, "FAILED", "blocked")
	fatal(t, store.RequestRetainedCancellation(root.ID, "test"))
	ready, problem := store.ReleaseCompletedChildWork(root.ID, middle.ID)
	fatal(t, problem)
	if !ready {
		t.Fatal("drained completed child was not released")
	}
	ready, problem = store.ReleaseRetainedWork(root.ID)
	fatal(t, problem)
	if ready {
		t.Fatal("parent completed while its grandchild retained work")
	}
	ready, problem = store.ReleaseCompletedChildWork(middle.ID, leaf.ID)
	fatal(t, problem)
	if ready {
		t.Fatal("success alone authorized a descendant's custody release")
	}
	ready, problem = store.ReleaseCompletedChildWork(root.ID, leaf.ID)
	fatal(t, problem)
	if !ready {
		t.Fatal("cancelled ancestor could not release a completed grandchild")
	}
	ready, problem = store.ReleaseRetainedWork(root.ID)
	fatal(t, problem)
	if !ready {
		t.Fatal("fully drained cancellation did not advance")
	}
}

func TestCancelledAncestorAuthorizesOnlyItsSuccessfulChildFinalization(t *testing.T) {
	store := successReleaseStore(t)
	root := offerChildParent(t, store, recordPrivateTransaction(t, store, "finalization-root", ""))
	child := operationHistory(t, store, "finalization-child", root, true)
	closeChild(t, store, child, "SUCCEEDED", "succeeded")
	otherRoot := offerChildParent(t, store, recordPrivateTransaction(t, store, "other-root", ""))
	other := operationHistory(t, store, "other-child", otherRoot, true)
	closeChild(t, store, other, "SUCCEEDED", "succeeded")
	closeChild(t, store, root, "FAILED", "blocked")
	intent := func(id string) records.WeightsFinalization {
		return records.WeightsFinalization{RequestID: id, Attempt: 1, InstanceID: "private-worker",
			OwnerScope: "owner", InvocationDigest: childDigest("1"), OutputSlot: "weights"}
	}
	fatal(t, store.RecordSuccessfulFinalization(root.ID, intent(child.ID)))
	pending, problem := store.PendingWeightsFinalizations(child.ID, 1)
	fatal(t, problem)
	if len(pending) != 0 {
		t.Fatal("native finalization preceded ancestor cancellation authority")
	}
	fatal(t, store.RequestRetainedCancellation(root.ID, "test"))
	fatal(t, store.RecordSuccessfulFinalization(root.ID, intent(child.ID)))
	fatal(t, store.RecordSuccessfulFinalization(root.ID, intent(other.ID)))
	pending, problem = store.PendingWeightsFinalizations(child.ID, 1)
	fatal(t, problem)
	if len(pending) != 1 {
		t.Fatal("successful child's native finalization was not recorded")
	}
	foreign, problem := store.PendingWeightsFinalizations(other.ID, 1)
	fatal(t, problem)
	if len(foreign) != 0 {
		t.Fatal("ancestor cancellation escaped its family")
	}
	ready, problem := store.ReleaseCompletedChildWork(root.ID, child.ID)
	fatal(t, problem)
	if ready {
		t.Fatal("completed custody cleared before native finalization confirmation")
	}
}
