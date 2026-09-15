package producttest

import (
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

func TestMachinePackageUploadProgressBelongsToItsSubmittingRoot(t *testing.T) {
	store, first, _ := machineObserverFixture(t)
	second, _, problem := store.Submit(records.Request{ID: "job-second", IdemKey: "second", Package: first.Package, Entrypoint: "main", Kind: "job", Payload: []byte(`{}`), BodyDigest: childDigest("1"), MachineExecutionObserver: true})
	fatal(t, problem)
	boot, revision := "retained-worker-boot", childDigest("7")
	progress := map[string]any{"worker_boot_id": boot, "revision": revision}
	fatal(t, store.AppendEvent(first.ID, "machine.package_uploaded", 0, progress))
	for _, row := range []struct {
		request, boot, revision string
		expected                bool
	}{
		{first.ID, boot, revision, true},
		{second.ID, boot, revision, false},
		{first.ID, "another-boot", revision, false},
		{first.ID, boot, childDigest("8"), false},
	} {
		_, count, problem := store.MachinePackageTransfer(row.request, row.boot, row.revision)
		got := count > 0
		fatal(t, problem)
		if got != row.expected {
			t.Fatalf("upload scope %+v: %v", row, got)
		}
	}
	// A concurrent second root must not hide the first root's own resumable upload.
	fatal(t, store.AppendEvent(second.ID, "machine.package_uploaded", 0, progress))
	for _, request := range []string{first.ID, second.ID} {
		_, count, problem := store.MachinePackageTransfer(request, boot, revision)
		got := count > 0
		fatal(t, problem)
		if !got {
			t.Fatal("same-root upload retry lost its progress")
		}
	}
}

func TestMachinePackageRepairTransferPreservesRootAndBootScope(t *testing.T) {
	store, first, _ := machineObserverFixture(t)
	boot, revision := "same-boot", childDigest("7")
	operation, count, problem := store.MachinePackageTransfer(first.ID, boot, revision)
	fatal(t, problem)
	if operation != "" || count != 0 {
		t.Fatal("new request has transfer progress")
	}
	fatal(t, store.AppendEvent(first.ID, "machine.package_uploaded", 0, map[string]any{"worker_boot_id": boot, "revision": revision}))
	operation, count, problem = store.MachinePackageTransfer(first.ID, boot, revision)
	fatal(t, problem)
	if operation != "" || count != 1 {
		t.Fatal("existing transfer without operation metadata was lost")
	}
	repair := first.ID + ".repair-1"
	fatal(t, store.AppendEvent(first.ID, "machine.package_uploaded", 0, map[string]any{"worker_boot_id": boot, "revision": revision, "operation_id": repair}))
	operation, count, problem = store.MachinePackageTransfer(first.ID, boot, revision)
	fatal(t, problem)
	if operation != repair || count != 2 {
		t.Fatal("repair progress did not remain separate from initial upload")
	}
	for _, scope := range [][3]string{{"another-root", boot, revision}, {first.ID, "another-boot", revision}, {first.ID, boot, childDigest("8")}} {
		operation, count, problem = store.MachinePackageTransfer(scope[0], scope[1], scope[2])
		fatal(t, problem)
		if operation != "" || count != 0 {
			t.Fatalf("another scope inherited transfer progress: %v", scope)
		}
	}
}
