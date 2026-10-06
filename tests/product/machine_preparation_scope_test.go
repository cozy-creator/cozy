package producttest

import (
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

func TestMachinePackageUploadProgressBelongsToItsSubmittingRoot(t *testing.T) {
	store, first := pendingNativeFixture(t)
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
		progress, problem := store.MachinePackageTransfer(row.request, row.boot, row.revision)
		got := progress.Uploaded
		fatal(t, problem)
		if got != row.expected {
			t.Fatalf("upload scope %+v: %v", row, got)
		}
	}
	// A concurrent second root must not hide the first root's own resumable upload.
	fatal(t, store.AppendEvent(second.ID, "machine.package_uploaded", 0, progress))
	for _, request := range []string{first.ID, second.ID} {
		progress, problem := store.MachinePackageTransfer(request, boot, revision)
		got := progress.Uploaded
		fatal(t, problem)
		if !got {
			t.Fatal("same-root upload retry lost its progress")
		}
	}
}

func TestMachinePackageRepairTransferPreservesRootAndBootScope(t *testing.T) {
	store, first := pendingNativeFixture(t)
	boot, revision := "same-boot", childDigest("7")
	progress, problem := store.MachinePackageTransfer(first.ID, boot, revision)
	fatal(t, problem)
	if progress.Operation != "" || progress.Completed != 0 || progress.Uploaded {
		t.Fatal("new request has transfer progress")
	}
	start := map[string]any{"worker_boot_id": boot, "revision": revision, "operation_id": first.ID}
	fatal(t, store.AppendEvent(first.ID, "machine.package_upload_started", 0, start))
	progress, problem = store.MachinePackageTransfer(first.ID, boot, revision)
	fatal(t, problem)
	if progress.Operation != first.ID || progress.Completed != 0 || progress.Uploaded {
		t.Fatal("initial partial upload cannot resume")
	}
	fatal(t, store.AppendEvent(first.ID, "machine.package_uploaded", 0, start))
	progress, problem = store.MachinePackageTransfer(first.ID, boot, revision)
	fatal(t, problem)
	if progress.Operation != first.ID || progress.Completed != 1 || !progress.Uploaded {
		t.Fatal("completed transfer was lost")
	}
	repair := first.ID + ".repair-1"
	next := map[string]any{"worker_boot_id": boot, "revision": revision, "operation_id": repair}
	fatal(t, store.AppendEvent(first.ID, "machine.package_upload_started", 0, next))
	progress, problem = store.MachinePackageTransfer(first.ID, boot, revision)
	fatal(t, problem)
	if progress.Operation != repair || progress.Completed != 1 || progress.Uploaded {
		t.Fatal("partial repair lost the previous completed transfer")
	}
	fatal(t, store.AppendEvent(first.ID, "machine.package_uploaded", 0, next))
	progress, problem = store.MachinePackageTransfer(first.ID, boot, revision)
	fatal(t, problem)
	if progress.Operation != repair || progress.Completed != 2 || !progress.Uploaded {
		t.Fatal("repair completion changed its operation")
	}
	for _, scope := range [][3]string{{"another-root", boot, revision}, {first.ID, "another-boot", revision}, {first.ID, boot, childDigest("8")}} {
		progress, problem = store.MachinePackageTransfer(scope[0], scope[1], scope[2])
		fatal(t, problem)
		if progress.Operation != "" || progress.Completed != 0 || progress.Uploaded {
			t.Fatalf("another scope inherited transfer progress: %v", scope)
		}
	}
}
