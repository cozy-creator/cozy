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
		got, problem := store.MachinePackageUploaded(row.request, row.boot, row.revision)
		fatal(t, problem)
		if got != row.expected {
			t.Fatalf("upload scope %+v: %v", row, got)
		}
	}
	// A concurrent second root must not hide the first root's own resumable upload.
	fatal(t, store.AppendEvent(second.ID, "machine.package_uploaded", 0, progress))
	for _, request := range []string{first.ID, second.ID} {
		got, problem := store.MachinePackageUploaded(request, boot, revision)
		fatal(t, problem)
		if !got {
			t.Fatal("same-root upload retry lost its progress")
		}
	}
}
