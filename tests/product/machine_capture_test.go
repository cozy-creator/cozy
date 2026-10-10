package producttest

import (
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

// A callable binding names one callee implementation and entrypoint for its caller's install,
// for good: a changed row is refused.
func TestChildBindingsAreImmutable(t *testing.T) {
	store, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	for _, id := range []string{"install-20", "install-10"} {
		fatal(t, store.RecordInstall(records.PackageInstall{ID: id, Package: "local/" + id, Version: "1.0.0"}))
	}
	binding := records.ChildBinding{ParentInstallID: "install-20", ChildInstallID: "install-10", Module: "ops", Export: "prepare", Entrypoint: "prepare"}
	fatal(t, store.RecordChildBindings([]records.ChildBinding{binding}))
	for _, field := range []string{"callee", "entrypoint"} {
		changed := binding
		if field == "callee" {
			changed.ChildInstallID = "install-20"
		} else {
			changed.Entrypoint = "changed"
		}
		if problem := store.RecordChildBindings([]records.ChildBinding{changed}); problem == nil {
			t.Fatal("changed immutable binding accepted", field)
		}
	}
}
