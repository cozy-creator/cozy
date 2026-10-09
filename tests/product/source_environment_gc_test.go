package producttest

import (
	"testing"

	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/records"
)

func TestUnreferencedRetainsNewestLocalEnvironment(t *testing.T) {
	layout, problem := home.Open(t.TempDir())
	if problem != nil {
		t.Fatal(problem)
	}
	store, problem := records.Open(layout.DB)
	if problem != nil {
		t.Fatal(problem)
	}
	defer store.Close()
	install := func(id, createdVersion string) records.PackageInstall {
		return records.PackageInstall{ID: id, Package: "local/example", Major: 1, Version: "1.0.0", SourceKind: "local", SourceRef: "file:///example/" + createdVersion, Dir: layout.InstallDir(id), ProjectDir: "/example/" + createdVersion}
	}
	first := install("first", "first")
	if _, problem := store.Activate(first); problem != nil {
		t.Fatal(problem)
	}
	if problem := store.Unpin(first.PinHub(), first.Package, first.Major); problem != nil {
		t.Fatal(problem)
	}
	second := install("second", "second")
	if _, problem := store.Activate(second); problem != nil {
		t.Fatal(problem)
	}
	if problem := store.Unpin(second.PinHub(), second.Package, second.Major); problem != nil {
		t.Fatal(problem)
	}
	rows, problem := store.Unreferenced()
	if problem != nil {
		t.Fatal(problem)
	}
	if len(rows) != 1 || rows[0].ID != first.ID {
		t.Fatalf("unreferenced local environments = %+v, want only the older candidate", rows)
	}
	if forgotten, problem := store.ForgetIfUnreferenced(second.ID); problem != nil || forgotten {
		t.Fatalf("newest local environment was reclaimable: forgotten=%v problem=%v", forgotten, problem)
	}
	if forgotten, problem := store.ForgetIfUnreferenced(first.ID); problem != nil || !forgotten {
		t.Fatalf("older local environment was not reclaimable: forgotten=%v problem=%v", forgotten, problem)
	}
}
