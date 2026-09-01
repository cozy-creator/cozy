package install

import "testing"

func TestRuntimePreparationDoesNotFeedRuntimeItsOwnWheel(t *testing.T) {
	if runtimePreparationDependency(PublishedWheel{Distribution: "cozy-runtime"}) {
		t.Fatal("the package Runtime was fed back to its own prepare-package command")
	}
	for _, distribution := range []string{"humanize", "msgspec", "tensorfs", "torch"} {
		if !runtimePreparationDependency(PublishedWheel{Distribution: distribution}) {
			t.Fatalf("ordinary dependency %s was hidden from package preparation", distribution)
		}
	}
}
