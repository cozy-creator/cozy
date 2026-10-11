package producttest

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/install"
	"github.com/cozy-creator/cozy/internal/records"
)

// A run's submission leases the install it resolved until the daemon has recorded the
// request that references it. An install that supersedes it meanwhile (an explicit
// reinstall, or the editable watcher's refresh) must not reclaim it in that window: the
// daemon would answer "request install … is no longer present" for work already sent.
// The lease is the submitter's own process claim, so here the test holds it exactly as
// a submitting process does while the real CLI installs over it.
func TestLeasedInstallSurvivesASupersedingInstall(t *testing.T) {
	root := t.TempDir()
	project := weightlessProject(t)
	if code, out := runCozy(t, root, "package", "install", project); code != 0 {
		t.Fatalf("install [exit %d]: %s", code, out)
	}
	leased := activePackageInstall(t, root)
	layout, problem := home.Open(root)
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	lease, problem := install.LeaseInstall(layout, store, leased.ID)
	fatal(t, problem)

	module := filepath.Join(project, "weightless.py")
	source, err := os.ReadFile(module)
	must(t, err)
	must(t, os.WriteFile(module, append(source, []byte("\n# a superseding edit\n")...), 0o644))
	if code, out := runCozy(t, root, "package", "install", project); code != 0 {
		t.Fatalf("superseding install [exit %d]: %s", code, out)
	}
	if active := activePackageInstall(t, root); active.ID == leased.ID {
		t.Fatal("the edited source did not produce a superseding install")
	}
	kept, problem := store.Install(leased.ID)
	fatal(t, problem)
	if kept == nil {
		t.Fatal("a superseding install reclaimed the install a submission holds")
	}
	if _, err := os.Stat(kept.Dir); err != nil {
		t.Fatalf("the leased install's tree is gone: %v", err)
	}

	// Once the submission ends, the superseded install is ordinary garbage again.
	lease.Release()
	if _, problem := install.Reclaim(layout, store, leased.ID); problem != nil {
		t.Fatal(problem.Message)
	}
	if gone, problem := store.Install(leased.ID); problem != nil || gone != nil {
		t.Fatalf("the released install was not reclaimed: %+v %v", gone, problem)
	}
}
