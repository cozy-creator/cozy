package cli

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/records"
)

func TestLocalLockHubRefusalPrecedesMachineSourceAndRun(t *testing.T) {
	root := t.TempDir()
	layout, problem := home.Open(root)
	if problem != nil {
		t.Fatal(problem)
	}
	store, problem := records.Open(layout.DB)
	if problem != nil {
		t.Fatal(problem)
	}
	defer store.Close()
	project := filepath.Join(root, "project")
	if err := os.Mkdir(project, 0o700); err != nil {
		t.Fatal(err)
	}
	for name, data := range map[string]string{
		"pyproject.toml": "[project]\nname='lock-proof'\nversion='1.0.0'\nrequires-python='>=3.12'\n",
		"package.toml":   "[application]\nobject='proof:app'\n",
		"proof.py":       "from cozy_runtime.author import App\napp = App()\n",
		"uv.lock":        "version=1\n[[package]]\nname='qwen-image-2'\nversion='0.5.0'\nsource={registry='https://tensorhub.com/v1/index/fidika/simple/'}\n",
	} {
		if err := os.WriteFile(filepath.Join(project, name), []byte(data), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	if problem := store.RecordInstall(records.PackageInstall{ID: "lock-proof", Package: "local/lock-proof", Major: 1,
		Version: "1.0.0", Hub: "https://tensorhub.com", SourceKind: "local", ProjectDir: project}); problem != nil {
		t.Fatal(problem)
	}
	m := machineRuns{resolver: NewResolver(store, config.Config{Home: root})}
	// No transport exists: even inspecting it would panic. The run's package/model
	// Hub must reject the lock before source upload or machine submission.
	manifest, sent, problem := m.localSourceV1(t.Context(), nil, "lock-proof", "http://127.0.0.1:8819")
	if problem == nil || problem.Name != "local_package_lock_hub_mismatch" || manifest != "" || sent {
		t.Fatalf("foreign lock reached the machine: manifest=%q sent=%v error=%v", manifest, sent, problem)
	}
}
