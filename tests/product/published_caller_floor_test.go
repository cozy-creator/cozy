package producttest

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/hostruntime"
	"github.com/cozy-creator/cozy/internal/packagepublish"
)

func TestPublishedCallerRuntimeFloorIsDeclaredAndLocked(t *testing.T) {
	for _, example := range []struct {
		name, requirement, locked string
		ok                        bool
	}{
		{"supported", "cozy-runtime[media]>=0.18.21", "0.18.21", true},
		{"newer-minor", "cozy-runtime==0.19.*", "0.19.1", true},
		{"old-declaration", "cozy-runtime>=0.18.20", "0.18.21", false},
		{"old-lock", "cozy-runtime>=0.18.21", "0.18.20", false},
		{"missing-lock", "cozy-runtime>=0.18.21", "", false},
		{"missing-floor", hostruntime.Distribution, "0.18.21", false},
	} {
		t.Run(example.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "pyproject.toml")
			err := os.WriteFile(path, []byte("[project]\nname='proof'\nversion='0.1.0'\nrequires-python='>=3.12'\ndependencies=['"+example.requirement+"']\n"), 0600)
			if err != nil {
				t.Fatal(err)
			}
			pack := &packagepublish.Package{Files: map[string]string{"pyproject.toml": path}}
			if example.locked != "" {
				pack.Registry = []packagepublish.RegistryRow{{Name: hostruntime.Distribution, Version: example.locked}}
			}
			problem := packagepublish.ValidateCallerRuntime(pack, "0.18.21")
			if (problem == nil) != example.ok {
				t.Fatalf("validation=%v, allowed=%v", problem, example.ok)
			}
			if problem != nil && problem.ErrName() != "package_publish.caller_runtime_floor" {
				t.Fatal(problem)
			}
		})
	}
}
