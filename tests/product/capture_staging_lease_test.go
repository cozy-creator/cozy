package producttest

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/packagepublish"
)

func TestProjectMetadataRefusalNamesTheLostPreparedFile(t *testing.T) {
	project, _, _ := directoryProofProject(t)
	must(t, os.WriteFile(filepath.Join(project, "uv.lock"), []byte("version=1\n"), 0600))
	prepared, problem := packagepublish.PrepareLocalFrom(project)
	fatal(t, problem)
	defer prepared.Close()
	path := filepath.Join(project, "pyproject.toml")
	must(t, os.Remove(path))
	_, readError := os.ReadFile(path)
	problem = prepared.Build(context.Background())
	if problem == nil || problem.ErrName() != "project_metadata_unreadable" || !strings.Contains(problem.Message, path) || !strings.Contains(problem.Message, readError.Error()) {
		t.Fatalf("lost source metadata refusal omits its path/cause: %v", problem)
	}
}
