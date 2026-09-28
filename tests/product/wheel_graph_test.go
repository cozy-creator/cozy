package producttest

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/packagepublish"
)

// Exercise uv's actual locked graph, including inactive extra roots, rather than
// reproducing its preview JSON. Captured installed versions remain authoritative.
func TestWheelGraphSelectsCalleeExtrasWithoutCallerOrUnrelatedDependencies(t *testing.T) {
	root := t.TempDir()
	projects := []struct{ name, version, details string }{
		{"base", "3", ""}, {"scoring", "4", "dependencies = [\"base\"]\n[tool.uv.sources]\nbase = {path = \"../base\"}\n"},
		{"unrelated", "5", ""}, {"unused", "6", ""},
		{"library", "2", "dependencies = [\"base\"]\n[project.optional-dependencies]\njobs = [\"scoring\"]\n[tool.uv.sources]\nbase = {path = \"../base\"}\nscoring = {path = \"../scoring\"}\n"},
		{"caller", "1", "dependencies = [\"library[jobs]\", \"unrelated\"]\n[project.optional-dependencies]\ndev = [\"unused\"]\n[tool.uv.sources]\nlibrary = {path = \"../library\"}\nunrelated = {path = \"../unrelated\"}\nunused = {path = \"../unused\"}\n"},
	}
	for _, project := range projects {
		dir := filepath.Join(root, project.name)
		must(t, os.MkdirAll(dir, 0700))
		metadata := fmt.Sprintf("[project]\nname = %q\nversion = %q\nrequires-python = \">=3.12,<3.13\"\n%s", project.name, project.version, project.details)
		must(t, os.WriteFile(filepath.Join(dir, "pyproject.toml"), []byte(metadata), 0600))
	}
	caller := filepath.Join(root, "caller")
	command := exec.Command("uv", "lock", "--directory", caller, "--python", "3.12")
	if out, err := command.CombinedOutput(); err != nil {
		t.Fatalf("uv lock: %v\n%s", err, out)
	}
	installed := "base==3\ncaller==1\nlibrary==2\nscoring==4\nunrelated==5\nunused==6"
	graph, problem := packagepublish.WheelClosures(context.Background(), caller, "python3.12", installed, "caller", "")
	fatal(t, problem)
	want := map[string]string{"library": "2", "base": "3", "scoring": "4"}
	if !reflect.DeepEqual(graph["library"], want) || graph["unused"] != nil || graph["caller"] != nil {
		t.Fatalf("callee closure includes inactive or caller dependencies: %#v", graph)
	}
	again, problem := packagepublish.WheelClosures(context.Background(), caller, "python3.12", strings.TrimSuffix(installed, "\nunused==6"), "caller", "")
	fatal(t, problem)
	if !reflect.DeepEqual(again["library"], want) {
		t.Fatalf("ambient install changed callee: %#v", again)
	}
	for _, observed := range []string{strings.Replace(installed, "library==2", "library==9", 1), strings.Replace(installed, "scoring==4\n", "", 1)} {
		if _, problem := packagepublish.WheelClosures(context.Background(), caller, "python3.12", observed, "caller", ""); problem == nil {
			t.Fatal("installed drift was accepted")
		}
	}
	if _, problem := packagepublish.WheelClosures(context.Background(), caller, "python3.12", installed, "caller", "missing"); problem == nil {
		t.Fatal("missing caller extra was accepted")
	}
}
