package producttest

import (
	"archive/zip"
	"bytes"
	"crypto/sha256"
	"encoding/base64"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/packagepublish"
	"github.com/pelletier/go-toml/v2"
)

func TestTransitiveCallableOverlayBecomesAnExactCallerDependency(t *testing.T) {
	root := t.TempDir()
	index, overlays := filepath.Join(root, "index"), filepath.Join(root, "overlays")
	for _, path := range []string{index, overlays} {
		must(t, os.MkdirAll(path, 0700))
	}
	makeWheel := func(dir, name, dependency, value string) string {
		t.Helper()
		module := strings.ReplaceAll(name, "-", "_")
		info := module + "-0.1.0.dist-info/"
		metadata := "Metadata-Version: 2.3\nName: " + name + "\nVersion: 0.1.0\n"
		if dependency != "" {
			metadata += "Requires-Dist: " + dependency + "==0.1.0\n"
		}
		members := map[string]string{
			module + ".py":    "VALUE = " + fmt.Sprintf("%q", value) + "\n",
			info + "METADATA": metadata + "\n",
			info + "WHEEL":    "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
		}
		names := make([]string, 0, len(members))
		for name := range members {
			names = append(names, name)
		}
		sort.Strings(names)
		var record strings.Builder
		for _, name := range names {
			digest := sha256.Sum256([]byte(members[name]))
			fmt.Fprintf(&record, "%s,sha256=%s,%d\n", name, base64.RawURLEncoding.EncodeToString(digest[:]), len(members[name]))
		}
		record.WriteString(info + "RECORD,,\n")
		members[info+"RECORD"] = record.String()
		names = append(names, info+"RECORD")
		path := filepath.Join(dir, module+"-0.1.0-py3-none-any.whl")
		file, err := os.Create(path)
		must(t, err)
		writer := zip.NewWriter(file)
		for _, name := range names {
			member, err := writer.Create(name)
			must(t, err)
			_, err = member.Write([]byte(members[name]))
			must(t, err)
		}
		must(t, writer.Close())
		must(t, file.Close())
		return path
	}
	makeWheel(index, "captured-leaf", "", "original")
	wrapper := makeWheel(index, "captured-wrapper", "captured-leaf", "wrapper")
	replacement := makeWheel(overlays, "captured-leaf", "", "overlay")
	project := filepath.Join(root, "project")
	must(t, os.Mkdir(project, 0700))
	metadata := fmt.Sprintf(`[project]
name = "captured-parent"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = ["captured-wrapper>=0.1.0"]
[tool.uv]
no-index = true
find-links = [%q]
[tool.uv.sources]
captured-wrapper = {path = %q}
captured-leaf = {path = %q}
`, index, wrapper, replacement)
	must(t, os.WriteFile(filepath.Join(project, "pyproject.toml"), []byte(metadata), 0600))
	must(t, os.WriteFile(filepath.Join(project, "package.toml"), []byte("[application]\nobject='root_app:app'\n"), 0600))
	must(t, os.WriteFile(filepath.Join(project, "root_app.py"), []byte("app = None\n"), 0600))
	run := func(dir string, args ...string) []byte {
		t.Helper()
		command := exec.CommandContext(t.Context(), "uv", args...)
		command.Dir = dir
		output, err := command.CombinedOutput()
		if err != nil {
			t.Fatalf("uv %v: %v\n%s", args, err, output)
		}
		return output
	}
	readFlavor := func(dir string) string {
		t.Helper()
		command := exec.CommandContext(t.Context(), filepath.Join(dir, ".venv", "bin", "python"), "-c", "import captured_leaf; print(captured_leaf.VALUE)")
		output, err := command.CombinedOutput()
		if err != nil {
			t.Fatalf("read installed wheel: %v\n%s", err, output)
		}
		return strings.TrimSpace(string(output))
	}
	run(project, "sync", "--no-dev", "--no-default-groups", "--no-install-project", "--python", "3.12")
	if got := readFlavor(project); got != "original" {
		t.Fatalf("baseline no longer reproduces ignored transitive source: %s", got)
	}
	lock, err := os.ReadFile(filepath.Join(project, "uv.lock"))
	must(t, err)
	pack, problem := packagepublish.PrepareLocalFrom(project)
	fatal(t, problem)
	defer pack.Close()
	copied, problem := packagepublish.WithChildInterfaces(t.Context(), pack, map[string]string{"captured-leaf": replacement}, nil)
	fatal(t, problem)
	defer copied.Close()
	run(copied.Tree, "sync", "--frozen", "--no-dev", "--no-default-groups", "--no-install-project", "--python", "3.12")
	if got := readFlavor(copied.Tree); got != "overlay" {
		t.Fatalf("captured caller ignored its advertised overlay: %s", got)
	}
	selected, problem := packagepublish.LocalDependencySelections(copied.Tree)
	fatal(t, problem)
	if selected["captured-leaf"].Path != replacement {
		t.Fatalf("staging cannot capture the selected overlay: %+v", selected)
	}
	var document struct {
		Project struct {
			Dependencies []string `toml:"dependencies"`
		} `toml:"project"`
	}
	raw, err := os.ReadFile(filepath.Join(copied.Tree, "pyproject.toml"))
	must(t, err)
	must(t, toml.Unmarshal(raw, &document))
	if strings.Join(document.Project.Dependencies, ",") != "captured-wrapper>=0.1.0,captured-leaf" {
		t.Fatalf("captured overlay changed unrelated requirements: %+v", document.Project.Dependencies)
	}
	current, err := os.ReadFile(filepath.Join(project, "pyproject.toml"))
	must(t, err)
	currentLock, err := os.ReadFile(filepath.Join(project, "uv.lock"))
	must(t, err)
	if string(current) != metadata || !bytes.Equal(currentLock, lock) {
		t.Fatal("interface capture changed the author's project or lock")
	}
	if _, problem := packagepublish.WithChildInterfaces(t.Context(), pack, map[string]string{"other": replacement}, nil); problem == nil {
		t.Fatal("replacement with a different distribution was accepted")
	}
}
