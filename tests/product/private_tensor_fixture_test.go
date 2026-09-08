package producttest

import (
	"flag"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

var privateChildTensorFSWheel = flag.String("child-tensorfs-wheel", "", "exact TensorFS wheel for local and remote native operation proofs")

// Both transports execute these same author files. Only installed wheel sources
// vary for prerelease qualification; captured code and source packages are shared.
func copyPrivateTensorProject(t *testing.T, root, source string) string {
	t.Helper()
	if source == "" {
		source = filepath.Join("testdata", "private_tensor_operations")
	}
	project := filepath.Join(root, "client-project")
	version := runtimeFixtureVersion(t, *privateChildRuntimeWheel)
	var sources, scriptSources strings.Builder
	for _, dependency := range []struct{ name, wheel string }{
		{"cozy-runtime", *privateChildRuntimeWheel}, {"tensorfs", *privateChildTensorFSWheel}, //cozy:allow distribution source metadata, not executable invocation
	} {
		if dependency.wheel == "" {
			continue
		}
		path, err := filepath.Abs(dependency.wheel)
		must(t, err)
		line := dependency.name + " = {path = " + strconv.Quote(path) + "}\n"
		sources.WriteString(line)
		scriptSources.WriteString("# " + line)
	}
	for _, relative := range []string{"recipe.py", "source/pyproject.toml", "source/package.toml", "source/uv.lock", "source/tensor_source.py", "candidate/pyproject.toml", "candidate/package.toml", "candidate/uv.lock", "candidate/tensor_candidate.py"} {
		body, err := os.ReadFile(filepath.Join(source, relative))
		if os.IsNotExist(err) && strings.HasSuffix(relative, "uv.lock") {
			continue // private intake resolves into its owned copy
		}
		must(t, err)
		text := strings.ReplaceAll(string(body), "__RUNTIME_VERSION__", version)
		if relative == "recipe.py" {
			text = strings.Replace(text, "factor=2", "factor=0", 1)
			text = strings.Replace(text, "# [tool.uv.sources]\n", "# [tool.uv.sources]\n"+scriptSources.String(), 1)
		}
		if relative == "candidate/tensor_candidate.py" {
			text = strings.Replace(text, "value * factor + 1 for value", "value * factor for value", 1)
		}
		if strings.HasSuffix(relative, "pyproject.toml") {
			text = strings.Replace(text, "[tool.uv.sources]\n", "[tool.uv.sources]\n"+sources.String(), 1)
		}
		target := filepath.Join(project, relative)
		must(t, os.MkdirAll(filepath.Dir(target), 0o700))
		must(t, os.WriteFile(target, []byte(text), 0o600))
	}
	return project
}
