package producttest

import (
	"encoding/json"
	"flag"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

var liveModelImportE2E = flag.String("live-model-import-e2e", "", "real local model-import fixture JSON")
var liveModelImportHF = flag.String("live-model-import-hf", "", "real public HF dry-run fixture JSON")

func TestModelImportGrammarAndFailFastRefusals(t *testing.T) {
	root := t.TempDir()
	env := []string{"COZY_HOME=" + root, "PATH=/usr/local/bin:/usr/bin:/bin"}
	if result := runCozyEnv(env, "model", "import", "--help"); result.code != 0 ||
		!strings.Contains(result.output, "<source>") || !strings.Contains(result.output, "--name") ||
		!strings.Contains(result.output, "--dry-run") || !strings.Contains(result.output, "--token-stdin") ||
		strings.Contains(result.output, "--hf") || strings.Contains(result.output, "--civitai") ||
		strings.Contains(result.output, "--url") || strings.Contains(result.output, "--rental") ||
		strings.Contains(result.output, "--release") || strings.Contains(result.output, "--lane") {
		t.Fatalf("model import help = exit %d\n%s", result.code, result.output)
	}
	if result := runCozyEnv(env, "model", "import", "hf://org/model@"+strings.Repeat("a", 40),
		"--name", "BadName"); result.code != 2 || !strings.Contains(result.output, "portable name") {
		t.Fatalf("invalid local name = exit %d\n%s", result.code, result.output)
	}
	file := filepath.Join(root, "local.safetensors")
	must(t, os.WriteFile(file, []byte("not-a-real-carrier"), 0o600))
	if result := runCozyEnv(env, "model", "import", file, "--name", "local-test", "--token-stdin"); result.code != 2 || !strings.Contains(result.output, "applies only") {
		t.Fatalf("local token stdin = exit %d\n%s", result.code, result.output)
	}
	if result := runCozyEnv(env, "model", "import",
		"https://civitai.com/api/download/models/1?token=leak", "--name", "leak"); result.code != 2 || !strings.Contains(result.output, "credentials must not appear") {
		t.Fatalf("URL credential = exit %d\n%s", result.code, result.output)
	}
	badConfig := t.TempDir()
	must(t, os.WriteFile(filepath.Join(badConfig, "config.yaml"), []byte(
		"tensorfs_registry: /tmp/operator-only.json\n"), 0o600))
	if result := runCozyEnv([]string{"COZY_HOME=" + badConfig, "PATH=/usr/local/bin:/usr/bin:/bin"}, "rental"); result.code != 2 || !strings.Contains(result.output, `unknown key "tensorfs_registry"`) {
		t.Fatalf("operator registry entered YAML = exit %d\n%s", result.code, result.output)
	}
}

type liveImport struct {
	TFS      string `json:"tfs"`
	Registry string `json:"registry,omitempty"`
	Source   string `json:"source"`
	Name     string `json:"name"`
}

func readLiveImport(t *testing.T, path string) liveImport {
	t.Helper()
	var fixture liveImport
	raw, err := os.ReadFile(path)
	must(t, err)
	must(t, json.Unmarshal(raw, &fixture))
	base := filepath.Dir(path)
	for _, field := range []*string{&fixture.TFS, &fixture.Registry, &fixture.Source} {
		if *field != "" && !strings.Contains(*field, "://") && !filepath.IsAbs(*field) {
			*field = filepath.Join(base, *field)
		}
	}
	if fixture.TFS == "" || fixture.Source == "" || fixture.Name == "" {
		t.Fatal("live model-import fixture is incomplete")
	}
	return fixture
}

func importEnv(root string, fixture liveImport) []string {
	env := []string{"COZY_HOME=" + root, "COZY_TFS=" + fixture.TFS,
		"PATH=/usr/local/bin:/usr/bin:/bin", "TENSORHUB_URL=http://127.0.0.1:1"}
	if fixture.Registry != "" {
		env = append(env, "COZY_TFS_REGISTRY="+fixture.Registry)
	}
	return env
}

func TestLiveLocalModelImport(t *testing.T) {
	if *liveModelImportE2E == "" {
		t.Skip("-live-model-import-e2e is not set")
	}
	fixture := readLiveImport(t, *liveModelImportE2E)
	root := t.TempDir()
	env := importEnv(root, fixture)
	if result := runCozyEnv(env, "model", "import", fixture.Source, "--name", fixture.Name, "--dry-run"); result.code != 0 || !strings.Contains(result.output, "planned") {
		t.Fatalf("local import dry-run = exit %d\n%s", result.code, result.output)
	}
	if result := runCozyEnv(env, "model", "import", fixture.Source, "--name", fixture.Name); result.code != 0 || !strings.Contains(result.output, "local/"+fixture.Name) ||
		!strings.Contains(result.output, "imported") {
		t.Fatalf("local import = exit %d\n%s", result.code, result.output)
	}
	if _, err := os.Stat(fixture.Source); err != nil {
		t.Fatalf("local source was deleted: %v", err)
	}
	if result := runCozyEnv(env, "model", "list"); result.code != 0 ||
		!strings.Contains(result.output, "local/"+fixture.Name) {
		t.Fatalf("model list after import = exit %d\n%s", result.code, result.output)
	}
}

func TestLivePublicHFModelImportPlan(t *testing.T) {
	if *liveModelImportHF == "" {
		t.Skip("-live-model-import-hf is not set")
	}
	fixture := readLiveImport(t, *liveModelImportHF)
	root := t.TempDir()
	result := runCozyEnv(importEnv(root, fixture), "model", "import", fixture.Source,
		"--name", fixture.Name, "--dry-run")
	if result.code != 0 || !strings.Contains(result.output, "hf://") ||
		!strings.Contains(result.output, "planned") {
		t.Fatalf("public HF import plan = exit %d\n%s", result.code, result.output)
	}
}
