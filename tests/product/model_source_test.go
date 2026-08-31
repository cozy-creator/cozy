package producttest

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/modelsource"
)

func TestModelSourceGrammarPinsProvidersAndLocalFiles(t *testing.T) {
	commit := strings.Repeat("a", 40)
	cases := []struct {
		input string
		want  string
	}{
		{"hf://org/model@" + commit, "hf://org/model@" + commit},
		{"https://huggingface.co/org/model/tree/" + commit, "hf://org/model@" + commit},
		{"https://huggingface.co/org/model", "hf://org/model"},
		{"civitai://2514310", "civitai://2514310"},
		{"https://civitai.com/models/827184?modelVersionId=2514310", "civitai://2514310"},
		{"https://civitai.com/api/download/models/2514310", "civitai://2514310"},
	}
	for _, test := range cases {
		source, problem := modelsource.Parse(test.input, t.TempDir())
		if problem != nil || source.Canonical != test.want {
			t.Fatalf("Parse(%q) = %#v, %v; want %q", test.input, source, problem, test.want)
		}
	}

	dir := t.TempDir()
	path := filepath.Join(dir, "model.safetensors")
	must(t, os.WriteFile(path, []byte("carrier"), 0o600))
	source, problem := modelsource.Parse("./model.safetensors", dir)
	if problem != nil || source.Kind != modelsource.LocalFile || source.Path != path || source.Bytes != 7 {
		t.Fatalf("local source = %#v, %v", source, problem)
	}
}

func TestModelSourceGrammarRefusesAmbiguityAndCredentials(t *testing.T) {
	bad := []string{
		"hf://org/model@main",
		"https://huggingface.co/org/model/resolve/main/model.safetensors",
		"https://civitai.com/models/827184",
		"https://civitai.com/models/827184?modelVersionId=1&modelVersionId=2",
		"https://civitai.com/api/download/models/1?token=leak",
		"https://example.com/model.safetensors",
		"model.safetensors",
	}
	for _, input := range bad {
		if _, problem := modelsource.Parse(input, t.TempDir()); problem == nil {
			t.Fatalf("Parse(%q) accepted", input)
		}
	}
}

func TestProviderSecretsRequireOwnerOnlyNonsymlinkConfig(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("POSIX owner and mode contract")
	}
	run := func(root string, args ...string) cozyResult {
		return runCozyEnv([]string{"COZY_HOME=" + root, "PATH=/usr/local/bin:/usr/bin:/bin"}, args...)
	}
	insecure := t.TempDir()
	must(t, os.WriteFile(filepath.Join(insecure, "config.yaml"), []byte("huggingface_token: hf_test\n"), 0o644))
	if result := run(insecure, "rental"); result.code != 2 || !strings.Contains(result.output, "mode 0600") {
		t.Fatalf("insecure provider config = exit %d\n%s", result.code, result.output)
	}

	symlinked := t.TempDir()
	target := filepath.Join(symlinked, "real.yaml")
	must(t, os.WriteFile(target, []byte("civitai_token: cv_test\n"), 0o600))
	must(t, os.Symlink(target, filepath.Join(symlinked, "config.yaml")))
	if result := run(symlinked, "rental"); result.code != 2 || !strings.Contains(result.output, "non-symlink") {
		t.Fatalf("symlinked provider config = exit %d\n%s", result.code, result.output)
	}

	protected := t.TempDir()
	must(t, os.WriteFile(filepath.Join(protected, "config.yaml"), []byte(
		"port: 0\nhuggingface_token: hf_test\ncivitai_token: cv_test\n"), 0o600))
	if result := run(protected, "rental"); result.code != 0 || !strings.Contains(result.output, "0 remote machines") {
		t.Fatalf("protected provider config = exit %d\n%s", result.code, result.output)
	}
}

func TestTensorhubDestinationCannotUseLocalNamespace(t *testing.T) {
	root := t.TempDir()
	result := runCozyEnv([]string{"COZY_HOME=" + root, "PATH=/usr/local/bin:/usr/bin:/bin"},
		"model", "publish", "local/model", "sha256:"+strings.Repeat("a", 64),
		"--release", "1.0.0", "--lane", "bf16")
	if result.code != 2 || !strings.Contains(result.output, "local/ is reserved") {
		t.Fatalf("local Tensorhub destination = exit %d\n%s", result.code, result.output)
	}
}
