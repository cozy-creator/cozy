package producttest

import (
	"net/http"
	"net/http/httptest"
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
		"model", "upload", "local/model", "hf://org/model@"+strings.Repeat("a", 40),
		"--dry-run")
	if result.code != 2 || !strings.Contains(result.output, "local/ is reserved") {
		t.Fatalf("local Tensorhub destination = exit %d\n%s", result.code, result.output)
	}
}

func TestForeignSourceLaneRefusesBeforeProviderNetwork(t *testing.T) {
	root := t.TempDir()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/v1/accounts/current" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"name":"acme"}`))
	}))
	defer server.Close()
	result := runCozyEnv([]string{"COZY_HOME=" + root, "PATH=/usr/local/bin:/usr/bin:/bin",
		"TENSORHUB_URL=" + server.URL, "TENSORHUB_TOKEN=proof-token"},
		"model", "upload", "acme/model", "hf://org/model@"+strings.Repeat("a", 40),
		"--lane", "bf16")
	if result.code != 2 || !strings.Contains(result.output, "--lane selects only") {
		t.Fatalf("recognized foreign source = exit %d\n%s", result.code, result.output)
	}
}

func TestMissingLocalPublishAliasRefusesBeforeAccountLookup(t *testing.T) {
	accountReads := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/accounts/current" {
			accountReads++
		}
		http.Error(w, "unexpected account lookup", http.StatusInternalServerError)
	}))
	defer server.Close()
	root := t.TempDir()
	tfs := filepath.Join(root, "tfs-missing-alias")
	must(t, os.WriteFile(tfs, []byte("#!/bin/sh\n"+
		"[ \"$1 $2\" = \"store init\" ] && exit 0\n"+
		"echo 'REFUSED NOT_FOUND: local alias missing' >&2\nexit 1\n"), 0o700))
	result := runCozyEnv([]string{"COZY_HOME=" + root, "PATH=/usr/local/bin:/usr/bin:/bin",
		"COZY_TFS=" + tfs, "TENSORHUB_URL=" + server.URL, "TENSORHUB_TOKEN=proof-token"},
		"model", "upload", "acme/model", "local/missing")
	if result.code == 0 || accountReads != 0 || !strings.Contains(result.output, "missing") {
		t.Fatalf("missing local alias = exit %d account reads %d\n%s",
			result.code, accountReads, result.output)
	}
}
