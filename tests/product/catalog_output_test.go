package producttest

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"
)

func TestModelSearchShowsAuthoritativeAvailableReleaseLanes(t *testing.T) {
	manifest := "sha256:" + strings.Repeat("a", 64)
	searches, alphaCards, betaCards, packageSearches, packageCards := 0, 0, 0, 0, 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/v1/models":
			searches++
			if r.Method != http.MethodGet || r.URL.Query().Get("q") != "" {
				t.Fatalf("model search = %s %s", r.Method, r.URL.String())
			}
			_, _ = io.WriteString(w, `{"models":[{"org":"acme","name":"alpha","created_at":"2026-09-01T00:00:00Z"},{"org":"acme","name":"beta","created_at":"2026-09-02T00:00:00Z"}],"search":{"total":2,"limit":1000,"capped":false,"q":""}}`)
		case "/v1/models/acme/alpha":
			alphaCards++
			_, _ = io.WriteString(w, `{"model":{"org":"acme","name":"alpha","created_at":"2026-09-01T00:00:00Z"},"releases":[{"release":"2.0.0","lanes":[{"lane":"fp8","manifest_id":"`+manifest+`"},{"lane":"bf16","manifest_id":"`+manifest+`"},{"lane":"bf16","manifest_id":"`+manifest+`"}]},{"release":"1.0.0","lanes":[{"lane":"bf16","manifest_id":"`+manifest+`"}]},{"release":"broken","yanked":true,"lanes":[{"lane":"q4","manifest_id":"`+manifest+`"}]}]}`)
		case "/v1/models/acme/beta":
			betaCards++
			_, _ = io.WriteString(w, `{"model":{"org":"acme","name":"beta","created_at":"2026-09-02T00:00:00Z"},"releases":[{"release":"stable","lanes":[{"lane":"int8","manifest_id":"`+manifest+`"}]}]}`)
		case "/v1/packages":
			packageSearches++
			_, _ = io.WriteString(w, `{"packages":[{"org":"acme","name":"package","created_at":"2026-09-03T00:00:00Z"}],"search":{"total":1,"limit":1000,"capped":false,"q":""}}`)
		case "/v1/packages/acme/package":
			packageCards++
			_, _ = io.WriteString(w, `{"package":{"org":"acme","name":"package","created_at":"2026-09-03T00:00:00Z"}}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()
	env := []string{"TENSORHUB_URL=" + server.URL}

	code, out := runCozyDir(t, t.TempDir(), ".", env, "model", "search")
	lines := strings.Split(strings.TrimSpace(out), "\n")
	if code != 0 || len(lines) != 3 || !reflect.DeepEqual(strings.Fields(lines[0]), []string{"MODEL", "LANES"}) ||
		!strings.Contains(lines[1], "acme/alpha") || !strings.Contains(lines[1], "1.0.0/bf16, 2.0.0/bf16, 2.0.0/fp8") ||
		!strings.Contains(lines[2], "acme/beta") || !strings.Contains(lines[2], "stable/int8") ||
		strings.Contains(out, "q4") || strings.Contains(out, manifest) {
		t.Fatalf("model search output [exit %d]\n%s", code, out)
	}

	code, out = runCozyDir(t, t.TempDir(), ".", env, "--json", "model", "search", "acme/alpha")
	var document struct {
		Models []map[string]string `json:"models"`
	}
	if code != 0 || json.Unmarshal([]byte(out), &document) != nil || len(document.Models) != 1 ||
		document.Models[0]["model"] != "acme/alpha" ||
		document.Models[0]["lanes"] != "1.0.0/bf16, 2.0.0/bf16, 2.0.0/fp8" ||
		strings.Contains(out, manifest) {
		t.Fatalf("exact model search output [exit %d]\n%s", code, out)
	}

	code, out = runCozyDir(t, t.TempDir(), ".", env, "package", "search")
	if code != 0 || strings.TrimSpace(out) != "- acme/package" {
		t.Fatalf("package search output changed [exit %d]\n%s", code, out)
	}
	code, out = runCozyDir(t, t.TempDir(), ".", env, "package", "search", "acme/package")
	if code != 0 || !strings.Contains(out, "ref:     acme/package") ||
		!strings.Contains(out, "created: 2026-09-03T00:00:00Z") {
		t.Fatalf("exact package search output changed [exit %d]\n%s", code, out)
	}
	if searches != 1 || alphaCards != 2 || betaCards != 1 || packageSearches != 1 || packageCards != 1 {
		t.Fatalf("catalog requests = searches %d alpha %d beta %d package searches %d cards %d",
			searches, alphaCards, betaCards, packageSearches, packageCards)
	}
}

func TestModelListHidesManifestDigestByDefault(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("shell-backed TensorFS product fixture")
	}
	root := t.TempDir()
	tfs := filepath.Join(root, "tfs-fixture")
	row := `{"org":"acme","name":"alpha","version":"2.0.0","lane":"bf16","manifest_sha256":"` + strings.Repeat("b", 64) + `","manifest_length":7,"evidence":{}}`
	script := "#!/bin/sh\n" +
		"if [ \"$1 $2\" = \"store init\" ]; then exit 0; fi\n" +
		"if [ \"$1 $2\" = \"repo list\" ]; then\n" +
		"  while [ \"$1\" != \"--rows\" ]; do shift; done\n" +
		"  printf '%s\\n' '" + row + "' > \"$2\"\n" +
		"  exit 0\n" +
		"fi\n" +
		"echo unexpected tfs invocation >&2\nexit 1\n"
	must(t, os.WriteFile(tfs, []byte(script), 0o700))
	env := []string{"COZY_TFS=" + tfs}
	manifest := "sha256:" + strings.Repeat("b", 64)

	code, out := runCozyDir(t, root, ".", env, "model", "list")
	lines := strings.Split(strings.TrimSpace(out), "\n")
	if code != 0 || len(lines) != 2 ||
		!reflect.DeepEqual(strings.Fields(lines[0]), []string{"MODEL", "RELEASE", "LANE"}) ||
		strings.Contains(out, "MANIFEST_ID") || strings.Contains(out, manifest) {
		t.Fatalf("default model list [exit %d]\n%s", code, out)
	}

	code, out = runCozyDir(t, root, ".", env, "--full", "model", "list")
	if code != 0 || !strings.Contains(out, "MANIFEST_ID") || !strings.Contains(out, manifest) {
		t.Fatalf("full model list [exit %d]\n%s", code, out)
	}

	code, out = runCozyDir(t, root, ".", env, "--json", "model", "list")
	if code != 0 || strings.Contains(out, "manifest_id") || strings.Contains(out, manifest) {
		t.Fatalf("default JSON model list [exit %d]\n%s", code, out)
	}

	code, out = runCozyDir(t, root, ".", env, "--json", "--fields", "model,manifest_id", "model", "list")
	if code != 0 || !strings.Contains(out, `"manifest_id":"`+manifest+`"`) {
		t.Fatalf("selected model list digest [exit %d]\n%s", code, out)
	}
}
