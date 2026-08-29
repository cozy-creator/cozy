package live

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"testing"

	"github.com/cozy-creator/cozy-creator/internal/endpointprofile"
	"github.com/cozy-creator/cozy-creator/internal/endpointpublish"
)

func TestEndpointPublishCLI(t *testing.T) {
	source := trackedEndpointFixture(t)
	var server *httptest.Server
	var lock sync.Mutex
	var declaration []byte
	var declared endpointpublish.Declaration
	put := map[string][]byte{}
	beginCount, finalizeCount := 0, 0
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.Method != http.MethodPut && r.Method != http.MethodGet && r.Header.Get("Authorization") != "Bearer admin" {
			writeHubError(w, http.StatusUnauthorized, "auth.token_invalid", "wrong token")
			return
		}
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/v1/endpoints":
			_, _ = w.Write([]byte(`{"endpoint":{"org":"cozy","name":"marco","created_at":"2026-08-28T00:00:00Z"}}`))
		case r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/begin"):
			body, _ := io.ReadAll(r.Body)
			lock.Lock()
			beginCount++
			if declaration == nil {
				declaration = append([]byte(nil), body...)
				if err := json.Unmarshal(body, &declared); err != nil {
					t.Errorf("begin declaration: %s", err)
				}
			} else if string(declaration) != string(body) {
				t.Errorf("begin replay changed declaration bytes")
			}
			lock.Unlock()
			roles := fixtureDeclarationRoles(declared)
			rows := make([]map[string]any, 0, len(roles))
			for _, role := range sortedKeys(roles) {
				ref := roles[role]
				held := role == "descriptor"
				row := map[string]any{"role": role, "ref": ref, "already_held": held,
					"required_headers": map[string]string{}, "expires_at": "2099-08-29T00:00:00Z"}
				if !held {
					row["url"] = server.URL + "/put/" + role
					row["required_headers"] = map[string]string{"X-Test-Role": role}
				}
				rows = append(rows, row)
			}
			digest := fixtureDigest(body)
			_ = json.NewEncoder(w).Encode(map[string]any{"declaration_digest": digest,
				"state": "pending", "uploads": rows})
		case r.Method == http.MethodPut && strings.HasPrefix(r.URL.Path, "/put/"):
			role := strings.TrimPrefix(r.URL.Path, "/put/")
			if r.Header.Get("X-Test-Role") != role {
				t.Errorf("PUT %s omitted signed header", role)
			}
			body, _ := io.ReadAll(r.Body)
			lock.Lock()
			put[role] = body
			lock.Unlock()
			w.WriteHeader(http.StatusOK)
		case r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/finalize"):
			body, _ := io.ReadAll(r.Body)
			lock.Lock()
			finalizeCount++
			if string(declaration) != string(body) {
				t.Errorf("finalize changed declaration bytes")
			}
			for role, ref := range fixtureDeclarationRoles(declared) {
				if role == "descriptor" {
					continue
				}
				body := put[role]
				if int64(len(body)) != ref.Length || fixtureDigest(body) != ref.Digest {
					t.Errorf("uploaded role %s disagrees with declaration", role)
				}
			}
			lock.Unlock()
			_ = json.NewEncoder(w).Encode(map[string]any{"created": true,
				"release": "1.0.0", "declaration_digest": fixtureDigest(declaration),
				"profiles": []map[string]any{
					{"profile": endpointprofile.CPU, "state": "qualified", "candidate_id": "candidate-cpu",
						"base_realization_kind": "oci", "base_realization_digest": "registry.invalid/tensorhub-worker@sha256:" + strings.Repeat("a", 64),
						"endpoint_environment_spec": map[string]any{"digest": "sha256:" + strings.Repeat("b", 64), "length": 1},
						"resolved_wheel_set":        map[string]any{"digest": "sha256:" + strings.Repeat("c", 64), "length": 1},
						"resolution_lock":           map[string]any{"digest": "sha256:" + strings.Repeat("d", 64), "length": 1}}},
				"endpoint_executions": []map[string]string{{"profile": endpointprofile.CPU, "function": "marco", "digest": "sha256:" + strings.Repeat("1", 64), "state": "qualified"}}})
		default:
			writeHubError(w, http.StatusNotFound, "route.not_found", r.Method+" "+r.URL.Path)
		}
	})
	server = httptest.NewServer(handler)
	defer server.Close()

	home := t.TempDir()
	run := func(args ...string) (int, string) {
		cmd := exec.Command("/usr/bin/nice", append([]string{"-n", "19", cozyBin}, args...)...)
		cmd.Env = childEnv(t, home, "TENSORHUB_URL="+server.URL, "TENSORHUB_TOKEN=admin",
			"PATH="+filepath.Join(source, ".test-bin")+":/usr/local/bin:/usr/bin:/bin")
		body, _ := cmd.CombinedOutput()
		return cmd.ProcessState.ExitCode(), string(body)
	}
	args := []string{"endpoint", "publish", "cozy/marco", "--release", "1.0.0", "--dir", source,
		"--reason", "fixture publish"}
	if code, out := run(args...); code != 0 || !strings.Contains(out, "status: published") ||
		!strings.Contains(out, "qualified") || strings.Contains(out, "candidate-cpu") {
		t.Fatalf("endpoint publish [exit %d]\n%s", code, out)
	}
	// Exact replay sends identical declaration bytes and converges without a client journal.
	if code, out := run(append(args, "--full")...); code != 0 ||
		!strings.Contains(out, "candidate-cpu") || !strings.Contains(out, "qualified") {
		t.Fatalf("endpoint publish replay [exit %d]\n%s", code, out)
	}
	lock.Lock()
	defer lock.Unlock()
	if beginCount != 2 || finalizeCount != 2 {
		t.Fatalf("calls begin=%d finalize=%d", beginCount, finalizeCount)
	}
}

func trackedEndpointFixture(t *testing.T) string {
	t.Helper()
	repo := t.TempDir()
	mustWrite(t, filepath.Join(repo, "pyproject.toml"), `[project]
name = "marco-polo"
version = "1.0.0"
dependencies = []
`)
	mustWrite(t, filepath.Join(repo, "endpoint.toml"), "[application]\nobject = \"marco_polo:app\"\n")
	mustWrite(t, filepath.Join(repo, "marco_polo.py"), "app = object()\n")
	mustWrite(t, filepath.Join(repo, "uv.lock"), "version = 1\n")
	mustWrite(t, filepath.Join(repo, ".gitignore"), ".test-bin/\n.venv/\n")
	git(t, repo, "init", "-q")
	git(t, repo, "config", "user.email", "fixture@example.invalid")
	git(t, repo, "config", "user.name", "Fixture")
	git(t, repo, "add", ".")
	git(t, repo, "commit", "-qm", "fixture")
	fakeUV := filepath.Join(repo, ".test-bin", "uv")
	mustWrite(t, fakeUV, `#!/bin/sh
if [ "${0##*/}" = "cozy-runtime" ]; then # //cozy:allow independent Runtime CLI fixture
  printf '%s\n' '{"application":"marco_polo:app","entrypoints":[],"format":"cozy.endpoint.descriptor/1","jobs":[]}'
  exit 0
fi
mkdir -p "$UV_PROJECT_ENVIRONMENT/bin"
cp "$0" "$UV_PROJECT_ENVIRONMENT/bin/cozy-runtime"
chmod 755 "$UV_PROJECT_ENVIRONMENT/bin/cozy-runtime"
`)
	must(t, os.Chmod(fakeUV, 0o755))
	return repo
}

func fixtureDeclarationRoles(d endpointpublish.Declaration) map[string]endpointpublish.ObjectRef {
	out := map[string]endpointpublish.ObjectRef{
		"source_archive": d.SourceArchive, "source_lock": d.SourceLock,
		"project_wheel": {Digest: d.ProjectWheel.Digest, Length: d.ProjectWheel.Length},
		"descriptor":    d.Descriptor,
	}
	return out
}

func sortedKeys[V any](values map[string]V) []string {
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

func fixtureDigest(body []byte) string {
	sum := sha256.Sum256(body)
	return "sha256:" + hex.EncodeToString(sum[:])
}

func writeHubError(w http.ResponseWriter, status int, code, message string) {
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{"error": map[string]string{
		"code": code, "message": message, "remedy": "fixture"}})
}
