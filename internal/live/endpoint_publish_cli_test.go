package live

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"testing"

	"github.com/cozy-creator/cozy-creator/internal/endpointprofile"
	"github.com/cozy-creator/cozy-creator/internal/endpointpublish"
)

func TestEndpointPublishQualifyPromoteCLI(t *testing.T) {
	source := trackedEndpointFixture(t)
	var server *httptest.Server
	var lock sync.Mutex
	var declaration []byte
	var declared endpointpublish.Declaration
	put := map[string][]byte{}
	beginCount, finalizeCount, qualificationReads := 0, 0, 0
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.Method != http.MethodPut && r.Header.Get("Authorization") != "Bearer admin" {
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
				"state": "pending", "uploads": rows,
				"profiles": []map[string]string{{"profile": endpointprofile.CU126, "state": "candidate_pending"}, {"profile": endpointprofile.CU130, "state": "candidate_pending"}}})
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
					{"profile": endpointprofile.CU126, "state": "candidate", "candidate_id": "candidate-126",
						"base_realization_kind": "oci", "base_realization_digest": "registry.invalid/tensorhub-worker@sha256:" + strings.Repeat("a", 64),
						"endpoint_environment_spec": map[string]any{"digest": "sha256:" + strings.Repeat("b", 64), "length": 1},
						"resolved_wheel_set":        map[string]any{"digest": "sha256:" + strings.Repeat("c", 64), "length": 1},
						"resolution_lock":           map[string]any{"digest": "sha256:" + strings.Repeat("d", 64), "length": 1}},
					{"profile": endpointprofile.CU130, "state": "candidate", "candidate_id": "candidate-130",
						"base_realization_kind": "oci", "base_realization_digest": "registry.invalid/tensorhub-worker@sha256:" + strings.Repeat("e", 64),
						"endpoint_environment_spec": map[string]any{"digest": "sha256:" + strings.Repeat("f", 64), "length": 1},
						"resolved_wheel_set":        map[string]any{"digest": "sha256:" + strings.Repeat("1", 64), "length": 1},
						"resolution_lock":           map[string]any{"digest": "sha256:" + strings.Repeat("2", 64), "length": 1}}},
				"endpoint_executions": []map[string]string{{"profile": endpointprofile.CU126, "function": "marco", "digest": "sha256:" + strings.Repeat("1", 64), "state": "candidate"}}})
		case r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/qualify"):
			var body map[string]any
			_ = json.NewDecoder(r.Body).Decode(&body)
			if body["accelerator_model"] != "NVIDIA GeForce RTX 4090" ||
				body["provider_exposure_limit_usd_micros"] != float64(250000) ||
				body["duration_cap_s"] != float64(900) {
				t.Errorf("qualification envelope changed: %v", body)
			}
			_ = json.NewEncoder(w).Encode(map[string]any{
				"qualification_id": "qualification-1", "candidate_id": "candidate-130",
				"state": "acquiring", "accelerator_model": body["accelerator_model"],
				"provider_exposure_limit_usd_micros": 250000, "duration_cap_s": 900,
				"model_qualification_spec_digest": "sha256:" + strings.Repeat("4", 64),
				"reclaim_proven":                  false})
		case r.Method == http.MethodGet && strings.HasSuffix(r.URL.Path, "/qualification"):
			lock.Lock()
			qualificationReads++
			lock.Unlock()
			_ = json.NewEncoder(w).Encode(map[string]any{
				"qualification_id": "qualification-1", "candidate_id": "candidate-130",
				"state": "qualified", "accelerator_model": "NVIDIA GeForce RTX 4090",
				"provider_exposure_limit_usd_micros": 250000, "duration_cap_s": 900,
				"observed_cost_usd_micros": 50000, "provider_resource_id": "pod-1",
				"model_qualification_spec_digest": "sha256:" + strings.Repeat("4", 64),
				"execution_observation_digest":    "sha256:" + strings.Repeat("5", 64),
				"model_admission_decision_digest": "sha256:" + strings.Repeat("6", 64),
				"reclaim_proven":                  true})
		case r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/promote"):
			var body struct {
				Serving []map[string]string `json:"serving"`
			}
			_ = json.NewDecoder(r.Body).Decode(&body)
			if len(body.Serving) != 2 || body.Serving[0]["function"] != "edit" || body.Serving[1]["function"] != "marco" {
				t.Errorf("serving set was not sorted/deduplicated: %v", body.Serving)
			}
			_ = json.NewEncoder(w).Encode(map[string]any{"release": "1.0.0", "serving": []map[string]any{
				{"endpoint_ref": "cozy/marco/v1/edit", "endpoint_execution_digests": []string{"sha256:" + strings.Repeat("2", 64)}},
				{"endpoint_ref": "cozy/marco/v1/marco", "endpoint_execution_digests": []string{"sha256:" + strings.Repeat("1", 64)}}}})
		default:
			writeHubError(w, http.StatusNotFound, "route.not_found", r.Method+" "+r.URL.Path)
		}
	})
	server = httptest.NewServer(handler)
	defer server.Close()

	home := t.TempDir()
	run := func(args ...string) (int, string) {
		cmd := exec.Command("/usr/bin/nice", append([]string{"-n", "19", cozyBin}, args...)...)
		cmd.Env = childEnv(t, home, "TENSORHUB_URL="+server.URL, "TENSORHUB_TOKEN=admin")
		body, _ := cmd.CombinedOutput()
		return cmd.ProcessState.ExitCode(), string(body)
	}
	args := []string{"endpoint", "publish", "cozy/marco", "--release", "1.0.0", "--dir", source,
		"--profile", endpointprofile.CU130, "--profile", endpointprofile.CU126,
		"--create", "--reason", "fixture publish"}
	if code, out := run(args...); code != 0 || !strings.Contains(out, "candidate-130") || !strings.Contains(out, "serving-pointer move ran") {
		t.Fatalf("endpoint publish [exit %d]\n%s", code, out)
	}
	// Exact replay omits create (resource exists) and must send identical declaration bytes.
	replay := append([]string(nil), args...)
	replay = deleteArg(replay, "--create")
	if code, out := run(replay...); code != 0 || !strings.Contains(out, "true") || !strings.Contains(out, "candidate") {
		t.Fatalf("endpoint publish replay [exit %d]\n%s", code, out)
	}
	if code, out := run("endpoint", "qualify", "cozy/marco@1.0.0", "--profile", endpointprofile.CU130,
		"--gpu", "NVIDIA GeForce RTX 4090", "--max-cost", "0.25", "--reason", "fixture qualification"); code != 0 || !strings.Contains(out, "acquiring") || !strings.Contains(out, "qualified") || !strings.Contains(out, "reclaimed:") || !strings.Contains(out, "true") {
		t.Fatalf("endpoint qualify [exit %d]\n%s", code, out)
	}
	if code, out := run("endpoint", "promote", "cozy/marco", "1.0.0", "--serve", "v1/marco",
		"--serve", "v1/edit", "--reason", "fixture promotion"); code != 0 || !strings.Contains(out, "serving:") || !strings.Contains(out, "one Tensorhub transaction") {
		t.Fatalf("endpoint promote [exit %d]\n%s", code, out)
	}
	lock.Lock()
	defer lock.Unlock()
	if beginCount != 2 || finalizeCount != 2 || qualificationReads != 1 {
		t.Fatalf("calls begin=%d finalize=%d qualification_reads=%d", beginCount, finalizeCount, qualificationReads)
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
	mustWrite(t, filepath.Join(repo, "endpoint.descriptor.json"), `{"application":"marco_polo:app","entrypoints":[],"format":"cozy.endpoint.descriptor/1","jobs":[]}`)
	mustWrite(t, filepath.Join(repo, "endpoint.release.json"), `{"compatible_accelerator_models":["NVIDIA GeForce RTX 4090"],"model_bindings":[],"model_roots":[]}`)
	git(t, repo, "init", "-q")
	git(t, repo, "config", "user.email", "fixture@example.invalid")
	git(t, repo, "config", "user.name", "Fixture")
	git(t, repo, "add", ".")
	git(t, repo, "commit", "-qm", "fixture")
	return repo
}

func fixtureDeclarationRoles(d endpointpublish.Declaration) map[string]endpointpublish.ObjectRef {
	out := map[string]endpointpublish.ObjectRef{
		"source_archive": d.SourceArchive, "source_lock": d.SourceLock,
		"project_wheel": {Digest: d.ProjectWheel.Digest, Length: d.ProjectWheel.Length},
		"descriptor":    d.Descriptor, "evaluated_config": d.EvaluatedConfig,
	}
	for _, custom := range d.CustomWheels {
		role := "custom_wheel:" + custom.Wheel.Distribution + ":" + strings.TrimPrefix(custom.Wheel.Digest, "sha256:")
		out[role] = endpointpublish.ObjectRef{Digest: custom.Wheel.Digest, Length: custom.Wheel.Length}
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

func deleteArg(values []string, target string) []string {
	for i, value := range values {
		if value == target {
			return append(values[:i], values[i+1:]...)
		}
	}
	return values
}
