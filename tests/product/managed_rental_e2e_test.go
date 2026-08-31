package producttest

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/daemon"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/records"
)

func TestRentalRunAcquiresCheapestOnceAndReleasesFailedPreAttempt(t *testing.T) {
	const releaseDigest = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	const modelManifest = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
	var mu sync.Mutex
	var descriptor []byte
	var descriptorDigest string
	var rentalRequest map[string]any
	var createReason, deleteReason string
	activeRentalID := "pr-managed-e2e"
	posts, deletes := 0, 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		switch {
		case r.Method == http.MethodGet && r.URL.Path == "/v1/models/cozy/tiny":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"model": map[string]any{"org": "cozy", "name": "tiny"},
				"releases": []map[string]any{{"release": "1.0.0", "lanes": []map[string]any{{
					"lane": "bf16", "manifest_id": modelManifest,
				}}}},
			})
		case r.Method == http.MethodGet && r.URL.Path == "/v1/packages/cozy/cozy-weightless-package/releases/1.0.0":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"release": map[string]any{
					"release": "1.0.0", "release_digest": releaseDigest,
					"package_descriptor_digest": descriptorDigest,
					"package_descriptor_length": len(descriptor),
					"created_at":                "2026-08-30T00:00:00Z",
				},
				"document": map[string]any{}, "package_descriptor": json.RawMessage(descriptor),
			})
		case r.Method == http.MethodGet && r.URL.Path == "/v1/packages/cozy/cozy-weightless-package":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"package": map[string]any{"org": "cozy", "name": "cozy-weightless-package",
					"created_at": "2026-08-30T00:00:00Z"},
				"releases": []map[string]any{{"release": "1.0.0", "cut_at": "2026-08-30T00:00:00Z"}},
			})
		case strings.HasSuffix(r.URL.Path, "/download"):
			t.Error("remote metadata resolution called the package download route")
			http.Error(w, "download route must be worker-only in this flow", http.StatusInternalServerError)
		case r.Method == http.MethodGet && r.URL.Path == "/v1/rental-skus":
			_ = json.NewEncoder(w).Encode([]map[string]any{
				{"name": "expensive", "accelerator_model": "GPU X", "compute_capability": "9.0", "vram_gb": 80, "price_usd_micros_per_hour": 900_000},
				{"name": "cheap", "accelerator_model": "GPU C", "compute_capability": "8.9", "vram_gb": 24, "price_usd_micros_per_hour": 300_000},
			})
		case r.Method == http.MethodPost && r.URL.Path == "/v1/rentals":
			posts++
			createReason = r.Header.Get("X-Tensorhub-Reason")
			if err := json.NewDecoder(r.Body).Decode(&rentalRequest); err != nil {
				t.Error(err)
			}
			w.WriteHeader(http.StatusAccepted)
			_ = json.NewEncoder(w).Encode(map[string]any{
				"rental_id": activeRentalID, "state": "pending_acquisition",
				"requested_accelerator_model": "GPU C", "hourly_rate_usd_micros": 300_000,
			})
		case r.Method == http.MethodDelete && r.URL.Path == "/v1/rentals/"+activeRentalID:
			deletes++
			deleteReason = r.Header.Get("X-Tensorhub-Reason")
			w.WriteHeader(http.StatusNoContent)
		case r.Method == http.MethodGet && r.URL.Path == "/v1/rentals/"+activeRentalID:
			state := "ready"
			if deletes > 0 {
				state = "released"
			}
			answer := map[string]any{
				"rental_id": activeRentalID, "state": state,
				"requested_accelerator_model": "GPU C", "hourly_rate_usd_micros": 300_000,
			}
			if state == "ready" {
				answer["worker_address"] = "127.0.0.1:9443"
				answer["media_address"] = "127.0.0.1:9444"
				answer["worker_id"] = "worker-e2e"
				answer["worker_boot_id"] = "boot-e2e"
				answer["cert_pem"] = "not-a-certificate"
				answer["creator_public_key"] = rentalRequest["creator_public_key"]
				answer["media_token_sha256"] = []any{rentalRequest["media_token_sha256"]}
			}
			_ = json.NewEncoder(w).Encode(answer)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	root := t.TempDir()
	must(t, os.WriteFile(filepath.Join(root, "config.yaml"), []byte(
		"tensorhub_url: "+server.URL+"\ntensorhub_token: test-token\nport: 0\n"+
			"rentals:\n  max_hourly_spend_usd: 1.00\n"), 0o600))
	t.Cleanup(func() { _, _ = runCozy(t, root, "down", "--all") })
	project := weightlessProject(t)
	if code, out := runCozy(t, root, "package", "install", project); code != 0 {
		t.Fatalf("fixture install [exit %d]\n%s", code, out)
	}
	local := activePackageInstall(t, root)
	descriptor, _ = os.ReadFile(launch.DescriptorPath(local.Dir))
	var descriptorDocument map[string]any
	must(t, json.Unmarshal(descriptor, &descriptorDocument))
	entrypoints := descriptorDocument["entrypoints"].([]any)
	for _, entrypoint := range entrypoints {
		row := entrypoint.(map[string]any)
		if row["name"] == "tile" {
			encoded, _ := json.Marshal(row)
			var job map[string]any
			must(t, json.Unmarshal(encoded, &job))
			job["name"], job["publishes"] = "tile_job", false
			descriptorDocument["jobs"] = []any{job}
			row["models"] = []any{map[string]any{
				"class": "TinyModel", "path": "tile.models.model",
				"stamps": map[string]any{}, "component_use": map[string]any{"core": []any{"tile"}},
			}}
			descriptorDocument["entrypoints"] = []any{row}
		}
	}
	descriptor, _ = json.Marshal(descriptorDocument)
	decoded, problem := launch.DecodeDescriptor(descriptor)
	fatal(t, problem)
	jobDescriptorID := decoded.Jobs[0].DescriptorID
	if jobDescriptorID == "" {
		t.Fatal("remote job descriptor id was not derived")
	}
	descriptorDigest = decoded.Digest
	store, problem := records.Open(filepath.Join(root, "records.db"))
	fatal(t, problem)
	store.Close()
	daemonPath := filepath.Join(root, "cozy-daemon")
	must(t, os.Symlink(cozyBin, daemonPath))
	daemonCommand := exec.Command(daemonPath)
	daemonCommand.Env = childEnv(t, root)
	var daemonOutput bytes.Buffer
	daemonCommand.Stdout, daemonCommand.Stderr = &daemonOutput, &daemonOutput
	must(t, daemonCommand.Start())
	t.Cleanup(func() {
		if daemonCommand.Process != nil {
			_ = daemonCommand.Process.Signal(os.Interrupt)
			_, _ = daemonCommand.Process.Wait()
		}
	})
	readyDeadline := time.Now().Add(5 * time.Second)
	for !daemon.Probe(config.Config{Home: root}).Up && time.Now().Before(readyDeadline) {
		time.Sleep(25 * time.Millisecond)
	}
	if !daemon.Probe(config.Config{Home: root}).Up {
		t.Fatalf("foreground daemon did not start:\n%s", daemonOutput.String())
	}

	code, out := runCozy(t, root, "run", weightlessRef+"/tile", "size=32", "seed=7",
		"--model", "model=cozy/tiny@1.0.0#"+modelManifest,
		"--rental", "--detach", "--idempotency-key", "managed-e2e", "--json")
	if code != 0 || !strings.Contains(out, `"changed":true`) {
		t.Fatalf("rental submission [exit %d]\n%s\ndaemon:\n%s", code, out, daemonOutput.String())
	}
	deadline := time.Now().Add(10 * time.Second)
	var request *records.Request
	var events []records.Event
	for time.Now().Before(deadline) {
		store, problem = records.Open(filepath.Join(root, "records.db"))
		fatal(t, problem)
		rows, rowsProblem := store.RequestsOfKind("serving", "", 20)
		fatal(t, rowsProblem)
		for i := range rows {
			if rows[i].IdemKey == "managed-e2e" {
				row := rows[i]
				request = &row
				events, _ = store.EventsAfter(row.ID, 0, 100)
			}
		}
		rentalRow, _ := store.RentalRow("pr-managed-e2e")
		store.Close()
		mu.Lock()
		done := posts == 1 && deletes == 1 && rentalRow == nil && request != nil && request.State == "failed"
		mu.Unlock()
		if done {
			break
		}
		time.Sleep(25 * time.Millisecond)
	}
	mu.Lock()
	gotPosts, gotDeletes, body := posts, deletes, rentalRequest
	gotCreateReason, gotDeleteReason := createReason, deleteReason
	mu.Unlock()
	if request == nil || request.Worker != "pr-managed-e2e" || request.InstallID != "" ||
		request.Release != "1.0.0" || request.PackageRevisionDigest != releaseDigest ||
		len(request.Models) != 1 || request.Models[0].Package != weightlessRef ||
		request.Models[0].Slot != "tile.models.model" || request.Models[0].Model != "cozy/tiny" ||
		request.Models[0].Release != "1.0.0" || request.Models[0].Manifest != modelManifest ||
		gotPosts != 1 || gotDeletes != 1 ||
		body["sku"] != "cheap" || body["max_cost_usd_micros"] != nil || body["max_duration_seconds"] != nil {
		t.Fatalf("managed lifecycle request=%+v posts=%d deletes=%d body=%v", request, gotPosts, gotDeletes, body)
	}
	if gotCreateReason != "" || gotDeleteReason != "" ||
		strings.Contains(fmt.Sprint(body), weightlessRef) || strings.Contains(fmt.Sprint(body), "cozy/tiny") {
		t.Fatalf("managed rental leaked work identity in reason/body: create=%q delete=%q body=%v",
			gotCreateReason, gotDeleteReason, body)
	}
	wantLines := map[string]bool{
		"rentals: 0 remote machines running · $0.00/hour of $1.00/hour": false,
		"rentals: 1 remote machine running · $0.30/hour of $1.00/hour":  false,
	}
	for _, event := range events {
		if line, ok := event.Payload["line"].(string); ok {
			if _, wanted := wantLines[line]; wanted {
				wantLines[line] = true
			}
		}
	}
	for line, seen := range wantLines {
		if !seen {
			t.Fatalf("managed lifecycle omitted %q: %+v", line, events)
		}
	}
	code, out = runCozy(t, root, "run", weightlessRef+"/tile", "size=32", "seed=7",
		"--model", "model=cozy/tiny@1.0.0#"+modelManifest,
		"--rental", "--detach", "--idempotency-key", "managed-e2e", "--json")
	mu.Lock()
	gotPosts = posts
	mu.Unlock()
	if code != 0 || !strings.Contains(out, `"changed":false`) || gotPosts != 1 {
		t.Fatalf("exact replay purchased again [exit %d posts=%d]\n%s", code, gotPosts, out)
	}

	mu.Lock()
	posts, deletes, rentalRequest, createReason, deleteReason = 0, 0, nil, "", ""
	activeRentalID = "pr-managed-job"
	mu.Unlock()
	code, out = runCozy(t, root, "run", weightlessRef+"/tile_job", "size=32", "seed=7",
		"--rental", "--detach", "--idempotency-key", "managed-job", "--json")
	if code != 0 || !strings.Contains(out, `"changed":true`) {
		t.Fatalf("remote job submission [exit %d]\n%s\ndaemon:\n%s", code, out, daemonOutput.String())
	}
	deadline = time.Now().Add(10 * time.Second)
	request = nil
	for time.Now().Before(deadline) {
		store, problem = records.Open(filepath.Join(root, "records.db"))
		fatal(t, problem)
		rows, rowsProblem := store.RequestsOfKind("job", "", 20)
		fatal(t, rowsProblem)
		for i := range rows {
			if rows[i].IdemKey == "managed-job" {
				row := rows[i]
				request = &row
			}
		}
		rentalRow, _ := store.RentalRow("pr-managed-job")
		store.Close()
		mu.Lock()
		done := posts == 1 && deletes == 1 && rentalRow == nil && request != nil && request.State == "failed"
		mu.Unlock()
		if done {
			break
		}
		time.Sleep(25 * time.Millisecond)
	}
	mu.Lock()
	gotPosts, gotDeletes, body = posts, deletes, rentalRequest
	mu.Unlock()
	if request == nil || request.Worker != "pr-managed-job" || request.PlanID != jobDescriptorID || request.Kind != "job" ||
		request.Release != "1.0.0" || request.PackageRevisionDigest != releaseDigest ||
		request.JobGPUCount != 0 || len(request.Models) != 0 || gotPosts != 1 || gotDeletes != 1 ||
		strings.Contains(fmt.Sprint(body), weightlessRef) || strings.Contains(fmt.Sprint(body), "tile_job") {
		t.Fatalf("remote job lifecycle request=%+v posts=%d deletes=%d body=%v",
			request, gotPosts, gotDeletes, body)
	}
	code, out = runCozy(t, root, "run", weightlessRef+"/tile_job", "size=32", "seed=7",
		"--rental", "--detach", "--idempotency-key", "managed-job", "--json")
	mu.Lock()
	gotPosts = posts
	mu.Unlock()
	if code != 0 || !strings.Contains(out, `"changed":false`) || gotPosts != 1 {
		t.Fatalf("remote job replay purchased again [exit %d posts=%d]\n%s", code, gotPosts, out)
	}
}
