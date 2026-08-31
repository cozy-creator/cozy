package producttest

import (
	"bytes"
	"encoding/json"
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
	var mu sync.Mutex
	var descriptor []byte
	var descriptorDigest string
	var rentalRequest map[string]any
	posts, deletes := 0, 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		switch {
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
		case r.Method == http.MethodGet && r.URL.Path == "/v1/rental-skus":
			_ = json.NewEncoder(w).Encode([]map[string]any{
				{"name": "expensive", "accelerator_model": "GPU X", "compute_capability": "9.0", "vram_gb": 80, "price_usd_micros_per_hour": 900_000},
				{"name": "cheap", "accelerator_model": "GPU C", "compute_capability": "8.9", "vram_gb": 24, "price_usd_micros_per_hour": 300_000},
			})
		case r.Method == http.MethodPost && r.URL.Path == "/v1/rentals":
			posts++
			if err := json.NewDecoder(r.Body).Decode(&rentalRequest); err != nil {
				t.Error(err)
			}
			w.WriteHeader(http.StatusAccepted)
			_ = json.NewEncoder(w).Encode(map[string]any{
				"rental_id": "pr-managed-e2e", "state": "pending_acquisition",
				"requested_accelerator_model": "GPU C", "hourly_rate_usd_micros": 300_000,
			})
		case r.Method == http.MethodDelete && r.URL.Path == "/v1/rentals/pr-managed-e2e":
			deletes++
			w.WriteHeader(http.StatusNoContent)
		case r.Method == http.MethodGet && r.URL.Path == "/v1/rentals/pr-managed-e2e":
			state := "ready"
			if deletes > 0 {
				state = "released"
			}
			answer := map[string]any{
				"rental_id": "pr-managed-e2e", "state": state,
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
			descriptorDocument["entrypoints"] = []any{row}
		}
	}
	descriptor, _ = json.Marshal(descriptorDocument)
	decoded, problem := launch.DecodeDescriptor(descriptor)
	fatal(t, problem)
	descriptorDigest = decoded.Digest
	published := local
	published.ID = "published-managed-e2e"
	published.SourceKind = "tensorhub"
	published.SourceRef = weightlessRef + "@1.0.0"
	published.SourceDigest = releaseDigest
	published.Version = "1.0.0"
	published.Verified = true
	store, problem := records.Open(filepath.Join(root, "records.db"))
	fatal(t, problem)
	_, problem = store.Activate(published)
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
	mu.Unlock()
	if request == nil || request.Worker != "pr-managed-e2e" || gotPosts != 1 || gotDeletes != 1 ||
		body["sku"] != "cheap" || body["max_cost_usd_micros"] != nil || body["max_duration_seconds"] != nil {
		t.Fatalf("managed lifecycle request=%+v posts=%d deletes=%d body=%v", request, gotPosts, gotDeletes, body)
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
		"--rental", "--detach", "--idempotency-key", "managed-e2e", "--json")
	mu.Lock()
	gotPosts = posts
	mu.Unlock()
	if code != 0 || !strings.Contains(out, `"changed":false`) || gotPosts != 1 {
		t.Fatalf("exact replay purchased again [exit %d posts=%d]\n%s", code, gotPosts, out)
	}
}
