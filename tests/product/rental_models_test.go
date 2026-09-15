package producttest

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/records"
)

// The actual command reaches the paid request boundary, but this local Hub
// refuses every POST. No provider or paid rental is involved in this proof.
func TestManualRentalDeclaresExactModelsAndReplaysPinnedBytes(t *testing.T) {
	var mu sync.Mutex
	var posts [][]byte
	lookups := 0
	changed := false
	stockOut := false
	catalogLookups := 0
	first, second := "sha256:"+strings.Repeat("1", 64), "sha256:"+strings.Repeat("2", 64)
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/rentals", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"rentals":[]}`))
	})
	mux.HandleFunc("GET /v1/rental-skus", func(w http.ResponseWriter, _ *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		catalogLookups++
		if stockOut {
			_ = json.NewEncoder(w).Encode([]any{})
			return
		}
		_ = json.NewEncoder(w).Encode([]map[string]any{{"name": "h200", "accelerator_model": "NVIDIA H200",
			"accelerator_count": 1, "compute_capability": "9.0", "vram_gb": 141, "minimum_ram_per_gpu_gb": 128, "price_usd_micros_per_hour": 1_000_000,
			"storage_usd_micros_per_hour": 100_000, "base_worker_profile": "torch2.13.0-cu130-cp312-linux-x86"}})
	})
	mux.HandleFunc("GET /v1/models/resolve", func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		lookups++
		ref, lane := r.URL.Query().Get("ref"), r.URL.Query().Get("lane")
		model, release, _ := strings.Cut(ref, "@")
		manifest := first
		if model == "proof/shared" {
			manifest = second
		}
		if changed {
			manifest = "sha256:" + strings.Repeat("3", 64)
		}
		if model == "proof/wrong" {
			model = "proof/another"
		}
		if model == "proof/absent" {
			w.WriteHeader(http.StatusNotFound)
			_, _ = w.Write([]byte(`{"error":{"code":"model.not_found","message":"absent"}}`))
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"model": model, "release": release, "lane": lane,
			"manifest_id": manifest, "bytes": int64(333_000_000_000), "objects": 100})
	})
	mux.HandleFunc("POST /v1/rentals", func(w http.ResponseWriter, r *http.Request) {
		raw, err := io.ReadAll(r.Body)
		if err != nil {
			t.Error(err)
		}
		mu.Lock()
		posts = append(posts, raw)
		mu.Unlock()
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte(`{"error":{"code":"proof.no_paid_create","message":"request captured; no rental created"}}`))
	})
	server := httptest.NewServer(mux)
	defer server.Close()
	base := filepath.Join(os.TempDir(), "cozy-product-test")
	must(t, os.MkdirAll(base, 0o755))
	root, err := os.MkdirTemp(base, "manual-models-")
	must(t, err)
	t.Cleanup(func() { _ = os.RemoveAll(root) })
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte(
		"tensorhub_url: "+server.URL+"\ntensorhub_token: rental-models-test\n"+
			"rentals:\n  max_hourly_spend_usd: 20.00\n  idle_release_s: 0\n"+
			"daemon:\n  idle_shutdown_s: 0\n"), 0o600))
	const sshKey = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAINdamAGCsQq31Uv+08lkBzoO4XLz2qYjJa8CGmj3B1Ea fixture"
	keyPath := filepath.Join(root, "operator.pub")
	must(t, os.WriteFile(keyPath, []byte(sshKey+"\n"), 0600))
	startDaemonProcess(t, root)
	runRental := func(args ...string) (int, string) {
		return runCozy(t, root, append(args, "--json")...)
	}
	args := []string{"rental", "new", "h200", "--idempotency-key", "manual-models-proof",
		"--development", "--ssh-public-key", keyPath,
		"--model", "proof/shared@1.0.0/bf16", "--model", "proof/h3@1.2.0/full",
		"--model", "proof/shared@1.0.0/bf16"}
	code, output := runRental(args...)
	if code == 0 || !strings.Contains(output, "proof.no_paid_create") {
		t.Fatalf("command did not reach isolated paid boundary [exit %d]: %s", code, output)
	}
	mu.Lock()
	if len(posts) != 1 {
		t.Fatalf("paid request count=%d", len(posts))
	}
	body := append([]byte(nil), posts[0]...)
	initialLookups := lookups
	mu.Unlock()
	request, problem := hub.ParseRentalRequestBytes(body)
	fatal(t, problem)
	if request.Development == nil || request.Development.SSHPublicKey != sshKey {
		t.Fatalf("explicit development key not retained: %+v", request.Development)
	}
	if request.SKU != "h200" || request.PlannedSourceBytes != 0 || len(request.ServingModels) != 2 {
		t.Fatalf("manual workload lost its exact models: %+v", request)
	}
	if request.ServingModels[0] != (hub.ServingModel{Model: "proof/h3", Release: "1.2.0", Lane: "full", Manifest: first}) ||
		request.ServingModels[1] != (hub.ServingModel{Model: "proof/shared", Release: "1.0.0", Lane: "bf16", Manifest: second}) {
		t.Fatalf("wrong sorted checkpoint pins: %+v", request.ServingModels)
	}
	if strings.Contains(string(body), "333000000000") || strings.Contains(string(body), "container_disk") {
		t.Fatalf("Creator estimated model bytes instead of declaring identity: %s", body)
	}
	st, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer st.Close()
	op, problem := st.RentalOperation("manual-models-proof")
	fatal(t, problem)
	if op == nil || !bytes.Equal(op.RequestBody, body) {
		t.Fatalf("the sent declaration was not persisted first: %+v", op)
	}
	mu.Lock()
	changed = true // later lane retarget must not change a paid replay
	stockOut = true
	initialCatalogLookups := catalogLookups
	mu.Unlock()
	for _, replayArgs := range [][]string{args, {"rental", "new", "h200", "--idempotency-key", "manual-models-proof"}} {
		code, output = runRental(replayArgs...)
		if code == 0 || !strings.Contains(output, "proof.no_paid_create") {
			t.Fatalf("exact replay refused before the captured boundary [exit %d]: %s", code, output)
		}
	}
	mu.Lock()
	if catalogLookups != initialCatalogLookups || lookups != initialLookups || len(posts) != 3 || !bytes.Equal(posts[1], body) || !bytes.Equal(posts[2], body) {
		t.Fatalf("retry re-resolved or changed paid bytes: lookups=%d/%d posts=%d", lookups, initialLookups, len(posts))
	}
	stockOut = false
	mu.Unlock()
	code, output = runRental("rental", "new", "another-gpu", "--idempotency-key", "manual-models-proof")
	if code == 0 || !strings.Contains(output, "rental.idempotency_conflict") {
		t.Fatalf("changed SKU not refused before replay [exit %d]: %s", code, output)
	}
	code, output = runRental("rental", "new", "h200", "--idempotency-key", "manual-models-proof",
		"--model", "proof/h3@1.2.0/changed")
	if code == 0 || !strings.Contains(output, "rental.idempotency_conflict") {
		t.Fatalf("changed serving set not refused [exit %d]: %s", code, output)
	}
	for _, ref := range []string{"proof/h3", "proof/h3@1.2.0", "local/h3@1.2.0/full",
		"proof/absent@1.0.0/full", "proof/wrong@1.0.0/full", "proof/h3@1.2.0/full#" + first} {
		code, output = runRental("rental", "new", "h200", "--model", ref)
		if code == 0 {
			t.Fatalf("invalid/mismatched --model %s admitted: %s", ref, output)
		}
	}
	mu.Lock()
	if len(posts) != 3 {
		t.Fatalf("changed/invalid model reached paid create: %d requests", len(posts))
	}
	mu.Unlock()
	must(t, os.WriteFile(keyPath, []byte(sshKey+" changed-comment\n"), 0600))
	code, output = runRental(args...)
	if code == 0 || !strings.Contains(output, "rental.idempotency_conflict") {
		t.Fatalf("changed development access was not refused: %s", output)
	}
	mu.Lock()
	if len(posts) != 3 {
		t.Fatalf("changed key reached paid create: %d", len(posts))
	}
	mu.Unlock()
	code, output = runRental("rental", "new", "--model", "proof/h3@1.2.0/full")
	if code == 0 || !strings.Contains(output, "require a GPU SKU") {
		t.Fatalf("--model without GPU was not refused [exit %d]: %s", code, output)
	}
	code, output = runRental("rental", "new", "h200", "--idempotency-key", "manual-undeclared")
	if code == 0 || !strings.Contains(output, "proof.no_paid_create") {
		t.Fatalf("undeclared rental did not reach isolated paid boundary [exit %d]: %s", code, output)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(posts) != 4 || bytes.Contains(posts[3], []byte("serving_models")) || !bytes.Contains(posts[3], []byte("development")) {
		t.Fatalf("undeclared manual rental changed its request shape: %q", posts)
	}
}
