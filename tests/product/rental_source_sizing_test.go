package producttest

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/hub"
)

// `cozy rental new --model <provider source>` sizes the pod for an ingest of a source that is
// not on the Hub yet: the same planning a rented upload runs (pinning, reviewed profiles,
// header preflight) declares the source's planned bytes, alongside any explicit --disk-gb.
// The stand-in Hub captures each paid request and refuses it; nothing is rented.
func TestManualRentalSizesDiskFromAProviderSource(t *testing.T) {
	const (
		civitaiAPI = "https://civitai.com/api/v1/model-versions/128078"
		hfRevision = "335001fb9e5455d68a0caa18ec2e319072150328"
		hfSource   = "hf://alibaba-pai/MiniMax-H3-Acc-LoRAs@" + hfRevision
		hfTree     = "https://huggingface.co/api/models/alibaba-pai/MiniMax-H3-Acc-LoRAs/tree/" + hfRevision
	)
	probe := &http.Client{Timeout: 20 * time.Second}
	fetch := func(url string, into any) {
		t.Helper()
		response, err := probe.Get(url)
		if err != nil {
			t.Skipf("provider unreachable from this runner: %v", err)
		}
		defer response.Body.Close()
		must(t, json.NewDecoder(response.Body).Decode(into))
	}
	var version struct {
		Files []struct {
			Name    string  `json:"name"`
			Primary bool    `json:"primary"`
			SizeKB  float64 `json:"sizeKB"`
		} `json:"files"`
	}
	fetch(civitaiAPI, &version)
	var civitaiBytes int64
	for _, file := range version.Files {
		if file.Primary && strings.HasSuffix(file.Name, ".safetensors") {
			civitaiBytes = int64(file.SizeKB * 1024)
		}
	}
	var tree []struct {
		Path string `json:"path"`
		Size int64  `json:"size"`
	}
	fetch(hfTree, &tree)
	var carrierBytes int64
	for _, file := range tree {
		if strings.HasSuffix(file.Path, ".safetensors") {
			carrierBytes += file.Size
		}
	}
	if civitaiBytes == 0 || carrierBytes == 0 {
		t.Fatalf("provider listings changed: civitai=%d hf=%d", civitaiBytes, carrierBytes)
	}

	var mu sync.Mutex
	var posts [][]byte
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/rentals", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"rentals":[]}`))
	})
	mux.HandleFunc("GET /v1/rental-skus", func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode([]map[string]any{{"name": "h200", "accelerator_model": "NVIDIA H200",
			"compute_capability": "9.0", "vram_gb": 141, "base_worker_profile": "torch2.13.0-cu130-cp312-linux-x86",
			"widths": []map[string]any{{"accelerator_count": 1, "price_usd_micros_per_hour": 1_000_000, "storage_usd_micros_per_hour": 100_000}}}})
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
	mux.HandleFunc("POST /v1/rental-quotes", listedRentalQuote(mux))
	server := httptest.NewServer(mux)
	defer server.Close()
	must(t, os.MkdirAll(scratchBase, 0o755))
	root, err := os.MkdirTemp(scratchBase, "source-sizing-")
	must(t, err)
	t.Cleanup(func() { _ = removeAllForce(root) })
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+server.URL+
		"\ntensorhub_token: source-sizing-test\ndaemon:\n  idle_shutdown_s: 0\n"), 0o600))
	startDaemonProcess(t, root)

	declared := func(key string, args ...string) hub.RentalRequest {
		t.Helper()
		mu.Lock()
		before := len(posts)
		mu.Unlock()
		code, out := runCozy(t, root, append([]string{"rental", "new", "h200", "--idempotency-key", key}, append(args, "--json")...)...)
		mu.Lock()
		defer mu.Unlock()
		if code == 0 || !strings.Contains(out, "proof.no_paid_create") || len(posts) != before+1 {
			t.Fatalf("rental new %v did not reach the paid boundary [exit %d]: %s", args, code, out)
		}
		request, problem := hub.ParseRentalRequestBytes(posts[len(posts)-1])
		fatal(t, problem)
		return request
	}

	civitai := declared("sized-civitai", "--model", "civitai://128078", "--source-profile", "civitai/sdxl/single-file/1")
	if civitai.PlannedSourceBytes != civitaiBytes || len(civitai.ServingModels) != 0 || civitai.ContainerDiskGB != 0 {
		t.Fatalf("a Civitai source declared %d planned bytes, want its %d-byte carrier: %+v", civitai.PlannedSourceBytes, civitaiBytes, civitai)
	}
	composed := declared("sized-hf", "--model", hfSource,
		"--source-profile", "hf/minimax-h3/pdd-fl2va-bf16/1", "--source-profile", "hf/minimax-h3/pdd-ref2va-bf16/1")
	if composed.PlannedSourceBytes != carrierBytes {
		t.Fatalf("two composed profiles declared %d planned bytes, want the %d bytes of their carriers", composed.PlannedSourceBytes, carrierBytes)
	}
	single := declared("sized-hf-one", "--model", hfSource, "--source-profile", "hf/minimax-h3/pdd-fl2va-bf16/1")
	if single.PlannedSourceBytes <= 0 || single.PlannedSourceBytes >= carrierBytes {
		t.Fatalf("one profile declared %d planned bytes, want only its own carrier of %d", single.PlannedSourceBytes, carrierBytes)
	}
	both := declared("sized-disk", "--model", "civitai://128078", "--source-profile", "civitai/sdxl/single-file/1", "--disk-gb=900")
	if both.ContainerDiskGB != 900 || both.PlannedSourceBytes != civitaiBytes {
		t.Fatalf("a source with --disk-gb=900 declared %+v", both)
	}

	for _, args := range [][]string{
		{"--source-profile", "civitai/sdxl/single-file/1"},
		{"--model", "civitai://128078", "--model", hfSource, "--source-profile", "civitai/sdxl/single-file/1"},
	} {
		code, out := runCozy(t, root, append([]string{"rental", "new", "h200"}, append(args, "--json")...)...)
		if code == 0 || strings.Contains(out, "proof.no_paid_create") {
			t.Fatalf("rental new %v was not refused before paying [exit %d]: %s", args, code, out)
		}
	}
}
