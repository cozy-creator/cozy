package producttest

import (
	"encoding/json"
	"net/http"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/cozy-creator/cozy/internal/hub"
)

func TestRentalMachineExclusionsAreCanonicalProviderScopedAndOptional(t *testing.T) {
	for _, tc := range []struct {
		provider string
		ids      []string
	}{{"", []string{"1"}}, {"runpod", []string{"1"}}, {"vast", []string{"-1"}}, {"vast", []string{"01"}}, {"vast", []string{"150864/x"}}, {"vast", []string{"0"}}} {
		if _, problem := hub.RentalMachineExclusions(tc.provider, tc.ids); problem == nil {
			t.Fatalf("accepted invalid selection: %+v", tc)
		}
	}
	ids, problem := hub.RentalMachineExclusions("vast", []string{"150864", "41285", "150864"})
	fatal(t, problem)
	if !slices.Equal(ids, []string{"150864", "41285"}) {
		t.Fatalf("not canonical: %v", ids)
	}
	body, problem := hub.RentalRequestBytes("twine", "cpu", 1, strings.Repeat("a", 64), "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", hub.DeclaredWorkload{}, nil, "", "vast", ids...)
	fatal(t, problem)
	got, problem := hub.ParseRentalRequestBytes(body)
	fatal(t, problem)
	if !slices.Equal(got.ExcludedProviderMachines, ids) {
		t.Fatalf("lost selection: %s", body)
	}
}

func TestRentalMachineExclusionsReachQuoteAndPaidIntentAndSurviveReplay(t *testing.T) {
	root, _, stand := rentalEndRoot(t, "machine-exclude")
	defer runCozy(t, root, "down")
	stand.setSKUs(map[string]any{"name": "rtx-4060", "accelerator_model": "NVIDIA GeForce RTX 4060", "accelerator_count": 1, "base_worker_profile": "proof", "compute_capability": "8.9", "vram_gb": 8, "price_usd_micros_per_hour": 260000})
	var mu sync.Mutex
	var quote, paid []string
	posts := 0
	stand.quote = func(req map[string]any) (int, string) {
		raw, _ := json.Marshal(req["excluded_provider_machines"])
		mu.Lock()
		_ = json.Unmarshal(raw, &quote)
		mu.Unlock()
		body, _ := json.Marshal(map[string]any{"price_usd_micros_per_hour": 260000, "maximum_total_hourly_rate_usd_micros": 260000, "container_disk_gb": 321, "excluded_provider_machines": req["excluded_provider_machines"]})
		return http.StatusOK, string(body)
	}
	stand.rent = func(req map[string]any) map[string]any {
		raw, _ := json.Marshal(req["excluded_provider_machines"])
		mu.Lock()
		_ = json.Unmarshal(raw, &paid)
		posts++
		mu.Unlock()
		return map[string]any{"rental_id": "pr-machine-exclude", "name": req["name"], "state": "acquiring", "requested_accelerator_model": "NVIDIA GeForce RTX 4060", "accelerator_count": 1, "hourly_rate_usd_micros": 260000}
	}
	args := []string{"rental", "new", "rtx-4060", "--provider=vast", "--idempotency-key=machine-exclude", "--development=false", "--json", "--timeout=1s"}
	_, out := runCozy(t, root, append(args, "--exclude-provider-machine=41285", "--exclude-provider-machine=150864", "--exclude-provider-machine=41285")...)
	mu.Lock()
	gotPosts := posts
	gotQuote := append([]string(nil), quote...)
	gotPaid := append([]string(nil), paid...)
	mu.Unlock()
	want := []string{"150864", "41285"}
	if gotPosts != 1 || !slices.Equal(gotQuote, want) || !slices.Equal(gotPaid, want) {
		t.Fatalf("quote/paid selection missing: posts%d quote%v paid%v %s", gotPosts, gotQuote, gotPaid, out)
	}
	// Omitted selection resumes the retained intent. A changed one cannot buy again.
	_, out = runCozy(t, root, args...)
	if strings.Contains(out, "idempotency_conflict") {
		t.Fatalf("omitted exclusions did not resume: %s", out)
	}
	code, out := runCozy(t, root, append(args, "--exclude-provider-machine=7")...)
	if code == 0 || !strings.Contains(out, "idempotency_conflict") {
		t.Fatalf("changed exclusion accepted: %d %s", code, out)
	}
}

func TestRentalMachineExclusionsRefuseUnawareHubBeforePaidPost(t *testing.T) {
	root, _, stand := rentalEndRoot(t, "exclude-old-hub")
	defer runCozy(t, root, "down")
	stand.setSKUs(map[string]any{"name": "rtx-4060", "accelerator_model": "NVIDIA GeForce RTX 4060", "accelerator_count": 1, "base_worker_profile": "proof", "compute_capability": "8.9", "vram_gb": 8, "price_usd_micros_per_hour": 260000})
	var mu sync.Mutex
	posts := 0
	stand.rent = func(req map[string]any) map[string]any { mu.Lock(); posts++; mu.Unlock(); return map[string]any{} }
	code, out := runCozy(t, root, "rental", "new", "rtx-4060", "--provider=vast", "--exclude-provider-machine=150864", "--development=false", "--json")
	mu.Lock()
	defer mu.Unlock()
	if code == 0 || !strings.Contains(out, "rental.machine_exclusion_unsupported") || posts != 0 {
		t.Fatalf("unaware Hub reached paid POST: posts%d exit%d %s", posts, code, out)
	}
}

func TestRentalMachineExclusionsReconfirmUnsentResumeAgainstOlderHub(t *testing.T) {
	root, _, stand := rentalEndRoot(t, "exclude-resume-old-hub")
	defer runCozy(t, root, "down")
	stand.setSKUs(map[string]any{"name": "rtx-4060", "accelerator_model": "NVIDIA GeForce RTX 4060", "accelerator_count": 1, "base_worker_profile": "proof", "compute_capability": "8.9", "vram_gb": 8, "price_usd_micros_per_hour": 260000})
	var mu sync.Mutex
	aware, posts := true, 0
	stand.quote = func(req map[string]any) (int, string) {
		mu.Lock()
		confirmed := aware
		mu.Unlock()
		body := map[string]any{"price_usd_micros_per_hour": 260000, "maximum_total_hourly_rate_usd_micros": 260000, "container_disk_gb": 321}
		if confirmed {
			body["excluded_provider_machines"] = req["excluded_provider_machines"]
		}
		raw, _ := json.Marshal(body)
		return 200, string(raw)
	}
	original := stand.server.Config.Handler
	stand.server.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == "POST" && r.URL.Path == "/v1/rentals" {
			mu.Lock()
			posts++
			mu.Unlock()
			w.WriteHeader(503)
			_, _ = w.Write([]byte(`{"error":{"code":"proof.no_create","message":"no paid request accepted"}}`))
			return
		}
		original.ServeHTTP(w, r)
	})
	args := []string{"rental", "new", "rtx-4060", "--provider=vast", "--idempotency-key=exclude-unsent", "--development=false", "--json"}
	_, out := runCozy(t, root, append(args, "--exclude-provider-machine=150864")...)
	if !strings.Contains(out, "proof.no_create") {
		t.Fatalf("did not retain unsent intent: %s", out)
	}
	mu.Lock()
	before := posts
	aware = false
	mu.Unlock()
	code, out := runCozy(t, root, args...)
	mu.Lock()
	defer mu.Unlock()
	if code == 0 || !strings.Contains(out, "rental.machine_exclusion_unsupported") || posts != before {
		t.Fatalf("older Hub received resumed paid POST: posts%d->%d exit%d %s", before, posts, code, out)
	}
}
