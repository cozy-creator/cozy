package producttest

import (
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/secret"
)

// `--provider` is carried in the exact request bytes, and an ask with no provider is
// byte-identical to one authored before providers existed.
func TestRentalRequestNamesItsProviderOnlyWhenAsked(t *testing.T) {
	key := "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
	vast, problem := hub.RentalRequestBytes("twine", "rtx-3070", 1, strings.Repeat("ab", 32), key, hub.DeclaredWorkload{}, nil, "", "vast")
	fatal(t, problem)
	if request, problem := hub.ParseRentalRequestBytes(vast); problem != nil || request.Provider != "vast" {
		t.Fatalf("provider lost from the persisted ask: %+v %v", request, problem)
	}
	plain, problem := hub.RentalRequestBytes("twine", "rtx-3070", 1, strings.Repeat("ab", 32), key, hub.DeclaredWorkload{}, nil, "", "")
	fatal(t, problem)
	if strings.Contains(string(plain), "provider") {
		t.Fatalf("an ask without --provider names one: %s", plain)
	}
	if _, problem := hub.RentalRequestBytes("twine", "rtx-3070", 1, strings.Repeat("ab", 32), key, hub.DeclaredWorkload{}, nil, "", "Vast AI"); problem == nil {
		t.Fatal("a malformed provider was accepted")
	}
}

// A named provider lists only that provider's machines. A hub that predates providers
// ignores the query and answers its default listing, which names none: the ask then
// finds nothing rather than silently buying from the default marketplace.
func TestRentalSKUsForANamedProvider(t *testing.T) {
	product := `{"name":"rtx-3070","accelerator_model":"NVIDIA GeForce RTX 3070","compute_capability":"8.6","vram_gb":8,` +
		`"widths":[{"accelerator_count":1,"price_usd_micros_per_hour":100000}]`
	for _, tc := range []struct {
		hub, provider, query string
		listed               int
	}{
		{`[` + product + `,"provider":"vast"}]`, "vast", "vast", 1},
		{`[` + product + `}]`, "vast", "vast", 0},
		{`[` + product + `,"provider":"runpod"}]`, "", "", 1},
	} {
		var asked string
		mux := http.NewServeMux()
		mux.HandleFunc("GET /v1/rental-skus", func(w http.ResponseWriter, r *http.Request) {
			asked = r.URL.Query().Get("provider")
			_, _ = w.Write([]byte(tc.hub))
		})
		server := httptest.NewServer(mux)
		client := hub.New(config.Config{HubURL: server.URL, HubToken: secret.New("proof")}, "rental-provider")
		skus, problem := client.RentalSKUs(t.Context(), tc.provider)
		server.Close()
		fatal(t, problem)
		if asked != tc.query || len(skus) != tc.listed || tc.listed == 1 && skus[0].Provider == "" && tc.provider != "" {
			t.Fatalf("provider %q against %s: asked %q, listed %+v", tc.provider, tc.hub, asked, skus)
		}
	}
}

// The development override is hidden from help, and it forces the marketplace: the catalog
// is asked for that provider's machines and the order names it. Without it, the order names
// no provider and Tensorhub places the rental.
func TestHiddenProviderOverrideForcesTheMarketplace(t *testing.T) {
	root, _, stand := rentalEndRoot(t, "rental-provider")
	if code, out := runCozy(t, root, "rental", "new", "--help"); code != 0 || strings.Contains(out, "--provider") {
		t.Fatalf("the development override shows in help [exit %d]\n%s", code, out)
	}
	stand.publishListing()
	stand.setSKUs(map[string]any{
		"name": "rtx-4060", "accelerator_model": "NVIDIA GeForce RTX 4060", "accelerator_count": 1,
		"base_worker_profile": "proof", "compute_capability": "8.9", "vram_gb": 8,
		"minimum_ram_per_gpu_gb": 16, "price_usd_micros_per_hour": 260_000,
	})
	ordered := map[string]any{}
	stand.mu.Lock()
	stand.rent = func(request map[string]any) map[string]any {
		ordered[request["name"].(string)] = request["provider"]
		return map[string]any{"rental_id": "pr-provider-" + request["name"].(string), "name": request["name"], "state": "failed",
			"requested_accelerator_model": "NVIDIA GeForce RTX 4060", "accelerator_count": 1, "hourly_rate_usd_micros": 260_000}
	}
	stand.mu.Unlock()
	asked := func() []string {
		stand.mu.Lock()
		defer stand.mu.Unlock()
		return append([]string(nil), stand.providers...)
	}

	if code, out := runCozy(t, root, "rental", "new", "--provider", "vast", "--json"); code != 0 || !strings.Contains(out, "rtx-4060") ||
		!slices.Contains(asked(), "vast") {
		t.Fatalf("the override's catalog was not that provider's [exit %d] asked %v\n%s", code, asked(), out)
	}
	_, _, _ = runCozyStreams(t, root, "rental", "new", "rtx-4060", "--provider", "vast", "--json", "--timeout=10s")
	_, _, _ = runCozyStreams(t, root, "rental", "new", "rtx-4060", "--json", "--timeout=10s")
	stand.mu.Lock()
	defer stand.mu.Unlock()
	providers := map[any]int{}
	for _, provider := range ordered {
		providers[provider]++
	}
	if len(ordered) != 2 || providers["vast"] != 1 || providers[nil] != 1 {
		t.Fatalf("the orders did not name the forced provider, and only it: %v", ordered)
	}
}
