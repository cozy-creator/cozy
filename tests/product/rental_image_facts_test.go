package producttest

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/secret"
)

func TestRentalImageFactsAdvertisePublicModelOriginWithoutPackagePublication(t *testing.T) {
	calls := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		if r.URL.Path != "/v1/rentals/pr-test/image-inventory" || r.Header.Get("Authorization") != "Bearer inventory-token" || r.URL.RawQuery != "" {
			t.Errorf("unexpected metadata request %s", r.URL)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"image_inventory": map[string]any{"format": "tensorhub.image_inventory/1"}, "public_origin": "https://private-hub-tunnel.example"})
	}))
	defer server.Close()
	client := hub.New(config.Config{HubURL: server.URL, HubToken: secret.New("inventory-token")}, "private-model-origin-test")
	facts, problem := client.RentalImageInventory(context.Background(), "pr-test")
	fatal(t, problem)
	if calls != 1 || facts.PublicOrigin != "https://private-hub-tunnel.example" || len(facts.ImageInventory) == 0 {
		t.Fatal("rental byte origin was lost", facts, calls)
	}
}
