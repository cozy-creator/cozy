package rental

// The facts fetch, driven through the production path: a hub answering the
// rental-scoped prepare-facts route exactly as tensorhub's privaterental.go
// spells it, read by the real hub client and admitted onto the wire shape.

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/secret"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

func factsServer(t *testing.T, inventory any) *hub.Client {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/rentals/pr-facts-01/prepare-facts" ||
			r.URL.Query().Get("package") != "acme/anima" || r.URL.Query().Get("release") != "1.2.0" ||
			!strings.HasPrefix(r.Header.Get("Authorization"), "Bearer ") {
			w.WriteHeader(http.StatusNotFound)
			_, _ = w.Write([]byte(`{"error":{"code":"rental.not_found","message":"no such rental"}}`))
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"application":      "comfyui",
			"model_slot_paths": []string{"unet", "vae"},
			"image_inventory":  inventory,
			"locked_requirements": "--index-url https://pypi.org/simple\n" +
				"--extra-index-url https://hub.example/v1/index/acme/simple/\n\n" +
				"anima==1.2.0 --hash=sha256:" + strings.Repeat("11", 32) + "\n",
		})
	}))
	t.Cleanup(server.Close)
	cfg := config.Config{HubURL: server.URL, HubToken: secret.New("test-token")}
	return hub.New(cfg, "cozy-test")
}

func registeredInventory() map[string]any {
	return map[string]any{
		"format": "tensorhub.image_inventory/1", "profile": "python3.12-cpu-linux-x86",
		"python": "3.12.8",
		"distributions": []map[string]string{
			{"name": "numpy", "version": "2.1.0"},
			{"name": "torch", "version": "2.6.0"},
		},
	}
}

func fetchFacts(t *testing.T, client *hub.Client) (orchestrator.PrepareFacts, *struct {
	Code string
	Text string
},
) {
	t.Helper()
	source := PrepareFactsSource(client)
	connection := &orchestrator.WorkerConnection{RentalID: "pr-facts-01"}
	facts, problem := source(context.Background(), connection,
		&pb.DownloadPackageRef{Package: "acme/anima", Release: "1.2.0"})
	if problem != nil {
		return facts, &struct {
			Code string
			Text string
		}{problem.ErrName(), problem.Message}
	}
	return facts, nil
}

func TestPrepareFactsSourceAnswersTheCallShape(t *testing.T) {
	facts, refused := fetchFacts(t, factsServer(t, registeredInventory()))
	if refused != nil {
		t.Fatalf("facts refused: %s: %s", refused.Code, refused.Text)
	}
	if facts.Application != "comfyui" {
		t.Fatalf("application = %q", facts.Application)
	}
	if len(facts.ModelSlotPaths) != 2 || facts.ModelSlotPaths[0] != "unet" || facts.ModelSlotPaths[1] != "vae" {
		t.Fatalf("model slot paths = %v", facts.ModelSlotPaths)
	}
	inventory := facts.ImageInventory
	if inventory.GetProfile() != "python3.12-cpu-linux-x86" || inventory.GetPython() != "3.12.8" ||
		len(inventory.GetDistributions()) != 2 ||
		inventory.Distributions[0].Distribution != "numpy" || inventory.Distributions[0].Version != "2.1.0" ||
		inventory.Distributions[1].Distribution != "torch" {
		t.Fatalf("inventory = %v", inventory)
	}
	rows := string(facts.LockedRequirements)
	if !strings.HasPrefix(rows, "--index-url https://pypi.org/simple\n") ||
		!strings.Contains(rows, "anima==1.2.0 --hash=sha256:") {
		t.Fatalf("locked requirements = %q", rows)
	}
}

func TestPrepareFactsSourceRefusesAbsentInventory(t *testing.T) {
	_, refused := fetchFacts(t, factsServer(t, nil))
	if refused == nil || refused.Code != "rental.prepare_facts_invalid" ||
		!strings.Contains(refused.Text, "no registered inventory") {
		t.Fatalf("absent inventory not refused typed: %+v", refused)
	}
}

func TestPrepareFactsSourceRefusesForeignInventoryFormat(t *testing.T) {
	inventory := registeredInventory()
	inventory["format"] = "tensorhub.image_inventory/2"
	_, refused := fetchFacts(t, factsServer(t, inventory))
	if refused == nil || refused.Code != "rental.prepare_facts_invalid" ||
		!strings.Contains(refused.Text, "tensorhub.image_inventory/1") {
		t.Fatalf("foreign format not refused typed: %+v", refused)
	}
}

func TestPrepareFactsSourceRelaysTheHubRefusal(t *testing.T) {
	client := factsServer(t, registeredInventory())
	source := PrepareFactsSource(client)
	_, problem := source(context.Background(),
		&orchestrator.WorkerConnection{RentalID: "pr-facts-02"},
		&pb.DownloadPackageRef{Package: "acme/anima", Release: "1.2.0"})
	if problem == nil || !strings.Contains(problem.Message, "rental.not_found") &&
		!strings.Contains(problem.Message, "no such rental") {
		t.Fatalf("hub refusal not relayed: %+v", problem)
	}
}
