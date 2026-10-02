package client

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/machineendpoint"
	"github.com/cozy-creator/cozy/internal/secret"
)

func TestOlderControllerCannotDropExplicitEndpointAndSubmitLocally(t *testing.T) {
	var posts atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			posts.Add(1)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"model_overrides":true}`))
	}))
	defer server.Close()
	c := At(config.Config{}, strings.TrimPrefix(server.URL, "http://"), secret.Mint())
	_, problem := c.Submit(api.Submission{Package: "local/fixture", Function: "classify", MachineEndpoint: &machineendpoint.Endpoint{}}, "key")
	if problem == nil || problem.ErrName() != "daemon.machine_endpoint_unavailable" || posts.Load() != 0 {
		t.Fatalf("endpoint absence must stop before POST: problem=%v posts=%d", problem, posts.Load())
	}
}
