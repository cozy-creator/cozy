package client

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/secret"
)

func TestExplicitGPUCountRequiresControllerSupportBeforeSubmission(t *testing.T) {
	for _, supported := range []bool{false, true} {
		for _, job := range []bool{false, true} {
			var posts int
			var captured uint32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				if r.Method == http.MethodGet {
					_ = json.NewEncoder(w).Encode(api.Capabilities{RunGPUs: supported, ModelOverrides: true})
					return
				}
				posts++
				var body struct {
					GPUs uint32 `json:"gpus"`
				}
				_ = json.NewDecoder(r.Body).Decode(&body)
				captured = body.GPUs
				_, _ = w.Write([]byte(`{"request_id":"recorded","job_id":"recorded"}`))
			}))
			c := At(config.Config{}, strings.TrimPrefix(server.URL, "http://"), secret.Mint())
			if job {
				_, problem := c.SubmitJob(api.JobSubmission{Package: "local/fixture", Function: "job", GPUs: 2}, "two")
				if (problem == nil) != supported {
					t.Fatalf("job support=%v: %v", supported, problem)
				}
			} else {
				_, problem := c.Submit(api.Submission{Package: "local/fixture", Function: "call", GPUs: 2}, "two")
				if (problem == nil) != supported {
					t.Fatalf("call support=%v: %v", supported, problem)
				}
			}
			server.Close()
			if supported && (posts != 1 || captured != 2) || !supported && posts != 0 {
				t.Fatalf("support=%v job=%v posts=%d count=%d", supported, job, posts, captured)
			}
		}
	}
}
