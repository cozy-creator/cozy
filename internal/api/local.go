package api

import (
	"encoding/json"
	"io"
	"net/http"
	"runtime"
	"time"

	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
)

// The LOCAL EXTENSION MODULE. Everything here is mounted under /v1/local/ so the boundary
// is visible in the URL rather than only in a document: a client that sees `/v1/local/`
// knows it has left the shared contract and is talking to this host's own surface. The
// cloud host serves none of these and never will; it serves catalog/billing/org routes
// this host serves none of. The CORE between them is byte-identical, and neither set of
// extensions pretends to be part of it.

// EndpointRow is one endpoint this host can serve.
type EndpointRow struct {
	Endpoint  string   `json:"endpoint"`
	ReleaseID string   `json:"release_id"`
	Source    string   `json:"source"` // "generation" | "dev"
	Functions []string `json:"functions"`
	Resident  bool     `json:"resident"`
}

func (s *Server) localEndpoints(w http.ResponseWriter, r *http.Request) {
	rows := []EndpointRow{}
	resident := map[string]bool{}
	for _, f := range s.coord.Workers() {
		resident[f.Endpoint] = true
	}
	if s.endpoints != nil {
		for _, name := range s.endpoints.List() {
			spec, e := s.endpoints.Resolve(name)
			if e != nil {
				continue
			}
			row := EndpointRow{
				Endpoint: spec.Endpoint, ReleaseID: spec.ReleaseID,
				Source: "dev", Resident: resident[spec.Endpoint], Functions: []string{},
			}
			if spec.Generation != "" {
				row.Source = "generation"
			}
			for _, b := range spec.Bindings {
				row.Functions = append(row.Functions, b.Entrypoint)
			}
			rows = append(rows, row)
		}
	}
	s.ok(w, r, http.StatusOK, map[string]any{"endpoints": rows, "count": len(rows)})
}

func (s *Server) localWorkers(w http.ResponseWriter, r *http.Request) {
	facts := s.coord.Workers()
	s.ok(w, r, http.StatusOK, map[string]any{"workers": facts, "count": len(facts)})
}

func (s *Server) startWorker(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Endpoint string `json:"endpoint"`
	}
	data, _ := io.ReadAll(io.LimitReader(r.Body, 1<<16))
	if err := json.Unmarshal(data, &body); err != nil || body.Endpoint == "" {
		s.refuse(w, r, http.StatusBadRequest, "invalid_request",
			`this route takes {"endpoint":"org/name"}`,
			"a client names an ENDPOINT; resolving it to an interpreter and a binding is this host's job")
		return
	}
	if s.endpoints == nil {
		s.refuse(w, r, http.StatusServiceUnavailable, "no_resolver",
			"this LocalService resolves no endpoints", "")
		return
	}
	spec, e := s.endpoints.Resolve(body.Endpoint)
	if e != nil {
		s.refuseTyped(w, r, e)
		return
	}
	// Already resident is an IDEMPOTENT 200 (cozy-creator.md: "already serving = 0").
	for _, f := range s.coord.Workers() {
		if f.Endpoint == spec.Endpoint {
			s.ok(w, r, http.StatusOK, map[string]any{
				"instance_id": f.InstanceID, "endpoint": f.Endpoint, "resident": true,
				"note": "already resident: this route is idempotent",
			})
			return
		}
	}
	instance, e := s.coord.StartWorker(spec)
	if e != nil {
		s.refuseTyped(w, r, e)
		return
	}
	s.ok(w, r, http.StatusAccepted, map[string]any{
		"instance_id": instance, "endpoint": spec.Endpoint, "resident": false,
		"note": "spawned; poll GET /v1/local/workers for intake READY",
	})
}

func (s *Server) stopWorker(w http.ResponseWriter, r *http.Request) {
	instance := r.PathValue("instance_id")
	found := false
	for _, f := range s.coord.Workers() {
		if f.InstanceID == instance {
			found = true
		}
	}
	if !found {
		// Not running is an idempotent 200: `stop` on something already stopped is a
		// no-op, not a refusal.
		s.ok(w, r, http.StatusOK, map[string]any{
			"instance_id": instance, "stopped": false,
			"note": "no such live worker: stopping is idempotent",
		})
		return
	}
	// The WHOLE process group drains and stops — the only yield mechanism there is. An
	// attempt is never killed to improve queue latency and no suspend path exists.
	s.coord.StopWorker(instance, 30*time.Second)
	s.ok(w, r, http.StatusOK, map[string]any{"instance_id": instance, "stopped": true})
}

func (s *Server) doctor(w http.ResponseWriter, r *http.Request) {
	counts, _ := s.store.Counts()
	head, _ := s.store.LastEventSeq()
	s.ok(w, r, http.StatusOK, map[string]any{
		"service": map[string]any{
			"address": s.addr, "bind": "loopback-only", "root": s.layout.Root,
			"records": s.layout.DB, "socket_families": s.families(),
		},
		"host": map[string]any{
			"os": runtime.GOOS, "arch": runtime.GOARCH, "cpus": runtime.NumCPU(),
		},
		"counts":     counts,
		"event_head": head,
		"workers":    s.coord.Workers(),
	})
}

// families names which loopback families are actually bound, so `doctor` reports what is
// true rather than what was intended: an IPv6 bind that failed on a v6-disabled host is a
// FACT to print, not a silent difference between the Host allowlist and reality.
func (s *Server) families() []string {
	if s.bound == nil {
		return []string{}
	}
	return s.bound
}

// UnknownEndpoint is how every LOCAL route refuses a ref this host does not serve. One
// sentence, one remedy, one place.
func UnknownEndpoint(name string) *exit.Error {
	return exit.New(exit.NotFound, "this host serves no endpoint %q", name).
		WithRemedy("GET /v1/local/endpoints lists what it does serve")
}
