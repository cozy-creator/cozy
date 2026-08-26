package api

import (
	"encoding/json"
	"io"
	"net/http"
	"runtime"

	"github.com/cozy-creator/cozy-creator-v2/internal/coord"
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
	// `warm` is a POINTER so its absence is a fact: unspecified means WARM, which is the
	// serving default, and only an explicit `false` turns the boot warm pass off.
	var body struct {
		Endpoint string `json:"endpoint"`
		Warm     *bool  `json:"warm"`
	}
	data, _ := io.ReadAll(io.LimitReader(r.Body, 1<<16))
	if err := json.Unmarshal(data, &body); err != nil || body.Endpoint == "" {
		s.refuse(w, r, http.StatusBadRequest, "invalid_request",
			`this route takes {"endpoint":"org/name"} and optionally {"warm":false}`,
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
	// THE BOOT WARM PASS, off by request. It normally earns its keep — it pays the
	// first-call tax (960-1380 ms on the four-component SDXL pipeline) before a user's
	// first request instead of inside it. cl-003 found the two cases where it does not:
	// an entrypoint whose warm shape does not fit degrades its binding at boot for
	// nothing (observed: `condition` OOMs the 1024px decode with every component
	// resident), and the warm pass is a BIT-LEVEL input to the first real image, because
	// the device-memory history it leaves is what cuDNN's algorithm selection reads.
	if body.Warm != nil && !*body.Warm {
		spec.NoWarm = true
	}
	// Already resident is an IDEMPOTENT 200 (cozy-creator.md: "already serving = 0").
	for _, f := range s.coord.Workers() {
		if f.Endpoint == spec.Endpoint {
			note := "already resident: this route is idempotent"
			if spec.NoWarm {
				// The flag asked for a boot that has already happened. Saying so is the
				// difference between an idempotent verb and one that quietly ignores an
				// argument — the same defect as a flag that parses and does nothing.
				note += "; `warm:false` changed nothing — the resident worker was booted with its own warm setting"
			}
			s.ok(w, r, http.StatusOK, map[string]any{
				"instance_id": f.InstanceID, "endpoint": f.Endpoint, "resident": true,
				"note": note,
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
	s.coord.StopWorker(instance, coord.StopGrace)
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
