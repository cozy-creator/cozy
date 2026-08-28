package api

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"runtime"

	"github.com/cozy-creator/cozy-creator/internal/exit"
	"github.com/cozy-creator/cozy-creator/internal/orchestrator"
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
	for _, f := range s.orchestrator.Workers() {
		resident[f.Endpoint] = true
	}
	if s.endpoints != nil {
		for _, name := range s.endpoints.List() {
			placement, e := s.endpoints.ResolvePlacement(name)
			if e != nil {
				continue
			}
			row := EndpointRow{
				Endpoint: placement.Endpoint, ReleaseID: placement.ReleaseID,
				Source: "dev", Resident: resident[placement.Endpoint], Functions: []string{},
			}
			if placement.InstallID != "" {
				row.Source = "generation"
			}
			for _, b := range placement.Bindings {
				row.Functions = append(row.Functions, b.Entrypoint)
			}
			rows = append(rows, row)
		}
	}
	s.ok(w, r, http.StatusOK, map[string]any{"endpoints": rows, "count": len(rows)})
}

func (s *Server) localWorkers(w http.ResponseWriter, r *http.Request) {
	facts := s.orchestrator.Workers()
	s.ok(w, r, http.StatusOK, map[string]any{"workers": facts, "count": len(facts)})
}

func (s *Server) startWorker(w http.ResponseWriter, r *http.Request) {
	// `warm` is a POINTER so its absence is a fact: unspecified means WARM, which is the
	// serving default, and only an explicit `false` turns the boot warm pass off.
	var body struct {
		Endpoint string `json:"endpoint"`
		Rental   string `json:"rental"`
		Warm     *bool  `json:"warm"`
	}
	data, _ := io.ReadAll(io.LimitReader(r.Body, 1<<16))
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	err := decoder.Decode(&body)
	var trailing any
	if err == nil {
		err = decoder.Decode(&trailing)
	}
	if err != io.EOF || (body.Endpoint == "") == (body.Rental == "") ||
		body.Rental != "" && body.Warm != nil {
		s.refuse(w, r, http.StatusBadRequest, "invalid_request",
			`this route takes exactly one of {"endpoint":"org/name","warm":false} or {"rental":"id"}`,
			"a rental probe claims an already-provisioned worker and never selects local warmup policy")
		return
	}
	var spec orchestrator.WorkerLaunchSpec
	var instance, endpoint string
	var change orchestrator.WorkerChange
	var e *exit.Error
	if body.Rental != "" {
		instance, endpoint, change, e = s.orchestrator.EnsureRental(body.Rental)
	} else {
		if s.endpoints == nil {
			s.refuse(w, r, http.StatusServiceUnavailable, "no_resolver",
				"this LocalService resolves no endpoints", "")
			return
		}
		spec, e = s.endpoints.Resolve(body.Endpoint)
		if e == nil {
			endpoint = spec.Placement.Endpoint
			// THE BOOT WARM PASS, off by request. A rental was already provisioned
			// under its exact execution and has no local warmup choice on this route.
			if body.Warm != nil && !*body.Warm {
				spec.Warmup = orchestrator.WarmupNone
			}
			instance, change, e = s.orchestrator.EnsureWorker(spec)
		}
	}
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
	// EnsureWorker is idempotent by construction and SAYS WHAT IT DID (#484). `resident`
	// is deleted: it answered "was it already there", which is true both of a no-op and of
	// a live worker that just gained a placement, and a client that has to tell those apart
	// cannot read it off a boolean.
	note := map[orchestrator.WorkerChange]string{
		orchestrator.ChangeNone: "already hosting this placement: this route is idempotent",
		orchestrator.ChangeWorkerStarted: "spawned; poll GET /v1/local/workers until the " +
			"placement's serving axis is DISPATCHABLE",
		orchestrator.ChangePlacementAdded: "the live worker was converged onto a desired set " +
			"carrying this placement; poll GET /v1/local/workers for its serving axis",
	}[change]
	if body.Endpoint != "" && change == orchestrator.ChangeNone && spec.Warmup == orchestrator.WarmupNone {
		// The flag asked for a boot that has already happened. Saying so is the
		// difference between an idempotent verb and one that quietly ignores an
		// argument — the same defect as a flag that parses and does nothing.
		note += "; `warm:false` changed nothing — the resident worker was booted with its own warm setting"
	}
	status := http.StatusAccepted
	if change == orchestrator.ChangeNone {
		status = http.StatusOK
	}
	s.ok(w, r, status, map[string]any{
		"instance_id": instance, "endpoint": endpoint,
		"change": string(change), "note": note,
	})
}

func (s *Server) stopWorker(w http.ResponseWriter, r *http.Request) {
	instance := r.PathValue("instance_id")
	found := false
	for _, f := range s.orchestrator.Workers() {
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
	s.orchestrator.ShutdownWorker(instance, orchestrator.StopGrace)
	s.ok(w, r, http.StatusOK, map[string]any{"instance_id": instance, "stopped": true})
}

func (s *Server) reviseRental(w http.ResponseWriter, r *http.Request) {
	var body struct {
		EndpointRef    string `json:"endpoint_ref"`
		IdempotencyKey string `json:"idempotency_key"`
		Reason         string `json:"reason"`
	}
	data, _ := io.ReadAll(io.LimitReader(r.Body, 1<<16))
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	err := decoder.Decode(&body)
	var trailing any
	if err == nil {
		err = decoder.Decode(&trailing)
	}
	if err != io.EOF || body.EndpointRef == "" || body.IdempotencyKey == "" || body.Reason == "" ||
		r.Header.Get("Idempotency-Key") != body.IdempotencyKey {
		s.refuse(w, r, http.StatusBadRequest, "invalid_request",
			"endpoint_ref, idempotency_key, and reason are required", "send one closed revision request")
		return
	}
	facts, revision, problem := s.orchestrator.ReviseRental(r.Context(), r.PathValue("rental_id"),
		body.EndpointRef, body.IdempotencyKey, body.Reason)
	if problem != nil {
		s.refuseTyped(w, r, problem)
		return
	}
	s.ok(w, r, http.StatusAccepted, map[string]any{
		"rental_id": r.PathValue("rental_id"), "placement_revision": revision,
		"instance_id": facts.InstanceID, "worker_boot_id": facts.BootID,
		"endpoint": facts.Endpoint, "desired_state_revision": facts.DesiredRevision,
		"artifact_grant_revision": facts.GrantRevision,
	})
}

// shutdownService is `cozy down`'s COOPERATIVE tier (#449): an authenticated ask that the
// service drain every worker and exit — the same act the Unix SIGTERM performs, spelled as
// a route so it exists on every platform (Windows has no signal to send a detached
// service). The reply races the exit deliberately: it is sent before the drain starts, and
// the caller's proof of completion is the service LOCK becoming free, never this body.
func (s *Server) shutdownService(w http.ResponseWriter, r *http.Request) {
	if s.shutdown == nil {
		s.refuse(w, r, http.StatusConflict, "conflict",
			"this server was built with no shutdown hook", "")
		return
	}
	s.ok(w, r, http.StatusAccepted, map[string]any{
		"shutting_down": true,
		"note":          "draining workers; the service lock frees when the exit is complete",
	})
	go s.shutdown()
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
		"workers":    s.orchestrator.Workers(),
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
