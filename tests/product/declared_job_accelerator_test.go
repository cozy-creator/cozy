package producttest

import (
	"encoding/json"
	"net/http"
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/records"
)

// The H3 conversion agent, 2026-09-27: every job of a package whose closure names torch was
// accelerator-class, so the pure-CPU conversions sdxl `prepare`/`fp8` and anima `fp8` read
// excluded:wrong_class on the owner's CPU pod. A job's own `accelerator` declaration decides
// its class. The package here names torch, binds its serving slot, and declares one CPU and
// one GPU conversion of the same shape; each is sent to the owner's named CPU rental through
// the real binary and daemon against the stand-in hub, which buys nothing.
func TestDeclaredJobAcceleratorDecidesTheRentalClass(t *testing.T) {
	h := newLadderHub(t)
	h.bind(goodLadder())
	var doc map[string]any
	must(t, json.Unmarshal(h.iface, &doc))
	// As sdxl's `generate`: invocable, so a CPU workflow could call it and reserve its slot.
	doc["entrypoints"].([]any)[0].(map[string]any)["invocable"] = map[string]any{"context": "ctx", "defaults": map[string]any{},
		"enum_members": map[string]any{}, "export": "generate", "module": "h3", "parameters": []any{"steps"}, "type_names": map[string]any{}}
	cpu := doc["jobs"].([]any)[0].(map[string]any)
	cpu["accelerator"] = false
	gpu := map[string]any{}
	for key, value := range cpu {
		gpu[key] = value
	}
	gpu["name"], gpu["accelerator"] = "tables", true
	gpu["models"] = []any{map[string]any{"class": "Source", "component_use": map[string]any{}, "path": "tables.models.pruned"}}
	doc["jobs"] = append(doc["jobs"].([]any), gpu)
	iface, err := json.Marshal(doc)
	must(t, err)
	h.iface = iface
	declared, problem := launch.DecodePackageInterface(iface)
	fatal(t, problem)
	h.mux.HandleFunc("POST /v1/packages/proof/h3/download", func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(declaredInstallPlan("h3", r.URL.Query().Get("release"), declared))
	})
	root := ladderRoot(t, h)
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	plantAttached(t, root, h, store, records.Rental{AcceleratorCount: 1, ID: "pr-karam", MachineName: "karam", SKU: "cpu",
		AcceleratorModel: "CPU", HourlyRateUSDMicros: 70_000, State: "ready", Hub: h.server.URL}, true)
	startDaemonProcess(t, root)
	source := "--model.pruned=" + ladderModel + "@" + ladderRelease + "/" + ladderLane
	run := func(function, key string) *records.Request {
		t.Helper()
		_, out := runCozy(t, root, "run", ladderPackage+"/"+function, source, "--rental=karam", "--json", "--idempotency-key", key)
		row, problem := store.RequestByIdempotencyKey(key)
		fatal(t, problem)
		if row == nil {
			t.Fatalf("%s was not submitted: %s", function, out)
		}
		return row
	}

	// A named rental is where each job goes; what the machine cannot run, it refuses. Each
	// job's own declaration still decides its class.
	converted := run("lane", "declared-cpu")
	tables := run("tables", "declared-gpu")
	waitFor(t, root, "both jobs pinned to the named rental", func() bool {
		converted, problem = store.RequestByIdempotencyKey("declared-cpu")
		if problem != nil || converted == nil {
			return false
		}
		tables, problem = store.RequestByIdempotencyKey("declared-gpu")
		return problem == nil && tables != nil && converted.Worker != "" && tables.Worker != ""
	})
	if converted.NeedsAccelerator || converted.Worker != "pr-karam" {
		t.Fatalf("the CPU-declared job went to %q (accelerator %v)", converted.Worker, converted.NeedsAccelerator)
	}
	if !tables.NeedsAccelerator || tables.Worker != "pr-karam" {
		t.Fatalf("the GPU-declared job went to %q (accelerator %v)", tables.Worker, tables.NeedsAccelerator)
	}
	if asks := h.postedSKUs(); len(asks) != 0 {
		t.Fatalf("a named rental bought %v", asks)
	}
}
