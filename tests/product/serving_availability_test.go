package producttest

import (
	"encoding/json"
	"strings"
	"testing"
)

func servingAvailabilityInterface(t *testing.T) []byte {
	t.Helper()
	var document map[string]any
	must(t, json.Unmarshal(admissionInterface(t, false, false), &document))
	entrypoints := document["entrypoints"].([]any)
	raw, err := json.Marshal(entrypoints[0])
	must(t, err)
	var variant map[string]any
	must(t, json.Unmarshal(raw, &variant))
	variant["name"] = "variant"
	variant["models"] = []any{
		map[string]any{"class": "H3", "component_use": map[string]any{}, "path": "variant.models.model", "default": h3Default},
		map[string]any{"class": "Adapter", "component_use": map[string]any{}, "path": "variant.models.adapter"},
	}
	// A Weights[M] argument is the caller's raw input: no default needed to be available.
	document["entrypoints"] = append(entrypoints, variant, map[string]any{
		"name": "ping", "request": map[string]any{"fields": []any{}}, "result": map[string]any{"fields": []any{}},
	}, map[string]any{
		"name": "quantize", "publishes": false, "accelerator": false,
		"models":  []any{map[string]any{"class": "Model", "component_use": map[string]any{}, "path": "quantize.models.source", "weights": true}},
		"request": map[string]any{"fields": []any{}}, "result": map[string]any{"fields": []any{}},
	})
	raw, err = json.Marshal(document)
	must(t, err)
	return raw
}

func TestServingListDistinguishesUnboundDeploymentFromRawInputs(t *testing.T) {
	root, path, _, _ := admissionRoot(t, servingAvailabilityInterface(t), "H100", false)
	code, out := runAdmissionCLI(t, root, path, "run", h3Package, "--json")
	if code != 0 {
		t.Fatalf("function listing failed [exit %d]: %s", code, out)
	}
	var result map[string]json.RawMessage
	must(t, json.Unmarshal([]byte(out), &result))
	var rows []struct {
		Function     string `json:"function"`
		Availability string `json:"availability"`
	}
	must(t, json.Unmarshal(result["functions"], &rows))
	got := map[string]string{}
	for _, row := range rows {
		got[row.Function] = row.Availability
	}
	for function, want := range map[string]string{
		"generate": "available", "variant": "disabled: no default for adapter", "ping": "available", "quantize": "available",
	} {
		if got[function] != want {
			t.Errorf("%s availability=%q, want %q: %s", function, got[function], want, out)
		}
	}
	if strings.Contains(string(result["next"]), "/variant") {
		t.Errorf("listing recommended a disabled default invocation: %s", out)
	}
}

func TestUnboundServingDefaultRefusesBeforeAcquisitionButExplicitModelsProceed(t *testing.T) {
	root, path, activity, probe := admissionRoot(t, servingAvailabilityInterface(t), "H100", false)
	args := []string{"run", h3Package + "/variant", "prompt=hello", "assets:=[]", "--rental-only", "--json"}
	code, out := runAdmissionCLI(t, root, path, args...)
	if code == 0 || !strings.Contains(out, "declares no default") || !strings.Contains(out, "adapter") {
		t.Fatalf("unbound serving default was not refused before acquisition: %s", out)
	}
	requests, _ := probe.snapshot()
	for _, request := range requests {
		if request != "GET /v1/packages/proof/h3" && request != "GET /v1/packages/proof/h3/releases/1.0.0" &&
			!calciferFleetHousekeeping(request) {
			t.Errorf("unbound function reached %s (activity %s)", request, tail(activity))
		}
	}
	model := h3Model + "@" + h3Release + "/" + h3Lane
	args = append(args, "model.model="+model, "model.adapter="+model)
	code, out = runAdmissionCLI(t, root, path, args...)
	if code == 0 || !strings.Contains(out, "fixture_finished") || strings.Contains(out, "declares no default") {
		t.Fatalf("explicit private model inputs failed default availability instead of reaching the fixture's capacity refusal: %s", out)
	}
	localArgs := make([]string, 0, len(args))
	for _, arg := range args {
		if arg != "--rental-only" {
			localArgs = append(localArgs, arg)
		}
	}
	// Accepted for this computer's machine, which describes nothing first (th-241).
	_, out = runAdmissionCLI(t, root, path, localArgs...)
	if !strings.Contains(out, `"machine":"local"`) || strings.Contains(out, "declares no default") {
		t.Fatalf("explicit local model inputs failed default availability instead of reaching this computer's machine: %s", out)
	}
}
