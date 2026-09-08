package producttest

import (
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/records"
)

func authoredH3(lane string) []launch.ModelDefaultRung {
	return []launch.ModelDefaultRung{{GPU: "H100", Lane: ladderModel + "@" + ladderRelease + "/" + lane}}
}

func TestAuthoredModelDefaultPrecedence(t *testing.T) {
	for _, mode := range []string{"authored", "owner", "explicit", "unbound"} {
		t.Run(mode, func(t *testing.T) {
			h := newLadderHub(t, authoredH3(ladderLane))
			root := ladderRoot(t, h)
			t.Cleanup(func() { _, _ = runCozy(t, root, "down", "--all") })
			want := ladderLane
			if mode != "authored" {
				owner := goodLadder()
				owner.Ladder = []hub.BindingRung{{GPU: "H100", Lane: "bf16-full"}}
				h.bind(owner)
				want = "bf16-full"
			}
			if mode == "unbound" {
				code, out := runCozy(t, root, "package", "unbind", ladderPackage, ladderSlot, "--json")
				if code != 0 || !strings.Contains(out, `"changed":true`) {
					t.Fatalf("unbind failed: %d %s", code, out)
				}
				want = ladderLane
			}
			args := []string{"run", ladderPackage + "/generate", "steps=1"}
			if mode == "explicit" {
				want = ladderLane
				args = append(args, "model.model="+ladderModel+"@"+ladderRelease+"/"+ladderLane)
			}
			args = append(args, "--rental-only", "--json", "--idempotency-key", mode)
			code, out := runCozy(t, root, args...)
			store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
			fatal(t, problem)
			defer store.Close()
			row, problem := store.RequestByIdempotencyKey(mode)
			fatal(t, problem)
			if row == nil || len(row.Models) != 1 {
				t.Fatalf("default never reached request: %d %s row=%+v", code, out, row)
			}
			model := row.Models[0]
			if model.Model != ladderModel || model.Release != ladderRelease {
				t.Fatalf("wrong default target: %+v", model)
			}
			lane := model.Lane
			if lane == "" && len(model.Ladder) > 0 {
				lane = model.Ladder[0].Lane
			}
			if lane != want {
				t.Fatalf("%s selected %s instead of %s: %+v", mode, lane, want, model)
			}
			h.mu.Lock()
			writes := len(h.puts)
			h.mu.Unlock()
			if writes != 0 {
				t.Fatal("invocation synchronized defaults into Hub rows")
			}
		})
	}
}

func TestAuthoredDefaultsDoNotHideOwnerReadFailure(t *testing.T) {
	h := newLadderHub(t, authoredH3(ladderLane))
	root := ladderRoot(t, h)
	t.Cleanup(func() { _, _ = runCozy(t, root, "down", "--all") })
	h.mu.Lock()
	h.bindingsUnavailable = true
	h.mu.Unlock()
	code, out := runCozy(t, root, "run", ladderPackage+"/generate", "steps=1", "--rental-only", "--json")
	if code == 0 || !strings.Contains(out, "package_default_model_unavailable") {
		t.Fatalf("unreadable owner choices silently fell back: %d %s", code, out)
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	if len(h.posts) != 0 {
		t.Fatal("unreadable override reached rental creation")
	}
}

func TestUnbindPreservesConcurrentOwnerChoice(t *testing.T) {
	h := newLadderHub(t)
	root := ladderRoot(t, h)
	t.Cleanup(func() { _, _ = runCozy(t, root, "down", "--all") })
	h.bind(goodLadder())
	h.mu.Lock()
	h.resetConflict = true
	h.mu.Unlock()
	code, out := runCozy(t, root, "package", "unbind", ladderPackage, ladderSlot, "--json")
	if code == 0 || !strings.Contains(out, "binding.revision_conflict") {
		t.Fatalf("stale unbind did not conflict: %d %s", code, out)
	}
	h.mu.Lock()
	if len(h.bindings) != 1 || len(h.deletes) != 1 || string(h.deletes[0]) != `{"expected_revision":3}` {
		t.Fatalf("unbind replaced concurrent row or omitted CAS: %+v %s", h.bindings, h.deletes)
	}
	h.bindings = nil
	h.resetConflict = false
	h.mu.Unlock()
	code, out = runCozy(t, root, "package", "unbind", ladderPackage, ladderSlot, "--json")
	if code != 0 || !strings.Contains(out, `"changed":false`) {
		t.Fatalf("empty unbind is not idempotent: %d %s", code, out)
	}
}

func TestAuthoredModelDefaultDescriptorIsClosed(t *testing.T) {
	for _, raw := range []string{`null`, `[]`, `[{"gpu":"H100","lane":"proof/m@1.0.0/fp8","extra":true}]`, `[{"gpu":1,"lane":"proof/m@1.0.0/fp8"}]`} {
		doc := map[string]any{"format": "cozy.package.interface/1", "application": "proof:app", "entrypoints": []any{map[string]any{"name": "generate", "request": map[string]any{"fields": []any{}}, "result": map[string]any{"fields": []any{}}, "models": []any{map[string]any{"class": "M", "path": "generate.models.model", "component_use": map[string]any{}, "default_ladder": json.RawMessage(raw)}}}}, "jobs": []any{}}
		body, err := json.Marshal(doc)
		must(t, err)
		if _, problem := launch.DecodePackageInterface(body); problem == nil {
			t.Fatalf("invalid authored ladder admitted: %s", raw)
		}
	}
}

func TestAuthoredDefaultRefusesMixedOrUnpinnedTargetsBeforeRental(t *testing.T) {
	for _, wrong := range []string{"proof/other@1.0.0-rc.1/fp8", "proof/minimax@2.0.0/fp8", "proof/minimax/fp8", "proof/minimax@1.0.0-rc.1", "proof/minimax@" + strings.Repeat("a", 64) + "/fp8", "proof/minimax@1.0.0-rc.1/bad lane"} {
		defaults := append(authoredH3(ladderLane), launch.ModelDefaultRung{GPU: "B200", Lane: wrong})
		h := newLadderHub(t, defaults)
		root := ladderRoot(t, h)
		t.Cleanup(func() { _, _ = runCozy(t, root, "down", "--all") })
		code, out := runCozy(t, root, "run", ladderPackage+"/generate", "steps=1", "--rental-only", "--json")
		if code == 0 || !strings.Contains(out, "package_model_default_invalid") {
			t.Fatalf("invalid authored target reached resolution: %q %d %s", wrong, code, out)
		}
		h.mu.Lock()
		buys := len(h.posts)
		h.mu.Unlock()
		if buys != 0 {
			t.Fatal("invalid default reached paid boundary")
		}
	}
}

func TestGPUFitPatternsAllowMultipleTokens(t *testing.T) {
	row := goodLadder()
	row.Ladder = []hub.BindingRung{{GPU: "NVIDIA H100", Lane: ladderLane}}
	if problem := hub.ValidateLadder(row.Ladder); problem != nil {
		t.Fatal(problem)
	}
	for _, gpu := range []string{" H100", "H100 ", strings.Repeat("x", 65), "---"} {
		if problem := hub.ValidateLadder([]hub.BindingRung{{GPU: gpu, Lane: ladderLane}}); problem == nil {
			t.Fatalf("invalid GPU pattern admitted: %q", gpu)
		}
	}
}
