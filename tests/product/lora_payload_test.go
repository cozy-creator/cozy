package producttest

import (
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/records"
	"path/filepath"
	"strings"
	"testing"
)

func TestLoRAOverridesPreserveOrderZeroAndExplicitComponents(t *testing.T) {
	ep := &launch.Entrypoint{Name: "generate", Models: []launch.Slot{{Param: "model", Path: "generate.models.model"}}}
	rows, problem := launch.ParseLoRAs(ep, []string{"model:fl2va_dit=owner/first@1", "model:ref2va_dit=owner/second#sha256:abc,0", "model:fl2va_dit=owner/third,-.25"})
	fatal(t, problem)
	if len(rows) != 3 || rows[0].Ref != "owner/first@1" || rows[0].Scale != "1" || rows[1].Scale != "0" || rows[1].Component != "ref2va_dit" || rows[2].Scale != "-0.25" {
		t.Fatalf("changed ordered input: %+v", rows)
	}
	for _, bad := range []string{"model=owner/x", "unknown:fl2va_dit=owner/x", "model:fl2va_dit=owner/x,", "model:fl2va_dit=owner/x,NaN", "model:fl2va_dit=owner/x,Inf", "model:fl2va_dit=owner/x,0x1p0", "model:fl2va_dit=owner/x,1e309"} {
		if _, problem := launch.ParseLoRAs(ep, []string{bad}); problem == nil {
			t.Fatalf("accepted %s", bad)
		}
	}
}

func TestLoRAScalesUseFiniteCanonicalDecimalStrings(t *testing.T) {
	for input, want := range map[string]string{"0": "0", "-0": "0", "0.0": "0", "+.5": "0.5", "-0.25": "-0.25", "1000000": "1e+06", "0.00001": "1e-05", "1.00": "1"} {
		got, problem := launch.CanonicalLoRAScale(input)
		fatal(t, problem)
		if got != want {
			t.Fatalf("%s -> %s want %s", input, got, want)
		}
	}
}

func TestLoRADiskDeclarationIncludesDistinctAdaptersAtZeroStrength(t *testing.T) {
	store, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	adapter := records.ModelAdapterRef{Component: "dit", Model: "owner/lora", Manifest: "sha256:" + strings.Repeat("b", 64), SourceComponent: "adapter", Scale: "0"}
	_, _, problem = store.Submit(records.Request{ID: "req-lora", IdemKey: "lora", BodyDigest: "sha256:" + strings.Repeat("c", 64), Package: "owner/package", Entrypoint: "generate", State: "queued", Payload: []byte("{}"), Models: []records.ModelRef{
		{Package: "owner/package", Slot: "generate.models.model", Model: "owner/base", Manifest: "sha256:" + strings.Repeat("a", 64), HubCheckpoint: true, Adapters: []records.ModelAdapterRef{adapter, adapter}},
	}})
	fatal(t, problem)
	models, problem := store.DeclaredServingModels("req-lora")
	fatal(t, problem)
	if len(models) != 2 || models[0].Model != "owner/base" || models[1].Model != "owner/lora" || !models[1].Downloadable() {
		t.Fatalf("incomplete disk closure: %+v", models)
	}
}
