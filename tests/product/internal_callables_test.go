package producttest

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/records"
)

const internalCallablesInterface = `{"format":"cozy.package.interface/1","application":"fixture:app","entrypoints":[{"name":"generate","request":{"fields":[]},"result":{"fields":[]}},{"name":"segment","internal":true,"request":{"fields":[]},"result":{"fields":[]}}],"jobs":[{"name":"long_form","publishes":false,"request":{"fields":[]},"result":{"fields":[]}},{"name":"internal_job","internal":true,"publishes":false,"request":{"fields":[]},"result":{"fields":[]}}]}`

func TestInternalCallablesStayInExecutionInterface(t *testing.T) {
	surface, problem := launch.DecodePackageInterface([]byte(internalCallablesInterface))
	fatal(t, problem)
	if !reflect.DeepEqual(surface.PublicNames(), []string{"generate", "long_form"}) {
		t.Fatal(surface.PublicNames())
	}
	if !reflect.DeepEqual(surface.Names(), []string{"generate", "internal_job", "long_form", "segment"}) {
		t.Fatal("execution capture lost internal names", surface.Names())
	}
	for _, name := range []string{"segment", "internal_job"} {
		entry, problem := surface.Function(name)
		fatal(t, problem)
		if !entry.Internal || entry.RequirePublic() == nil {
			t.Fatal("lost internal boundary", name)
		}
	}
	for _, value := range []string{"null", `"true"`, "1", "{}"} {
		raw := strings.ReplaceAll(internalCallablesInterface, `"internal":true`, `"internal":`+value)
		if _, problem := launch.DecodePackageInterface([]byte(raw)); problem == nil {
			t.Fatal("accepted non-boolean visibility", value)
		}
	}
}

func TestInternalCallablesAreAbsentFromCLIAndRefuseDirectRun(t *testing.T) {
	raw := []byte(internalCallablesInterface)
	surface, problem := launch.DecodePackageInterface(raw)
	fatal(t, problem)
	var detail hub.PackageReleaseDetail
	detail.PackageInterface = surface.Raw
	detail.Release.Release = "1.0.0"
	detail.Release.PackageInterfaceDigest = surface.Digest
	detail.Release.PackageInterfaceLength = int64(len(surface.Raw))
	detail.ExecutionRequirements = []string{"cozy-runtime>=0.2.25"}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/packages/proof/internal", func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(hub.PackageCard{Package: hub.Resource{Org: "proof", Name: "internal"}, Releases: []hub.ReleaseSummary{{Release: "1.0.0"}}})
	})
	mux.HandleFunc("GET /v1/packages/proof/internal/releases/1.0.0", func(w http.ResponseWriter, _ *http.Request) { _ = json.NewEncoder(w).Encode(detail) })
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		t.Errorf("public selection unexpectedly reached %s %s", r.Method, r.URL.Path)
		w.WriteHeader(503)
	})
	server := httptest.NewServer(mux)
	t.Cleanup(server.Close)
	root := t.TempDir()
	t.Cleanup(func() { _, _ = runCozy(t, root, "down", "--json") })
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+server.URL+"\nrentals:\n  max_hourly_spend_usd: 20\n"), 0600))
	code, out := runCozy(t, root, "run", "proof/internal", "--rental-only", "--json", "--full")
	if code != 0 || strings.Contains(out, `"segment"`) || strings.Contains(out, "internal_job") || !strings.Contains(out, "long_form") || !strings.Contains(out, "generate") {
		t.Fatalf("public list: %d %s", code, out)
	}
	for _, name := range []string{"segment", "internal_job"} {
		for _, extra := range [][]string{nil, {"--describe"}} {
			args := append([]string{"run", "proof/internal/" + name, "--rental-only", "--json"}, extra...)
			code, out = runCozy(t, root, args...)
			if code == 0 || !strings.Contains(out, `"code":"callable_internal"`) {
				t.Fatalf("internal root escaped: %d %s", code, out)
			}
		}
	}
	if _, err := os.Stat(filepath.Join(root, "creator.sqlite")); err == nil {
		store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
		fatal(t, problem)
		defer store.Close()
		if request, problem := store.RequestByReference("1"); problem != nil || request != nil {
			t.Fatalf("internal root was submitted: %+v %v", request, problem)
		}
	}
}
