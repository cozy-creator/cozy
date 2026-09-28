package producttest

import (
	"encoding/json"
	"fmt"
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
)

func runInputJSONRoot(t *testing.T) string {
	t.Helper()
	raw := []byte(`{"application":"fixture:app","entrypoints":[],"format":"cozy.package.interface/1","jobs":[{"name":"prepare","publishes":false,"request":{"fields":[{"name":"shots","type":{"list":{"fields":[{"name":"prompt","type":"str"},{"name":"seed","type":"int"}]}}},{"name":"steps","type":"int"}]},"result":{"fields":[]}}]}`)
	iface, problem := launch.DecodePackageInterface(raw)
	fatal(t, problem)
	var detail hub.PackageReleaseDetail
	detail.PackageInterface = raw
	detail.Release.Release = "1.0.0"
	detail.Release.PackageInterfaceDigest = assessmentDigest(iface.Raw)
	detail.Release.PackageInterfaceLength = int64(len(raw))
	detail.ExecutionRequirements = []string{"cozy-runtime>=0.2.25"}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/packages/proof/input", func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(hub.PackageCard{Package: hub.Resource{Org: "proof", Name: "input"}, Releases: []hub.ReleaseSummary{{Release: "1.0.0"}}})
	})
	mux.HandleFunc("GET /v1/packages/proof/input/releases/1.0.0", func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(detail)
	})
	mux.HandleFunc("/", http.NotFound)
	server := httptest.NewServer(mux)
	t.Cleanup(server.Close)
	root, err := os.MkdirTemp(scratchBase, "submitted-run-")
	must(t, err)
	t.Cleanup(func() { _ = removeAllForce(root) })
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+server.URL+"\ntensorhub_token: proof\n"), 0600))
	return root
}

func TestRunInputJSONFileAndAliasPreserveNestedPayload(t *testing.T) {
	root := runInputJSONRoot(t)
	infile := filepath.Join(root, "scene=shots.json")
	// Field names fold onto the declared spelling, as CLI terms do.
	payload := `{"Shots":[{"prompt":"First shot","seed":101},{"prompt":"Second shot","seed":102}],"STEPS":30}`
	must(t, os.WriteFile(infile, []byte(payload), 0600))
	var expected any
	must(t, json.Unmarshal([]byte(strings.NewReplacer(`"Shots"`, `"shots"`, `"STEPS"`, `"steps"`).Replace(payload)), &expected))
	expected.(map[string]any)["steps"] = float64(7)
	for _, flag := range []string{"--input"} {
		for _, inline := range []bool{false, true} {
			args := []string{"run", "proof/input/prepare", "steps=7"}
			if inline {
				args = append(args, flag+"="+infile)
			} else {
				args = append(args, flag, infile)
			}
			args = append(args, "--rental-only", "--json")
			request, _, out := submitRun(t, root, fmt.Sprintf("input-%s-%t", flag, inline), args...)
			if request == nil || !reflect.DeepEqual(any(submittedPayload(t, request)), expected) {
				t.Fatalf("%s inline=%t: %+v %s", flag, inline, request, out)
			}
		}
	}
	request, _, out := submitRun(t, root, "input-inline-override", "run", "proof/input/prepare", "--input="+infile,
		`shots:=[{"prompt":"Edited shot","seed":999}]`, "steps=8", "--rental-only", "--json")
	if request == nil || !strings.Contains(string(request.Payload), `"prompt":"Edited shot"`) || !strings.Contains(string(request.Payload), `"seed":999`) ||
		!strings.Contains(string(request.Payload), `"steps":8`) || strings.Contains(string(request.Payload), "First shot") {
		t.Fatalf("nested inline override changed: %+v %s", request, out)
	}
	// The tree option still binds a directory rather than the JSON payload despite the
	// old internal Values key also being "--input".
	code, out := runCozy(t, root, "run", "proof/input/prepare", "--input="+infile,
		"--input-tree", "prior="+filepath.Join(root, "missing"), "--rental-only", "--json")
	if code == 0 || !strings.Contains(out, "--input-tree prior") || !strings.Contains(out, "not a directory") {
		t.Fatalf("JSON input suppressed tree validation: %d %s", code, out)
	}
}

func TestRunHelpUsesInputJSONName(t *testing.T) {
	code, out := runCozy(t, t.TempDir(), "run", "proof/input/prepare", "--help")
	if code != 0 || !strings.Contains(out, "--input=") || !strings.Contains(out, "JSON file") || !strings.Contains(out, "--input-tree=") {
		t.Fatalf("input help: %d %s", code, out)
	}
}
