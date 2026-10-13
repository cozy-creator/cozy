package producttest

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// What a package logs reaches `cozy run show`, from this computer's machine and a rental: info
// lines in its log, warnings beside the run's own, and the machine's warnings (a kernel pin no
// call applied) too.
func TestRunShowCarriesThePackagesLog(t *testing.T) {
	_, root, _, store := parityMachines(t)
	project := parityProject(t)
	must(t, os.WriteFile(filepath.Join(project, "machine_parity.py"), []byte(`import msgspec
from cozy_runtime.author import App, Telemetry


class AddRequest(msgspec.Struct):
    value: int


class AddResult(msgspec.Struct):
    value: int


app = App()


@app.job
def add(payload: AddRequest, tel: Telemetry) -> AddResult:
    tel.log("adding one", amount=payload.value)
    tel.log("the value is odd", level="warning")
    return AddResult(value=payload.value + 1)
`), 0o600))
	if code, out := runCozy(t, root, "package", "install", project); code != 0 {
		t.Fatalf("install [exit %d]\n%s", code, out)
	}
	for _, venue := range [][]string{nil, {"--rental=tessa"}} {
		key := "logs-" + strings.Join(venue, "")
		args := append([]string{"run", parityPackage + "/add", "value=41", "--attention-kernel=sdpa", "--await", "--json", "--idempotency-key", key}, venue...)
		if code, out := runCozy(t, root, args...); code != 0 || !strings.Contains(out, `"value":42`) {
			t.Fatalf("%v [exit %d]\n%s", venue, code, out)
		}
		request, problem := store.RequestByIdempotencyKey(key)
		fatal(t, problem)
		_, human := runCozy(t, root, "run", "show", request.ID)
		for _, want := range []string{"\nLOG\n", "adding one amount=41", "warning: the value is odd", "warning: attention pin sdpa was never applied"} {
			if !strings.Contains(human, want) {
				t.Fatalf("%v: run show lacks %q\n%s", venue, want, human)
			}
		}
		_, document := runCozy(t, root, "run", "show", request.ID, "--json")
		var shown struct {
			Logs []struct{ Level, Text string } `json:"logs"`
		}
		if json.Unmarshal([]byte(document), &shown) != nil || len(shown.Logs) != 1 || shown.Logs[0].Text != "adding one amount=41" {
			t.Fatalf("%v: run show --json logs are not the package's line\n%s", venue, document)
		}
	}
}
