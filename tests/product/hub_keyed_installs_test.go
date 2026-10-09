package producttest

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/machines"
)

// One org/name published at two hubs is two packages. Each installs beside the other, the
// list shows both, and each hub's reference runs that hub's own install on this computer's
// machine with the real Host and Runtime: a (+1) answers 42 and b (+100) answers 141.
func TestOneNameFromTwoHubsInstallsAndRunsSeparately(t *testing.T) {
	a, root, _, store := parityMachines(t)
	b := newMachineHub(t)
	publishParityRelease(t, a, root, parityProject(t))
	project := parityProject(t)
	body := filepath.Join(project, "machine_parity.py")
	raw, err := os.ReadFile(body)
	must(t, err)
	must(t, os.WriteFile(body, []byte(strings.ReplaceAll(string(raw), "payload.value + 1", "payload.value + 100")), 0o600))
	publishParityRelease(t, b, root, project)
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: a\ntensorhub_token: rental-idle-test\nhubs:\n  a: "+
		a.server.URL+"\n  b: "+b.server.URL+"\ndaemon:\n  idle_shutdown_s: 0\n"), 0o600))
	fixtureExecutionAccess(t, root, b.server, b.worker.Config.Handler)

	for _, hub := range []string{"a", "b"} {
		if code, out := runCozy(t, root, "package", "install", parityPublished, "--tensorhub="+hub, "--json"); code != 0 || !strings.Contains(out, `"hub":"`+hub+`"`) {
			t.Fatalf("install from hub %s [exit %d]\n%s", hub, code, out)
		}
	}
	code, listed := runCozy(t, root, "package", "list", "--json", "--full")
	var doc struct {
		Packages []struct{ Package, Hub, Scope string }
	}
	must(t, json.Unmarshal([]byte(listed), &doc))
	hubs := map[string]string{}
	for _, row := range doc.Packages {
		if row.Package == parityPublished {
			hubs[row.Hub] = row.Scope
		}
	}
	if code != 0 || hubs["a"] != "current hub" || hubs["b"] != "other hub" {
		t.Fatalf("both hubs' installations are not listed [exit %d]: %v\n%s", code, hubs, listed)
	}
	for hub, origin := range map[string]string{"a": a.server.URL, "b": b.server.URL} {
		if _, installed, problem := store.ActivePackage(origin, parityPublished); problem != nil || installed == nil || installed.Hub != origin {
			t.Fatalf("hub %s's install is not active beside the other: %+v %v", hub, installed, problem)
		}
	}

	for hub, want := range map[string]string{"a": `"value":42`, "b": `"value":141`} {
		key := "two-hubs-" + hub
		code, out := runCozy(t, root, "run", parityPublished+"/add", "value=41", "--tensorhub="+hub, "--await", "--json", "--idempotency-key", key)
		if code != 0 || !strings.Contains(out, want) {
			t.Fatalf("run at hub %s did not run that hub's package [exit %d]\n%s", hub, code, out)
		}
		request, problem := store.RequestByIdempotencyKey(key)
		fatal(t, problem)
		link, problem := store.MachineExecution(request.ID)
		fatal(t, problem)
		if link == nil || link.MachineID != machines.Local || !link.Collected {
			t.Fatalf("run at hub %s did not run its install on this computer: %+v %+v", hub, request, link)
		}
	}
}
