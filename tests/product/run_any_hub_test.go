package producttest

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
)

// A run goes straight to its machine; the hub that rented the machine does not matter. A
// run on a rental bought from a hub other than the current one, which does not even answer,
// finds it by its name with no --tensorhub; `run list` shows its machine and no hub; and
// show, watch and cancel need no hub. (A machine name is unique on this computer: its
// records refuse a second rental by one name, so a name is never ambiguous.)
func TestARunFindsItsRentalOnAnyHub(t *testing.T) {
	h, root, _, store := parityMachines(t)
	const elsewhere = "http://127.0.0.1:1"
	// The current hub is another one, which does not even answer; the rental's hub is b.
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: a\nhubs:\n  a: "+elsewhere+
		"\n  b: "+h.server.URL+"\ndaemon:\n  idle_shutdown_s: 0\n"), 0o600))
	if code, out := runCozy(t, root, "package", "install", parityProject(t), "--editable"); code != 0 {
		t.Fatalf("editable install [exit %d]\n%s", code, out)
	}
	code, out := runCozy(t, root, "run", parityPackage+"/add", "value=41", "--rental=tessa", "--await", "--json", "--idempotency-key", "any-hub")
	if code != 0 || !strings.Contains(out, `"value":42`) {
		t.Fatalf("a run on a rental of another hub [exit %d]\n%s", code, out)
	}
	request, problem := store.RequestByIdempotencyKey("any-hub")
	fatal(t, problem)
	link, problem := store.MachineExecution(request.ID)
	fatal(t, problem)
	if link == nil || link.MachineID != parityRental || !link.Collected {
		t.Fatalf("the run did not execute on tessa: %+v", link)
	}
	if code, out = runCozy(t, root, "run", "list", "--no-watch"); code != 0 || !regexp.MustCompile(`(?m)^NUMBER +TARGET +MACHINE `).MatchString(out) ||
		regexp.MustCompile(`(?m)^NUMBER .*\bHUB\b`).MatchString(out) || !strings.Contains(out, "tessa") {
		t.Fatalf("run list does not show the run's machine without a hub [exit %d]\n%s", code, out)
	}

	// Every run-side verb answers from this computer's records and the run's machine, with
	// the current hub not answering.
	for _, verb := range [][]string{{"run", "show", request.ID}, {"run", "watch", request.ID}, {"run", "cancel", request.ID}} {
		if code, out := runCozy(t, root, append(verb, "--json")...); code != 0 || !strings.Contains(out, "completed") {
			t.Fatalf("%v needed a hub [exit %d]\n%s", verb, code, out)
		}
	}
}
