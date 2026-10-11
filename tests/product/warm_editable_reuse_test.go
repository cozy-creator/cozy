package producttest

import (
	"strings"
	"testing"
)

// An unchanged editable package run again is neither captured nor uploaded again: a job runs
// the snapshot it already has, and a machine that holds an installation reopens it. On this
// computer's machine and on a rental, with the real Host and Runtime.
func TestAnUnchangedEditablePackageIsNotCapturedOrUploadedAgain(t *testing.T) {
	_, root, _, store := parityMachines(t)
	if code, out := runCozy(t, root, "package", "install", parityProject(t)); code != 0 {
		t.Fatalf("editable install [exit %d]\n%s", code, out)
	}
	for venue, args := range map[string][]string{"local": nil, "rental": {"--rental=tessa"}} {
		for _, call := range []struct{ function, want string }{{"add", `"value":42`}, {"echo", `"value":82`}} {
			var installations []string
			for _, key := range []string{"cold", "warm"} {
				idem := "reuse-" + venue + "-" + call.function + "-" + key
				code, out := runCozy(t, root, append([]string{"run", parityPackage + "/" + call.function, "value=41", "--await", "--json",
					"--idempotency-key", idem}, args...)...)
				if code != 0 || !strings.Contains(out, call.want) {
					t.Fatalf("%s %s on %s [exit %d]\n%s", key, call.function, venue, code, out)
				}
				request, problem := store.RequestByIdempotencyKey(idem)
				fatal(t, problem)
				installations = append(installations, request.LocalInstallationID)
				events, problem := store.EvidenceEvents(request.ID, 1000)
				fatal(t, problem)
				uploaded := false
				for _, event := range events {
					uploaded = uploaded || event.Type == "machine.package_upload_started"
				}
				if key == "warm" && uploaded {
					t.Fatalf("the warm %s on %s uploaded its package again", call.function, venue)
				}
			}
			if installations[0] == "" || installations[0] != installations[1] {
				t.Fatalf("the warm %s on %s ran installation %q, the cold one %q", call.function, venue, installations[1], installations[0])
			}
		}
	}
}
