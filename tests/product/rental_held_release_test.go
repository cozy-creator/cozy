package producttest

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// A run on a rental executes the release that machine holds, checked against that release's
// interface; this computer's installation decides only this computer's runs. A newer field
// the held release lacks is ignored with a warning naming the release, and `cozy package
// update --rental` moves the machine to the newest.
func TestARentalRunsTheReleaseItsMachineHolds(t *testing.T) {
	h, root, _, store := parityMachines(t)
	publishParityRelease(t, h, root, parityProject(t))
	if code, out := runCozy(t, root, "package", "install", parityPublished); code != 0 {
		t.Fatalf("install here [exit %d]\n%s", code, out)
	}
	run := func(key, want, release string, extra ...string) string {
		t.Helper()
		args := append([]string{"run", parityPublished + "/add", "value=41", "--rental=tessa", "--await", "--json", "--idempotency-key", key}, extra...)
		code, out := runCozy(t, root, args...)
		if code != 0 || !strings.Contains(out, want) {
			t.Fatalf("%s [exit %d]\n%s", key, code, out)
		}
		request, problem := store.RequestByIdempotencyKey(key)
		fatal(t, problem)
		if request.Release != release {
			t.Fatalf("%s ran %s, not %s", key, request.Release, release)
		}
		_, shown := runCozy(t, root, "run", "show", request.ID)
		if !strings.Contains(shown, parityPublished+"/add "+release) {
			t.Fatalf("run show does not name release %s\n%s", release, shown)
		}
		return out
	}
	run("held-none", `"value":42`, parityVersion) // the machine holds nothing: the newest, installed there

	next := parityProject(t)
	patch := func(name, old, new string) {
		path := filepath.Join(next, name)
		raw, err := os.ReadFile(path)
		must(t, err)
		must(t, os.WriteFile(path, []byte(strings.Replace(string(raw), old, new, 1)), 0o600))
	}
	patch("pyproject.toml", `version="0.0.1"`, `version="0.0.2"`)
	patch("machine_parity.py", "    value: int\n\n\nclass AddResult", "    value: int\n    bonus: int = 0\n\n\nclass AddResult")
	patch("machine_parity.py", "payload.value + 1)", "payload.value + 1 + payload.bonus)")
	if out, err := exec.Command("uv", "lock", "--project", next).CombinedOutput(); err != nil {
		t.Fatalf("locking 0.0.2: %v\n%s", err, out)
	}
	publishParityReleaseAt(t, h, root, next, "0.0.2")

	out := run("held-old", `"value":42`, parityVersion, "bonus=1")
	if !strings.Contains(out, "bonus") || !strings.Contains(out, "add "+parityVersion+"'s interface") ||
		!strings.Contains(out, "cozy package update "+parityPublished+" --rental=tessa") {
		t.Fatalf("the ignored field does not name the held release and its update\n%s", out)
	}
	if code, out := runCozy(t, root, "package", "update", parityPublished, "--rental=tessa"); code != 0 {
		t.Fatalf("update on the rental [exit %d]\n%s", code, out)
	}
	run("held-new", `"value":43`, "0.0.2", "bonus=1") // this computer still holds 0.0.1
}
