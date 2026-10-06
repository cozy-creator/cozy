package producttest

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/machinev1"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	"github.com/cozy-creator/cozy/internal/secret"
	"github.com/cozy-creator/cozy/internal/workertls"
)

// A rental's newly attached boot runs the Hub's target software before it takes work: the
// daemon sends its machine Run kind: update with the target's published versions, once per
// boot, older or newer alike. A boot already on the target is left as it is; a target the
// machine cannot install leaves it serving its own software, and that boot is not tried again.
func TestARentalBootFollowsTheHubsTarget(t *testing.T) {
	root, store, launch, identity, _, h := statusRentalHub(t)
	layout, problem := home.Open(root)
	fatal(t, problem)
	token, problem := rental.MediaToken(layout, parityRental)
	fatal(t, problem)
	cert := string(pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: launch.Leaf}))
	pin, err := workertls.ParsePin([]byte(cert))
	must(t, err)
	public, err := base64.RawURLEncoding.DecodeString(identity.PublicKey())
	must(t, err)
	machine, err := machinev1.Dial(launch.Addr, pin.TLSConfig(), launch.WorkerID, machinev1.Signer{Public: public, Sign: identity.Sign})
	must(t, err)
	defer machine.Close()
	running := func() map[string]string {
		t.Helper()
		ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
		defer cancel()
		for {
			frame, err := machine.Status(ctx)
			if err == nil && frame.Phase == "ready" {
				if frame.BootId != launch.BootID {
					t.Fatalf("the update changed the machine's boot: %s", frame.BootId)
				}
				return map[string]string{"runtime": frame.Runtime, "tensorfs": frame.Tensorfs}
			}
			if ctx.Err() != nil {
				t.Fatalf("the machine did not answer ready: %v", err)
			}
			time.Sleep(200 * time.Millisecond)
		}
	}
	booted := running()
	previous := ""
	// The Hub lists the rental on its machine's boot while this host still holds an earlier one:
	// the next reconcile attaches the boot and brings it to target.
	boot := func(target map[string]string) *records.RuntimeUpdate {
		t.Helper()
		fatal(t, store.RebootRental(parityRental, launch.WorkerID, "an-earlier-boot", launch.Addr, launch.MediaAddr))
		h.fakeRentalHub.mu.Lock()
		h.software = target
		listed := h.rentals[parityRental]
		listed["worker_id"], listed["worker_boot_id"], listed["cert_pem"] = launch.WorkerID, launch.BootID, cert
		listed["creator_public_key"], listed["media_token_sha256"] = identity.PublicKey(), []string{secret.HashHex(token)}
		h.fakeRentalHub.mu.Unlock()
		if code, out := runCozy(t, root, "rental", "list", "--no-watch", "--json"); code != 0 {
			t.Fatalf("rental list [exit %d]\n%s", code, out)
		}
		eventually(t, root, "the boot's update ends", func() bool {
			row, problem := store.RuntimeUpdate(parityRental)
			return problem == nil && row != nil && row.ID != previous && row.BootID == launch.BootID && !row.Active()
		})
		row, problem := store.RuntimeUpdate(parityRental)
		fatal(t, problem)
		previous = row.ID
		return row
	}
	result := func(row *records.RuntimeUpdate) (out struct {
		To        map[string]string
		Unchanged bool
	}) {
		must(t, json.Unmarshal(row.Result, &out))
		return out
	}

	if row := boot(booted); row.State != "succeeded" || !result(row).Unchanged {
		t.Fatalf("a boot already on the target was updated: %+v", row)
	}
	// Moving the target back to an older release is a rollback; forward again, a release.
	if older := olderRelease(t, "cozy-runtime", booted["runtime"]); older != "" {
		for _, version := range []string{older, booted["runtime"]} {
			target := map[string]string{"runtime": version, "tensorfs": booted["tensorfs"]}
			if row := boot(target); row.State != "succeeded" || result(row).Unchanged || !samePair(running(), target) {
				t.Fatalf("the boot did not follow the target to %v: %+v, running %v", target, row, running())
			}
		}
	} else {
		t.Logf("Runtime %s has no older published release: the install in both directions is not exercised", booted["runtime"])
	}
	failed := boot(map[string]string{"runtime": "999.0.0", "tensorfs": booted["tensorfs"]})
	if failed.State != "failed" || !strings.Contains(failed.Error, "999.0.0") || !samePair(running(), booted) {
		t.Fatalf("an uninstallable target did not leave the boot serving its own software: %+v, running %v", failed, running())
	}
	if code, out := runCozy(t, root, "rental", "list", "--no-watch", "--json"); code != 0 || strings.Contains(out, "updating its software") {
		t.Fatalf("the failed boot still holds the rental [exit %d]\n%s", code, out)
	}
	if again, problem := store.RuntimeUpdate(parityRental); problem != nil || again.ID != failed.ID {
		t.Fatalf("the same boot was tried again: %+v %v", again, problem)
	}
	// Asked with nothing named, an update is the Hub's target too.
	h.fakeRentalHub.mu.Lock()
	h.software = booted
	h.fakeRentalHub.mu.Unlock()
	code, out := runCozy(t, root, "rental", "update", "tessa", "--json")
	var shown map[string]any
	if code != 0 || json.Unmarshal([]byte(lastJSONLine(out)), &shown) != nil || shown["status"] != "unchanged: it already runs this software" {
		t.Fatalf("rental update with nothing named [exit %d]\n%s", code, out)
	}
}

// This computer's machine follows the Hub's target when it starts, unless its owner pinned it
// by installing named files; one that cannot install the target keeps serving its own.
func TestALocalMachineStartFollowsTheHubsTargetUnlessPinned(t *testing.T) {
	if *machineHostBinary == "" || *machineRuntimeWheel == "" || *machineTensorFSWheel == "" {
		t.Skip("requires -machine-host and a -machine-runtime-wheel/-machine-tensorfs-wheel pair")
	}
	h := newMachineHub(t)
	root, err := os.MkdirTemp("", "czf")
	must(t, err)
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+h.server.URL+
		"\ntensorhub_token: rental-idle-test\ndaemon:\n  idle_shutdown_s: 0\n"), 0o600))
	t.Cleanup(func() {
		_, _ = runCozy(t, root, "machine", "stop")
		_, _ = runCozy(t, root, "down")
		if t.Failed() {
			t.Logf("evidence retained at %s\nlocal machine log:\n%s", root, tail(filepath.Join(root, "machine", "host.log")))
		} else {
			_ = removeAllForce(root)
		}
	})
	target := func(runtime, tensorfs string) {
		h.fakeRentalHub.mu.Lock()
		h.software = map[string]string{"runtime": runtime, "tensorfs": tensorfs}
		h.fakeRentalHub.mu.Unlock()
	}
	start := func() (map[string]any, string) {
		t.Helper()
		_, _ = runCozy(t, root, "machine", "stop")
		code, stdout, stderr := runCozyStreams(t, root, "machine", "start")
		if code != 0 {
			t.Fatalf("machine start [exit %d]\n%s%s", code, stdout, stderr)
		}
		code, out := runCozy(t, root, "machine", "show", "--json")
		var shown map[string]any
		if code != 0 || json.Unmarshal([]byte(lastJSONLine(out)), &shown) != nil || shown["phase"] != "ready" {
			t.Fatalf("machine show [exit %d]\n%s", code, out)
		}
		return shown, stderr
	}
	if code, out := runCozy(t, root, "machine", "install", "--host", *machineHostBinary, "--runtime-wheel", *machineRuntimeWheel, "--tensorfs-wheel", *machineTensorFSWheel); code != 0 {
		t.Fatalf("machine install [exit %d]\n%s", code, out)
	}
	installed, _ := start()
	runtime, tensorfs := installed["runtime_version"].(string), installed["tensorfs_version"].(string)

	target("999.0.0", "999.0.0")
	if shown, said := start(); shown["runtime_version"] != runtime || strings.Contains(said, "keeps its software") {
		t.Fatalf("a pinned machine followed the target: %v\n%s", shown, said)
	}
	// As an install of published releases records it: the machine follows the Hub.
	record := filepath.Join(root, "machine", "installed.json")
	raw, err := os.ReadFile(record)
	must(t, err)
	var pinned map[string]any
	must(t, json.Unmarshal(raw, &pinned))
	delete(pinned, "pinned")
	raw, err = json.Marshal(pinned)
	must(t, err)
	must(t, os.WriteFile(record, raw, 0o600))
	if shown, said := start(); shown["runtime_version"] != runtime || !strings.Contains(said, "keeps its software") || !strings.Contains(said, "999.0.0") {
		t.Fatalf("an uninstallable target did not leave the machine serving its own software: %v\n%s", shown, said)
	}
	if older := olderRelease(t, "cozy-runtime", runtime); older != "" {
		target(older, tensorfs)
		if shown, said := start(); shown["runtime_version"] != older {
			t.Fatalf("the machine did not follow the target back to %s: %v\n%s", older, shown, said)
		}
		target(runtime, tensorfs)
		if code, out := runCozy(t, root, "machine", "install"); code != 0 {
			t.Fatalf("machine install of the target [exit %d]\n%s", code, out)
		}
		if shown, _ := start(); shown["runtime_version"] != runtime {
			t.Fatalf("installing with nothing named did not take the target %s: %v", runtime, shown)
		}
	}
}

// olderRelease is the published release of distribution just before version, or "" when
// version is not published or is its first.
func olderRelease(t *testing.T, distribution, version string) string {
	t.Helper()
	response, err := http.Get("https://pypi.org/pypi/" + distribution + "/json")
	if err != nil {
		t.Logf("the package index did not answer: %v", err)
		return ""
	}
	defer response.Body.Close()
	var project struct {
		Releases map[string][]struct {
			Yanked bool `json:"yanked"`
		} `json:"releases"`
	}
	if json.NewDecoder(response.Body).Decode(&project) != nil {
		return ""
	}
	var all []string
	for release, files := range project.Releases {
		if len(files) > 0 && !files[0].Yanked {
			all = append(all, release)
		}
	}
	slices.SortFunc(all, comparePatch)
	at := slices.Index(all, version)
	if at < 1 {
		return ""
	}
	return all[at-1]
}

// comparePatch orders plain X.Y.Z versions numerically; any other spelling sorts first.
func comparePatch(a, b string) int {
	parse := func(v string) []int {
		var out []int
		for _, part := range strings.Split(v, ".") {
			n := 0
			for _, c := range part {
				if c < '0' || c > '9' {
					return nil
				}
				n = n*10 + int(c-'0')
			}
			out = append(out, n)
		}
		return out
	}
	return slices.Compare(parse(a), parse(b))
}

func samePair(a, b map[string]string) bool {
	return a["runtime"] == b["runtime"] && a["tensorfs"] == b["tensorfs"]
}
