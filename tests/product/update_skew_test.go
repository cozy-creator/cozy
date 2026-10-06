package producttest

import (
	"encoding/json"
	"flag"
	"fmt"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

var daemonBeforeTarget = flag.String("daemon-before-target", "",
	"a cozy (0.2.0 candidate a2200172) whose daemon installs the newest published pair for an update naming nothing")

// A cozy and a daemon a release apart never update a rental to software nobody named. This
// cozy names the Hub's target versions in every update it asks for, and asks for none when the
// Hub names no target; the older daemon installed the newest published release for a request
// naming nothing (cut condition 13: a fresh rental downgraded at boot). The other way round, this
// daemon treats such a request from the older cozy as installing nothing.
func TestUpdatesAcrossAReleaseNameTheirSoftware(t *testing.T) {
	if *daemonBeforeTarget == "" {
		t.Skip("requires -daemon-before-target: the a2200172 cozy")
	}
	if version, err := exec.Command(*daemonBeforeTarget, "--version").CombinedOutput(); err != nil || !strings.Contains(string(version), "a2200172") {
		t.Fatalf("-daemon-before-target is not the a2200172 cozy: %s %v", version, err)
	}
	selection := func(root, rental string) map[string]any {
		t.Helper()
		store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
		fatal(t, problem)
		defer store.Close()
		row, problem := store.RuntimeUpdate(rental)
		fatal(t, problem)
		if row == nil {
			return nil
		}
		var chosen map[string]any
		must(t, json.Unmarshal(row.Selection, &chosen))
		return chosen
	}

	t.Run("new cozy, older daemon", func(t *testing.T) {
		root, _, stand := rentalEndRoot(t, "skew-boot")
		stand.publishListing()
		stand.setSKUs(cpuSKU)
		stand.mu.Lock()
		stand.rent = func(request map[string]any) map[string]any {
			id := fmt.Sprintf("pr-skew-%s", request["name"])
			return map[string]any{"rental_id": id, "name": request["name"], "state": "ready",
				"requested_accelerator_model": "CPU", "accelerator_count": 1, "hourly_rate_usd_micros": 100_000,
				"worker_address": "127.0.0.1:1", "media_address": "127.0.0.1:2", "cert_pem": "fixture",
				"worker_id": "skew-worker", "worker_boot_id": "skew-boot",
				"creator_public_key": request["creator_public_key"], "media_token_sha256": []any{request["media_token_sha256"]}}
		}
		stand.mu.Unlock()
		startDaemonBinary(t, *daemonBeforeTarget, root)

		// No target: the new rental keeps its image's software, and the older daemon is asked nothing.
		if code, out := runCozy(t, root, "rental", "new", "cpu", "--idempotency-key=no-target", "--json"); code != 0 {
			t.Fatalf("rental new [exit %d]: %s", code, out)
		}
		rental := rentalIDByOperation(t, root, "no-target")
		if chosen := selection(root, rental); chosen != nil {
			t.Fatalf("with no target the older daemon was asked to update %s: %v", rental, chosen)
		}
		// A target: the request names its exact versions, which the older daemon installs as named.
		stand.mu.Lock()
		stand.software = map[string]string{"runtime": "0.18.102", "tensorfs": "0.3.93"}
		stand.mu.Unlock()
		if code, out := runCozy(t, root, "rental", "new", "cpu", "--idempotency-key=a-target", "--json"); code != 0 {
			t.Fatalf("rental new [exit %d]: %s", code, out)
		}
		rental = rentalIDByOperation(t, root, "a-target")
		if chosen := selection(root, rental); chosen["runtime_version"] != "0.18.102" || chosen["tensorfs_version"] != "0.3.93" {
			t.Fatalf("the update the older daemon received named no versions: %v", chosen)
		}
		// Asked by hand with nothing named and no target, this cozy refuses rather than send it.
		stand.mu.Lock()
		stand.software = nil
		stand.mu.Unlock()
		code, out := runCozy(t, root, "rental", "update", rental, "--json")
		if code == 0 || !strings.Contains(out, "rental.no_target_software") {
			t.Fatalf("rental update with nothing named and no target [exit %d]: %s", code, out)
		}
	})

	t.Run("older cozy, new daemon", func(t *testing.T) {
		root, store, _, _, _, _ := statusRentalHub(t)
		command := exec.Command("/usr/bin/nice", "-n", "19", *daemonBeforeTarget, "rental", "update", "tessa", "--json")
		command.Env = childEnv(t, root)
		data, _ := command.CombinedOutput()
		code, out := command.ProcessState.ExitCode(), string(data)
		if code != 0 {
			t.Fatalf("the older cozy's rental update [exit %d]: %s", code, out)
		}
		row, problem := store.RuntimeUpdate(parityRental)
		fatal(t, problem)
		var result struct {
			Unchanged bool
			Note      string
		}
		if row == nil || row.State != "succeeded" || json.Unmarshal(row.Result, &result) != nil || !result.Unchanged || result.Note != "the update named no software" {
			t.Fatalf("an update naming nothing changed the machine: %+v %s", row, row.Result)
		}
	})
}

// rentalIDByOperation is the rental a `rental new` with this idempotency key bought.
func rentalIDByOperation(t *testing.T, root, key string) string {
	t.Helper()
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	operation, problem := store.RentalOperation(key)
	fatal(t, problem)
	if operation == nil || operation.RentalID == "" {
		t.Fatalf("no rental recorded for %s: %+v", key, operation)
	}
	return operation.RentalID
}
