package producttest

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/records"
	v1 "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
)

// A ready rental its Hub has not reached since a probe missed is listed as unreachable for that
// long, not plainly ready (adashino, 2026-10-06); a reached one is plainly ready.
func TestRentalListingSaysWhenItsHubCannotReachARental(t *testing.T) {
	root := t.TempDir()
	t.Cleanup(func() { _, _ = runCozy(t, root, "down") })
	hub := newFakeRentalHub(t, 0)
	hubURL := fmt.Sprintf("http://127.0.0.1:%d", hub.port())
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte(
		"tensorhub_url: "+hubURL+"\ntensorhub_token: rental-idle-test\n"), 0o600))
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	for id, machine := range map[string]string{"pr-lost": "adashino", "pr-fine": "mokou"} {
		hub.add(id, machine)
		fatal(t, store.RecordRental(records.Rental{AcceleratorCount: 1, ID: id, MachineName: machine,
			SKU: "cpu", AcceleratorModel: "CPU", HourlyRateUSDMicros: 100_000, State: "ready", Hub: hubURL}))
	}
	store.Close()
	since := time.Now().Add(-6 * time.Minute).UTC().Format(time.RFC3339)
	hub.set("pr-lost", "unreachable_since", since)

	code, out := runCozy(t, root, "rental", "list", "--no-watch")
	lines := map[string]string{}
	for _, line := range strings.Split(out, "\n") {
		if fields := strings.Fields(line); len(fields) > 0 {
			lines[fields[0]] = line
		}
	}
	if code != 0 || !strings.Contains(lines["adashino"], "ready (unreachable 6m") || strings.Contains(lines["mokou"], "unreachable") {
		t.Fatalf("the listing does not say which rental its Hub cannot reach [exit %d]:\n%s", code, out)
	}
	code, out = runCozy(t, root, "rental", "list", "--json")
	var doc struct {
		Rentals []map[string]any `json:"rentals"`
	}
	if code != 0 || json.Unmarshal([]byte(out), &doc) != nil {
		t.Fatalf("rental list --json [exit %d]:\n%s", code, out)
	}
	for _, row := range doc.Rentals {
		if want := map[string]any{"adashino": since, "mokou": nil}[row["machine"].(string)]; row["unreachable_since"] != want {
			t.Fatalf("%s: unreachable_since %v, want %v", row["machine"], row["unreachable_since"], want)
		}
	}
}

// The Hub ends a rental it could not reach through its idle window (idle_unreached). A run the
// machine confirmed ends FAILED naming that cause, never silently; one it never confirmed goes
// back to the outbox, the cause recorded.
func TestARunOnARentalItsHubCouldNotReachFailsWithTheCause(t *testing.T) {
	root := filepath.Join(scratchBase, "rental-unreached-runs")
	must(t, os.RemoveAll(root))
	must(t, os.MkdirAll(root, 0o755))
	t.Cleanup(func() {
		_, _ = runCozy(t, root, "down")
		if !t.Failed() {
			_ = os.RemoveAll(root)
		}
	})
	hub := newFakeRentalHub(t, 0)
	hubURL := fmt.Sprintf("http://127.0.0.1:%d", hub.port())
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+hubURL+"\n"+
		"tensorhub_token: rental-idle-test\ndaemon:\n  idle_shutdown_s: 0\n"), 0o600))
	hub.packageReleases = map[string]any{"fake/lost@1": rentalReleaseFacts()}
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	hub.add("rental-unreached", "adashino")
	hub.setState("rental-unreached", "degraded", "")
	fatal(t, store.RecordRental(records.Rental{AcceleratorCount: 1, ID: "rental-unreached", MachineName: "adashino", SKU: "cpu",
		AcceleratorModel: "CPU", HourlyRateUSDMicros: 100_000, State: "degraded", Hub: hubURL, Address: "127.0.0.1:1",
		ReadyAt: time.Now().Add(-time.Hour).UTC().Format(time.RFC3339Nano)}))
	for id, accepted := range map[string]bool{"req-confirmed": true, "req-sent": false} {
		body, _ := canonical.Spell(canonical.Digest([]byte(id)))
		_, _, problem := store.Submit(records.Request{ID: id, IdemKey: "idem-" + id, BodyDigest: body, Package: "fake/lost",
			Release: "1", Entrypoint: "generate", Payload: []byte("{}"), Rental: true, Worker: "rental-unreached",
			MachineExecutionObserver: true})
		fatal(t, problem)
		fatal(t, store.LinkMachineExecution(id, "rental-unreached"))
		if send, problem := store.MarkRunV1Sent(id); problem != nil || !send {
			t.Fatalf("%s was not sent: %v %v", id, send, problem)
		}
		if accepted {
			fatal(t, store.AcceptRunV1(id, "rental-unreached", &v1.RunState{Id: id, Number: 1, State: "running", Attempt: 1}))
		}
	}

	startDaemonProcess(t, root)
	hub.set("rental-unreached", "release_cause", "idle_unreached")
	hub.setState("rental-unreached", "released", "")

	deadline := time.Now().Add(60 * time.Second)
	for {
		row, problem := store.RequestRow("req-confirmed")
		fatal(t, problem)
		if row.State == "failed" {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("the confirmed run did not settle after its rental ended: %+v", row)
		}
		time.Sleep(500 * time.Millisecond)
	}
	errType, _, errText, problem := store.SettledFailure("req-confirmed")
	fatal(t, problem)
	events, problem := store.EventsAfter("req-confirmed", 0, 100)
	fatal(t, problem)
	var lost map[string]any
	for _, event := range events {
		if event.Type == "client.machine_lost" {
			lost = event.Payload
		}
	}
	if errType != "machine_execution.state_lost" || !strings.Contains(errText, "idle_unreached") || lost["cause"] != "idle_unreached" {
		t.Fatalf("the confirmed run does not name why its rental ended: %s %q %v", errType, errText, lost)
	}
	sent, problem := store.RequestRow("req-sent")
	fatal(t, problem)
	events, problem = store.EventsAfter("req-sent", 0, 100)
	fatal(t, problem)
	requeued := false
	for _, event := range events {
		requeued = requeued || event.Type == "request.queued" && strings.Contains(fmt.Sprint(event.Payload["reason"]), "idle_unreached")
	}
	if settled(sent.State) || !requeued {
		t.Fatalf("the unconfirmed run was not put back in the outbox with the cause: %+v %v", sent, events)
	}
}
