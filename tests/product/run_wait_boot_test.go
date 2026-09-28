//go:build !windows

package producttest

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/records"
)

// Run 1558 as behaviour. It was queued on nonomiya while the Hub booted the rental, and
// `cozy run list` said `queued - 0.0s`: its park reason was in its own record and the boot
// was on the Hub, and no view said either. The real daemon, against a stand-in Hub whose
// rental stays booting with an advancing boot log, now says what the run waits on in the
// run list, run show and the live watch, as the boot advances and when the Hub replans it
// onto another host; the rental list says the same of the rental.
func TestWaitingRunNamesItsBootingRental(t *testing.T) {
	root := filepath.Join(scratchBase, "run-wait-boot")
	must(t, os.RemoveAll(root))
	must(t, os.MkdirAll(root, 0o755))
	stand := newFakeRentalHub(t, 0)
	hubURL := fmt.Sprintf("http://127.0.0.1:%d", stand.port())
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte(
		"tensorhub_url: "+hubURL+"\ntensorhub_token: rental-idle-test\ndaemon:\n  idle_shutdown_s: 0\n"), 0o600))
	const rental, run = "pr-0886bdd3d3be9a2ec01c", "req-1558"
	stand.add(rental, "nonomiya")
	stand.set(rental, "state", "booting")
	stand.set(rental, "requested_accelerator_model", "NVIDIA H100 80GB HBM3")
	stand.set(rental, "accelerator_count", 4)
	stand.set(rental, "hourly_rate_usd_micros", 11_960_000)

	// The Hub reads the provider back every few seconds and answers on its own clock; the
	// boot log advances while the host works, and stops when it goes silent.
	started := time.Now().Add(-6 * time.Minute)
	var mu sync.Mutex
	boot := map[string]any{"attempt": 1, "state": "booting", "datacenter": "AP-IN-1", "started_at": started,
		"phase": "container", "phase_started_at": started}
	logging := true
	stop := make(chan struct{})
	defer close(stop)
	hubReads := func() {
		mu.Lock()
		defer mu.Unlock()
		now := time.Now().UTC()
		served := map[string]any{"observed_at": now}
		for key, value := range boot {
			served[key] = value
		}
		if logging {
			served["boot_log_at"], served["last_progress_at"] = now.Add(-2*time.Second), now.Add(-2*time.Second)
		}
		stand.set(rental, "boot", served)
	}
	hubReads()
	go func() {
		for {
			select {
			case <-stop:
				return
			case <-time.After(500 * time.Millisecond):
				hubReads()
			}
		}
	}()

	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	fatal(t, store.RecordRental(records.Rental{ID: rental, MachineName: "nonomiya", SKU: "h100-sxm",
		AcceleratorModel: "NVIDIA H100 80GB HBM3", AcceleratorCount: 4, HourlyRateUSDMicros: 11_960_000,
		State: "acquiring", Hub: hubURL}))
	body, _ := canonical.Spell(canonical.Digest([]byte(run)))
	_, _, problem = store.Submit(records.Request{ID: run, IdemKey: "idem-" + run, BodyDigest: body,
		Package: "paul/minimax-h3", Release: "1.22.0", Entrypoint: "long_form", Payload: []byte("{}"),
		Rental: true, Worker: rental, MachineExecutionObserver: true})
	fatal(t, problem)
	fatal(t, store.LinkMachineExecution(run, rental))
	startDaemonProcess(t, root)
	logPath := filepath.Join(root, "daemon.log")

	list := func() (api.Lifecycle, string) {
		t.Helper()
		code, out := runCozy(t, root, "run", "list", "--json", "--full")
		var doc struct {
			Invocations []api.Lifecycle `json:"invocations"`
		}
		if code != 0 || json.Unmarshal([]byte(out), &doc) != nil || len(doc.Invocations) != 1 {
			t.Fatalf("run list [%d]: %s", code, out)
		}
		code, human := runCozy(t, root, "run", "list", "--no-watch")
		if code != 0 {
			t.Fatalf("run list [%d]: %s", code, human)
		}
		return doc.Invocations[0], human
	}
	until := func(what string, done func() bool) {
		t.Helper()
		for deadline := time.Now().Add(30 * time.Second); !done(); time.Sleep(250 * time.Millisecond) {
			if time.Now().After(deadline) {
				t.Fatalf("%s not seen in 30s\n%s", what, tail(logPath))
			}
		}
	}

	// The park is the run's own record; the boot is the Hub's.
	var life api.Lifecycle
	var human string
	until("the parked run's boot", func() bool {
		life, human = list()
		return life.RentalBoot != nil && life.WaitReason != ""
	})
	if life.Status != "queued" || life.Machine != "nonomiya" || life.RentalBoot.Datacenter != "AP-IN-1" ||
		!strings.Contains(life.WaitReason, "rental "+rental+" is acquiring; its retained worker is not attachable") {
		t.Fatalf("the waiting run's record: %+v", life)
	}
	activeLog := regexp.MustCompile(`boot log active (just now|[0-9]s ago)`)
	for _, want := range []string{"waiting for nonomiya to boot · 6m", "AP-IN-1 · container not started · boot log active"} {
		if !strings.Contains(human, want) || !activeLog.MatchString(human) {
			t.Fatalf("run list does not say %q with an advancing boot log:\n%s", want, human)
		}
	}
	code, shown := runCozy(t, root, "run", "show", run)
	if code != 0 || !regexp.MustCompile(`run \d+ queued  paul/minimax-h3/long_form  on nonomiya\n`+
		`waiting for nonomiya to boot · AP-IN-1 · 6m · container not started · boot log active`).MatchString(shown) {
		t.Fatalf("run show [%d]:\n%s", code, shown)
	}
	code, fleet := runCozy(t, root, "rental", "list", "--no-watch")
	if code != 0 || !regexp.MustCompile(`booting · AP-IN-1 · container not started · boot log active`).MatchString(fleet) {
		t.Fatalf("rental list does not name the boot [%d]:\n%s", code, fleet)
	}

	// The live view on a terminal follows the boot as the Hub reports it.
	master, watch := startPTY(t, root, 24, ptyColumns, "run", "watch", run)
	var screenMu sync.Mutex
	var raw strings.Builder
	go func() {
		chunk := make([]byte, 4096)
		for {
			n, err := master.Read(chunk)
			screenMu.Lock()
			raw.Write(chunk[:n])
			screenMu.Unlock()
			if err != nil {
				return
			}
		}
	}()
	drawn := func(what string, pattern *regexp.Regexp) {
		t.Helper()
		until("the live view drawing "+what, func() bool {
			screenMu.Lock()
			defer screenMu.Unlock()
			return pattern.MatchString(screen(raw.String()))
		})
	}
	drawn("the boot", regexp.MustCompile(`(?m)^  ▸ waiting for nonomiya to boot · AP-IN-1 · 6m\d+s\n`+
		`    container not started · boot log active (just now|\ds ago)\n    4 × NVIDIA H100 80GB HBM3 · \$11\.96/hour`))

	// The host goes silent: the boot log stops, and every view says how long ago it moved.
	mu.Lock()
	logging = false
	boot["boot_log_at"] = time.Now().UTC().Add(-2 * time.Second)
	mu.Unlock()
	drawn("a silent boot log", regexp.MustCompile(`boot log active 1[0-9]s ago`))

	// The container starts, then the pod's supervisor answers; the boot log is no longer read.
	mu.Lock()
	boot["container"], boot["runtime_observed"] = "running", true
	boot["phase"], boot["phase_started_at"] = "host", time.Now().UTC()
	mu.Unlock()
	drawn("the started container", regexp.MustCompile(`(?m)^    starting supervisor \(\d+s\)\n`))
	mu.Lock()
	boot["phase"], boot["phase_started_at"] = "runtime", time.Now().UTC()
	mu.Unlock()
	until("run list naming the answering host", func() bool {
		_, human = list()
		return regexp.MustCompile(`AP-IN-1 · starting Runtime \(\d+s\)`).MatchString(human)
	})

	// The Hub ends the attempt and replans it onto another host; the run stays parked on
	// the rental and continues on the replacement.
	mu.Lock()
	boot = map[string]any{"attempt": 2, "state": "booting", "datacenter": "EU-RO-1", "activity": "pulling_image",
		"started_at": time.Now().UTC().Add(-40 * time.Second), "phase": "container",
		"phase_started_at": time.Now().UTC().Add(-35 * time.Second),
		"replanned_from": map[string]any{"attempt": 1, "datacenter": "AP-IN-1", "started_at": started,
			"ended_at": started.Add(5 * time.Minute), "failure_code": "provider_never_started", "observed_max_ms": 294_000}}
	logging = true
	mu.Unlock()
	drawn("the replan", regexp.MustCompile(`(?m)^  ▸ boot attempt 2 of rental nonomiya · now EU-RO-1 · 6m\d+s\n`+
		`    replanned from AP-IN-1 after 5m \(no progress past the observed 4\.9m max\)\n`+
		`    pulling image · boot log active (just now|\ds ago)\n`))
	until("run show naming the replan", func() bool {
		_, shown = runCozy(t, root, "run", "show", run)
		return regexp.MustCompile(`boot attempt 2 of rental nonomiya · replanned from AP-IN-1 after 5m ` +
			`\(no progress past the observed 4\.9m max\) · now EU-RO-1 · \d+s · pulling image · boot log active`).MatchString(shown)
	})
	life, human = list()
	if life.Status != "queued" || life.RentalBoot == nil || life.RentalBoot.Attempt != 2 ||
		!strings.Contains(human, "boot attempt 2 of rental nonomiya · ") ||
		!strings.Contains(human, "now EU-RO-1 · pulling image · boot log active") {
		t.Fatalf("the run does not continue on the replacement: %+v\n%s", life.RentalBoot, human)
	}
	code, fleet = runCozy(t, root, "rental", "list", "--no-watch")
	if code != 0 || !strings.Contains(fleet, "booting · attempt 2 · now EU-RO-1 · pulling image · boot log active") {
		t.Fatalf("rental list does not name the replan [%d]:\n%s", code, fleet)
	}

	// Ctrl-C detaches the watcher; the run keeps waiting.
	_, _ = master.Write([]byte{3})
	_ = watch.Wait()
	screenMu.Lock()
	final := screen(raw.String())
	screenMu.Unlock()
	if !strings.Contains(final, "detached from run") {
		t.Fatalf("the watch did not detach:\n%s", final)
	}
	if row, problem := store.RequestRow(run); problem != nil || row == nil || settled(row.State) {
		t.Fatalf("the watch changed the run: %+v %v", row, problem)
	}
}
