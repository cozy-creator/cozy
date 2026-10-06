package producttest

import (
	"crypto/ed25519"
	"crypto/rand"
	"crypto/x509"
	"encoding/pem"
	"math/big"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/flock"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	"github.com/cozy-creator/cozy/internal/secret"
	v1 "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
)

// A native Run may already have been accepted despite a lost first state frame.
// Cancellation stays pending until the same machine can answer for the run;
// an unreachable endpoint is not proof that work never started.
//
// This computer's stopped machine is proof: its unit and agent have ended, so nothing of the
// run executes. Its cancel settles on record at once, whether the machine never answered or
// was executing the run, and the intent is kept for the machine's journal (runs 2677-2679 and
// 2801-2802 stayed canceling after `cozy machine stop` until the machine next started).
func TestCancelWithUnknownAcceptanceStaysPending(t *testing.T) {
	if *machineHostBinary == "" {
		t.Skip("requires -machine-host: the current native machine fixture")
	}
	for _, arm := range []struct{ name, machine string }{{"local machine never boots", machines.Local},
		{"rental unavailable", "pr-deadpoddeadpoddead0"},
		{"rental cancel recorded while daemon down", "pr-deadpoddeadpoddead0"},
		{"rental native signing key lost", "pr-deadpoddeadpoddead0"},
		{"local machine stopped while executing", machines.Local}} {
		machine := arm.machine
		t.Run(arm.name, func(t *testing.T) {
			root := t.TempDir()
			must(t, os.WriteFile(filepath.Join(root, config.FileName),
				[]byte("tensorhub_url: http://127.0.0.1:1\ntensorhub_token: unreachable\n"), 0o600))
			store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
			fatal(t, problem)
			rented := machine != machines.Local
			if rented {
				layout, problem := home.Open(root)
				fatal(t, problem)
				identity, problem := rental.PendingCreatorIdentity(layout, "unavailable-native-run")
				fatal(t, problem)
				fatal(t, rental.Attach(layout, store, records.Rental{ID: machine, MachineName: "deadpod", SKU: "cpu", AcceleratorModel: "CPU",
					AcceleratorCount: 1, HourlyRateUSDMicros: 100_000, State: "ready", Address: "127.0.0.1:1", Hub: "http://127.0.0.1:1",
					ExpectedWorkerID: "unavailable-native-machine", ExpectedWorkerBootID: "unavailable-native-boot"},
					unavailableNativeCertificate(t), secret.New(""), identity))
			}
			request, _, problem := store.Submit(records.Request{ID: "req-stuck-cancel", IdemKey: "stuck-cancel", Package: "proof/stuck",
				Entrypoint: "generate", Payload: []byte(`{}`), BodyDigest: childDigest("9"), MachineExecutionObserver: true,
				Rental: rented, RequestedRental: map[bool]string{true: machine}[rented]})
			fatal(t, problem)
			fatal(t, store.LinkMachineExecution(request.ID, machine))
			// This is the marker the native transport writes before sending Run.
			fatal(t, store.AppendEvent(request.ID, records.RunV1Sent, 0, map[string]any{"machine": machine}))
			executing := arm.name == "local machine stopped while executing"
			if executing {
				fatal(t, store.AcceptRunV1(request.ID, &v1.RunState{Id: request.ID, Number: 1, State: "running", Attempt: 1}))
			}
			if arm.name == "rental cancel recorded while daemon down" {
				_, problem := store.RequestMachineCancellation(request.ID, "cozy run cancel")
				fatal(t, problem)
			}
			if arm.name == "rental native signing key lost" {
				must(t, os.Remove(home.Paths(root).RentalCreatorIdentity(machine)))
			}
			store.Close()
			if !rented {
				// The full native suite provisions machines automatically. Hold this
				// fixture's real lifecycle lock so it remains genuinely unavailable.
				provisionMachine(t, root)
				dir := filepath.Join(root, "machine")
				must(t, os.MkdirAll(dir, 0700))
				lease, err := os.OpenFile(filepath.Join(dir, "host.lock"), os.O_CREATE|os.O_RDWR, 0600)
				must(t, err)
				must(t, flock.Exclusive(lease))
				defer lease.Close()
				defer flock.Release(lease)
			}
			began := time.Now()
			startDaemonProcess(t, root)
			if code, out := runCozy(t, root, "run", "cancel", request.ID, "--json"); code != 0 ||
				!strings.Contains(out, `"canceling"`) {
				observations, problem := records.Open(filepath.Join(root, "creator.sqlite"))
				fatal(t, problem)
				events, problem := observations.EventsAfter(request.ID, 0, 100)
				fatal(t, problem)
				for _, event := range events {
					t.Logf("%s: %v", event.Type, event.Payload)
				}
				observations.Close()
				t.Fatalf("cancel did not end the run [exit %d]: %s\n%s", code, out, tail(filepath.Join(root, "daemon.log")))
			}

			store, problem = records.Open(filepath.Join(root, "creator.sqlite"))
			fatal(t, problem)
			defer store.Close()
			row, problem := store.RequestRow(request.ID)
			fatal(t, problem)
			owed, problem := store.MachineExecutionOwesWork(request.ID)
			fatal(t, problem)
			if row.State != "canceling" || !owed || time.Since(began) > 10*time.Second {
				t.Fatalf("the run is %s (owes the machine: %v) %s after its cancel", row.State, owed, time.Since(began))
			}
			// Intent stays pending without declaring an unknown acceptance canceled.
			time.Sleep(2 * time.Second)
			row, problem = store.RequestRow(request.ID)
			fatal(t, problem)
			link, problem := store.MachineExecution(request.ID)
			fatal(t, problem)
			if row.State != "canceling" || executing != (len(link.Receipt) > 0) || !link.CancelRequested {
				t.Fatalf("after its cancel the run is %s (receipt %d bytes, intent kept %v)", row.State, len(link.Receipt), link.CancelRequested)
			}
		})
	}
}

func unavailableNativeCertificate(t *testing.T) string {
	t.Helper()
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	template := &x509.Certificate{SerialNumber: big.NewInt(1), NotBefore: time.Now().Add(-time.Minute),
		NotAfter: time.Now().Add(time.Hour), KeyUsage: x509.KeyUsageDigitalSignature,
		ExtKeyUsage: []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth}, DNSNames: []string{"localhost"}}
	der, err := x509.CreateCertificate(rand.Reader, template, template, public, private)
	must(t, err)
	return string(pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der}))
}
