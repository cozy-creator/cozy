package producttest

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// A cancel never waits on a machine that has not accepted the run. A run whose submission
// reached this computer's machine, which then never boots again, and one sent to a rental
// whose pod died, both end canceled the moment they are canceled (runs 1583, 1589 sat in
// "canceling" until a machine that never came back, or a preparation, answered).
func TestACancelBeforeAcceptanceEndsTheRunAtOnce(t *testing.T) {
	for _, machine := range []string{machines.Local, "pr-deadpoddeadpoddead0"} {
		t.Run(map[bool]string{true: "local machine never boots", false: "rental died"}[machine == machines.Local], func(t *testing.T) {
			root := t.TempDir()
			must(t, os.WriteFile(filepath.Join(root, config.FileName),
				[]byte("tensorhub_url: http://127.0.0.1:1\ntensorhub_token: unreachable\n"), 0o600))
			store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
			fatal(t, problem)
			rented := machine != machines.Local
			if rented {
				cert := filepath.Join(root, "pod.pem")
				must(t, os.WriteFile(cert, []byte("pod"), 0o600))
				fatal(t, store.RecordRental(records.Rental{ID: machine, MachineName: "deadpod", SKU: "cpu", AcceleratorModel: "CPU",
					AcceleratorCount: 1, HourlyRateUSDMicros: 100_000, State: "ready", Address: "127.0.0.1:1", CertPath: cert, Hub: "http://127.0.0.1:1"}))
			}
			request, _, problem := store.Submit(records.Request{ID: "req-stuck-cancel", IdemKey: "stuck-cancel", Package: "proof/stuck",
				Entrypoint: "generate", Payload: []byte(`{}`), BodyDigest: childDigest("9"), MachineExecutionObserver: true,
				Rental: rented, RequestedRental: map[bool]string{true: machine}[rented]})
			fatal(t, problem)
			fatal(t, store.LinkMachineExecution(request.ID, machine))
			// The machine was preparing the release when it went away: sent, never accepted.
			fatal(t, store.RecordMachineSubmission(request.ID, &pb.MachineExecutionSubmit{SubmissionId: "stuck-cancel",
				ExpectedExecutionWorkspaceId: "workspace", Offer: &pb.AttemptOffer{RequestId: request.ID},
				PayloadCanonicalBytes: []byte(`{}`), ReleaseRoot: &pb.ReleaseRoot{Package: "proof/stuck", Release: "1.0.0", Entrypoint: "generate"}}))
			store.Close()
			startDaemonProcess(t, root)

			began := time.Now()
			code, out := runCozy(t, root, "run", "cancel", request.ID, "--json")
			if code != 0 || !strings.Contains(out, `"canceled"`) {
				t.Fatalf("cancel did not end the run [exit %d]: %s", code, out)
			}
			store, problem = records.Open(filepath.Join(root, "creator.sqlite"))
			fatal(t, problem)
			defer store.Close()
			row, problem := store.RequestRow(request.ID)
			fatal(t, problem)
			owed, problem := store.MachineExecutionOwesWork(request.ID)
			fatal(t, problem)
			if row.State != "canceled" || owed || time.Since(began) > 10*time.Second {
				t.Fatalf("the run is %s (owes the machine: %v) %s after its cancel", row.State, owed, time.Since(began))
			}
			// It stays canceled: the daemon neither submits it again nor waits on the machine.
			time.Sleep(2 * time.Second)
			row, problem = store.RequestRow(request.ID)
			fatal(t, problem)
			link, problem := store.MachineExecution(request.ID)
			fatal(t, problem)
			if row.State != "canceled" || len(link.Receipt) != 0 || !link.CancelRequested {
				t.Fatalf("after its cancel the run is %s (receipt %d bytes, intent kept %v)", row.State, len(link.Receipt), link.CancelRequested)
			}
		})
	}
}
