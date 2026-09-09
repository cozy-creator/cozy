package producttest

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
)

// Two independent database connections model the foreground handoff racing the
// daemon's release. Channels stop at the actual credential-staging boundary.
func TestRentalAttachmentSerializesWithRelease(t *testing.T) {
	for _, mode := range []string{"released", "release_requested", "attachment-first", "stage-failed", "already-attached"} {
		t.Run(mode, func(t *testing.T) {
			layout, problem := home.Open(t.TempDir())
			fatal(t, problem)
			first, problem := records.Open(layout.DB)
			fatal(t, problem)
			defer first.Close()
			second, problem := records.Open(layout.DB)
			fatal(t, problem)
			defer second.Close()
			operation, _, problem := first.BeginRentalOperation(records.RentalOperation{Key: "attachment-race", Hub: "proof", Reason: "manual", HourlyRateUSDMicros: 1}, 100, 0,
				func(name string) ([]byte, string, *exit.Error) { return []byte(`{}`), "proof", nil }, nil)
			fatal(t, problem)
			row := records.Rental{ID: "pr-attachment-race", MachineName: "attachment-race", SKU: "cpu", AcceleratorModel: "CPU", AcceleratorCount: 1, HourlyRateUSDMicros: 1, State: "acquiring", Hub: "proof"}
			fatal(t, first.RecordRental(row))
			fatal(t, first.AdvanceRentalOperation(operation.Key, row.ID, "acquiring"))
			// This is the stale observation held by a foreground caller before release.
			observed, problem := first.RentalOperation(operation.Key)
			fatal(t, problem)
			ready := row
			ready.State = "ready"
			ready.Address = "worker:1"
			ready.CertPath = layout.RentalCert(row.ID)
			staged := false
			stage := func() *exit.Error {
				staged = true
				if err := os.MkdirAll(filepath.Dir(ready.CertPath), 0700); err != nil {
					return exit.Internalf("stage: %s", err)
				}
				if err := os.WriteFile(ready.CertPath, []byte("proof"), 0600); err != nil {
					return exit.Internalf("stage: %s", err)
				}
				return nil
			}
			switch mode {
			case "released", "release_requested":
				if mode == "released" {
					_, problem = rental.Forget(layout, second, row.ID)
				} else {
					_, problem = second.RequestRentalRelease(row.ID)
				}
				fatal(t, problem)
				problem = first.CompleteRentalAttachment(observed.Key, ready, stage)
				if problem == nil || problem.ErrName() != "rental.operation_moved" || staged {
					t.Fatalf("stale attachment crossed release: problem=%v staged=%v", problem, staged)
				}
			case "attachment-first":
				entered, finish := make(chan struct{}), make(chan struct{})
				attached := make(chan *exit.Error, 1)
				go func() {
					attached <- first.CompleteRentalAttachment(observed.Key, ready, func() *exit.Error { close(entered); <-finish; return stage() })
				}()
				<-entered
				started, released := make(chan struct{}), make(chan *exit.Error, 1)
				go func() { close(started); _, problem := rental.Forget(layout, second, row.ID); released <- problem }()
				<-started
				select {
				case problem := <-released:
					close(finish)
					t.Fatalf("release passed the active SQLite handoff: %v", problem)
				case <-time.After(100 * time.Millisecond):
				}
				close(finish)
				fatal(t, <-attached)
				fatal(t, <-released)
			case "stage-failed":
				problem = first.CompleteRentalAttachment(observed.Key, ready, func() *exit.Error { return exit.New(exit.Failed, "staging refused") })
				if problem == nil {
					t.Fatal("staging refusal committed")
				}
				current, problem := first.RentalRow(row.ID)
				fatal(t, problem)
				op, problem := first.RentalOperation(observed.Key)
				fatal(t, problem)
				if current.State != "acquiring" || current.Address != "" || op.State != "acquiring" {
					t.Fatal("failed staging advanced target or operation")
				}
				return
			case "already-attached":
				fatal(t, first.CompleteRentalAttachment(observed.Key, ready, stage))
				staged = false
				ready.Address = "stale:2"
				fatal(t, second.CompleteRentalAttachment(observed.Key, ready, stage))
				current, problem := first.RentalRow(row.ID)
				fatal(t, problem)
				if staged || current.Address != "worker:1" {
					t.Fatal("late attachment replaced the completed target")
				}
				return
			}
			current, problem := first.RentalRow(row.ID)
			fatal(t, problem)
			op, problem := first.RentalOperation(observed.Key)
			fatal(t, problem)
			if mode == "release_requested" {
				if op.State != "release_requested" || current.Address != "" {
					t.Fatal("release intent lost")
				}
			} else if op.State != "released" || current != nil {
				t.Fatalf("release resurrected target: operation=%s row=%+v", op.State, current)
			}
			if _, err := os.Stat(ready.CertPath); !os.IsNotExist(err) {
				t.Fatal("release left or recreated credential files")
			}
		})
	}
}
