package producttest

import (
	"fmt"
	"github.com/cozy-creator/cozy/internal/records"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/cozy-creator/cozy/internal/exit"
)

// Cancellation and first dispatch must choose one order. A canceled unsent run must
// never be dispatched; a possibly sent run must not claim a terminal cancellation.
func TestNativeDispatchAndCancellationChooseOneOrder(t *testing.T) {
	s, problem := records.Open(filepath.Join(t.TempDir(), "records.sqlite"))
	if problem != nil {
		t.Fatal(problem)
	}
	defer s.Close()
	for _, order := range []string{"cancel first", "dispatch first", "concurrent"} {
		t.Run(order, func(t *testing.T) {
			for i := range 20 {
				id := fmt.Sprintf("%s-%d", order, i)
				_, _, problem := s.Submit(records.Request{ID: id, IdemKey: id, Package: "local/test", Entrypoint: "main",
					Payload: []byte(`{}`), BodyDigest: "sha256:" + strings.Repeat("1", 64), MachineExecutionObserver: true})
				if problem != nil {
					t.Fatal(problem)
				}
				if problem := s.LinkMachineExecution(id, "machine"); problem != nil {
					t.Fatal(problem)
				}
				var allowed bool
				var dispatchError, cancelError *exit.Error
				dispatch := func() { allowed, dispatchError = s.MarkRunV1Sent(id) }
				cancel := func() { _, cancelError = s.RequestMachineCancellation(id, "test user") }
				switch order {
				case "cancel first":
					cancel()
					dispatch()
				case "dispatch first":
					dispatch()
					cancel()
				default:
					var group sync.WaitGroup
					group.Add(2)
					go func() { defer group.Done(); dispatch() }()
					go func() { defer group.Done(); cancel() }()
					group.Wait()
				}
				if dispatchError != nil || cancelError != nil {
					t.Fatalf("dispatch: %v; cancel: %v", dispatchError, cancelError)
				}
				if order == "cancel first" && allowed || order == "dispatch first" && !allowed {
					t.Fatalf("%s allowed dispatch: %v", order, allowed)
				}
				row, problem := s.RequestRow(id)
				if problem != nil {
					t.Fatal(problem)
				}
				want := "canceled"
				if allowed {
					want = "canceling"
				}
				if row.State != want {
					t.Fatalf("dispatch allowed=%v left state %q, want %q", allowed, row.State, want)
				}
				if sent, problem := s.RunV1Marked(id, records.RunV1Sent); problem != nil || sent != allowed {
					t.Fatalf("dispatch allowed=%v, sent=%v: %v", allowed, sent, problem)
				}
				if retry, problem := s.MarkRunV1Sent(id); problem != nil || retry {
					t.Fatalf("canceled run allowed a later dispatch=%v: %v", retry, problem)
				}
			}
		})
	}
}
