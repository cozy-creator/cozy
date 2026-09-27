package producttest

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
)

func TestRentalReassignmentPreservesExecutionAndPurchaseCustody(t *testing.T) {
	for _, mode := range []string{"unattempted", "explicit", "retained", "uploaded", "purchased", "attempted"} {
		t.Run(mode, func(t *testing.T) {
			st, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
			fatal(t, problem)
			defer st.Close()
			row := records.Rental{ID: "pr-busy", MachineName: "busy", SKU: "cpu", AcceleratorModel: "CPU", AcceleratorCount: 1, HourlyRateUSDMicros: 1, State: "ready", Hub: "proof"}
			fatal(t, st.RecordRental(row))
			request := records.Request{ID: "job-replan", IdemKey: "replan", BodyDigest: "sha256:" + strings.Repeat("a", 64), Package: "proof/video", Entrypoint: "generate", PlanID: "plan", Payload: []byte(`{}`), Outputs: "[]", Kind: "job", Worker: row.ID, Rental: true, RentalRequired: true}
			switch mode {
			case "explicit":
				request.RequestedRental = row.ID
			case "retained":
				request.RetainWork = true
			case "uploaded":
				request.LocalPackageUploadedBootID = "boot-busy"
			}
			request, _, problem = st.Submit(request)
			fatal(t, problem)
			if mode == "purchased" {
				op, _, problem := st.BeginRentalOperation(records.RentalOperation{Key: "purchase", Hub: "proof", Reason: "job", ManagedRequestID: request.ID, HourlyRateUSDMicros: 1}, func(string) ([]byte, string, *exit.Error) { return []byte(`{}`), "proof", nil })
				fatal(t, problem)
				fatal(t, st.AdvanceRentalOperation(op.Key, row.ID, "ready"))
			}
			if mode == "attempted" {
				fatal(t, st.SpawnWorker(records.WorkerProcess{InstanceID: "instance", Package: request.Package, WorkerID: "worker", Devices: []string{"cpu"}}))
				_, problem := st.Dispatch(records.Attempt{RequestID: request.ID, InstanceID: "instance", SessionID: "session", InvocationCanonical: []byte(`{}`), WeightsOutputs: `[]`})
				fatal(t, problem)
			}
			changed, problem := st.ReleaseUnattemptedRentalAssignment(request.ID, row.ID)
			fatal(t, problem)
			if changed != (mode == "unattempted") {
				t.Fatalf("assignment released=%t for %s", changed, mode)
			}
			got, problem := st.RequestRow(request.ID)
			fatal(t, problem)
			if !changed && got.Worker != row.ID {
				t.Fatal("retained assignment changed")
			}
		})
	}
}
