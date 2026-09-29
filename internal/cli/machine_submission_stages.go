package cli

import (
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/localpackage"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// submissionStage keeps one ended step of building a machine submission on the run as
// `request.preparing`, so `cozy run show` says where the time before acceptance went.
// It is evidence, not custody: a failed append changes nothing about the submission.
func (m *machineRuns) submissionStage(request, stage, detail string, began time.Time) {
	payload := map[string]any{"stage": stage, "started_unix_ms": began.UnixMilli(), "ms": time.Since(began).Milliseconds()}
	if detail != "" {
		payload["detail"] = detail
	}
	_ = m.store.AppendEvent(request, "request.preparing", 0, payload)
}

func (m *machineRuns) captureForSubmission(request records.Request) (localpackage.ExecutionCapture, *exit.Error) {
	began := time.Now()
	defer m.submissionStage(request.ID, "installation capture", "immutable installation graph", began)
	return m.resolver.CaptureMachineExecution(request)
}

func (m *machineRuns) recordSubmission(request string, submission *pb.MachineExecutionSubmit) *exit.Error {
	began := time.Now()
	defer m.submissionStage(request, "submission persistence", "", began)
	return m.store.RecordMachineSubmission(request, submission)
}
