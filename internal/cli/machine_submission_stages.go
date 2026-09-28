package cli

import (
	"time"
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
