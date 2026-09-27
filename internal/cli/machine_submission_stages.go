package cli

import (
	"fmt"
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

func preparedDetail(pkg, release string, prepared *publishedPreparation) string {
	detail := pkg + "@" + release
	if prepared.Retained {
		detail += ", reused the machine's preparation"
	}
	return detail
}

func modelDefaultsDetail(rungs, probes int) string {
	return fmt.Sprintf("%d rung(s), %d checkpoint probe(s)", rungs, probes)
}
