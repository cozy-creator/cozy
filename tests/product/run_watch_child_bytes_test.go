package producttest

import (
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
	v1 "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
)

// A job's child call downloading its model shows the bytes the machine attaches to the job's
// progress, as a run's own download does; an older machine's stage, without them, shows as
// before.
func TestRunWatchShowsAChildCallsDownloadBytes(t *testing.T) {
	o := hostOwner(t, "child-download-bytes")
	id := "job-child-bytes"
	_, _, problem := o.store.Submit(records.Request{ID: id, IdemKey: id, Package: "paul/minimax-h3",
		Entrypoint: "long_form", Kind: "job", Payload: []byte(`{}`), BodyDigest: childDigest("d"),
		MachineExecutionObserver: true})
	fatal(t, problem)
	fatal(t, o.store.LinkMachineExecution(id, "pr-unreachable"))
	fatal(t, o.store.AcceptRunV1(id, "pr-unreachable", &v1.RunState{Id: id, Number: 1, State: "running", Attempt: 1}))
	const gib = 1 << 30
	for i, progress := range []*v1.Progress{
		{Stage: "Creating reference Subject 1 / downloading paul/reference-image@0.1.0 original",
			Fraction: -1, BytesDone: 123 * gib / 10, BytesTotal: 308 * gib / 10},
		{Stage: "Creating reference Subject 2 / downloading paul/reference-image@0.1.0 original", Fraction: -1},
	} {
		fatal(t, o.store.ObserveRunV1(id, &v1.RunEvent{Sequence: uint64(i + 1), Event: &v1.RunEvent_Progress{Progress: progress}}, nil))
	}
	fatal(t, o.store.ObserveRunV1(id, &v1.RunEvent{Sequence: 3, Event: &v1.RunEvent_State{State: &v1.RunState{
		Id: id, Number: 1, State: "running", Attempt: 1}}}, nil))
	fatal(t, o.store.RecordRunOutcomeV1(id, records.RunEndV1{Outcome: &v1.Outcome{Status: "succeeded"}}))
	defer publicationControlAPI(t, o)()

	code, watched := runCozy(t, o.root, "run", "watch", id)
	t.Logf("cozy run watch:\n%s", watched)
	if code != 0 || !strings.Contains(watched,
		"  Creating reference Subject 1 · downloading paul/reference-image@0.1.0 original · 12.3GiB / 30.8GiB · 40% stage") {
		t.Fatalf("run watch [%d] lacks the child's download bytes:\n%s", code, watched)
	}
	older := strings.Index(watched, "Creating reference Subject 2 · downloading paul/reference-image@0.1.0 original")
	if older < 0 || strings.Contains(watched[older:], "GiB") {
		t.Fatalf("run watch shows an older machine's stage otherwise:\n%s", watched)
	}
}
