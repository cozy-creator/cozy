package cli

import (
	"strings"
	"testing"
	"time"
)

func TestPackagePublishTimingsRecordsMachineAndHumanValues(t *testing.T) {
	timings := newPackagePublishTimings()
	started := time.Now().Add(-25 * time.Millisecond)
	timings.measure("object_upload", started)
	timings.finish(started)
	if timings.Total <= 0 {
		t.Fatalf("total duration was not recorded: %#v", timings)
	}
	if timings.Phases["object_upload"] <= 0 {
		t.Fatalf("object-upload duration was not recorded: %#v", timings.Phases)
	}
	human := timings.Human()
	if human == "" || !strings.Contains(human, "object upload") || !strings.Contains(human, "total") {
		t.Fatalf("human timings omitted recorded phases: %q", human)
	}
}
