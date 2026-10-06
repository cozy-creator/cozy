package cli

import (
	"testing"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/records"
)

func TestExportedNativeOutputUsesCommittedFactsWithoutPublication(t *testing.T) {
	life := api.Lifecycle{
		OutputExport: &api.OutputExportRef{State: "published", Paths: []string{"/result/report.json", "/result/unmatched"}},
		Output: []records.OutputItem{
			{ID: "17/report", Name: "report", Path: "/result/report.json", Status: "completed", Length: 244, MediaType: "application/json", Sha256: "sha256:held"},
			{ID: "17/pending", Name: "pending", Path: "/result/unmatched", Status: "in_progress", Length: 100},
		},
	}
	got := exportedOutputs(life)
	if len(got) != 2 || got[0] != (savedFile{Output: "report", Path: "/result/report.json", Bytes: 244, Mime: "application/json", Digest: "sha256:held"}) || got[1] != (savedFile{Path: "/result/unmatched"}) {
		t.Fatalf("export metadata must come from completed native products: %#v", got)
	}
}
