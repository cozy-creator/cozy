package producttest

import (
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/output"
)

func TestLiveTableSeparatesFixedSections(t *testing.T) {
	list := output.List{
		Name: "files", Fields: []string{"name", "bytes"}, AllFields: []string{"name", "bytes"},
		Bytes: []string{"bytes"}, Lead: []string{"retained files"},
		Rows:       []map[string]string{{"name": "first\nline", "bytes": "1024"}, {"name": "second", "bytes": "2048"}},
		Aggregates: []output.Field{{K: "count", V: 2}, {K: "stored", V: "3 KiB"}},
		Trail:      []string{"files remain retained"}, Notes: []string{"read only"}, Next: []string{"cozy storage"},
	}
	mode := output.Mode{Human: true}
	table, err := list.Table(mode)
	must(t, err)
	if !strings.Contains(table.Header, "NAME") || !strings.Contains(table.Header, "BYTES") || len(table.Rows) != 2 {
		t.Fatalf("heading or data row identity lost: %+v", table)
	}
	if !strings.Contains(table.Rows[0], "1.0KiB") || !strings.Contains(table.Rows[1], "2.0KiB") {
		t.Fatalf("live byte columns differ from snapshot formatting: %+v", table.Rows)
	}
	footer := strings.Join(table.Footer, "\n")
	for _, want := range []string{"stored: 3 KiB", "files remain retained", "Note: read only", "Next: cozy storage"} {
		if !strings.Contains(footer, want) {
			t.Fatalf("fixed footer omitted %q: %q", want, footer)
		}
	}
	if strings.Contains(footer, "count:") {
		t.Fatalf("hidden aggregate acquired a physical line: %q", footer)
	}
	// Both empty and --fields=one-column boards must retain a heading, unlike
	// static snapshots, whose existing empty sentence and bullet list remain useful.
	list.Rows = nil
	mode.Fields = []string{"name"}
	table, err = list.Table(mode)
	must(t, err)
	if table.Header != "NAME" || len(table.Rows) != 0 {
		t.Fatalf("empty single-column table lost its heading: %+v", table)
	}
	list.Lead, list.Aggregates, list.Trail, list.Notes, list.Next = nil, nil, nil, nil, nil
	var snapshot strings.Builder
	must(t, list.Emit(&snapshot, mode))
	if snapshot.String() != "No files found.\n" {
		t.Fatalf("static empty list changed: %q", snapshot.String())
	}
}
