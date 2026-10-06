package producttest

import (
	"bytes"
	"database/sql"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/records"
)

func TestArchivedWorkerSubmissionIsAbandonedNeverResent(t *testing.T) {
	for _, accepted := range []bool{false, true} {
		t.Run(map[bool]string{false: "acceptance unknown", true: "accepted"}[accepted], func(t *testing.T) {
			root := t.TempDir()
			must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: http://127.0.0.1:1\ntensorhub_token: unreachable\n"), 0o600))
			path := filepath.Join(root, "creator.sqlite")
			store, problem := records.Open(path)
			fatal(t, problem)
			request, _, problem := store.Submit(records.Request{ID: "archive-run", IdemKey: "saved-worker",
				Package: "local/archived", Entrypoint: "main", Kind: "job", Payload: []byte(`{}`),
				BodyDigest: childDigest("1"), RetainWork: true, MachineExecutionObserver: true})
			fatal(t, problem)
			fatal(t, store.LinkMachineExecution(request.ID, "local"))
			submission, err := os.ReadFile("testdata/record-archive/submission.bin")
			must(t, err)
			receipt := []byte{}
			if accepted {
				receipt, err = os.ReadFile("testdata/record-archive/receipt.bin")
				must(t, err)
			}
			db, err := sql.Open("sqlite", path)
			must(t, err)
			_, err = db.Exec(`UPDATE machine_executions SET submission=?,receipt=? WHERE request_id=?`, submission, receipt, request.ID)
			must(t, err)
			must(t, db.Close())
			store.Close()
			_ = startDaemonProcess(t, root)
			reader, problem := records.OpenReadOnly(path)
			fatal(t, problem)
			defer reader.Close()
			waitUntil(t, "archived worker run abandonment", func() bool {
				row, problem := reader.RequestRow(request.ID)
				fatal(t, problem)
				return row != nil && row.State == "abandoned"
			})
			link, problem := reader.MachineExecution(request.ID)
			fatal(t, problem)
			if !link.Abandoned || !bytes.Equal(link.Submission, submission) || !bytes.Equal(link.Receipt, receipt) || len(link.Outcome) != 0 {
				t.Fatal("retirement altered uncertain execution evidence")
			}
			if sent, problem := reader.RunV1Marked(request.ID, records.RunV1Sent); problem != nil || sent {
				t.Fatalf("archived submission was dispatched as native: %t %v", sent, problem)
			}
			events, problem := reader.EventsAfter(request.ID, 0, 100)
			fatal(t, problem)
			found := false
			for _, event := range events {
				found = found || event.Type == "client.machine_abandoned" && strings.Contains(string(event.Raw), "whether it ran is unknown")
			}
			if !found {
				t.Fatal("retirement omitted its durable uncertainty reason")
			}
		})
	}
}
