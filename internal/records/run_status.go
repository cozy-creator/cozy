package records

import (
	"database/sql"
	"strings"
)

type runStatusGroup struct {
	public string
	stored []string
}

var runStatusGroups = []runStatusGroup{
	{"queued", []string{"submitted", "queued", "requeue_pending"}},
	{"in_progress", []string{"dispatching"}},
	{"completed", []string{"succeeded"}},
	{"failed", []string{"failed", "abandoned", "refused"}},
	{"canceled", []string{"canceled"}},
	{"canceling", []string{"releasing"}},
}

func PublicRunStatus(state string) string {
	for _, group := range runStatusGroups {
		for _, stored := range group.stored {
			if state == stored {
				return group.public
			}
		}
	}
	return state
}

func publicRunStatusSQL() string {
	var text strings.Builder
	text.WriteString("CASE")
	for _, group := range runStatusGroups {
		text.WriteString(" WHEN requests.state IN ('" + strings.Join(group.stored, "','") + "') THEN '" + group.public + "'")
	}
	text.WriteString(" ELSE requests.state END")
	return text.String()
}

// RetainedRetryAvailable uses the same read-only admission checks as --retry.
// It creates no descendant, rental hold, operation, or authority receipt.
func (s *Store) RetainedRetryAvailable(row Request) bool {
	if !row.IsJob() || row.State != "failed" {
		return false
	}
	var machine bool
	if err := s.db.QueryRow(`SELECT EXISTS(SELECT 1 FROM machine_executions WHERE request_id=?)`, row.ID).Scan(&machine); err != nil {
		return false
	}
	candidate := Request{Kind: "job", RetainWork: true, RetryOf: row.ID, MachineExecutionObserver: machine}
	return retainRetryFrom(s.db, &candidate, row) == nil
}

type retryReader interface{ QueryRow(string, ...any) *sql.Row }
