package records

import (
	"database/sql"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
)

type runStatusGroup struct {
	public string
	stored []string
}

var runStatusGroups = []runStatusGroup{
	{"queued", []string{"submitted", "queued", "requeue_pending"}},
	{"in_progress", []string{"dispatching"}},
	{"completed", []string{"succeeded"}},
	{"failed", []string{"failed", "abandoned", "refused", "blocked"}},
	{"canceled", []string{"canceled"}},
	{"canceling", []string{"releasing"}},
}

// This predicate is shared by point reads and bounded public history filters.
// A sent offer with no receipt is still reconciling possible acceptance, not a
// stopped failure that authorizes a new manual submission.
const pendingMachineAcceptanceSQL = `EXISTS(SELECT 1 FROM machine_executions e
 WHERE e.request_id=requests.id AND (length(e.submission)>0 OR
 EXISTS(SELECT 1 FROM request_events sent WHERE sent.request_id=e.request_id AND sent.type='` + RunV1Sent + `')) AND length(e.receipt)=0
 AND e.cancel_requested=0 AND NOT ` + machineExecutionLost + `)`

func PublicRunStatus(state string, acceptancePending bool) string {
	if state == "blocked" && acceptancePending {
		return "queued"
	}
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
	text.WriteString("CASE WHEN requests.state='blocked' AND " + pendingMachineAcceptanceSQL + " THEN 'queued'")
	for _, group := range runStatusGroups {
		text.WriteString(" WHEN requests.state IN ('" + strings.Join(group.stored, "','") + "') THEN '" + group.public + "'")
	}
	text.WriteString(" ELSE requests.state END")
	return text.String()
}

func (s *Store) PublicRunState(row Request) (string, *exit.Error) {
	if row.State != "blocked" {
		return PublicRunStatus(row.State, false), nil
	}
	var pending bool
	if err := s.db.QueryRow(`SELECT `+pendingMachineAcceptanceSQL+` FROM requests WHERE id=?`, row.ID).Scan(&pending); err != nil {
		return "", exit.Internalf("cannot read pending run acceptance: %s", err)
	}
	return PublicRunStatus(row.State, pending), nil
}

// RetainedRetryAvailable uses the same read-only admission checks as --retry.
// It creates no descendant, rental hold, operation, or authority receipt.
func (s *Store) RetainedRetryAvailable(row Request) bool {
	if !row.IsJob() || (row.State != "blocked" && row.State != "failed") {
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

// StoppedEventID identifies the current retained manual stop without changing
// the stored event type. Watchers must not stop at an older superseded stop.
func (s *Store) StoppedEventID(row Request) int64 {
	if row.State != "blocked" {
		return 0
	}
	var sequence int64
	_ = s.db.QueryRow(`SELECT COALESCE(MAX(seq),0) FROM request_events WHERE request_id=? AND type='request.blocked'`, row.ID).Scan(&sequence)
	return sequence
}

func (s *Store) StoppedEventAt(row Request) string {
	if row.State != "blocked" {
		return ""
	}
	var at string
	_ = s.db.QueryRow(`SELECT at FROM request_events WHERE request_id=? AND type='request.blocked' ORDER BY seq DESC LIMIT 1`, row.ID).Scan(&at)
	return at
}
