package records

import (
	"database/sql"
	"encoding/json"
	"errors"

	"github.com/cozy-creator/cozy/internal/exit"
)

// Rows are armed only by new, verified root submissions. Historical
// success is not evidence that its result schema or custody can be reconstructed.
const successfulWorkDDL = `CREATE TABLE IF NOT EXISTS successful_work_releases (
 request_id TEXT PRIMARY KEY REFERENCES requests(id),
 body_digest TEXT NOT NULL, plan_id TEXT NOT NULL,
 attempt INTEGER NOT NULL DEFAULT 0, invocation_digest TEXT NOT NULL DEFAULT '',
 terminal_id TEXT NOT NULL DEFAULT '', terminal_digest TEXT NOT NULL DEFAULT '',
 state TEXT NOT NULL DEFAULT 'armed' CHECK(state IN ('armed','draining','release_work','complete','deferred')),
 members TEXT NOT NULL DEFAULT '[]', reason TEXT NOT NULL DEFAULT ''
)`

type SuccessfulWorkRelease struct {
	RequestID, BodyDigest, PlanID                 string
	Attempt                                       int64
	Invocation, TerminalID, TerminalDigest, State string
	Members                                       []string
}

func armSuccessfulWorkTx(tx *sql.Tx, r Request) *exit.Error {
	if !r.ReleaseImplicitWork || !r.RetainWork || r.ParentRequestID != "" || !r.IsJob() ||
		(r.WeightsOutputs != "" && r.WeightsOutputs != "[]") || r.ModelTransfer != nil {
		return nil
	}
	if _, err := tx.Exec(`INSERT INTO successful_work_releases(request_id,body_digest,plan_id) VALUES(?,?,?)`, r.ID, r.BodyDigest, r.PlanID); err != nil {
		return exit.Internalf("cannot arm successful work release: %s", err)
	}
	return nil
}

func (s *Store) SuccessfulWorkRelease(id string) (*SuccessfulWorkRelease, *exit.Error) {
	var r SuccessfulWorkRelease
	var members string
	err := s.db.QueryRow(`SELECT request_id,body_digest,plan_id,attempt,invocation_digest,terminal_id,terminal_digest,state,members FROM successful_work_releases WHERE request_id=?`, id).
		Scan(&r.RequestID, &r.BodyDigest, &r.PlanID, &r.Attempt, &r.Invocation, &r.TerminalID, &r.TerminalDigest, &r.State, &members)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, exit.Internalf("cannot read successful work release: %s", err)
	}
	if json.Unmarshal([]byte(members), &r.Members) != nil {
		return nil, exit.Internalf("successful work release has invalid membership")
	}
	return &r, nil
}

// Bind one successful attempt before asynchronous native cleanup. It never arms
// an older request or rewrites an execution outcome.
func (s *Store) BeginSuccessfulWorkRelease(r Request, a Attempt) (bool, *exit.Error) {
	if r.ParentRequestID != "" || r.State != "succeeded" || a.State != "closed" || a.TerminalStatus != "SUCCEEDED" || r.ID != a.RequestID {
		return false, nil
	}
	_, err := s.db.Exec(`UPDATE successful_work_releases SET state='draining',attempt=?,invocation_digest=?,terminal_id=?,terminal_digest=?,
		members=(WITH RECURSIVE family(id) AS (SELECT successful_work_releases.request_id UNION ALL SELECT r.id FROM requests r JOIN family f ON r.parent_request_id=f.id) SELECT json_group_array(r.id) FROM requests r WHERE r.id IN (SELECT id FROM family) AND r.retain_work=1 AND (r.id<>successful_work_releases.request_id OR (r.child_artifacts=0 AND r.weights_outputs IN ('','[]'))))
		WHERE request_id=? AND state='armed' AND body_digest=? AND plan_id=?
		AND EXISTS(SELECT 1 FROM requests WHERE id=? AND state='succeeded' AND ordinal=?)`,
		a.Attempt, a.InvocationDigest, a.TerminalID, a.TerminalDigest, r.ID, r.BodyDigest, r.PlanID, r.ID, a.Attempt)
	if err != nil {
		return false, exit.Internalf("cannot begin successful work release: %s", err)
	}
	row, problem := s.SuccessfulWorkRelease(r.ID)
	if problem != nil || row == nil {
		return false, problem
	}
	return (row.State == "draining" || row.State == "release_work") && row.Attempt == a.Attempt && row.Invocation == a.InvocationDigest && row.TerminalID == a.TerminalID && row.TerminalDigest == a.TerminalDigest, nil
}

// SuccessfulWorkFamily returns children before their parent. Closed caller
// admission cannot add another child; every state is rechecked at the phase cut.
func (s *Store) SuccessfulWorkFamily(root string) ([]Request, *exit.Error) {
	rows, err := s.db.Query(`WITH RECURSIVE family(id,depth) AS (SELECT ?,0 UNION ALL SELECT r.id,f.depth+1 FROM requests r JOIN family f ON r.parent_request_id=f.id)
		SELECT `+requestCols+` FROM requests WHERE id IN (SELECT id FROM family) ORDER BY (SELECT depth FROM family WHERE family.id=requests.id) DESC,created_at`, root)
	if err != nil {
		return nil, exit.Internalf("cannot inspect successful work family: %s", err)
	}
	defer rows.Close()
	var out []Request
	for rows.Next() {
		r, err := scanRequest(rows)
		if err != nil {
			return nil, exit.Internalf("cannot read successful work member: %s", err)
		}
		out = append(out, r)
	}
	if err := rows.Err(); err != nil {
		return nil, exit.Internalf("cannot finish successful work census: %s", err)
	}
	return out, nil
}

func (s *Store) DeferSuccessfulWorkRelease(id, reason string) *exit.Error {
	_, err := s.db.Exec(`UPDATE successful_work_releases SET state='deferred',reason=? WHERE request_id=? AND state IN ('armed','draining')`, reason, id)
	if err != nil {
		return exit.Internalf("cannot defer successful work release: %s", err)
	}
	return nil
}

// Only confirmed native holder releases precede this cut. RetainWork becomes
// false atomically with the durable intent that requires false ACK replay.
func (s *Store) ReadySuccessfulWorkRelease(root string, members []Request) (bool, *exit.Error) {
	tx, err := s.db.Begin()
	if err != nil {
		return false, exit.Internalf("cannot begin successful release cut: %s", err)
	}
	defer tx.Rollback()
	var phase, priorMembers string
	if err := tx.QueryRow(`SELECT state,members FROM successful_work_releases WHERE request_id=?`, root).Scan(&phase, &priorMembers); err != nil {
		return false, exit.Internalf("cannot read successful release cut: %s", err)
	}
	if phase == "release_work" || phase == "complete" {
		return true, nil
	}
	if phase != "draining" {
		return false, nil
	}
	var sealed []string
	if json.Unmarshal([]byte(priorMembers), &sealed) != nil {
		return false, exit.Internalf("successful release membership is malformed")
	}
	expected := map[string]bool{}
	for _, id := range sealed {
		expected[id] = true
	}
	ids := make([]string, 0, len(members))
	for _, member := range members {
		var state string
		var retained bool
		var open int
		if err := tx.QueryRow(`SELECT state,retain_work,(SELECT COUNT(*) FROM attempts WHERE request_id=r.id AND state IN (`+openAttemptStates+`)) FROM requests r WHERE id=?`, member.ID).Scan(&state, &retained, &open); err != nil {
			return false, exit.Internalf("cannot seal successful release member: %s", err)
		}
		if state != "succeeded" || open != 0 {
			return false, nil
		}
		keepResult := member.ID == root && member.RetainsLocalOutputs()
		var owed bool
		if err := tx.QueryRow(`SELECT EXISTS(SELECT 1 FROM request_weights_retentions WHERE request_id=? AND state!='released' AND (?=0 OR kind!='result'))
            OR EXISTS(SELECT 1 FROM native_artifact_retentions WHERE consumer_id=? AND state!='released' AND (?=0 OR kind!='result'))
            OR EXISTS(SELECT 1 FROM weights_finalizations WHERE request_id=? AND completed_at='')
            OR EXISTS(SELECT 1 FROM request_operation_lookups WHERE request_id=? AND state='pending')`, member.ID, keepResult, member.ID, keepResult, member.ID, member.ID).Scan(&owed); err != nil {
			return false, exit.Internalf("cannot verify successful native drain: %s", err)
		}
		if owed {
			return false, nil
		}
		if retained && !keepResult {
			if !expected[member.ID] {
				return false, exit.Internalf("successful release changed its sealed members")
			}
			delete(expected, member.ID)
			ids = append(ids, member.ID)
		}
	}
	if len(expected) != 0 {
		return false, exit.Internalf("successful release omitted sealed members")
	}
	raw, _ := json.Marshal(ids)
	if _, err := tx.Exec(`UPDATE successful_work_releases SET state='release_work',members=? WHERE request_id=? AND state='draining'`, string(raw), root); err != nil {
		return false, exit.Internalf("cannot record successful release intent: %s", err)
	}
	for _, id := range ids {
		if _, err := tx.Exec(`UPDATE requests SET retain_work=0 WHERE id=? AND state='succeeded'`, id); err != nil {
			return false, exit.Internalf("cannot release successful request ownership: %s", err)
		}
	}
	if err := appendEventTx(tx, root, "request.work_release_requested", 0, map[string]any{"status": "SUCCEEDED", "members": len(ids)}); err != nil {
		return false, exit.Internalf("cannot journal successful release: %s", err)
	}
	if err := tx.Commit(); err != nil {
		return false, exit.Internalf("cannot commit successful release: %s", err)
	}
	return true, nil
}

func (s *Store) CompleteSuccessfulWorkRelease(id string) *exit.Error {
	tx, err := s.db.Begin()
	if err != nil {
		return exit.Internalf("cannot begin successful release completion: %s", err)
	}
	defer tx.Rollback()
	r, err := tx.Exec(`UPDATE successful_work_releases SET state='complete' WHERE request_id=? AND state='release_work'`, id)
	if err != nil {
		return exit.Internalf("cannot complete successful release: %s", err)
	}
	n, _ := r.RowsAffected()
	if n == 0 {
		return nil
	}
	if err := appendEventTx(tx, id, "request.work_released", 0, map[string]any{"status": "SUCCEEDED"}); err != nil {
		return exit.Internalf("cannot journal released successful work: %s", err)
	}
	if err := tx.Commit(); err != nil {
		return exit.Internalf("cannot commit released successful work: %s", err)
	}
	return nil
}

func (s *Store) PendingSuccessfulWorkReleases() ([]string, *exit.Error) {
	rows, err := s.db.Query(`SELECT w.request_id FROM successful_work_releases w JOIN requests r ON r.id=w.request_id WHERE r.state='succeeded' AND w.state IN ('armed','draining','release_work') ORDER BY r.created_at`)
	if err != nil {
		return nil, exit.Internalf("cannot restore successful work releases: %s", err)
	}
	defer rows.Close()
	var ids []string
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return nil, exit.Internalf("cannot read successful release queue: %s", err)
		}
		ids = append(ids, id)
	}
	if err := rows.Err(); err != nil {
		return nil, exit.Internalf("cannot finish successful release queue: %s", err)
	}
	return ids, nil
}

// Request idempotency compares the original retention intent even after normal
// completion has released its operational hold. No old row is re-armed here.
func submittedRetainWorkTx(tx *sql.Tx, r Request) (bool, error) {
	if r.RetainWork {
		return true, nil
	}
	var retained bool
	err := tx.QueryRow(`SELECT json_extract(payload,'$.retain_work') FROM request_events WHERE request_id=? AND type='request.submitted' AND json_type(payload,'$.retain_work') IN ('true','false') ORDER BY seq LIMIT 1`, r.ID).Scan(&retained)
	if err == nil {
		return retained, nil
	}
	if err != sql.ErrNoRows {
		return false, err
	}
	var released bool
	err = tx.QueryRow(`SELECT EXISTS(SELECT 1 FROM successful_work_releases w,json_each(w.members) m WHERE w.state IN ('release_work','complete') AND m.value=?)`, r.ID).Scan(&released)
	return released, err
}
