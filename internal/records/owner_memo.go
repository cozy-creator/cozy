package records

import (
	"encoding/json"

	"github.com/cozy-creator/cozy/internal/exit"
)

// ownerMemoIndex finds the owner's memo.record events by key. The records stay on this
// machine: they are the owner's own, and nothing about them is sent to a Hub.
const ownerMemoIndex = `CREATE INDEX IF NOT EXISTS request_events_memo
  ON request_events(json_extract(payload,'$.computation_digest')) WHERE type='machine.memo.record'`

// OwnerMemoResults are the recorded results of one memoized operation under one key on one
// Hub, newest first, as the exact bytes the machine reported.
func (s *Store) OwnerMemoResults(hub, operation, key string) ([]json.RawMessage, *exit.Error) {
	rows, err := s.db.Query(`SELECT e.payload FROM request_events e JOIN requests r ON r.id=e.request_id
		WHERE e.type='machine.memo.record' AND json_extract(e.payload,'$.computation_digest')=?
		AND json_extract(e.payload,'$.operation')=? AND r.hub=? ORDER BY e.seq DESC`, key, operation, hub)
	if err != nil {
		return nil, exit.Internalf("cannot read the owner's memo: %s", err)
	}
	defer rows.Close()
	var out []json.RawMessage
	for rows.Next() {
		var payload string
		var record struct {
			Result json.RawMessage `json:"result"`
		}
		if err := rows.Scan(&payload); err != nil {
			return nil, exit.Internalf("cannot read an owner memo record: %s", err)
		}
		if json.Unmarshal([]byte(payload), &record) == nil && len(record.Result) > 0 {
			out = append(out, record.Result)
		}
	}
	return out, nil
}
