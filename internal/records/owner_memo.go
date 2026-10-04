package records

import (
	"encoding/json"

	"github.com/cozy-creator/cozy/internal/exit"
	v1 "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
)

// MemoRecorded is a memoized call's result its machine reported (a v1 `memo` event, or a
// worker.v1 `memo.record`): this owner's own, never sent to a Hub.
const MemoRecorded = "machine.memo.record"

// A job spec carries at most this many known results, of at most these bytes each and in all.
const (
	maxKnownResults = 256
	maxKnownResult  = 48 << 10
	maxKnownBytes   = 1 << 20
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

// KnownResults are the newest result of each memoized computation the package's runs reported,
// for its next job's spec: a machine answers a matching call from one without running it.
func (s *Store) KnownResults(pkg string) ([]*v1.MemoResult, *exit.Error) {
	rows, err := s.db.Query(`SELECT e.payload FROM request_events e JOIN requests r ON r.id=e.request_id
		WHERE e.type='machine.memo.record' AND r.package=? ORDER BY e.seq DESC`, pkg)
	if err != nil {
		return nil, exit.Internalf("cannot read the package's known results: %s", err)
	}
	defer rows.Close()
	var known []*v1.MemoResult
	seen, total := map[string]bool{}, 0
	for rows.Next() && len(known) < maxKnownResults {
		var payload string
		var memo struct {
			Operation string          `json:"operation"`
			Digest    string          `json:"computation_digest"`
			Result    json.RawMessage `json:"result"`
		}
		if err := rows.Scan(&payload); err != nil {
			return nil, exit.Internalf("cannot read a known result: %s", err)
		}
		if json.Unmarshal([]byte(payload), &memo) != nil || memo.Digest == "" || seen[memo.Digest] ||
			len(memo.Result) == 0 || total+len(memo.Result) > maxKnownBytes {
			continue
		}
		seen[memo.Digest], total = true, total+len(memo.Result)
		known = append(known, &v1.MemoResult{Operation: memo.Operation, ComputationDigest: memo.Digest, Result: memo.Result})
	}
	return known, nil
}
