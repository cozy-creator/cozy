package records

import (
	"database/sql"
	"encoding/json"
	"errors"

	"github.com/cozy-creator/cozy/internal/exit"
)

// The durable trace of an editable package's life outside any request: one row per refresh
// the daemon's source watcher performed (cl-097), the same shape request_events has, keyed by
// package because no request asked for it. `cozy package list --full` reads the latest row.
var packageEventSchema = []string{`
CREATE TABLE IF NOT EXISTS package_events (
  seq     INTEGER PRIMARY KEY AUTOINCREMENT,
  package TEXT    NOT NULL,
  type    TEXT    NOT NULL,
  payload TEXT    NOT NULL,
  at      TEXT    NOT NULL
)`, `
CREATE INDEX IF NOT EXISTS package_events_by_package ON package_events(package, seq)`}

// PackageEvent is one durable package lifecycle row.
type PackageEvent struct {
	Seq     int64
	Package string
	Type    string
	Payload map[string]any
	At      string
}

// AppendPackageEvent records one package lifecycle event.
func (s *Store) AppendPackageEvent(pkg, eventType string, payload map[string]any) *exit.Error {
	body, e := encodePayload(payload)
	if e != nil {
		return e
	}
	if _, err := s.db.Exec(`INSERT INTO package_events(package,type,payload,at) VALUES(?,?,?,?)`,
		pkg, eventType, body, now()); err != nil {
		return exit.Internalf("cannot append the %s event for %s: %s", eventType, pkg, err)
	}
	return nil
}

// LastPackageEvent answers the newest event recorded for one package, or nil.
func (s *Store) LastPackageEvent(pkg string) (*PackageEvent, *exit.Error) {
	var event PackageEvent
	var body string
	err := s.db.QueryRow(`SELECT seq,package,type,payload,at FROM package_events
		WHERE package=? ORDER BY seq DESC LIMIT 1`, pkg).
		Scan(&event.Seq, &event.Package, &event.Type, &body, &event.At)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, exit.Internalf("cannot read package events for %s: %s", pkg, err)
	}
	if err := json.Unmarshal([]byte(body), &event.Payload); err != nil {
		return nil, exit.Internalf("package event %d for %s has an unreadable payload: %s", event.Seq, pkg, err)
	}
	return &event, nil
}
