package records

import (
	"database/sql"
	"errors"

	"github.com/cozy-creator/cozy/internal/exit"
)

// The catalog revision this client knows: it changes whenever this client binds, unbinds,
// publishes or yanks a package or model. A run carries it, and a machine re-resolves a held
// model resolution only when it differs (cozy.machine.v1 RunSpec.binding_revision).
const bindingRevisionDDL = `CREATE TABLE IF NOT EXISTS binding_revision (
 id INTEGER PRIMARY KEY CHECK(id = 1),
 revision TEXT NOT NULL
)`

// BindingRevision is the current catalog revision, "" before this client changed anything.
func (s *Store) BindingRevision() (string, *exit.Error) {
	var revision string
	err := s.db.QueryRow(`SELECT revision FROM binding_revision WHERE id=1`).Scan(&revision)
	if errors.Is(err, sql.ErrNoRows) {
		return "", nil
	}
	if err != nil {
		return "", exit.Internalf("cannot read the binding revision: %s", err)
	}
	return revision, nil
}

// ChangeBindingRevision records that this client changed a package's or model's catalog rows.
func (s *Store) ChangeBindingRevision() (string, *exit.Error) {
	revision := NewID("binding")
	if _, err := s.db.Exec(`INSERT INTO binding_revision(id,revision) VALUES(1,?) ON CONFLICT(id) DO UPDATE SET revision=excluded.revision`, revision); err != nil {
		return "", exit.Internalf("cannot record the binding revision: %s", err)
	}
	return revision, nil
}
