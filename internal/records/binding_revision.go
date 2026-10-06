package records

import (
	"fmt"

	"github.com/cozy-creator/cozy/internal/exit"
)

// The catalog revision this client knows: it changes whenever this client binds, unbinds,
// publishes or yanks a package or model. A run carries it, and a machine re-resolves a held
// model resolution only when it differs (cozy.machine.v1 RunSpec.binding_revision).
const bindingRevisionDDL = `CREATE TABLE IF NOT EXISTS binding_revision (
 id INTEGER PRIMARY KEY CHECK(id = 1),
 revision TEXT NOT NULL
)`

// The account's own bindings revision as each Hub last stated it on a rental listing: a
// rebind from another computer or the Hub moves it.
const hubBindingsRevisionDDL = `CREATE TABLE IF NOT EXISTS hub_bindings_revisions (
 hub TEXT PRIMARY KEY,
 revision INTEGER NOT NULL
)`

// BindingRevision is the catalog revision runs on hub carry: this client's own, joined with
// the one hub last stated; "" before either changed anything.
func (s *Store) BindingRevision(hub string) (string, *exit.Error) {
	var revision string
	var stated int64
	err := s.db.QueryRow(`SELECT COALESCE((SELECT revision FROM binding_revision WHERE id=1),''),
 COALESCE((SELECT revision FROM hub_bindings_revisions WHERE hub=?),0)`, hub).Scan(&revision, &stated)
	if err != nil {
		return "", exit.Internalf("cannot read the binding revision: %s", err)
	}
	if stated > 0 {
		revision += fmt.Sprintf("+hub.%d", stated)
	}
	return revision, nil
}

// StateHubBindingsRevision records the bindings revision hub stated, written only when it moved.
func (s *Store) StateHubBindingsRevision(hub string, revision int64) *exit.Error {
	var stated int64
	if err := s.db.QueryRow(`SELECT COALESCE((SELECT revision FROM hub_bindings_revisions WHERE hub=?),0)`, hub).Scan(&stated); err != nil {
		return exit.Internalf("cannot read the Hub's bindings revision: %s", err)
	}
	if revision <= 0 || revision == stated {
		return nil
	}
	if _, err := s.db.Exec(`INSERT INTO hub_bindings_revisions(hub,revision) VALUES(?,?)
 ON CONFLICT(hub) DO UPDATE SET revision=excluded.revision WHERE hub_bindings_revisions.revision<>excluded.revision`, hub, revision); err != nil {
		return exit.Internalf("cannot record the Hub's bindings revision: %s", err)
	}
	return nil
}

// ChangeBindingRevision records that this client changed a package's or model's catalog rows.
func (s *Store) ChangeBindingRevision() (string, *exit.Error) {
	revision := NewID("binding")
	if _, err := s.db.Exec(`INSERT INTO binding_revision(id,revision) VALUES(1,?) ON CONFLICT(id) DO UPDATE SET revision=excluded.revision`, revision); err != nil {
		return "", exit.Internalf("cannot record the binding revision: %s", err)
	}
	return revision, nil
}
