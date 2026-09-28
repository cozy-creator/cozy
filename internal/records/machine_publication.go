package records

import (
	"bytes"
	"database/sql"
	"encoding/json"

	"github.com/cozy-creator/cozy/internal/exit"
)

func (s *Store) RequestPublicationRepositories(id string) ([]string, *exit.Error) {
	var raw []byte
	err := s.db.QueryRow(`SELECT COALESCE(json_extract(payload,'$.allow_publish'),'[]') FROM request_events WHERE request_id=? AND type='run.created' ORDER BY seq LIMIT 1`, id).Scan(&raw)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	var names []string
	if err != nil || json.Unmarshal(raw, &names) != nil {
		return nil, exit.Internalf("cannot read recorded publication consent")
	}
	return names, nil
}

// A frozen authorization names one machine (this host's name for it: a rental id or
// "local") and its certificate; a run released from a lost machine is authorized again for
// the machine it is placed on next.
const linkedPublication = `json_extract(CAST(payload AS TEXT),'$.machine')=(SELECT machine_id FROM machine_executions WHERE request_id=?)`

func (s *Store) MachinePublicationIntent(id string) ([]byte, *exit.Error) {
	var raw []byte
	err := s.db.QueryRow(`SELECT payload FROM request_events WHERE request_id=? AND type='machine.publication_authorization' AND `+linkedPublication+` ORDER BY seq LIMIT 1`, id, id).Scan(&raw)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, exit.Internalf("cannot read frozen machine publication authorization: %s", err)
	}
	return raw, nil
}

// RecordMachinePublicationIntent precedes the first AuthKit request. An ambiguous
// reply reuses this exact public certificate, scope, expiry, and authorization ID.
func (s *Store) RecordMachinePublicationIntent(id string, raw []byte) *exit.Error {
	if len(raw) == 0 || len(raw) > 16<<10 || !json.Valid(raw) {
		return exit.New(exit.Validation, "machine publication intent is not a bounded JSON document")
	}
	tx, err := s.db.Begin()
	if err != nil {
		return exit.Internalf("cannot begin machine publication authorization: %s", err)
	}
	defer tx.Rollback()
	var prior []byte
	err = tx.QueryRow(`SELECT payload FROM request_events WHERE request_id=? AND type='machine.publication_authorization' AND `+linkedPublication+` ORDER BY seq LIMIT 1`, id, id).Scan(&prior)
	if err == nil {
		if !bytes.Equal(prior, raw) {
			return exit.New(exit.Conflict, "machine publication authorization was already frozen")
		}
		return nil
	}
	if err != sql.ErrNoRows {
		return exit.Internalf("cannot inspect existing machine publication authorization: %s", err)
	}
	var allowed bool
	err = tx.QueryRow(`SELECT EXISTS(SELECT 1 FROM machine_executions e JOIN requests r ON r.id=e.request_id WHERE r.id=? AND e.machine_id=json_extract(?,'$.machine') AND length(e.submission)=0 AND r.state!='canceled')`, id, string(raw)).Scan(&allowed)
	if err != nil || !allowed {
		return exit.New(exit.Conflict, "publication authority has no pending machine submission")
	}
	if _, err = tx.Exec(`INSERT INTO request_events(request_id,type,attempt,payload,at) VALUES(?,'machine.publication_authorization',0,?,?)`, id, raw, now()); err != nil {
		return exit.Internalf("cannot freeze machine publication authorization: %s", err)
	}
	if err = tx.Commit(); err != nil {
		return exit.Internalf("cannot commit machine publication authorization: %s", err)
	}
	return nil
}
