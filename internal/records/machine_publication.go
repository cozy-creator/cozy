package records

import (
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
