package records

import (
	"database/sql"
	"encoding/json"
	"errors"
	"sort"

	"github.com/cozy-creator/cozy/internal/exit"
)

const outputExportSchema = `
CREATE TABLE IF NOT EXISTS request_output_exports (
  request_id       TEXT PRIMARY KEY REFERENCES requests(id),
  directory        TEXT    NOT NULL,
  outputs          TEXT    NOT NULL,
  state            TEXT    NOT NULL,
  attempts         INTEGER NOT NULL DEFAULT 0,
  error_code       TEXT    NOT NULL DEFAULT '',
  safe_error       TEXT    NOT NULL DEFAULT '',
  published_paths  TEXT    NOT NULL DEFAULT '[]',
  updated_at       TEXT    NOT NULL,
  CHECK (state IN ('pending','exporting','published','failed','skipped'))
)`

// OutputExportEntry is one pre-execution output contract derived from the package
// PackageInterface: which result field, and the one media type its bytes must carry. The
// filename is not here — it is the verified file's own content digest plus the media
// type's extension, known only once the terminal is accepted, and never a terminal's
// to supply.
type OutputExportEntry struct {
	OutputID  string `json:"output_id"`
	MediaType string `json:"media_type"`
}

// OutputExportIntent is the immutable submit-time portion of an export row. Directory is
// the caller's --out, or the package's own store under outputs/ when none was given.
type OutputExportIntent struct {
	Directory string              `json:"directory"`
	Outputs   []OutputExportEntry `json:"outputs"`
}

// OutputExport is the durable daemon-owned publication obligation and its settlement.
type OutputExport struct {
	RequestID string
	OutputExportIntent
	State          string
	Attempts       int64
	ErrorCode      string
	SafeError      string
	PublishedPaths []string
	UpdatedAt      string
}

func prepareOutputExport(intent *OutputExportIntent) (string, *exit.Error) {
	if intent == nil {
		return "", nil
	}
	copy := *intent
	copy.Outputs = append([]OutputExportEntry(nil), intent.Outputs...)
	sort.Slice(copy.Outputs, func(i, j int) bool { return copy.Outputs[i].OutputID < copy.Outputs[j].OutputID })
	data, err := json.Marshal(copy.Outputs)
	if err != nil {
		return "", exit.Internalf("cannot encode output export intent: %s", err)
	}
	intent.Outputs = copy.Outputs
	return string(data), nil
}

func recordOutputExportTx(tx *sql.Tx, requestID string, intent *OutputExportIntent,
	outputs string,
) *exit.Error {
	if intent == nil {
		return nil
	}
	if _, err := tx.Exec(`INSERT INTO request_output_exports(request_id,directory,
		outputs,state,updated_at) VALUES(?,?,?, 'pending', ?)`, requestID, intent.Directory,
		outputs, now()); err != nil {
		return exit.Internalf("cannot record output export for %s: %s", requestID, err)
	}
	return nil
}

// OutputExportOf reads one request's export row. Nil means the run exports nothing (a
// callable with no result files).
func (s *Store) OutputExportOf(requestID string) (*OutputExport, *exit.Error) {
	row, err := scanOutputExport(s.db.QueryRow(`SELECT request_id,directory,outputs,
		state,attempts,error_code,safe_error,published_paths,updated_at
		FROM request_output_exports WHERE request_id=?`, requestID))
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, exit.Internalf("cannot read output export for %s: %s", requestID, err)
	}
	return &row, nil
}

// OutputExportsOwed is every non-settled export of a request that has ended.
func (s *Store) OutputExportsOwed() ([]OutputExport, *exit.Error) {
	rows, err := s.db.Query(`SELECT e.request_id,e.directory,e.outputs,
		e.state,e.attempts,e.error_code,e.safe_error,e.published_paths,e.updated_at
		FROM request_output_exports e JOIN requests r ON r.id=e.request_id
		WHERE e.state IN ('pending','exporting','failed')
		  AND r.state IN ('succeeded','failed','canceled','refused','abandoned')
		ORDER BY r.created_at,r.id`)
	if err != nil {
		return nil, exit.Internalf("cannot read owed output exports: %s", err)
	}
	defer rows.Close()
	var out []OutputExport
	for rows.Next() {
		row, err := scanOutputExport(rows)
		if err != nil {
			return nil, exit.Internalf("cannot decode an owed output export: %s", err)
		}
		out = append(out, row)
	}
	return out, nil
}

func scanOutputExport(row interface{ Scan(...any) error }) (OutputExport, error) {
	var out OutputExport
	var outputs, paths string
	err := row.Scan(&out.RequestID, &out.Directory, &outputs, &out.State,
		&out.Attempts, &out.ErrorCode, &out.SafeError, &paths, &out.UpdatedAt)
	if err == nil {
		err = json.Unmarshal([]byte(outputs), &out.Outputs)
	}
	if err == nil {
		err = json.Unmarshal([]byte(paths), &out.PublishedPaths)
	}
	return out, err
}

// SettleOutputExport records the paths of the outputs verified in their folder: published
// when that is all of them, failed with why when it is not. An output missing from a failed
// export's paths was never delivered.
func (s *Store) SettleOutputExport(requestID string, paths []string, failure *exit.Error) *exit.Error {
	state, code, message := "published", "", ""
	if failure != nil {
		state, code, message = "failed", failure.ErrName(), failure.Message
	}
	encoded, _ := json.Marshal(append([]string{}, paths...))
	if _, err := s.db.Exec(`UPDATE request_output_exports SET state=?,published_paths=?,error_code=?,
		safe_error=?,updated_at=? WHERE request_id=?`, state, string(encoded), code, message, now(), requestID); err != nil {
		return exit.Internalf("cannot settle output export for %s: %s", requestID, err)
	}
	return nil
}

func (s *Store) SkipOutputExport(requestID, reason string) *exit.Error {
	if _, err := s.db.Exec(`UPDATE request_output_exports SET state='skipped',safe_error=?,
		updated_at=? WHERE request_id=? AND state IN ('pending','exporting','failed')`,
		reason, now(), requestID); err != nil {
		return exit.Internalf("cannot skip output export for %s: %s", requestID, err)
	}
	return nil
}
