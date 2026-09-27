package records

import (
	"encoding/json"

	"github.com/cozy-creator/cozy/internal/exit"
)

// OutputUpload is one retained output's upload, from the machine holding it to a private
// checkpoint, journaled on the run that produced it. Operation is fixed by run, output
// and destination, so a repeated upload resumes the same Hub publication.
type OutputUpload struct {
	Output      string `json:"output"`
	Destination string `json:"destination"`
	Operation   string `json:"operation"`
	State       string `json:"state"` // uploading | uploaded | failed
	Checkpoint  string `json:"checkpoint,omitempty"`
	ErrorCode   string `json:"error_code,omitempty"`
	Error       string `json:"error,omitempty"`
}

func (u OutputUpload) Finished() bool { return u.State == "uploaded" || u.State == "failed" }

func (s *Store) RecordOutputUpload(request string, upload OutputUpload) *exit.Error {
	raw, err := json.Marshal(upload)
	if err != nil || len(raw) > 16<<10 {
		return exit.New(exit.Validation, "output upload record exceeds its metadata bound")
	}
	if _, err := s.db.Exec(`INSERT INTO request_events(request_id,type,attempt,payload,at) VALUES(?,'machine.output_upload',0,?,?)`,
		request, raw, now()); err != nil {
		return exit.Internalf("cannot record output upload: %s", err)
	}
	return nil
}

const latestOutputUploads = `SELECT upload.request_id,upload.payload FROM request_events upload
 WHERE upload.type='machine.output_upload' AND NOT EXISTS(SELECT 1 FROM request_events newer
 WHERE newer.request_id=upload.request_id AND newer.type=upload.type AND newer.seq>upload.seq
 AND json_extract(newer.payload,'$.operation')=json_extract(upload.payload,'$.operation'))`

// OutputUploads is the latest state of each of a run's output uploads.
func (s *Store) OutputUploads(request string) ([]OutputUpload, *exit.Error) {
	byRun, problem := s.outputUploads(latestOutputUploads+` AND upload.request_id=? ORDER BY upload.seq`, request)
	return byRun[request], problem
}

// UnfinishedOutputUploads is every upload a restarted daemon resumes, by run.
func (s *Store) UnfinishedOutputUploads() (map[string][]OutputUpload, *exit.Error) {
	return s.outputUploads(latestOutputUploads + ` AND json_extract(upload.payload,'$.state')='uploading' ORDER BY upload.seq`)
}

func (s *Store) outputUploads(query string, args ...any) (map[string][]OutputUpload, *exit.Error) {
	rows, err := s.db.Query(query, args...)
	if err != nil {
		return nil, exit.Internalf("cannot read output uploads: %s", err)
	}
	defer rows.Close()
	out := map[string][]OutputUpload{}
	for rows.Next() {
		var request string
		var raw []byte
		var upload OutputUpload
		if err := rows.Scan(&request, &raw); err != nil || json.Unmarshal(raw, &upload) != nil {
			return nil, exit.Internalf("cannot decode an output upload")
		}
		out[request] = append(out[request], upload)
	}
	return out, nil
}
