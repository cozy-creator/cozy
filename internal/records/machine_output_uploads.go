package records

import (
	"encoding/json"

	"github.com/cozy-creator/cozy/internal/exit"
)

// OutputUpload is one retained output's upload, from the machine holding it to a private
// checkpoint: a numbered operation of the run that produced it. Operation is fixed by run,
// output and destination, so a repeated upload resumes the same Hub publication.
type OutputUpload struct {
	Number      int64  `json:"number,omitempty"`
	ID          string `json:"id,omitempty"`
	Request     string `json:"request"`
	Output      string `json:"output"`
	Destination string `json:"destination"`
	Operation   string `json:"operation"`
	State       string `json:"state"` // uploading | uploaded | failed | canceled
	Checkpoint  string `json:"checkpoint,omitempty"`
	ErrorCode   string `json:"error_code,omitempty"`
	Error       string `json:"error,omitempty"`
}

func (u OutputUpload) Finished() bool { return u.State != "uploading" }

// OutputUploadOf is the upload an operation row records.
func OutputUploadOf(o Operation) OutputUpload {
	upload := o.Upload
	upload.Number, upload.ID, upload.ErrorCode, upload.Error = o.Number, o.ID, o.ErrorCode, o.Error
	upload.State = map[string]string{"succeeded": "uploaded"}[o.State]
	if upload.State == "" {
		upload.State = o.State
	}
	var result struct {
		Checkpoint string `json:"checkpoint"`
	}
	if json.Unmarshal(o.Result, &result) == nil {
		upload.Checkpoint = result.Checkpoint
	}
	return upload
}

// BeginOutputUpload journals an upload, or answers the one still uploading or uploaded for
// the same publication. A failed or canceled upload is asked again under a new number.
func (s *Store) BeginOutputUpload(machine, hub string, upload OutputUpload) (OutputUpload, bool, *exit.Error) {
	priors, problem := s.uploadsWhere(`json_extract(o.selection,'$.request')=? AND json_extract(o.selection,'$.operation')=?`, upload.Request, upload.Operation)
	if problem != nil {
		return upload, false, problem
	}
	for _, prior := range priors {
		if prior.State == "uploading" || prior.State == "uploaded" {
			return prior, false, nil
		}
	}
	raw, err := json.Marshal(OutputUpload{Request: upload.Request, Output: upload.Output, Destination: upload.Destination, Operation: upload.Operation})
	if err != nil {
		return upload, false, exit.Internalf("cannot encode upload: %s", err)
	}
	tx, err := s.db.Begin()
	if err != nil {
		return upload, false, exit.Internalf("cannot begin upload: %s", err)
	}
	defer tx.Rollback()
	o := Operation{ID: NewID("upload"), Kind: "upload", Machine: machine, Hub: hub, State: "uploading"}
	if problem := s.insertOperation(tx, &o, raw); problem != nil {
		return upload, false, problem
	}
	if err := tx.Commit(); err != nil {
		return upload, false, exit.Internalf("cannot commit upload: %s", err)
	}
	upload.Number, upload.ID, upload.State = o.Number, o.ID, "uploading"
	return upload, true, nil
}

// SettleOutputUpload records where an upload ended; a canceled one stays canceled.
func (s *Store) SettleOutputUpload(upload OutputUpload) *exit.Error {
	state := map[string]string{"uploaded": "succeeded", "failed": "failed"}[upload.State]
	if state == "" {
		return exit.New(exit.Validation, "invalid upload settlement")
	}
	result, _ := json.Marshal(map[string]string{"checkpoint": upload.Checkpoint})
	if _, err := s.db.Exec(`UPDATE operations SET state=?,result=?,error_code=?,error=?,updated_at=? WHERE id=? AND state='uploading'`,
		state, result, upload.ErrorCode, upload.Error, now(), upload.ID); err != nil {
		return exit.Internalf("cannot record output upload: %s", err)
	}
	return nil
}

// OutputUploads is every upload of a run's outputs, oldest first.
func (s *Store) OutputUploads(request string) ([]OutputUpload, *exit.Error) {
	return s.uploadsWhere(`json_extract(o.selection,'$.request')=?`, request)
}

// UnfinishedOutputUploads is every upload a restarted daemon resumes.
func (s *Store) UnfinishedOutputUploads() ([]OutputUpload, *exit.Error) {
	return s.uploadsWhere(`o.state='uploading'`)
}

func (s *Store) uploadsWhere(where string, args ...any) ([]OutputUpload, *exit.Error) {
	rows, problem := s.queryOperations(`o.kind='upload' AND `+where+` ORDER BY j.number`, args...)
	out := make([]OutputUpload, 0, len(rows))
	for _, row := range rows {
		out = append(out, OutputUploadOf(row))
	}
	return out, problem
}
