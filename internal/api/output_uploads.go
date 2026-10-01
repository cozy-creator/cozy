package api

import (
	"encoding/json"
	"io"
	"net/http"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
)

// OutputUploadRequest names one retained output of a run and the repository its private
// checkpoint goes to. An empty output selects the run's only retained output.
type OutputUploadRequest struct {
	Output      string `json:"output,omitempty"`
	Destination string `json:"destination"`
}

type outputUploader interface {
	Upload(records.Request, string, string) (records.OutputUpload, *exit.Error)
}

func (s *Server) uploadJobOutput(w http.ResponseWriter, r *http.Request) {
	row, ok := s.jobRow(w, r)
	if !ok {
		return
	}
	var body OutputUploadRequest
	decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, 4096))
	var trailing any
	if err := decoder.Decode(&body); err != nil || body.Destination == "" || decoder.Decode(&trailing) != io.EOF {
		s.refuseTyped(w, r, exit.New(exit.Validation, "an upload names one destination and at most one output"))
		return
	}
	uploader, ok := s.machineExecutions.(outputUploader)
	if !ok {
		s.refuseTyped(w, r, exit.Unavailablef("this daemon cannot upload retained outputs"))
		return
	}
	upload, problem := uploader.Upload(row, body.Output, body.Destination)
	if problem == nil {
		var o *records.Operation
		if o, problem = s.store.Operation(upload.ID); problem == nil && o == nil {
			problem = exit.Internalf("the upload was journaled and cannot be read back")
		}
		if problem == nil {
			s.ok(w, r, http.StatusAccepted, s.operationLifecycle(*o))
			return
		}
	}
	s.refuseTyped(w, r, problem)
}
