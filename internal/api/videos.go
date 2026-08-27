package api

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"

	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
	"github.com/cozy-creator/cozy-creator-v2/internal/video"
)

type VideoController interface {
	Compose(video.ComposeRequest) (video.Composition, *exit.Error)
}

type VideoComposeRequest struct {
	Source             []byte `json:"source,omitempty"`
	BaseDir            string `json:"base_dir,omitempty"`
	CreativePlanDigest string `json:"creative_plan_digest,omitempty"`
	H3Endpoint         string `json:"h3_endpoint"`
	AssemblyEndpoint   string `json:"assembly_endpoint"`
	Worker             string `json:"worker,omitempty"`
}

func (s *Server) composeVideo(w http.ResponseWriter, r *http.Request) {
	if s.videos == nil {
		s.refuseTyped(w, r, exit.Unavailablef("this LocalService has no video composer"))
		return
	}
	if !s.cliAuthenticated(r) {
		s.refuse(w, r, http.StatusForbidden, "cli_credential_required",
			"video composition reads and stages caller-owned local assets",
			"use the OS-protected Cozy CLI on this host")
		return
	}
	body, err := io.ReadAll(io.LimitReader(r.Body, MaxBody+1))
	if err != nil || len(body) > MaxBody {
		s.refuse(w, r, http.StatusRequestEntityTooLarge, "body_too_large",
			"the video composition request is unreadable or oversized", "source assets never ride JSON")
		return
	}
	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.DisallowUnknownFields()
	var request VideoComposeRequest
	if err := decoder.Decode(&request); err != nil {
		s.refuse(w, r, http.StatusBadRequest, "malformed_body",
			"the video composition request is not one strict object: "+err.Error(), "")
		return
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		s.refuse(w, r, http.StatusBadRequest, "malformed_body",
			"the video composition request carries trailing JSON", "")
		return
	}
	composition, problem := s.videos.Compose(video.ComposeRequest{
		Source: request.Source, BaseDir: request.BaseDir,
		CreativePlanDigest: request.CreativePlanDigest,
		H3Endpoint:         request.H3Endpoint, AssemblyEndpoint: request.AssemblyEndpoint,
		Worker: request.Worker,
	})
	if problem != nil {
		s.refuseTyped(w, r, problem)
		return
	}
	s.ok(w, r, http.StatusOK, composition)
}
