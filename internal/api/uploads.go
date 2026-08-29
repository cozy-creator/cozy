package api

import (
	"io"
	"net/http"
	"time"

	"github.com/cozy-creator/cozy-creator/internal/upload"
)

func (s *Server) putUpload(w http.ResponseWriter, r *http.Request) {
	stored, problem := upload.Put(s.layout, r.Body, r.ContentLength, r.Header.Get("Content-Type"))
	if problem != nil {
		s.refuseTyped(w, r, problem)
		return
	}
	s.ok(w, r, http.StatusCreated, stored)
}

func (s *Server) getUpload(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("upload_id")
	file, digest, problem := upload.Open(s.layout, id)
	if problem != nil {
		s.refuseTyped(w, r, problem)
		return
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		s.refuse(w, r, http.StatusInternalServerError, "upload_unreadable", "cannot size upload", "")
		return
	}
	prefix := make([]byte, min(512, info.Size()))
	n, _ := io.ReadFull(file, prefix)
	_, _ = file.Seek(0, io.SeekStart)
	w.Header().Set("Content-Type", http.DetectContentType(prefix[:n]))
	w.Header().Set("X-Cozy-Digest", digest)
	w.Header().Set("Content-Disposition", "inline; filename=\""+id+"\"")
	http.ServeContent(w, r, id, time.Time{}, file)
}
