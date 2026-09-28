package api

func (s *Server) outputExportOf(requestID string) *OutputExportRef {
	export, problem := s.store.OutputExportOf(requestID)
	if problem != nil || export == nil {
		return nil
	}
	return &OutputExportRef{
		Directory: export.Directory, State: export.State,
		ErrorCode: export.ErrorCode, Error: export.SafeError,
		Paths: append([]string{}, export.PublishedPaths...),
	}
}
