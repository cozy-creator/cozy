package api

import (
	"embed"
	"net/http"
	"strings"
)

// The STUB PAGE — cl-007's unblocked half, landing here because it is what proves the
// serving path, the security posture, and the API wiring end to end. The real UI is
// deliberately undesigned; nothing beyond this stub may be built before the owner's
// definition exists, and this file is small on purpose.
//
// Three assets, embedded with go:embed, and the CSP is what makes the split worth it:
// script and style are SEPARATE FILES, so the page needs neither `script-src
// 'unsafe-inline'` nor `style-src 'unsafe-inline'`. A page that holds a credential and
// permits inline script is one HTML-injection bug away from handing it over — and the
// cheapest way never to have that bug is to have no inline script at all.

//go:embed stub/index.html stub/app.js stub/app.css
var stubFS embed.FS

// stubAssets is the CLOSED set of names this route serves. It is a map rather than a
// directory walk so "which files can be fetched from this origin" is a list a reader can
// see, and a file added to the embed by accident is not automatically public.
var stubAssets = map[string]struct {
	file        string
	contentType string
}{
	"/":        {"stub/index.html", "text/html; charset=utf-8"},
	"/app.js":  {"stub/app.js", "text/javascript; charset=utf-8"},
	"/app.css": {"stub/app.css", "text/css; charset=utf-8"},
}

func (s *Server) stub(w http.ResponseWriter, r *http.Request) {
	asset, ok := stubAssets[r.URL.Path]
	if !ok {
		// Anything else under `/` is not a page this host has. The typed envelope, so a
		// client meets one error shape everywhere.
		s.refuse(w, r, http.StatusNotFound, "not_found",
			"this host serves no "+r.URL.Path, "the local client API is under /v1/")
		return
	}
	data, err := stubFS.ReadFile(asset.file)
	if err != nil {
		s.refuse(w, r, http.StatusInternalServerError, "internal",
			"the embedded stub asset is unreadable", "")
		return
	}
	h := w.Header()
	h.Set("Content-Type", asset.contentType)
	// The stub's own policy, tighter than the baseline: its script and style are
	// same-origin FILES, so neither needs 'unsafe-inline'. `connect-src 'self'` is what
	// lets it reach this API and nothing else.
	h.Set("Content-Security-Policy",
		"default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; "+
			"connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(data)
}

// StubReachable is the one fact `--open` needs before it launches a browser: the page
// exists in this binary. A `--open` that opened a 404 would be a worse first impression
// than one that said so.
func StubReachable() bool {
	data, err := stubFS.ReadFile("stub/index.html")
	return err == nil && strings.Contains(string(data), "cozy LocalService")
}
