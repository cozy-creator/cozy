// Package web embeds the localhost frontend shipped with Cozy Creator.
package web

import (
	"embed"
	"net/http"
)

//go:embed index.html app.css app.js
var assets embed.FS

// Handler serves only files embedded in the release binary.
func Handler() http.Handler { return http.FileServer(http.FS(assets)) }
