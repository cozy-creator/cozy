package output

import (
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

// hostname is resolved once: some terminal emulators refuse to open file://
// links whose authority is not this machine's name.
var hostname = sync.OnceValue(func() string {
	name, err := os.Hostname()
	if err != nil {
		return ""
	}
	return name
})

// FileURL spells an absolute local path as a file:// URL carrying this host's
// name, percent-encoded per RFC 8089/3986.
func FileURL(path string) string {
	p := filepath.ToSlash(path)
	if !strings.HasPrefix(p, "/") {
		p = "/" + p
	}
	u := url.URL{Scheme: "file", Host: hostname(), Path: p}
	return u.String()
}

// Hyperlink wraps an absolute local path in an OSC 8 terminal hyperlink when
// this mode writes to a human on a terminal. Every other mode — --json, a pipe,
// a redirect — gets the path back byte-for-byte unchanged, and a path that is
// not absolute is never linked.
func (m Mode) Hyperlink(path string) string {
	if !m.Human || m.JSON || !m.TTY || !filepath.IsAbs(path) {
		return path
	}
	return "\x1b]8;;" + FileURL(path) + "\x1b\\" + path + "\x1b]8;;\x1b\\"
}
