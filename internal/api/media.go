package api

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"
)

// MEDIA BY OPAQUE ID, and there is no other shape.
//
// The class of bug this route is designed against has a name and a history: ComfyUI's
// `/view?filename=…` took a CLIENT-SUPPLIED PATH and served whatever it resolved to, and
// every patch since has been another attempt to sanitize a string that should never have
// been a parameter. The fix is structural rather than defensive — this route's ONLY input
// is a media id, the id is a random 24-hex-character token minted inside the terminal
// transaction, and it resolves through a database row that exists only because a terminal
// was accepted. Consequently:
//
//   - There is no representation of "serve this path". Not a rejected one — none.
//   - An output the orchestrator never published has no id, so it cannot be reached even
//     by a caller who knows exactly where the runtime wrote it.
//   - `..`, absolute paths, symlinks and encodings of them are not special cases: they are
//     simply not media ids, and a non-id is a 404.
//
// Three more properties on the way out: an explicit MIME allowlist (an unknown type is
// served as a download, never guessed), `nosniff` on every response, no inline SVG (an
// SVG is a script), and a Range implementation bounded to ONE range.

// mimeAllowed is the closed set of types served INLINE. Everything else is served as an
// attachment with a generic type: a browser that cannot render it must not be persuaded
// to try. `image/svg+xml` is deliberately absent — an SVG is an executable document, and
// serving one inline from this origin would hand a page script execution here.
var mimeAllowed = map[string]bool{
	"image/png":                 true,
	"image/jpeg":                true,
	"image/webp":                true,
	"image/gif":                 true,
	"video/mp4":                 true,
	"video/webm":                true,
	"audio/wav":                 true,
	"audio/mpeg":                true,
	"audio/flac":                true,
	"application/json":          true,
	"text/plain; charset=utf-8": true,
}

// extensions gives the download filename a suffix. The filename is derived from the
// OPAQUE ID and the recorded type — never from anything a client sent — so there is no
// header-injection or path surface in `Content-Disposition` at all.
var extensions = map[string]string{
	"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp", "image/gif": "gif",
	"video/mp4": "mp4", "video/webm": "webm", "audio/wav": "wav", "audio/mpeg": "mp3",
	"audio/flac": "flac", "application/json": "json",
}

// mediaIDLen is what NewID("med") produces: "med-" + 24 hex characters.
const mediaIDLen = 4 + 24

func plausibleMediaID(id string) bool {
	if len(id) != mediaIDLen || !strings.HasPrefix(id, "med-") {
		return false
	}
	for _, c := range id[4:] {
		if (c < '0' || c > '9') && (c < 'a' || c > 'f') {
			return false
		}
	}
	return true
}

func (s *Server) media(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("media_id")
	// The shape check comes first so a traversal attempt is refused before it ever
	// reaches the authority. It is not the defense — the opaque id is — but it keeps a
	// scanner's noise out of the query log.
	if !plausibleMediaID(id) {
		s.refuse(w, r, http.StatusNotFound, "not_found",
			"no media by that id on this host",
			"media is addressed by the opaque id a terminal published; there is no path form")
		return
	}
	out, requestID, attempt, e := s.store.Media(id)
	if e != nil {
		s.refuseTyped(w, r, e)
		return
	}
	if out == nil {
		s.refuse(w, r, http.StatusNotFound, "not_found",
			"no media by that id on this host",
			"only an output whose terminal this orchestrator accepted has an id at all")
		return
	}

	f, err := os.Open(out.Path)
	if err != nil {
		// The row says the bytes were published and they are not there: an honest 410,
		// not a 404. The difference matters — one means "never existed", the other
		// "was reclaimed", and a client retries differently.
		s.refuse(w, r, http.StatusGone, "media_reclaimed",
			fmt.Sprintf("%s was published by %s#%d and its bytes are no longer on disk",
				id, requestID, attempt),
			"local outputs live under the local root and are reclaimed with it")
		return
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		s.refuse(w, r, http.StatusInternalServerError, "internal", "cannot size the media", "")
		return
	}
	// The row is the authority on length. A file that disagrees is not the published
	// object, and serving it would put the record's digest on the wrong bytes.
	if info.Size() != out.Length {
		s.refuse(w, r, http.StatusInternalServerError, "media_length_mismatch",
			fmt.Sprintf("%s holds %d B on disk where its record declares %d B",
				id, info.Size(), out.Length),
			"the published bytes were altered after the terminal was accepted")
		return
	}
	size := out.Length

	contentType := out.MimeType
	disposition := "inline"
	if !mimeAllowed[contentType] {
		// An unlisted type is served as an opaque download. This is the ONE place the
		// declared type is overridden, and it overrides toward safety.
		contentType = "application/octet-stream"
		disposition = "attachment"
	}
	ext, ok := extensions[out.MimeType]
	if !ok {
		ext = "bin"
	}
	h := w.Header()
	h.Set("Content-Type", contentType)
	h.Set("Content-Disposition", fmt.Sprintf(`%s; filename="%s.%s"`, disposition, id, ext))
	h.Set("Accept-Ranges", "bytes")
	// The digest the manifest declared, so a client can verify the bytes it received
	// against what the terminal published rather than trusting the transfer.
	h.Set("X-Cozy-Digest", out.Digest)

	if r.Method == http.MethodHead {
		h.Set("Content-Length", strconv.FormatInt(size, 10))
		w.WriteHeader(http.StatusOK)
		return
	}

	start, end, ranged, bad := parseRange(r.Header.Get("Range"), size)
	if bad {
		h.Set("Content-Range", "bytes */"+strconv.FormatInt(size, 10))
		s.refuse(w, r, http.StatusRequestedRangeNotSatisfiable, "range_not_satisfiable",
			"the requested range lies outside the media",
			"one range per request; multipart ranges are not served")
		return
	}
	if !ranged {
		h.Set("Content-Length", strconv.FormatInt(size, 10))
		w.WriteHeader(http.StatusOK)
		_, _ = io.Copy(w, f)
		return
	}
	if _, err := f.Seek(start, io.SeekStart); err != nil {
		s.refuse(w, r, http.StatusInternalServerError, "internal", "cannot seek the media", "")
		return
	}
	h.Set("Content-Range", fmt.Sprintf("bytes %d-%d/%d", start, end, size))
	h.Set("Content-Length", strconv.FormatInt(end-start+1, 10))
	w.WriteHeader(http.StatusPartialContent)
	_, _ = io.CopyN(w, f, end-start+1)
}

// parseRange implements exactly ONE byte range and refuses everything else. Multipart
// ranges are a parser, a boundary generator and a second serialization for no consumer;
// a video element asks for one range at a time.
func parseRange(header string, size int64) (start, end int64, ranged, bad bool) {
	spec, ok := strings.CutPrefix(strings.TrimSpace(header), "bytes=")
	if !ok || spec == "" {
		return 0, 0, false, header != ""
	}
	if strings.Contains(spec, ",") {
		return 0, 0, false, true // a multi-range request is refused, never partly honoured
	}
	first, last, ok := strings.Cut(spec, "-")
	if !ok {
		return 0, 0, false, true
	}
	switch {
	case first == "" && last == "": // "bytes=-"
		return 0, 0, false, true
	case first == "": // a suffix range: the LAST n bytes
		n, err := strconv.ParseInt(last, 10, 64)
		if err != nil || n <= 0 {
			return 0, 0, false, true
		}
		if n > size {
			n = size
		}
		return size - n, size - 1, true, false
	default:
		start, err := strconv.ParseInt(first, 10, 64)
		if err != nil || start < 0 || start >= size {
			return 0, 0, false, true
		}
		end := size - 1
		if last != "" {
			n, err := strconv.ParseInt(last, 10, 64)
			if err != nil || n < start {
				return 0, 0, false, true
			}
			if n < end {
				end = n
			}
		}
		return start, end, true, false
	}
}
