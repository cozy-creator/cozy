package api

import (
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"

	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/resultfiles"
)

// A RUN'S PRODUCTS FOR A STANDARD PLAYER. A player cannot present the daemon's bearer, so
// these URLs carry a capability instead: an HMAC of the run under a key this daemon draws
// at start. The one route answers, from this daemon's mirror of the run's output log:
//
//	<output>.m3u8        an HLS EVENT playlist of a growing video (init as EXT-X-MAP, one
//	                     EXTINF per part), EXT-X-ENDLIST once the run ended: mpv, VLC,
//	                     Safari, hls.js
//	<output>.<ext>       the output's current bytes (a single output's latest product)
//	<sha256 hex>.<ext>   one stored part or product, by digest (`.mp4` init, `.m4s` media)
//
// with one Range per request. The Hub is never involved; the bytes are the ones already here.

var streamKey = sync.OnceValue(func() []byte {
	key := make([]byte, 32)
	_, _ = rand.Read(key)
	return key
})

func streamToken(requestID string) string {
	mac := hmac.New(sha256.New, streamKey())
	mac.Write([]byte(requestID))
	return hex.EncodeToString(mac.Sum(nil))[:32]
}

// StreamPath is where a run's products are served to players, relative to the daemon.
func StreamPath(row records.Request) string {
	return fmt.Sprintf("/v1/local/runs/%d/%s/", row.Number, streamToken(row.ID))
}

func (s *Server) runProduct(w http.ResponseWriter, r *http.Request) {
	notFound := func() {
		s.refuse(w, r, http.StatusNotFound, "not_found", "no such product of this run", "")
	}
	row, problem := s.store.RequestByReference(r.PathValue("number"))
	if problem != nil || row == nil || !hmac.Equal([]byte(r.PathValue("token")), []byte(streamToken(row.ID))) {
		notFound()
		return
	}
	products, problem := s.store.Products(row.ID)
	if problem != nil {
		s.refuseTyped(w, r, problem)
		return
	}
	store := s.layout.Products(row.Org, row.ID)
	file := r.PathValue("file")
	stem, extension, _ := strings.Cut(file, ".")
	latest := map[string]records.Product{}
	for _, product := range products {
		if product.Op == records.ProductSet {
			latest[product.Output] = product
		}
	}
	switch product, ok := latest[stem]; {
	case ok && extension == "m3u8" && len(product.Parts) > 1:
		ended := records.Settled(row.State)
		w.Header().Set("Content-Type", "application/vnd.apple.mpegurl")
		_, _ = io.WriteString(w, playlist(product, ended))
	case ok && strings.TrimPrefix(resultfiles.Extension(product.MediaType), ".") == extension:
		serveParts(w, r, s, product.MediaType, partFiles(store, product))
	case len(stem) == 64:
		for _, product := range products {
			for _, part := range append(product.Parts, records.ProductPart{Digest: product.Digest, Length: product.Length}) {
				path := filepath.Join(store, strings.TrimPrefix(part.Digest, "sha256:"))
				if info, err := os.Stat(path); part.Digest == "sha256:"+stem && err == nil && info.Size() == part.Length {
					serveParts(w, r, s, product.MediaType, []partFile{{path, part.Length}})
					return
				}
			}
		}
		notFound()
	default:
		notFound()
	}
}

// playlist is a composite video product as HLS: its first part is the init segment.
func playlist(product records.Product, ended bool) string {
	var b strings.Builder
	target := 1.0
	for _, part := range product.Parts[1:] {
		target = math.Max(target, math.Ceil(float64(part.DurationUs)/1e6))
	}
	fmt.Fprintf(&b, "#EXTM3U\n#EXT-X-VERSION:7\n#EXT-X-PLAYLIST-TYPE:EVENT\n#EXT-X-TARGETDURATION:%d\n", int(target))
	// ffmpeg-based players (mpv, ffplay) admit segments by extension: `.mp4` and `.m4s`.
	fmt.Fprintf(&b, "#EXT-X-INDEPENDENT-SEGMENTS\n#EXT-X-MAP:URI=\"%s.mp4\"\n", strings.TrimPrefix(product.Parts[0].Digest, "sha256:"))
	for _, part := range product.Parts[1:] {
		fmt.Fprintf(&b, "#EXTINF:%.6f,\n%s.m4s\n", float64(part.DurationUs)/1e6, strings.TrimPrefix(part.Digest, "sha256:"))
	}
	if ended {
		b.WriteString("#EXT-X-ENDLIST\n")
	}
	return b.String()
}

type partFile struct {
	path   string
	length int64
}

func partFiles(store string, product records.Product) []partFile {
	if len(product.Parts) == 0 {
		return []partFile{{filepath.Join(store, strings.TrimPrefix(product.Digest, "sha256:")), product.Length}}
	}
	files := make([]partFile, len(product.Parts))
	for index, part := range product.Parts {
		files[index] = partFile{filepath.Join(store, strings.TrimPrefix(part.Digest, "sha256:")), part.Length}
	}
	return files
}

// serveParts answers the files concatenated, with one Range.
func serveParts(w http.ResponseWriter, r *http.Request, s *Server, mediaType string, files []partFile) {
	var size int64
	for _, file := range files {
		size += file.length
	}
	h := w.Header()
	if !mimeAllowed[mediaType] {
		mediaType = "application/octet-stream"
	}
	h.Set("Content-Type", mediaType)
	h.Set("Accept-Ranges", "bytes")
	start, end, ranged, bad := parseRange(r.Header.Get("Range"), size)
	if bad {
		h.Set("Content-Range", "bytes */"+strconv.FormatInt(size, 10))
		s.refuse(w, r, http.StatusRequestedRangeNotSatisfiable, "range_not_satisfiable", "the requested range lies outside the product", "")
		return
	}
	if !ranged {
		start, end = 0, size-1
	}
	h.Set("Content-Length", strconv.FormatInt(end-start+1, 10))
	if ranged {
		h.Set("Content-Range", fmt.Sprintf("bytes %d-%d/%d", start, end, size))
		w.WriteHeader(http.StatusPartialContent)
	} else {
		w.WriteHeader(http.StatusOK)
	}
	if r.Method == http.MethodHead {
		return
	}
	remaining := end - start + 1
	for _, file := range files {
		if start >= file.length {
			start -= file.length
			continue
		}
		f, err := os.Open(file.path)
		if err != nil {
			return
		}
		_, _ = f.Seek(start, io.SeekStart)
		copied, _ := io.CopyN(w, f, min(remaining, file.length-start))
		f.Close()
		remaining -= copied
		start = 0
		if remaining <= 0 {
			return
		}
	}
}
