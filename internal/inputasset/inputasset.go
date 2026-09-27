// Package inputasset owns the local byte half of request asset bindings. It is the one
// place that reads, fingerprints, and later re-verifies those files; launch owns
// the schema field path, while the orchestrator owns the protocol grant.
package inputasset

import (
	"crypto/sha256"
	"encoding/hex"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
)

// MaxBytes is the compatibility fallback for an older PackageInterface with no effective
// field bound. Current PackageInterfaces carry each field's selected max_bytes; no private
// aggregate cap exists here.
const MaxBytes = int64(64 << 20)

// ValidateID keeps asset field paths out of the protocol's two reserved input
// namespaces. Without it, map decoding could collapse two grant entries onto one id.
func ValidateID(fieldPath string) *exit.Error {
	if fieldPath == "payload" || strings.HasPrefix(fieldPath, "tree:") {
		return exit.New(exit.Validation,
			"input asset field %q collides with a reserved protocol input id", fieldPath).
			WithRemedy("`payload` names the request document and `tree:` names job tree capabilities")
	}
	return nil
}

// Probe reads only a regular file's stat and MIME prefix. It creates no identity.
func Probe(path string) (int64, string, *exit.Error) {
	file, err := os.Open(path)
	if err != nil {
		return 0, "", exit.New(exit.NotFound, "input asset %s: %s", path, err)
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		return 0, "", exit.New(exit.NotFound, "input asset %s: %s", path, err)
	}
	if !info.Mode().IsRegular() {
		return 0, "", exit.New(exit.Validation, "input asset %s is not a regular file", path)
	}
	prefix := make([]byte, 512)
	n, err := io.ReadFull(file, prefix)
	if err != nil && err != io.EOF && err != io.ErrUnexpectedEOF {
		return 0, "", exit.New(exit.NotFound, "input asset %s: %s", path, err)
	}
	return info.Size(), normalizeMediaType(http.DetectContentType(prefix[:n])), nil
}

// Facts are one file's identity as Fingerprint read it.
type Facts struct {
	Length    int64
	Digest    string
	MediaType string
	// ModTime is the file's modification time, in Unix nanoseconds, before it was read.
	ModTime int64
}

// Fingerprint streams one bounded regular file and returns the identity facts used by
// both the request record and InvocationSpec. Encoded media never enters a whole-file
// buffer. Unknown media type stays empty.
func Fingerprint(path string, max int64) (Facts, *exit.Error) {
	file, err := os.Open(path)
	if err != nil {
		return Facts{}, exit.New(exit.NotFound, "input asset %s: %s", path, err)
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		return Facts{}, exit.New(exit.NotFound, "input asset %s: %s", path, err)
	}
	if !info.Mode().IsRegular() {
		return Facts{}, exit.New(exit.Validation, "input asset %s is not a regular file", path)
	}
	if max > 0 && info.Size() > max {
		return Facts{}, overCap(path, info.Size(), max)
	}
	hash := sha256.New()
	prefix := make([]byte, 0, 512)
	buffer := make([]byte, 64<<10)
	var length int64
	for {
		n, readErr := file.Read(buffer)
		if n > 0 {
			length += int64(n)
			if max > 0 && length > max {
				return Facts{}, overCap(path, length, max)
			}
			_, _ = hash.Write(buffer[:n])
			if len(prefix) < cap(prefix) {
				take := min(n, cap(prefix)-len(prefix))
				prefix = append(prefix, buffer[:take]...)
			}
		}
		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			return Facts{}, exit.New(exit.NotFound, "input asset %s: %s", path, readErr)
		}
	}
	if length != info.Size() {
		return Facts{}, exit.Named(exit.Conflict, "input_asset_changed",
			"input asset %s changed length while it was fingerprinted", filepath.Base(path))
	}
	return Facts{Length: length, Digest: "sha256:" + hex.EncodeToString(hash.Sum(nil)),
		MediaType: normalizeMediaType(http.DetectContentType(prefix)), ModTime: info.ModTime().UnixNano()}, nil
}

// unchanged answers whether a file still has the size and modification time its recorded
// digest was computed at, so that digest still names its bytes without reading them.
func unchanged(path string, binding records.AssetBinding) (os.FileInfo, bool) {
	info, err := os.Stat(path)
	if err != nil || !info.Mode().IsRegular() {
		return nil, false
	}
	return info, binding.ModTime != 0 && info.Size() == binding.Length && info.ModTime().UnixNano() == binding.ModTime
}

// Verify proves that the borrowed file still holds the request's bytes before a grant is
// minted, so a mutable path can never silently change an idempotent request. The digest
// recorded when the asset was attached is reused while the file keeps that size and
// modification time; a changed file is hashed again and must still match.
func Verify(binding records.AssetBinding, max int64) *exit.Error {
	if _, same := unchanged(binding.LocalPath, binding); same {
		return nil
	}
	facts, e := Fingerprint(binding.LocalPath, max)
	if e != nil {
		return e
	}
	if facts.Digest != binding.Digest || facts.Length != binding.Length {
		return exit.Named(exit.Conflict, "input_asset_changed",
			"input asset %s no longer matches its request identity", binding.FieldPath).
			WithRemedy("keep the original file available and unchanged until the request finishes; restore its original bytes before retrying")
	}
	return nil
}

// Bind admits a caller's asset while retaining the original path. No file is copied or
// owned by Creator. A caller that fingerprinted the file already names its digest, size and
// modification time; while the file still has them, that digest is its identity and the
// bytes are not read again. Otherwise the file is hashed here and must match what the
// caller declared.
func Bind(binding records.AssetBinding, max int64) (records.AssetBinding, *exit.Error) {
	if binding.FieldPath == "" {
		return binding, exit.New(exit.Validation, "an input asset names no request field path")
	}
	if e := ValidateID(binding.FieldPath); e != nil {
		return binding, e
	}
	path, err := filepath.Abs(binding.LocalPath)
	if err != nil {
		return binding, exit.New(exit.NotFound, "cannot resolve input asset %s: %s", binding.LocalPath, err)
	}
	if info, same := unchanged(path, binding); same && binding.Digest != "" {
		if max > 0 && info.Size() > max {
			return binding, overCap(path, info.Size(), max)
		}
		_, mediaType, e := Probe(path)
		if e != nil {
			return binding, e
		}
		binding.LocalPath, binding.MediaType = path, mediaType
		return binding, nil
	}
	facts, e := Fingerprint(path, max)
	if e != nil {
		return binding, e
	}
	if binding.Digest != "" && binding.Digest != facts.Digest {
		return binding, exit.Named(exit.Validation, "input_digest_mismatch",
			"input asset %s was declared as %s and its bytes hash to %s",
			binding.FieldPath, binding.Digest, facts.Digest)
	}
	if binding.Length != 0 && binding.Length != facts.Length {
		return binding, exit.Named(exit.Validation, "input_length_mismatch",
			"input asset %s was declared as %d B and holds %d B",
			binding.FieldPath, binding.Length, facts.Length)
	}
	// The type the bytes sniff as wins over a caller's label for the same bytes.
	binding.LocalPath = path
	binding.Digest, binding.Length, binding.MediaType, binding.ModTime = facts.Digest, facts.Length, facts.MediaType, facts.ModTime
	return binding, nil
}

func normalizeMediaType(mediaType string) string {
	mediaType = strings.ToLower(strings.TrimSpace(strings.Split(mediaType, ";")[0]))
	switch mediaType {
	case "audio/wave", "audio/x-wav":
		mediaType = "audio/wav"
	case "application/octet-stream":
		mediaType = ""
	}
	return mediaType
}

func overCap(path string, size, max int64) *exit.Error {
	return exit.Named(exit.Validation, "input_over_cap",
		"input asset %s is %d B and this deployment admits %d B per input", filepath.Base(path), size, max).
		WithRemedy("resize or recompress the input before submitting it")
}
