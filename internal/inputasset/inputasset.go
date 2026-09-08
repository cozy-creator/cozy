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

// Probe reads only a regular file's stat and MIME prefix. It creates no identity;
// the prepared or original bytes still pass through Fingerprint before admission.
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

// Fingerprint streams one bounded regular file and returns the identity facts used by
// both the request record and InvocationSpec. Encoded media never enters a whole-file
// buffer. Unknown media type stays empty.
func Fingerprint(path string, max int64) (int64, string, string, *exit.Error) {
	file, err := os.Open(path)
	if err != nil {
		return 0, "", "", exit.New(exit.NotFound, "input asset %s: %s", path, err)
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		return 0, "", "", exit.New(exit.NotFound, "input asset %s: %s", path, err)
	}
	if !info.Mode().IsRegular() {
		return 0, "", "", exit.New(exit.Validation, "input asset %s is not a regular file", path)
	}
	if max > 0 && info.Size() > max {
		return 0, "", "", overCap(path, info.Size(), max)
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
				return 0, "", "", overCap(path, length, max)
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
			return 0, "", "", exit.New(exit.NotFound, "input asset %s: %s", path, readErr)
		}
	}
	if length != info.Size() {
		return 0, "", "", exit.Named(exit.Conflict, "input_asset_changed",
			"input asset %s changed length while it was fingerprinted", filepath.Base(path))
	}
	digest := "sha256:" + hex.EncodeToString(hash.Sum(nil))
	mediaType := normalizeMediaType(http.DetectContentType(prefix))
	return length, digest, mediaType, nil
}

// Verify proves that the borrowed file still matches the request row before a grant is
// minted. A mutable path can therefore never silently change an idempotent request.
func Verify(binding records.AssetBinding, max int64) *exit.Error {
	length, digest, mediaType, e := Fingerprint(binding.LocalPath, max)
	if e != nil {
		return e
	}
	if digest != binding.Digest || length != binding.Length {
		return exit.Named(exit.Conflict, "input_asset_changed",
			"input asset %s no longer matches its request identity", binding.FieldPath).
			WithRemedy("keep the original file available and unchanged until the request finishes; restore its original bytes before retrying")
	}
	if binding.MediaType != "" && mediaType != "" && binding.MediaType != mediaType {
		return exit.Named(exit.Conflict, "input_asset_type_changed",
			"input asset %s was recorded as %s and now sniffs as %s",
			binding.FieldPath, binding.MediaType, mediaType)
	}
	return nil
}

// Bind verifies a caller's identity claims while retaining the original path. No file
// is copied or owned by Creator; dispatch verifies it again before minting access.
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
	length, digest, mediaType, e := Fingerprint(path, max)
	if e != nil {
		return binding, e
	}
	if binding.Digest != "" && binding.Digest != digest {
		return binding, exit.Named(exit.Validation, "input_digest_mismatch",
			"input asset %s was declared as %s and its bytes hash to %s",
			binding.FieldPath, binding.Digest, digest)
	}
	if binding.Length != 0 && binding.Length != length {
		return binding, exit.Named(exit.Validation, "input_length_mismatch",
			"input asset %s was declared as %d B and holds %d B",
			binding.FieldPath, binding.Length, length)
	}
	if binding.MediaType != "" && mediaType != "" && binding.MediaType != mediaType {
		return binding, exit.Named(exit.Validation, "input_media_type",
			"input asset %s was declared as %s and its bytes sniff as %s",
			binding.FieldPath, binding.MediaType, mediaType)
	}
	binding.LocalPath = path
	binding.Digest, binding.Length, binding.MediaType = digest, length, mediaType
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
