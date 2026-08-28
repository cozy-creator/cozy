// Package inputasset owns the local byte half of request asset bindings. It is the one
// place that reads, fingerprints, stages, and later re-verifies those files; launch owns
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
	"sync"

	"github.com/cozy-creator/cozy-creator/internal/canonical"
	"github.com/cozy-creator/cozy-creator/internal/exit"
	"github.com/cozy-creator/cozy-creator/internal/home"
	"github.com/cozy-creator/cozy-creator/internal/records"
)

// MaxBytes is the compatibility fallback for an older descriptor with no effective
// field bound. Current descriptors carry each field's selected max_bytes; no private
// aggregate cap exists here.
const MaxBytes = int64(64 << 20)

var ownership sync.Mutex

// Guard serializes the short stage+request-record and live-reference+drop critical
// sections. Files and SQLite cannot share a transaction; this lock is the single process
// boundary that prevents terminal cleanup from deleting a digest between its staging and
// the request row that takes ownership of it.
func Guard() func() {
	ownership.Lock()
	return ownership.Unlock
}

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

// Verify proves that the durable file still matches the request row before a grant is
// minted. A mutable path can therefore never silently change an idempotent request.
func Verify(binding records.AssetBinding, max int64) *exit.Error {
	length, digest, mediaType, e := Fingerprint(binding.LocalPath, max)
	if e != nil {
		return e
	}
	if digest != binding.Digest || length != binding.Length {
		return exit.Named(exit.Conflict, "input_asset_changed",
			"input asset %s no longer matches its request identity", binding.FieldPath).
			WithRemedy("the service stages immutable input bytes before recording a request; restore the local input store before retrying")
	}
	if binding.MediaType != "" && mediaType != "" && binding.MediaType != mediaType {
		return exit.Named(exit.Conflict, "input_asset_type_changed",
			"input asset %s was recorded as %s and now sniffs as %s",
			binding.FieldPath, binding.MediaType, mediaType)
	}
	return nil
}

// Stage verifies a caller's claims and atomically copies the bytes into the service's
// private content-addressed input store. If the original path has disappeared but the
// claimed digest is already staged, the durable copy answers; this is what makes an
// idempotent replay independent of the submitting CLI still being alive.
func Stage(layout home.Layout, binding records.AssetBinding, max int64) (records.AssetBinding, *exit.Error) {
	if binding.FieldPath == "" {
		return binding, exit.New(exit.Validation, "an input asset names no request field path")
	}
	if e := ValidateID(binding.FieldPath); e != nil {
		return binding, e
	}
	source := binding.LocalPath
	if _, err := os.Stat(source); err != nil && validDigest(binding.Digest) {
		source = layout.InputAsset(binding.Digest)
	}
	stagingPath, length, digest, mediaType, e := stageSource(layout, source, max)
	if e != nil {
		return binding, e
	}
	defer os.Remove(stagingPath)
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

	binding.Digest, binding.Length, binding.MediaType = digest, length, mediaType
	destination := layout.InputAsset(digest)
	if _, err := os.Stat(destination); err == nil {
		_, existingDigest, _, check := Fingerprint(destination, max)
		if check != nil || existingDigest != digest {
			return binding, exit.Named(exit.Conflict, "input_store_corrupt",
				"the staged object for input asset %s does not match %s", binding.FieldPath, digest).
				WithRemedy("repair or remove that exact object from the local input store, then submit again")
		}
	} else if !os.IsNotExist(err) {
		return binding, exit.Internalf("cannot inspect the input store for %s: %s", binding.FieldPath, err)
	} else {
		err := os.Rename(stagingPath, destination)
		if err != nil {
			// Another concurrent submission of the same digest may have won the rename.
			if _, statErr := os.Stat(destination); statErr != nil {
				return binding, exit.Internalf("cannot commit input asset %s: %s", binding.FieldPath, err)
			}
		}
	}
	binding.LocalPath = destination
	return binding, nil
}

func stageSource(layout home.Layout, source string, max int64) (string, int64, string, string, *exit.Error) {
	file, err := os.Open(source)
	if err != nil {
		return "", 0, "", "", exit.New(exit.NotFound, "input asset %s: %s", source, err)
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil || !info.Mode().IsRegular() {
		return "", 0, "", "", exit.New(exit.Validation, "input asset %s is not a regular file", source)
	}
	if max > 0 && info.Size() > max {
		return "", 0, "", "", overCap(source, info.Size(), max)
	}
	staging, err := os.CreateTemp(layout.Inputs, ".asset-*")
	if err != nil {
		return "", 0, "", "", exit.Internalf("cannot stage input asset: %s", err)
	}
	path := staging.Name()
	keep := false
	defer func() {
		_ = staging.Close()
		if !keep {
			_ = os.Remove(path)
		}
	}()
	hash := sha256.New()
	prefix := make([]byte, 0, 512)
	buffer := make([]byte, 64<<10)
	var length int64
	for {
		n, readErr := file.Read(buffer)
		if n > 0 {
			length += int64(n)
			if max > 0 && length > max {
				return "", 0, "", "", overCap(source, length, max)
			}
			_, _ = hash.Write(buffer[:n])
			if len(prefix) < cap(prefix) {
				take := min(n, cap(prefix)-len(prefix))
				prefix = append(prefix, buffer[:take]...)
			}
			if _, err := staging.Write(buffer[:n]); err != nil {
				return "", 0, "", "", exit.Internalf("cannot stage input asset: %s", err)
			}
		}
		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			return "", 0, "", "", exit.New(exit.NotFound, "input asset %s: %s", source, readErr)
		}
	}
	if length != info.Size() {
		return "", 0, "", "", exit.Named(exit.Conflict, "input_asset_changed",
			"input asset %s changed length while it was staged", filepath.Base(source))
	}
	if err := staging.Sync(); err != nil {
		return "", 0, "", "", exit.Internalf("cannot sync staged input asset: %s", err)
	}
	if err := staging.Close(); err != nil {
		return "", 0, "", "", exit.Internalf("cannot close staged input asset: %s", err)
	}
	mediaType := normalizeMediaType(http.DetectContentType(prefix))
	keep = true
	return path, length, "sha256:" + hex.EncodeToString(hash.Sum(nil)), mediaType, nil
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

// Drop removes one exact content-addressed staging object after the records authority
// proves no unsettled request references it. Digest validation keeps cleanup from ever
// accepting a path-like spelling.
func Drop(layout home.Layout, digest string) *exit.Error {
	if !validDigest(digest) {
		return exit.New(exit.Validation, "cannot clean an invalid input asset digest %q", digest)
	}
	if err := os.Remove(layout.InputAsset(digest)); err != nil && !os.IsNotExist(err) {
		return exit.Internalf("cannot remove staged input asset %s: %s", digest, err)
	}
	return nil
}

// DropUnowned removes the distinct staged objects that no unsettled request owns. The
// caller holds Guard across the surrounding ownership transition.
func DropUnowned(layout home.Layout, store *records.Store, assets []records.AssetBinding) *exit.Error {
	seen := map[string]bool{}
	for _, asset := range assets {
		if seen[asset.Digest] {
			continue
		}
		seen[asset.Digest] = true
		used, e := store.AssetInUse(asset.Digest)
		if e != nil {
			return e
		}
		if !used {
			if e := Drop(layout, asset.Digest); e != nil {
				return e
			}
		}
	}
	return nil
}

// Sweep removes objects a crash left without a request owner. It runs during service
// reconciliation, before submissions are admitted, under the same ownership guard as
// staging and terminal cleanup. Only this package's digest and temporary spellings are
// touched; an unrelated file in the directory is not guessed to be ours.
func Sweep(layout home.Layout, store *records.Store) *exit.Error {
	entries, err := os.ReadDir(layout.Inputs)
	if err != nil {
		return exit.Internalf("cannot scan the input asset store: %s", err)
	}
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		if strings.HasPrefix(name, ".asset-") {
			if err := os.Remove(filepath.Join(layout.Inputs, name)); err != nil && !os.IsNotExist(err) {
				return exit.Internalf("cannot remove orphan input staging file %s: %s", name, err)
			}
			continue
		}
		digest := "sha256:" + name
		if len(name) != 64 || !validDigest(digest) {
			continue
		}
		used, e := store.AssetInUse(digest)
		if e != nil {
			return e
		}
		if !used {
			if e := Drop(layout, digest); e != nil {
				return e
			}
		}
	}
	return nil
}

func validDigest(digest string) bool {
	_, err := canonical.Raw(digest)
	return err == nil
}

func overCap(path string, size, max int64) *exit.Error {
	return exit.Named(exit.Validation, "input_over_cap",
		"input asset %s is %d B and this deployment admits %d B per input", filepath.Base(path), size, max).
		WithRemedy("resize or recompress the input before submitting it")
}
